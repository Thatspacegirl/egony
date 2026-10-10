"""Text-prompted video object masks with SAM3 (HF transformers Sam3VideoModel, facebook/sam3) -- host venv, no Docker.

For each episode: prompts = one phrase per object slot (OBJECT_PROMPTS) + optionally "hand".
Per prompt keeps the highest-scoring instance on the first frame where it appears (instance id is then fixed),
and writes per-frame binary masks:
  OUT/<split>/episode_XXXXXX/<cam>_s<scale>/<object_name>/000000.png   (uint8 0/255; missing frame = empty mask)
  OUT/<split>/episode_XXXXXX/<cam>_s<scale>/masks.json                 (scores, chosen ids, prompt map)
--cam cam_a (colour, undistorted) or left (rectified mono -> fed as 3-channel gray).
GPU expected (bf16).  --cpu --max_frames 3 --scale 0.25 is a slow CPU smoke test of the API.

  python -I t3_masks_sam3.py --frames /mnt/secondary/v2d/t3/frames --out /mnt/secondary/v2d/t3/masks \
      --split evaluation --episodes 3 --cam cam_a --scale 0.5 [--hands]
"""
import argparse
import glob
import json
import os
import threading
import time

import cv2
import numpy as np

# Object name -> SAM3 noun phrase.  Chosen 2026-10-08 from t3_prompt_check.py overlays (13 episodes x 3 frames,
# /mnt/secondary/v2d/t3/checks/prompts/): "watering can" missed the pitcher on public 21 f0 -> "white jug" (0.93);
# "dustpan" missed public 39 f0 -> "yellow dustpan"; "dish rack" 0.48-0.78 -> "white dish rack" 0.77-0.90;
# "lid" also fires on the pot body -> "pot lid"; "white pot" keeps the handle (eval 3 f0/f102/f205).
# wooden_piece_1/2 share "wooden block": two instances on the first frame showing both, assigned LEFT->RIGHT.
# Verified rule: the public scans differ only by the mocap balls (wooden_piece_1 has 4, wooden_piece_2 has 3) and in
# public 23, 24 and eval 25 frame 0 the 4-ball board is the LEFT one (checks/overview/wooden_crops.jpg).
OBJECT_PROMPTS = {
    "white_pot": "white pot",
    "white_pot_lid": "pot lid",
    "wooden_spoon": "wooden spoon",
    "blue_cup": "blue cup",
    "beige_cup": "beige cup",
    "plastic_dish_rack": "white dish rack",
    "water_pitcher": "white jug",
    "mini_sweeper": "hand brush",
    "mini_dust_pan": "yellow dustpan",
    "wooden_piece_1": "wooden block",
    "wooden_piece_2": "wooden block",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/mnt/secondary/v2d/t3/frames")
    ap.add_argument("--out", default="/mnt/secondary/v2d/t3/masks")
    ap.add_argument("--weights", default="/mnt/secondary/v2d/weights/sam3/sam3")
    ap.add_argument("--split", nargs="*", default=["public", "evaluation"])
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--cam", choices=["cam_a", "left"], default="cam_a")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--hands", action="store_true", help="also segment 'hand' (for VO / depth cleaning)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--max_rss_gb", type=float, default=10.0)
    ap.add_argument("--storage", default="cpu", help="device holding the preprocessed bf16 frames")
    a = ap.parse_args()

    def watchdog():
        while True:
            rss = int([l for l in open("/proc/self/status") if l.startswith("VmRSS")][0].split()[1]) / 1e6
            if rss > a.max_rss_gb:
                print(f"ABORT: RSS {rss:.2f} GB > {a.max_rss_gb}", flush=True)
                os._exit(3)
            time.sleep(0.2)
    threading.Thread(target=watchdog, daemon=True).start()

    import torch
    from PIL import Image
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    dev = "cpu" if a.cpu else "cuda"
    dtype = torch.float32 if a.cpu else torch.bfloat16
    model = Sam3VideoModel.from_pretrained(a.weights, torch_dtype=dtype).to(dev).eval()
    proc = Sam3VideoProcessor.from_pretrained(a.weights)

    eps = []
    for split in a.split:
        for d in sorted(glob.glob(f"{a.frames}/{split}/episode_*")):
            if not a.episodes or int(d[-6:]) in a.episodes:
                eps.append(d)
    for ep_dir in eps:
        meta = json.load(open(f"{ep_dir}/meta.json"))
        n = meta["n_frames"] if not a.max_frames else min(a.max_frames, meta["n_frames"])
        frames = []
        for i in range(n):
            im = cv2.imread(f"{ep_dir}/{a.cam}/{i:06d}.jpg", cv2.IMREAD_COLOR)
            im = cv2.resize(im, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        H, W = frames[0].size[1], frames[0].size[0]

        def unpack(bits):
            return np.unpackbits(bits, axis=-1, count=W).astype(bool)
        objs = meta["objects"]
        prompts = list(dict.fromkeys(OBJECT_PROMPTS[o] for o in objs)) + (["hand"] if a.hands else [])
        t0 = time.time()
        # Memory: init_video_session(video=frames) preprocesses the WHOLE clip at once in float32 (1008x1008:
        # 12 MB/frame) -> host RSS > 10 GB on a 325-frame clip, or a CUDA OOM when done on the GPU (2026-10-08).
        # Instead: preprocess chunks of 16 frames on the processing device and store each frame in bf16 on the CPU
        # (6 MB/frame, 2.5 GB for 421 frames) through the session's own add_new_frame; tracking is unchanged
        # (offline propagation incl. the hotstart heuristics).
        sess = proc.init_video_session(video=None, inference_device=dev, video_storage_device=a.storage,
                                       processing_device=dev, dtype=dtype)
        sess.video_height, sess.video_width = H, W
        for c0 in range(0, len(frames), 16):
            pv = proc.video_processor(videos=[frames[c0:c0 + 16]], device=dev, return_tensors="pt").pixel_values_videos[0]
            for j in range(pv.shape[0]):
                sess.add_new_frame(pv[j], c0 + j)
            del pv
        del frames
        sess = proc.add_text_prompt(sess, prompts)
        per_frame = {}
        with torch.inference_mode():
            for out in model.propagate_in_video_iterator(sess):
                r = proc.postprocess_outputs(sess, out)
                per_frame[out.frame_idx] = dict(
                    ids=r["object_ids"].tolist(), scores=r["scores"].float().tolist(),
                    masks=np.packbits(r["masks"].cpu().numpy(), axis=-1),  # bit-packed to bound host RAM
                    p2o={k: list(v) for k, v in r["prompt_to_obj_ids"].items()})
        # choose instance ids: per prompt, the highest-score instance on its first frame of appearance;
        # prompts shared by several slots (two wooden blocks) take the top-k instances sorted by x of first box
        choice, info = {}, {}
        for p in prompts:
            slots = [o for o in objs if OBJECT_PROMPTS[o] == p] or ["hand"]
            # first frame with at least as many instances as slots (else the first frame with any instance)
            cand = [f for f in sorted(per_frame) if len(per_frame[f]["p2o"].get(p, [])) >= len(slots)] or \
                [f for f in sorted(per_frame) if per_frame[f]["p2o"].get(p, [])]
            for f in cand[:1]:
                ids = per_frame[f]["p2o"].get(p, [])
                if ids:
                    sc = {i: per_frame[f]["scores"][per_frame[f]["ids"].index(i)] for i in ids}
                    top = sorted(ids, key=lambda i: -sc[i])[:len(slots)]
                    if len(slots) > 1:
                        cx = {i: np.nonzero(unpack(per_frame[f]["masks"][per_frame[f]["ids"].index(i)]).any(0))[0].mean()
                              for i in top}
                        top = sorted(top, key=lambda i: cx[i])
                    for s, i in zip(slots, top):
                        choice[s] = i
                        info[s] = dict(prompt=p, obj_id=int(i), first_frame=int(f), first_score=float(sc[i]))
                    break
        # per-frame id per slot: keep the chosen id while SAM3 reports it; when it vanishes and the same prompt
        # reports a NEW id that no other slot owns (SAM3 re-detection after an occlusion), hand the slot over to the
        # new id if its mask is near the slot's last mask (centroid within 0.25 * image width).
        slot_id = {f: {} for f in range(n)}
        owner = {i: s for s, i in choice.items()}
        for s0 in [o for o in objs if o in choice]:
            cur, last_c, switches = choice[s0], None, []
            p = OBJECT_PROMPTS[s0]
            for f in range(n):
                pf = per_frame.get(f)
                if pf is None:
                    continue
                if cur not in pf["ids"]:
                    new_ids = [i for i in pf["p2o"].get(p, []) if i not in owner]
                    best, bd = None, 0.25 * W
                    for i in new_ids:
                        mm = unpack(pf["masks"][pf["ids"].index(i)])
                        if not mm.any():
                            continue
                        ys, xs = np.nonzero(mm)
                        dd = 0.0 if last_c is None else np.hypot(xs.mean() - last_c[0], ys.mean() - last_c[1])
                        if dd < bd:
                            best, bd = i, dd
                    if best is not None:
                        owner[best] = s0
                        switches.append([f, int(cur), int(best)])
                        cur = best
                if cur in pf["ids"]:
                    slot_id[f][s0] = cur
                    mm = unpack(pf["masks"][pf["ids"].index(cur)])
                    if mm.any():
                        ys, xs = np.nonzero(mm)
                        last_c = (xs.mean(), ys.mean())
            info[s0]["switches"] = switches
        od = f"{a.out}/{meta['split']}/episode_{meta['episode']:06d}/{a.cam}_s{a.scale:g}"
        cover = {}
        for s in list(objs) + (["hand"] if a.hands else []):
            os.makedirs(f"{od}/{s}", exist_ok=True)
            k = 0
            for f in range(n):
                m = np.zeros((H, W), np.uint8)
                pf = per_frame.get(f)
                if s == "hand" and pf is not None:  # union of ALL hand instances (both hands, any arm pieces)
                    for i in pf["p2o"].get("hand", []):
                        m |= unpack(pf["masks"][pf["ids"].index(i)]).astype(np.uint8) * 255
                    k += bool(m.any())
                elif pf is not None and s in slot_id[f]:
                    m = unpack(pf["masks"][pf["ids"].index(slot_id[f][s])]).astype(np.uint8) * 255
                    k += bool(m.any())
                cv2.imwrite(f"{od}/{s}/{f:06d}.png", m)
            cover[s] = k / n
        json.dump(dict(prompts=prompts, chosen=info, coverage=cover, frames=n, scale=a.scale, cam=a.cam,
                       size_wh=[W, H], seconds=round(time.time() - t0, 1)), open(f"{od}/masks.json", "w"), indent=1)
        print(f"{od}: {n} frames, coverage {cover}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
