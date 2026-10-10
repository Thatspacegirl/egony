#!/usr/bin/env python3
"""SAM3 text-prompted actor + object masks on the processing window of one Track 1 / FORM-HOI val episode.

Env: /mnt/secondary/v2d/envs/t3-seg (HF transformers Sam3VideoModel, weights /mnt/secondary/v2d/weights/sam3/sam3).
GPU, under the shared lock.  Derived from t1_masks_sam3.py (same prompts / tracking / hand-over logic) with:
  * only the window t1_items.window(item) (scored span - 60 / + 30 frames), every --stride-th frame;
  * ACTOR choice by object proximity (Track 1 has seated / standing bystanders): for each person instance, the
    number of frames where its mask, dilated by 2% of the width, touches the chosen object mask; ties -> area.
    The object instance is chosen first (largest summed score over the clip, prompts: sentence + noun).
  * output <out>/masks.npz: frames (N,), human (N,h,ceil(w/8)) / object packed bits at --scale resolution,
    obj_score (N,), plus masks.json (choices, coverage, timing).  Atomic writes.

  python -I t1_masks.py --split val --episode 24 --out /mnt/secondary/v2d/t1/masks/val/episode_000024
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_items as I  # noqa: E402

PERSON = "person"


def read_frames(path, idx, scale):
    import cv2
    from PIL import Image
    want = set(int(i) for i in idx)
    cap = cv2.VideoCapture(str(path))
    frames, n, full = [], 0, None
    last = max(want)
    while n <= last:
        ok, im = cap.read()
        if not ok:
            break
        if n in want:
            full = im.shape[:2]
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        n += 1
    assert len(frames) == len(idx), (len(frames), len(idx))
    return frames, full


def follow(per_frame, prompts, start_id, n, unpack, W, owner):
    ids_out, cur, last_c, switches = {}, start_id, None, []
    for f in range(n):
        pf = per_frame.get(f)
        if pf is None:
            continue
        if cur not in pf["ids"]:
            best, bd = None, 0.25 * W
            for i in [i for p in prompts for i in pf["p2o"].get(p, []) if i not in owner]:
                mm = unpack(pf["masks"][pf["ids"].index(i)])
                if not mm.any():
                    continue
                ys, xs = np.nonzero(mm)
                dd = 0.0 if last_c is None else float(np.hypot(xs.mean() - last_c[0], ys.mean() - last_c[1]))
                if dd < bd:
                    best, bd = i, dd
            if best is not None:
                owner[best] = prompts[0]
                switches.append([f, int(cur), int(best)])
                cur = best
        if cur in pf["ids"]:
            ids_out[f] = cur
            mm = unpack(pf["masks"][pf["ids"].index(cur)])
            if mm.any():
                ys, xs = np.nonzero(mm)
                last_c = (xs.mean(), ys.mean())
    return ids_out, switches


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["val", "track1"])
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="/mnt/secondary/v2d/weights/sam3/sam3")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--pre", type=int, default=60)
    ap.add_argument("--post", type=int, default=30)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--max-rss-gb", type=float, default=9.0)
    ap.add_argument("--seg", type=int, default=240, help="frames per SAM3 session (memory bound)")
    a = ap.parse_args()

    def watchdog():
        while True:
            # anonymous RSS only: file-backed pages (the mmap'ed safetensors during from_pretrained) are reclaimable
            rss = int([l for l in open("/proc/self/status") if l.startswith("RssAnon")][0].split()[1]) / 1e6
            if rss > a.max_rss_gb:
                print(f"ABORT: RSS {rss:.2f} GB > {a.max_rss_gb}", flush=True)
                os._exit(3)
            time.sleep(0.2)
    threading.Thread(target=watchdog, daemon=True).start()

    import cv2
    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    it = I.item(a.split, a.episode)
    s, e = I.window(it, a.pre, a.post)
    idx = np.arange(s, e, a.stride)
    if a.max_frames:
        idx = idx[: a.max_frames]
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dev = "cpu" if a.cpu else "cuda"
    dtype = torch.float32 if a.cpu else torch.bfloat16
    model = Sam3VideoModel.from_pretrained(a.weights, torch_dtype=dtype).to(dev).eval()
    proc = Sam3VideoProcessor.from_pretrained(a.weights)
    frames, (FH, FW) = read_frames(it["video"], idx, a.scale)
    n = len(frames)
    W, H = frames[0].size

    def unpack(bits):
        return np.unpackbits(bits, axis=-1, count=W).astype(bool)
    obj_prompts = list(dict.fromkeys([it["prompt"], it["noun"]]))
    prompts = [PERSON] + obj_prompts
    # SAM3 runs on SEGMENTS of --seg frames (each its own session, preprocessed in chunks straight to the device in
    # bf16): one session over a whole 900-frame window keeps every frame's tracking state on the GPU (CUDA OOM after
    # ~600 frames on the 24 GB card) and Sam3VideoProcessor.init_video_session would first build the clip as one
    # float32 1008x1008 CPU tensor (12 MB/frame).  Instance ids are made global (seg * 100000 + id) and linked across
    # segment boundaries by mask IoU (last frame of segment k vs first frame of k+1, same prompt group, IoU > 0.3).
    from transformers.models.sam3_video.modeling_sam3_video import Sam3VideoInferenceSession
    store = "cpu" if a.cpu else "cuda"
    segs, c = [], 0
    while c < n:
        b = min(n, c + a.seg)
        if n - b < a.seg // 4:
            b = n
        segs.append((c, b)); c = b
    per_frame = {}
    for si, (sa, sb) in enumerate(segs):
        chunks, vh, vw = [], None, None
        for c0 in range(sa, sb, 32):
            pv = proc.video_processor(videos=frames[c0:min(sb, c0 + 32)], device=store, return_tensors="pt")
            chunks.append(pv.pixel_values_videos[0].to(store, dtype=dtype))
            if vh is None:
                vh, vw = int(pv.original_sizes[0][0]), int(pv.original_sizes[0][1])
            del pv
        video_t = torch.cat(chunks); del chunks
        sess = Sam3VideoInferenceSession(video=video_t, video_height=vh, video_width=vw, inference_device=dev,
                                         video_storage_device=store, inference_state_device=dev, dtype=dtype)
        del video_t
        sess = proc.add_text_prompt(sess, prompts)
        off = si * 100000
        with torch.inference_mode():
            for out in model.propagate_in_video_iterator(sess):
                r = proc.postprocess_outputs(sess, out)
                per_frame[sa + out.frame_idx] = dict(
                    ids=[off + int(i) for i in r["object_ids"].tolist()], scores=r["scores"].float().tolist(),
                    masks=np.packbits(r["masks"].cpu().numpy().astype(bool), axis=-1),
                    p2o={k: [off + int(i) for i in v] for k, v in r["prompt_to_obj_ids"].items()})
        del sess
        if not a.cpu:
            torch.cuda.empty_cache()
        print(f"segment {si} [{sa},{sb}) done {time.time() - t0:.0f}s", flush=True)
    frames = None
    t_sam = time.time() - t0
    # ---- chains of ids linked across segment boundaries
    parent = {}

    def find_(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    links = []
    for k in range(len(segs) - 1):
        fa, fb = segs[k][1] - 1, segs[k + 1][0]
        A, B = per_frame.get(fa), per_frame.get(fb)
        if A is None or B is None:
            continue
        for group in ([PERSON], obj_prompts):
            ia = sorted({i for p in group for i in A["p2o"].get(p, [])})
            ib = sorted({i for p in group for i in B["p2o"].get(p, [])})
            pairs = []
            for x in ia:
                mx = unpack(A["masks"][A["ids"].index(x)])
                for y in ib:
                    my = unpack(B["masks"][B["ids"].index(y)])
                    u = (mx | my).sum()
                    if u:
                        pairs.append(((mx & my).sum() / u, x, y))
            used = set()
            for iou, x, y in sorted(pairs, reverse=True):
                if iou < 0.3 or x in used or y in used:
                    continue
                used |= {x, y}
                parent[find_(y)] = find_(x)
                links.append([int(x), int(y), round(float(iou), 3)])
    seg_of = np.zeros(n, int)
    for si, (sa, sb) in enumerate(segs):
        seg_of[sa:sb] = si

    def chain_follow(prompts_, root, owner_):
        """per frame id of chain `root`; inside a segment where the chain has no id (or loses it), hand over to the
        nearest unowned instance of the same prompts (follow() logic)."""
        out, cur, last_c, switches = {}, None, None, []
        for f in range(n):
            pf = per_frame.get(f)
            if pf is None:
                continue
            if f == 0 or seg_of[f] != seg_of[f - 1]:
                mem = [i for i in pf["ids"] if find_(i) == root]
                if mem:
                    cur = mem[0]
            if cur not in pf["ids"]:
                best, bd = None, 0.25 * W
                for i in [i for p in prompts_ for i in pf["p2o"].get(p, []) if owner_.get(i) is None]:
                    mm = unpack(pf["masks"][pf["ids"].index(i)])
                    if not mm.any():
                        continue
                    ys, xs = np.nonzero(mm)
                    dd = 0.0 if last_c is None else float(np.hypot(xs.mean() - last_c[0], ys.mean() - last_c[1]))
                    if last_c is None and find_(i) != root:
                        continue
                    if dd < bd:
                        best, bd = i, dd
                if best is not None:
                    switches.append([f, int(cur) if cur is not None else -1, int(best)])
                    cur = best
            if cur in pf["ids"]:
                owner_[cur] = prompts_[0]
                out[f] = cur
                mm = unpack(pf["masks"][pf["ids"].index(cur)])
                if mm.any():
                    ys, xs = np.nonzero(mm)
                    last_c = (xs.mean(), ys.mean())
        return out, switches
    # ---- object first: chain with the largest summed score
    score = {}
    for pf in per_frame.values():
        for i in {i for p in obj_prompts for i in pf["p2o"].get(p, [])}:
            score[find_(i)] = score.get(find_(i), 0.0) + pf["scores"][pf["ids"].index(i)]
    owner, info, slot_ids = {}, {"segments": [list(x) for x in segs], "links": links}, {}
    if score:
        c0_ = max(score, key=score.get)
        slot_ids["object"], sw = chain_follow(obj_prompts, c0_, owner)
        info["object"] = {"chain": int(c0_), "switches": sw, "candidates": len(score),
                          "score": round(score[c0_], 2)}
    # ---- actor: person chain touching the object most often (ties -> area)
    rad = max(3, int(0.02 * W))
    ker = np.ones((2 * rad + 1, 2 * rad + 1), np.uint8)
    touch, area = {}, {}
    for f, pf in per_frame.items():
        om = None
        if f in slot_ids.get("object", {}):
            om = unpack(pf["masks"][pf["ids"].index(slot_ids["object"][f])])
        for i in pf["p2o"].get(PERSON, []):
            pm = unpack(pf["masks"][pf["ids"].index(i)])
            ci = find_(i)
            area[ci] = area.get(ci, 0) + int(pm.sum())
            if om is not None and om.any() and pm.any():
                if (cv2.dilate(pm.astype(np.uint8), ker) > 0)[om].any():
                    touch[ci] = touch.get(ci, 0) + 1
    if area:
        maxa = max(area.values())
        h0 = max(area, key=lambda i: (touch.get(i, 0), area[i] / maxa))
        slot_ids["human"], sw = chain_follow([PERSON], h0, owner)
        info["human"] = {"chain": int(h0), "switches": sw, "candidates": len(area),
                         "touch": {int(k): int(v) for k, v in touch.items()},
                         "area_frac": {int(k): round(v / maxa, 3) for k, v in area.items()},
                         "largest_area_chain": int(max(area, key=area.get))}
    wb = (W + 7) // 8
    hum = np.zeros((n, H, wb), np.uint8); obj = np.zeros((n, H, wb), np.uint8); osc = np.zeros(n, np.float32)
    cover = {"human": 0, "object": 0}
    for f in range(n):
        pf = per_frame.get(f)
        for slot, arr in (("human", hum), ("object", obj)):
            if pf is not None and f in slot_ids.get(slot, {}):
                k = pf["ids"].index(slot_ids[slot][f])
                arr[f] = pf["masks"][k]
                cover[slot] += 1
                if slot == "object":
                    osc[f] = pf["scores"][k]
    tmp = od / "masks.tmp.npz"
    np.savez_compressed(tmp, frames=idx, human=hum, object=obj, obj_score=osc, W=W, H=H, scale=a.scale,
                        full_wh=np.array([FW, FH]))
    os.replace(tmp, od / "masks.npz")
    rep = {"split": a.split, "episode": a.episode, "sequence_id": it["sequence_id"], "video": it["video"],
           "window": [int(s), int(e)], "stride": a.stride, "n": n, "scale": a.scale, "prompts": prompts,
           "chosen": info, "coverage": {k: v / n for k, v in cover.items()}, "sam3_seconds": round(t_sam, 1),
           "seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(od / "masks.json", "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("episode", "n", "coverage", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
