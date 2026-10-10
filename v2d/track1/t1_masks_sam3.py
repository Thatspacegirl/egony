#!/usr/bin/env python3
"""Text-prompted human + object video masks with SAM3 -> CARI4D wild mask H5 (Track 1 and the FORM-HOI val set).

Replaces the toolkit's interactive SAM2 annotation step (v2d.sam2.docker.run_annotate / run_video_to_masks /
v2d.cari4d.docker.run_pack_masks) with a non-interactive host run (env /mnt/secondary/v2d/envs/t3-seg, HF
transformers Sam3VideoModel, weights /mnt/secondary/v2d/weights/sam3/sam3). Logic adapted from
../track3/t3_masks_sam3.py.

Prompts: "person" + the episode's object prompt (Track 1: meta/episodes_metadata.jsonl `object_prompt`;
FORM-HOI val: hoi_metadata.yaml `object.prompt`) + the object id as a short noun phrase ("air_purifier" ->
"air purifier"; SAM3 is trained on noun phrases). Both object prompts compete for the object slot. Instance choice:
  person : the instance with the largest mask area summed over the clip (the actor, not a background person)
  object : the instance (from either object prompt) with the largest score summed over the clip
  then the chosen id is followed; when SAM3 drops it and the same prompt reports a NEW id near the last
  centroid (< 0.25 * width), the slot is handed over (re-detection after occlusion).
Masks are computed at --scale and resized back to full resolution (bilinear, threshold 0.5).

Output per episode in <out>/episode_%06d/:
  episode_%06d.0.color.mp4      hard link (or copy) of the source video (CARI4D requires the <seq>.0.color.mp4 name)
  episode_%06d_masks_k0.h5      exactly lib/pack_masks.py's schema (v2d.cari4d.wild_masks.v1): group <seq>,
                                datasets "<stem>-k0.person_mask.png" / "<stem>-k0.obj_rend_mask.png" (uint8 0/255)
  masks.json                    prompts, chosen ids, hand-overs, coverage, seconds (+ IoU vs FORM-HOI masks)

usage (GPU):
  python -I t1_masks_sam3.py --track1 --out /mnt/secondary/v2d/t1/masks/track1 [--episodes 0 3]
  python -I t1_masks_sam3.py --formhoi-val --out /mnt/secondary/v2d/t1/masks/formhoi_val --compare-formhoi
CPU smoke test (3 frames, quarter resolution, ~5 GB RSS):
  python -I t1_masks_sam3.py --formhoi-val --episodes 18 --cpu --max-frames 3 --scale 0.25 --compare-formhoi \
      --out /mnt/secondary/v2d/t1/masks/smoke
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

DATASET = Path("/home/asubuntudesktop/TestingGrounds/egony/video_to_data_challenge")
VAL_ROOT = Path("/mnt/secondary/v2d/t1/formhoi_val")
MASK_SCHEMA = "v2d.cari4d.wild_masks.v1"
PERSON = "person"


def track1_items():
    t1 = DATASET / "track_1"
    out = []
    for line in open(t1 / "meta" / "episodes_metadata.jsonl"):
        m = json.loads(line)
        e = int(m["episode_index"])
        out.append({"episode": e, "video": t1 / "videos" / "chunk-000" / m["video_key"] / f"episode_{e:06d}.mp4",
                    "prompt": m["object_prompt"].rstrip(". "), "noun": m["object"].replace("_", " "),
                    "sequence_id": m["sequence_id"]})
    return out


def formhoi_items():
    import yaml
    sel = json.load(open(VAL_ROOT / "selection.json"))["sequences"]
    out = []
    for r in sel:
        d = VAL_ROOT / r["sequence_id"]
        meta = yaml.safe_load(open(d / "hoi_metadata.yaml"))
        out.append({"episode": int(r["val_index"]), "video": d / f"videos__{r['camera']}.mp4",
                    "prompt": str(meta["object"]["prompt"]).rstrip(". "), "noun": str(meta["object"]["id"]).replace("_", " "),
                    "sequence_id": r["sequence_id"],
                    "gt_masks": {"human": d / f"human_masks__{r['camera']}.h5", "object": d / f"object_masks__{r['camera']}.h5"}})
    return out


def read_frames(path, scale, max_frames):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(path))
    frames, full = [], None
    while True:
        ok, im = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        full = im.shape[:2]
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frames, full, n_video


def follow(per_frame, prompts, start_id, n, unpack, W, owner):
    """per-frame id of one slot, starting from start_id, with re-detection hand-over (see module doc)."""
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
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--track1", action="store_true")
    g.add_argument("--formhoi-val", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--weights", default="/mnt/secondary/v2d/weights/sam3/sam3")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--max-rss-gb", type=float, default=12.0)
    ap.add_argument("--compare-formhoi", action="store_true", help="IoU vs the FORM-HOI release masks (val only)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    def watchdog():
        while True:
            rss = int([l for l in open("/proc/self/status") if l.startswith("VmRSS")][0].split()[1]) / 1e6
            if rss > a.max_rss_gb:
                print(f"ABORT: RSS {rss:.2f} GB > {a.max_rss_gb}", flush=True)
                os._exit(3)
            time.sleep(0.2)
    threading.Thread(target=watchdog, daemon=True).start()

    import cv2
    import h5py
    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    dev = "cpu" if a.cpu else "cuda"
    dtype = torch.float32 if a.cpu else torch.bfloat16
    model = Sam3VideoModel.from_pretrained(a.weights, torch_dtype=dtype).to(dev).eval()
    proc = Sam3VideoProcessor.from_pretrained(a.weights)

    items = track1_items() if a.track1 else formhoi_items()
    if a.episodes:
        items = [it for it in items if it["episode"] in a.episodes]
    for it in items:
        seq = f"episode_{it['episode']:06d}"
        od = Path(a.out) / seq
        h5_path = od / f"{seq}_masks_k0.h5"
        if h5_path.exists() and not a.overwrite:
            print("exists", h5_path); continue
        od.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        frames, (FH, FW), n_video = read_frames(it["video"], a.scale, a.max_frames)
        n = len(frames)
        W = frames[0].size[0]

        def unpack(bits):
            return np.unpackbits(bits, axis=-1, count=W).astype(bool)
        obj_prompts = list(dict.fromkeys([it["prompt"], it["noun"]]))
        prompts = [PERSON] + obj_prompts
        sess = proc.init_video_session(video=frames, inference_device=dev, video_storage_device="cpu",
                                       processing_device="cpu", dtype=dtype)
        sess = proc.add_text_prompt(sess, prompts)
        per_frame = {}
        with torch.inference_mode():
            for out in model.propagate_in_video_iterator(sess):
                r = proc.postprocess_outputs(sess, out)
                per_frame[out.frame_idx] = dict(
                    ids=r["object_ids"].tolist(), scores=r["scores"].float().tolist(),
                    masks=np.packbits(r["masks"].cpu().numpy().astype(bool), axis=-1),
                    p2o={k: list(v) for k, v in r["prompt_to_obj_ids"].items()})
        del sess
        # initial choice per slot over the whole clip
        area, score = {}, {}
        for pf in per_frame.values():
            for i in pf["p2o"].get(PERSON, []):
                k = pf["ids"].index(i)
                area[i] = area.get(i, 0) + int(np.unpackbits(pf["masks"][k]).sum())
            for i in {i for p in obj_prompts for i in pf["p2o"].get(p, [])}:
                score[i] = score.get(i, 0.0) + pf["scores"][pf["ids"].index(i)]
        choice = {}
        if area:
            choice["human"] = ([PERSON], max(area, key=area.get))
        if score:
            choice["object"] = (obj_prompts, max(score, key=score.get))
        owner = {i: p[0] for p, i in choice.values()}
        info, slot_ids = {}, {}
        for slot, (p, i0) in choice.items():
            slot_ids[slot], sw = follow(per_frame, p, i0, n, unpack, W, owner)
            src = [q for q in p if any(i0 in pf["p2o"].get(q, []) for pf in per_frame.values())]
            info[slot] = {"prompt": src, "start_id": int(i0), "switches": sw,
                          "candidates": len(area) if slot == "human" else len(score)}
        stems = [f"{f:06d}" for f in range(n)]
        cover = {"human": 0, "object": 0}
        iou = {"human": [], "object": []}
        gt = {}
        if a.compare_formhoi and "gt_masks" in it:
            gt = {s: h5py.File(p, "r")["frames"] for s, p in it["gt_masks"].items()}
        tmp = h5_path.with_name(f".{h5_path.name}.{os.getpid()}.tmp")
        with h5py.File(tmp, "w") as h5:
            h5.attrs["schema"] = MASK_SCHEMA
            h5.attrs["sequence"] = seq
            h5.attrs["frame_count"] = n
            h5.attrs["width"] = FW
            h5.attrs["height"] = FH
            grp = h5.require_group(seq)
            for f, stem in enumerate(stems):
                full = {}
                for slot in ("human", "object"):
                    pf = per_frame.get(f)
                    m = np.zeros((frames[0].size[1], W), np.uint8)
                    if pf is not None and f in slot_ids.get(slot, {}):
                        m = unpack(pf["masks"][pf["ids"].index(slot_ids[slot][f])]).astype(np.uint8) * 255
                    m = (cv2.resize(m, (FW, FH), interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8) * 255
                    cover[slot] += bool(m.any())
                    full[slot] = m
                    if slot in gt:
                        g_ = gt[slot][f] > 0
                        u = np.logical_or(g_, m > 0).sum()
                        iou[slot].append(float(np.logical_and(g_, m > 0).sum() / u) if u else 1.0)
                grp.create_dataset(f"{stem}-k0.person_mask.png", data=full["human"], compression="lzf", shuffle=True)
                grp.create_dataset(f"{stem}-k0.obj_rend_mask.png", data=full["object"], compression="lzf", shuffle=True)
        os.replace(tmp, h5_path)
        link = od / f"{seq}.0.color.mp4"
        if not link.exists():   # hard link (Docker bind mounts do not follow symlinks out of the mount); else copy
            try:
                os.link(Path(it["video"]).resolve(), link)
            except OSError:
                import shutil
                shutil.copyfile(Path(it["video"]).resolve(), link)
        rep = {"episode": it["episode"], "sequence_id": it["sequence_id"], "video": str(it["video"]),
               "frames": n, "video_frames": n_video, "complete": n == n_video, "scale": a.scale, "prompts": prompts,
               "chosen": info, "coverage": {k: v / n for k, v in cover.items()}, "seconds": round(time.time() - t0, 1)}
        if gt:
            rep["iou_vs_formhoi"] = {k: float(np.mean(v)) for k, v in iou.items() if v}
        json.dump(rep, open(od / "masks.json", "w"), indent=1)
        print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
