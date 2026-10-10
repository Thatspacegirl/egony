#!/usr/bin/env python3
"""VAL ONLY: FORM-HOI's released per-frame human/object masks (nvidia/form-hoi @ c63db107, CC-BY-4.0; files
formhoi_val/<seq>/{human,object}_masks__<camera>.h5) -> the masks.npz schema of t1_masks.py, so CARI4D / mesh
stages can run on the val set without spending GPU time on SAM3 (the GPU is shared by three tracks + training).

Why this is acceptable for validation: on val ep 18 our SAM3 masks match these at IoU 0.957 (human) / 0.969 (object)
(t1_masks_sam3.py smoke), and val is used to compare pipeline variants, which all get the same masks.  It is NOT a
Track 1 input: Track 1 always uses our SAM3 masks (t1_masks.py), and val eps 11 + 5 are run with BOTH mask sources
to measure the gap.  Output: /mnt/secondary/v2d/t1/masks_fh/val/episode_X/{masks.npz,masks.json} (window = the same
t1_items.window(), stride 1, scale 0.5, packed bits; obj_score = 1 where the object mask is non-empty).  Atomic.

  python -I t1_masks_formhoi.py --episodes 0-24      (env t1-cari4d: h5py + cv2; CPU, ~1 GB RAM)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_items as I  # noqa: E402

OUT = Path("/mnt/secondary/v2d/t1/masks_fh/val")


def parse_eps(s):
    out = []
    for p in s.split(","):
        if "-" in p:
            a, b = p.split("-"); out += list(range(int(a), int(b) + 1))
        elif p:
            out.append(int(p))
    return out


def main():
    import cv2
    import h5py
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", default="0-24"); ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    for ep in parse_eps(a.episodes):
        it = I.item("val", ep)
        od = OUT / f"episode_{ep:06d}"
        if (od / "masks.npz").is_file():
            print("exists", od); continue
        od.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        d = I.VAL_ROOT / it["sequence_id"]
        s, e = I.window(it)
        idx = np.arange(s, e)
        packed = {}
        for key in ("human", "object"):
            ds = h5py.File(d / f"{key}_masks__{it['camera']}.h5", "r")["frames"]
            assert ds.shape[0] == it["T"], (ds.shape, it["T"])
            FH, FW = ds.shape[1:]
            W, H = int(round(FW * a.scale)), int(round(FH * a.scale))
            out = np.zeros((len(idx), H, (W + 7) // 8), np.uint8)
            for i, f in enumerate(idx):
                m = cv2.resize(ds[int(f)], (W, H), interpolation=cv2.INTER_AREA) > 127
                out[i] = np.packbits(m, axis=-1)
            packed[key] = out
        cov = {k: float(np.mean(packed[k].reshape(len(idx), -1).any(1))) for k in packed}
        osc = packed["object"].reshape(len(idx), -1).any(1).astype(np.float32)
        tmp = od / "masks.tmp.npz"
        np.savez_compressed(tmp, frames=idx, human=packed["human"], object=packed["object"], obj_score=osc, W=W, H=H,
                            scale=a.scale, full_wh=np.array([FW, FH]))
        os.replace(tmp, od / "masks.npz")
        rep = {"split": "val", "episode": ep, "sequence_id": it["sequence_id"], "source": "FORM-HOI released masks (val only)",
               "window": [int(s), int(e)], "stride": 1, "n": len(idx), "scale": a.scale, "coverage": cov,
               "seconds": round(time.time() - t0, 1)}
        json.dump(rep, open(od / "masks.json", "w"), indent=1)
        print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
