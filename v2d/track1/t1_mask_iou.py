#!/usr/bin/env python3
"""VAL ONLY: per-frame IoU of our SAM3 masks (masks/val/E/masks.npz) against FORM-HOI's released masks
(masks_fh/val/E/masks.npz), both at scale 0.5 on the same window frames.

  python -I t1_mask_iou.py --episodes 11 [--json out.json]        (scratch venv; CPU, seconds)
"""
import argparse
import json
from pathlib import Path

import numpy as np

R = Path("/mnt/secondary/v2d/t1")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", default="11"); ap.add_argument("--json")
    a = ap.parse_args()
    out = []
    for ep in [int(x) for x in a.episodes.split(",")]:
        E = f"episode_{ep:06d}"
        p, q = R / "masks" / "val" / E / "masks.npz", R / "masks_fh" / "val" / E / "masks.npz"
        if not (p.exists() and q.exists()):
            print("missing", ep); continue
        A, B = np.load(p), np.load(q)
        W = int(A["W"]); assert W == int(B["W"])
        fa, fb = list(A["frames"]), list(B["frames"])
        common = sorted(set(fa) & set(fb))
        ia, ib = {f: i for i, f in enumerate(fa)}, {f: i for i, f in enumerate(fb)}
        row = {"ep": ep, "frames": len(common)}
        for key in ("human", "object"):
            ious = []
            for f in common:
                x = np.unpackbits(A[key][ia[f]], axis=-1, count=W).astype(bool)
                y = np.unpackbits(B[key][ib[f]], axis=-1, count=W).astype(bool)
                u = (x | y).sum()
                ious.append(1.0 if u == 0 else (x & y).sum() / u)
            ious = np.array(ious)
            row[key] = {"mean": round(float(ious.mean()), 4), "median": round(float(np.median(ious)), 4),
                        "p10": round(float(np.percentile(ious, 10)), 4), "frac_below_0.5": round(float((ious < 0.5).mean()), 4)}
        out.append(row); print(json.dumps(row))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
