#!/usr/bin/env python3
"""Consistency checks between stages (CPU, no GT):
  actor : GEM-X's tracked person box (preprocess/bbx.pt, largest-area YOLOX/ByteTrack track) vs the SAM3 actor
          mask (t1_masks.py, chosen by object contact) on the window frames: median box IoU and fraction of frames
          with IoU < 0.3 (=> GEM-X followed someone else -> rerun with --bbx from the SAM3 actor masks).
  K     : per-video MoGe-2 focal (human/<ep>/moge/K.json) grouped by camera.

  python t1_check.py --split track1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_items as I  # noqa: E402

R = Path("/mnt/secondary/v2d/t1")


def box_iou(a, b):
    x0, y0 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x1, y1 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(ua, 1e-6)


def mask_boxes(M):
    W = int(M["W"]); sc = float(M["full_wh"][0]) / W
    out = np.full((len(M["frames"]), 4), np.nan)
    for i in range(len(M["frames"])):
        m = np.unpackbits(M["human"][i], axis=-1, count=W).astype(bool)
        if m.any():
            ys, xs = np.nonzero(m)
            out[i] = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]) * sc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    import torch
    rows = {}
    for it in I.items(a.split):
        ep = it["episode"]; E = f"episode_{ep:06d}"
        r = {"camera": it["camera"]}
        kj = R / "human" / a.split / E / "moge" / "K.json"
        if kj.exists():
            K = json.load(open(kj)); r["f"] = K["f_med"]; r["f_mad"] = K["f_mad"]
        bb = list((R / "human" / a.split / E / "gemx").glob("*/preprocess/bbx.pt"))
        mk = R / "masks" / a.split / E / "masks.npz"
        if bb and mk.exists():
            B = torch.load(bb[0], map_location="cpu", weights_only=False)["bbx_xyxy"].numpy()
            M = dict(np.load(mk))
            mb = mask_boxes(M)
            fr = M["frames"]
            ok = np.isfinite(mb).all(1)
            iou = box_iou(B[fr[ok]], mb[ok])
            r.update({"actor_iou_med": float(np.median(iou)), "actor_bad_frac": float((iou < 0.3).mean()),
                      "sam3_actor": json.load(open(mk.parent / "masks.json"))["chosen"].get("human", {}).get("start_id")})
        rows[ep] = r
        print(ep, json.dumps(r))
    by = {}
    for ep, r in rows.items():
        if "f" in r:
            by.setdefault(r["camera"], []).append((ep, round(r["f"], 1)))
    for c, v in by.items():
        f = np.array([x[1] for x in v])
        print(f"{c:28s} n={len(v)} f median {np.median(f):.1f} min {f.min():.1f} max {f.max():.1f}  {v}")


if __name__ == "__main__":
    main()
