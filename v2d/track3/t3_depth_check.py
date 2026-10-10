"""CPU check of FoundationStereo depth (depth_rect/*.png) against OpenCV SGBM on the same rectified pair.

Both use the same rectification + baseline, so this checks the network's disparity (not the metric scale; the metric
scale check against public scans is t3_scale_check.py).  Per sampled frame: median FS/SGBM depth ratio over pixels
where both are valid in 0.25-1.6 m, the fraction within 2 % / 5 %, and an overlay PNG (FS | SGBM | rel. diff).

  python -I t3_depth_check.py --ep_dir FRAMES/public/episode_000012 --depth_dir DEPTH/public/episode_000012 \
      --frames 0 90 180 --out_png /mnt/secondary/v2d/t3/checks/fs_vs_sgbm_pub12.png
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t3_depth_warp import decode_inv_depth  # noqa: E402


def sgbm_depth(L, R, fx, base):
    sg = cv2.StereoSGBM_create(0, 192, 5, P1=200, P2=800, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                               mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sg.compute(L, R).astype(np.float32) / 16
    return np.where(disp > 2, fx * base / np.maximum(disp, 1e-3), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_dir", required=True)
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--out_png")
    a = ap.parse_args()
    meta = json.load(open(f"{a.ep_dir}/meta.json"))
    fx, base = meta["stereo"]["fx"], meta["stereo"]["baseline_m"]
    n = meta["n_frames"]
    frames = a.frames or [0, n // 2, n - 1]
    rows, res = [], []
    for f in frames:
        L = cv2.imread(f"{a.ep_dir}/left/{f:06d}.jpg", 0)
        R = cv2.imread(f"{a.ep_dir}/right/{f:06d}.jpg", 0)
        ds = sgbm_depth(L, R, fx, base)
        df = decode_inv_depth(cv2.imread(f"{a.depth_dir}/depth_rect/{f:06d}.png", cv2.IMREAD_UNCHANGED))
        both = (ds > 0.25) & (ds < 1.6) & (df > 0.25) & (df < 1.6)
        r = df[both] / ds[both]
        st = dict(frame=f, fs_valid=float(np.mean(df > 0)), sgbm_valid=float(np.mean(ds > 0)), n_both=int(both.sum()),
                  median_ratio=float(np.median(r)), within_2pct=float(np.mean(np.abs(r - 1) < 0.02)),
                  within_5pct=float(np.mean(np.abs(r - 1) < 0.05)),
                  fs_median_depth=float(np.median(df[(df > 0.1) & (df < 3)])))
        res.append(st)
        print(json.dumps(st))
        vis = lambda d: np.where(d[..., None] > 0, cv2.applyColorMap(
            np.clip(255 * (1 - (d - 0.3) / 1.5), 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO), 0)
        rel = np.zeros_like(df)
        rel[both] = (r - 1)
        rv = cv2.applyColorMap(np.clip(128 + rel * 128 / 0.1, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        rv[~both] = 0
        row = np.hstack([cv2.cvtColor(L, cv2.COLOR_GRAY2BGR), vis(df), vis(ds), rv])
        rows.append(cv2.resize(row, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA))
    if a.out_png:
        os.makedirs(os.path.dirname(a.out_png), exist_ok=True)
        cv2.imwrite(a.out_png, np.vstack(rows))
    print("SUMMARY median_ratio", np.round([s["median_ratio"] for s in res], 4).tolist(),
          "within5", np.round([s["within_5pct"] for s in res], 3).tolist())


if __name__ == "__main__":
    main()
