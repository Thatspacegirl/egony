"""Transfer SAM3 cam_a masks to the rectified-LEFT grid through the FoundationStereo depth (CPU; replaces a second
SAM3 run on the mono left images, and keeps the SAME instance ids in both views).

Stereo frame i takes the cam_a mask of frame i + lag[i] (t3_sync.py; depth_cam_a must be written with the same lags).
For every left pixel with depth: X_rect -> X_a = T_a_rect X_rect -> pixel in the cam_a mask grid (scale s); the label is
taken there if the point lies on the surface cam_a sees (|z_a - cam_a depth| < --occ_tol, depth_cam_a_s{s} with
5x5 hole filling); occluded / background-through-parallax / no-depth pixels get 0.  Output layout = what t3_masks_sam3.py --cam left writes:
  MASKS/<split>/episode_X/left_s1/<slot>/000000.png  (uint8 0/255) + masks.json (source + coverage)

  python -I t3_masks_to_left.py --split public --episode 12 [--scale 0.5]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t3_depth_warp import decode_inv_depth  # noqa: E402

R0 = "/mnt/secondary/v2d/t3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--occ_tol", type=float, default=0.03)
    a = ap.parse_args()
    e = f"episode_{a.episode:06d}"
    meta = json.load(open(f"{R0}/frames/{a.split}/{e}/meta.json"))
    K = np.array(meta["images"]["left"]["K"])
    W, H = meta["images"]["left"]["size_wh"]
    Ka = np.array(meta["images"]["cam_a"]["K"], float)
    Ka[:2] *= a.scale
    T = np.array(meta["transforms"]["T_a_rect"])
    src = f"{R0}/masks/{a.split}/{e}/cam_a_s{a.scale:g}"
    dst = f"{R0}/masks/{a.split}/{e}/left_s1"
    slots = [s for s in meta["objects"] + ["hand"] if os.path.isdir(f"{src}/{s}")]
    for s in slots:
        os.makedirs(f"{dst}/{s}", exist_ok=True)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    cover = {s: 0 for s in slots}
    from t3_sync import load_lag
    n = meta["n_frames"]
    lag = load_lag(a.split, e, n)
    for f in range(n):
        j = int(np.clip(f + lag[f], 0, n - 1))  # cam_a frame showing stereo frame f (t3_sync.py)
        d = decode_inv_depth(cv2.imread(f"{R0}/depth/{a.split}/{e}/depth_rect/{f:06d}.png", cv2.IMREAD_UNCHANGED))
        da = decode_inv_depth(cv2.imread(f"{R0}/depth/{a.split}/{e}/depth_cam_a_s{a.scale:g}/{j:06d}.png",
                                         cv2.IMREAD_UNCHANGED))
        X = rays * d[..., None]
        Xa = X @ T[:3, :3].T + T[:3, 3]
        z = Xa[..., 2]
        ok = (d > 0) & (z > 0.05)
        ua = np.round(Xa[..., 0] / np.where(ok, z, 1) * Ka[0, 0] + Ka[0, 2]).astype(np.int32)
        va = np.round(Xa[..., 1] / np.where(ok, z, 1) * Ka[1, 1] + Ka[1, 2]).astype(np.int32)
        Ha, Wa = da.shape
        ok &= (ua >= 0) & (ua < Wa) & (va >= 0) & (va < Ha)
        ua, va = np.where(ok, ua, 0), np.where(ok, va, 0)
        # splat holes of the forward-warped cam_a depth filled with the nearest (min) valid depth in 5x5
        big = np.where(da > 0, da, 1e6).astype(np.float32)
        da_f = np.where(da > 0, da, cv2.erode(big, np.ones((5, 5), np.uint8)))
        za = da_f[va, ua]
        # the left point must lie ON the surface cam_a sees there: rejects background seen past an object edge in
        # the left view (it lands inside the object's cam_a mask through parallax) and points occluded in cam_a
        ok &= (za < 1e5) & (np.abs(z - za) < a.occ_tol)
        for s in slots:
            m = cv2.imread(f"{src}/{s}/{j:06d}.png", 0)
            out = np.zeros((H, W), np.uint8)
            if m is not None and m.any():
                out[ok & (m[va, ua] > 0)] = 255
                out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))  # grazing walls: sparse hits
                cover[s] += bool(out.any())
            cv2.imwrite(f"{dst}/{s}/{f:06d}.png", out)
    json.dump(dict(source=src, method="depth transfer cam_a -> rect-left", occ_tol=a.occ_tol,
                   coverage={s: cover[s] / meta["n_frames"] for s in slots}, frames=meta["n_frames"], size_wh=[W, H]),
              open(f"{dst}/masks.json", "w"), indent=1)
    print(dst, {s: round(cover[s] / meta["n_frames"], 3) for s in slots})


if __name__ == "__main__":
    main()
