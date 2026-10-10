"""Validate the exported frames + transforms of one preprocessed episode (CPU only).

SGBM depth on the written rectified pair (left/right) -> 3D in the rectified-left frame -> T_a_rect ->
project with the undistorted cam_a K -> NCC between warped left intensity and the written cam_a gray.
Compares against wrong-transform hypotheses (identity, inverted) as a control.

  python -I t3_check_cam_a.py /mnt/secondary/v2d/t3/frames/public/episode_000012 [frame ...]
"""
import json
import sys

import cv2
import numpy as np

ep_dir = sys.argv[1]
frames = [int(f) for f in sys.argv[2:]] or [0]
meta = json.load(open(f"{ep_dir}/meta.json"))
Kr = np.array(meta["images"]["left"]["K"])
Ka = np.array(meta["images"]["cam_a"]["K"])
T_a_rect = np.array(meta["transforms"]["T_a_rect"])
base = meta["stereo"]["baseline_m"]
W, H = meta["images"]["cam_a"]["size_wh"]
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5, P1=8 * 25, P2=32 * 25,
                             uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)
for f in frames:
    L = cv2.imread(f"{ep_dir}/left/{f:06d}.jpg", cv2.IMREAD_GRAYSCALE)
    R = cv2.imread(f"{ep_dir}/right/{f:06d}.jpg", cv2.IMREAD_GRAYSCALE)
    A = cv2.imread(f"{ep_dir}/cam_a/{f:06d}.jpg", cv2.IMREAD_GRAYSCALE)
    disp = sgbm.compute(L, R).astype(np.float32) / 16
    v, u = np.nonzero(disp > 4)
    z = Kr[0, 0] * base / disp[v, u]
    keep = z < 3.0
    u, v, z = u[keep], v[keep], z[keep]
    X = np.stack([(u - Kr[0, 2]) / Kr[0, 0] * z, (v - Kr[1, 2]) / Kr[1, 1] * z, z], 1)
    I = L[v, u].astype(np.float32)
    out = [f"frame {f}: {len(X)} stereo pts, median depth {np.median(z):.2f} m"]
    for lab, T in [("T_a_rect (exported)", T_a_rect), ("identity", np.eye(4)), ("inv(T_a_rect)", np.linalg.inv(T_a_rect))]:
        Xa = X @ T[:3, :3].T + T[:3, 3]
        uv = Xa[:, :2] / Xa[:, 2:3] * [Ka[0, 0], Ka[1, 1]] + [Ka[0, 2], Ka[1, 2]]
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1) & (Xa[:, 2] > 0.1)
        x, y = uv[ok, 0], uv[ok, 1]
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - x0, y - y0
        Af = A.astype(np.float32)
        a = (Af[y0, x0] * (1 - fx) * (1 - fy) + Af[y0, x0 + 1] * fx * (1 - fy)
             + Af[y0 + 1, x0] * (1 - fx) * fy + Af[y0 + 1, x0 + 1] * fx * fy)
        out.append(f"  {lab:22s} in-view {ok.mean():.2f}  NCC {np.corrcoef(a, I[ok])[0, 1]:.3f}")
    print("\n".join(out))
