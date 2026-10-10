"""Warp rectified-left stereo depth into the undistorted ego_cam_a pinhole grid (CPU, z-buffered forward splat).

Library:  warp_depth(depth_src, K_src, T_dst_src, K_dst, size_dst) -> depth_dst (metres, 0 = no data)
CLI (one episode; reads depth_rect/*.png written by t3_stereo_depth.py in v2d inverse-depth uint16 format):
  python -I t3_depth_warp.py --ep_dir FRAMES/<split>/episode_X --depth_dir DEPTH/<split>/episode_X [--scale 0.5]
writes DEPTH/<split>/episode_X/depth_cam_a_s{scale}/000000.png (same uint16 inverse-depth encoding) + K.txt + sync.json.
cam_a frame j gets the depth of stereo frame src[j] = j - lag (per-frame lag from t3_sync.py: the cam_a stream runs
1-2 frames behind the stereo pair), so RGB and depth are simultaneous.
--test_sgbm FRAME: CPU self-test with OpenCV SGBM depth instead of FoundationStereo; writes an overlay PNG.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def decode_inv_depth(png):
    a = png.astype(np.float32)
    d = np.zeros_like(a)
    v = a > 0
    d[v] = 65535.0 / a[v] - 1.0
    d[a >= 65535] = 0.0
    return d


def encode_inv_depth(d):
    # v2d DepthImage convention: pixel = 65535/(depth_m+1); "no data" (depth 0) MUST be 65535, because
    # v2d.common.datatypes.DepthImage.from_array decodes pixel 0 as depth = +inf (which FoundationPose's
    # `obs_depth > 0.001` validity tests would treat as a valid measurement).  Valid depths map to [1, 65534].
    out = np.full(d.shape, 65535, np.uint16)
    v = (d > 0) & np.isfinite(d)
    out[v] = np.clip(np.round(65535.0 / (d[v] + 1.0)), 1, 65534).astype(np.uint16)
    return out


def warp_depth(depth_src, K_src, T_dst_src, K_dst, size_dst, max_depth=5.0, fill=True):
    h, w = depth_src.shape
    v, u = np.nonzero((depth_src > 0) & (depth_src < max_depth))
    z = depth_src[v, u]
    X = np.stack([(u - K_src[0, 2]) / K_src[0, 0] * z, (v - K_src[1, 2]) / K_src[1, 1] * z, z], 1)
    Xd = X @ T_dst_src[:3, :3].T + T_dst_src[:3, 3]
    zd = Xd[:, 2]
    W, H = size_dst
    ok = zd > 0.05
    ud = Xd[ok, 0] / zd[ok] * K_dst[0, 0] + K_dst[0, 2]
    vd = Xd[ok, 1] / zd[ok] * K_dst[1, 1] + K_dst[1, 2]
    zd = zd[ok]
    ui, vi = np.round(ud).astype(np.int64), np.round(vd).astype(np.int64)
    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, zd = ui[inb], vi[inb], zd[inb]
    out = np.full(H * W, np.inf, np.float32)
    np.minimum.at(out, vi * W + ui, zd.astype(np.float32))
    out = out.reshape(H, W)
    out[~np.isfinite(out)] = 0
    if fill:  # close 1-2 px splat holes with the nearest (min) valid neighbour, never across big gaps
        big = np.where(out > 0, out, 1e6).astype(np.float32)
        mn = cv2.erode(big, np.ones((3, 3), np.uint8))
        hole = (out == 0) & (mn < 1e6)
        cnt = cv2.filter2D((out > 0).astype(np.float32), -1, np.ones((3, 3), np.float32))
        hole &= cnt >= 3
        out[hole] = mn[hole]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_dir", required=True)
    ap.add_argument("--depth_dir")
    ap.add_argument("--scale", type=float, default=0.5, help="cam_a output scale (0.5 -> 1014x760, f=520)")
    ap.add_argument("--test_sgbm", type=int, default=None)
    ap.add_argument("--out_png", default=None)
    ap.add_argument("--no_sync", action="store_true", help="ignore t3_sync.py lags (cam_a frame j <- stereo frame j)")
    a = ap.parse_args()
    meta = json.load(open(f"{a.ep_dir}/meta.json"))
    K_src = np.array(meta["images"]["left"]["K"])
    Ka = np.array(meta["images"]["cam_a"]["K"])
    W, H = meta["images"]["cam_a"]["size_wh"]
    s = a.scale
    K_dst = Ka.copy()
    K_dst[:2] *= s  # same convention as v2d run_video_to_poses --target_width/--target_height
    size = (int(round(W * s)), int(round(H * s)))
    T = np.array(meta["transforms"]["T_a_rect"])
    if a.test_sgbm is not None:
        f = a.test_sgbm
        L = cv2.imread(f"{a.ep_dir}/left/{f:06d}.jpg", 0)
        R = cv2.imread(f"{a.ep_dir}/right/{f:06d}.jpg", 0)
        sg = cv2.StereoSGBM_create(0, 160, 5, P1=200, P2=800, uniquenessRatio=10, speckleWindowSize=100,
                                   speckleRange=2)
        disp = sg.compute(L, R).astype(np.float32) / 16
        d = np.where(disp > 2, meta["stereo"]["fx"] * meta["stereo"]["baseline_m"] / np.maximum(disp, 1e-3), 0)
        dw = warp_depth(d, K_src, T, K_dst, size)
        A = cv2.resize(cv2.imread(f"{a.ep_dir}/cam_a/{f:06d}.jpg"), size, interpolation=cv2.INTER_AREA)
        # edge agreement: depth discontinuities vs image edges in cam_a
        de = cv2.Canny(np.clip(dw * 120, 0, 255).astype(np.uint8), 30, 90) > 0
        ie = cv2.Canny(cv2.cvtColor(A, cv2.COLOR_BGR2GRAY), 60, 160) > 0
        dist = cv2.distanceTransform((~ie).astype(np.uint8), cv2.DIST_L2, 3)
        print(f"valid {np.mean(dw > 0):.2f} of cam_a@{s} grid; median depth {np.median(dw[dw > 0]):.2f} m; "
              f"median dist depth-edge -> image-edge {np.median(dist[de]):.2f} px (n={de.sum()})")
        vis = cv2.applyColorMap(np.clip(255 * (1 - (dw - 0.3) / 1.7), 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[dw == 0] = 0
        over = cv2.addWeighted(A, 0.55, vis, 0.45, 0)
        over[de] = (255, 255, 255)
        cv2.imwrite(a.out_png or "/tmp/depth_warp_check.png", over)
        return
    od = f"{a.depth_dir}/depth_cam_a_s{s:g}"
    os.makedirs(od, exist_ok=True)
    np.savetxt(f"{od}/K.txt", K_dst)
    # cam_a frame j shows the scene of stereo frame src[j] = j - lag (t3_sync.py); without a sync file src[j] = j
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from t3_sync import load_lag, stereo_index_for_cam_a
    n = meta["n_frames"]
    lag = load_lag(meta["split"], f"episode_{meta['episode']:06d}", n) if not a.no_sync else np.zeros(n, int)
    src = stereo_index_for_cam_a(lag) if lag.any() else np.arange(n)
    for j in range(n):
        d = decode_inv_depth(cv2.imread(f"{a.depth_dir}/depth_rect/{src[j]:06d}.png", cv2.IMREAD_UNCHANGED))
        cv2.imwrite(f"{od}/{j:06d}.png", encode_inv_depth(warp_depth(d, K_src, T, K_dst, size)))
    json.dump(dict(src_stereo_frame=src.tolist(), lag_applied=bool(lag.any())), open(f"{od}/sync.json", "w"))
    print("wrote", od, "lag values", sorted(set(lag.tolist())))


if __name__ == "__main__":
    main()
