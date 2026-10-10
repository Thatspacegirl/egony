"""Stage 2: sample stereo depth around every raw 2D hand keypoint (CPU).

Per frame with at least one detection: rectified-left depth (FoundationStereo if its PNG exists, else SGBM), warped
into the undistorted cam_a grid at scale 0.5 (t3_depth_warp.warp_depth, z-buffered), then a 9x9 window of cam_a-frame
z values around each keypoint of each raw detection is stored (0 = no depth). Lifting (lift_hands.py) is tuned on
these samples without recomputing stereo.

Output HANDS/<split>/episode_X/depth_samples_{fs,sgbm}.npz (fs only when the episode's FoundationStereo depth is complete)
  win  (P,T,K,21,81) f2   cam_a-frame depth (m) in the window, row-major, 0 = none
  src  (T,) i1            0 no detection / skipped, 1 FoundationStereo, 2 SGBM
  source  str             'foundationstereo' (only if the episode's FS depth is complete) or 'sgbm' 

  /mnt/secondary/v2d/envs/t3-hands/bin/python -I sample_depth.py --episodes all --workers 3 [--prefer sgbm]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handlib as hl  # noqa: E402


def run(args):
    ep_dir, prefer, overwrite = args
    import cv2
    cv2.setNumThreads(3)
    hl.track3_import()
    from t3_depth_warp import warp_depth
    meta, Ka, Kr, T_a_rect = hl.load_meta(ep_dir)
    od = hl.out_dir(meta)
    ds = hl.RectDepth(ep_dir, meta, prefer)
    out = f"{od}/depth_samples_{'fs' if ds.use_fs else 'sgbm'}.npz"
    det_p = f"{od}/det_mp_cam_a.npz"
    if not os.path.exists(det_p):
        return f"MISSING {det_p}"
    legacy = f"{od}/depth_samples.npz"  # first run (2026-10-08) wrote SGBM samples under this name
    if not ds.use_fs and os.path.exists(legacy) and not os.path.exists(out):
        os.rename(legacy, out)
    if os.path.exists(out) and not overwrite:
        return f"skip {out}"
    det = np.load(det_p)
    lm = det["lm2d"]
    P, T, K = lm.shape[:3]
    s = hl.DEPTH_SCALE
    Ks = Ka.copy()
    Ks[:2] *= s
    W, H = int(round(meta["images"]["cam_a"]["size_wh"][0] * s)), int(round(meta["images"]["cam_a"]["size_wh"][1] * s))
    w = hl.WIN
    dy, dx = np.mgrid[-w:w + 1, -w:w + 1]
    dy, dx = dy.ravel(), dx.ravel()
    win = np.zeros((P, T, K, 21, (2 * w + 1) ** 2), np.float16)
    src = np.zeros(T, np.int8)
    t0 = time.time()
    has = ~np.isnan(lm[..., 0, 0])  # (P,T,K)
    for f in range(T):
        if not has[:, f].any():
            continue
        d, src[f] = ds.get(f)
        dw = warp_depth(d, Kr, T_a_rect, Ks, (W, H))
        pad = np.pad(dw, w)
        for p in range(P):
            for k in range(K):
                if not has[p, f, k]:
                    continue
                u = np.round(lm[p, f, k, :, :2] * s).astype(int)  # (21,2)
                ok = (u[:, 0] >= 0) & (u[:, 0] < W) & (u[:, 1] >= 0) & (u[:, 1] < H)
                uu = np.clip(u, 0, [W - 1, H - 1])
                vals = pad[uu[:, 1, None] + w + dy[None], uu[:, 0, None] + w + dx[None]]
                vals[~ok] = 0
                win[p, f, k] = vals
    np.savez_compressed(out, win=win, src=src, scale=np.array(s), K_s=Ks, source=np.array(ds.source))
    nf = int((src > 0).sum())
    return (f"{meta['split']}/episode_{meta['episode']:06d} frames={nf}/{T} fs={int((src == 1).sum())} "
            f"sgbm={int((src == 2).sum())} {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--splits", default="public,evaluation")
    ap.add_argument("--prefer", default="fs", choices=("fs", "sgbm"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    jobs = [(d, a.prefer, a.overwrite) for d in hl.episode_dirs(a.episodes, a.splits.split(","))]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for msg in pool.imap_unordered(run, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())
