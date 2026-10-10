"""Temporal offset between ego_cam_a and the stereo pair (ego_cam_c/b), per episode, from the images only (CPU).

Found 2026-10-08: the three released streams have equal frame counts but cam_a frame i+k shows the scene of stereo
frame i (k = 1 on public 12 / eval 1, k = 2 on public 21).  For frames with head rotation > --min_rot deg/frame,
the LEFT image is warped into cam_a (0.5 scale) with the stereo depth and T_a_rect (z-buffered splat) and the
gradient NCC against cam_a frames i+k, k in [-2, 5], is computed; the per-episode offset is the argmax of the mean
curve (with per-segment votes to detect a change inside the clip).
Output: SYNC/<split>/episode_X.json {a_lag, subframe, mean_ncc, votes, segments}; a_lag is used by
t3_depth_warp.py / t3_masks_to_left.py / t3_fpose_track.py / t3_assemble.py:  cam_a frame j <-> stereo frame j - a_lag.

  python -I t3_sync.py --eps public:21 evaluation:1 ...   (or --all)
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t3_depth_warp import decode_inv_depth  # noqa: E402

R0 = "/mnt/secondary/v2d/t3"
KS = list(range(-2, 6))


def grad(img):
    g = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def ncc(a, b, m):
    a = a[m]
    b = b[m]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def load_lag(split, e, n=None):
    """per-frame lag array (stereo frame i <-> cam_a frame i + lag[i]); zeros if no sync file."""
    p = f"{R0}/sync/{split}/{e}.json"
    if not os.path.exists(p):
        return np.zeros(n or 0, int)
    return np.array(json.load(open(p))["lag_per_frame"], int)


def stereo_index_for_cam_a(lag):
    """src[j] = stereo frame whose scene cam_a frame j shows (j - lag, inverted per frame, clipped)."""
    n = len(lag)
    src = np.full(n, -1)
    for i in range(n):
        j = i + lag[i]
        if 0 <= j < n:
            src[j] = i
    for j in range(n):  # frames not hit (lag change) -> nearest neighbour rule
        if src[j] < 0:
            src[j] = int(np.clip(j - lag[min(j, n - 1)], 0, n - 1))
    return src


def dp_lags(fr, C, switch_cost=1.5):
    """piecewise-constant lag sequence over the sampled frames fr (Viterbi on per-frame z-scored NCC curves)."""
    Z = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
    nK = C.shape[1]
    cost = -Z[0].copy()
    back = np.zeros(C.shape, int)
    for t in range(1, len(C)):
        trans = cost[None, :] + switch_cost * (1 - np.eye(nK))  # [to, from]
        back[t] = np.argmin(trans, 1)
        cost = trans[np.arange(nK), back[t]] - Z[t]
    path = [int(np.argmin(cost))]
    for t in range(len(C) - 1, 0, -1):
        path.append(back[t][path[-1]])
    return [KS[i] for i in path[::-1]]


def episode(sp, ep, step=3, min_rot=0.4, scale=0.5):
    e = f"episode_{ep:06d}"
    meta = json.load(open(f"{R0}/frames/{sp}/{e}/meta.json"))
    K = np.array(meta["images"]["left"]["K"])
    Ka = np.array(meta["images"]["cam_a"]["K"], float)
    Ka[:2] *= scale
    T = np.array(meta["transforms"]["T_a_rect"])
    W = np.load(f"{R0}/vo/{sp}/{e}/vo_cuvslam.npz")["world_T_rect"]
    n = meta["n_frames"]
    Wa, Ha = [int(round(x * scale)) for x in meta["images"]["cam_a"]["size_wh"]]
    curves, fr = [], []
    for thr in (min_rot, 0.2, 0.0):  # relax if the head hardly moves
        for f in range(3, n - 6, step):
            rot = np.degrees(np.linalg.norm(cv2.Rodrigues((np.linalg.inv(W[f - 1]) @ W[f + 1])[:3, :3])[0])) / 2
            if rot < thr or f in fr:
                continue
            d = decode_inv_depth(cv2.imread(f"{R0}/depth/{sp}/{e}/depth_rect/{f:06d}.png", -1))
            L = cv2.imread(f"{R0}/frames/{sp}/{e}/left/{f:06d}.jpg", 0).astype(np.float32)
            v, u = np.nonzero((d > 0.2) & (d < 3))
            z = d[v, u]
            X = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1) @ T[:3, :3].T + T[:3, 3]
            ua = np.round(X[:, 0] / X[:, 2] * Ka[0, 0] + Ka[0, 2]).astype(int)
            va = np.round(X[:, 1] / X[:, 2] * Ka[1, 1] + Ka[1, 2]).astype(int)
            ok = (ua >= 0) & (ua < Wa) & (va >= 0) & (va < Ha) & (X[:, 2] > 0.05)
            zb = np.full((Ha, Wa), np.inf, np.float32)
            np.minimum.at(zb, (va[ok], ua[ok]), X[ok, 2].astype(np.float32))
            sel = ok.copy()
            sel[ok] = X[ok, 2] <= zb[va[ok], ua[ok]] + 1e-4
            img = np.zeros((Ha, Wa), np.float32)
            img[va[sel], ua[sel]] = L[v[sel], u[sel]]
            m = np.zeros((Ha, Wa), np.uint8)
            m[va[sel], ua[sel]] = 1
            m = cv2.erode(cv2.dilate(m, np.ones((3, 3))), np.ones((7, 7))) > 0
            gi = grad(cv2.dilate(img, np.ones((2, 2))))
            row = []
            for k in KS:
                g = min(max(f + k, 0), n - 1)
                A = cv2.resize(cv2.imread(f"{R0}/frames/{sp}/{e}/cam_a/{g:06d}.jpg", 0), (Wa, Ha), interpolation=cv2.INTER_AREA)
                row.append(ncc(gi, grad(A), m))
            curves.append(row)
            fr.append(f)
        if len(curves) >= 8:
            break
    C = np.array(curves)
    mean = C.mean(0)
    i = int(np.argmax(mean))
    sub = KS[i] + (0.5 * (mean[i - 1] - mean[i + 1]) / (mean[i - 1] - 2 * mean[i] + mean[i + 1]) if 0 < i < len(KS) - 1 else 0)
    best = [KS[int(np.argmax(r))] for r in C]
    segs = []
    for s0 in range(0, n, 60):
        b = [x for f, x in zip(fr, best) if s0 <= f < s0 + 60]
        if b:
            segs.append([s0, int(np.bincount(np.array(b) - KS[0]).argmax() + KS[0]), len(b)])
    order = np.argsort(fr)
    frs = [fr[j] for j in order]
    lag_s = dp_lags(frs, C[order])
    lag = np.interp(np.arange(n), frs, lag_s)  # piecewise constant between samples (nearest)
    lag = np.array([lag_s[int(np.argmin(np.abs(np.array(frs) - t)))] for t in range(n)], int)
    changes = [[int(t), int(lag[t - 1]), int(lag[t])] for t in range(1, n) if lag[t] != lag[t - 1]]
    return dict(split=sp, episode=ep, a_lag=int(KS[i]), lag_per_frame=lag.tolist(), changes=changes,
                subframe=float(sub), n=len(C), mean_ncc=np.round(mean, 4).tolist(),
                ks=KS, votes={str(k): int(sum(1 for x in best if x == k)) for k in KS}, segments=segs,
                margin=float(np.sort(mean)[-1] - np.sort(mean)[-2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--step", type=int, default=2)
    a = ap.parse_args()
    eps = a.eps
    if a.all:
        eps = [f"{p.split('/')[-2]}:{int(p[-6:])}" for p in sorted(glob.glob(f"{R0}/frames/*/episode_*"))]
    for spec in eps:
        sp, ep = spec.split(":")
        r = episode(sp, int(ep), step=a.step)
        os.makedirs(f"{R0}/sync/{sp}", exist_ok=True)
        json.dump(r, open(f"{R0}/sync/{sp}/episode_{int(ep):06d}.json", "w"), indent=1)
        print(json.dumps({k: r[k] for k in ("split", "episode", "a_lag", "subframe", "n", "votes", "changes", "margin")}), flush=True)


if __name__ == "__main__":
    main()
