"""Small geometry helpers for Track 1 (CPU): floor plane from MoGe-2 depth, point back-projection."""
from __future__ import annotations

import numpy as np


def unpack(bits, W):
    return np.unpackbits(bits, axis=-1, count=W).astype(bool)


def backproject(D, K, mask=None, step=4):
    h, w = D.shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    z = D[ys, xs]
    ok = z > 0
    if mask is not None:
        ok &= mask[ys, xs]
    z = z[ok]; u = xs[ok]; v = ys[ok]
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)


def ransac_plane(P, iters=400, thr=0.03, seed=0):
    rng = np.random.default_rng(seed)
    best, bn = None, -1
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[1]) < 0.7:                      # floor normal is roughly the camera's vertical axis
            continue
        d = -n @ a
        cnt = int((np.abs(P @ n + d) < thr).sum())
        if cnt > bn:
            best, bn = (n, d), cnt
    n, d = best
    inl = np.abs(P @ n + d) < thr
    # least-squares refit on inliers
    Q = P[inl]; c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][-1]
    d = -n @ c
    if d < 0:                                     # camera centre on the positive side
        n, d = -n, -d
    return n, float(d), float(inl.mean())


def floor_plane(D, M, K, r=None, n_frames=12, lower_frac=0.45):
    """D: (N,h,w) depth (metres, 0 invalid) at mask resolution; M: masks dict (t1_masks.py); K at that resolution."""
    import cv2
    W, H = int(M["W"]), int(M["H"])
    N = len(D)
    idx = np.linspace(0, N - 1, min(n_frames, N)).round().astype(int)
    pts = []
    ker = np.ones((15, 15), np.uint8)
    for i in idx:
        fg = unpack(M["human"][i], W) | unpack(M["object"][i], W)
        fg = cv2.dilate(fg.astype(np.uint8), ker) > 0
        m = ~fg
        m[: int(H * (1 - lower_frac))] = False
        d = np.asarray(D[i], np.float32) * (1.0 if r is None else float(r[i]))
        pts.append(backproject(d, K, m, step=6))
    P = np.concatenate(pts)
    P = P[P[:, 2] < 12]
    return ransac_plane(P)
