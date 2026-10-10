"""Pose-track post-processing in the VO world (library; used by t3_assemble.py).

  static_segments(contact, min_len)      boolean runs where the object is NOT touched by a hand
  clamp_static(T, segs)                  replace each untouched run by one robust pose (object rests): translation =
                                         median, rotation = chordal L2 mean of the rotations within 1.5 MAD
  smooth_track(T, w, sig_t, sig_r)       weighted Gaussian smoothing of translation and rotation (rotation via
                                         tangent-space residuals about a running anchor), w = per-frame confidence
All tracks are [N,4,4] world_T_obj.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def robust_mean_pose(T):
    t = T[:, :3, 3]
    med = np.median(t, 0)
    d = np.linalg.norm(t - med, axis=1)
    keep = d <= np.median(d) + 1.5 * (np.median(np.abs(d - np.median(d))) + 1e-4)
    R = Rotation.from_matrix(T[keep, :3, :3]).mean().as_matrix()
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.median(t[keep], 0)
    return M


def clamp_static(T, untouched, min_len=5, margin=2):
    """untouched: bool[N] (no hand contact).  Each run (shrunk by `margin` frames at inner ends, i.e. ends that border
    a touched frame) longer than min_len is replaced by its robust mean pose."""
    T = T.copy()
    N = len(T)
    segs = []
    for a, b in runs(untouched):
        a2 = a + (margin if a > 0 else 0)
        b2 = b - (margin if b < N else 0)
        if b2 - a2 >= min_len:
            T[a2:b2] = robust_mean_pose(T[a2:b2])
            segs.append((a2, b2))
    return T, segs


def smooth_track(T, w=None, sig_t=1.5, sig_r=1.5, keep=None):
    """Gaussian (sigma in frames) confidence-weighted smoothing; frames with keep=True are left untouched."""
    N = len(T)
    w = np.ones(N) if w is None else np.clip(np.nan_to_num(w, nan=0.0), 1e-3, None)
    out = T.copy()
    rad = int(np.ceil(3 * max(sig_t, sig_r)))
    R = Rotation.from_matrix(T[:, :3, :3])
    for i in range(N):
        if keep is not None and keep[i]:
            continue
        lo, hi = max(0, i - rad), min(N, i + rad + 1)
        k = np.arange(lo, hi)
        gt = np.exp(-0.5 * ((k - i) / max(sig_t, 1e-6)) ** 2) * w[k]
        gr = np.exp(-0.5 * ((k - i) / max(sig_r, 1e-6)) ** 2) * w[k]
        out[i, :3, 3] = (T[k, :3, 3] * gt[:, None]).sum(0) / gt.sum()
        rv = (R[i].inv() * R[k]).as_rotvec()
        out[i, :3, :3] = (R[i] * Rotation.from_rotvec((rv * gr[:, None]).sum(0) / gr.sum())).as_matrix()
    return out
