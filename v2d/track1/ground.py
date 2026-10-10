"""Feet-on-floor depth refinement for Track 1 episode dicts (camera frame, CPU).  EXPERIMENTAL: accept only if it
improves CD-H / CD-O on the FORM-HOI val set with real model outputs.

Monocular human depth drifts along the viewing ray; the static floor does not.  With the human's own feet:
  1. per frame, the kit-body vertices (603, kit asset) in the camera frame;
  2. a floor plane n.x = d (RANSAC over the per-frame lowest vertices along the current up estimate, then least
     squares on the inliers; up initialised from the median body axis);
  3. per frame the foot height h_t (mean of the 3 lowest vertices; the asset has ~6 sole vertices) and the factor
     that puts that foot point f_t on the plane along ITS viewing ray, lam_t = 1 - h_t / (n . f_t);
  4. frames with |h_t| < `contact_h` get weight 1 (standing), others 0; log(lam) is smoothed with a robust Whittaker
     smoother (weak prior to 0) and the shift (lam_t - 1) f_t is applied to the human root translation AND the object
     translation (the object keeps its offset from the human, so contact/PEN are unchanged).
Body size/articulation are untouched.  Returns the new dict and a report.
"""
from __future__ import annotations

import numpy as np

from smoothing import whittaker


def _body():
    import t1lib as L
    return L.body()


def human_vertices(ep):
    pose = np.asarray(ep["pose"], np.float64)
    mp = np.concatenate([pose, np.repeat(np.asarray(ep["scales"], np.float64)[None], len(pose), 0)], 1)
    V, J = _body()(mp, ep["shape"])
    return V, J


def fit_floor(V, frames, iters=300, thr=0.03, k=5, seed=0):
    """plane (n, d) with n pointing UP from the k lowest vertices of each given frame (RANSAC + SVD refit, 3 rounds,
    up initialised as camera -y)."""
    rng = np.random.default_rng(seed)
    Vf = V[frames]
    up = np.array([0.0, -1.0, 0.0])
    inl_frac = 0.0
    for _ in range(3):
        h = Vf @ up
        low = np.take_along_axis(Vf, np.argsort(h, 1)[:, :k, None], 1).reshape(-1, 3)
        best = None
        for _ in range(iters):
            p = low[rng.choice(len(low), 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0]); nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n = n / nn
            if n @ up < 0:
                n = -n
            if n @ up < 0.7:
                continue
            inl = np.abs(low @ n - n @ p[0]) < thr
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None:
            break
        P = low[best]
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c)
        up = vt[2] if vt[2] @ up > 0 else -vt[2]
        inl_frac = float(best.mean())
    return up, float(up @ c), inl_frac


def refine(ep, scored, contact_h=0.08, lam_w=300.0, prior=1e-3):
    ep = {k: np.array(v, copy=True) for k, v in ep.items()}
    V, J = human_vertices(ep)
    T = len(V)
    sc = np.asarray(sorted(scored))
    n, d, inl = fit_floor(V, sc)
    H = V @ n - d
    lo = np.argsort(H, 1)[:, :3]                                 # the asset has only ~6 sole vertices
    h = np.take_along_axis(H, lo, 1).mean(1)                     # foot height per frame
    f = np.take_along_axis(V, lo[:, :, None], 1).mean(1)          # the foot point; we move along ITS ray
    nf = f @ n                                                    # ~ d (< 0: floor below the camera): well conditioned
    lam = 1.0 - h / np.where(np.abs(nf) < 1e-3, -1e-3, nf)
    w = (np.abs(h) < contact_h).astype(float)
    y = np.log(np.clip(lam, 0.5, 2.0))
    ys = whittaker(np.where(w > 0, y, 0.0), lam_w, w=w + prior, iters=3, robust="huber")
    lam_s = np.exp(ys)
    shift = (lam_s - 1.0)[:, None] * f                            # metres, camera frame (whole human + object)
    ep["pose"][:, :3] += shift / 0.1 * np.array([1.0, -1.0, -1.0])
    ep["object_translation"] = ep["object_translation"] + shift
    rep = {"floor_n": n.round(4).tolist(), "floor_d": round(d, 4), "inlier_frac": round(inl, 3),
           "contact_frac": round(float(w[sc].mean()), 3), "h_med_cm": round(float(np.median(h[sc]) * 100), 2),
           "lam_range": [round(float(lam_s.min()), 4), round(float(lam_s.max()), 4)],
           "shift_max_cm": round(float(np.linalg.norm(shift, axis=1).max() * 100), 2)}
    return ep, rep
