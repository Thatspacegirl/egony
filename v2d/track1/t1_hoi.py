"""Human-object interaction refinement for Track 1 episode dicts (camera frame; CPU).

contact_depth(): during the scored frames (= the hand-object CONTACT span by the competition's definition) the
object must touch a hand.  Monocular depth is the weakest axis of both tracks, while their 2D image positions are
reliable, so we move the object only along its viewing ray: t_obj(t) <- lambda_t * t_obj(t).  Per frame lambda* =
argmin_lambda min_{hand point h, object point o(lambda)} |h - o|, kept at 1 when the hand already touches
(distance < thr).  log(lambda) is then smoothed over the scored frames with a robust Whittaker smoother (frames
where no lambda reaches contact get weight 0, plus a weak prior to lambda = 1) and applied to every frame (nearest
scored value outside the span).  Hand points = the kit body asset's left_hand + right_hand roles.
"""
from __future__ import annotations

import numpy as np

from smoothing import whittaker


def _body():
    import t1lib as L
    return L.body()


def hand_points(epd, frames):
    b = _body()
    pose = np.asarray(epd["pose"], np.float64)[frames]
    params = np.concatenate([pose, np.broadcast_to(np.asarray(epd["scales"]).reshape(68), (len(pose), 68))], 1)
    V, _ = b(params, np.asarray(epd["shape"], np.float64).reshape(45))
    idx = np.concatenate([b.roles["left_hand"], b.roles["right_hand"]])
    return V[:, idx]


def mesh_points(mesh_vf, n=800, seed=0):
    import trimesh
    v, f = mesh_vf
    m = trimesh.Trimesh(v, f, process=False)
    p, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    return np.asarray(p)


def contact_depth(epd, mesh_vf, frames, lam_grid=None, thr=0.02, whit=300.0, prior=0.05, max_dev=0.35):
    from scipy.spatial import cKDTree
    frames = np.asarray(frames)
    lam_grid = np.exp(np.linspace(np.log(1 - max_dev), np.log(1 + max_dev), 71)) if lam_grid is None else lam_grid
    H = hand_points(epd, frames)
    P = mesh_points(mesh_vf)
    s = float(np.asarray(epd["object_scale"]).reshape(()))
    Rt = np.asarray(epd["object_rotation"])[frames]
    tt = np.asarray(epd["object_translation"])[frames]
    n = len(frames)
    lam_star = np.ones(n); d1 = np.zeros(n); dstar = np.zeros(n)
    for i in range(n):
        tree = cKDTree(H[i])
        base = s * P @ Rt[i].T
        X = base[None] + lam_grid[:, None, None] * tt[i][None, None]
        d = tree.query(X.reshape(-1, 3))[0].reshape(len(lam_grid), -1).min(1)
        j1 = int(np.argmin(np.abs(lam_grid - 1.0)))
        d1[i] = d[j1]
        if d1[i] < thr:
            lam_star[i], dstar[i] = 1.0, d1[i]
        else:
            # among lambdas reaching contact, the one closest to 1; else the global minimum
            ok = np.flatnonzero(d < thr)
            j = ok[np.argmin(np.abs(np.log(lam_grid[ok])))] if len(ok) else int(np.argmin(d))
            lam_star[i], dstar[i] = lam_grid[j], d[j]
    w = (dstar < thr).astype(float) + prior
    y = np.log(lam_star)
    # Whittaker with weights, then robust re-weighting (Huber-like) by hand
    ys = whittaker(y[:, None], whit, w, iters=1)[:, 0]
    r = np.abs(y - ys)
    sc = 1.4826 * np.median(r[w > 1]) + 1e-6 if (w > 1).any() else 1.0
    w2 = w * np.where(r <= 2.5 * sc, 1.0, 2.5 * sc / np.maximum(r, 1e-9))
    ys = whittaker(y[:, None], whit, w2, iters=1)[:, 0]
    lam_s = np.exp(ys)
    T = len(epd["object_translation"])
    lam_all = np.interp(np.arange(T), frames, lam_s)
    out = dict(epd)
    out["object_translation"] = np.asarray(epd["object_translation"]) * lam_all[:, None]
    info = {"d_before_med_cm": float(100 * np.median(d1)), "d_star_med_cm": float(100 * np.median(dstar)),
            "contact_frac_before": float((d1 < thr).mean()), "contact_frac_reachable": float((dstar < thr).mean()),
            "lambda_med": float(np.median(lam_s)), "lambda_p5_p95": [float(np.percentile(lam_s, 5)), float(np.percentile(lam_s, 95))]}
    return out, info
