"""Model-agnostic temporal smoother for Track 1 episode dicts (MHR params + object Sim(3) pose).

Every component is a (robust, optionally anchored) second-order Whittaker-Henderson smoother:

    x* = argmin_x  sum_t w_t |x_t - y_t|^2  +  lam * sum_t |x_{t-1} - 2 x_t + x_{t+1}|^2

which penalises exactly the second difference the ACC-H / ACC-O metrics measure, has no window edge
effects, handles gaps, and has cutoff period ~ 2*pi*lam**0.25 frames (lam 1e2 ~ 20 f, 1e3 ~ 35 f,
1e4 ~ 63 f, 1e5 ~ 112 f).

  * root rotation (pose[:, 3:6], Euler xyz, R = Rz Ry Rx) and object rotation are smoothed on SO(3):
    rotation -> unit quaternion -> sign-continuous path -> Whittaker on the 4 components -> renormalise.
    Never the raw Euler angles (they wrap at +-pi and pass gimbal lock in the FORM-HOI GT).
  * root translation (pose[:, 0:3]), body pose (pose[:, 6:136], hands separately) and object translation
    are smoothed as Euclidean signals.
  * frame-0 protection ("anchor"): the first scored frame f0 defines the scorer's Sim(3) for the whole
    episode. anchor='raw' pins x(f0) to the input value, anchor='light' pins it to a lightly smoothed value
    (lam_anchor), anchor='none' lets the smoother move it. The pin is a large weight in the same quadratic
    problem, so the trajectory stays C1-smooth through f0 (no kink -> no ACC spike).
  * robust='huber' re-weights frames with large residuals (IRLS), so isolated glitches are not smeared.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.linalg import solveh_banded
from scipy.spatial.transform import Rotation

POSE_TRANS = slice(0, 3)
POSE_ROT = slice(3, 6)
HAND_COLS = np.arange(68, 122)            # finger/thumb params (kit mesh_to_mhr_params.HAND_COLS)
BODY_COLS = np.setdiff1d(np.arange(6, 136), HAND_COLS)


def whittaker(y, lam, w=None, iters=1, robust=None, k=2.5):
    """Second-order Whittaker smoother along axis 0 of y [T, ...]. w: per-frame weights [T] (>=0)."""
    y = np.asarray(y, np.float64)
    T = y.shape[0]
    flat = y.reshape(T, -1)
    if lam <= 0 or T < 3:
        return y.copy()
    w = np.ones(T) if w is None else np.asarray(w, np.float64).copy()
    w0 = w.copy()
    # D^T D for the second difference, as symmetric banded (upper form, 3 bands)
    d0 = np.full(T, 6.0); d0[[0, -1]] = 1.0; d0[[1, -2]] = 5.0
    d1 = np.full(T - 1, -4.0); d1[[0, -1]] = -2.0
    d2 = np.ones(T - 2)
    x = flat
    for it in range(max(1, iters)):
        ab = np.zeros((3, T))
        ab[2] = lam * d0 + w
        ab[1, 1:] = lam * d1
        ab[0, 2:] = lam * d2
        x = solveh_banded(ab, w[:, None] * flat, lower=False)
        if robust != "huber" or it == iters - 1:
            break
        r = np.linalg.norm(flat - x, axis=1)
        s = 1.4826 * np.median(r) + 1e-12
        u = r / (k * s)
        w = w0 * np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
    return x.reshape(y.shape)


def _anchored_weights(T, anchor_frames, big=1e8):
    w = np.ones(T)
    if anchor_frames is not None:
        w[np.asarray(anchor_frames, int)] = big
    return w


def smooth_euclid(y, lam, anchor_frames=None, anchor_values=None, robust=None, iters=3):
    """Whittaker with optional pins: anchor_values replace y at anchor_frames before the (weighted) solve."""
    y = np.asarray(y, np.float64).copy()
    if lam <= 0:
        return y
    if anchor_frames is not None and anchor_values is not None:
        y[np.asarray(anchor_frames, int)] = anchor_values
    w = _anchored_weights(len(y), anchor_frames)
    return whittaker(y, lam, w, iters=iters if robust else 1, robust=robust)


def quat_continuous(q):
    q = np.asarray(q, np.float64).copy()
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    return q


def smooth_rotmats(R, lam, anchor_frames=None, anchor_R=None, robust=None, iters=3):
    """SO(3) smoothing via sign-continuous quaternions + Whittaker + renormalisation."""
    R = np.asarray(R, np.float64)
    if lam <= 0:
        return R.copy()
    q = quat_continuous(Rotation.from_matrix(R).as_quat())
    if anchor_frames is not None and anchor_R is not None:
        qa = Rotation.from_matrix(np.asarray(anchor_R).reshape(-1, 3, 3)).as_quat()
        for i, f in enumerate(np.atleast_1d(anchor_frames)):
            q[f] = qa[i] if np.dot(qa[i], q[f]) >= 0 else -qa[i]
    w = _anchored_weights(len(q), anchor_frames)
    qs = whittaker(q, lam, w, iters=iters if robust else 1, robust=robust)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    return Rotation.from_quat(qs).as_matrix()


def euler_to_R(e):
    return Rotation.from_euler("xyz", np.asarray(e).reshape(-1, 3)).as_matrix()


def R_to_euler(R, ref=None):
    """Euler xyz (scorer convention). If ref is given, pick the equivalent triple closest to it, so the
    stored parameters stay continuous (cosmetic: the scorer only uses the rotation)."""
    e = Rotation.from_matrix(R).as_euler("xyz")
    if ref is None:
        return e
    alt = np.stack([e[:, 0] + np.pi, np.pi - e[:, 1], e[:, 2] + np.pi], 1)
    def wrap(a, r):
        return r + (a - r + np.pi) % (2 * np.pi) - np.pi
    e1, e2 = wrap(e, ref), wrap(alt, ref)
    use2 = np.abs(e2 - ref).sum(1) < np.abs(e1 - ref).sum(1)
    return np.where(use2[:, None], e2, e1)


@dataclass
class SmoothCfg:
    lam_root_rot: float = 1e3
    lam_root_trans: float = 1e3
    lam_body: float = 1e2
    lam_hand: float = 1e2
    lam_obj_rot: float = 1e3
    lam_obj_trans: float = 1e3
    anchor: str = "light"          # none | raw | light
    lam_anchor: float = 10.0       # smoothing used to form the 'light' anchor value
    anchor_body: bool = True       # also pin body params at f0
    anchor_object: bool = False    # object pose is not part of the Sim(3); pin optional
    robust: str | None = None      # None | 'huber'

    def tag(self):
        anc = self.anchor + (f"{self.lam_anchor:g}" if self.anchor == "light" else "")
        return (f"rr{self.lam_root_rot:g}_rt{self.lam_root_trans:g}_b{self.lam_body:g}_h{self.lam_hand:g}"
                f"_or{self.lam_obj_rot:g}_ot{self.lam_obj_trans:g}_a{anc}{'_hub' if self.robust else ''}")


def smooth_episode(ep: dict, f0: int | None, cfg: SmoothCfg) -> dict:
    """Return a smoothed copy of a Track 1 episode dict. f0 = first scored frame (Sim(3) frame)."""
    out = {k: np.array(v, dtype=np.float64, copy=True) for k, v in ep.items()}
    pose = out["pose"]
    T = len(pose)
    af = None if (cfg.anchor == "none" or f0 is None) else np.array([int(f0)])

    def anchor_val(sig, lam_main, rot=False):
        if af is None:
            return None
        if cfg.anchor == "raw":
            return sig[af]
        if rot:
            return smooth_rotmats(sig, cfg.lam_anchor)[af]
        return smooth_euclid(sig, cfg.lam_anchor)[af]

    # root rotation on SO(3)
    Rr = euler_to_R(pose[:, POSE_ROT])
    Rs = smooth_rotmats(Rr, cfg.lam_root_rot, af, anchor_val(Rr, cfg.lam_root_rot, rot=True), cfg.robust)
    pose[:, POSE_ROT] = R_to_euler(Rs, ref=ep["pose"][:, POSE_ROT])
    # root translation
    tr = ep["pose"][:, POSE_TRANS]
    pose[:, POSE_TRANS] = smooth_euclid(tr, cfg.lam_root_trans, af, anchor_val(tr, cfg.lam_root_trans), cfg.robust)
    # body / hands
    abody = af if cfg.anchor_body else None
    b = ep["pose"][:, BODY_COLS]
    pose[:, BODY_COLS] = smooth_euclid(b, cfg.lam_body, abody, anchor_val(b, cfg.lam_body) if abody is not None else None, cfg.robust)
    h = ep["pose"][:, HAND_COLS]
    pose[:, HAND_COLS] = smooth_euclid(h, cfg.lam_hand, None, None, cfg.robust)
    out["pose"] = pose
    # object
    ao = af if cfg.anchor_object else None
    Ro = np.asarray(ep["object_rotation"], np.float64)
    out["object_rotation"] = smooth_rotmats(Ro, cfg.lam_obj_rot, ao, anchor_val(Ro, cfg.lam_obj_rot, rot=True) if ao is not None else None, cfg.robust)
    to = np.asarray(ep["object_translation"], np.float64)
    out["object_translation"] = smooth_euclid(to, cfg.lam_obj_trans, ao, anchor_val(to, cfg.lam_obj_trans) if ao is not None else None, cfg.robust)
    return out


def cfg_dict(cfg: SmoothCfg) -> dict:
    return asdict(cfg)
