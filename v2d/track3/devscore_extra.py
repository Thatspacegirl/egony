"""Follow-up sweep: isolate how much FRAME-0 error matters vs. per-frame error (frame-0 alignment amplification)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devscore as D
from scipy.spatial.transform import Rotation as Rot

def frame0_rot(deg):
    def f(P, rng):
        Q = P.copy()
        for b in range(P.shape[1]):
            e = Rot.from_rotvec(D._unit(rng.normal(size=3)) * np.deg2rad(deg))
            Q[0, b, 3:] = (Rot.from_quat(P[0, b, 3:]) * e).as_quat()
        return Q
    return f
def frame0_pos(m):
    def f(P, rng):
        Q = P.copy(); Q[0, :, :3] += np.stack([D._unit(rng.normal(size=3)) for _ in range(P.shape[1])]) * m; return Q
    return f
def jitter_skip0(s, d):
    def f(P, rng):
        Q = D.p_jitter(s, d)(P, rng); Q[0] = P[0]; return Q
    return f
def smooth_noise(s, d, win=15):
    """temporally correlated (low-pass) noise, frame 0 included"""
    def f(P, rng):
        T, B, _ = P.shape; k = np.ones(win) / win
        def lp(x): return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, x) * np.sqrt(win)
        Q = P.copy(); Q[..., :3] += lp(rng.normal(size=(T, B * 3))).reshape(T, B, 3) * s
        rv = lp(rng.normal(size=(T, B * 3))).reshape(-1, 3) * np.deg2rad(d)
        Q[..., 3:] = (Rot.from_quat(P[..., 3:].reshape(-1, 4)) * Rot.from_rotvec(rv)).as_quat().reshape(T, B, 4)
        return Q
    return f
eps = [0, 2, 11, 12, 13, 18, 21, 22, 23, 24, 26, 27, 31, 39, 40, 41, 42, 43, 44, 45]
for d in (2, 5, 10): D.run(eps, f"frame-0-only rot err {d}deg (all objs)", frame0_rot(d))
for m in (0.01, 0.03): D.run(eps, f"frame-0-only pos err {m*100:.0f}cm (all objs)", frame0_pos(m))
D.run(eps, "jitter 1cm/5deg but frame 0 exact", jitter_skip0(0.01, 5))
D.run(eps, "jitter 2cm/10deg but frame 0 exact", jitter_skip0(0.02, 10))
D.run(eps, "low-pass noise 1cm/5deg (15-frame)", smooth_noise(0.01, 5))
