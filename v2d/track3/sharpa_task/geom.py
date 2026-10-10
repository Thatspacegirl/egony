"""Pose / quaternion / mesh-geometry utilities (numpy + scipy + trimesh only).

Quaternions are wxyz everywhere in this package (ManoSharpaData convention).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


# ----------------------------------------------------------------------------- quaternions
def qnorm(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-9):
        raise ValueError("zero quaternion")
    return q / n


def rot_from_wxyz(q: np.ndarray) -> R:
    return R.from_quat(np.asarray(q, dtype=np.float64), scalar_first=True)


def wxyz_from_rot(r: R) -> np.ndarray:
    return r.as_quat(scalar_first=True)


def hemisphere_continuous(q: np.ndarray, axis: int = 0) -> np.ndarray:
    """Flip signs so consecutive quaternions along ``axis`` have non-negative dot product."""
    q = np.array(q, dtype=np.float64, copy=True)
    q = np.moveaxis(q, axis, 0)
    for t in range(1, q.shape[0]):
        flip = (q[t] * q[t - 1]).sum(-1) < 0.0
        q[t][flip] *= -1.0
    return np.moveaxis(q, 0, axis)


def fill_invalid_poses(pos: np.ndarray, quat: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill invalid frames of one trajectory: lerp/slerp inside gaps, hold at the ends.

    pos (T,3), quat (T,4) wxyz, valid (T,) bool. Returns filled copies and the number of filled frames.
    """
    pos = np.array(pos, dtype=np.float64, copy=True)
    quat = np.array(quat, dtype=np.float64, copy=True)
    valid = np.asarray(valid, dtype=bool)
    T = len(valid)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise ValueError("trajectory has no valid frame")
    missing = np.flatnonzero(~valid)
    if missing.size == 0:
        return pos, qnorm(quat), 0
    t_all = np.arange(T, dtype=np.float64)
    for k in range(3):
        pos[:, k] = np.interp(t_all, idx.astype(np.float64), pos[idx, k])
    rots = rot_from_wxyz(qnorm(quat[idx]))
    if idx.size == 1:
        quat[:] = wxyz_from_rot(rots)[0]
    else:
        slerp = Slerp(idx.astype(np.float64), rots)
        tt = np.clip(t_all, idx[0], idx[-1])
        quat = wxyz_from_rot(slerp(tt))
    return pos, qnorm(quat), int(missing.size)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking unit vector a onto unit vector b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # 180 deg: any axis orthogonal to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return R.from_rotvec(np.pi * axis).as_matrix()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def frame_from_two_dirs(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Orthonormal 3x3 with columns [primary, secondary_orth, primary x secondary_orth]."""
    p = np.asarray(primary, float)
    p = p / np.linalg.norm(p)
    s = np.asarray(secondary, float) - p * (p @ secondary)
    if np.linalg.norm(s) < 1e-8:
        raise ValueError("degenerate frame directions")
    s = s / np.linalg.norm(s)
    return np.stack([p, s, np.cross(p, s)], axis=1)


def apply_rigid(R_: np.ndarray, t: np.ndarray, pos: np.ndarray, quat: np.ndarray | None = None):
    """Apply x' = R x + t to positions (...,3) and (optionally) left-multiply quats (...,4 wxyz)."""
    pos2 = np.asarray(pos, float) @ R_.T + t
    if quat is None:
        return pos2
    shp = np.asarray(quat).shape
    q2 = wxyz_from_rot(R.from_matrix(R_) * rot_from_wxyz(np.asarray(quat, float).reshape(-1, 4)))
    return pos2, q2.reshape(shp)


def world_vertices(verts_body: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Body-frame vertices (V,3) -> world for one pose."""
    return verts_body @ rot_from_wxyz(quat_wxyz).as_matrix().T + pos


# ----------------------------------------------------------------------------- mass properties
# Plausible masses (kg) for the Track 3 roster: rough household-object estimates, NOT measured
# (Track 3 releases no physical properties; public URDFs use a 0.3 kg placeholder for everything).
MASS_PRIOR_KG = {
    "white_pot": 0.90,
    "white_pot_lid": 0.35,
    "blue_cup": 0.30,
    "beige_cup": 0.30,
    "plastic_dish_rack": 0.60,
    "wooden_spoon": 0.05,
    "mini_sweeper": 0.08,
    "mini_dust_pan": 0.12,
    "water_pitcher": 0.45,
    "wooden_piece_1": 0.15,
    "wooden_piece_2": 0.15,
}
EFFECTIVE_DENSITY = 250.0  # kg / m^3 of convex-hull volume (hollow household objects)
MASS_RANGE = (0.03, 1.5)


def mass_properties(vertices: np.ndarray, faces: np.ndarray, name: str, mass: float | None = None) -> dict:
    """Mass, COM (hull centroid) and inertia about the COM in body axes, from the convex hull.

    The hull is used because reconstructed meshes are rarely watertight; inertia is that of a
    uniform-density solid hull scaled to ``mass``.
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    hull = mesh.convex_hull
    vol = float(hull.volume)
    if vol <= 0:
        raise ValueError(f"{name}: degenerate convex hull")
    source = "given"
    if mass is None:
        if name in MASS_PRIOR_KG:
            mass, source = MASS_PRIOR_KG[name], "name_prior"
        else:
            mass, source = float(np.clip(vol * EFFECTIVE_DENSITY, *MASS_RANGE)), "hull_volume"
    mass = float(mass)
    if not (MASS_RANGE[0] <= mass <= 5.0):
        raise ValueError(f"{name}: implausible mass {mass} kg")
    com = np.asarray(hull.center_mass, float)
    inertia = np.asarray(hull.moment_inertia, float) * (mass / float(hull.mass))
    inertia = 0.5 * (inertia + inertia.T)
    return {
        "mass": mass,
        "mass_source": source,
        "com": com,
        "inertia": inertia,
        "hull_volume_m3": vol,
        "radius": float(np.linalg.norm(vertices - vertices.mean(0), axis=1).max()),
    }


# ----------------------------------------------------------------------------- table / support
def bottom_heights(verts_body: list[np.ndarray], pos: np.ndarray, quat: np.ndarray, up: np.ndarray) -> np.ndarray:
    """min_v up.(R v + p) per (frame, body). pos (T,B,3), quat (T,B,4)."""
    T, B = pos.shape[:2]
    out = np.empty((T, B))
    for b in range(B):
        Rm = rot_from_wxyz(quat[:, b]).as_matrix()  # (T,3,3)
        # up . (R v) = (R^T up) . v
        u_body = np.einsum("tji,j->ti", Rm, up)  # (T,3)
        out[:, b] = (u_body @ verts_body[b].T).min(axis=1) + pos[:, b] @ up
    return out


def estimate_table_height(bottoms: np.ndarray, speeds: np.ndarray, still_speed: float = 0.02) -> float:
    """Robust table height from object bottoms on frames where that object is (nearly) still.

    Uses the 10th percentile of still-frame bottoms over all bodies (objects resting on other
    objects sit higher; the lowest resting cluster is the table).
    """
    still = speeds < still_speed
    vals = bottoms[still] if still.any() else bottoms.ravel()
    return float(np.percentile(vals, 10.0))


def speeds(pos: np.ndarray, fps: float) -> np.ndarray:
    """Central-difference linear speed (T,B) in m/s."""
    v = np.gradient(pos, axis=0) * fps
    return np.linalg.norm(v, axis=-1)
