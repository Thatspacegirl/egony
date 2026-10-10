"""Frame-0 physical-consistency checks (numpy/scipy/trimesh; no rtree needed, runs in either env)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geom import rot_from_wxyz


def support_gap_report(verts_body, pos0, quat0, names, top_z, center_xy, size_xy) -> dict:
    """Per object at frame 0: lowest vertex height minus support top (negative = interpenetration)."""
    out = {}
    half = 0.5 * np.asarray(size_xy)
    for k, name in enumerate(names):
        V = verts_body[k] @ rot_from_wxyz(quat0[k]).as_matrix().T + pos0[k]
        i = int(np.argmin(V[:, 2]))
        inside = bool(np.all(np.abs(V[i, :2] - center_xy) <= half))
        out[name] = {"gap_m": float(V[i, 2] - top_z), "lowest_point": V[i].tolist(), "inside_footprint": inside}
    return out


def _world_samples(mesh, pos, quat, n, seed):
    pts, fid = mesh.sample(n, return_index=True, seed=seed) if _sample_has_seed(mesh) else mesh.sample(n, return_index=True)
    Rm = rot_from_wxyz(quat).as_matrix()
    return pts @ Rm.T + pos, mesh.face_normals[fid] @ Rm.T


def _sample_has_seed(mesh) -> bool:
    import inspect

    return "seed" in inspect.signature(mesh.sample).parameters


def pair_penetration_report(meshes, pos0, quat0, names, n_samples: int = 20000) -> dict:
    """Approximate object-object penetration at frame 0.

    For each ordered pair (A, B): points sampled on A's surface are matched to the nearest of
    ``n_samples`` points on B's surface; the signed distance uses B's outward face normal at that sample.
    ``max_penetration_m`` = deepest A point behind B's surface within 2 cm of it (pseudo-signed distance,
    ~2 mm resolution; thin open scans may give false positives near rims).
    """
    out = {}
    B = len(meshes)
    for a in range(B):
        for b in range(B):
            if a == b:
                continue
            pa, _ = _world_samples(meshes[a], pos0[a], quat0[a], 5000, 1)
            pb, nb = _world_samples(meshes[b], pos0[b], quat0[b], n_samples, 2)
            d, j = cKDTree(pb).query(pa)
            signed = np.einsum("ij,ij->i", pa - pb[j], nb[j])
            near = d < 0.02
            pen = float(np.clip(-signed[near], 0, None).max()) if near.any() else 0.0
            out[f"{names[a]}->{names[b]}"] = {"min_dist_m": float(d.min()), "max_penetration_m": pen,
                                              "frac_within_5mm": float((d < 0.005).mean())}
    return out
