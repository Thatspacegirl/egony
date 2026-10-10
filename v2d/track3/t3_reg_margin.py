"""DEV: registration margin of a mesh vs the public scan, replicating AUC.py _register_object_body: run the scorer's
ICP from all 25 PCA starts and report every distinct converged basin (rotation, symmetric mean distance).  The margin
= distance of the best basin whose rotation differs by > 30 deg from the winner, minus the winner's.
  /mnt/secondary/v2d/scratch/venv/bin/python -I t3_reg_margin.py OBJ MESH [EP SLOT]
"""
import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

KIT = "/mnt/secondary/v2d/kit/v2d_submission_kit"
DS = os.path.expanduser("~/TestingGrounds/egony/video_to_data_challenge/track_3")
obj, mesh = sys.argv[1], sys.argv[2]
ep, slot = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 0)
sys.argv = [sys.argv[0], KIT, DS]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devscore as D  # noqa: E402
from v2dlb.mesh_budget import budget_mesh  # noqa: E402
A = D.MODS["AUC"]
v, f = budget_mesh(mesh, 4096, 4096)
g = D.ref_geometry(obj)
seed = 1_000_003 * ep + 101 * slot
cc, cr = A._surface_moments(v, f)
rc, rr = A._surface_moments(g["v"], g["f"])
cp = (A._sample_mesh_surface(v, f, 3000, seed) - cc) * (rr / cr)
rp = A._sample_mesh_surface(g["v"], g["f"], 3000, seed) - rc
tree = cKDTree(rp)
res = []
for start in A._initial_rotations(rp, cp, A._ALIGNMENT_STARTS):
    R, t = start.copy(), rp.mean(0) - start @ cp.mean(0)
    prev = math.inf
    for _ in range(A._ICP_ITERATIONS):
        mv = cp @ R.T + t
        d, idx = tree.query(mv)
        keep = max(3, int(math.ceil(len(d) * A._TRIM_FRACTION)))
        ch = np.argpartition(d, keep - 1)[:keep] if keep < len(d) else np.arange(len(d))
        dR, dt = A._kabsch(mv[ch], rp[idx[ch]])
        R, t = dR @ R, dR @ t + dt
        err = float(np.mean(d[ch]))
        if math.isfinite(prev) and abs(prev - err) <= 1e-8 * max(prev, 1.0):
            break
        prev = err
    res.append((A._symmetric_mean_distance(rp, cp @ R.T + t) * 100, R))
res.sort(key=lambda x: x[0])
best = res[0]
alts = [(s, np.degrees(Rot.from_matrix(R @ best[1].T).magnitude())) for s, R in res[1:]]
far = [s for s, a in alts if a > 30]
print(f"{obj} {os.path.basename(mesh)}: winner {best[0]:.4f} cm; best other basin (>30 deg away) "
      f"{(min(far) if far else float('nan')):.4f} cm -> margin {(min(far) - best[0] if far else float('nan')):.4f} cm")
