"""DEV: stability of the official scorer's registration of a mesh onto the public scan under different sampling seeds
(the hidden evaluation uses seed 1_000_003*episode + 101*slot).  Prints, for N seeds, the angle between each seed's
chosen body rotation and the reference seed's (or a given true rotation); a robust mesh gives ~0 deg for all seeds.
  /mnt/secondary/v2d/scratch/venv/bin/python -I t3_reg_seeds.py OBJ MESH EP SLOT [N]
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

KIT = "/mnt/secondary/v2d/kit/v2d_submission_kit"
DS = os.path.expanduser("~/TestingGrounds/egony/video_to_data_challenge/track_3")
obj, mesh, ep, slot = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
N = int(sys.argv[5]) if len(sys.argv) > 5 else 10
sys.argv = [sys.argv[0], KIT, DS]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devscore as D  # noqa: E402
from v2dlb.mesh_budget import budget_mesh  # noqa: E402
A = D.MODS["AUC"]
v, f = budget_mesh(mesh, 4096, 4096)
g = D.ref_geometry(obj)
seeds = [1_000_003 * ep + 101 * slot] + [1_000_003 * e + 101 * s for e, s in zip(range(100, 100 + N), [0, 1] * N)]
R0 = None
out = []
for sd in seeds:
    _, R = A._register_object_body((v, f), (g["v"], g["f"]), np.zeros(3), sd)
    if R0 is None:
        R0 = R
    out.append(np.degrees(Rot.from_matrix(R @ R0.T).magnitude()))
out = np.array(out)
print(f"{obj} {os.path.basename(os.path.dirname(mesh))}/{os.path.basename(mesh)}: angle to the episode-seed choice over "
      f"{N} other seeds: {np.round(out[1:], 1).tolist()}  -> {np.mean(out[1:] < 15) * 100:.0f}% agree")
