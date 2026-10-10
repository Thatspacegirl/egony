"""Budget a mesh EXACTLY like the Track 3 packer, minus the padding, and write it as OBJ.

Run with the submission-kit venv (needs trimesh + fast_simplification):

    /mnt/secondary/v2d/venv-kit/bin/python mesh_prep.py IN_MESH OUT.obj [--kit DIR]

Why: tools/pack_submission.py budgets every submitted mesh to 4096 vertices / 4096 faces with
v2dlb.mesh_budget.budget_mesh (weld + fast_simplification + zero-area padding). If we simulate the
*un*-budgeted mesh, the scored mesh differs from the simulated one. So we run the same function
here, strip only the padding, and use the result both as the URDF visual/collision mesh and as the
file passed to pack_submission.py --meshes. Re-budgeting the written OBJ is then a no-op apart from
padding (checked below, the script fails otherwise).

Prints one JSON line: {"vertices": V, "faces": F, "input_vertices": ..., "input_faces": ..., ...}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_KIT = "/mnt/secondary/v2d/kit/v2d_submission_kit"
BUDGET = 4096


def strip_padding(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undo budget_mesh's padding: trailing copies of v[0] and trailing [0,0,0] faces."""
    real_f = f[~np.all(f == 0, axis=1)]
    n_v = len(v)
    while n_v > 1 and np.array_equal(v[n_v - 1], v[0]):
        n_v -= 1
    used_max = int(real_f.max()) if len(real_f) else 0
    if used_max >= n_v:
        raise RuntimeError("padding detection failed: faces reference stripped vertices")
    return v[:n_v], real_f


def write_obj(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    lines = ["# budgeted with v2dlb.mesh_budget (4096/4096), padding stripped; v + f only"]
    lines += [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in v]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh_in")
    ap.add_argument("obj_out")
    ap.add_argument("--kit", default=DEFAULT_KIT)
    args = ap.parse_args()
    sys.path.insert(0, args.kit)
    import trimesh
    from v2dlb.mesh_budget import budget_mesh  # the packer's own function

    src = trimesh.load(args.mesh_in, force="mesh", process=False)
    v, f = budget_mesh(args.mesh_in, BUDGET, BUDGET)
    v, f = strip_padding(np.asarray(v, np.float64), np.asarray(f, np.int64))
    out = Path(args.obj_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_obj(out, v, f)

    # Round-trip check through the packer path: budgeting our OBJ must not simplify again.
    v2, f2 = budget_mesh(str(out), BUDGET, BUDGET)
    v2, f2 = strip_padding(np.asarray(v2), np.asarray(f2))
    same_counts = (len(v2), len(f2)) == (len(v), len(f))
    tri1 = np.sort(np.round(v[f].reshape(-1, 9), 6), axis=0)
    tri2 = np.sort(np.round(v2[f2].reshape(-1, 9), 6), axis=0)
    same_geom = same_counts and np.allclose(tri1, tri2, atol=1e-6)
    if not same_geom:
        print(json.dumps({"error": "re-budget changed the mesh", "counts": [len(v), len(f), len(v2), len(f2)]}))
        return 2
    mesh = trimesh.Trimesh(v, f, process=False)
    print(
        json.dumps(
            {
                "input": str(Path(args.mesh_in).resolve()),
                "output": str(out.resolve()),
                "input_vertices": int(len(src.vertices)),
                "input_faces": int(len(src.faces)),
                "vertices": int(len(v)),
                "faces": int(len(f)),
                "watertight": bool(mesh.is_watertight),
                "extents": [float(x) for x in mesh.extents],
                "obj_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "repack_identical": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
