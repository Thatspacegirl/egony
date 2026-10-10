"""How much body-frame error does the host registration (_register_object_body from the official scorer)
introduce when the submitted mesh is imperfect?  Candidate meshes are degraded versions of the scan,
in the SAME body frame, so the ideal result is rotation=I, candidate_anchor=0.  Reports rotation error (deg)
and anchor error (mm).  A body-rotation error on the rotation-anchor object becomes a global scene rotation."""
import sys, os, importlib.util, tempfile, numpy as np, trimesh
KIT, DS = sys.argv[1], sys.argv[2]
sys.path.insert(0, KIT)
from v2dlb.mesh_budget import budget_mesh
from v2dlb.vertices import mesh_path_for
spec = importlib.util.spec_from_file_location("auc", f"{KIT}/metric_code/track_3/AUC.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__)))
def budget(m):
    p = os.path.join(tmp, "m.ply"); m.export(p); return budget_mesh(p, 4096, 4096)
AXIAL = {"beige_cup", "blue_cup"}
rng = np.random.default_rng(0)
print(f"{'object':18s} " + " ".join(f"{k:>22s}" for k in ["noise2mm", "thicken5mm", "drop_bottom15%", "drop_side", "aniso1.2", "convex_hull", "OBB"]))
for name in ["beige_cup", "blue_cup", "mini_dust_pan", "mini_sweeper", "plastic_dish_rack", "water_pitcher",
             "white_pot", "white_pot_lid", "wooden_piece_1", "wooden_piece_2"]:
    path = mesh_path_for(f"{DS}/public/mesh", name)
    full = trimesh.load(path, force="mesh", process=False); full = trimesh.Trimesh(full.vertices, full.faces, process=True)
    v, f = np.asarray(full.vertices), np.asarray(full.faces); n = full.vertex_normals
    ref = budget_mesh(path, 4096, 4096)
    lo = v[:, 2].min(); keep = v[f][:, :, 2].min(axis=1) > lo + 0.15 * (v[:, 2].max() - lo)
    side = v[f][:, :, 0].mean(axis=1) > np.percentile(v[:, 0], 70)
    ext = v.max(0) - v.min(0); S = np.ones(3); S[int(np.argmax(ext))] = 1.2
    obb = full.bounding_box_oriented
    cands = [trimesh.Trimesh(v + rng.normal(size=v.shape) * 0.002, f, process=False), trimesh.Trimesh(v + n * 0.005, f, process=False),
             trimesh.Trimesh(v, f[keep], process=False), trimesh.Trimesh(v, f[~side], process=False),
             trimesh.Trimesh(v * S, f, process=False), full.convex_hull, obb.to_mesh() if hasattr(obb, "to_mesh") else obb]
    out = []
    for c in cands:
        cand = budget(c)
        anchor_c, R = M._register_object_body(cand, ref, np.zeros(3), 1_000_003 * 3 + 101 * 0)
        if name in AXIAL:  # only the axis direction matters
            ang = np.degrees(np.arccos(np.clip(abs((R @ [0, 0, 1])[2]), -1, 1)))
        else:
            ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        out.append(f"{ang:7.1f}deg {np.linalg.norm(anchor_c)*1000:6.1f}mm")
    print(f"{name:18s} " + " ".join(f"{o:>22s}" for o in out), flush=True)
