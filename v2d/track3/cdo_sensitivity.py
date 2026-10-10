"""CD-O sensitivity: official Track 3 CD-O code (_object_chamfer_cm from metric_code/track_3/CD-O.py)
applied to controlled degradations of the public scans.  Reference = budget_mesh(public scan, 4096, 4096)
(INFERRED: host reference is a 4096-budget mesh, as the CD-O solution rows require exactly 4096 vertex and
4096 face rows per object).  Usage: python -I cdo_sensitivity.py KIT DATASET_TRACK3"""
import sys, importlib.util, tempfile, os, time
import numpy as np, trimesh
KIT, DS = sys.argv[1], sys.argv[2]
sys.path.insert(0, KIT)
from v2dlb.mesh_budget import budget_mesh
from v2dlb.vertices import mesh_path_for
spec = importlib.util.spec_from_file_location("cdo", f"{KIT}/metric_code/track_3/CD-O.py")
CDO = importlib.util.module_from_spec(spec); spec.loader.exec_module(CDO)
tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__)))

def budget(mesh):
    p = os.path.join(tmp, "m.ply"); mesh.export(p); return budget_mesh(p, 4096, 4096)

def cd(ref, cand, seed=1_000_003 * 1 + 101 * 0):
    return CDO._object_chamfer_cm(ref, cand, seed)

def variants(full, rng):
    v, f = np.asarray(full.vertices), np.asarray(full.faces)
    out = {}
    out["identical (same budget mesh)"] = None
    p = os.path.join(tmp, "s.ply"); full.export(p)
    out["re-decimated to 1024 faces"] = budget_mesh(p, 1024, 1024)
    out["re-decimated to 256 faces"] = budget_mesh(p, 256, 256)
    n = full.vertex_normals
    for s in (0.001, 0.002, 0.005):
        out[f"vertex noise sigma {s*1000:.0f}mm"] = budget(trimesh.Trimesh(v + rng.normal(size=v.shape) * s, f, process=False))
    for d in (0.002, 0.005):
        out[f"normal offset +{d*1000:.0f}mm (thicken)"] = budget(trimesh.Trimesh(v + n * d, f, process=False))
    ext = v.max(0) - v.min(0); ax = int(np.argmax(ext))
    for s in (1.05, 1.10, 1.20):
        S = np.ones(3); S[ax] = s
        out[f"anisotropic x{s} on longest axis"] = budget(trimesh.Trimesh(v * S, f, process=False))
    lo = v[:, 2].min(); keep = (v[f][:, :, 2].min(axis=1) > lo + 0.15 * (v[:, 2].max() - lo))
    out["drop lowest 15% (z) of surface"] = budget(trimesh.Trimesh(v, f[keep], process=False))
    side = v[f][:, :, 0].mean(axis=1) > np.percentile(v[:, 0], 70)
    out["drop one side (x>p70)"] = budget(trimesh.Trimesh(v, f[~side], process=False))
    out["convex hull"] = budget(full.convex_hull)
    out["oriented bounding box"] = budget(full.bounding_box_oriented.to_mesh() if hasattr(full.bounding_box_oriented, "to_mesh") else full.bounding_box_oriented)
    return out

rng = np.random.default_rng(0)
names = ["beige_cup", "blue_cup", "mini_dust_pan", "mini_sweeper", "plastic_dish_rack", "water_pitcher",
         "white_pot", "white_pot_lid", "wooden_piece_1", "wooden_piece_2"]
table = {}
for name in names:
    path = mesh_path_for(f"{DS}/public/mesh", name)
    full = trimesh.load(path, force="mesh", process=False)
    full = trimesh.Trimesh(full.vertices, full.faces, process=True)
    ref = budget_mesh(path, 4096, 4096)
    t0 = time.time()
    for label, cand in variants(full, rng).items():
        cand = ref if cand is None else cand
        # put the candidate in an arbitrary similarity frame (CD-O is Sim(3)-invariant by construction)
        R = trimesh.transformations.random_rotation_matrix(rng.random(3))[:3, :3]
        cv = (cand[0] * 1.7) @ R.T + rng.normal(size=3)
        table.setdefault(label, {})[name] = cd(ref, (cv, cand[1]))
    print(f"{name}: done in {time.time()-t0:.0f}s", flush=True)
print()
print(f"{'variant':38s} " + " ".join(f"{n[:10]:>10s}" for n in names) + "   mean")
for label, row in table.items():
    vals = [row[n] for n in names]
    print(f"{label:38s} " + " ".join(f"{x:10.3f}" for x in vals) + f" {np.mean(vals):7.3f}")
