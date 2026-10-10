"""Score YOUR perception/rollout outputs on the PUBLIC Track 3 split with the official Kaggle scorer code.

  python -I t3_devscore.py KIT DATASET_TRACK3 --pred PRED_DIR [--episodes 0 2 11 ...] [--cdo]

PRED_DIR layout (any rigid world frame, metres; poses must place YOUR mesh):
  PRED_DIR/episode_000011.parquet   columns: frame_index, object_slot, pos_x,pos_y,pos_z, quat_x,quat_y,quat_z,quat_w
                                     (object_slot follows the public parquet's per-frame object order)
  PRED_DIR/episode_000011/<object_name>.{glb,obj,ply,stl,off}
Missing frames are an error (the host requires every frame 0..N-1).
Reference geometry is built from the public scans exactly as devscore.py documents (anchor/points/axes INFERRED).
"""
import sys, argparse, glob, os
import numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("kit"); ap.add_argument("ds"); ap.add_argument("--pred", required=True)
ap.add_argument("--episodes", type=int, nargs="*"); ap.add_argument("--cdo", action="store_true")
ap.add_argument("--per_episode", action="store_true", help="also print each episode's own scores")
a = ap.parse_args()
sys.argv = [sys.argv[0], a.kit, a.ds]
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import devscore as D
from v2dlb.mesh_budget import budget_mesh

def find_mesh(d, name):
    for ext in ("glb", "obj", "ply", "stl", "off"):
        p = os.path.join(d, f"{name}.{ext}")
        if os.path.exists(p): return p
    raise FileNotFoundError(f"no mesh for {name} in {d}")

eps = a.episodes or sorted(int(os.path.basename(p)[8:14]) for p in glob.glob(f"{a.pred}/episode_*.parquet"))
items, cdo = [], []
for ep in eps:
    names, ref = D.load_episode(ep)
    T, B, _ = ref.shape
    df = pd.read_parquet(f"{a.pred}/episode_{ep:06d}.parquet").sort_values(["frame_index", "object_slot"])
    if len(df) != T * B or not (df.frame_index.to_numpy().reshape(T, B) == np.arange(T)[:, None]).all():
        raise SystemExit(f"episode {ep}: need {T} frames x {B} objects, got {len(df)} rows")
    pred = df[D.MODS["AUC"]._POSE_COLUMNS].to_numpy(np.float64).reshape(T, B, 7)
    meshes = {b: budget_mesh(find_mesh(f"{a.pred}/episode_{ep:06d}", n), 4096, 4096) for b, n in enumerate(names)}
    items.append((ep, names, ref, pred, meshes))
    if a.cdo:
        for b, n in enumerate(names):
            g = D.ref_geometry(n)
            cdo.append(D.CDO._object_chamfer_cm((g["v"], g["f"]), meshes[b], D.CDO._object_seed(ep, b)))
if a.per_episode:
    for it in items:
        sc = D.score_all([it])
        print(f"ep {it[0]:3d} " + "  ".join(f"{k}={v:.4f}" for k, v in sc.items()) + "  " + ",".join(it[1]))
scores = D.score_all(items)
if a.cdo: scores["CD-O"] = float(np.mean(cdo))
print("  ".join(f"{k}={v:.4f}" for k, v in scores.items()))
