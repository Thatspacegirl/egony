"""Write a GT-as-prediction PRED_DIR (random mesh body frame + random world frame) to validate t3_devscore.py."""
import sys, os, numpy as np, pandas as pd, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devscore as D
IDENTITY = "--identity" in sys.argv  # plain GT-vs-GT: scan mesh as-is, no body/world frame change
argv = [x for x in sys.argv if x != "--identity"]
out = argv[3]; eps = [int(e) for e in argv[4:]] or [0, 2, 11, 12, 13, 18, 21, 22, 23, 24, 26, 27, 31, 39, 40, 41, 42, 43, 44, 45]
rng = np.random.default_rng(7)
for ep in eps:
    names, ref = D.load_episode(ep)
    G = (np.eye(3), np.zeros(3)) if IDENTITY else D.random_rigid(rng)
    os.makedirs(f"{out}/episode_{ep:06d}", exist_ok=True)
    pred = ref.copy()
    for b, n in enumerate(names):
        g = D.ref_geometry(n)
        Rc, tc = (np.eye(3), np.zeros(3)) if IDENTITY else D.random_rigid(rng, trans=0.1)
        sc = 1.0 if IDENTITY else 1.15
        trimesh.Trimesh((g["v"] * sc) @ Rc.T + tc, g["f"][np.linalg.norm(np.cross(g["v"][g["f"][:,1]]-g["v"][g["f"][:,0]], g["v"][g["f"][:,2]]-g["v"][g["f"][:,0]]),axis=1)>0], process=False).export(f"{out}/episode_{ep:06d}/{n}.ply")
        pred[:, b] = D.to_candidate_body(ref[:, b], Rc, tc)
    pred = D.compose(G, pred)
    T, B, _ = pred.shape
    rows = [dict(frame_index=t, object_slot=b, **dict(zip(D.MODS["AUC"]._POSE_COLUMNS, pred[t, b]))) for t in range(T) for b in range(B)]
    pd.DataFrame(rows).to_parquet(f"{out}/episode_{ep:06d}.parquet", index=False)
print("wrote", out)
