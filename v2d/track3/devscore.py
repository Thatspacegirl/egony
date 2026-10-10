"""Local Track 3 dev scorer: runs the OFFICIAL Kaggle scorer code (metric_code/track_3/*.py,
imported unmodified) on public-split episodes, with a host-like reference geometry built from the
public scans.  INFERRED host details: anchor = scan origin (= surface centroid, verified ~0 for all
public scans), points = v2dlb.vertices.sample_vertices (500 verts, seed 0), cup axis = scan +z
(order 360), lid half-turn axis = scan +x (thin axis, order 2), reference mesh = budget_mesh(scan,4096,4096).

Usage: python -I devscore.py KIT DATASET_TRACK3 [episodes...]
"""
from __future__ import annotations
import sys, json, glob, importlib.util, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as Rot

KIT, DS = sys.argv[1], sys.argv[2]
sys.path.insert(0, KIT)
from v2dlb.mesh_budget import budget_mesh
from v2dlb.vertices import sample_vertices, mesh_path_for
from v2dlb.track3_pose import _encode_archive

METRICS = {"AUC": "add_auc", "SP-SR": "spider_sr", "MP-SR": "maniptrans_sr", "RPE": "rpe_cm", "MPPE": "mppe_cm"}
def load_metric(name):
    spec = importlib.util.spec_from_file_location(f"t3_{name}", f"{KIT}/metric_code/track_3/{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
MODS = {k: load_metric(k) for k in METRICS}
CDO = load_metric("CD-O")

# memoize the (deterministic) mesh registration across the 5 metric modules and across scenarios
import hashlib
_REG_CACHE = {}
def _cached(orig):
    def f(candidate, reference, anchor, seed):
        key = (hashlib.sha1(candidate[0].tobytes() + reference[0].tobytes()).hexdigest(), int(seed))
        if key not in _REG_CACHE:
            _REG_CACHE[key] = orig(candidate, reference, anchor, seed)
        return _REG_CACHE[key]
    return f
for _m in MODS.values():
    _m._register_object_body = _cached(_m._register_object_body)

AXIAL = {"beige_cup", "blue_cup"}; HALF = {"white_pot_lid"}
_geo_cache = {}
def ref_geometry(name):
    if name not in _geo_cache:
        path = mesh_path_for(f"{DS}/public/mesh", name)
        v, f = budget_mesh(path, 4096, 4096)
        pts = sample_vertices(path)
        axis = np.array([0., 0., 1.]) if name in AXIAL else (np.array([1., 0., 0.]) if name in HALF else np.zeros(3))
        order = 360 if name in AXIAL else (2 if name in HALF else 1)
        _geo_cache[name] = dict(v=v, f=f, anchor=np.zeros(3), points=pts, axis=axis, order=order)
    return _geo_cache[name]

def load_episode(ep):
    t = pq.read_table(f"{DS}/public/data/chunk-000/episode_{ep:06d}.parquet").to_pandas()
    objs = t["observation.objects"]
    names = [o["name"] for o in objs.iloc[0]]
    P = np.stack([np.stack([np.asarray(o[i]["pose"], np.float64) for i in range(len(names))]) for o in objs])  # T,B,7 wxyz
    vis = np.stack([[o[i]["visible"] for i in range(len(names))] for o in objs])
    assert vis.all(), "public episodes are fully visible"
    pose = np.concatenate([P[..., :3], P[..., 4:7], P[..., 3:4]], axis=-1)  # -> xyzw
    pose[..., 3:] /= np.linalg.norm(pose[..., 3:], axis=-1, keepdims=True)
    return names, pose

PROV = {c: "0" * 64 for c in ["eval_script_sha256", "eval_code_sha256", "checkpoint_sha256",
                               "packer_script_sha256", "uploader_script_sha256"]}
PROV["code_commit_url"] = "https://github.com/x/y/commit/0000000"

def build_frames(episodes):
    """episodes: list of (ep, names, ref_pose[T,B,7] xyzw, sub_pose[T,B,7] xyzw, cand_meshes{slot:(v,f)})"""
    names_all = sorted({n for e in episodes for n in e[1]})
    geo = {}
    for i, n in enumerate(names_all):
        g = ref_geometry(n)
        geo.update({f"v{i}": g["v"], f"f{i}": g["f"], f"anchor{i}": g["anchor"], f"points{i}": g["points"],
                    f"axis{i}": g["axis"], f"order{i}": np.int64(g["order"])})
    geo["names"] = np.array(names_all)
    sol, sub = [], []
    for ep, names, ref, pred, meshes in episodes:
        T, B, _ = ref.shape
        for t in range(T):
            for b in range(B):
                rid = f"track_3/e{ep:06d}/w000/f{t:06d}/o{b}"
                sol.append([rid, ep, 0, t, b, names[b], *ref[t, b]])
                mesh = _encode_archive(vertices=meshes[b][0], faces=meshes[b][1]) if t == 0 else ""
                sub.append([rid, *pred[t, b], mesh])
    sol = pd.DataFrame(sol, columns=["row_id", "episode_index", "world_index", "frame_index", "object_slot",
                                     "object_name"] + ["ref_" + c for c in MODS["AUC"]._POSE_COLUMNS])
    sol["reference_geometry_b64"] = ""
    sol.loc[0, "reference_geometry_b64"] = _encode_archive(**geo)
    sub = pd.DataFrame(sub, columns=["row_id"] + MODS["AUC"]._POSE_COLUMNS + ["object_mesh_b64"])
    for k, v in PROV.items(): sub[k] = v
    return sol, sub

def score_all(episodes):
    sol, sub = build_frames(episodes)
    out = {}
    for k, mod in MODS.items():
        out[k] = mod.score(sol.copy(), sub.copy(), "row_id")
    return out

# ---------------- perturbations ----------------
def compose(T_a, pose):
    """apply rigid world transform (R,t) to xyzw poses"""
    R, t = T_a
    out = pose.copy()
    out[..., :3] = pose[..., :3] @ R.T + t
    out[..., 3:] = (Rot.from_matrix(R) * Rot.from_quat(pose[..., 3:].reshape(-1, 4))).as_quat().reshape(pose[..., 3:].shape)
    return out

def to_candidate_body(pose, Rc, tc):
    """mesh vertices expressed in candidate frame v_c = Rc v + tc  =>  world_T_cand = world_T_obj * inv(Tc)"""
    R = Rot.from_quat(pose[..., 3:].reshape(-1, 4))
    Rc_inv = Rot.from_matrix(Rc).inv()
    newR = R * Rc_inv
    p = pose[..., :3].reshape(-1, 3) - newR.apply(tc)
    out = pose.copy(); out[..., :3] = p.reshape(pose[..., :3].shape)
    out[..., 3:] = newR.as_quat().reshape(pose[..., 3:].shape)
    return out

rng_global = np.random.default_rng(123)
def random_rigid(rng, ang=None, trans=1.0):
    R = Rot.random(random_state=int(rng.integers(1 << 31))).as_matrix() if ang is None else \
        Rot.from_rotvec(ang * (lambda a: a / np.linalg.norm(a))(rng.normal(size=3))).as_matrix()
    return R, rng.normal(size=3) * trans

def candidate_meshes(names, rng, mesh_scale=1.0, frame_change=True):
    meshes, frames = {}, {}
    for b, n in enumerate(names):
        g = ref_geometry(n)
        Rc, tc = random_rigid(rng, trans=0.1) if frame_change else (np.eye(3), np.zeros(3))
        v = (g["v"] * mesh_scale) @ Rc.T + tc
        meshes[b] = (v, g["f"]); frames[b] = (Rc, tc, mesh_scale)
    return meshes, frames

def make_pred(ref, names, frames, perturb, rng):
    pred = perturb(ref.copy(), rng)
    out = pred.copy()
    for b in range(len(names)):
        Rc, tc, s = frames[b]
        out[:, b] = to_candidate_body(pred[:, b], Rc, tc * 1.0 / 1.0 if s == 1 else tc)  # tc already in scaled mesh space
        if s != 1.0:  # mesh scaled about origin: anchor (origin) maps to tc; pose positions stay metric
            pass
    return compose(random_rigid(rng), out)  # arbitrary sim/world frame

# perturbation library (operate on xyzw world_T_obj, T,B,7)
def p_identity(P, rng): return P
def p_static(P, rng): return np.repeat(P[:1], len(P), axis=0)
def p_jitter(sig_m, sig_deg):
    def f(P, rng):
        Q = P.copy(); Q[..., :3] += rng.normal(size=Q[..., :3].shape) * sig_m
        r = Rot.from_rotvec(rng.normal(size=Q[..., :3].shape).reshape(-1, 3) * np.deg2rad(sig_deg))
        Q[..., 3:] = (Rot.from_quat(Q[..., 3:].reshape(-1, 4)) * r).as_quat().reshape(Q[..., 3:].shape)
        return Q
    return f
def p_drift(end_m):
    def f(P, rng):
        Q = P.copy(); T = len(P)
        d = rng.normal(size=(P.shape[1], 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        Q[..., :3] += (np.linspace(0, 1, T)[:, None, None] * d[None] * end_m)
        return Q
    return f
def p_scale(s):
    def f(P, rng):
        Q = P.copy(); c = P[0, :, :3].mean(0)
        Q[..., :3] = (P[..., :3] - c) * s + c; return Q
    return f
def p_lag(k):
    def f(P, rng):
        return np.concatenate([np.repeat(P[:1], k, 0), P[:-k]], 0) if k > 0 else P
    return f
def _unit(a): return a / np.linalg.norm(a)
def p_anchor_bias(deg):
    """constant body-frame orientation bias on every object (e.g. biased mesh registration / FP tracking)"""
    def f(P, rng):
        Q = P.copy()
        for b in range(P.shape[1]):
            e = Rot.from_rotvec(_unit(rng.normal(size=3)) * np.deg2rad(deg))
            Q[:, b, 3:] = (Rot.from_quat(P[:, b, 3:]) * e).as_quat()
        return Q
    return f
def p_combo(sig_m, sig_deg, drift_m, lag, bias_deg):
    def f(P, rng):
        return p_jitter(sig_m, sig_deg)(p_anchor_bias(bias_deg)(p_drift(drift_m)(p_lag(lag)(P, rng), rng), rng), rng)
    return f
def p_init_offset(m):
    """object 1..B-1 initial placement error: constant world offset of m metres (relative pose error)"""
    def f(P, rng):
        Q = P.copy()
        for b in range(1, P.shape[1]):
            d = rng.normal(size=3); d[2] = 0; d /= np.linalg.norm(d); Q[:, b, :3] += d * m
        return Q
    return f
def p_world_tilt(deg):
    """our world frame is tilted (gravity error) - pure global rotation, should be absorbed"""
    def f(P, rng):
        R = Rot.from_rotvec([np.deg2rad(deg), 0, 0]).as_matrix(); return compose((R, np.zeros(3)), P)
    return f

def run(eps, label, perturb, mesh_scale=1.0, seed=0, frame_change=True):
    rng = np.random.default_rng(seed)
    items = []
    for ep in eps:
        names, ref = load_episode(ep)
        meshes, frames = candidate_meshes(names, np.random.default_rng(1000 + ep), mesh_scale, frame_change)
        pred = make_pred(ref, names, frames, perturb, rng)
        items.append((ep, names, ref, pred, meshes))
    t0 = time.time(); s = score_all(items)
    print(f"{label:38s} " + "  ".join(f"{k}={v:.4f}" for k, v in s.items()) + f"   ({time.time()-t0:.0f}s)", flush=True)
    return s

if __name__ == "__main__":
    eps = [int(e) for e in sys.argv[3:]] or [0, 2, 11, 12, 13, 18, 21, 22, 23, 24, 26, 27, 31, 39, 40, 41, 42, 43, 44, 45]
    print("episodes", eps)
    run(eps, "identity (no frame change)", p_identity, frame_change=False)
    run(eps, "identity + random body/world frames", p_identity)
    run(eps, "identity + mesh scaled x1.2", p_identity, mesh_scale=1.2)
    run(eps, "world tilt 10deg (global)", p_world_tilt(10))
    run(eps, "STATIC (objects never move)", p_static)
    for s_, d_ in [(0.005, 2), (0.01, 5), (0.02, 10)]:
        run(eps, f"jitter {s_*100:.1f}cm/{d_}deg iid", p_jitter(s_, d_))
    for d in [0.02, 0.05, 0.10]:
        run(eps, f"linear drift to {d*100:.0f}cm at end", p_drift(d))
    for s_ in [0.9, 1.1, 1.25]:
        run(eps, f"trajectory scale x{s_}", p_scale(s_))
    for k in [2, 5, 10, 20]:
        run(eps, f"time lag {k} frames", p_lag(k))
    for d in [5, 10, 20]:
        run(eps, f"const body-frame rot bias {d}deg", p_anchor_bias(d))
    for m in [0.01, 0.03, 0.05]:
        run(eps, f"obj>=1 init offset {m*100:.0f}cm", p_init_offset(m))
    run(eps, "combo A: 0.5cm/2deg jit,2cm drift,lag2,bias3", p_combo(0.005, 2, 0.02, 2, 3))
    run(eps, "combo B: 1cm/5deg jit,5cm drift,lag5,bias5", p_combo(0.01, 5, 0.05, 5, 5))
