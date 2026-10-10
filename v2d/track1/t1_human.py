#!/usr/bin/env python3
"""Track 1 human route: GEM-X / SAM-3D-Body outputs (t1_gemx.py) -> kit episode fields (pose [T,136], scales [68],
shape [45]) in OUR camera frame, plus a CPU evaluator on the FORM-HOI val set (CD-H / ACC-H via the kit metric
code, score_t1.FastScorer).

Variants of the human trajectory (all keep SAM-3D-Body's per-frame native MHR articulation decoded with our K):
  s3      : per-frame SAM-3D-Body camera translation (k_cam_t)
  s3gx    : SAM-3D-Body body, pelvis moved along its viewing ray to GEM-X's Hips depth (GEM-X = temporally
            consistent depth; the 2D position of the pelvis is kept)
  s3gxd   : as s3gx but the full GEM-X Hips 3D position (not only depth)
Identity (scales, shape) is constant per episode: per-parameter median over the frames ('median').
MHR translation unit: model param = 10 * metres (verified: param 1.0 -> 10 cm), kit frame = MHR/100*diag(1,-1,-1).

Python: /mnt/secondary/v2d/scratch/venv/bin/python -I (CPU torch; MHR TorchScript).
  eval  --split val --variants s3,s3gx --smooth default,none [--episodes 0-12]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FLIP = np.array([1.0, -1.0, -1.0])
MHR_TS = Path(os.environ.get("V2D_MHR", "/mnt/secondary/v2d/scratch/track1/mhr_assets/mhr_model.pt"))
HUMAN_ROOT = Path("/mnt/secondary/v2d/t1/human")
PELVIS = 1          # MHR skeleton joint 1 (rest y = 92.4 cm) ; joint 0 = model origin


_MHR = {}


def mhr_model(device="cpu"):
    import torch
    if device not in _MHR:
        _MHR[device] = torch.jit.load(str(MHR_TS), map_location=device).eval()
    return _MHR[device]


def mhr_forward(params, shape, device="cpu", batch=128, verts=True):
    """params [T,204] (pose 136 + scales 68), shape [45] or [T,45] -> kit-frame metres: V [T,18439,3], J [T,127,3]."""
    import torch
    m = mhr_model(device)
    P = torch.as_tensor(np.asarray(params, np.float32))
    S = np.asarray(shape, np.float32)
    S = torch.as_tensor(np.broadcast_to(S, (len(P), 45)).copy())
    Vs, Js = [], []
    with torch.no_grad():
        for s in range(0, len(P), batch):
            p, sh = P[s:s + batch].to(device), S[s:s + batch].to(device)
            v, sk = m(sh, p, torch.zeros(len(p), 72, device=device))
            if verts:
                Vs.append(v.float().cpu().numpy() / 100.0 * FLIP)
            Js.append(sk[..., :3].float().cpu().numpy() / 100.0 * FLIP)
    J = np.concatenate(Js)
    return (np.concatenate(Vs) if verts else None), J


def find_outputs(split, ep):
    d = HUMAN_ROOT / split / f"episode_{ep:06d}"
    s3 = list((d / "gemx").glob("*/sam3db_mhr.npz"))
    gx = list((d / "gemx").glob("*/gemx_dump.npz"))
    return d, (s3[0] if s3 else None), (gx[0] if gx else None)


def build_episode(s3_path, gx_path, variant="s3gx", identity="median"):
    """Return (episode dict with pose/scales/shape, info). Object fields are NOT set here."""
    s3 = dict(np.load(s3_path))
    mp = s3["k_mhr_model_params"].astype(np.float64)
    T = len(mp)
    if identity == "median":
        scales = np.median(mp[:, 136:204], 0)
        shape = np.median(s3["k_shape"].astype(np.float64), 0)
    else:
        raise ValueError(identity)
    pose = mp[:, :136].copy()
    assert np.abs(pose[:, :3]).max() < 1e-6, "SAM-3D-Body root translation should be 0"
    # pelvis of the per-frame identity vs the episode identity (both without translation)
    own = np.concatenate([mp[:, :136], mp[:, 136:204]], 1)
    _, J_own = mhr_forward(own, s3["k_shape"], verts=False)
    info = {"T": T, "variant": variant}
    if variant.endswith("r"):
        # root orientation from GEM-X (temporally consistent), mapped to the MHR root by one constant offset
        from scipy.spatial.transform import Rotation as Rot
        g = dict(np.load(gx_path))
        R_gx = Rot.from_rotvec(g["global_orient"].astype(np.float64)).as_matrix()
        R_s3 = np.diag(FLIP) @ Rot.from_euler("xyz", pose[:, 3:6]).as_matrix()
        C = np.einsum("tji,tjk->tik", R_gx, R_s3)
        Cbar = Rot.from_matrix(C).mean().as_matrix()
        resid = Rot.from_matrix(np.einsum("ij,tjk->tik", Cbar.T, C)).magnitude()
        info["gx_rot_offset_resid_deg_med"] = float(np.degrees(np.median(resid)))
        R_new = np.diag(FLIP) @ (R_gx @ Cbar)
        pose[:, 3:6] = Rot.from_matrix(R_new).as_euler("xyz")
        variant = variant[:-1]
    med = np.concatenate([pose, np.broadcast_to(scales, (T, 68))], 1)
    _, J_med = mhr_forward(med, shape, verts=False)
    cam_t = s3["k_cam_t"].astype(np.float64)
    p_s3 = J_own[:, PELVIS] + cam_t                               # SAM-3D-Body pelvis, camera frame (m)
    if variant == "s3":
        p = p_s3
    else:
        g = dict(np.load(gx_path))
        names = [str(x) for x in g["joint_names"]]
        hips = g["joints_incam"][:, names.index("Hips")].astype(np.float64)
        assert len(hips) == T, (len(hips), T)
        if variant == "s3gx":
            lam = hips[:, 2] / np.maximum(p_s3[:, 2], 1e-3)
            p = p_s3 * lam[:, None]
            info["depth_ratio_gx_over_s3_median"] = float(np.median(lam))
        elif variant == "s3gxd":
            p = hips
        else:
            raise ValueError(variant)
    trans_m = p - J_med[:, PELVIS]                                   # kit-frame translation of the model origin
    pose[:, :3] = 10.0 * trans_m * FLIP
    info["pelvis_depth_median_m"] = float(np.median(p[:, 2]))
    return {"pose": pose, "scales": scales, "shape": shape}, info


def dummy_object(T):
    return {"object_rotation": np.broadcast_to(np.eye(3), (T, 3, 3)).copy(), "object_translation": np.zeros((T, 3)),
            "object_scale": np.array(1.0)}


def parse_eps(s, all_eps):
    if not s:
        return sorted(all_eps)
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [e for e in out if e in all_eps]


def cmd_eval(a):
    import score_t1 as S
    from smoothing import SmoothCfg, smooth_episode
    fs = S.FastScorer()
    eps = parse_eps(a.episodes, fs.index)
    cfgs = {}
    for name in a.smooth.split(","):
        if name == "none":
            cfgs[name] = None
        else:
            p = HERE / f"smooth_{name}.json"
            cfgs[name] = SmoothCfg(**json.load(open(p)))
    rows = []
    for ep in eps:
        d, s3p, gxp = find_outputs("val", ep)
        if s3p is None:
            print(f"ep {ep}: no outputs yet"); continue
        mesh_vf = fs.ref(ep)[1]
        f0 = fs.index[ep]["frames"][0]
        for var in a.variants.split(","):
            t0 = time.time()
            hum, info = build_episode(s3p, gxp, var)
            T_gt = len(fs.ref(ep)[0]["pose"])
            T = len(hum["pose"])
            if T < T_gt:
                print(f"ep {ep}: pred T {T} < GT T {T_gt}"); continue
            epd = {**hum, **dummy_object(T)}
            for cname, cfg in cfgs.items():
                e2 = smooth_episode(epd, f0, cfg) if cfg is not None else epd
                m = fs.score_episode(ep, e2, mesh_vf, pen=False)
                r = {"ep": ep, "variant": var, "smooth": cname, "cd_h": m["cd_h_cm"], "acc_h": m["acc_h_cm"], **info}
                rows.append(r)
                print(f"ep {ep:2d} {var:6s} {cname:10s} CD-H {m['cd_h_cm']:7.3f} ACC-H {m['acc_h_cm']:7.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    if len(df):
        print(df.groupby(["variant", "smooth"])[["cd_h", "acc_h"]].mean().round(4).to_string())
    if a.json:
        df.to_json(a.json, orient="records", indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("eval")
    e.add_argument("--variants", default="s3,s3gx")
    e.add_argument("--smooth", default="none,default")
    e.add_argument("--episodes", default="")
    e.add_argument("--json")
    a = ap.parse_args()
    {"eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
