"""DEV ONLY (public split): is the official scorer's mesh registration of OUR mesh onto the scan correct?

The Track 3 pose metrics first register each submitted mesh onto the hidden scan (AUC.py _register_object_body:
RMS-radius scale + ICP from 25 PCA starts, best symmetric mean distance), convert our poses into the scan body frame
with that rotation, then align the whole episode with ONE rigid transform from frame 0 of the first non-symmetric
object.  A wrong (e.g. 180-degree flipped) registration of that anchor object rotates the whole scene.

Here the TRUE body rotation is recovered independently of the registration:
  1. world alignment W (rotation + translation, no scale) = Umeyama on the surface-centroid trajectories of all objects
     (our mesh centroid under our poses vs scan centroid under GT poses, every 2nd frame);
  2. per object Q_true = mean_t R_o(t)^T R_W^T R_g(t)   (our body -> scan body; spread reported);
  3. registration error = angle(R_reg Q_true) with the scorer's R_reg; cups: angle between symmetry axes; lid: min over
     the half-turn about its axis.
It also reports the symmetric mean distance (cm, scorer CD-O units, after RMS-scale normalisation) of our mesh vs the
scan at the scorer's chosen alignment and at the TRUE alignment (refined by the scorer's own trimmed ICP from there):
true < chosen means the 25 starts missed the right basin; true > chosen means our SHAPE prefers the wrong orientation.

  /mnt/secondary/v2d/scratch/venv/bin/python -I t3_reg_check.py --pred DEVSCORE_PRED_DIR [--episodes ...] [--json OUT]
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

KIT = "/mnt/secondary/v2d/kit/v2d_submission_kit"
DS = os.path.expanduser("~/TestingGrounds/egony/video_to_data_challenge/track_3")


def umeyama(X, Y):
    """Y ~= R X + t (no scale)."""
    mx, my = X.mean(0), Y.mean(0)
    C = (Y - my).T @ (X - mx)
    U, S, Vt = np.linalg.svd(C)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, my - R @ mx, S


def mean_rotation(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    spread = np.degrees([Rot.from_matrix(R.T @ r).magnitude() for r in Rs])
    return R, float(np.median(spread)), float(np.percentile(spread, 90))


def ang(R):
    return float(np.degrees(Rot.from_matrix(R).magnitude()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    sys.argv = [sys.argv[0], KIT, DS]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import devscore as D
    from v2dlb.mesh_budget import budget_mesh
    A = D.MODS["AUC"]

    def find_mesh(d, name):
        for ext in ("glb", "obj", "ply", "stl", "off"):
            if os.path.exists(f"{d}/{name}.{ext}"):
                return f"{d}/{name}.{ext}"
        raise FileNotFoundError(name)

    eps = a.episodes or sorted(int(os.path.basename(p)[8:14]) for p in glob.glob(f"{a.pred}/episode_*.parquet"))
    rows = []
    for ep in eps:
        names, ref = D.load_episode(ep)
        T, B, _ = ref.shape
        df = pd.read_parquet(f"{a.pred}/episode_{ep:06d}.parquet").sort_values(["frame_index", "object_slot"])
        pred = df[A._POSE_COLUMNS].to_numpy(np.float64).reshape(T, B, 7)
        Rg = Rot.from_quat(ref[..., 3:].reshape(-1, 4)).as_matrix().reshape(T, B, 3, 3)
        Ro = Rot.from_quat(pred[..., 3:].reshape(-1, 4)).as_matrix().reshape(T, B, 3, 3)
        info = []
        for b, n in enumerate(names):
            v, f = budget_mesh(find_mesh(f"{a.pred}/episode_{ep:06d}", n), 4096, 4096)
            g = D.ref_geometry(n)
            seed = 1_000_003 * ep + 101 * b
            _, Rreg = A._register_object_body((v, f), (g["v"], g["f"]), np.zeros(3), seed)
            co, ro = A._surface_moments(v, f)
            cs, rs = A._surface_moments(g["v"], g["f"])
            info.append(dict(v=v, f=f, g=g, Rreg=Rreg, co=co, cs=cs, ro=ro, rs=rs, seed=seed))
        ts = np.arange(0, T, 2)
        X = np.concatenate([pred[ts, b, :3] + Ro[ts, b] @ info[b]["co"] for b in range(B)])
        Y = np.concatenate([ref[ts, b, :3] + Rg[ts, b] @ info[b]["cs"] for b in range(B)])
        RW, tW, S = umeyama(X, Y)
        resid = np.linalg.norm(X @ RW.T + tW - Y, axis=1)
        print(f"== episode {ep}: world fit resid median {np.median(resid) * 100:.2f} cm p90 "
              f"{np.percentile(resid, 90) * 100:.2f} cm, singular values {np.round(S / len(X), 5)}")
        for b, n in enumerate(names):
            I = info[b]
            Qs = np.einsum("tji,jk,tkl->til", Ro[:, b], RW.T, Rg[:, b])  # R_o^T R_W^T R_g
            Q, sp50, sp90 = mean_rotation(Qs)
            Rreg = I["Rreg"]
            if n in D.AXIAL:
                z = np.array([0, 0, 1.0])
                a_reg, a_true = Rreg.T @ z, Q @ z
                err = float(np.degrees(np.arccos(np.clip(a_reg @ a_true, -1, 1))))
            elif n in D.HALF:
                Sx = Rot.from_rotvec([math.pi, 0, 0]).as_matrix()
                err = min(ang(Rreg @ Q), ang(Rreg @ Q @ Sx.T), ang(Sx @ Rreg @ Q))
            else:
                err = ang(Rreg @ Q)
            # chamfer at the scorer's chosen alignment vs at the true one (scorer units: cm after RMS scale)
            cv = (I["v"] - I["co"]) * (I["rs"] / I["ro"])
            gs = A._sample_mesh_surface(I["g"]["v"], I["g"]["f"], 10000, I["seed"]) - I["cs"]
            cs_ = A._sample_mesh_surface(cv, I["f"], 10000, I["seed"])
            ch = []
            for R0 in (Rreg, Q.T):
                Rr, tr = R0.copy(), np.zeros(3)
                tree = cKDTree(gs)
                for _ in range(40):
                    mv = cs_ @ Rr.T + tr
                    d, idx = tree.query(mv)
                    keep = np.argsort(d)[: int(0.9 * len(d))]
                    dR, dt = A._kabsch(mv[keep], gs[idx[keep]])
                    Rr, tr = dR @ Rr, dR @ tr + dt
                ch.append(100 * A._symmetric_mean_distance(gs, cs_ @ Rr.T + tr))
            row = dict(episode=ep, slot=b, obj=n, reg_err_deg=round(err, 1), q_spread_deg=round(sp50, 1),
                       cd_chosen=round(ch[0], 3), cd_true=round(ch[1], 3),
                       anchor_kind="axial" if n in D.AXIAL else ("half" if n in D.HALF else "fixed"))
            rows.append(row)
            flag = "FLIP" if err > 30 else ""
            print(f"  {n:18s} reg_err {err:6.1f} deg (Q spread med {sp50:4.1f} p90 {sp90:4.1f}) | CD chosen "
                  f"{ch[0]:.3f} true {ch[1]:.3f} cm  {flag}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
