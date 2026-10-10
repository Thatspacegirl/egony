"""DEV-ONLY diagnostics of a t3_devscore.py prediction dir on PUBLIC episodes (replicates the official registration +
frame-0 alignment of metric_code/track_3/AUC.py, then reports where the error comes from).

Per episode and slot: position error (cm) after the scorer's own alignment (mean / median / p90 / frame 0 / last),
orientation error (deg; cups: axis tilt only, lid: half-turn min), and the same after an ORACLE rigid alignment over
all frames (removes the frame-0 alignment error, leaves tracking error).  Also the registration rotation angle of our
mesh onto the scan (large values near 180 deg indicate flips when the mesh frame is near-symmetric).

  /mnt/secondary/v2d/scratch/venv/bin/python -I t3_dev_diag.py --pred DIR --episodes 21 12 ...
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

KIT = "/mnt/secondary/v2d/kit/v2d_submission_kit"
DS = os.path.expanduser("~/TestingGrounds/egony/video_to_data_challenge/track_3")


def rot_err_deg(qa, qr, axis, order):
    Ra, Rr = Rot.from_quat(qa).as_matrix(), Rot.from_quat(qr).as_matrix()
    if np.any(axis) and order >= 360:  # axial: tilt of the axis
        a, r = Ra @ axis, Rr @ axis
        return np.degrees(np.arccos(np.clip((a * r).sum(-1), -1, 1)))
    d = np.degrees((Rot.from_matrix(Rr).inv() * Rot.from_matrix(Ra)).magnitude())
    if np.any(axis) and order == 2:
        flip = Rot.from_rotvec(np.pi * axis)
        d2 = np.degrees((Rot.from_matrix(Rr).inv() * Rot.from_matrix(Ra) * flip).magnitude())
        d = np.minimum(d, d2)
    return d


def body_rotation_lsq(Rg, Ro, iters=30):
    """R_gt(t) ~= RA @ R_o(t) @ RB : alternating Procrustes; returns RB (our body -> scan body rotation)."""
    RB = np.eye(3)
    for _ in range(iters):
        M = sum(Rg[t] @ (Ro[t] @ RB).T for t in range(len(Rg)))
        U, _, Vt = np.linalg.svd(M)
        RA = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        M = sum((RA @ Ro[t]).T @ Rg[t] for t in range(len(Rg)))
        U, _, Vt = np.linalg.svd(M)
        RB = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    res = np.degrees([Rot.from_matrix(Rg[t].T @ RA @ Ro[t] @ RB).magnitude() for t in range(len(Rg))])
    return RB, float(np.median(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    a = ap.parse_args()
    sys.argv = [sys.argv[0], KIT, DS]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import devscore as D
    from v2dlb.mesh_budget import budget_mesh

    def find_mesh(d, name):
        for ext in ("glb", "obj", "ply", "stl", "off"):
            if os.path.exists(f"{d}/{name}.{ext}"):
                return f"{d}/{name}.{ext}"
        raise FileNotFoundError(name)
    M = D.MODS["AUC"]
    for ep in a.episodes:
        names, ref = D.load_episode(ep)
        T, B, _ = ref.shape
        df = pd.read_parquet(f"{a.pred}/episode_{ep:06d}.parquet").sort_values(["frame_index", "object_slot"])
        pred = df[M._POSE_COLUMNS].to_numpy(np.float64).reshape(T, B, 7)
        ach = pred[:, None].copy()
        rf = ref[:, None].copy()
        axes, orders, regs, regerr = [], [], [], []
        for b, n in enumerate(names):
            g = D.ref_geometry(n)
            v, f = budget_mesh(find_mesh(f"{a.pred}/episode_{ep:06d}", n), 4096, 4096)
            anc, R = M._register_object_body((v, f), (g["v"], g["f"]), g["anchor"], 1_000_003 * ep + 101 * b)
            ach[:, :, b] = M._convert_object_body(ach[:, :, b], anc, R)
            rf[:, :, b, :3] += np.einsum("...ij,j->...i", M._rotation_matrix(rf[:, :, b, 3:]), g["anchor"])
            axes.append(g["axis"])
            orders.append(g["order"])
            regs.append(np.degrees(Rot.from_matrix(R).magnitude()))
        al = M._align_registered_scene(ach, rf, axes)[:, 0]
        r0 = rf[:, 0]
        # oracle: best rigid transform over all frames/objects (positions)
        P, Q = ach[:, 0, :, :3].reshape(-1, 3), r0[..., :3].reshape(-1, 3)
        Pc, Qc = P - P.mean(0), Q - Q.mean(0)
        U, S, Vt = np.linalg.svd(Pc.T @ Qc)
        dd = np.sign(np.linalg.det(Vt.T @ U.T))
        Ro = Vt.T @ np.diag([1, 1, dd]) @ U.T
        orc = ach[:, 0].copy()
        orc[..., :3] = (ach[:, 0, :, :3] - P.mean(0)) @ Ro.T + Q.mean(0)
        orc[..., 3:] = (Rot.from_matrix(Ro) * Rot.from_quat(ach[:, 0, :, 3:].reshape(-1, 4))).as_quat().reshape(T, B, 4)
        print(f"== episode {ep} ({T} frames)")
        for b, n in enumerate(names):
            e = np.linalg.norm(al[:, b, :3] - r0[:, b, :3], axis=1) * 100
            eo = np.linalg.norm(orc[:, b, :3] - r0[:, b, :3], axis=1) * 100
            re = rot_err_deg(al[:, b, 3:], r0[:, b, 3:], axes[b], orders[b])
            reo = rot_err_deg(orc[:, b, 3:], r0[:, b, 3:], axes[b], orders[b])
            gtm = np.linalg.norm(r0[:, b, :3] - r0[0, b, :3], axis=1).max() * 100
            om = np.linalg.norm(al[:, b, :3] - al[0, b, :3], axis=1).max() * 100
            print(f"  {n:18s} reg_rot {regs[b]:6.1f}deg | scorer-aligned pos cm mean {e.mean():5.2f} med {np.median(e):5.2f} "
                  f"p90 {np.percentile(e, 90):5.2f} f0 {e[0]:5.2f} last {e[-1]:5.2f} | rot deg mean {re.mean():5.1f} f0 {re[0]:5.1f} "
                  f"| oracle pos mean {eo.mean():5.2f} rot {reo.mean():5.1f} | max disp gt {gtm:5.1f} ours {om:5.1f}")


if __name__ == "__main__":
    main()
