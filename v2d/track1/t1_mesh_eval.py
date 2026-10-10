#!/usr/bin/env python3
"""Val-only diagnostics of our object mesh / reference pose against FORM-HOI GT (GT is read ONLY here, for scoring).

Per val episode (needs mesh/<ep>/mesh_0.ply + fpose/<ep>/fpose.npz):
  shape_cd_cm     symmetric Chamfer (mean of both directions, cm) between OUR mesh at OUR metric scale (FoundationPose
                  scale, MoGe-2 metric) and the GT mesh, after the best RIGID alignment (multi-start ICP)
  shape_cd_sim_cm same after a similarity alignment (scale free: shape only), reported in GT units
  scale_ratio     our scale / best similarity scale  (1 = metric size right)
  ref_pose_cd_cm  Chamfer between our posed mesh and the GT posed mesh at the reference frame, camera frame
                  (GT pose = poses.npy composed with the edex camera; measures registration + depth error)
  ref_t_err_cm    centroid distance at the reference frame;  ref_depth_ratio our/GT centroid depth
  kit_cd64_cm     the scorer's CD-O sampling (64 area-weighted points per mesh, seed = episode) at the reference
                  frame (rigid, no Sim(3)) - same units as CD-O

  python t1_mesh_eval.py --episodes 0-24 --json /mnt/secondary/v2d/t1/work/mesh_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_items as I  # noqa: E402

CAM_INDEX = {"front_stereo_camera_left": 0, "back_stereo_camera_left": 2, "left_stereo_camera_left": 4,
             "right_stereo_camera_left": 6}
R = Path("/mnt/secondary/v2d/t1")


def sample(mesh, n, seed=0):
    import trimesh
    p, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(p)


def chamfer(A, B):
    from scipy.spatial import cKDTree
    return 0.5 * (cKDTree(B).query(A)[0].mean() + cKDTree(A).query(B)[0].mean())


def icp(src, dst, iters=30, with_scale=False):
    """align src -> dst; returns (s, R, t, cd)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(dst)
    s, Rm, t = 1.0, np.eye(3), np.zeros(3)
    for _ in range(iters):
        X = s * src @ Rm.T + t
        _, nn = tree.query(X)
        Y = dst[nn]
        mx, my = src.mean(0), Y.mean(0)
        A, B = src - mx, Y - my
        U, S, Vt = np.linalg.svd(A.T @ B)
        Dm = np.eye(3); Dm[2, 2] = np.sign(np.linalg.det(U @ Vt))
        Rm = (U @ Dm @ Vt).T
        if with_scale:
            s = (S * np.diag(Dm)).sum() / (A ** 2).sum()
        t = my - s * mx @ Rm.T
    X = s * src @ Rm.T + t
    return s, Rm, t, chamfer(X, dst)


def rot_candidates():
    from scipy.spatial.transform import Rotation
    return Rotation.create_group("O").as_matrix()          # 24 rotations


def best_align(src, dst, with_scale):
    """returns (cd, total_scale_applied_to_src, R, Rc)."""
    best = None
    src0 = src - src.mean(0)
    dst0 = dst - dst.mean(0)
    pre = 1.0
    if with_scale:
        pre = np.sqrt((dst0 ** 2).sum(1).mean()) / np.sqrt((src0 ** 2).sum(1).mean())
    for Rc in rot_candidates():
        s, Rm, t, cd = icp(pre * src0 @ Rc.T, dst0, iters=20, with_scale=with_scale)
        if best is None or cd < best[0]:
            best = (cd, s * pre, Rm, Rc)
    return best


def gt_cam_pose(it, frame):
    d = json.load(open(I.VAL_ROOT / it["sequence_id"] / "edex"))
    c = d[0]["cameras"][CAM_INDEX[it["camera"]]]
    Tcw = np.asarray(c["transform"], float)
    Rcw, tcw = Tcw[:, :3], Tcw[:, 3]
    Pw = np.load(I.VAL_ROOT / it["sequence_id"] / "poses.npy")[frame]
    Rw, tw = Pw[:3, :3], Pw[:3, 3]
    return Rcw.T @ Rw, Rcw.T @ (tw - tcw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="0-24")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--json")
    a = ap.parse_args()
    import trimesh
    from t1_run import parse_eps
    its = {it["episode"]: it for it in I.items("val")}
    rows = {}
    for ep in parse_eps(a.episodes, its):
        E = f"episode_{ep:06d}"
        mp, fp = R / "mesh" / "val" / E / "mesh_0.ply", R / "fpose" / "val" / E / "fpose.npz"
        if not (mp.exists() and fp.exists()):
            continue
        it = its[ep]
        ours = trimesh.load(mp, process=False)
        gt = trimesh.load(I.VAL_ROOT / it["sequence_id"] / "object_mesh__output_aligned.glb", force="mesh", process=False)
        F = dict(np.load(fp))
        s = float(F["scale"])
        A = sample(ours, a.n, 0) * s
        B = sample(gt, a.n, 1)
        cd_r = best_align(A, B, False)[0]
        cd_s, s_sim, _, _ = best_align(A, B, True)
        scale_ratio = 1.0 / s_sim
        ri = int(F["ref_index"]); fr = int(F["frames"][ri])
        Tm = F["cam_T_obj"][ri]
        Ao = sample(ours, a.n, 0) * s @ Tm[:3, :3].T + Tm[:3, 3]
        Rg, tg = gt_cam_pose(it, fr)
        Bo = sample(gt, a.n, 1) @ Rg.T + tg
        # kit CD-O sampling (64 points each, rigid) at the reference frame
        import t1lib as L
        cdh = L.metric_module("CD-H")
        from v2dlb.mesh_budget import budget_mesh
        tmpo = trimesh.Trimesh(np.asarray(ours.vertices) * s, ours.faces, process=False)
        po = Path("/tmp") / f"_t1me_{ep}.ply"; tmpo.export(po)
        mo, fo = budget_mesh(po, 4096, 4096); po.unlink()
        mg, fg = budget_mesh(I.VAL_ROOT / it["sequence_id"] / "object_mesh__output_aligned.glb", 4096, 4096)
        lo = cdh._sample_mesh_surface(mo, fo, 64, ep) @ Tm[:3, :3].T + Tm[:3, 3]
        lg = cdh._sample_mesh_surface(mg, fg, 64, ep) @ Rg.T + tg
        rows[ep] = {"object": it["object"], "ref_frame": fr, "scale_m": s, "gt_extent_m": gt.extents.tolist(),
                    "our_extent_m": (ours.extents * s).tolist(),
                    "shape_cd_cm": 100 * cd_r, "shape_cd_sim_cm": 100 * cd_s, "scale_ratio": float(scale_ratio),
                    "ref_pose_cd_cm": 100 * chamfer(Ao, Bo),
                    "ref_t_err_cm": 100 * float(np.linalg.norm(Ao.mean(0) - Bo.mean(0))),
                    "ref_depth_ratio": float(Ao.mean(0)[2] / Bo.mean(0)[2]),
                    "kit_cd64_cm": 100 * float(2 * chamfer(lo, lg))}
        r = rows[ep]
        print(f"ep {ep:2d} {it['object']:28s} shapeCD {r['shape_cd_cm']:6.2f} simCD {r['shape_cd_sim_cm']:6.2f} "
              f"scale x{r['scale_ratio']:.2f} refCD {r['ref_pose_cd_cm']:6.1f} t_err {r['ref_t_err_cm']:6.1f} "
              f"depth x{r['ref_depth_ratio']:.3f} kitCD64 {r['kit_cd64_cm']:6.1f}", flush=True)
    if rows:
        keys = ("shape_cd_cm", "shape_cd_sim_cm", "scale_ratio", "ref_pose_cd_cm", "ref_t_err_cm", "ref_depth_ratio", "kit_cd64_cm")
        med = {k: float(np.median([r[k] for r in rows.values()])) for k in keys}
        mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in keys}
        print("median", json.dumps({k: round(v, 3) for k, v in med.items()}))
        print("mean  ", json.dumps({k: round(v, 3) for k, v in mean.items()}))
        if a.json:
            json.dump({"per_episode": rows, "median": med, "mean": mean}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
