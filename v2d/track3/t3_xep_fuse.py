"""Cross-episode fusion of ONE physical object seen in several episodes of the SAME split (CPU, Open3D).

Every input is a per-episode mesh in its own body frame plus its JSON (t3_tsdf_fuse.py / t3_mesh_complete.py:
frame_T_obj = VO-world pose of the body during the static window, `frames` = window stereo frames, `poses` = VO npz).
1. Up vector per episode: the table normal in that body frame (the object rests on the table in the window).
2. Each episode's mesh is registered to the reference episode by an up-constrained yaw search (36 yaws x 2 flips of the
   up axis sign excluded -> objects keep resting on their base) + point-to-plane ICP on 4000 surface samples; the
   result is kept if fitness >= --min_fitness and RMSE <= --max_rmse.
3. One TSDF over the window frames of all accepted episodes in the reference body frame
   (canon_T_rect = canon_T_body_e @ inv(frame_T_obj_e) @ world_T_rect_e(f)), largest component, Taubin.
4. Per episode e: OUT_ROOT/<split>/episode_e/<obj>.ply (the fused mesh, identical for all e) and .json with
   frame_T_obj = frame_T_obj_e @ inv(canon_T_body_e), so t3_fpose_track.py --stage1_json works unchanged.

  python -I t3_xep_fuse.py --split public --obj plastic_dish_rack --episodes 12 13 26 \
      --in_root /mnt/secondary/v2d/t3/meshes/stage1 --out_root /mnt/secondary/v2d/t3/meshes/xep
Use only episodes of ONE split (evaluation meshes are never built from public videos).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t3_depth_warp import decode_inv_depth  # noqa: E402
from t3_mesh_complete import table_in_body  # noqa: E402
from t3_tsdf_fuse import clean, clean_mask_depth  # noqa: E402

R0 = "/mnt/secondary/v2d/t3"


def rot_about(axis, ang):
    return cv2.Rodrigues(np.asarray(axis, float) * ang)[0]


def sample(mesh_path, n=4000):
    m = o3d.io.read_triangle_mesh(mesh_path)
    m.compute_vertex_normals()
    p = m.sample_points_uniformly(n)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    return p


def protrusion_azimuth(pcd, up):
    """azimuth (rad, in the plane normal to `up`) of a handle-like protrusion: points beyond 1.35x the median radial
    distance from the vertical axis through the robust centre; None if fewer than 3 % of the points."""
    P = np.asarray(pcd.points)
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(up, a)
    u /= np.linalg.norm(u)
    v = np.cross(up, u)
    xy = np.c_[P @ u, P @ v]
    c = np.median(xy, 0)
    r = np.linalg.norm(xy - c, axis=1)
    out = r > 1.35 * np.median(r)
    if out.mean() < 0.03:
        return None, (u, v)
    d = (xy[out] - c).mean(0)
    return float(np.arctan2(d[1], d[0])), (u, v)


def register(src, dst, up_src, up_dst, max_dist=0.01):
    """T (4x4) mapping src body -> dst body with up_src -> up_dst, best yaw by ICP fitness."""
    # rotation taking up_src to up_dst
    v = np.cross(up_src, up_dst)
    c = float(np.dot(up_src, up_dst))
    R0_ = np.eye(3) if np.linalg.norm(v) < 1e-9 else rot_about(v / np.linalg.norm(v), np.arccos(np.clip(c, -1, 1)))
    cs = np.asarray(src.points).mean(0)
    cd = np.asarray(dst.points).mean(0)
    best = None
    yaws = [2 * np.pi * k / 36 for k in range(36)]
    az_s, _ = protrusion_azimuth(src, up_src)
    az_d, (ud, vd) = protrusion_azimuth(dst, up_dst)
    if az_s is not None and az_d is not None:  # handle-like protrusion in both: only yaws that align them (+-20 deg)
        # direction of the source protrusion after R0_, expressed as an azimuth in the destination plane basis
        Ps = np.asarray(src.points)
        a = np.array([1.0, 0, 0]) if abs(up_src[0]) < 0.9 else np.array([0, 1.0, 0])
        us = np.cross(up_src, a)
        us /= np.linalg.norm(us)
        vs = np.cross(up_src, us)
        dir_s = R0_ @ (np.cos(az_s) * us + np.sin(az_s) * vs)
        az_s_d = np.arctan2(dir_s @ vd, dir_s @ ud)
        base = az_d - az_s_d
        yaws = [base + np.radians(x) for x in range(-20, 21, 5)]
    for yaw in yaws:
        R = rot_about(up_dst, yaw) @ R0_
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = cd - R @ cs
        r = o3d.pipelines.registration.registration_icp(
            src, dst, max_dist * 2, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        r = o3d.pipelines.registration.registration_icp(
            src, dst, max_dist, r.transformation, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        # keep the base on the table: penalise solutions that tilt the up axis
        tilt = np.degrees(np.arccos(np.clip(np.dot(r.transformation[:3, :3] @ up_src, up_dst), -1, 1)))
        score = r.fitness - (0.5 if tilt > 15 else 0.0)
        if best is None or score > best[0]:
            best = (score, r, tilt)
    return best[1].transformation, float(best[1].fitness), float(best[1].inlier_rmse), float(best[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--ref", type=int, default=None, help="reference episode (default: largest surface area)")
    ap.add_argument("--in_root", default=f"{R0}/meshes/stage1")
    ap.add_argument("--out_root", default=f"{R0}/meshes/xep")
    ap.add_argument("--min_fitness", type=float, default=0.5)
    ap.add_argument("--max_rmse", type=float, default=0.006)
    ap.add_argument("--voxel", type=float, default=0.002)
    a = ap.parse_args()
    eps = []
    for ep in a.episodes:
        e = f"episode_{ep:06d}"
        p = f"{a.in_root}/{a.split}/{e}/{a.obj}"
        if not os.path.exists(p + ".ply"):
            continue
        s1 = json.load(open(p + ".json"))
        F = np.array(s1["frame_T_obj"])
        vo = s1["poses"].split(":")[0]
        tb = table_in_body(a.split, e, s1["frames"], F, vo)
        if tb is None:
            print("no table plane", e)
            continue
        m = o3d.io.read_triangle_mesh(p + ".ply")
        eps.append(dict(ep=ep, e=e, path=p + ".ply", s1=s1, F=F, vo=vo, up=tb[0], area=m.get_surface_area(),
                        pcd=sample(p + ".ply")))
    if not eps:
        raise SystemExit("no inputs")
    ref = next((x for x in eps if x["ep"] == a.ref), None) or max(eps, key=lambda x: x["area"])
    report = {}
    for x in eps:
        if x is ref:
            x["T"] = np.eye(4)
            report[x["ep"]] = dict(ref=True)
            continue
        T, fit, rmse, tilt = register(x["pcd"], ref["pcd"], x["up"], ref["up"])
        ok = fit >= a.min_fitness and rmse <= a.max_rmse
        x["T"] = T if ok else None
        report[x["ep"]] = dict(fitness=round(fit, 3), rmse_mm=round(rmse * 1000, 2), tilt_deg=round(tilt, 1), used=ok)
        print(a.obj, x["e"], "->", ref["e"], report[x["ep"]], flush=True)
    # multi-episode TSDF in the reference body frame
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=a.voxel, sdf_trunc=4 * a.voxel, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    n_views = 0
    for x in eps:
        if x["T"] is None:
            continue
        meta = json.load(open(f"{R0}/frames/{a.split}/{x['e']}/meta.json"))
        K = np.array(meta["images"]["left"]["K"])
        W, H = meta["images"]["left"]["size_wh"]
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        Wr = np.load(x["vo"])["world_T_rect"]
        canon_T_world = x["T"] @ np.linalg.inv(x["F"])
        for f in x["s1"]["frames"]:
            if not np.isfinite(Wr[f]).all():
                continue
            d = decode_inv_depth(cv2.imread(f"{R0}/depth/{a.split}/{x['e']}/depth_rect/{f:06d}.png", cv2.IMREAD_UNCHANGED))
            mk = cv2.imread(f"{R0}/masks/{a.split}/{x['e']}/left_s1/{a.obj}/{f:06d}.png", 0) > 0
            mk = clean_mask_depth(d, mk, 3)
            L = cv2.imread(f"{R0}/frames/{a.split}/{x['e']}/left/{f:06d}.jpg")
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(L[..., ::-1])), o3d.geometry.Image(np.where(mk, d, 0).astype(np.float32)),
                depth_scale=1.0, depth_trunc=2.0, convert_rgb_to_intensity=False)
            vol.integrate(rgbd, intr, np.linalg.inv(canon_T_world @ Wr[f]))
            n_views += 1
    mesh = clean(vol.extract_triangle_mesh())
    V = np.asarray(mesh.vertices)
    c = (V.max(0) + V.min(0)) / 2  # bbox centre at the origin (FoundationPose convention)
    mesh.translate(-c)
    Tc = np.eye(4)
    Tc[:3, 3] = c
    V = np.asarray(mesh.vertices)
    print(f"{a.obj}: fused {n_views} views from {sum(x['T'] is not None for x in eps)} episodes -> {len(V)} verts, "
          f"extent {np.round(np.ptp(V, 0) * 100, 1)} cm")
    for x in eps:
        if x["T"] is None:
            continue
        od = f"{a.out_root}/{a.split}/{x['e']}"
        os.makedirs(od, exist_ok=True)
        o3d.io.write_triangle_mesh(f"{od}/{a.obj}.ply", mesh)
        s1 = dict(x["s1"])
        s1["frame_T_obj"] = (x["F"] @ np.linalg.inv(x["T"]) @ Tc).tolist()
        s1["xep"] = dict(ref=ref["e"], episodes=[y["e"] for y in eps if y["T"] is not None], report=report,
                         canon_T_body=x["T"].tolist(), in_root=a.in_root)
        json.dump(s1, open(f"{od}/{a.obj}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
