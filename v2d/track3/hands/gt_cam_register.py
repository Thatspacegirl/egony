"""DEV ONLY (PUBLIC episodes): camera pose in the mocap world by registering the public GT-posed scans to stereo depth.

For every frame t (stride --stride, others interpolated): scene model = public scan points of every object at its
GT pose at t (mocap world); target = rectified-left stereo point cloud (SGBM, or FoundationStereo when complete),
0.15-1.3 m, with points within 4 cm of our own hand joints removed (so ICP locks onto objects, not hands).
Point-to-plane ICP (3 -> 1.5 -> 0.8 cm) from the previous frame's pose gives rect_T_mocap(t); the first frame is
initialised by aligning mocap +z to the table-plane normal, centring the model on the above-table points and trying
36 yaw angles. Only objects that are static in GT around t (< 2 cm/s, < 5 deg/s) form the model when any is static
(hand-held objects are occluded and suffer mocap/video timing offsets); ICP starts from a constant-velocity prediction
(then from the last accepted pose); a result is accepted if its fitness (1 cm) >= 0.4, or >= 0.2 with a jump < 5 cm /
6 deg from its init; otherwise the frame is invalid. Output, in the vo_cuvslam.npz layout so hands_to_bundle.py can use it like VO:

  HANDS/public/episode_X/gt_cam_register.npz: world_T_rect (T,4,4) [world = mocap], valid (T,), fitness (T,),
  rmse (T,)  + .json summary + an overlay PNG (projected GT scans on cam_a) for a visual check.

Refuses evaluation episodes. Public GT is used ONLY for development scoring of the hand pipeline.
  /mnt/secondary/v2d/envs/t3-fpose/bin/python -I gt_cam_register.py --episodes 12 --stride 2
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handlib as hl  # noqa: E402

PUBLIC = "/home/asubuntudesktop/TestingGrounds/egony/video_to_data_challenge/track_3/public"


def load_gt(ep):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(f"{PUBLIC}/data/chunk-000/episode_{ep:06d}.parquet")
    assert (f.schema_arrow.metadata or {}).get(b"pose_convention") == b"world_T_object"
    objs = f.read(columns=["observation.objects"])["observation.objects"].to_pylist()
    names = [o["name"] for o in objs[0]]
    pose = np.array([[o["pose"] for o in fr] for fr in objs], np.float64)
    vis = np.array([[o["visible"] for o in fr] for fr in objs], bool)
    return names, pose, vis


def load_scan_points(name, n=6000):
    import trimesh
    d = f"{PUBLIC}/mesh/{name}"
    p = next(f"{d}/{c}" for c in (f"{name}.glb", f"{name}_visual.glb") if os.path.exists(f"{d}/{c}"))
    m = trimesh.load(p, force="mesh")
    pts, fi = trimesh.sample.sample_surface_even(m, n, seed=0)
    return np.asarray(pts), np.asarray(m.face_normals[fi]), m


def quat_R(q):
    from scipy.spatial.transform import Rotation as R
    return R.from_quat(np.asarray(q)[..., [1, 2, 3, 0]]).as_matrix()


def static_mask(pose, t, half=3, v_max=0.02, w_max=5.0, fps=20.0):
    """(B,) objects whose GT pose is static over [t-half, t+half] (< 2 cm/s and < 5 deg/s)."""
    from scipy.spatial.transform import Rotation as R
    a, b = max(0, t - half), min(len(pose) - 1, t + half)
    if b == a:
        return np.ones(pose.shape[1], bool)
    dt = (b - a) / fps
    v = np.linalg.norm(pose[b, :, :3] - pose[a, :, :3], axis=-1) / dt
    ra, rb = R.from_quat(pose[a, :, [4, 5, 6, 3]].T), R.from_quat(pose[b, :, [4, 5, 6, 3]].T)
    w = np.degrees((ra.inv() * rb).magnitude()) / dt
    return (v < v_max) & (w < w_max)


def model_at(t, pose, scans, use=None):
    P, N = [], []
    for b, (pts, nrm, _) in enumerate(scans):
        if use is not None and not use[b]:
            continue
        Rb = quat_R(pose[t, b, 3:])
        P.append(pts @ Rb.T + pose[t, b, :3])
        N.append(nrm @ Rb.T)
    return np.concatenate(P), np.concatenate(N)


def cloud(depth, Kr, hands_X=None, zmin=0.15, zmax=1.3, step=2):
    v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    z = depth[v, u]
    ok = (z > zmin) & (z < zmax)
    X = np.stack([(u[ok] - Kr[0, 2]) / Kr[0, 0] * z[ok], (v[ok] - Kr[1, 2]) / Kr[1, 1] * z[ok], z[ok]], 1)
    if hands_X is not None and len(hands_X):
        from scipy.spatial import cKDTree
        d, _ = cKDTree(hands_X).query(X, distance_upper_bound=0.04)
        X = X[~np.isfinite(d)]
    return X


def run(args):
    ep_dir, stride, overwrite = args
    import cv2
    import open3d as o3d
    cv2.setNumThreads(2)
    meta, Ka, Kr, T_a_rect = hl.load_meta(ep_dir)
    if meta["split"] != "public":
        raise SystemExit("dev-only: refusing a non-public episode")
    ep = meta["episode"]
    od = hl.out_dir(meta)
    out = f"{od}/gt_cam_register.npz"
    if os.path.exists(out) and not overwrite:
        return f"skip {out}"
    t0 = time.time()
    names, pose, vis = load_gt(ep)
    T = meta["n_frames"]
    assert pose.shape[0] == T, (pose.shape, T)
    scans = [load_scan_points(n) for n in names]
    hz = np.load(f"{od}/hands_rect.npz") if os.path.exists(f"{od}/hands_rect.npz") else None
    ds = hl.RectDepth(ep_dir, meta)
    reg = o3d.pipelines.registration

    def pcd(X, N=None):
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
        if N is not None:
            p.normals = o3d.utility.Vector3dVector(N)
        return p

    def target_at(t):
        d, _ = ds.get(t)
        hx = None
        if hz is not None:
            hx = np.concatenate([hz[s][t] for s in hl.SIDES if hz[f"{s}_valid"][t]] or [np.zeros((0, 3))])
        tg = pcd(cloud(d, Kr, hx)).voxel_down_sample(0.006)
        tg.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        tg.orient_normals_towards_camera_location(np.zeros(3))
        return tg

    def icp(src, tg, init, dists=(0.03, 0.015, 0.008), iters=20):
        Tm = init
        for dd in dists:
            r = reg.registration_icp(src, tg, dd, Tm, reg.TransformationEstimationPointToPlane(
                reg.TukeyLoss(k=dd)), reg.ICPConvergenceCriteria(max_iteration=iters))
            Tm = r.transformation
        return Tm, r.fitness, r.inlier_rmse

    def global_init(t, tg):
        X = np.asarray(tg.points)
        plane, inl = tg.segment_plane(0.01, 3, 500)
        n = np.array(plane[:3])
        dpl = plane[3]
        if n @ np.array([0, -1.0, 0]) < 0:  # up in the rect frame is roughly -y (camera y points down)
            n, dpl = -n, -dpl
        h = X @ n + dpl
        above = X[(h > 0.01) & (h < 0.3)]
        c_obs = np.median(above, 0) if len(above) > 50 else X.mean(0)
        src_P, src_N = model_at(t, pose, scans)
        src = pcd(src_P, src_N)
        c_m = src_P.mean(0)
        # rotation taking mocap +z to n, then yaw about n
        z = np.array([0, 0, 1.0])
        v = np.cross(z, n)
        s, c = np.linalg.norm(v), z @ n
        Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R0 = np.eye(3) + Vx + Vx @ Vx * ((1 - c) / max(s * s, 1e-12))
        best = None
        for yaw in np.radians(np.arange(0, 360, 10)):
            Ry = cv2.Rodrigues(n * yaw)[0] @ R0
            Tm = np.eye(4)
            Tm[:3, :3] = Ry
            Tm[:3, 3] = c_obs - Ry @ c_m
            Tm2, _, _ = icp(src, tg, Tm, dists=(0.05, 0.03, 0.015, 0.008), iters=30)
            ev = reg.evaluate_registration(src, tg, 0.01, Tm2)
            if best is None or ev.fitness > best[0]:
                best = (ev.fitness, Tm2)
        return best[1]

    frames = list(range(0, T, stride))
    if frames[-1] != T - 1:
        frames.append(T - 1)
    rect_T_m = np.full((T, 4, 4), np.nan)
    fitness = np.full(T, np.nan)
    rmse = np.full(T, np.nan)
    from scipy.spatial.transform import Rotation as Rr
    hist = []  # (t, rect_T_mocap) of accepted frames
    n_reject = 0
    ok_frame = {}
    n_static_model = 0
    for t in frames:
        tg = target_at(t)
        use = static_mask(pose, t)
        if not use.any():
            use = np.ones(len(names), bool)
        else:
            n_static_model += 1
        src_P, src_N = model_at(t, pose, scans, use)
        src = pcd(src_P, src_N)
        if not hist:
            init = global_init(t, tg)
            Tm, fit, rm = icp(src, tg, init)
            ok = True  # first frame: the best of the yaw search is accepted (checked on the overlay PNG)
        else:
            # constant-velocity prediction from the last two accepted frames (in rect_T_mocap)
            tp, Tp = hist[-1]
            init = Tp
            if len(hist) >= 2:
                tq, Tq = hist[-2]
                D = Tp @ np.linalg.inv(Tq)  # motion over (tp - tq) frames
                k = (t - tp) / max(tp - tq, 1)
                if k <= 3:
                    rv = Rr.from_matrix(D[:3, :3]).as_rotvec() * k
                    Dk = np.eye(4)
                    Dk[:3, :3] = Rr.from_rotvec(rv).as_matrix()
                    Dk[:3, 3] = D[:3, 3] * k
                    init = Dk @ Tp
            def jump(A, B):
                J = A @ np.linalg.inv(B)
                return np.linalg.norm(J[:3, 3]), np.degrees(Rr.from_matrix(J[:3, :3]).magnitude())

            best = None
            for ini in ([init, Tp] if len(hist) >= 2 else [Tp]):
                Tm, fit, rm = icp(src, tg, ini)
                jt, jr = jump(Tm, ini)
                ok = fit >= 0.4 or (fit >= 0.2 and jt < 0.05 and jr < 6.0)
                if best is None or (ok, fit) > (best[0], best[2]):
                    best = (ok, Tm, fit, rm)
                if ok:
                    break
            ok, Tm, fit, rm = best
            if not ok:
                n_reject += 1
                Tm = init
        rect_T_m[t], fitness[t], rmse[t] = Tm, fit, rm
        ok_frame[t] = ok
        if ok:
            hist.append((t, Tm))
    # interpolate skipped frames (SLERP rotation, linear translation)
    from scipy.spatial.transform import Rotation as Rr, Slerp
    W = np.array([np.linalg.inv(rect_T_m[t]) for t in frames])  # world_T_rect
    sl = Slerp(frames, Rr.from_matrix(W[:, :3, :3]))
    world_T_rect = np.tile(np.eye(4), (T, 1, 1))
    world_T_rect[:, :3, :3] = sl(np.arange(T)).as_matrix()
    for c in range(3):
        world_T_rect[:, c, 3] = np.interp(np.arange(T), frames, W[:, c, 3])
    valid = np.zeros(T, bool)
    good = np.array([ok_frame[t] for t in frames])
    for i, t in enumerate(frames):
        valid[t] = good[i]
    for i in range(len(frames) - 1):
        if good[i] and good[i + 1]:
            valid[frames[i]:frames[i + 1] + 1] = True
    np.savez_compressed(out, world_T_rect=world_T_rect, valid=valid, fitness=fitness, rmse=rmse,
                        frames=np.array(frames), object_names=np.array(names), depth_source=np.array(ds.source))
    # overlay check on cam_a at 3 frames
    rows = []
    for t in (frames[0], frames[len(frames) // 2], frames[-1]):
        A = cv2.imread(f"{ep_dir}/cam_a/{t:06d}.jpg")
        P, _ = model_at(t, pose, scans)
        Xr = P @ np.linalg.inv(world_T_rect[t])[:3, :3].T + np.linalg.inv(world_T_rect[t])[:3, 3]
        uv = hl.project(hl.transform(T_a_rect, Xr), Ka)
        for u in uv[::3]:
            if 0 <= u[0] < A.shape[1] and 0 <= u[1] < A.shape[0]:
                cv2.circle(A, (int(u[0]), int(u[1])), 3, (0, 255, 0), -1)
        cv2.putText(A, f"ep{ep} f{t} fit={fitness[t]:.2f} rmse={rmse[t] * 1000:.1f}mm", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        rows.append(cv2.resize(A, (676, 507)))
    cv2.imwrite(f"{od}/gt_cam_register.jpg", np.hstack(rows))
    summ = {"episode": ep, "objects": names, "frames_registered": len(frames), "valid_frac": float(valid.mean()),
            "fitness_median": float(np.nanmedian(fitness)), "rmse_median_mm": float(np.nanmedian(rmse) * 1000),
            "n_rejected": n_reject, "frames_static_model": n_static_model, "depth_source": ds.source, "seconds": round(time.time() - t0, 1)}
    json.dump(summ, open(f"{od}/gt_cam_register.json", "w"), indent=1)
    return json.dumps(summ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    jobs = [(d, a.stride, a.overwrite) for d in hl.episode_dirs(a.episodes, ("public",))]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for msg in pool.imap_unordered(run, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())
