"""Assemble per-episode Track 3 perception output (object world poses in the VO world) + dev-score adapter.

Inputs per episode (defaults under /mnt/secondary/v2d/t3):
  frames/<split>/episode_X/meta.json                 objects (slot order), T_a_rect, n_frames
  vo_masked|vo/<split>/episode_X/vo_cuvslam.npz       world_T_rect[T,4,4] (world = rect-left frame at frame 0)
  fpose/<split>/episode_X/<obj>.npz                   cam_T_obj[T,4,4] in the undistorted cam_a frame (mesh coords),
                                                      iou[T] (t3_fpose_track.py)
  meshes/<stage>/<split>/episode_X/<obj>.ply          the mesh those poses place
  depth/<split>/episode_X/depth_rect/000000.png      for the table plane

Output perception/<split>/episode_X/:
  perception.npz   t3_perception_v1-compatible arrays (drop-in for sharpa_task/bundle.py, minus `schema`):
     episode_index, num_frames, split, object_names (B,), object_mesh_paths (B,),
     object_pose (N,B,7) [x,y,z,qw,qx,qy,qz] world_T_object in the VO world, object_valid (N,B),
     object_conf (N,B) (FoundationPose rendered-vs-SAM3 mask IoU, NaN = no mask),
     world_T_rect (N,4,4), world_T_cam_a (N,4,4), cam_a_T_obj (N,B,4,4),
     table_plane (4,) [nx,ny,nz,d] (n.x + d = 0, n up = towards the camera), up (3,)
  perception.json  summary + provenance
--devscore DIR additionally writes DIR/episode_X.parquet + DIR/episode_X/<obj>.ply for t3_devscore.py (PUBLIC only).

  python -I t3_assemble.py --split public --episode 12 [--mesh_stage stage1] [--devscore /mnt/.../pred_v1]
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t3_depth_warp import decode_inv_depth  # noqa: E402
from t3_smooth import clamp_static, smooth_track  # noqa: E402

R0 = "/mnt/secondary/v2d/t3"


def fill_poses(T):
    """Fill NaN frames of a [N,4,4] pose track: interpolate inside, hold at the ends."""
    T = T.copy()
    ok = np.isfinite(T).all((1, 2))
    if ok.all() or not ok.any():
        return T, ok
    idx = np.nonzero(ok)[0]
    t = T[idx, :3, 3]
    sl = Slerp(idx, Rotation.from_matrix(T[idx, :3, :3]))
    q = np.clip(np.arange(len(T)), idx[0], idx[-1])
    T[:, :3, :3] = sl(q).as_matrix()
    for k in range(3):
        T[:, k, 3] = np.interp(np.arange(len(T)), idx, t[:, k])
    T[:, 3] = [0, 0, 0, 1]
    return T, ok


def hand_contact(split, e, obj, N, contact_px=10, cam="cam_a_s0.5"):
    """bool[N]: dilated hand mask overlaps the object mask (or the object is not visible at all -> True)."""
    md = f"{R0}/masks/{split}/{e}/{cam}"
    k = np.ones((2 * contact_px + 1, 2 * contact_px + 1), np.uint8)
    out = np.ones(N, bool)
    for f in range(N):
        m = cv2.imread(f"{md}/{obj}/{f:06d}.png", 0)
        h = cv2.imread(f"{md}/hand/{f:06d}.png", 0)
        if m is None or not m.any():
            continue
        out[f] = h is not None and (cv2.dilate((h > 0).astype(np.uint8), k) > 0)[m > 0].any()
    return out


def table_plane_rect(depth, K, excl=None, n_iter=300, thr=0.008, seed=0, max_depth=1.0):
    """RANSAC plane on rect-left depth (0.25-max_depth m, lower 2/3 of the image, excluding masks).  n points to
    camera.  max_depth 1.0 keeps the table (0.4-0.9 m away) and drops the floor (>= 1.1 m below the head)."""
    H, W = depth.shape
    m = (depth > 0.25) & (depth < max_depth)
    m[: H // 3] = False
    if excl is not None:
        m &= ~excl
    v, u = np.nonzero(m)
    if len(u) < 500:
        return None, 0.0
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(u), min(len(u), 40000), replace=False)
    z = depth[v[sel], u[sel]]
    P = np.stack([(u[sel] - K[0, 2]) / K[0, 0] * z, (v[sel] - K[1, 2]) / K[1, 1] * z, z], 1)
    best, bn = None, 0
    for _ in range(n_iter):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        inl = np.abs(P @ n + d) < thr
        if inl.sum() > bn:
            bn, best = inl.sum(), inl
    Q = P[best]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][2]
    if n @ (-c) < 0:  # normal towards the camera (= up for a table seen from above)
        n = -n
    return np.r_[n, -n @ c], float(bn / len(P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--vo", default="auto", help="auto = vo_masked if present else vo; or a path")
    ap.add_argument("--fpose_dir", default=None, help="default R0/fpose/<split>/episode_X")
    ap.add_argument("--mesh_stage", default="stage1")
    ap.add_argument("--out", default=f"{R0}/perception")
    ap.add_argument("--devscore", default=None)
    ap.add_argument("--min_mask_px", type=int, default=300, help="cam_a s0.5 mask pixels for a trusted measurement")
    ap.add_argument("--min_iou", type=float, default=0.3, help="FoundationPose render/mask IoU for a trusted measurement")
    ap.add_argument("--clamp_static", action="store_true", help="hold one robust pose over every hand-free run")
    ap.add_argument("--static_min_len", type=int, default=5)
    ap.add_argument("--static_margin", type=int, default=2)
    ap.add_argument("--contact_px", type=int, default=10, help="hand-object contact dilation (cam_a s0.5 px)")
    ap.add_argument("--smooth", type=float, default=0.0, help="Gaussian sigma (frames) for IoU-weighted smoothing")
    ap.add_argument("--time_base", default="stereo",
                    help="stereo (frame i = stereo frame i; DEFAULT since 2026-10-09: public dev, 11 eps, clamp: AUC "
                         "0.5992 -> 0.6072, MP-SR 0.455 -> 0.545, better in 9/11 eps, i.e. the GT follows the stereo "
                         "clock) | cam_a (= cam_a frame i) | shift<k> (cam_a frame i+k)")
    a = ap.parse_args()
    e = f"episode_{a.episode:06d}"
    meta = json.load(open(f"{R0}/frames/{a.split}/{e}/meta.json"))
    N, objs = meta["n_frames"], meta["objects"]
    T_a_rect = np.array(meta["transforms"]["T_a_rect"])
    vo = a.vo
    if vo == "auto":
        vo = f"{R0}/vo_masked/{a.split}/{e}/vo_cuvslam.npz"
        if not os.path.exists(vo):
            vo = f"{R0}/vo/{a.split}/{e}/vo_cuvslam.npz"
    W_rect, vo_ok = fill_poses(np.load(vo)["world_T_rect"])
    from t3_sync import load_lag, stereo_index_for_cam_a
    lag = load_lag(a.split, e, N)
    src = stereo_index_for_cam_a(lag) if lag.any() else np.arange(N)
    W_a = W_rect[src] @ np.linalg.inv(T_a_rect)  # world_T_cam_a at the instant of cam_a frame j
    fd = a.fpose_dir or f"{R0}/fpose/{a.split}/{e}"
    B = len(objs)
    pose = np.zeros((N, B, 7))
    valid = np.zeros((N, B), bool)
    conf = np.full((N, B), np.nan)
    camT = np.full((N, B, 4, 4), np.nan)
    meshes = []
    obj_info = {}
    for b, o in enumerate(objs):
        z = np.load(f"{fd}/{o}.npz")
        C = z["cam_T_obj"]
        camT[:, b] = C
        meshes.append(str(z["mesh"]) if "mesh" in z else f"{R0}/meshes/{a.mesh_stage}/{a.split}/{e}/{o}.ply")
        Wo = W_a @ C
        iou = z["iou"] if "iou" in z else np.full(N, np.nan)
        mpx = z["mask_px"] if "mask_px" in z else np.full(N, 1e9)
        # a measurement is trusted when the object is visibly segmented and the rendered mesh overlaps the mask
        meas_ok = np.isfinite(Wo).all((1, 2)) & (mpx >= a.min_mask_px) & ~(iou < a.min_iou)
        Wo[~meas_ok] = np.nan
        Wo, okb = fill_poses(Wo)
        contact = hand_contact(a.split, e, o, N, a.contact_px) if (a.clamp_static or a.smooth > 0) else np.zeros(N, bool)
        untouched = ~contact & (mpx >= a.min_mask_px)
        info_b = {}
        if a.clamp_static:
            Wo, segs = clamp_static(Wo, untouched, min_len=a.static_min_len, margin=a.static_margin)
            info_b["static_segments"] = [list(map(int, x)) for x in segs]
        if a.smooth > 0:
            keep = np.zeros(N, bool)
            for s0, s1 in info_b.get("static_segments", []):
                keep[s0:s1] = True
            Wo = smooth_track(Wo, np.where(meas_ok, np.nan_to_num(iou, nan=0.5), 0.05), a.smooth, a.smooth, keep)
        obj_info[o] = dict(meas_ok_frac=float(meas_ok.mean()), contact_frac=float(contact.mean()), **info_b)
        valid[:, b] = okb & vo_ok
        conf[:, b] = np.where(meas_ok, iou, np.nan)
        pose[:, b, :3] = Wo[:, :3, 3]
        q = Rotation.from_matrix(Wo[:, :3, :3]).as_quat()  # xyzw
        pose[:, b, 3:] = np.c_[q[:, 3], q[:, :3]]
    if a.time_base == "stereo":  # output frame i = stereo frame i = cam_a frame i + lag[i]
        idx = np.clip(np.arange(N) + lag, 0, N - 1)
        pose, valid, conf, camT = pose[idx], valid[idx], conf[idx], camT[idx]
    elif a.time_base.startswith("shift"):  # dev experiments: output frame i = cam_a frame i + k
        k = int(a.time_base[5:])
        idx = np.clip(np.arange(N) + k, 0, N - 1)
        pose, valid, conf, camT = pose[idx], valid[idx], conf[idx], camT[idx]
    # table plane from frame-0 depth (objects + hands excluded), into the VO world
    K = np.array(meta["images"]["left"]["K"])
    d0 = decode_inv_depth(cv2.imread(f"{R0}/depth/{a.split}/{e}/depth_rect/{0:06d}.png", cv2.IMREAD_UNCHANGED))
    excl = np.zeros(d0.shape, bool)
    for s in objs + ["hand"]:
        mp = f"{R0}/masks/{a.split}/{e}/left_s1/{s}/{0:06d}.png"
        if os.path.exists(mp):
            mm = cv2.imread(mp, 0)
            excl |= cv2.dilate(cv2.resize(mm, d0.shape[::-1], interpolation=cv2.INTER_NEAREST), np.ones((15, 15))) > 0
    pl, frac = table_plane_rect(d0, K, excl)
    table_plane = up = None
    if pl is not None:
        n_w = W_rect[0, :3, :3] @ pl[:3]
        p_w = W_rect[0, :3, :3] @ (-pl[3] * pl[:3]) + W_rect[0, :3, 3]
        table_plane = np.r_[n_w, -n_w @ p_w]
        up = n_w
    od = f"{a.out}/{a.split}/{e}"
    os.makedirs(od, exist_ok=True)
    arrays = dict(episode_index=a.episode, num_frames=N, split=a.split, object_names=np.array(objs),
                  object_mesh_paths=np.array(meshes), object_pose=pose, object_valid=valid, object_conf=conf,
                  world_T_rect=W_rect, world_T_cam_a=W_a, cam_a_T_obj=camT, vo_valid=vo_ok, a_lag=lag,
                  time_base=a.time_base)
    if table_plane is not None:
        arrays.update(table_plane=table_plane, up=up)
    np.savez_compressed(f"{od}/perception.tmp.npz", **arrays)  # atomic write (tmp + rename)
    os.replace(f"{od}/perception.tmp.npz", f"{od}/perception.npz")
    summ = dict(split=a.split, episode=a.episode, frames=N, objects=objs, meshes=meshes, vo=vo,
                vo_valid_frac=float(vo_ok.mean()), fpose_dir=fd,
                valid_frac={o: float(valid[:, b].mean()) for b, o in enumerate(objs)},
                iou_median={o: float(np.nanmedian(conf[:, b])) for b, o in enumerate(objs)},
                table_plane=None if table_plane is None else table_plane.round(5).tolist(), table_inlier_frac=frac,
                objects_info=obj_info, options=dict(min_mask_px=a.min_mask_px, min_iou=a.min_iou,
                                                    clamp_static=a.clamp_static, smooth=a.smooth,
                                                    contact_px=a.contact_px, time_base=a.time_base),
                lag_values=sorted(set(lag.tolist())))
    json.dump(summ, open(f"{od}/perception.json.tmp", "w"), indent=1)
    os.replace(f"{od}/perception.json.tmp", f"{od}/perception.json")
    print(json.dumps(summ))
    if a.devscore:
        if a.split != "public":
            raise SystemExit("--devscore is for PUBLIC episodes only")
        import pandas as pd
        rows = []
        for t in range(N):
            for b in range(B):
                p = pose[t, b]
                rows.append([t, b, *p[:3], p[4], p[5], p[6], p[3]])
        df = pd.DataFrame(rows, columns=["frame_index", "object_slot", "pos_x", "pos_y", "pos_z",
                                         "quat_x", "quat_y", "quat_z", "quat_w"])
        os.makedirs(f"{a.devscore}/{e}", exist_ok=True)
        df.to_parquet(f"{a.devscore}/{e}.parquet")
        for o, m in zip(objs, meshes):
            shutil.copy(m, f"{a.devscore}/{e}/{o}{os.path.splitext(m)[1]}")


if __name__ == "__main__":
    main()
