"""Stereo visual odometry for the head-mounted ego rig with cuVSLAM 17 (pip wheel, no Docker).

Input: a preprocessed episode dir (t3_preprocess.py): left/ right/ rectified JPEGs + meta.json (K_rect, baseline).
Rig frame = rectified-LEFT camera frame (OpenCV x right, y down, z forward); right camera at +baseline along x.
Output (OUT/<split>/episode_XXXXXX/):
  vo_cuvslam.npz   world_T_rect[T,4,4] (world = rect-left frame at the first tracked frame), valid[T],
                   cov[T,6,6]; frames where tracking failed are NaN.
  vo_cuvslam.json  summary (n valid, path length, rotation span, timing, config).
Optional --mask_dir DIR[,DIR...]: per-frame PNGs (000000.png) in the LEFT rectified grid (any scale; resized), nonzero =
dynamic (hands/objects); OR-ed, dilated 5 px and passed to cuVSLAM for both cameras (right reuses the left mask).

  source envs/t3-geom.env
  python -I t3_vo_cuvslam.py --ep_dir /mnt/secondary/v2d/t3/frames/public/episode_000012 --out /mnt/secondary/v2d/t3/vo
  (add --cpu to run cuVSLAM without the GPU: use_gpu=False)
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("--ep_dir", required=True)
ap.add_argument("--out", default="/mnt/secondary/v2d/t3/vo")
ap.add_argument("--cpu", action="store_true", help="cuVSLAM use_gpu=False")
ap.add_argument("--mask_dir", default=None)
ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--no_rectified_flag", action="store_true")
a = ap.parse_args()

import cuvslam as vslam  # noqa: E402

meta = json.load(open(f"{a.ep_dir}/meta.json"))
st = meta["stereo"]
w, h = meta["images"]["left"]["size_wh"]
cam_l = vslam.Camera(size=[w, h], principal=[st["cx"], st["cy"]], focal=[st["fx"], st["fy"]],
                     rig_from_camera=vslam.Pose(rotation=[0, 0, 0, 1], translation=[0, 0, 0]))
cam_r = vslam.Camera(size=[w, h], principal=[st["cx"], st["cy"]], focal=[st["fx"], st["fy"]],
                     rig_from_camera=vslam.Pose(rotation=[0, 0, 0, 1], translation=[st["baseline_m"], 0, 0]))
rig = vslam.Rig([cam_l, cam_r])
cfg = vslam.Tracker.OdometryConfig(odometry_mode=vslam.Tracker.OdometryMode.Multicamera,
                                   use_gpu=not a.cpu, async_sba=False, use_motion_model=True,
                                   use_denoising=False, rectified_stereo_camera=not a.no_rectified_flag,
                                   enable_observations_export=False, enable_landmarks_export=False)
tracker = vslam.Tracker(rig, cfg)
n = meta["n_frames"] if not a.max_frames else min(a.max_frames, meta["n_frames"])
ts = meta["timestamps"]
T = np.full((n, 4, 4), np.nan)
cov = np.full((n, 6, 6), np.nan)
valid = np.zeros(n, bool)
t0 = time.time()
for i in range(n):
    L = cv2.imread(f"{a.ep_dir}/left/{i:06d}.jpg", cv2.IMREAD_GRAYSCALE)
    R = cv2.imread(f"{a.ep_dir}/right/{i:06d}.jpg", cv2.IMREAD_GRAYSCALE)
    masks = None
    if a.mask_dir:  # comma-separated dirs (e.g. every object + hand) are OR-ed; dilated 5 px for safety
        m = np.zeros_like(L)
        for md in a.mask_dir.split(","):
            x = cv2.imread(f"{md}/{i:06d}.png", cv2.IMREAD_GRAYSCALE)
            if x is not None:
                m |= cv2.resize((x > 0).astype(np.uint8), (L.shape[1], L.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = cv2.dilate(m, np.ones((11, 11), np.uint8))
        masks = [m, m]
    est, _ = tracker.track(int(round(ts[i] * 1e9)), [L, R], masks=masks)
    p = est.world_from_rig
    if p is not None:
        q, t = np.array(p.pose.rotation), np.array(p.pose.translation)
        M = np.eye(4)
        M[:3, :3] = Rotation.from_quat(q).as_matrix()
        M[:3, 3] = t
        T[i] = M
        cov[i] = np.array(p.covariance_xyz_rpy).reshape(6, 6)
        valid[i] = True
dt = time.time() - t0
od = f"{a.out}/{meta['split']}/episode_{meta['episode']:06d}"
os.makedirs(od, exist_ok=True)
np.savez_compressed(f"{od}/vo_cuvslam.npz", world_T_rect=T, valid=valid, cov=cov)
v = np.nonzero(valid)[0]
summ = dict(split=meta["split"], episode=meta["episode"], frames=n, n_valid=int(valid.sum()),
            first_invalid=[int(x) for x in np.nonzero(~valid)[0][:20]], seconds=round(dt, 1),
            fps=round(n / dt, 1), use_gpu=not a.cpu, masks=bool(a.mask_dir))
if len(v) > 1:
    P = T[v, :3, 3]
    summ["path_length_m"] = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    summ["max_displacement_m"] = float(np.linalg.norm(P - P[0], axis=1).max())
    rots = Rotation.from_matrix(T[v, :3, :3])
    summ["max_rotation_deg"] = float(np.degrees((rots[0].inv() * rots).magnitude()).max())
json.dump(summ, open(f"{od}/vo_cuvslam.json", "w"), indent=1)
print(json.dumps(summ))
