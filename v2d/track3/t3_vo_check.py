"""Sanity checks of cuVSLAM VO trajectories (CPU): jumps, and consistency of the STATIC table plane in the VO world.

Per episode: frame-to-frame translation/rotation steps (max, p99), the number of 'jumps' (> --jump_cm or
> --jump_deg per frame), and the table plane (RANSAC on FoundationStereo depth, excluding object/hand masks when
present) estimated every --every frames in the rect frame and mapped into the VO world with world_T_rect: the
spread of its normal (deg) and of its offset (mm) must be small if VO is metric and drift-free (a wrong VO scale
shows up as an offset that changes with the head height).  Optional --static_obj NAME: world-frame centroid of that
object's masked depth (left_s1 masks) over frames where the hand does not touch it.

  python -I t3_vo_check.py --split public --episodes 12 13 [--vo_root /mnt/secondary/v2d/t3/vo] [--out J.json]
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t3_assemble import table_plane_rect  # noqa: E402
from t3_depth_warp import decode_inv_depth  # noqa: E402

R0 = "/mnt/secondary/v2d/t3"


def check(split, ep, vo_root, every, jump_cm, jump_deg, static_obj=None, a_max_depth=1.0):
    e = f"episode_{ep:06d}"
    meta = json.load(open(f"{R0}/frames/{split}/{e}/meta.json"))
    K = np.array(meta["images"]["left"]["K"])
    z = np.load(f"{vo_root}/{split}/{e}/vo_cuvslam.npz")
    T, valid = z["world_T_rect"], z["valid"]
    n = len(T)
    res = dict(split=split, episode=ep, frames=n, valid_frac=float(valid.mean()))
    idx = np.nonzero(valid)[0]
    d = np.linalg.norm(np.diff(T[idx, :3, 3], axis=0), axis=1) * 100
    r = np.degrees((Rotation.from_matrix(T[idx[:-1], :3, :3]).inv() * Rotation.from_matrix(T[idx[1:], :3, :3])).magnitude())
    res.update(step_cm_max=float(d.max()), step_cm_p99=float(np.percentile(d, 99)), step_deg_max=float(r.max()),
               n_jumps=int(((d > jump_cm) | (r > jump_deg)).sum()),
               path_cm=float(d.sum()), span_cm=float(np.ptp(T[idx, :3, 3], axis=0).max() * 100))
    md = f"{R0}/masks/{split}/{e}/left_s1"
    normals, offsets, heights, cents = [], [], [], []
    for f in range(0, n, every):
        if not valid[f]:
            continue
        p = f"{R0}/depth/{split}/{e}/depth_rect/{f:06d}.png"
        if not os.path.exists(p):
            continue
        dep = decode_inv_depth(cv2.imread(p, cv2.IMREAD_UNCHANGED))
        excl = np.zeros(dep.shape, bool)
        for mp in glob.glob(f"{md}/*/{f:06d}.png"):
            m = cv2.imread(mp, 0)
            excl |= cv2.dilate(cv2.resize(m, dep.shape[::-1], interpolation=cv2.INTER_NEAREST), np.ones((15, 15))) > 0
        pl, frac = table_plane_rect(dep, K, excl, n_iter=200, max_depth=a_max_depth)
        if pl is None or frac < 0.2 or (a_max_depth <= 1.0 and not (0.3 < pl[3] < 0.8)):
            continue  # with the table setting, drop fits that latched onto the floor (camera ~1.1 m above it)
        nw = T[f, :3, :3] @ pl[:3]
        pw = T[f, :3, :3] @ (-pl[3] * pl[:3]) + T[f, :3, 3]
        normals.append(nw)
        offsets.append(-nw @ pw)
        heights.append(pl[3])  # camera height above the table plane (m): n.0 + d with n towards the camera
        if static_obj and os.path.exists(f"{md}/{static_obj}/{f:06d}.png"):
            m = cv2.imread(f"{md}/{static_obj}/{f:06d}.png", 0) > 0
            h = cv2.imread(f"{md}/hand/{f:06d}.png", 0) if os.path.exists(f"{md}/hand/{f:06d}.png") else None
            touch = h is not None and (cv2.dilate((h > 0).astype(np.uint8), np.ones((41, 41))) > 0)[m].any()
            m = cv2.erode(m.astype(np.uint8), np.ones((5, 5))) > 0
            v, u = np.nonzero(m & (dep > 0.2) & (dep < 2))
            if len(u) > 200 and not touch:
                zz = dep[v, u]
                X = np.stack([(u - K[0, 2]) / K[0, 0] * zz, (v - K[1, 2]) / K[1, 1] * zz, zz], 1)
                cents.append(T[f, :3, :3] @ np.median(X, 0) + T[f, :3, 3])
    if normals:
        N = np.array(normals)
        nm = N.mean(0) / np.linalg.norm(N.mean(0))
        res.update(n_plane=len(N), plane_normal_spread_deg=float(np.degrees(np.arccos(np.clip(N @ nm, -1, 1))).max()),
                   plane_offset_spread_mm=float(np.ptp(offsets) * 1000), plane_offset_std_mm=float(np.std(offsets) * 1000),
                   cam_height_range_cm=[float(min(heights) * 100), float(max(heights) * 100)],
                   offset_vs_height_slope=float(np.polyfit(heights, offsets, 1)[0]) if np.ptp(heights) > 0.01 else None)
    if cents:
        C = np.array(cents)
        res.update(static_obj=static_obj, static_n=len(C), static_spread_mm=float(np.linalg.norm(C - np.median(C, 0), axis=1).max() * 1000),
                   static_std_mm=float(np.linalg.norm(C - C.mean(0), axis=1).std() * 1000))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="public")
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--vo_root", default=f"{R0}/vo")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--jump_cm", type=float, default=3.0)
    ap.add_argument("--jump_deg", type=float, default=5.0)
    ap.add_argument("--static_obj", default=None)
    ap.add_argument("--max_depth", type=float, default=1.0, help="plane points nearer than this (1.0 = table, 3 = floor)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    eps = a.episodes or sorted(int(p[-6:]) for p in glob.glob(f"{a.vo_root}/{a.split}/episode_*"))
    out = []
    for ep in eps:
        r = check(a.split, ep, a.vo_root, a.every, a.jump_cm, a.jump_deg, a.static_obj, a.max_depth)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        out.append(r)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
