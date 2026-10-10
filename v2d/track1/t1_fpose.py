#!/usr/bin/env python3
"""Object 6-DoF track for one Track 1 / FORM-HOI val episode: FoundationPose (NVlabs PyTorch backend, toolkit wrapper
v2d.foundation_pose.lib.foundation_pose_tracker) with OUR TRELLIS mesh, SAM3 masks and MoGe-2 metric depth.

Env: /mnt/secondary/v2d/envs/t3-fpose.  GPU, under the shared lock.
Inputs (all on the same window frames, every 2nd frame, half resolution):
  <masks>/masks.npz (t1_masks.py), <depth>/depth_s0.5.npy + depth_frames.npy (t1_moge.py depth, K = <K>),
  <mesh>/mesh_0.ply + trellis.json (t1_trellis.py; canonical unit-cube frame, vertex colours).
Steps
  1. depth stabilisation: per-frame scale r_t = median(ref / D_t) over static background pixels (valid, outside the
     dilated actor/object masks; ref = per-pixel temporal median), D_t <- r_t D_t  (removes MoGe's per-frame
     metric-scale jitter; the camera is static).
  2. reference frame = the TRELLIS view frame (unoccluded object).  Initial metric size s0 from the masked depth
     points (robust extent along the principal axis / mesh extent), then the toolkit's
     estimate_scale_grid_search (FP register at 0.5x..2x, IoU + depth score, 3 levels).
  3. --recolor: vertex colours re-sampled from the reference image at the registered pose (visible vertices; the
     rest from the nearest visible vertex), FP re-initialised and re-registered.
  4. track forward / backward from the reference frame (depth outside the dilated object mask zeroed when
     --mask-depth), re-register when the rendered-vs-SAM3 mask IoU < --rereg-iou and the mask is substantial.
Output <out>/fpose.npz: frames, cam_T_obj (N,4,4) for mesh_0 scaled by `scale` (metres, camera frame of the FULL
image K), scale, iou, rereg, depth_ratio, ref_index; fpose.json summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_items as I  # noqa: E402

FP_DIR = "/mnt/secondary/v2d/video_to_data/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose"


def unpack(bits, W):
    return np.unpackbits(bits, axis=-1, count=W).astype(bool)


def read_frames(video, idx, size):
    import cv2
    want = {int(i): k for k, i in enumerate(idx)}
    out = [None] * len(idx)
    cap = cv2.VideoCapture(str(video))
    n, last = 0, int(max(idx))
    while n <= last:
        ok, im = cap.read()
        if not ok:
            break
        if n in want:
            out[want[n]] = cv2.cvtColor(cv2.resize(im, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        n += 1
    assert all(o is not None for o in out)
    return out


def stabilize(D, hum, obj, W, sub=4):
    """D (N,h,w) float32 depth (0 invalid).  Returns ratios r (N,) so that r_t*D_t is consistent with the static bg."""
    import cv2
    N = len(D)
    Ds = D[:, ::sub, ::sub].astype(np.float32)
    bg = np.ones_like(Ds, bool)
    ker = np.ones((9, 9), np.uint8)
    for t in range(N):
        fg = unpack(hum[t], W) | unpack(obj[t], W)
        fg = cv2.dilate(fg.astype(np.uint8), ker) > 0
        bg[t] = ~fg[::sub, ::sub]
    bg &= Ds > 0
    X = np.where(bg, Ds, np.nan)
    ref = np.nanmedian(X, 0)
    r = np.ones(N)
    for t in range(N):
        v = bg[t] & np.isfinite(ref)
        if v.sum() > 100:
            r[t] = float(np.median(ref[v] / Ds[t][v]))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--mesh", required=True, help="dir with mesh_0.ply + trellis.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="/mnt/secondary/v2d/weights/foundationpose")
    ap.add_argument("--register-iter", type=int, default=10)
    ap.add_argument("--track-iter", type=int, default=5)
    ap.add_argument("--rereg-iou", type=float, default=0.5)
    ap.add_argument("--mask-depth", type=int, default=1)
    ap.add_argument("--mask-dilate", type=int, default=15)
    ap.add_argument("--recolor", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    import cv2
    import torch
    sys.path.append(FP_DIR)
    from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask, Transform3d
    from v2d.common.datatypes import Image as V2dImage
    from v2d.mesh.lib.mesh import Mesh
    from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker
    import trimesh
    from scipy.spatial import cKDTree

    t0 = time.time()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    it = I.item(a.split, a.episode)
    M = dict(np.load(Path(a.masks) / "masks.npz"))
    W, H = int(M["W"]), int(M["H"])
    frames = M["frames"]
    dfr = np.load(Path(a.depth) / "depth_frames.npy")
    assert np.array_equal(dfr, frames), "depth and mask frames differ"
    dinfo = json.load(open(Path(a.depth) / "depth_info.json"))
    Kfull = np.array(dinfo["K"], float)
    D = np.load(Path(a.depth) / "depth_s0.5.npy", mmap_mode="r")
    assert D.shape[1:] == (H, W), (D.shape, H, W)
    n = len(frames) if not a.max_frames else min(a.max_frames, len(frames))
    sx = W / float(M["full_wh"][0])
    K = Kfull.copy(); K[:2] *= sx
    intr = CameraIntrinsics(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), width=W, height=H)
    D = np.asarray(D[:n], np.float32)
    r = stabilize(D, M["human"][:n], M["object"][:n], W)
    D *= r[:, None, None].astype(np.float32)
    rgb = read_frames(it["video"], frames[:n], (W, H))
    om = [unpack(M["object"][t], W) for t in range(n)]
    areas = np.array([m.sum() for m in om], float)
    a95 = np.percentile(areas[areas > 0], 95) if (areas > 0).any() else 0
    tj = json.load(open(Path(a.mesh) / "trellis.json"))
    ref = int(tj["views"][0]["mask_index"])
    assert ref < n
    ker = np.ones((a.mask_dilate, a.mask_dilate), np.uint8)

    def obs(t):
        d = D[t]
        if a.mask_depth and om[t].any():
            md = cv2.dilate(om[t].astype(np.uint8), ker) > 0
            d = np.where(md, d, 0)
        return V2dImage(data=rgb[t]), DepthImage(depth=d.astype(np.float32)), Mask(mask=om[t].astype(np.float32))

    # ---- initial metric size from the masked depth points at the reference frame
    ys, xs = np.nonzero(om[ref] & (D[ref] > 0))
    z = D[ref][ys, xs]
    P = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], 1)
    Pc = P - np.median(P, 0)
    u = np.linalg.svd(Pc[np.random.default_rng(0).choice(len(Pc), min(len(Pc), 5000), replace=False)],
                      full_matrices=False)[2][0]
    proj = Pc @ u
    obj_len = float(np.percentile(proj, 97) - np.percentile(proj, 3))
    tm = trimesh.load(Path(a.mesh) / "mesh_0.ply", process=False)
    s0 = obj_len / float(tm.extents.max())
    cols = np.asarray(tm.visual.vertex_colors)[:, :3] if hasattr(tm.visual, "vertex_colors") else None

    def make_mesh(scale, colors):
        vc = None
        if colors is not None:
            vc = np.concatenate([colors[:, :3].astype(np.uint8), np.full((len(colors), 1), 255, np.uint8)], 1)
        return Mesh(vertices=np.asarray(tm.vertices) * scale, faces=np.asarray(tm.faces), vertex_colors=vc)
    tracker = FoundationPoseTracker(make_mesh(s0, cols), a.weights, backend="nvlabs_pytorch")
    rgb0, dep0, msk0 = obs(ref)
    with torch.no_grad():
        rel = tracker.estimate_scale_grid_search(rgb0, dep0, msk0, intr, lo=0.5, hi=2.0, n_samples=7, n_levels=3,
                                                 registration_iterations=5)
    scale = s0 * rel
    with torch.no_grad():
        pose0 = tracker.register(rgb0, dep0, msk0, intr, iteration=a.register_iter)
    iou0 = tracker._mask_iou(msk0, intr, pose0)
    info = {"s0": s0, "rel": float(rel), "scale": float(scale), "obj_len_m": obj_len, "iou_ref_gray": float(iou0)}
    if a.recolor:
        # vertex colours from the reference image (visible vertices), rest by nearest visible vertex
        from Utils import nvdiffrast_render
        V = np.asarray(tm.vertices) * scale
        Tm = pose0.to_matrix()
        Vc = V @ Tm[:3, :3].T + Tm[:3, 3]
        uv = Vc @ K.T
        uv = uv[:, :2] / uv[:, 2:3]
        with torch.no_grad():
            _, rd, _ = nvdiffrast_render(K, H, W, torch.as_tensor(Tm[None], device="cuda", dtype=torch.float),
                                         glctx=tracker._glctx, mesh_tensors=tracker._est.mesh_tensors, get_normal=False)
        rd = rd[0].cpu().numpy()
        ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        vis = np.zeros(len(V), bool)
        vis[inb] = (np.abs(rd[vi[inb], ui[inb]] - Vc[inb, 2]) < 0.01 + 0.02 * float(scale)) & om[ref][vi[inb], ui[inb]]
        if vis.sum() > 50:
            newc = np.zeros((len(V), 3), np.uint8)
            newc[vis] = rgb[ref][vi[vis], ui[vis]]
            _, nn = cKDTree(V[vis]).query(V[~vis])
            newc[~vis] = newc[vis][nn]
            tracker._original_mesh = make_mesh(1.0, newc)
            tracker.rescale_to(scale)
            tracker.reset_to_pose(pose0)
            with torch.no_grad():
                pose0 = tracker.track_one(rgb0, dep0, intr, iteration=a.register_iter)
            iou1 = tracker._mask_iou(msk0, intr, pose0)
            info.update({"recolor_visible": int(vis.sum()), "iou_ref_recolor": float(iou1)})
            np.save(Path(a.out) / "vertex_colors.npy", newc)
    Pm = np.full((n, 4, 4), np.nan)
    iou = np.full(n, np.nan)
    rereg = np.zeros(n, bool)
    Pm[ref] = pose0.to_matrix()
    iou[ref] = tracker._mask_iou(msk0, intr, pose0)

    def run(order):
        tracker.reset_to_pose(Transform3d.from_matrix(Pm[ref]))
        for t in order:
            rg, dp, mk = obs(t)
            with torch.no_grad():
                pose = tracker.track_one(rg, dp, intr, iteration=a.track_iter)
            if om[t].any():
                iou[t] = tracker._mask_iou(mk, intr, pose)
                if a.rereg_iou > 0 and iou[t] < a.rereg_iou and areas[t] > 0.3 * a95:
                    with torch.no_grad():
                        p2 = tracker.register(rg, dp, mk, intr, iteration=a.register_iter)
                    i2 = tracker._mask_iou(mk, intr, p2)
                    if i2 > iou[t]:
                        pose, iou[t], rereg[t] = p2, i2, True
                    else:
                        tracker.reset_to_pose(pose)
            Pm[t] = pose.to_matrix()
    run(range(ref + 1, n))
    run(range(ref - 1, -1, -1))
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    tmp = od / "fpose.tmp.npz"
    np.savez(tmp, frames=frames[:n], cam_T_obj=Pm, scale=scale, iou=iou, rereg=rereg, depth_ratio=r, ref_index=ref,
             K_full=Kfull)
    os.replace(tmp, od / "fpose.npz")
    info.update({"episode": a.episode, "split": a.split, "n": int(n), "ref_index": ref, "ref_frame": int(frames[ref]),
                 "iou_mean": float(np.nanmean(iou)), "iou_p10": float(np.nanpercentile(iou, 10)),
                 "n_rereg": int(rereg.sum()), "depth_ratio_p5_p95": [float(np.percentile(r, 5)), float(np.percentile(r, 95))],
                 "seconds": round(time.time() - t0, 1)})
    json.dump(info, open(od / "fpose.json", "w"), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
