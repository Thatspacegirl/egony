"""FoundationPose (NVLabs PyTorch backend) object tracking on undistorted cam_a for one episode/object -- GPU.

Our own driver around v2d.foundation_pose.lib.foundation_pose_tracker.FoundationPoseTracker (same model calls as the
module CLI run_video_to_poses) but
  * reads the preprocessed JPEG frames directly (no VideoCapture seeking), resized to the depth/mask grid;
  * registers at a reference frame (default: auto = first frame whose mask area >= 50 % of the clip's 95th pct),
    tracks forward and backward; optional re-registration when the rendered-vs-observed mask IoU drops;
  * optional --mask_depth (depth outside the dilated object mask is zeroed, so hands/table cannot pull the pose);
  * --stage1_json + --vo: initial pose from the stage-1 mesh frame (object static in the VO world during the window)
    composed with VO at the matching stereo instant (t3_sync.py lag) instead of FoundationPose's global registration;
  * stores everything in one NPZ: cam_T_obj[T,4,4] (cam_a frame, mesh coordinates), iou[T], mask_px[T],
    rereg[T] (bool), ref_frame, plus timing.

  PYTHONPATH=$FP python -I t3_fpose_track.py --ep_dir FRAMES/public/episode_000012 --obj blue_cup \
     --depth_dir DEPTH/public/episode_000012/depth_cam_a_s0.5 --mask_dir MASKS/public/episode_000012/cam_a_s0.5/blue_cup \
     --mesh MESH.ply --out OUT/blue_cup.npz [--mask_depth] [--rereg_iou 0.5]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

FP_DIR = "/mnt/secondary/v2d/video_to_data/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose"
if FP_DIR not in sys.path:
    sys.path.append(FP_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t3_depth_warp import decode_inv_depth  # noqa: E402


def auto_ref(areas):
    if not (areas > 0).any():
        return 0
    thr = 0.5 * np.percentile(areas[areas > 0], 95)
    return int(np.argmax(areas >= thr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_dir", required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="/mnt/secondary/v2d/weights/foundationpose")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--ref", default="auto", help="reference frame index or 'auto'")
    ap.add_argument("--register_iter", type=int, default=10)
    ap.add_argument("--track_iter", type=int, default=5)
    ap.add_argument("--mask_depth", action="store_true")
    ap.add_argument("--mask_dilate", type=int, default=7)
    ap.add_argument("--rereg_iou", type=float, default=0.0, help="re-register when IoU < this (0 = never)")
    ap.add_argument("--init_pose", default=None,
                    help="NPZ with cam_T_obj_init[4,4] for the reference frame (skips global registration)")
    ap.add_argument("--stage1_json", default=None,
                    help="stage-1 mesh JSON (t3_tsdf_fuse.py: frame_T_obj in the VO world + static-window frames); "
                         "with --vo the reference pose is world->cam_a composed, no global registration")
    ap.add_argument("--vo", default=None, help="vo_cuvslam.npz (world_T_rect) for --stage1_json")
    ap.add_argument("--anchor0", action="store_true",
                    help="second track forward from frame 0 (init = stage-1 pose under the static assumption or a "
                         "global registration, whichever renders closer to the mask); per frame the track with the "
                         "higher smoothed IoU wins (2-state Viterbi, switch cost --switch_cost)")
    ap.add_argument("--switch_cost", type=float, default=0.5)
    ap.add_argument("--max_frames", type=int, default=0)
    a = ap.parse_args()

    import torch
    from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask, Transform3d
    from v2d.common.datatypes import Image as V2dImage
    from v2d.mesh.lib.mesh import Mesh
    from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker

    meta = json.load(open(f"{a.ep_dir}/meta.json"))
    n = meta["n_frames"] if not a.max_frames else min(a.max_frames, meta["n_frames"])
    Ka = np.array(meta["images"]["cam_a"]["K"], float)
    W0, H0 = meta["images"]["cam_a"]["size_wh"]
    W, H = int(round(W0 * a.scale)), int(round(H0 * a.scale))
    K = Ka.copy()
    K[:2] *= a.scale
    intr = CameraIntrinsics(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), width=W, height=H)

    masks = []
    for f in range(n):
        m = cv2.imread(f"{a.mask_dir}/{f:06d}.png", 0)
        m = np.zeros((H, W), bool) if m is None else cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0
        masks.append(m)
    areas = np.array([m.sum() for m in masks])
    init_from_s1 = None
    if a.stage1_json:
        s1 = json.load(open(a.stage1_json))
        Wr = np.load(a.vo)["world_T_rect"]
        T_a_rect = np.array(meta["transforms"]["T_a_rect"])
        from t3_sync import load_lag, stereo_index_for_cam_a
        lag = load_lag(meta["split"], f"episode_{meta['episode']:06d}", meta["n_frames"])
        src = stereo_index_for_cam_a(lag) if lag.any() else np.arange(meta["n_frames"])
        Wa = Wr[src]  # world_T_rect at the instant of cam_a frame j
        # stage-1 window frames are STEREO frames; the matching cam_a frames are f + lag[f]
        win = sorted({int(f + lag[f]) for f in s1["frames"] if f < len(lag)})
        win = [j for j in win if j < n and np.isfinite(Wa[j]).all() and areas[j] > 0]
        if a.ref == "auto" and win:
            ref = int(win[int(np.argmax(areas[win]))])
        else:
            ref = auto_ref(areas) if a.ref == "auto" else int(a.ref)
        if np.isfinite(Wa[ref]).all():
            init_from_s1 = T_a_rect @ np.linalg.inv(Wa[ref]) @ np.array(s1["frame_T_obj"])  # cam_a_T_obj
    else:
        ref = auto_ref(areas) if a.ref == "auto" else int(a.ref)

    def load(f):
        rgb = cv2.cvtColor(cv2.resize(cv2.imread(f"{a.ep_dir}/cam_a/{f:06d}.jpg"), (W, H), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        d = decode_inv_depth(cv2.imread(f"{a.depth_dir}/{f:06d}.png", cv2.IMREAD_UNCHANGED))
        if a.mask_depth and masks[f].any():
            md = cv2.dilate(masks[f].astype(np.uint8), np.ones((a.mask_dilate, a.mask_dilate), np.uint8)) > 0
            d = np.where(md, d, 0)
        return V2dImage(data=rgb), DepthImage(depth=d.astype(np.float32)), Mask(mask=masks[f].astype(np.float32))

    mesh = Mesh.load(a.mesh, force_mesh=a.mesh.lower().endswith(".glb"))
    tracker = FoundationPoseTracker(mesh, a.weights, backend="nvlabs_pytorch")
    t0 = time.time()
    P = np.full((n, 4, 4), np.nan)
    iou = np.full(n, np.nan)
    rereg = np.zeros(n, bool)
    rgb, dep, msk = load(ref)
    if a.init_pose or init_from_s1 is not None:
        init = init_from_s1 if init_from_s1 is not None else np.load(a.init_pose)["cam_T_obj_init"]
        init_T = Transform3d.from_matrix(init)
        tracker.reset_to_pose(init_T)
        # refine the given pose on the reference frame with a few tracking iterations
        pose0 = tracker.track_one(rgb, dep, intr, iteration=a.register_iter)
    else:
        pose0 = tracker.register(rgb, dep, msk, intr, iteration=a.register_iter)
    P[ref] = pose0.to_matrix()
    iou[ref] = tracker._mask_iou(msk, intr, pose0) if masks[ref].any() else np.nan

    def run(order):
        tracker.reset_to_pose(Transform3d.from_matrix(P[ref]))
        for f in order:
            rgb, dep, msk = load(f)
            pose = tracker.track_one(rgb, dep, intr, iteration=a.track_iter)
            if masks[f].any():
                iou[f] = tracker._mask_iou(msk, intr, pose)
                if a.rereg_iou > 0 and iou[f] < a.rereg_iou and areas[f] > 0.3 * np.percentile(areas[areas > 0], 95):
                    p2 = tracker.register(rgb, dep, msk, intr, iteration=a.register_iter)
                    i2 = tracker._mask_iou(msk, intr, p2)
                    if i2 > iou[f]:
                        pose, iou[f], rereg[f] = p2, i2, True
                    else:  # keep the tracked pose; restore the tracker state to it
                        tracker.reset_to_pose(pose)
            P[f] = pose.to_matrix()
    with torch.no_grad():
        run(range(ref + 1, n))
        run(range(ref - 1, -1, -1))
    extra = {}
    if a.anchor0 and ref > 0:
        PA, iouA = P.copy(), iou.copy()
        cands = []
        rgb, dep, msk = load(0)
        with torch.no_grad():
            if init_from_s1 is not None and np.isfinite(Wa[0]).all():
                # static assumption: the object did not move between frame 0 and the stage-1 window
                T0 = T_a_rect @ np.linalg.inv(Wa[0]) @ np.array(s1["frame_T_obj"])
                tracker.reset_to_pose(Transform3d.from_matrix(T0))
                p = tracker.track_one(rgb, dep, intr, iteration=a.register_iter)
                cands.append(("static", p, tracker._mask_iou(msk, intr, p) if masks[0].any() else 0.0))
            if masks[0].any():
                p = tracker.register(rgb, dep, msk, intr, iteration=a.register_iter)
                cands.append(("register", p, tracker._mask_iou(msk, intr, p)))
        if cands:
            name0, p0, i0 = max(cands, key=lambda c: c[2])
            P[:] = np.nan
            iou[:] = np.nan
            P[0], iou[0] = p0.to_matrix(), i0
            ref_saved = ref
            ref = 0
            with torch.no_grad():
                run(range(1, n))
            ref = ref_saved
            PB, iouB = P.copy(), iou.copy()
            # 2-state Viterbi on smoothed IoU (NaN IoU = no mask -> no evidence)
            k = np.ones(9) / 9
            def sm(x):
                y = np.nan_to_num(x, nan=0.0)
                w = np.isfinite(x).astype(float)
                return np.convolve(y, k, "same") / np.maximum(np.convolve(w, k, "same"), 1e-6)
            S = np.stack([sm(iouA), sm(iouB)], 1)
            cost = np.zeros(2)
            back = np.zeros((n, 2), int)
            for f in range(n):
                if f > 0:
                    stay = cost
                    sw = cost[::-1] + a.switch_cost
                    back[f] = np.where(stay <= sw, [0, 1], [1, 0])
                    cost = np.minimum(stay, sw)
                cost = cost - S[f]
            st = np.zeros(n, int)
            st[-1] = int(np.argmin(cost))
            for f in range(n - 1, 0, -1):
                st[f - 1] = back[f, st[f]]
            P = np.where(st[:, None, None] == 0, PA, PB)
            iou = np.where(st == 0, iouA, iouB)
            extra = dict(cam_T_obj_ref=PA, iou_ref=iouA, cam_T_obj_f0=PB, iou_f0=iouB, choice_f0=st,
                         anchor0_init=name0, anchor0_cands=np.array([[c[2] for c in cands]]))
            print(f"anchor0: init {name0} iou {i0:.3f} ({[(c[0], round(float(c[2]), 3)) for c in cands]}); "
                  f"frames from the frame-0 track: {int(st.sum())}/{n}", flush=True)
    dt = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, cam_T_obj=P, iou=iou, mask_px=areas, rereg=rereg, ref_frame=ref, K=K,
                        init=np.full((4, 4), np.nan) if init_from_s1 is None else init_from_s1,
                        size_wh=np.array([W, H]), mesh=a.mesh, scale=a.scale, **extra)
    print(json.dumps(dict(obj=a.obj, frames=n, ref=ref, seconds=round(dt, 1), fps=round(n / dt, 2),
                          iou_median=float(np.nanmedian(iou)), iou_p10=float(np.nanpercentile(iou, 10)),
                          n_rereg=int(rereg.sum()), out=a.out)), flush=True)


if __name__ == "__main__":
    main()
