"""Independent stereo cross-check of hands_rect.npz (method A = cam_a 2D + dense stereo depth + temporal opt.).

Method B: an independent top-down 2D detector on the rectified LEFT and RIGHT gray views, boxed by A's projection
(--det rtmpose [default]: RTMPose-m hand5, box +15 %, joints with score < 0.3 dropped; --det mediapipe: MediaPipe
on an upscaled crop, IMAGE mode, recall on these small gray hands is only ~15-20 %), then
per-joint triangulation on the rectified pair (same row; Z = fx * baseline / (uL - uR)). The stereo views are not
used for 2D by method A, so:
  * 2D error of A's projection vs the B detections in rect-left / rect-right (px), and the disparity error
    -> depth error (Z^2/(f B) * d_err);
  * per-joint 3D distance A vs B (frames where both exist), wrist depth difference (bias), and the raw per-frame
    bone-length coefficient of variation of B (noise proxy; A raw is reported by lift stats).

Output HANDS/<split>/episode_X/stereo_check_<det>.json (+ stereo_tri_<det>.npz with B's raw triangulated joints).
  python -I stereo_check.py --episodes 0,12,41 --stride 3
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

MODEL = "/mnt/secondary/v2d/weights/mediapipe/hand_landmarker.task"


def detect_rtm(rtm, img_gray, uv_pred, min_score=0.3):
    """RTMPose-hand on the box of uv_pred (+15 %, padding 1.25 inside); joints with score < min_score -> NaN."""
    lo, hi = np.nanmin(uv_pred, 0), np.nanmax(uv_pred, 0)
    m = 0.15 * (hi - lo) + 8
    k, sc = rtm(img_gray, (lo[0] - m[0], lo[1] - m[1], hi[0] + m[0], hi[1] + m[1]))
    k[sc < min_score] = np.nan
    return k, sc


def detect_crop(lm, mp, img_gray, uv_pred, target=320):
    """Run MediaPipe on a crop around uv_pred (21,2); return the 21x2 detection (full-image px) closest to uv_pred."""
    import cv2
    H, W = img_gray.shape
    lo, hi = np.nanmin(uv_pred, 0), np.nanmax(uv_pred, 0)
    c = 0.5 * (lo + hi)
    side = 1.5 * max(hi - lo) + 40
    x0, y0 = int(c[0] - side / 2), int(c[1] - side / 2)
    x1, y1 = int(c[0] + side / 2), int(c[1] + side / 2)
    pad = cv2.copyMakeBorder(img_gray, max(0, -y0), max(0, y1 - H), max(0, -x0), max(0, x1 - W), cv2.BORDER_CONSTANT, 0)
    crop = pad[y0 + max(0, -y0):y1 + max(0, -y0), x0 + max(0, -x0):x1 + max(0, -x0)]
    if crop.size == 0:
        return None
    s = target / crop.shape[0]
    crop = cv2.resize(crop, (int(round(crop.shape[1] * s)), target), interpolation=cv2.INTER_CUBIC)
    rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB))
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    best, bd = None, 1e9
    for h in res.hand_landmarks:
        u = np.array([[q.x * crop.shape[1], q.y * crop.shape[0]] for q in h]) / s + [x0, y0]
        d = np.linalg.norm(u - uv_pred, axis=1).mean()
        if d < bd:
            best, bd = u, d
    if best is None or bd > 0.5 * side:
        return None
    return best


def run(args):
    ep_dir, stride, det = args
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    cv2.setNumThreads(1)
    meta, Ka, Kr, T_a_rect = hl.load_meta(ep_dir)
    od = hl.out_dir(meta)
    z = np.load(f"{od}/hands_rect.npz")
    Bm = meta["stereo"]["baseline_m"]
    f = Kr[0, 0]
    T = int(z["num_frames"])
    from rtmpose_hand import RTMPoseHand
    rtm = RTMPoseHand(threads=2)
    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))
    tri = {s: np.full((T, 21, 3), np.nan, np.float32) for s in hl.SIDES}
    det2d = {f"{s}_{v}": np.full((T, 21, 2), np.nan, np.float32) for s in hl.SIDES for v in ("rectL", "rectR")}
    errL, errR, dZ, d3, dwrist, n_try, n_both = [], [], [], [], [], 0, 0
    t0 = time.time()
    for t in range(0, T, stride):
        L = R = None
        for side in hl.SIDES:
            if not z[f"{side}_observed"][t]:
                continue
            X = z[side][t].astype(float)
            uvL = hl.project(X, Kr)
            uvR = hl.project(X - [Bm, 0, 0], Kr)
            if L is None:
                L = cv2.imread(f"{ep_dir}/left/{t:06d}.jpg", 0)
                R = cv2.imread(f"{ep_dir}/right/{t:06d}.jpg", 0)
            n_try += 1
            if det == "rtmpose":
                dl, _ = detect_rtm(rtm, L, uvL)
                dr, _ = detect_rtm(rtm, R, uvR)
            else:
                dl = detect_crop(lm, mp, L, uvL)
                dr = detect_crop(lm, mp, R, uvR)
            inL = (uvL[:, 1] < L.shape[0] - 3) & (uvL[:, 1] > 3) & (uvL[:, 0] > 3) & (uvL[:, 0] < L.shape[1] - 3)
            inR = (uvR[:, 1] < R.shape[0] - 3) & (uvR[:, 1] > 3) & (uvR[:, 0] > 3) & (uvR[:, 0] < R.shape[1] - 3)
            if dl is not None:
                det2d[f"{side}_rectL"][t] = dl
            if dr is not None:
                det2d[f"{side}_rectR"][t] = dr
            if dl is not None:
                e = np.linalg.norm(dl - uvL, axis=1)[inL]
                errL += e[np.isfinite(e)].tolist()
            if dr is not None:
                e = np.linalg.norm(dr - uvR, axis=1)[inR]
                errR += e[np.isfinite(e)].tolist()
            if dl is not None and dr is not None:
                ok = inL & inR & np.isfinite(dl).all(1) & np.isfinite(dr).all(1)
                disp = dl[:, 0] - dr[:, 0]
                ok &= np.nan_to_num(disp) > 5
                if ok.sum() < 10:
                    continue
                n_both += 1
                Zb = f * Bm / np.where(ok, disp, np.nan)
                v = 0.5 * (dl[:, 1] + dr[:, 1])
                Xb = np.stack([(dl[:, 0] - Kr[0, 2]) / f * Zb, (v - Kr[1, 2]) / Kr[1, 1] * Zb, Zb], 1)
                tri[side][t] = Xb
                # disparity error of A vs B, in depth units
                dispA = uvL[:, 0] - uvR[:, 0]
                dZ += ((f * Bm / disp - f * Bm / dispA)[ok]).tolist()
                d3 += np.linalg.norm(Xb - X, axis=1)[ok].tolist()
                if ok[0]:
                    dwrist.append(float(Zb[0] - X[0, 2]))
    lm.close()

    def bl_cv(J):
        ok = np.isfinite(J).all((1, 2))
        if ok.sum() < 5:
            return None
        bl = np.linalg.norm(J[ok][:, [b for _, b in hl.BONES]] - J[ok][:, [a for a, _ in hl.BONES]], axis=-1)
        return float(np.nanmedian(np.nanstd(bl, 0) / np.nanmean(bl, 0)))

    def pct(x, q):
        return None if not len(x) else float(np.percentile(x, q))

    rep = {"episode": meta["episode"], "split": meta["split"], "stride": stride, "detector": det, "hands_tried": n_try,
           "hands_triangulated": n_both, "seconds": round(time.time() - t0, 1),
           "A_vs_Bdet_rectL_px_median": pct(errL, 50), "A_vs_Bdet_rectL_px_p90": pct(errL, 90),
           "A_vs_Bdet_rectR_px_median": pct(errR, 50), "A_vs_Bdet_rectR_px_p90": pct(errR, 90),
           "depth_diff_B_minus_A_median_m": pct(dZ, 50), "depth_absdiff_median_m": pct(np.abs(dZ), 50),
           "joint_3d_dist_A_B_median_m": pct(d3, 50), "joint_3d_dist_A_B_p90_m": pct(d3, 90),
           "wrist_depth_B_minus_A_median_m": pct(dwrist, 50),
           "B_raw_bone_cv": {s: bl_cv(tri[s]) for s in hl.SIDES}}
    np.savez_compressed(f"{od}/stereo_tri_{det}.npz", **tri, **det2d, baseline=np.array(Bm), K_rect=Kr)
    json.dump(rep, open(f"{od}/stereo_check_{det}.json", "w"), indent=1)
    return json.dumps(rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--splits", default="public")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--det", default="rtmpose", choices=("rtmpose", "mediapipe"))
    a = ap.parse_args()
    jobs = [(d, a.stride, a.det) for d in hl.episode_dirs(a.episodes, a.splits.split(","))]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for msg in pool.imap_unordered(run, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())
