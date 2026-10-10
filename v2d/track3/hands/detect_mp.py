"""Stage 1: 21-keypoint 2D hand detection on the undistorted ego_cam_a colour stream with MediaPipe HandLandmarker
(Tasks API, Apache-2.0 code + model; no MANO). CPU only.

Two VIDEO-mode passes per episode (forward in time and backward in time, each a fresh tracker) because the
tracker needs a few frames to lock on after an occlusion; the backward pass recovers the frames just before a
re-acquisition. Raw per-pass results are stored; merging, handedness and 3D lifting happen in lift_hands.py.

Output OUT/<split>/episode_XXXXXX/det_mp_<view>.npz
  lm2d    (P,T,K,21,3) f4  pixel x, pixel y (in the <view> image), MediaPipe relative z (x-normalised)  NaN = none
  world   (P,T,K,21,3) f4  MediaPipe hand_world_landmarks (metres, origin ~ hand centre, camera-aligned axes)
  label   (P,T,K)      i1  MediaPipe handedness 0=Left 1=Right -1=none
  score   (P,T,K)      f4  handedness score
  P = 2 passes (0 forward, 1 backward), K = num_hands slots (order is MediaPipe's, not an identity)

  /mnt/secondary/v2d/envs/t3-hands/bin/python -I detect_mp.py --episodes all --workers 3
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

FRAMES = "/mnt/secondary/v2d/t3/frames"
OUT = "/mnt/secondary/v2d/t3/hands"
MODEL = "/mnt/secondary/v2d/weights/mediapipe/hand_landmarker.task"


def episode_dirs(spec):
    dirs = sorted(glob.glob(f"{FRAMES}/*/episode_*"))
    if spec in ("all", None):
        return dirs
    want = {int(x) for x in spec.split(",")}
    return [d for d in dirs if int(d.rsplit("_", 1)[1]) in want]


def make_landmarker(num_hands, conf):
    from mediapipe.tasks.python import BaseOptions, vision
    opt = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands, min_hand_detection_confidence=conf, min_hand_presence_confidence=conf,
        min_tracking_confidence=conf)
    return vision.HandLandmarker.create_from_options(opt)


def run_episode(args):
    ep_dir, view, num_hands, conf, overwrite = args
    import cv2
    import mediapipe as mp
    cv2.setNumThreads(2)
    meta = json.load(open(f"{ep_dir}/meta.json"))
    split, ep = meta["split"], meta["episode"]
    od = f"{OUT}/{split}/episode_{ep:06d}"
    out = f"{od}/det_mp_{view}.npz"
    if os.path.exists(out) and not overwrite:
        return f"skip {out}"
    os.makedirs(od, exist_ok=True)
    T = meta["n_frames"]
    K = num_hands
    lm2d = np.full((2, T, K, 21, 3), np.nan, np.float32)
    world = np.full((2, T, K, 21, 3), np.nan, np.float32)
    label = np.full((2, T, K), -1, np.int8)
    score = np.zeros((2, T, K), np.float32)
    t0 = time.time()
    for p, order in enumerate((range(T), range(T - 1, -1, -1))):
        lm = make_landmarker(num_hands, conf)
        for i, f in enumerate(order):
            rgb = np.ascontiguousarray(cv2.cvtColor(cv2.imread(f"{ep_dir}/{view}/{f:06d}.jpg"), cv2.COLOR_BGR2RGB))
            H, W = rgb.shape[:2]
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 50))
            for k, hl in enumerate(res.hand_landmarks[:K]):
                lm2d[p, f, k] = [[q.x * W, q.y * H, q.z * W] for q in hl]
                world[p, f, k] = [[q.x, q.y, q.z] for q in res.hand_world_landmarks[k]]
                h = res.handedness[k][0]
                label[p, f, k] = 1 if h.category_name == "Right" else 0
                score[p, f, k] = h.score
        lm.close()
    np.savez_compressed(out, lm2d=lm2d, world=world, label=label, score=score,
                        view=np.array(view), num_hands=np.array(K), conf=np.array(conf),
                        model=np.array(MODEL), image_wh=np.array([W, H]))
    n = (~np.isnan(lm2d[..., 0, 0])).sum(-1)  # (2,T) hands per frame
    return (f"{split}/episode_{ep:06d} T={T} {time.time() - t0:.0f}s fwd 0/1/2={np.bincount(n[0], minlength=3)[:3]} "
            f"bwd 0/1/2={np.bincount(n[1], minlength=3)[:3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="all", help="'all' or comma list of episode indices")
    ap.add_argument("--view", default="cam_a")
    ap.add_argument("--num_hands", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    jobs = [(d, a.view, a.num_hands, a.conf, a.overwrite) for d in episode_dirs(a.episodes)]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for msg in pool.imap_unordered(run_episode, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())
