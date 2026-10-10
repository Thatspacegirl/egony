"""Shared helpers for the MANO-free hand pipeline (Track 3).

Keypoint order (21): MediaPipe HandLandmarker == OpenPose == the 21-joint MANO order used by HaMeR and by
sharpa_task/hands.py (MANO_HAND_LINKS, TIP_JOINTS=[4,8,12,16,20], palm [0,5,9,13,17]):
  0 wrist | 1-4 thumb CMC,MCP,IP,TIP | 5-8 index MCP,PIP,DIP,TIP | 9-12 middle | 13-16 ring | 17-20 pinky
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

TRACK3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = "/mnt/secondary/v2d/t3/frames"
DEPTH = "/mnt/secondary/v2d/t3/depth"
HANDS = "/mnt/secondary/v2d/t3/hands"
SIDES = ("left", "right")
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
PALM_EDGES = [(5, 9), (9, 13), (13, 17), (5, 17), (1, 5)]  # near-rigid palm distances (extra length terms)
LEN_EDGES = BONES + PALM_EDGES
TIPS = [4, 8, 12, 16, 20]
PALM = [0, 5, 9, 13, 17]
# joint centre lies behind the visible (camera-facing) skin surface: ~half the local thickness
SKIN_OFFSET = np.array([0.012, 0.010, 0.009, 0.008, 0.007, 0.010, 0.008, 0.007, 0.006, 0.010, 0.008, 0.007, 0.006,
                        0.010, 0.008, 0.007, 0.006, 0.009, 0.007, 0.006, 0.005])
DEPTH_SCALE = 0.5  # depth sampled on the cam_a grid at this scale (1014x760, f=520), same as depth_cam_a_s0.5
WIN = 4  # sampling half-window on that grid -> 9x9 values per joint


def episode_dirs(spec="all", splits=("public", "evaluation")):
    dirs = [d for s in splits for d in sorted(glob.glob(f"{FRAMES}/{s}/episode_*"))]
    if spec in ("all", None):
        return dirs
    want = {int(x) for x in str(spec).split(",")}
    return [d for d in dirs if int(d.rsplit("_", 1)[1]) in want]


def load_meta(ep_dir):
    m = json.load(open(f"{ep_dir}/meta.json"))
    Ka = np.array(m["images"]["cam_a"]["K"], float)
    Kr = np.array(m["images"]["left"]["K"], float)
    T_a_rect = np.array(m["transforms"]["T_a_rect"], float)
    return m, Ka, Kr, T_a_rect


def out_dir(meta):
    return f"{HANDS}/{meta['split']}/episode_{meta['episode']:06d}"


def track3_import():
    if TRACK3 not in sys.path:
        sys.path.insert(0, TRACK3)


class RectDepth:
    """Rectified-left metric depth per frame. One source per episode (never mixed): FoundationStereo PNGs
    (t3_stereo_depth.py output) when prefer='fs' and ALL n_frames PNGs exist, else OpenCV SGBM on the rectified pair
    (same settings as t3_depth_check.sgbm_depth + a left-right check)."""

    def __init__(self, ep_dir, meta, prefer="fs"):
        import cv2
        self.cv2 = cv2
        self.ep_dir, self.meta, self.prefer = ep_dir, meta, prefer
        self.fs_dir = f"{DEPTH}/{meta['split']}/episode_{meta['episode']:06d}/depth_rect"
        self.fx, self.base = meta["stereo"]["fx"], meta["stereo"]["baseline_m"]
        self.sg = cv2.StereoSGBM_create(0, 192, 5, P1=200, P2=800, disp12MaxDiff=2, uniquenessRatio=10,
                                        speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        n_fs = len(glob.glob(f"{self.fs_dir}/[0-9]*.png"))
        self.use_fs = prefer == "fs" and n_fs >= meta["n_frames"]
        self.source = "foundationstereo" if self.use_fs else "sgbm"

    def get(self, f):
        cv2 = self.cv2
        p = f"{self.fs_dir}/{f:06d}.png"
        if self.use_fs:
            track3_import()
            from t3_depth_warp import decode_inv_depth
            return decode_inv_depth(cv2.imread(p, cv2.IMREAD_UNCHANGED)), 1
        L = cv2.imread(f"{self.ep_dir}/left/{f:06d}.jpg", 0)
        R = cv2.imread(f"{self.ep_dir}/right/{f:06d}.jpg", 0)
        disp = self.sg.compute(L, R).astype(np.float32) / 16
        return np.where(disp > 2, self.fx * self.base / np.maximum(disp, 1e-3), 0).astype(np.float32), 2


def project(X, K):
    return X[..., :2] / X[..., 2:3] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]


def backproject(u, z, K):
    return np.concatenate([(u - K[:2, 2]) / np.array([K[0, 0], K[1, 1]]) * z[..., None], z[..., None]], -1)


def transform(T, X):
    return X @ T[:3, :3].T + T[:3, 3]
