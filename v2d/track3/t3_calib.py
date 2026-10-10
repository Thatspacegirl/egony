"""Track 3 ego-rig calibration helpers (verified conventions, see README.md).

Conventions (verified empirically on public frames, Phase-1 report index 5):
  * JSON intrinsics/distortion (14-coef OpenCV rational+thin-prism+tilt) apply to the released
    upright frames unchanged (do NOT rotate them by 180 deg).
  * Stereo pair: left = ego_cam_c, right = ego_cam_b, R/t = stereo_pairs[ego].left_to_right
    (== ego_cam_c.extrinsics_to_stereo_left = M_c), X_b = M_c X_c.  Rectify with alpha=0.
  * Colour camera: X_a = inv(M_a) X_b, i.e. T_a<-c = inv(M_a) @ M_c.
  * Rectified-left frame: X_rect = R1 X_c  =>  T_rect<-c = [R1 | 0].

All 4x4 transforms are "T_dst<-src" (X_dst = T @ X_src), metres.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

STEREO_SIZE = (1280, 800)       # (w, h) of ego_cam_b / ego_cam_c
CAM_A_SIZE = (2028, 1520)       # (w, h) of ego_cam_a
CAM_A_UNDIST_F = 1040.0         # square-pixel pinhole focal for undistorted cam_a (100% valid pixels)


def _KD(cams, name):
    c = cams[name]
    return np.array(c["intrinsic_matrix"], np.float64), np.array(c["distortion_coefficients"], np.float64)


def _h(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.ravel(t)
    return T


def load_ego_calibration(calib_json: str) -> dict:
    """Return every matrix the downstream perception stack needs (numpy arrays)."""
    C = json.load(open(calib_json))
    cams = C["cameras"]
    pair = [p for p in C["stereo_pairs"] if p["camera"] == "ego"][0]
    assert pair["left"] == "ego_cam_c" and pair["right"] == "ego_cam_b", pair
    Kc, Dc = _KD(cams, "ego_cam_c")
    Kb, Db = _KD(cams, "ego_cam_b")
    Ka, Da = _KD(cams, "ego_cam_a")
    T_lr = np.array(pair["left_to_right"], np.float64)            # X_b = T_lr X_c
    Mc = np.array(cams["ego_cam_c"]["extrinsics_to_stereo_left"], np.float64)
    Ma = np.array(cams["ego_cam_a"]["extrinsics_to_stereo_left"], np.float64)
    assert np.abs(T_lr - Mc).max() < 1e-5, "stereo_pairs.left_to_right should equal M_c"

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        Kc, Dc, Kb, Db, STEREO_SIZE, np.ascontiguousarray(T_lr[:3, :3]),
        np.ascontiguousarray(T_lr[:3, 3:4]), alpha=0)
    baseline = float(-P2[0, 3] / P2[0, 0])
    K_rect = P1[:3, :3].copy()

    Ka_u = np.array([[CAM_A_UNDIST_F, 0, Ka[0, 2]], [0, CAM_A_UNDIST_F, Ka[1, 2]], [0, 0, 1]], np.float64)

    T_b_c = T_lr
    T_a_c = np.linalg.inv(Ma) @ Mc
    T_rect_c = _h(R1)
    T_c_rect = _h(R1.T)
    T_a_rect = T_a_c @ T_c_rect
    return dict(
        Kc=Kc, Dc=Dc, Kb=Kb, Db=Db, Ka=Ka, Da=Da, Ma=Ma, Mc=Mc,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, roi1=roi1, roi2=roi2,
        K_rect=K_rect, baseline=baseline, Ka_undist=Ka_u,
        T_b_c=T_b_c, T_a_c=T_a_c, T_rect_c=T_rect_c, T_c_rect=T_c_rect, T_a_rect=T_a_rect,
        cross_device_skew_ms=C.get("cross_device_skew_ms"),
    )


def rectify_maps(cal: dict):
    """Fixed-point remap tables for (left=cam_c, right=cam_b, cam_a undistort)."""
    def fp(K, D, R, P, size):
        m1, m2 = cv2.initUndistortRectifyMap(K, D, R, P, size, cv2.CV_32FC1)
        return cv2.convertMaps(m1, m2, cv2.CV_16SC2)
    left = fp(cal["Kc"], cal["Dc"], cal["R1"], cal["P1"], STEREO_SIZE)
    right = fp(cal["Kb"], cal["Db"], cal["R2"], cal["P2"], STEREO_SIZE)
    cam_a = fp(cal["Ka"], cal["Da"], np.eye(3), cal["Ka_undist"], CAM_A_SIZE)
    return left, right, cam_a


def calib_to_jsonable(cal: dict) -> dict:
    out = {}
    for k, v in cal.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out
