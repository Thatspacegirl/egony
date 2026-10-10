"""Synthetic 'monocular-like' corruption of FORM-HOI GT episodes, for tuning post-processing.

Errors are generated in the camera frame of the validation camera (edex camera-to-world; world = front-left
camera of the rig), so depth errors are larger than lateral ones, like a single-view method. Components:
  iid jitter (per frame)          root translation, root rotation, body pose, hands, object translation/rotation
  low-frequency drift (LF)         Gaussian-filtered white noise, correlation `lf_frames`
  glitches                         rare 1-3 frame bursts (root rotation / translation, object translation)
  scene depth-scale error          human root + object positions scaled about the camera centre by (1+eps)
                                   (monocular scale inconsistency; smoothing cannot fix it, it sets the CD floor)
Presets are calibrated so the UNSMOOTHED corrupted GT lands near the public baseline (CARI4D) leaderboard
row (ACC-H ~2.0, ACC-O ~0.73, CD-H ~15, CD-O ~23) or a raw per-frame regime with 2-3x more jitter.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

FLIP = np.array([1.0, -1.0, -1.0])
CAM_INDEX = {"front_stereo_camera_left": 0, "back_stereo_camera_left": 2, "left_stereo_camera_left": 4, "right_stereo_camera_left": 6}
FLEX_COLS = np.array([130, 131, 132, 133, 134, 135])   # *_length/width_flexible (identity-like), no noise
HAND_COLS = np.arange(68, 122)
BODY_ROT_COLS = np.setdiff1d(np.arange(6, 130), HAND_COLS)


@dataclass
class NoiseCfg:
    # defaults = 'cari4d_like', calibrated on the 25 FORM-HOI val episodes so the UNSMOOTHED corruption scores
    # CD-H 15.55 / CD-O 20.04 / ACC-H 2.07 / ACC-O 0.72 (public CARI4D baseline row: 15.6 / ~23 / 1.97 / 0.73)
    name: str = "cari4d_like"
    root_t_lat: float = 0.0015     # m iid, camera x/y
    root_t_dep: float = 0.003      # m iid, camera z
    root_r: float = 0.22           # deg iid per axis
    body: float = 0.005            # rad iid
    hand: float = 0.05             # rad iid
    obj_t_lat: float = 0.001
    obj_t_dep: float = 0.002
    obj_r: float = 1.0             # deg iid per axis
    lf_frames: float = 45.0
    lf_root_t_lat: float = 0.015   # m
    lf_root_t_dep: float = 0.03
    lf_root_r: float = 1.0         # deg
    lf_body: float = 0.03          # rad
    lf_obj_t_lat: float = 0.025
    lf_obj_t_dep: float = 0.06
    lf_obj_r: float = 8.0          # deg
    glitch_rate: float = 0.003     # bursts per frame
    glitch_root_r: float = 20.0    # deg
    glitch_root_t: float = 0.08    # m
    glitch_obj_t: float = 0.05     # m
    depth_scale_sd: float = 0.015


PRESETS = {
    "clean": NoiseCfg(name="clean", root_t_lat=0, root_t_dep=0, root_r=0, body=0, hand=0, obj_t_lat=0, obj_t_dep=0, obj_r=0,
                      lf_root_t_lat=0, lf_root_t_dep=0, lf_root_r=0, lf_body=0, lf_obj_t_lat=0, lf_obj_t_dep=0, lf_obj_r=0,
                      glitch_rate=0, depth_scale_sd=0),
    "cari4d_like": NoiseCfg(),
}
# per-frame (un-refined) SAM 3D Body + FoundationPose: ~3x the iid jitter, 3x more glitches
PRESETS["raw_sam3d_like"] = replace(NoiseCfg(), name="raw_sam3d_like", root_t_lat=0.0045, root_t_dep=0.009, root_r=0.66, body=0.015,
                                    hand=0.1, obj_t_lat=0.003, obj_t_dep=0.006, obj_r=3.0, glitch_rate=0.01)
PRESETS["hf_only"] = replace(NoiseCfg(), name="hf_only", lf_root_t_lat=0, lf_root_t_dep=0, lf_root_r=0, lf_body=0, lf_obj_t_lat=0,
                             lf_obj_t_dep=0, lf_obj_r=0, depth_scale_sd=0)


def camera_from_edex(seq_dir, camera):
    d = json.load(open(Path(seq_dir) / "edex"))
    T = np.asarray(d[0]["cameras"][CAM_INDEX[camera]]["transform"], np.float64)   # camera-to-world [3,4]
    return T[:, :3], T[:, 3]


def _lf(rng, T, dims, frames):
    x = gaussian_filter1d(rng.standard_normal((T + 6 * int(frames), dims)), frames, axis=0, mode="reflect")
    x = x[3 * int(frames):3 * int(frames) + T]
    return x / (x.std(0, keepdims=True) + 1e-12)


def _glitch_mask(rng, T, rate):
    m = np.zeros(T)
    n = rng.poisson(rate * T)
    for s in rng.integers(0, max(T - 3, 1), n):
        m[s:s + int(rng.integers(1, 4))] = 1.0
    return m


def corrupt(ep: dict, seq_dir, camera, cfg: NoiseCfg, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {k: np.array(v, dtype=np.float64, copy=True) for k, v in ep.items()}
    pose = out["pose"]
    T = len(pose)
    R_wc, c = camera_from_edex(seq_dir, camera)
    deg = np.pi / 180

    def cam_noise(lat, dep, lf_lat, lf_dep, glitch_amp):
        n_cam = rng.standard_normal((T, 3)) * np.array([lat, lat, dep])
        n_cam += _lf(rng, T, 3, cfg.lf_frames) * np.array([lf_lat, lf_lat, lf_dep])
        g = _glitch_mask(rng, T, cfg.glitch_rate)
        n_cam += g[:, None] * rng.standard_normal((T, 3)) * glitch_amp
        return n_cam @ R_wc.T                                  # world-frame metres

    eps = rng.normal(0, cfg.depth_scale_sd) if cfg.depth_scale_sd > 0 else 0.0
    # --- human root translation (param units: scorer metres = FLIP * param / 10)
    root_world = None
    if eps:
        # approximate the body root position by the param translation itself (world, metres)
        root_world = FLIP * pose[:, :3] / 10.0
    n_root = cam_noise(cfg.root_t_lat, cfg.root_t_dep, cfg.lf_root_t_lat, cfg.lf_root_t_dep, cfg.glitch_root_t)
    if eps:
        n_root += eps * (root_world - c)
    pose[:, :3] += FLIP * n_root * 10.0
    # --- root rotation (isotropic small rotations composed on the left of the local root rotation)
    w = rng.standard_normal((T, 3)) * cfg.root_r * deg + _lf(rng, T, 3, cfg.lf_frames) * cfg.lf_root_r * deg
    g = _glitch_mask(rng, T, cfg.glitch_rate)
    w += g[:, None] * rng.standard_normal((T, 3)) * cfg.glitch_root_r * deg
    R = Rotation.from_rotvec(w) * Rotation.from_euler("xyz", pose[:, 3:6])
    pose[:, 3:6] = R.as_euler("xyz")
    # --- body / hands
    nb = len(BODY_ROT_COLS)
    pose[:, BODY_ROT_COLS] += rng.standard_normal((T, nb)) * cfg.body + _lf(rng, T, nb, cfg.lf_frames) * cfg.lf_body
    pose[:, HAND_COLS] += rng.standard_normal((T, len(HAND_COLS))) * cfg.hand
    out["pose"] = pose
    # --- object
    n_obj = cam_noise(cfg.obj_t_lat, cfg.obj_t_dep, cfg.lf_obj_t_lat, cfg.lf_obj_t_dep, cfg.glitch_obj_t)
    if eps:
        n_obj += eps * (out["object_translation"] - c)
    out["object_translation"] = out["object_translation"] + n_obj
    wo = rng.standard_normal((T, 3)) * cfg.obj_r * deg + _lf(rng, T, 3, cfg.lf_frames) * cfg.lf_obj_r * deg
    out["object_rotation"] = (Rotation.from_rotvec(wo) * Rotation.from_matrix(out["object_rotation"])).as_matrix()
    return out


def cfg_dict(cfg):
    return asdict(cfg)
