"""Shared helpers for the Track 1 local validation harness (FORM-HOI pseudo-GT, kit metric code).

Nothing here re-implements a metric: metrics come from the kit's own metric_code/track_1/*.py modules,
loaded by path, with `_BODY` set to a kit-format MHR asset rebuilt from Meta's Apache-2.0 TorchScript
model by mhr_asset.py (kit `_MHRBody` forward == TorchScript to < 0.001 mm).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

KIT = Path(os.environ.get("V2D_KIT", "/mnt/secondary/v2d/kit/v2d_submission_kit"))
MHR_TS = Path(os.environ.get("V2D_MHR", "/mnt/secondary/v2d/scratch/track1/mhr_assets/mhr_model.pt"))
ASSET = Path(os.environ.get("V2D_MHR_ASSET", "/mnt/secondary/v2d/t1/assets/mhr_asset_603.npz"))
VAL_ROOT = Path(os.environ.get("V2D_T1_VAL", "/mnt/secondary/v2d/t1/formhoi_val"))
MIN_STRETCH = 8                       # config/tracks.json track_1.min_stretch_frames
EPISODE_FRAME = 999999
FAKE_COMMIT = "https://github.com/local/v2d-local/commit/" + "0" * 40

if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

_MODULES: dict = {}


def metric_module(name: str):
    """Load metric_code/track_1/<name>.py (CD-H, CD-O, ACC-H, ACC-O, PEN) with our body asset."""
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(f"t1_{name.replace('-', '_')}", KIT / "metric_code" / "track_1" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._BODY = body()
        _MODULES[name] = mod
    return _MODULES[name]


_BODY_CACHE = {}


def body():
    if "b" not in _BODY_CACHE:
        spec = importlib.util.spec_from_file_location("t1_body_loader", KIT / "metric_code" / "track_1" / "CD-H.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _BODY_CACHE["b"] = mod._load_body(ASSET.read_bytes())
    return _BODY_CACHE["b"]


# ------------------------------------------------------------------------------------------ rotations
def euler_xyz_to_mat(e):
    """The scorer's convention (CD-H.py _euler_xyz_to_mat): R = Rz(e2) @ Ry(e1) @ Rx(e0)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("xyz", np.asarray(e).reshape(-1, 3)).as_matrix().reshape(*np.shape(e)[:-1], 3, 3)


def mat_to_euler_xyz(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R).reshape(-1, 3, 3)).as_euler("xyz").reshape(*np.shape(R)[:-2], 3)


# ------------------------------------------------------------------------------------------ FORM-HOI GT
def formhoi_scored_frames(seq_dir, T: int, min_stretch: int = MIN_STRETCH) -> np.ndarray:
    """Track 1 rule: valid frames inside the hand-object contact span, minus failure segments, in
    contiguous stretches of >= min_stretch frames (mhr_submission docstring + tracks.json)."""
    seq_dir = Path(seq_dir)
    it = json.load(open(seq_dir / "interaction_trim.json"))
    if it.get("first_contact_frame_source") is None or it.get("last_contact_frame_source") is None:
        return np.zeros(0, int)            # 'no_stable_contact_keep_full': nothing would be scored
    a = it["first_contact_frame_source"] - it["export_source_start_frame"]
    b = it["last_contact_frame_source"] - it["export_source_start_frame"]
    m = np.zeros(T, bool)
    m[max(a, 0):min(b + 1, T)] = True
    valid = np.load(seq_dir / "pose_valid_mask.npy").astype(bool)
    m[: len(valid)] &= valid[:T]
    for s in json.load(open(seq_dir / "failure_segments.json")):
        m[s["start_frame"]:s["end_frame"]] = False
    frames = np.flatnonzero(m)
    if len(frames) == 0:
        return frames
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    keep = [p for p in np.split(frames, breaks) if len(p) >= min_stretch]
    return np.concatenate(keep) if keep else np.zeros(0, int)


def load_formhoi_gt(seq_dir) -> dict:
    """FORM-HOI GT as a Track 1 episode dict (same keys as the kit NPZ)."""
    import torch
    seq_dir = Path(seq_dir)
    gp = torch.load(seq_dir / "mhr_params_mv.pt", map_location="cpu", weights_only=True)
    mp = gp["mhr_model_params"].double().numpy()
    poses = np.load(seq_dir / "poses.npy").astype(np.float64)
    T = min(len(mp), len(poses))
    R = poses[:T, :3, :3]
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    return {"pose": mp[:T, :136], "scales": mp[0, 136:204].copy(), "shape": gp["shape_params"][0].double().numpy(),
            "object_rotation": R, "object_translation": poses[:T, :3, 3].copy(), "object_scale": np.array(1.0)}


def save_episode(path, ep: dict):
    np.savez(path, **{k: np.asarray(ep[k]) for k in ("pose", "scales", "shape", "object_rotation", "object_translation", "object_scale")})


def load_episode(path) -> dict:
    d = dict(np.load(path))
    return {k: np.asarray(v, np.float64) for k, v in d.items()}


def stretches(frames):
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    return np.split(np.asarray(frames), breaks)
