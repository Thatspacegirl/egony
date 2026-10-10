"""Per-episode perception bundle (``t3_perception_v1``): the converter's input contract.

One ``.npz`` per episode, written by the perception pipeline (or, for dev only, by make_dev_inputs.py):

  schema              str   "t3_perception_v1"
  episode_index       int
  num_frames          int   N == number of video frames (eval: 151..421); rollout step i = frame i
  object_names        (B,)  str   EXACTLY the roster order of data/track_3_evaluation_objects.json for eval
  object_mesh_paths   (B,)  str   our reconstructed meshes (.obj/.ply/.glb/.stl/.off); poses refer to the
                                  vertices of trimesh.load(path, force="mesh") (GLB node transforms baked)
  object_pose         (N,B,7) f   world_T_object [x, y, z, qw, qx, qy, qz] in the perception world (e.g. VO)
  object_valid        (N,B) bool  optional, default all True; invalid frames are interpolated / held
  table_plane         (4,)  f     optional [nx, ny, nz, d] with n.x + d = 0 and n pointing UP (away from table)
  up                  (3,)  f     optional gravity-up in the perception world (default: table normal, else +z)
  object_mass         (B,)  f     optional kg, NaN = auto (name prior, else hull volume x 250 kg/m^3)
  hand_<side>_joints       (N,21,3) f  optional MANO-order joints in the perception world (side = right/left)
  hand_<side>_joints_wxyz  (N,21,4) f  optional per-joint global orientations (MANO FK); if absent they are
                                       Kabsch-fitted from the palm keypoints
  hand_<side>_valid        (N,) bool   optional
  dev_only            bool  True for bundles built from public GT (never packed / submitted)
  provenance          str   free-form JSON (who/what produced it)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import BUNDLE_SCHEMA
from .geom import qnorm

ROSTER_JSON = "/mnt/secondary/v2d/kit/v2d_submission_kit/data/track_3_evaluation_objects.json"
PUBLIC_EPISODES = {0, 2, 11, 12, 13, 18, 21, 22, 23, 24, 26, 27, 31, 39, 40, 41, 42, 43, 44, 45}


@dataclass
class Bundle:
    path: Path
    episode_index: int
    num_frames: int
    object_names: list[str]
    object_mesh_paths: list[str]
    object_pos: np.ndarray  # (N,B,3)
    object_quat: np.ndarray  # (N,B,4) wxyz
    object_valid: np.ndarray  # (N,B)
    table_plane: np.ndarray | None
    up: np.ndarray | None
    object_mass: np.ndarray  # (B,) NaN = auto
    hands: dict = field(default_factory=dict)  # side -> {"joints", "quats"|None, "valid"}
    dev_only: bool = False
    provenance: dict = field(default_factory=dict)

    @property
    def sequence_id(self) -> str:
        return f"episode_{self.episode_index:06d}"


def eval_roster(path: str = ROSTER_JSON) -> dict[int, list[str]]:
    return {int(k): list(v) for k, v in json.loads(Path(path).read_text())["episodes"].items()}


def save_bundle(path: Path, **arrays) -> Path:
    arrays.setdefault("schema", BUNDLE_SCHEMA)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_bundle(path: str | Path, roster_json: str = ROSTER_JSON) -> Bundle:
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    schema = str(z["schema"]) if "schema" in z else ""
    if schema != BUNDLE_SCHEMA:
        raise ValueError(f"{path}: schema {schema!r} != {BUNDLE_SCHEMA!r}")
    ep = int(z["episode_index"])
    N = int(z["num_frames"])
    names = [str(x) for x in z["object_names"]]
    meshes = [str(x) for x in z["object_mesh_paths"]]
    pose = np.asarray(z["object_pose"], np.float64)
    B = len(names)
    if pose.shape != (N, B, 7):
        raise ValueError(f"object_pose shape {pose.shape} != {(N, B, 7)}")
    if len(meshes) != B or len(set(names)) != B or not all(names):
        raise ValueError("object_names must be unique/non-empty and match object_mesh_paths")
    for m in meshes:
        if not Path(m).is_file():
            raise FileNotFoundError(m)
    valid = np.asarray(z["object_valid"], bool) if "object_valid" in z else np.ones((N, B), bool)
    if valid.shape != (N, B):
        raise ValueError(f"object_valid shape {valid.shape} != {(N, B)}")
    valid &= np.isfinite(pose).all(-1) & (np.linalg.norm(pose[..., 3:], axis=-1) > 1e-6)
    quat = pose[..., 3:].copy()
    quat[valid] = qnorm(quat[valid])
    dev_only = bool(z["dev_only"]) if "dev_only" in z else False
    roster = eval_roster(roster_json)
    if ep in roster:
        if names != roster[ep]:
            raise ValueError(f"episode {ep}: object_names {names} != roster order {roster[ep]}")
        if dev_only:
            raise ValueError(f"episode {ep} is an EVAL episode but the bundle is marked dev_only")
    elif ep not in PUBLIC_EPISODES:
        raise ValueError(f"episode {ep} is neither an eval nor a public Track 3 episode")
    if dev_only and ep not in PUBLIC_EPISODES:
        raise ValueError("dev_only bundles are allowed for public episodes only")
    hands = {}
    for side in ("right", "left"):
        key = f"hand_{side}_joints"
        if key not in z:
            continue
        J = np.asarray(z[key], np.float64)
        if J.shape != (N, 21, 3):
            raise ValueError(f"{key} shape {J.shape} != {(N, 21, 3)}")
        Q = np.asarray(z[f"hand_{side}_joints_wxyz"], np.float64) if f"hand_{side}_joints_wxyz" in z else None
        if Q is not None and Q.shape != (N, 21, 4):
            raise ValueError(f"hand_{side}_joints_wxyz shape {Q.shape} != {(N, 21, 4)}")
        hv = np.asarray(z[f"hand_{side}_valid"], bool) if f"hand_{side}_valid" in z else np.ones(N, bool)
        hv &= np.isfinite(J).all((1, 2))
        hands[side] = {"joints": J, "quats": Q, "valid": hv}
    mass = np.asarray(z["object_mass"], np.float64) if "object_mass" in z else np.full(B, np.nan)
    prov = json.loads(str(z["provenance"])) if "provenance" in z else {}
    return Bundle(
        path=path,
        episode_index=ep,
        num_frames=N,
        object_names=names,
        object_mesh_paths=meshes,
        object_pos=pose[..., :3],
        object_quat=quat,
        object_valid=valid,
        table_plane=np.asarray(z["table_plane"], np.float64) if "table_plane" in z else None,
        up=np.asarray(z["up"], np.float64) if "up" in z else None,
        object_mass=mass,
        hands=hands,
        dev_only=dev_only,
        provenance=prov,
    )
