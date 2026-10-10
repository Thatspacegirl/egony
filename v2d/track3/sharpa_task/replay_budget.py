"""Exact FlashSAC replay-buffer size for a generated Track 3 task (CPU only, no training).

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= /mnt/secondary/v2d/envs/flash_chord/bin/python replay_budget.py MANIFEST.json

Composes the same Hydra config as train_3090.sh (3090 preset and the canonical 48 GB recipe), builds the
real RLEnv on the Warp CPU device with world_count=1 to read the runtime dimensions (observation, action,
objective terms, critic context), and evaluates FlashCHORD's own ``replay_memory`` (exact logical bytes of
the replay ring, without allocating it) for both configurations. Replay is the dominant VRAM term of the
recipe; simulator state, networks and XLA workspaces come on top (measure them with the smoke run's
gpu_trace.csv).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2D_FC = Path("/mnt/secondary/v2d/video_to_data/robotic_grounding/flash_chord")


def compose(task: str, preset: bool):
    from omegaconf import OmegaConf

    cmd = [sys.executable, str(V2D_FC / "scripts" / "train_flash_sac.py")]
    cmd += (["--config-dir", str(HERE / "configs"), "experiment=sharpa_flash_sac_3090"] if preset
            else ["experiment=sharpa_flash_sac", "task.motion_speed=1.0"])
    cmd += [f"task.parquet='{task}'", "--cfg", "job", "--resolve"]
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True,
                           env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""})
    return OmegaConf.create(r.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text())
    task = man["task_parquet"]

    import warp as wp

    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene
    from flash_chord.training.flash_sac.config import TrainingConfig
    from flash_chord.training.flash_sac.replay import ReplaySpec, replay_memory

    cfgs = {"3090_preset": compose(task, True), "recipe_48GB": compose(task, False)}
    cfg = cfgs["3090_preset"]
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    with wp.ScopedDevice("cpu"):
        setup = setup_scene(
            parquet=task, control_fps=env_config.sim.fps, motion_speed=cfg.task.motion_speed,
            embodiment=instantiate_typed(cfg.embodiment, Embodiment),
            collision=instantiate_typed(cfg.collision, CollisionPolicy), world_count=1,
            include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects,
            object_scale_min=cfg.scene.object_scale_min, object_scale_max=cfg.scene.object_scale_max,
            object_scale_seed=cfg.scene.object_scale_seed,
            contact_friction=cfg.scene.get("contact_friction", 1.0),
            object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
            source_frame_playback=bool(cfg.task.source_frame_playback),
            motion_start_frame=int(cfg.task.motion_start_frame), motion_end_frame=int(cfg.task.motion_end_frame),
        )
        env = RLEnv(setup.scene, setup.reference, config=env_config)
    dims = {
        "observation_dim": int(env.observation_dim),
        "action_dim": int(env.action.action_dim),
        "objective_term_count": len(env.objective.term_names),
        "critic_context_dim": len(tuple(getattr(env, "critic_context_names", ()))),
        "reference_frames": int(setup.reference.num_frames),
        "objects": man["object_body_names"],
    }
    out = {"task": task, "dims": dims}
    for label, c in cfgs.items():
        tr = instantiate_typed(c.training, TrainingConfig)
        spec = ReplaySpec.from_config(tr.replay, discount=tr.discount, world_count=int(c.scene.world_count), **{
            k: dims[k] for k in ("observation_dim", "action_dim", "objective_term_count", "critic_context_dim")})
        mem = replay_memory(spec)
        out[label] = {
            "world_count": spec.world_count, "capacity": spec.capacity, "obs_dtype": spec.observation_storage_dtype,
            "updates_per_collection": float(c.training.updates_per_collection),
            "bytes_per_transition": int(mem.bytes_per_transition),
            "replay_ring_GiB": round(mem.ring_bytes / 2**30, 3),
            "replay_pending_GiB": round(mem.pending_bytes / 2**30, 4),
            "sampled_batch_MiB": round(mem.sampled_batch_bytes / 2**20, 2),
        }
    print(json.dumps(out, indent=1))
    a.manifest.with_name(a.manifest.stem + "_replay_budget.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
