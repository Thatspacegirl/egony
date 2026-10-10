"""Convert FlashCHORD evaluator rollouts into the public-split dev scorer's prediction layout.

    python rollout_to_devscore.py RESULTS_DIR MANIFEST_DIR OUT_DIR [--which achieved|reference]

RESULTS_DIR holds evaluate_policy.py / kit run_policy_evaluation.py outputs (episode_XXXXXX.parquet + .json).
For each episode it checks the packer-relevant invariants (step_count == N video frames, every world valid,
object_body_name order == manifest roster) and writes
    OUT_DIR/episode_XXXXXX.parquet  (frame_index, object_slot, pos_x..z, quat_x..w; world 0)
    OUT_DIR/episode_XXXXXX/<name>.obj  (the exact mesh that was simulated)
which /mnt/secondary/v2d/scratch/track3_perception/t3_devscore.py scores with the official metric code
(public episodes only). ``--which reference`` emits the reference poses instead (policy-free upper bound).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("manifests", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--which", choices=("achieved", "reference"), default="achieved")
    ap.add_argument("--task-reference", action="store_true",
                    help="ignore RESULTS_DIR; emit the converter's training reference (task parquet) for every "
                         "manifest: measures the error the converter itself adds (snapping, frames), no rollout")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.task_reference:
        return task_reference(a.manifests, a.out)
    n_ok = 0
    for traj in sorted(a.results.glob("episode_*.parquet")):
        seq = traj.stem
        man = json.loads((a.manifests / f"{seq}.json").read_text())
        f = pq.ParquetFile(traj)
        md = {k.decode(): v.decode() for k, v in (f.schema_arrow.metadata or {}).items()}
        steps = int(md["flash_chord.step_count"])
        df = f.read().to_pandas()
        names = df.sort_values("object_body_id").drop_duplicates("object_body_id")["object_body_name"].astype(str).tolist()
        problems = []
        if steps != man["num_frames"]:
            problems.append(f"step_count {steps} != N {man['num_frames']} (motion_speed/fps mismatch?)")
        if not df["environment_valid"].all():
            problems.append("some world is environment_valid=false (packer would reject)")
        if names != man["object_body_names"]:
            problems.append(f"body names {names} != roster {man['object_body_names']}")
        if int(md.get("flash_chord.episode_index", -1)) != man["episode_index"]:
            problems.append(f"episode_index metadata {md.get('flash_chord.episode_index')} != {man['episode_index']}")
        if problems:
            print(f"{seq}: FAIL " + "; ".join(problems))
            continue
        w0 = df[df.environment_id == 0].sort_values(["step", "object_body_id"])
        p = a.which
        out = {
            "frame_index": w0["reference_frame"].to_numpy(np.int64),
            "object_slot": w0["object_body_id"].to_numpy(np.int64),
            "pos_x": w0[f"{p}_position_x"].to_numpy(np.float64),
            "pos_y": w0[f"{p}_position_y"].to_numpy(np.float64),
            "pos_z": w0[f"{p}_position_z"].to_numpy(np.float64),
            "quat_x": w0[f"{p}_quaternion_x"].to_numpy(np.float64),
            "quat_y": w0[f"{p}_quaternion_y"].to_numpy(np.float64),
            "quat_z": w0[f"{p}_quaternion_z"].to_numpy(np.float64),
            "quat_w": w0[f"{p}_quaternion_w"].to_numpy(np.float64),
        }
        import pandas as pd

        pd.DataFrame(out).to_parquet(a.out / f"{seq}.parquet", index=False)
        mdir = a.out / seq
        mdir.mkdir(exist_ok=True)
        for o in man["objects"]:
            shutil.copyfile(o["obj"], mdir / f"{o['name']}.obj")
        n_ok += 1
        print(f"{seq}: OK steps={steps} bodies={names} -> {a.out / (seq + '.parquet')}")
    return 0 if n_ok else 1


def task_reference(manifests: Path, out: Path) -> int:
    import pandas as pd
    import pyarrow.dataset as pads

    n = 0
    import re

    for mpath in sorted(manifests.glob("episode_*.json")):
        if not re.fullmatch(r"episode_\d{6}", mpath.stem):
            continue  # skip _verify / _settle side files
        man = json.loads(mpath.read_text())
        t = pads.dataset(man["task_parquet"], format="parquet").to_table(
            columns=["object_body_position", "object_body_wxyz", "object_body_names", "fps"])
        pos = np.asarray(t["object_body_position"][0].as_py(), np.float64)
        q = np.asarray(t["object_body_wxyz"][0].as_py(), np.float64)  # wxyz
        T, B, _ = pos.shape
        assert T == man["num_frames"] and t["object_body_names"][0].as_py() == man["object_body_names"]
        fr, sl = np.meshgrid(np.arange(T), np.arange(B), indexing="ij")
        pd.DataFrame({
            "frame_index": fr.ravel(), "object_slot": sl.ravel(),
            "pos_x": pos[..., 0].ravel(), "pos_y": pos[..., 1].ravel(), "pos_z": pos[..., 2].ravel(),
            "quat_x": q[..., 1].ravel(), "quat_y": q[..., 2].ravel(), "quat_z": q[..., 3].ravel(), "quat_w": q[..., 0].ravel(),
        }).to_parquet(out / f"{man['sequence_id']}.parquet", index=False)
        mdir = out / man["sequence_id"]
        mdir.mkdir(exist_ok=True)
        for o in man["objects"]:
            shutil.copyfile(o["obj"], mdir / f"{o['name']}.obj")
        print(f"{man['sequence_id']}: task reference T={T} B={B} fps={t['fps'][0].as_py()} -> {out}")
        n += 1
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
