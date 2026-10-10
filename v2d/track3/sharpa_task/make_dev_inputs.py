"""DEV ONLY: build t3_perception_v1 bundles from PUBLIC Track 3 ground truth + public scans.

Stand-ins for perception output so the policy side (converter, IK, FlashCHORD training/eval, dev scoring)
can be developed before our own perception exists. Bundles are marked ``dev_only`` (the converter never
writes submit meshes for them) and only PUBLIC episodes are accepted. Never use for eval episodes or for a
submission: Track 3 requires our own reconstructed meshes and our own perception.

    python make_dev_inputs.py --episodes 12 41 0 --out /mnt/secondary/v2d/t3/dev_bundles
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sharpa_task.bundle import PUBLIC_EPISODES, eval_roster, save_bundle  # noqa: E402

PUBLIC = Path("/home/asubuntudesktop/TestingGrounds/egony/video_to_data_challenge/track_3/public")
DATASET_REVISION = "5f68335acc802033d1e80728c1633197521de8"


def public_mesh(name: str) -> Path:
    d = PUBLIC / "mesh" / name
    for cand in (d / f"{name}.glb", d / f"{name}_visual.glb"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no public scan for {name} in {d}")


def build(ep: int, out_dir: Path) -> Path:
    if ep not in PUBLIC_EPISODES or ep in eval_roster():
        raise SystemExit(f"episode {ep} is not a public Track 3 episode; refusing (dev-only tool)")
    p = PUBLIC / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    f = pq.ParquetFile(p)
    if (f.schema_arrow.metadata or {}).get(b"pose_convention", b"") != b"world_T_object":
        raise SystemExit(f"{p}: unexpected pose convention")
    objs = f.read(columns=["observation.objects"])["observation.objects"].to_pylist()
    N = len(objs)
    names = [o["name"] for o in objs[0]]
    for fr in objs:
        if [o["name"] for o in fr] != names:
            raise SystemExit(f"{p}: object membership changes within the episode")
    pose = np.array([[o["pose"] for o in fr] for fr in objs], dtype=np.float64)  # x y z qw qx qy qz
    vis = np.array([[o["visible"] for o in fr] for fr in objs], dtype=bool)
    out = out_dir / f"dev_gt_episode_{ep:06d}.npz"
    save_bundle(
        out,
        episode_index=np.int64(ep),
        num_frames=np.int64(N),
        object_names=np.array(names),
        object_mesh_paths=np.array([str(public_mesh(n)) for n in names]),
        object_pose=pose,
        object_valid=vis,
        up=np.array([0.0, 0.0, 1.0]),
        dev_only=np.bool_(True),
        provenance=np.array(json.dumps({
            "source": "public_gt", "parquet": str(p), "dataset_revision": DATASET_REVISION,
            "meshes": "public scans (dev stand-ins, NOT our reconstructions)", "hands": "none",
        })),
    )
    print(f"{out}: N={N} objects={names} visible={vis.mean(0).round(3).tolist()}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("/mnt/secondary/v2d/t3/dev_bundles"))
    a = ap.parse_args()
    for ep in a.episodes:
        build(ep, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
