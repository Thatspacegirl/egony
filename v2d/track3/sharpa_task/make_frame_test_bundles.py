"""DEV TEST: re-express a dev bundle in a tilted/offset 'VO' world with an explicit table plane and input hands.

Exercises the converter paths that GT bundles do not (gravity alignment from a table plane, xy recentering,
'auto' table height when the input table is below the ground plane, MANO / keypoint-only input hands).
The scored result must not change: Track 3 metrics are invariant to the world frame.

    python make_frame_test_bundles.py SRC_BUNDLE LOADED_TASK_DIR OUT_DIR TABLE_Z
      LOADED_TASK_DIR = .../loaded/sequence_id=episode_X/robot_name=sharpa_wave of a previous run (its MANO
      joints/orientations, in that run's sim frame == the source bundle frame when run with --no-recenter-xy
      and an in-range table, are used as 'perception hands')
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sharpa_task.bundle import save_bundle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("loaded", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("table_z", type=float)
    a = ap.parse_args()
    z = dict(np.load(a.src, allow_pickle=False))
    rot = R.from_euler("zyx", [70.0, 12.0, -15.0], degrees=True)  # yaw + 19 deg total tilt
    t = np.array([1.3, -0.4, -0.9])  # puts the table below z=0 -> converter must re-place it
    pose = z["object_pose"].copy()
    pose[..., :3] = pose[..., :3] @ rot.as_matrix().T + t
    q = R.from_quat(pose[..., 3:].reshape(-1, 4), scalar_first=True)
    pose[..., 3:] = (rot * q).as_quat(scalar_first=True).reshape(pose.shape[:-1] + (4,))
    n = rot.apply([0.0, 0.0, 1.0])
    x0 = rot.apply([0.0, 0.0, a.table_z]) + t
    tbl = np.array([*n, -float(n @ x0)])
    tab = pads.dataset(str(a.loaded), format="parquet").to_table(
        columns=[f"mano_{s}_{k}" for s in ("right", "left") for k in ("joints", "joints_wxyz")])
    base = {k: v for k, v in z.items() if k not in ("object_pose", "up", "provenance")}
    prov = {"source": "frame-invariance test of " + str(a.src), "R_zyx_deg": [70, 12, -15], "t": t.tolist()}
    for variant in ("mano", "keypoints"):
        arrays = dict(base, object_pose=pose, table_plane=tbl, up=n, provenance=np.array(json.dumps(prov)))
        for s in ("right", "left"):
            J = np.asarray(tab[f"mano_{s}_joints"][0].as_py(), float) @ rot.as_matrix().T + t
            arrays[f"hand_{s}_joints"] = J
            if variant == "mano":
                Q = R.from_quat(np.asarray(tab[f"mano_{s}_joints_wxyz"][0].as_py(), float).reshape(-1, 4), scalar_first=True)
                arrays[f"hand_{s}_joints_wxyz"] = (rot * Q).as_quat(scalar_first=True).reshape(J.shape[:2] + (4,))
            v = np.ones(J.shape[0], bool)
            v[40:45] = False  # exercise gap filling
            arrays[f"hand_{s}_valid"] = v
        ov = np.ones(pose.shape[:2], bool)
        ov[60:63, 0] = False
        arrays["object_valid"] = ov
        out = a.out / f"frametest_{variant}_{a.src.name}"
        save_bundle(out, **arrays)
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
