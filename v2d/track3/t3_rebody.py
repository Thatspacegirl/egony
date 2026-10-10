"""DEV: swap the mesh of one or more (episode, object) in a t3_devscore.py prediction dir WITHOUT re-running
FoundationPose: the tracked motion is the same physical motion, only the body frame changes.  Both meshes carry a
VO-world pose of their body during the static window (stage JSON frame_T_obj), so
    world_T_new(t) = world_T_old(t) @ inv(frame_T_old) @ frame_T_new.
Used to measure how a different mesh changes the scorer's registration / the pose metrics before spending GPU time.

  python -I t3_rebody.py --pred IN_DIR --out OUT_DIR --swap 23:wooden_piece_1:/path/new.ply:/path/new.json [...]
  (the old JSON is meshes/final/... unless --old_stage is given)
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

R0 = "/mnt/secondary/v2d/t3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--swap", nargs="+", required=True, help="EP:OBJ:NEW_MESH:NEW_JSON")
    ap.add_argument("--old_stage", default="final")
    a = ap.parse_args()
    if os.path.abspath(a.out) != os.path.abspath(a.pred):
        if os.path.exists(a.out):
            shutil.rmtree(a.out)
        shutil.copytree(a.pred, a.out, ignore=shutil.ignore_patterns("perception"))
    for sw in a.swap:
        ep, obj, mesh, js = sw.split(":")
        ep = int(ep)
        e = f"episode_{ep:06d}"
        names = json.load(open(f"{R0}/frames/public/{e}/meta.json"))["objects"]
        b = names.index(obj)
        Fo = np.array(json.load(open(f"{R0}/meshes/{a.old_stage}/public/{e}/{obj}.json"))["frame_T_obj"])
        jn = json.load(open(js))
        Fn = np.array(jn["frame_T_obj"])
        pq = f"{a.out}/{e}.parquet"
        df = pd.read_parquet(pq).sort_values(["frame_index", "object_slot"])
        sel = (df.object_slot == b).to_numpy()
        P = df.loc[sel, ["pos_x", "pos_y", "pos_z"]].to_numpy()
        Q = df.loc[sel, ["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy()
        T = np.tile(np.eye(4), (len(P), 1, 1))
        T[:, :3, :3] = Rot.from_quat(Q).as_matrix()
        T[:, :3, 3] = P
        if "board" in jn and jn.get("info", {}).get("window") == "static_prefix":
            # the new mesh was placed in a different static window than the old one: tie the bodies at that time
            t0 = int(jn["board"]["views"][0])
            D = np.linalg.inv(T[t0]) @ Fn
        else:
            D = np.linalg.inv(Fo) @ Fn
        T2 = T @ D
        df.loc[sel, ["pos_x", "pos_y", "pos_z"]] = T2[:, :3, 3]
        df.loc[sel, ["quat_x", "quat_y", "quat_z", "quat_w"]] = Rot.from_matrix(T2[:, :3, :3]).as_quat()
        df.to_parquet(pq + ".tmp")
        os.replace(pq + ".tmp", pq)
        for ext in ("glb", "obj", "ply", "stl", "off"):
            p = f"{a.out}/{e}/{obj}.{ext}"
            if os.path.exists(p):
                os.remove(p)
        shutil.copy(mesh, f"{a.out}/{e}/{obj}.{mesh.rsplit('.', 1)[1]}")
        print(f"ep {ep} {obj}: body change {np.degrees(Rot.from_matrix(D[:3, :3]).magnitude()):.1f} deg, "
              f"{np.linalg.norm(D[:3, 3]) * 100:.1f} cm")


if __name__ == "__main__":
    main()
