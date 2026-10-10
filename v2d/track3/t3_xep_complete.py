"""Complete a cross-episode fused mesh ONCE (in its shared frame) and give every episode the same completed mesh.

t3_xep_fuse.py writes the same fused mesh into every accepted episode with per-episode frame_T_obj.  Completing each
copy separately re-estimates the table plane per episode (a 23 deg tilt error on public 12 broke one cup); here the
reference episode's completion (t3_mesh_complete.py) is reused and only frame_T_obj is re-expressed:
frame_T_obj_e(completed) = frame_T_obj_e(xep) @ T(recentre offset of the reference completion).

  python -I t3_xep_complete.py --split public --obj blue_cup --mode revolve --in_root .../xep --out_root .../xepc
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

R0 = "/mnt/secondary/v2d/t3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--in_root", default=f"{R0}/meshes/xep")
    ap.add_argument("--out_root", default=f"{R0}/meshes/xepc")
    a = ap.parse_args()
    root = f"{a.in_root}/{a.split}"
    eps = sorted(d for d in os.listdir(root) if os.path.exists(f"{root}/{d}/{a.obj}.json"))
    if not eps:
        raise SystemExit("no xep outputs")
    ref = json.load(open(f"{root}/{eps[0]}/{a.obj}.json"))["xep"]["ref"]
    ep_ref = int(ref[-6:])
    subprocess.run([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "t3_mesh_complete.py"),
                    "--split", a.split, "--episode", str(ep_ref), "--obj", a.obj, "--mode", a.mode,
                    "--in_root", a.in_root, "--out_root", a.out_root], check=True)
    cref = json.load(open(f"{a.out_root}/{a.split}/{ref}/{a.obj}.json"))
    Tc = np.eye(4)
    Tc[:3, 3] = cref["completion"]["recentred_by_m"]
    for e in eps:
        if e == ref:
            continue
        j = json.load(open(f"{root}/{e}/{a.obj}.json"))
        j["frame_T_obj"] = (np.array(j["frame_T_obj"]) @ Tc).tolist()
        j["completion"] = dict(cref["completion"], shared_from=ref)
        os.makedirs(f"{a.out_root}/{a.split}/{e}", exist_ok=True)
        shutil.copy(f"{a.out_root}/{a.split}/{ref}/{a.obj}.ply", f"{a.out_root}/{a.split}/{e}/{a.obj}.ply")
        json.dump(j, open(f"{a.out_root}/{a.split}/{e}/{a.obj}.json", "w"), indent=1)
    print(a.obj, "completed in", ref, "shared with", [e for e in eps if e != ref])


if __name__ == "__main__":
    main()
