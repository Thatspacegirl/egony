#!/usr/bin/env python3
"""VAL ONLY: object-mesh SHAPE quality of several mesh sources against the FORM-HOI GT mesh (GT read only here).

Per val episode and source (a glob with {E}):
  shape_cd_sim_cm  symmetric Chamfer (mean of both directions) after the best SIMILARITY alignment (24 start rotations
                   + ICP with scale), in GT centimetres: shape only, size-free
  scale_ratio      (source size) / (GT size) from that alignment; meaningful only for metric meshes (sam3do raw,
                   CARI4D's output_aligned.glb after the FoundationPose scale hook)
  shape_cd_rigid_cm same with a RIGID alignment (size errors included; for metric meshes)
  python t1_mesh_shape.py --src sam3do=/mnt/secondary/v2d/t1/mesh_sam3do_fh/val/{E}/mesh.glb \
        --src trellis2=/mnt/secondary/v2d/t1/mesh_trellis2_fh/val/{E}/mesh_0.ply --episodes 5,11 [--json out]
Python: /mnt/secondary/v2d/scratch/venv/bin/python -I (trimesh, scipy).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_items as I  # noqa: E402
from t1_mesh_eval import best_align  # noqa: E402


def load_points(path, n, seed):
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    p, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    return np.asarray(p, float), m


def parse_eps(s):
    out = []
    for p in s.split(","):
        if "-" in p:
            a, b = p.split("-"); out += list(range(int(a), int(b) + 1))
        elif p:
            out.append(int(p))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", default=[])
    ap.add_argument("--episodes", default="0-24")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--json")
    a = ap.parse_args()
    srcs = [s.split("=", 1) for s in a.src]
    rows = []
    for ep in parse_eps(a.episodes):
        it = I.item("val", ep)
        E = f"episode_{ep:06d}"
        G, _ = load_points(I.VAL_ROOT / it["sequence_id"] / "object_mesh__output_aligned.glb", a.n, 0)
        gsize = float(np.sqrt(((G - G.mean(0)) ** 2).sum(1).mean()))
        for name, pat in srcs:
            hits = sorted(glob.glob(pat.replace("{E}", E)))
            if not hits:
                continue
            P, _ = load_points(hits[0], a.n, 0)
            cd_s, s_s, R_s, Rc_s = best_align(P, G, True)
            cd_r, _, _, _ = best_align(P, G, False)
            row = {"ep": ep, "object": it["object"], "src": name, "shape_cd_sim_cm": round(cd_s * 100, 3),
                   "scale_ratio": round(1.0 / s_s, 4), "shape_cd_rigid_cm": round(cd_r * 100, 3),
                   "gt_rms_radius_cm": round(gsize * 100, 2), "mesh": hits[0]}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "mesh"}), flush=True)
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        print(df.groupby("src")[["shape_cd_sim_cm", "shape_cd_rigid_cm", "scale_ratio"]].agg(["mean", "median"]).round(3).to_string())
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
