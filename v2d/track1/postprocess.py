#!/usr/bin/env python3
"""Apply the tuned post-processing to a directory of Track 1 episode NPZs (any model's output).

  scored frames per episode come from the competition sample submission (row ids t1_<ep>_<frame>_...), or from
  the local val_index.json; f0 = first scored frame (the scorer's Sim(3) frame).
  1. smoothing.smooth_episode (SO(3) root/object rotation, Whittaker, robust, frame-0 anchor per cfg)
  2. pen_refine.refine against the BUDGETED mesh the scorer will see: hand-side finger/wrist de-penetration, then
     (only if PEN is still > --object-if-pen-above) a smooth object-offset curve
Writes <out>/episode_%06d.npz + the mesh copied unchanged + episode_%06d_post.json reports.

usage:
  postprocess.py --in <dir> --out <dir> --sample data/track_1_sample_submission.parquet [--smooth-json cfg.json] [--no-pen]
  postprocess.py --in <dir> --out <dir> --val-index /mnt/secondary/v2d/t1/formhoi_val/kitval/val_index.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1lib as L  # noqa: E402
import smoothing as SM  # noqa: E402

DEFAULT_SMOOTH = Path(__file__).resolve().parent / "smooth_default.json"


def scored_frames_from_sample(path):
    import pandas as pd
    ids = pd.read_parquet(path, columns=["row_id"])["row_id"].astype(str)
    p = ids.str.split("_", expand=True)
    ep, fr = p[1].astype(int), p[2].astype(int)
    keep = fr != L.EPISODE_FRAME
    out = {}
    for e, g in pd.DataFrame({"e": ep[keep], "f": fr[keep]}).groupby("e"):
        out[int(e)] = np.unique(g["f"].to_numpy())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample")
    ap.add_argument("--val-index")
    ap.add_argument("--smooth-json", default=str(DEFAULT_SMOOTH))
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--no-pen", action="store_true")
    ap.add_argument("--margin", type=float, default=0.002)
    ap.add_argument("--pen-stages", default="hand,object", help="pen_refine stages, applied in order")
    ap.add_argument("--object-if-pen-above", type=float, default=0.0005,
                    help="cm; run the object stage only on episodes whose PEN after the hand stage exceeds this")
    ap.add_argument("--object-lam", type=float, default=100.0, help="Whittaker lambda of the object offset curve")
    ap.add_argument("--object-accept-ratio", type=float, default=0.5,
                    help="keep the object offset only if PEN drops to <= ratio * PEN after the hand stage")
    ap.add_argument("--episodes")
    a = ap.parse_args()
    if a.sample:
        frames = scored_frames_from_sample(a.sample)
    elif a.val_index:
        frames = {e["episode"]: np.asarray(e["frames"]) for e in json.load(open(a.val_index))}
    else:
        raise SystemExit("need --sample or --val-index (scored frames)")
    inp, out = Path(a.inp), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SM.SmoothCfg(**json.load(open(a.smooth_json)))
    hands = None
    eps = sorted(frames) if not a.episodes else [int(x) for x in a.episodes.split(",")]
    for e in eps:
        src = inp / f"episode_{e:06d}.npz"
        if not src.exists():
            print("missing", src); continue
        mesh = next(p for p in (inp / f"episode_{e:06d}_object{s}" for s in (".glb", ".obj", ".ply", ".stl", ".off")) if p.exists())
        ep = L.load_episode(src)
        fr = frames[e]
        rep = {"episode": e, "f0": int(fr[0]), "n_scored": int(len(fr))}
        t0 = time.time()
        if not a.no_smooth:
            ep = SM.smooth_episode(ep, int(fr[0]), cfg)
            rep["smooth"] = SM.cfg_dict(cfg)
        if not a.no_pen:
            import pen_refine as P
            from v2dlb.mesh_budget import budget_mesh
            hands = hands or P.HandPoints(L.MHR_TS)
            ep, prep = P.refine(ep, budget_mesh(mesh, 4096, 4096), fr, hands, margin=a.margin,
                                stages=tuple(a.pen_stages.split(",")), lam=a.object_lam,
                                object_if_pen_above=a.object_if_pen_above,
                                object_accept_ratio=a.object_accept_ratio)
            rep["pen"] = prep
        rep["seconds"] = round(time.time() - t0, 1)
        L.save_episode(out / f"episode_{e:06d}.npz", ep)
        shutil.copyfile(mesh, out / mesh.name)
        json.dump(rep, open(out / f"episode_{e:06d}_post.json", "w"), indent=1)
        print(json.dumps({k: v for k, v in rep.items() if k != "smooth"})[:400], flush=True)


if __name__ == "__main__":
    main()
