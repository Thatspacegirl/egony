#!/usr/bin/env python3
"""Score Track 1 prediction variants on the FORM-HOI val set with the kit metric code (score_t1.FastScorer, identical
numbers to the kit parquet path), in memory, with optional smoothing (smoothing.py configs) -- no PEN refinement
here (postprocess.py does PEN on the frozen candidates; PEN is reported raw).

A "source" is a glob with {E} for the episode directory/name, e.g.
  cari4d_trellis:refined=/mnt/secondary/v2d/t1/cari4d_trellis/val/{E}/t1/refined/{E}.npz
The mesh is the sibling {E}_object.{glb,ply,obj}.

  python t1_val.py --src NAME=GLOB [--src ...] --smooth none,default --episodes 0-12 [--pen] [--json out.json]
     [--hybrid NAME=HUMAN_SRC+OBJECT_SRC]   human fields from one source, object fields + mesh from another
     [--diag]   adds: cdh_pa (per-frame Sim(3): articulation only), oracle object (GT object through the f0 Sim(3):
                the CD-O our human alone allows), f0 human error.  VAL ONLY (reads GT).
Tune split = val episodes 0-12, held-out = 13-24.  Leaderboard rows printed for reference.
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
import t1lib as L  # noqa: E402

KEYS = ("cd_h_cm", "cd_o_cm", "acc_h_cm", "acc_o_cm", "interpenetration_cm")
SHORT = ("CD-H", "CD-O", "ACC-H", "ACC-O", "PEN")
LB = {"ByeByeHelloHaHaHa": (14.205, 31.406, 0.16957, 0.13563, 0.00058), "um": (14.857, 105.26, 0.899, 0.995, 0.0702),
      "RoboPanda": (15.175, 22.087, 0.13902, 0.10591, 0.0017), "Open4D": (15.413, 24.567, 0.17835, 0.19278, 0.00286),
      "Lattice": (15.446, 29.401, 0.17902, 0.11903, 0.0582), "CARI4D baseline": (15.575, 23.162, 1.974, 0.726, 0.00423),
      "SNU CVLAB": (17.213, 28.689, 0.15118, 0.46771, 0.01011)}
HUMAN_KEYS = ("pose", "scales", "shape")
OBJ_KEYS = ("object_rotation", "object_translation", "object_scale")


def parse_eps(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def find(src_glob, ep):
    E = f"episode_{ep:06d}"
    hits = sorted(glob.glob(src_glob.replace("{E}", E)))
    if not hits:
        return None, None
    npz = Path(hits[0])
    mesh = next((p for p in (npz.with_name(f"{E}_object{s}") for s in (".glb", ".ply", ".obj")) if p.exists()), None)
    return npz, mesh


def rank_sum(row):
    """our rank-sum if we were added to the public leaderboard (lower is better for all 5)."""
    tot = 0
    for j in range(5):
        v = row[j]
        tot += 1 + sum(1 for k, r in LB.items() if k != "CARI4D baseline" and r[j] < v)
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", default=[])
    ap.add_argument("--hybrid", action="append", default=[])
    ap.add_argument("--smooth", default="none,default")
    ap.add_argument("--episodes", default="0-12")
    ap.add_argument("--pen", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--ground", action="store_true", help="also score each smoothing variant + ground.refine (feet on floor)")
    ap.add_argument("--json")
    a = ap.parse_args()
    import pandas as pd
    import score_t1 as S
    from smoothing import SmoothCfg, smooth_episode
    fs = S.FastScorer()
    eps = [e for e in parse_eps(a.episodes) if e in fs.index]
    srcs = dict(s.split("=", 1) for s in a.src)
    hyb = {}
    for h in a.hybrid:
        n, spec = h.split("=", 1); hs, os_ = spec.split("+"); hyb[n] = (hs, os_)
    def _cfg(n):   # "none", a name (smooth_<name>.json here) or name=path/to/cfg.json
        if n == "none":
            return n, None
        if "=" in n:
            n, path = n.split("=", 1)
            return n, SmoothCfg(**json.load(open(path)))
        return n, SmoothCfg(**json.load(open(HERE / f"smooth_{n}.json")))
    cfgs = dict(_cfg(n) for n in a.smooth.split(","))
    rows = []
    for ep in eps:
        f0 = int(fs.index[ep]["frames"][0])
        loaded = {}
        for name, g in srcs.items():
            npz, mesh = find(g, ep)
            if npz is not None and mesh is not None:
                loaded[name] = (L.load_episode(npz), mesh)
        for name, (hs, os_) in hyb.items():
            if hs in loaded and os_ in loaded:
                d = dict(loaded[hs][0])
                for k in OBJ_KEYS:
                    d[k] = loaded[os_][0][k]
                loaded[name] = (d, loaded[os_][1])
        for name, (d, mesh) in loaded.items():
            vf = fs.budget(mesh)
            variants = []
            for cn, cfg in cfgs.items():
                e2 = smooth_episode(d, f0, cfg) if cfg is not None else d
                variants.append((cn, e2))
                if a.ground:
                    import ground
                    e3, grep_ = ground.refine(e2, fs.index[ep]["frames"])
                    print(f"   ground ep {ep} {name} {cn}: {json.dumps(grep_)}", flush=True)
                    variants.append((cn + "+gnd", e3))
            for cn, e2 in variants:
                m = fs.score_episode(ep, e2, vf, pen=a.pen)
                r = {"ep": ep, "src": name, "smooth": cn, **m}
                if a.diag:
                    _, _, ref = fs.ref(ep)
                    pred, frames, _ = fs.arrays(ep, e2, vf)
                    cdh = fs.cdh
                    vals = []
                    for i in range(0, len(frames), 5):
                        tr = cdh.fit_similarity_transform(pred["mhr_vertices"][i], ref["mhr_vertices"][i])
                        ph = cdh.apply_similarity_transform(pred["human_surface_points"][i:i + 1], tr)
                        vals.append(float(np.mean(cdh.symmetric_chamfer_distance(ph, ref["human_surface_points"][i:i + 1]))) * 100)
                    r["cdh_pa"] = float(np.mean(vals))
                    r["cdh_f0"] = vals[0]
                rows.append(r)
                print(f"ep {ep:2d} {name:24s} {cn:8s} " + " ".join(f"{s_} {m.get(k, np.nan):8.4f}" for s_, k in zip(SHORT, KEYS) if k in m)
                      + (f"  cdh_pa {r['cdh_pa']:.3f} cdh_f0 {r['cdh_f0']:.3f}" if a.diag else ""), flush=True)
    df = pd.DataFrame(rows)
    if not len(df):
        print("no predictions found"); return
    df["split"] = np.where(df.ep <= 12, "tune", "heldout")
    cols = [k for k in KEYS if k in df] + [c for c in ("cdh_pa", "cdh_f0") if c in df]
    pd.set_option("display.width", 250)
    for split, g in df.groupby("split"):
        n_by = g.groupby(["src", "smooth"]).ep.nunique()
        t = g.groupby(["src", "smooth"])[cols].mean()
        t["n_ep"] = n_by
        if "interpenetration_cm" in t:
            t["rank_sum_vs_LB"] = [rank_sum(list(r[list(KEYS)])) for _, r in t.iterrows()]
        print(f"\n=== {split} ({sorted(g.ep.unique())})\n" + t.round(4).to_string())
    # common-episode table (only episodes every source has)
    common = set.intersection(*[set(g.ep) for _, g in df.groupby("src")])
    if len(df.src.unique()) > 1:
        c = df[df.ep.isin(common)]
        print(f"\n=== common episodes {sorted(common)}\n" + c.groupby(["src", "smooth"])[cols].mean().round(4).to_string())
    print("\nleaderboard:", json.dumps(LB))
    if a.json:
        df.to_json(a.json, orient="records", indent=1)


if __name__ == "__main__":
    main()
