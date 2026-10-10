"""DEV (public): join t3_mesh_select.py results with CD-O of every candidate (t3_mesh_eval.py JSON outputs) and report
how a label-free selection rule would do vs always-TSDF and vs the oracle (best CD-O candidate).

  python3 t3_select_report.py --select_root /mnt/secondary/v2d/t3/select/k3 --cdo CDO_TSDF.json CDO_SAM3D.json \
      [--rule iou|iou_dz] [--min_gain 0.0]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select_root", required=True)
    ap.add_argument("--cdo", nargs="+", required=True, help="t3_mesh_eval.py --out JSONs (any mix)")
    ap.add_argument("--split", default="public")
    ap.add_argument("--rule", default="iou")
    ap.add_argument("--min_gain", type=float, default=0.0, help="SAM3D must beat TSDF's view IoU by this much")
    a = ap.parse_args()
    cdo = {}
    for p in a.cdo:
        for r in json.load(open(p)):
            if "cdo_cm" in r:
                cdo[os.path.realpath(r["mesh"])] = r["cdo_cm"]
    rows = []
    per_obj = defaultdict(list)
    for p in sorted(glob.glob(f"{a.select_root}/{a.split}/episode_*/*.json")):
        if p.endswith(".cands.json"):
            continue
        s = json.load(open(p))
        cs = s["cands"]
        for c in cs:
            c["cdo"] = cdo.get(os.path.realpath(c["mesh"]), np.nan)
        tsdf = next((c for c in cs if c["name"].startswith("tsdf")), None)
        gens = [c for c in cs if not c["name"].startswith("tsdf")]

        def key(c):
            iou = c["iou"]
            if c["name"].startswith("tsdf"):  # same as t3_policy_apply.py: the unrefined stage-1 pose counts too
                iou = max(iou, c.get("iou_init", iou))
            dz = c["dz_mm"] if c["dz_mm"] == c["dz_mm"] else 50.0
            return iou - (0.002 * dz if a.rule == "iou_dz" else 0.0)
        pick = max(cs, key=key)
        if tsdf is not None and pick is not tsdf and key(pick) < key(tsdf) + a.min_gain:
            pick = tsdf
        best_gen = max(gens, key=key) if gens else None
        oracle = min(cs, key=lambda c: c["cdo"] if np.isfinite(c["cdo"]) else 1e9)
        row = dict(ep=s["episode"], obj=s["obj"], tsdf_cdo=tsdf["cdo"] if tsdf else np.nan,
                   tsdf_iou=tsdf["iou"] if tsdf else np.nan, gen_best_cdo=best_gen["cdo"] if best_gen else np.nan,
                   gen_best_iou=best_gen["iou"] if best_gen else np.nan,
                   gen_min_cdo=min((c["cdo"] for c in gens), default=np.nan),
                   pick=pick["name"], pick_cdo=pick["cdo"], oracle=oracle["name"], oracle_cdo=oracle["cdo"])
        rows.append(row)
        per_obj[s["obj"]].append(row)
        print(f"{row['ep']:3d} {row['obj']:18s} tsdf {row['tsdf_cdo']:.3f} (iou {row['tsdf_iou']:.3f}) | s3d-by-iou "
              f"{row['gen_best_cdo']:.3f} (iou {row['gen_best_iou']:.3f}) s3d-min {row['gen_min_cdo']:.3f} | pick "
              f"{row['pick']:22s} {row['pick_cdo']:.3f} | oracle {row['oracle_cdo']:.3f}")
    for k in ("tsdf_cdo", "gen_best_cdo", "gen_min_cdo", "pick_cdo", "oracle_cdo"):
        v = np.array([r[k] for r in rows], float)
        print(f"MEAN {k:13s} {np.nanmean(v):.4f} over {np.isfinite(v).sum()}")
    print("per object (mean tsdf / s3d-by-iou / pick):")
    for o, rr in sorted(per_obj.items()):
        print(f"  {o:18s} n={len(rr)}  {np.nanmean([r['tsdf_cdo'] for r in rr]):.3f} / "
              f"{np.nanmean([r['gen_best_cdo'] for r in rr]):.3f} / {np.nanmean([r['pick_cdo'] for r in rr]):.3f}")


if __name__ == "__main__":
    main()
