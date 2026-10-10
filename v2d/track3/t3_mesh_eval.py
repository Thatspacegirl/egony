"""DEV-ONLY: CD-O of OUR meshes against the public scans with the official Track 3 CD-O core (PUBLIC episodes only).

For every mesh MESH_ROOT/public/episode_X/<obj>.(ply|obj|glb) with a public scan: budget_mesh(4096, 4096) exactly as
the packer does, then CD.._object_chamfer_cm(reference=budget(scan), candidate, seed=_object_seed(ep, slot)).
Also prints the RMS-radius ratio (our/scan; the scorer normalises it away for CD-O but it is a direct check of the
metric scale when the mesh is complete) and the mesh extents.  wooden_spoon (no scan) is reported with dims only.

  /mnt/secondary/v2d/scratch/venv/bin/python -I t3_mesh_eval.py --root /mnt/secondary/v2d/t3/meshes/stage1 \
      [--episodes 12 13] [--out J.json]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

KIT = "/mnt/secondary/v2d/kit/v2d_submission_kit"
DS = os.path.expanduser("~/TestingGrounds/egony/video_to_data_challenge/track_3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir with public/episode_X/<obj>.<ext>")
    ap.add_argument("--candidates", action="store_true",
                    help="layout ROOT/public/episode_X/<obj>/<name>.ply (t3_sam3do.py output): score every file")
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.argv = [sys.argv[0], KIT, DS]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import devscore as D
    from v2dlb.mesh_budget import budget_mesh
    res = []
    for d in sorted(glob.glob(f"{a.root}/public/episode_*")):
        ep = int(d[-6:])
        if a.episodes and ep not in a.episodes:
            continue
        meta = json.load(open(f"/mnt/secondary/v2d/t3/frames/public/episode_{ep:06d}/meta.json"))
        for b, o in enumerate(meta["objects"]):
            if a.candidates:
                plist = sorted(glob.glob(f"{d}/{o}/*.ply"))
            else:
                plist = [x for x in (f"{d}/{o}.{e}" for e in ("ply", "obj", "glb")) if os.path.exists(x)][:1]
            for pp in plist:
              v, f = budget_mesh(pp, 4096, 4096)
              r = dict(episode=ep, slot=b, obj=o, mesh=pp, extent_cm=(np.ptp(v, 0) * 100).round(1).tolist())
              try:
                  g = D.ref_geometry(o)
                  r["cdo_cm"] = float(D.CDO._object_chamfer_cm((g["v"], g["f"]), (v, f), D.CDO._object_seed(ep, b)))
                  r["rms_ratio"] = float(D.CDO._surface_moments(v, f)[1] / D.CDO._surface_moments(g["v"], g["f"])[1])
                  r["scan_extent_cm"] = (np.ptp(g["v"], 0) * 100).round(1).tolist()
              except FileNotFoundError:
                  pass
              print(json.dumps(r), flush=True)
              res.append(r)
    c = [r["cdo_cm"] for r in res if "cdo_cm" in r]
    print(f"MEAN CD-O over {len(c)} pairs: {np.mean(c):.4f} cm" if c else "no scored pairs")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
