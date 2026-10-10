"""Apply a mesh-source policy (decided on PUBLIC, applied unchanged to evaluation) to the t3_mesh_select.py results of
one split: for every (episode, object) pick TSDF-final or one SAM 3D Objects candidate with a label-free rule, adopt
the SAM 3D picks into meshes/<stage>/<split>/episode_X/<obj>.{ply,json} (t3_mesh_adopt.py) and remove stale files of
that stage for objects that keep the TSDF mesh (FPOSE_STAGE="board <stage> final" then falls through to final).

Rule (same as t3_select_report.py): key(c) = view IoU (- 0.002 x dz_mm for iou_dz); the best SAM 3D candidate must
beat TSDF's key by --min_gain; --classes limits which objects may switch at all (default: every object).
Boards (wooden_piece_*) are left to the board stage and never switched here.

  python3 t3_policy_apply.py --split public --select_root /mnt/secondary/v2d/t3/select/k3 --stage select_k3 \
      [--rule iou] [--min_gain 0.0] [--classes white_pot water_pitcher ...] [--dry_run] [--swaps_out SWAPS.txt]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

R0 = "/mnt/secondary/v2d/t3"
T3 = os.path.dirname(os.path.abspath(__file__))
PY = "/mnt/secondary/v2d/envs/t3-fpose/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--select_root", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--rule", default="iou")
    ap.add_argument("--min_gain", type=float, default=0.0)
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--swaps_out", default=None, help="write EP:OBJ:MESH:JSON lines for t3_rebody.py (public dev)")
    ap.add_argument("--broken_iou", type=float, default=0.0,
                    help="BROKEN-MESH FALLBACK (0 = off): if the TSDF final's view IoU is below this, try (1) the "
                         "episode's own stage1c mesh (cross-episode fusion can misplace a mesh), then (2) the best "
                         "SAM 3D candidate if its IoU >= --s3d_min_iou")
    ap.add_argument("--fallback_gain", type=float, default=0.15)
    ap.add_argument("--s3d_min_iou", type=float, default=0.6)
    a = ap.parse_args()

    def key(c):
        iou = c["iou"]
        if c["name"].startswith("tsdf"):
            # FoundationPose starts the TSDF mesh from its own stage-1 pose, so its view IoU before t3_mesh_select's ICP
            # refinement counts too (the refinement occasionally diverges, e.g. eval 3 pot: 0.56 -> 0.0)
            iou = max(iou, c.get("iou_init", iou))
        dz = c["dz_mm"] if c["dz_mm"] == c["dz_mm"] else 50.0
        return iou - (0.002 * dz if a.rule == "iou_dz" else 0.0)

    swaps, decisions = [], []
    for p in sorted(glob.glob(f"{a.select_root}/{a.split}/episode_*/*.json")):
        if p.endswith(".cands.json"):
            continue
        s = json.load(open(p))
        ep, obj = int(s["episode"]), s["obj"]
        if a.episodes and ep not in a.episodes:
            continue
        if obj.startswith("wooden_piece"):
            continue
        cs = s["cands"]
        tsdf = next((c for c in cs if c["name"].startswith("tsdf")), None)
        gens = [c for c in cs if not c["name"].startswith("tsdf")]
        pick = tsdf
        if gens and (a.classes is None or obj in a.classes):
            g = max(gens, key=key)
            if tsdf is None or key(g) >= key(tsdf) + a.min_gain:
                pick = g
        e = f"episode_{ep:06d}"
        od = f"{R0}/meshes/{a.stage}/{a.split}/{e}"
        fallback = None
        if a.broken_iou > 0 and tsdf is not None and pick is tsdf and key(tsdf) < a.broken_iou:
            s1c = f"{R0}/meshes/stage1c/{a.split}/{e}/{obj}"
            fin = f"{R0}/meshes/final/{a.split}/{e}/{obj}"
            iou_s1c = -1.0
            if os.path.exists(s1c + ".ply") and open(s1c + ".ply", "rb").read() != open(fin + ".ply", "rb").read():
                cj = f"{a.select_root}/{a.split}/{e}/{obj}.s1c.cands.json"
                json.dump([dict(name="tsdf_stage1c", mesh=s1c + ".ply", json=s1c + ".json")], open(cj, "w"))
                oj = f"{a.select_root}/{a.split}/{e}/{obj}.s1c.json"
                if not os.path.exists(oj):
                    subprocess.run([PY, "-I", f"{T3}/t3_mesh_select.py", "--split", a.split, "--episode", str(ep),
                                    "--obj", obj, "--cands", cj, "--out", oj], capture_output=True, text=True,
                                   env=dict(os.environ, OMP_NUM_THREADS="2"))
                if os.path.exists(oj):
                    c1 = json.load(open(oj))["cands"][0]
                    iou_s1c = max(c1["iou"], c1.get("iou_init", c1["iou"]))
            if iou_s1c >= a.broken_iou and iou_s1c >= key(tsdf) + a.fallback_gain:
                fallback = ("stage1c", s1c, iou_s1c)
            elif gens and max(gens, key=key)["iou"] >= a.s3d_min_iou:
                pick = max(gens, key=key)
                fallback = ("s3d", pick["name"], pick["iou"])
        d = dict(ep=ep, obj=obj, pick=pick["name"] if pick else None, iou=pick["iou"] if pick else None,
                 tsdf_iou=tsdf["iou"] if tsdf else None, tsdf_key=round(key(tsdf), 4) if tsdf else None,
                 fallback=fallback)
        decisions.append(d)
        print(json.dumps(d), flush=True)
        if a.dry_run:
            continue
        if fallback and fallback[0] == "stage1c":
            import shutil
            os.makedirs(od, exist_ok=True)
            for ext in (".ply", ".json"):
                shutil.copy(fallback[1] + ext, f"{od}/{obj}.tmp{ext}")
                os.replace(f"{od}/{obj}.tmp{ext}", f"{od}/{obj}{ext}")
            swaps.append(f"{ep}:{obj}:{od}/{obj}.ply:{od}/{obj}.json")
            continue
        if pick is None or pick is tsdf:
            for ext in (".ply", ".json"):
                if os.path.exists(f"{od}/{obj}{ext}"):
                    os.remove(f"{od}/{obj}{ext}")
            continue
        r = subprocess.run([PY, "-I", f"{T3}/t3_mesh_adopt.py", "--select", p, "--cand", pick["name"], "--stage", a.stage],
                           capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="2"))
        if r.returncode != 0:
            print(f"ADOPT FAILED {e} {obj}: {r.stderr[-500:]}", file=sys.stderr, flush=True)
            continue
        swaps.append(f"{ep}:{obj}:{od}/{obj}.ply:{od}/{obj}.json")
    if a.swaps_out and not a.dry_run:
        with open(a.swaps_out + ".tmp", "w") as f:
            f.write("\n".join(swaps) + ("\n" if swaps else ""))
        os.replace(a.swaps_out + ".tmp", a.swaps_out)
    n_s3d = sum(1 for d in decisions if d["pick"] and not d["pick"].startswith("tsdf"))
    n_fb = sum(1 for d in decisions if d["fallback"])
    print(f"{len(decisions)} objects, {n_s3d} switched to SAM 3D, {n_fb} broken-mesh fallbacks", flush=True)


if __name__ == "__main__":
    main()
