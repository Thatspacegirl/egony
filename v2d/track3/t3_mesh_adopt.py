"""Turn a scored candidate mesh (t3_mesh_select.py output) into a stage-style mesh for FoundationPose + assembly:
meshes/<stage>/<split>/episode_X/<obj>.{ply,json}, mesh bbox-centred in its own body frame, decimated to <= 4000
faces (packer budget 4096 is then a no-op), JSON = the stage-1 JSON (window frames, VO file, ...) with frame_T_obj
replaced by the candidate's refined VO-world pose of the new body frame.

  python -I t3_mesh_adopt.py --select SEL.json --cand NAME --stage sam3do        (or --best: highest view IoU)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
R0 = "/mnt/secondary/v2d/t3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True)
    ap.add_argument("--cand", default=None)
    ap.add_argument("--best", action="store_true", help="highest mean view IoU (generated must beat TSDF by --min_gain)")
    ap.add_argument("--min_gain", type=float, default=0.0)
    ap.add_argument("--stage", default="sam3do")
    ap.add_argument("--max_faces", type=int, default=4000)
    a = ap.parse_args()
    import trimesh
    from t3_mesh_complete import decimate
    from t3_mesh_select import Episode
    sel = json.load(open(a.select))
    cands = sel["cands"]
    if a.best:
        c = max(cands, key=lambda r: r["iou"])
        tsdf = next((r for r in cands if r["name"].startswith("tsdf")), None)
        if tsdf is not None and c is not tsdf and c["iou"] < tsdf["iou"] + a.min_gain:
            c = tsdf
    else:
        c = next(r for r in cands if r["name"] == a.cand)
    ep = Episode(sel["split"], sel["episode"], sel["obj"])
    m = trimesh.load(c["mesh"], force="mesh", process=False)
    A = np.array(c["world_T_mesh_sim"])  # similarity: world = A @ mesh
    sc = float(np.cbrt(np.linalg.det(A[:3, :3])))
    Tw = np.eye(4)
    Tw[:3, :3] = A[:3, :3] / sc
    Tw[:3, 3] = A[:3, 3]
    U, _, Vt = np.linalg.svd(Tw[:3, :3])
    Tw[:3, :3] = U @ Vt
    V = np.asarray(m.vertices, float) * sc  # scale baked into the body mesh; Tw is rigid
    lo, hi = V.min(0), V.max(0)
    ctr = (lo + hi) / 2
    body = trimesh.Trimesh(V - ctr, np.asarray(m.faces), vertex_colors=getattr(m.visual, "vertex_colors", None),
                           process=False)
    body = decimate(body, a.max_faces)
    # re-centre after decimation (bbox may shift slightly)
    Vb = np.asarray(body.vertices)
    c2 = (Vb.max(0) + Vb.min(0)) / 2
    body.vertices = Vb - c2
    Tc = np.eye(4)
    Tc[:3, 3] = ctr + c2
    s1 = json.load(open(f"{R0}/meshes/stage1/{sel['split']}/{ep.e}/{sel['obj']}.json"))
    s1 = dict(s1)
    s1["frame_T_obj"] = (Tw @ Tc).tolist()
    s1["source"] = dict(select=os.path.abspath(a.select), cand=c["name"], mesh=c["mesh"], iou=c["iou"], dz_mm=c["dz_mm"],
                        obs_med_mm=c["obs_med_mm"], scale_applied=sc)
    od = f"{R0}/meshes/{a.stage}/{sel['split']}/{ep.e}"
    os.makedirs(od, exist_ok=True)
    body.export(f"{od}/{sel['obj']}.tmp.ply")
    os.replace(f"{od}/{sel['obj']}.tmp.ply", f"{od}/{sel['obj']}.ply")
    json.dump(s1, open(f"{od}/{sel['obj']}.json.tmp", "w"), indent=1)
    os.replace(f"{od}/{sel['obj']}.json.tmp", f"{od}/{sel['obj']}.json")
    print(json.dumps(dict(out=f"{od}/{sel['obj']}.ply", cand=c["name"], faces=len(body.faces),
                          extent_cm=(np.ptp(np.asarray(body.vertices), 0) * 100).round(1).tolist())))


if __name__ == "__main__":
    main()
