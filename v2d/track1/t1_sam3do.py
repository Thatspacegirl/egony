#!/usr/bin/env python3
"""Object mesh from the episode video with SAM 3D Objects (facebook/sam-3d-objects, SAM License; code
facebookresearch/sam-3d-objects @ f91db41) through the toolkit's own wrapper v2d.sam3d.lib.image_to_mesh, the way
the toolkit's pipelines call it (run_hand_masks.py / run_v2d_ego_e2e.py: with_mesh_postprocess, metric point map,
then the SAM3D rotation*scale baked into the vertices so the mesh is metric).  Track 2 meshes are never read.

  prep (env t1-gen or any env with cv2+av): pick the reference frame from our SAM3 masks (t1_trellis.pick_frames:
        unoccluded by the actor, high SAM3 score, large, not touching the border), write <out>/image.png (full-res
        RGB), <out>/mask.png (full-res object mask), <out>/frame.json
  [queue] t1_moge.py points --frame F --out <out>/moge   (MoGe-2 metric point map of that frame, free FoV)
  gen  (env t1-sam3do): image_to_mesh(image, mask, pointmap=<out>/moge/points.npy, with_mesh_postprocess=True)
        -> <out>/sam3d_raw.glb + transform.json + intrinsics.json; then R*s baked -> <out>/mesh.glb (metres)
Metric scale is refined later against CARI4D's human-aligned depth (t1_cari4d.py run, FoundationPose scale search).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_items as I  # noqa: E402


def cmd_prep(a):
    import cv2
    from t1_trellis import pick_frames, read_frame, unpack
    it = I.item(a.split, a.episode)
    M = dict(np.load(Path(a.masks) / "masks.npz"))
    W = int(M["W"]); FW, FH = [int(x) for x in M["full_wh"]]
    chosen, st = pick_frames(M, 1)
    fi = chosen[0]
    vf = int(M["frames"][fi])
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    rgb = read_frame(it["video"], vf)
    om = unpack(M["object"][fi], W).astype(np.uint8) * 255
    om = (cv2.resize(om, (FW, FH), interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8) * 255
    cv2.imwrite(str(od / "image.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(od / "mask.png"), om)
    rec = {"split": a.split, "episode": a.episode, "object": it["object"], "video": it["video"], "video_frame": vf,
           "mask_index": int(fi), "area_px_s05": float(st["area"][fi]), "occ": float(st["occ"][fi]),
           "score": float(M["obj_score"][fi])}
    json.dump(rec, open(od / "frame.json", "w"), indent=1)
    print(json.dumps(rec), flush=True)


def bake(raw_glb, transform_json, out_glb):
    """toolkit run_hand_masks._apply_sam3d_transform: vertices <- R @ diag(s) @ v (no translation)."""
    import trimesh
    t = json.load(open(transform_json))
    qw, qx, qy, qz = t["rotation"]
    sx, sy, sz = t["scale"]
    Rm = np.array([[1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
                   [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
                   [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy]])
    RS = Rm @ np.diag([sx, sy, sz])
    scene = trimesh.load(raw_glb)
    if isinstance(scene, trimesh.Scene):
        meshes = list(scene.dump()) if hasattr(scene, "dump") else list(scene.geometry.values())
        for m in meshes:
            m.vertices = (RS @ np.asarray(m.vertices).T).T
        result = trimesh.util.concatenate(meshes)
    else:
        scene.vertices = (RS @ np.asarray(scene.vertices).T).T
        result = scene
    tmp = Path(out_glb).with_suffix(".tmp.glb")
    result.export(tmp)
    os.replace(tmp, out_glb)
    return result, [sx, sy, sz]


def cmd_gen(a):
    import torch
    od = Path(a.out)
    rec = json.load(open(od / "frame.json"))
    t0 = time.time()
    from v2d.sam3d.lib.image_to_mesh import image_to_mesh
    kw = {}
    if (od / "moge" / "points.npy").is_file():
        # like the toolkit's depth_mask_path route (run_hand_masks.py): only object pixels condition the layout
        import cv2
        P = np.load(od / "moge" / "points.npy")
        m = cv2.imread(str(od / "mask.png"), 0) > 0
        me = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        if me.sum() >= 200:
            m = me
        P = P.copy(); P[~m] = np.nan
        np.save(od / "moge" / "points_obj.npy", P.astype(np.float32))
        rec["pointmap_obj_px"] = int(np.isfinite(P[..., 2]).sum())
        rec["pointmap_obj_median_z"] = float(np.nanmedian(P[..., 2]))
        kw = {"pointmap_path": str(od / "moge" / "points_obj.npy"), "pointmap_intrinsics_path": str(od / "moge" / "intrinsics.json")}
    torch.manual_seed(a.seed)
    image_to_mesh(str(od / "image.png"), str(od / "mask.png"), str(od / "sam3d_raw.glb"), str(od / "transform.json"),
                  str(od / "intrinsics.json"), os.environ.get("SAM3DO_WEIGHTS", "/mnt/secondary/v2d/weights/sam3d_objects"),
                  seed=a.seed, with_mesh_postprocess=True, with_texture_baking=a.texture, use_vertex_color=not a.texture, **kw)
    m, s = bake(od / "sam3d_raw.glb", od / "transform.json", od / "mesh.glb")
    rec.update({"seconds": round(time.time() - t0, 1), "pointmap": bool(kw), "sam3d_scale": s,
                "extents_m": [float(x) for x in m.extents], "faces": int(len(m.faces)), "watertight": bool(m.is_watertight),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None})
    json.dump(rec, open(od / "sam3do.json", "w"), indent=1)
    print(json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "gen"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--masks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--texture", action="store_true", help="texture baking (gsplat) instead of vertex colours")
    a = ap.parse_args()
    {"prep": cmd_prep, "gen": cmd_gen}[a.cmd](a)


if __name__ == "__main__":
    main()
