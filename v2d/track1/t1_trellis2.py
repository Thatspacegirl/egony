#!/usr/bin/env python3
"""Object mesh from the episode video with TRELLIS.2 (microsoft/TRELLIS.2-4B, MIT; DINOv3 image features
facebook/dinov3-vitl16-pretrain-lvd1689m, DINOv3 License; its BiRefNet background model briaai/RMBG-2.0 is loaded by
the pipeline but NOT used, because we pass our own object mask as the alpha channel).  Track 2 meshes are never read.

Env: /mnt/secondary/v2d/envs/t1-trellis2 (source v2d_env.sh).  GPU, under the shared lock.
Frame choice: identical to t1_trellis.py (pick_frames on <masks>/masks.npz: unoccluded by the actor, large, not on
the border, high score).  Output <out>/: view0_rgba.png, mesh_raw_0.ply (TRELLIS.2 canonical frame, unit cube,
vertex colours = queried base colour), mesh_0.ply (largest components, <= --faces faces), trellis2.json.
Metric scale/pose: like TRELLIS v1 (t1_cari4d.py scale hook: silhouette s0 + FoundationPose scale grid search).

  python t1_trellis2.py --split val --episode 11 --masks /mnt/secondary/v2d/t1/masks_fh/val/episode_000011 \
      --out /mnt/secondary/v2d/t1/mesh_trellis2_fh/val/episode_000011 [--pipeline-type 512|1024_cascade]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_items as I  # noqa: E402
from t1_trellis import clean_mesh, pick_frames, read_frame, unpack  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--faces", type=int, default=20000)
    ap.add_argument("--pipeline-type", default="512", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    a = ap.parse_args()
    import cv2
    import torch
    import trimesh
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    it = I.item(a.split, a.episode)
    M = dict(np.load(Path(a.masks) / "masks.npz"))
    W = int(M["W"]); FW, FH = [int(x) for x in M["full_wh"]]
    chosen, st = pick_frames(M, 1)
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()
    t_load = time.time() - t0
    rec = {"episode": a.episode, "split": a.split, "object": it["object"], "pipeline_type": a.pipeline_type,
           "views": [], "load_s": round(t_load, 1), "masks": str(Path(a.masks).resolve())}
    fi = chosen[0]
    vf = int(M["frames"][fi])
    rgb = read_frame(it["video"], vf)
    om = unpack(M["object"][fi], W).astype(np.uint8) * 255
    om = cv2.resize(om, (FW, FH), interpolation=cv2.INTER_LINEAR)
    rgba = np.dstack([rgb, om])
    Image.fromarray(rgba).save(od / "view0_rgba.png")
    t1 = time.time()
    with torch.no_grad():
        mesh = pipe.run(Image.fromarray(rgba), seed=a.seed, pipeline_type=a.pipeline_type)[0]
        col = mesh.query_vertex_attrs()[:, pipe.pbr_attr_layout["base_color"]].float().clamp(0, 1).cpu().numpy()
    v = mesh.vertices.detach().float().cpu().numpy()
    f = mesh.faces.detach().cpu().numpy().astype(np.int64)
    peak = torch.cuda.max_memory_allocated() / 2**30
    raw = trimesh.Trimesh(v, f, process=False)
    raw.visual.vertex_colors = np.concatenate([(col * 255).astype(np.uint8), np.full((len(col), 1), 255, np.uint8)], 1)
    raw.export(od / "mesh_raw_0.ply")
    m = clean_mesh(v, f, a.faces, col)
    tmp = od / "mesh_0.tmp.ply"
    m.export(tmp)
    os.replace(tmp, od / "mesh_0.ply")
    rec["views"].append({"k": 0, "video_frame": vf, "mask_index": int(fi), "area_px": float(st["area"][fi]), "a95": st["a95"],
                         "occ": float(st["occ"][fi]), "score": float(M["obj_score"][fi]), "raw_faces": int(len(f)),
                         "faces": int(len(m.faces)), "watertight": bool(m.is_watertight), "extent": m.extents.tolist(),
                         "gen_s": round(time.time() - t1, 1), "peak_alloc_gb": round(peak, 2)})
    print(json.dumps(rec["views"][-1]), flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    tmpj = od / "trellis2.json.tmp"
    json.dump(rec, open(tmpj, "w"), indent=1)
    os.replace(tmpj, od / "trellis2.json")


if __name__ == "__main__":
    main()
