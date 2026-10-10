#!/usr/bin/env python3
"""Object mesh from the video itself: TRELLIS-image-large (microsoft/TRELLIS-image-large, MIT, ungated) on the best
SAM3-masked frame(s) of the episode.  Track 2 meshes are never read.

Env: /mnt/secondary/v2d/envs/t1-gen (source v2d_env.sh). GPU, under the shared lock.
Frame choice (from <masks>/masks.npz, t1_masks.py): object mask area >= 60% of the clip's 95th-percentile area,
not touching the image border, smallest overlap with the dilated actor mask, highest SAM3 score; ties -> earlier
(pre-contact frames come first in the window).  --n-views K picks K well-separated frames (each -> one mesh).

Output <out>/: view{k}_rgba.png (TRELLIS input), mesh_raw_{k}.ply (TRELLIS canonical frame, unit cube),
mesh_{k}.ply (largest components, <= --faces faces), trellis.json (frames, scores, timing).
Metric scale / pose: t1_fpose.py.
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


def unpack(bits, W):
    return np.unpackbits(bits, axis=-1, count=W).astype(bool)


def frame_scores(M, dil_frac=0.02):
    import cv2
    W, H = int(M["W"]), int(M["H"])
    n = len(M["frames"])
    rad = max(2, int(dil_frac * W))
    ker = np.ones((2 * rad + 1, 2 * rad + 1), np.uint8)
    area = np.zeros(n); border = np.zeros(n, bool); occ = np.zeros(n)
    for f in range(n):
        om = unpack(M["object"][f], W)
        a = om.sum()
        area[f] = a
        if a == 0:
            continue
        ys, xs = np.nonzero(om)
        border[f] = xs.min() <= 1 or ys.min() <= 1 or xs.max() >= W - 2 or ys.max() >= H - 2
        hm = unpack(M["human"][f], W)
        if hm.any():
            occ[f] = (cv2.dilate(hm.astype(np.uint8), ker) > 0)[om].mean()
    return area, border, occ


def pick_frames(M, k):
    area, border, occ = frame_scores(M)
    ok = area > 0
    if not ok.any():
        raise RuntimeError("no object mask in the window")
    a95 = np.percentile(area[ok], 95)
    good = ok & (area >= 0.6 * a95) & ~border
    if not good.any():
        good = ok & (area >= 0.3 * a95)
    sc = M["obj_score"].astype(float)
    # lexicographic: low occlusion, then high score, then larger area
    cost = occ * 10 + (1 - sc) + 0.2 * (1 - area / a95)
    cost[~good] = np.inf
    order = np.argsort(cost, kind="stable")
    chosen = []
    for f in order:
        if not np.isfinite(cost[f]):
            break
        if all(abs(int(M["frames"][f]) - int(M["frames"][c])) >= 60 for c in chosen):
            chosen.append(int(f))
        if len(chosen) == k:
            break
    return chosen, {"area": area, "occ": occ, "border": border, "a95": float(a95)}


def read_frame(video, n):
    import av
    with av.open(str(video)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i == n:
                return fr.to_ndarray(format="rgb24")
    raise IndexError(n)


def clean_mesh(v, f, max_faces, colors=None):
    """largest components, <= max_faces faces; vertex colors (uint8 RGBA) carried over by nearest raw vertex."""
    import fast_simplification
    import trimesh
    from scipy.spatial import cKDTree
    m = trimesh.Trimesh(v, f, process=True)
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        areas = np.array([p.area for p in parts])
        keep = [p for p, a in zip(parts, areas) if a >= 0.05 * areas.max()]
        m = trimesh.util.concatenate(keep)
    if len(m.faces) > max_faces:
        vv, ff = fast_simplification.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int32),
                                              target_reduction=1 - max_faces / len(m.faces))
        m = trimesh.Trimesh(vv, ff, process=True)
    if colors is not None:
        _, nn = cKDTree(v).query(m.vertices)
        c = np.clip(colors[nn] * 255, 0, 255).astype(np.uint8)
        m.visual.vertex_colors = np.concatenate([c, np.full((len(c), 1), 255, np.uint8)], 1)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-views", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--faces", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=25)
    a = ap.parse_args()
    import cv2
    import torch
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    it = I.item(a.split, a.episode)
    M = dict(np.load(Path(a.masks) / "masks.npz"))
    W = int(M["W"])
    FW, FH = [int(x) for x in M["full_wh"]]
    chosen, st = pick_frames(M, a.n_views)
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()
    t_load = time.time() - t0
    rec = {"episode": a.episode, "split": a.split, "object": it["object"], "views": [], "load_s": round(t_load, 1)}
    for k, fi in enumerate(chosen):
        vf = int(M["frames"][fi])
        rgb = read_frame(it["video"], vf)
        om = unpack(M["object"][fi], W).astype(np.uint8) * 255
        om = cv2.resize(om, (FW, FH), interpolation=cv2.INTER_LINEAR)
        rgba = np.dstack([rgb, om])
        Image.fromarray(rgba).save(od / f"view{k}_rgba.png")
        t1 = time.time()
        with torch.no_grad():
            out = pipe.run(Image.fromarray(rgba), seed=a.seed, formats=["mesh"], preprocess_image=True,
                           sparse_structure_sampler_params={"steps": a.steps, "cfg_strength": 7.5},
                           slat_sampler_params={"steps": a.steps, "cfg_strength": 3.0})
        mesh = out["mesh"][0]
        v = mesh.vertices.detach().float().cpu().numpy()
        f = mesh.faces.detach().cpu().numpy().astype(np.int64)
        col = None
        if getattr(mesh, "vertex_attrs", None) is not None:
            col = mesh.vertex_attrs[:, :3].detach().float().clamp(0, 1).cpu().numpy()
        import trimesh
        raw = trimesh.Trimesh(v, f, process=False)
        if col is not None:
            raw.visual.vertex_colors = np.concatenate([(col * 255).astype(np.uint8), np.full((len(col), 1), 255, np.uint8)], 1)
        raw.export(od / f"mesh_raw_{k}.ply")
        m = clean_mesh(v, f, a.faces, col)
        tmp = od / f"mesh_{k}.tmp.ply"
        m.export(tmp)
        os.replace(tmp, od / f"mesh_{k}.ply")
        rec["views"].append({"k": k, "video_frame": vf, "mask_index": int(fi), "area_px": float(st["area"][fi]),
                             "a95": st["a95"], "occ": float(st["occ"][fi]), "score": float(M["obj_score"][fi]),
                             "raw_faces": int(len(f)), "faces": int(len(m.faces)), "watertight": bool(m.is_watertight),
                             "extent": m.extents.tolist(), "gen_s": round(time.time() - t1, 1)})
        print(json.dumps(rec["views"][-1]), flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    json.dump(rec, open(od / "trellis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
