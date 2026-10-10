#!/usr/bin/env python3
"""MoGe-2 (Ruicheng/moge-2-vitl-normal, MIT, ungated) for Track 1: camera focal estimate and metric depth.

Env: /mnt/secondary/v2d/envs/t1-gen (source v2d_env.sh). GPU, under the shared lock.

  focal  : run MoGe-2 with free FoV on --n evenly spaced frames of a static-camera video; K = median focal (px),
           principal point = image centre (MoGe's model).  Writes <out>/K.json:
             {"K": 3x3, "fx_px": [...per frame...], "fx_med", "fx_mad", "frames": [...], "W", "H"}
  depth  : run MoGe-2 with the FoV fixed to K.json on the frames [--start, --end) (step --stride) and write
           <out>/depth_s{scale}.npy (N, H*scale, W*scale) float16 metres (inf -> 0 = invalid), <out>/depth_frames.npy,
           <out>/depth_info.json (per-frame median depth, time).  Atomic writes (tmp + rename).

Only the episode's own video is read (no calibration from any dataset).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

MODEL = "Ruicheng/moge-2-vitl-normal"


def read_frames(video, idx):
    import av
    want = set(int(i) for i in idx)
    out = {}
    with av.open(str(video)) as c:
        for n, fr in enumerate(c.decode(video=0)):
            if n in want:
                out[n] = fr.to_ndarray(format="rgb24")
            if n >= max(want):
                break
    return [out[i] for i in idx]


def iter_frames(video, start, end, stride):
    import av
    with av.open(str(video)) as c:
        for n, fr in enumerate(c.decode(video=0)):
            if n >= end:
                break
            if n >= start and (n - start) % stride == 0:
                yield n, fr.to_ndarray(format="rgb24")


def n_frames(video):
    import av
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        n = s.frames
        W, H = s.codec_context.width, s.codec_context.height
    return int(n), int(W), int(H)


def load_model():
    import torch
    from moge.model.v2 import MoGeModel
    dev = os.environ.get("T1_DEVICE", "cuda")
    return MoGeModel.from_pretrained(MODEL).to(dev).eval(), torch


def atomic_json(obj, path):
    tmp = str(path) + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def cmd_focal(a):
    model, torch = load_model()
    T, W, H = n_frames(a.video)
    idx = np.linspace(0, T - 1, a.n + 2)[1:-1].round().astype(int)
    fx, fy, t0 = [], [], time.time()
    for im in read_frames(a.video, idx):
        x = torch.from_numpy(im).to(model.device).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            o = model.infer(x, resolution_level=a.res_level)
        K = o["intrinsics"].float().cpu().numpy()
        fx.append(float(K[0, 0] * W)); fy.append(float(K[1, 1] * H))
    fx, fy = np.array(fx), np.array(fy)
    f = float(np.median(np.concatenate([fx, fy])))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rec = {"video": str(a.video), "model": MODEL, "W": W, "H": H, "T": T, "frames": idx.tolist(),
           "fx_px": fx.round(2).tolist(), "fy_px": fy.round(2).tolist(), "f_med": f,
           "f_mad": float(np.median(np.abs(np.concatenate([fx, fy]) - f))),
           "K": [[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]], "seconds": round(time.time() - t0, 1)}
    atomic_json(rec, out / "K.json")
    print(json.dumps({k: rec[k] for k in ("f_med", "f_mad", "seconds")}), flush=True)


def cmd_depth(a):
    import cv2
    model, torch = load_model()
    T, W, H = n_frames(a.video)
    K = np.array(json.load(open(a.K))["K"], float)
    fov_x = float(np.degrees(2 * np.arctan(W / (2 * K[0, 0]))))
    end = T if a.end is None or a.end <= 0 else min(a.end, T)
    frames = np.arange(a.start, end, a.stride)
    h, w = int(round(H * a.scale)), int(round(W * a.scale))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"s{a.scale:g}"
    dpath, tmp = out / f"depth_{tag}.npy", out / f".depth_{tag}.tmp.npy"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(frames), h, w))
    med, t0, k = [], time.time(), 0
    batch, bidx = [], []

    def flush():
        nonlocal k
        x = torch.stack(batch).to(model.device).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            o = model.infer(x, resolution_level=a.res_level, fov_x=fov_x)
        d = o["depth"].float()
        d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
        d = torch.nn.functional.interpolate(d[:, None], size=(h, w), mode="nearest")[:, 0].cpu().numpy()
        for j in range(len(batch)):
            arr[k] = d[j].astype(np.float16)
            v = d[j][d[j] > 0]
            med.append(float(np.median(v)) if v.size else 0.0)
            k += 1
        batch.clear(); bidx.clear()
    for n, im in iter_frames(a.video, a.start, end, a.stride):
        batch.append(torch.from_numpy(im)); bidx.append(n)
        if len(batch) == a.batch:
            flush()
    if batch:
        flush()
    assert k == len(frames), (k, len(frames))
    arr.flush(); del arr
    os.replace(tmp, dpath)
    np.save(out / "depth_frames.npy", frames)
    atomic_json({"video": str(a.video), "model": MODEL, "K": K.tolist(), "fov_x_deg": fov_x, "scale": a.scale,
                 "shape": [len(frames), h, w], "start": a.start, "end": int(end), "stride": a.stride,
                 "median_depth": np.round(med, 4).tolist(), "seconds": round(time.time() - t0, 1),
                 "sec_per_frame": round((time.time() - t0) / max(len(frames), 1), 4)}, out / "depth_info.json")
    print(f"depth {dpath} {len(frames)} frames {(time.time() - t0) / max(len(frames), 1):.3f} s/frame", flush=True)


def cmd_points(a):
    """Metric point map (OpenCV camera frame, metres) of ONE frame with free FoV (MoGe-2's own focal), for
    SAM-3D-Objects' layout.  Writes <out>/points.npy (H,W,3) float32 (NaN invalid) + intrinsics.json (v2d_common
    CameraIntrinsics: fx, fy, cx, cy, width, height)."""
    model, torch = load_model()
    T, W, H = n_frames(a.video)
    im = read_frames(a.video, [a.frame])[0]
    x = torch.from_numpy(im).to(model.device).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        o = model.infer(x, resolution_level=a.res_level)
    P = o["points"].float().cpu().numpy()
    m = o["mask"].cpu().numpy().astype(bool) if "mask" in o else np.isfinite(P).all(-1)
    P[~m] = np.nan
    K = o["intrinsics"].float().cpu().numpy()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "points.npy", P.astype(np.float32))
    atomic_json({"fx": float(K[0, 0] * W), "fy": float(K[1, 1] * H), "cx": float(K[0, 2] * W), "cy": float(K[1, 2] * H),
                 "width": int(W), "height": int(H)}, out / "intrinsics.json")
    print(json.dumps({"frame": a.frame, "fx": float(K[0, 0] * W), "median_depth": float(np.nanmedian(P[..., 2]))}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("focal")
    f.add_argument("--video", required=True); f.add_argument("--out", required=True)
    f.add_argument("--n", type=int, default=24); f.add_argument("--res-level", type=int, default=9)
    d = sp.add_parser("depth")
    d.add_argument("--video", required=True); d.add_argument("--out", required=True); d.add_argument("--K", required=True)
    d.add_argument("--start", type=int, default=0); d.add_argument("--end", type=int, default=0)
    d.add_argument("--stride", type=int, default=1); d.add_argument("--scale", type=float, default=0.5)
    d.add_argument("--batch", type=int, default=4); d.add_argument("--res-level", type=int, default=9)
    pt = sp.add_parser("points")
    pt.add_argument("--video", required=True); pt.add_argument("--out", required=True)
    pt.add_argument("--frame", type=int, required=True); pt.add_argument("--res-level", type=int, default=9)
    a = ap.parse_args()
    {"focal": cmd_focal, "depth": cmd_depth, "points": cmd_points}[a.cmd](a)


if __name__ == "__main__":
    main()
