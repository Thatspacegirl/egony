#!/usr/bin/env python3
"""Assemble a Track 1 episode prediction (kit NPZ + mesh) from our stage outputs, all in OUR camera frame.

  human  : t1_human.build_episode(variant)            (SAM-3D-Body MHR articulation, GEM-X depth)
  object : t1_fpose.py track (FoundationPose on MoGe-2 depth, TRELLIS mesh, metric scale)
  scale  : k = median_t median_px( human-model depth / MoGe depth )  over actor-mask pixels (depth stabilised with the
           same per-frame ratios as the object track).  The object is mapped into the human's metric frame:
           object_translation = k * t,  object_scale = k * s.   (GEM-X/SAM-3D-Body put the person ~10% too far and
           ~10% too big; the scorer's Sim(3) removes that, so the object must live in the SAME scaled world.)
  frames : object poses known on every 2nd window frame -> SLERP / linear interpolation, constant outside.
Then optional HOI refinement (t1_hoi.py), smoothing (smoothing.py) and PEN (postprocess / pen_refine) are applied by
t1_run.py; this module only builds the raw episode.

Python: /mnt/secondary/v2d/scratch/venv/bin/python -I
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_items as I  # noqa: E402
import t1_human as HU  # noqa: E402

R = Path("/mnt/secondary/v2d/t1")


def unpack(bits, W):
    return np.unpackbits(bits, axis=-1, count=W).astype(bool)


def paths(split, ep):
    E = f"episode_{int(ep):06d}"
    return {k: R / k / split / E for k in ("human", "masks", "depth", "mesh", "fpose")}


def human_moge_scale(hum, K_full, M, D, frames_idx, r, n_eval=40, grid=8):
    """k = human-model depth / MoGe depth at actor pixels (median over frames).  hum: episode dict (kit frame = camera)."""
    W, H = int(M["W"]), int(M["H"])
    sx = W / float(M["full_wh"][0])
    K = np.array(K_full, float).copy(); K[:2] *= sx
    n = len(frames_idx)
    cand = [i for i in range(n) if M["human"][i].any()]
    if not cand:
        return np.nan, {}
    sel = np.array(cand)[np.linspace(0, len(cand) - 1, min(n_eval, len(cand))).round().astype(int)]
    vf = np.asarray(frames_idx)[sel]
    params = np.concatenate([hum["pose"][vf], np.broadcast_to(hum["scales"], (len(vf), 68))], 1)
    V, _ = HU.mhr_forward(params, hum["shape"])
    gh, gw = H // grid, W // grid
    ks, per = [], []
    for j, i in enumerate(sel):
        X = V[j]
        z = X[:, 2]
        u = X[:, 0] / z * K[0, 0] + K[0, 2]
        v = X[:, 1] / z * K[1, 1] + K[1, 2]
        gi, gj = (v / grid).astype(int), (u / grid).astype(int)
        ok = (z > 0.1) & (gi >= 0) & (gi < gh) & (gj >= 0) & (gj < gw)
        zb = np.full((gh, gw), np.inf)
        np.minimum.at(zb, (gi[ok], gj[ok]), z[ok])
        hm = unpack(M["human"][i], W)[: gh * grid, : gw * grid].reshape(gh, grid, gw, grid).mean((1, 3)) > 0.9
        d = np.asarray(D[i], np.float32)[: gh * grid, : gw * grid].reshape(gh, grid, gw, grid)
        dv = np.where(d > 0, d, np.nan)
        dm = np.nanmedian(dv.reshape(gh, grid, gw, grid).transpose(0, 2, 1, 3).reshape(gh, gw, -1), 2) * r[i]
        m = hm & np.isfinite(zb) & np.isfinite(dm) & (dm > 0)
        if m.sum() < 10:
            continue
        kt = float(np.median(zb[m] / dm[m]))
        ks.append(kt); per.append((int(vf[j]), kt, int(m.sum())))
    if not ks:
        return np.nan, {}
    return float(np.median(ks)), {"k_frames": per, "k_iqr": [float(np.percentile(ks, 25)), float(np.percentile(ks, 75))]}


def moge_anchor(hum, K_full, M, D, frames_idx, r, grid=8, min_cells=20):
    """Per processed frame i: additive depth offset dz_i = median over actor-mask cells of (r_i * MoGe depth -
    model front-surface depth) for the body as placed by SAM-3D-Body; rho_i = (z_pelvis + dz_i) / z_pelvis.
    Returns rho (n,) (nan where not measurable) and the pelvis positions (n,3) of the processed frames."""
    W, H = int(M["W"]), int(M["H"])
    sx = W / float(M["full_wh"][0])
    K = np.array(K_full, float).copy(); K[:2] *= sx
    vf = np.asarray(frames_idx)
    params = np.concatenate([hum["pose"][vf], np.broadcast_to(hum["scales"], (len(vf), 68))], 1)
    gh, gw = H // grid, W // grid
    rho = np.full(len(vf), np.nan); pel = np.zeros((len(vf), 3)); ncell = np.zeros(len(vf), int)
    for s0 in range(0, len(vf), 64):
        V, J = HU.mhr_forward(params[s0:s0 + 64], hum["shape"])
        for j in range(len(V)):
            i = s0 + j
            pel[i] = J[j, HU.PELVIS]
            if not M["human"][i].any():
                continue
            X = V[j]; z = X[:, 2]
            u = X[:, 0] / z * K[0, 0] + K[0, 2]; v = X[:, 1] / z * K[1, 1] + K[1, 2]
            gi, gj = (v / grid).astype(int), (u / grid).astype(int)
            ok = (z > 0.1) & (gi >= 0) & (gi < gh) & (gj >= 0) & (gj < gw)
            zb = np.full((gh, gw), np.inf)
            np.minimum.at(zb, (gi[ok], gj[ok]), z[ok])
            hm = unpack(M["human"][i], W)[: gh * grid, : gw * grid].reshape(gh, grid, gw, grid).mean((1, 3)) > 0.9
            d = np.asarray(D[i], np.float32)[: gh * grid, : gw * grid].reshape(gh, grid, gw, grid).transpose(0, 2, 1, 3)
            d = np.where(d > 0, d, np.nan).reshape(gh, gw, -1)
            with np.errstate(all="ignore"):
                dm = np.nanmedian(d, 2) * r[i]
            m = hm & np.isfinite(zb) & np.isfinite(dm)
            ncell[i] = int(m.sum())
            if m.sum() < min_cells:
                continue
            dz = float(np.median(dm[m] - zb[m]))
            rho[i] = (pel[i, 2] + dz) / pel[i, 2]
    return rho, pel, ncell


def apply_depth_multiplier(hum, mult):
    """Move the body of every frame along the camera ray through its pelvis by factor mult[t] (pelvis depth x mult)."""
    T = len(hum["pose"])
    params = np.concatenate([hum["pose"], np.broadcast_to(hum["scales"], (T, 68))], 1)
    _, J = HU.mhr_forward(params, hum["shape"], verts=False)
    pel = J[:, HU.PELVIS]
    out = dict(hum)
    pose = np.array(hum["pose"], np.float64, copy=True)
    pose[:, :3] += 10.0 * ((mult[:, None] - 1.0) * pel) * HU.FLIP
    out["pose"] = pose
    return out


def interp_poses(frames_known, Rk, tk, T):
    """SLERP / linear between known frames (sorted), hold outside; returns R [T,3,3], t [T,3]."""
    from scipy.spatial.transform import Rotation, Slerp
    fk = np.asarray(frames_known)
    rot = Rotation.from_matrix(Rk)
    Rs = np.empty((T, 3, 3)); ts = np.empty((T, 3))
    allf = np.arange(T)
    inside = (allf >= fk[0]) & (allf <= fk[-1])
    if len(fk) >= 2:
        Rs[inside] = Slerp(fk, rot)(allf[inside]).as_matrix()
    else:
        Rs[inside] = Rk[0]
    for d in range(3):
        ts[:, d] = np.interp(allf, fk, tk[:, d])
    Rs[allf < fk[0]] = Rk[0]
    Rs[allf > fk[-1]] = Rk[-1]
    return Rs, ts


def build(split, ep, variant="s3gx", k_override=None, use_k=True):
    it = I.item(split, ep)
    P = paths(split, ep)
    _, s3p, gxp = HU.find_outputs(split, ep)
    mg = variant.startswith("mg")              # 'mg' / 'mgr': SAM-3D-Body body (+GEM-X rotation), MoGe depth anchor
    hum, hinfo = HU.build_episode(s3p, gxp, ("s3" + variant[2:]) if mg else variant)
    T = len(hum["pose"])
    F = dict(np.load(P["fpose"] / "fpose.npz"))
    M = dict(np.load(P["masks"] / "masks.npz"))
    D = np.load(P["depth"] / "depth_s0.5.npy", mmap_mode="r")
    frames = F["frames"]
    Pm = F["cam_T_obj"]
    ok = np.isfinite(Pm).all((1, 2))
    info = {"human": hinfo, "fpose_ok": int(ok.sum()), "fpose_n": int(len(ok))}
    if mg:
        rho, _, ncell = moge_anchor(hum, F["K_full"], M, D, frames, F["depth_ratio"])
        good = np.isfinite(rho)
        kappa = 1.0 / float(np.median(rho[good]))
        mult_k = kappa * rho[good]
        mult = np.interp(np.arange(T), frames[good], mult_k)
        hum = apply_depth_multiplier(hum, mult)
        info.update({"mg_kappa": kappa, "mg_valid": int(good.sum()), "mg_mult_p5_p95": [float(np.percentile(mult_k, 5)), float(np.percentile(mult_k, 95))]})
        k_override, use_k = kappa, True
    if use_k:
        if k_override is not None:
            k, kinfo = float(k_override), {}
        else:
            k, kinfo = human_moge_scale(hum, F["K_full"], M, D, frames, F["depth_ratio"])
        if not np.isfinite(k):
            k = 1.0
        info.update({"k": k, **kinfo})
    else:
        k = 1.0
    Rk = Pm[ok, :3, :3]
    u, _, vt = np.linalg.svd(Rk)
    Rk = u @ vt
    tk = Pm[ok, :3, 3] * k
    Ro, to = interp_poses(frames[ok], Rk, tk, T)
    epd = {**hum, "object_rotation": Ro, "object_translation": to, "object_scale": np.array(float(F["scale"]) * k)}
    info["object_scale"] = float(epd["object_scale"])
    return epd, P["mesh"] / "mesh_0.ply", info


def write(out_dir, ep, epd, mesh_path, info=None):
    import shutil
    import t1lib as L
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    L.save_episode(out_dir / f"episode_{int(ep):06d}.npz", epd)
    for old in out_dir.glob(f"episode_{int(ep):06d}_object.*"):
        old.unlink()
    dst = out_dir / f"episode_{int(ep):06d}_object{Path(mesh_path).suffix}"
    shutil.copyfile(Path(mesh_path).resolve(), dst)
    if info is not None:
        json.dump(info, open(out_dir / f"episode_{int(ep):06d}_assemble.json", "w"), indent=1, default=float)
