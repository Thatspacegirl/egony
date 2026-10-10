"""Stage 3: tracks + handedness + 3D lifting + temporal optimisation -> hands_rect.npz (CPU).

Inputs (per episode): det_mp_cam_a.npz (detect_mp.py) and depth_samples_{fs,sgbm}.npz (sample_depth.py;
--depth auto prefers FoundationStereo).

1. Merge the forward/backward MediaPipe passes per frame (mean of matched detections, de-duplicated).
2. Link detections into tracklets (Hungarian on mean 2D joint distance / hand size, gaps <= 4 frames) and label
   each tracklet left/right by voting (MediaPipe handedness score summed over frames and passes, plus a weak
   image-side prior); tracklets that co-occur must get different labels (conflicts -> the weaker is flipped or
   dropped). Handedness is therefore constant along a tracklet.
3. Per frame, absolute depth from stereo: each joint's 9x9 depth window (cam_a frame) gives a surface depth D_j;
   the joint centre is D_j + skin offset. Depth targets per joint (--reldepth, see depth_targets): 'hybrid'
   (default) = relative depth of MediaPipe's metric world landmarks after PnP (shape prior), root refitted robustly to
   the dense depth, and each joint whose own dense depth agrees within 3 cm uses it directly (sigma 1 cm).
   Without stereo support (hand outside the rectified FOV) the depth comes from a PnP of the world landmarks
   rescaled to the episode's hand size (sigma 4 cm).
   Detections at Zw > 1.0 m or < 0.12 m (bystander hands, garbage) are dropped.
4. Hand scale: s = median(palm size of the stereo-placed hand, PnP relative depth) / median(palm size of MediaPipe
   world landmarks);
   bone and palm-edge lengths L = s * median(MediaPipe world lengths), shared by both hands of the episode.
5. Sparse robust least squares per hand segment (segments split at gaps > --max_gap frames) over all joints and
   frames: cam_a reprojection (sigma 8 px), depth prior (palm 1.2 cm / fingers 2.5 cm; PnP-only 4 cm), bone and palm
   lengths (4 / 6 mm), second-difference smoothness (8 mm), soft-L1 loss. Frames whose median reprojection error
   after the first solve exceeds 30 px are rejected as outliers and the segment is re-solved. Short gaps are filled
   by the smoothness term (marked valid but not observed).

Output HANDS/<split>/episode_X/hands_rect.npz  (frame = rectified-LEFT camera = depth frame; metres)
  left, right            (T,21,3) f4  joints, NaN where not valid
  left_valid, right_valid    (T,) bool  observed or gap-filled
  left_observed, ...         (T,) bool  a detection was used in this frame
  left_conf, right_conf      (T,21) f4  0..1 heuristic confidence (0 where invalid)
  left_depth_src, ...        (T,) i1    0 none, 1 stereo, 2 PnP size prior, 3 gap fill
  left_uv, right_uv          (T,21,2) f4  final joints projected into cam_a (px)
  bone_lengths           (25,) f4  LEN_EDGES lengths used (BONES + PALM_EDGES)
  T_a_rect               (4,4)     to go to cam_a: X_a = T_a_rect X_rect
  provenance             str (JSON)
and hands_rect.json (stats).

  /mnt/secondary/v2d/envs/t3-hands/bin/python -I lift_hands.py --episodes all --workers 3
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.sparse import lil_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handlib as hl  # noqa: E402

E = np.array(hl.LEN_EDGES)
NB = len(hl.BONES)


# ----------------------------------------------------------------------------------------------- 1. merge passes
def hand_scale(u):
    return max(float(np.ptp(u[:, 0])), float(np.ptp(u[:, 1])), 20.0)


def mean_dist(a, b):
    return float(np.linalg.norm(a - b, axis=-1).mean())


def merge_passes(det, win):
    lm, world, label, score = det["lm2d"], det["world"], det["label"], det["score"]
    P, T, K = lm.shape[:3]
    frames = []
    for t in range(T):
        raw = [(p, k) for p in range(P) for k in range(K) if not np.isnan(lm[p, t, k, 0, 0])]
        A = [d for d in raw if d[0] == 0]
        B = [d for d in raw if d[0] == 1]
        groups, usedB = [], set()
        if A and B:
            C = np.array([[mean_dist(lm[a[0], t, a[1], :, :2], lm[b[0], t, b[1], :, :2]) /
                           hand_scale(lm[a[0], t, a[1], :, :2]) for b in B] for a in A])
            ri, ci = linear_sum_assignment(C)
            pairs = {int(r): int(c) for r, c in zip(ri, ci) if C[r, c] < 0.25}
        else:
            pairs = {}
        for i, a in enumerate(A):
            if i in pairs:
                groups.append([a, B[pairs[i]]])
                usedB.add(pairs[i])
            else:
                groups.append([a])
        groups += [[b] for j, b in enumerate(B) if j not in usedB]
        dets = []
        for g in groups:
            ix = tuple(np.array(g).T)
            u = lm[ix[0], t, ix[1], :, :2].mean(0)
            dets.append(dict(
                u=u, zrel=lm[ix[0], t, ix[1], :, 2].mean(0), world=world[ix[0], t, ix[1]].mean(0),
                vote=float(sum((1.0 if label[p, t, k] == 1 else -1.0) * score[p, t, k] for p, k in g)),
                score=float(np.mean([score[p, t, k] for p, k in g])), npass=len(g),
                win=None if win is None else win[ix[0], t, ix[1]].astype(np.float32).reshape(len(g), 21, -1)))
        # de-duplicate overlapping detections of the same hand
        dets.sort(key=lambda d: -(d["npass"] + d["score"]))
        keep = []
        for d in dets:
            if all(mean_dist(d["u"], k["u"]) / hand_scale(k["u"]) > 0.3 for k in keep):
                keep.append(d)
        frames.append(keep)
    return frames


# ----------------------------------------------------------------------------------------------- 2. tracklets
def build_tracklets(frames, max_link_gap=4):
    tracks = []
    for t, dets in enumerate(frames):
        cand = [i for i, tr in enumerate(tracks) if t - tr["t"][-1] <= max_link_gap]
        assigned = set()
        if cand and dets:
            C = np.full((len(cand), len(dets)), 1e3)
            for a, i in enumerate(cand):
                tr = tracks[i]
                gap = t - tr["t"][-1]
                for b, d in enumerate(dets):
                    c = mean_dist(tr["d"][-1]["u"], d["u"]) / max(hand_scale(tr["d"][-1]["u"]), hand_scale(d["u"]))
                    if c < 0.6 + 0.3 * (gap - 1):
                        C[a, b] = c
            ri, ci = linear_sum_assignment(C)
            for r, c in zip(ri, ci):
                if C[r, c] < 1e3:
                    tracks[cand[r]]["t"].append(t)
                    tracks[cand[r]]["d"].append(dets[c])
                    assigned.add(c)
        for b, d in enumerate(dets):
            if b not in assigned:
                tracks.append({"t": [t], "d": [d]})
    return tracks


def label_tracklets(tracks, W, w_pos=0.3):
    for tr in tracks:
        x = np.array([d["u"][0, 0] for d in tr["d"]])
        tr["E_mp"] = float(sum(d["vote"] for d in tr["d"]))
        tr["E_pos"] = float(w_pos * np.clip((x - W / 2) / (0.25 * W), -1, 1).sum())
        tr["E"] = tr["E_mp"] + tr["E_pos"]
        tr["set"] = set(tr["t"])
    order = sorted(range(len(tracks)), key=lambda i: -abs(tracks[i]["E"]) - 0.01 * len(tracks[i]["t"]))
    side_frames = {0: set(), 1: set()}  # 0 left, 1 right: frames already claimed
    n_flip = n_drop_frames = 0
    for i in order:
        tr = tracks[i]
        pref = 1 if tr["E"] >= 0 else 0
        for lab in (pref, 1 - pref):
            if not (tr["set"] & side_frames[lab]):
                tr["side"] = lab
                n_flip += lab != pref
                break
        else:  # conflicts with both: keep the frames free on the preferred side, then on the other
            lab = pref
            free = tr["set"] - side_frames[lab]
            if len(free) < len(tr["set"] - side_frames[1 - lab]):
                lab = 1 - lab
                free = tr["set"] - side_frames[lab]
            n_drop_frames += len(tr["set"]) - len(free)
            keep = [k for k, t in enumerate(tr["t"]) if t in free]
            tr["t"] = [tr["t"][k] for k in keep]
            tr["d"] = [tr["d"][k] for k in keep]
            tr["set"] = set(tr["t"])
            tr["side"] = lab
        side_frames[tr["side"]] |= tr["set"]
    return {"n_tracklets": len(tracks), "n_flipped": n_flip, "n_conflict_frames_dropped": n_drop_frames}


# ----------------------------------------------------------------------------------------------- 3. per-frame lift
def root_depth(d, fa, min_inl=3):
    """Robust wrist depth Zw from the joint depth windows; returns (Zw or nan, n_inliers, per-joint D or nan)."""
    D = np.full(21, np.nan)
    if d["win"] is None:
        return np.nan, 0, D
    v = d["win"].transpose(1, 0, 2).reshape(21, -1)  # (21, n*81)
    for j in range(21):
        x = v[j][(v[j] > 0.1) & (v[j] < 1.6)]
        if x.size >= 10:
            D[j] = np.median(x)
    rel = 1.0 + d["zrel"] / fa
    e = (D + hl.SKIN_OFFSET) / rel
    ok = np.isfinite(e)
    if ok.sum() < min_inl:
        return np.nan, int(ok.sum()), D
    z0 = np.median(e[ok])
    inl = ok & (np.abs(e - z0) < 0.02)
    if inl.sum() < min_inl:
        return np.nan, int(inl.sum()), D
    return float(np.median(e[inl])), int(inl.sum()), D


def pnp_depths(d, Ka, scale):
    import cv2
    w = d["world"].astype(np.float64) * scale
    ok, rv, tv = cv2.solvePnP(w, d["u"].astype(np.float64), Ka, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    Rm = cv2.Rodrigues(rv)[0]
    return (w @ Rm.T + tv.ravel())[:, 2]


def depth_targets(d, Ka, scale, mode, fa):
    """Per-joint depth targets (cam_a z, m) and sigmas for a stereo-supported detection.

    zrel   : Z_j = Zw (1 + zrel_j / f) (MediaPipe landmark z), palm 1.2 cm / fingers 2.5 cm
    pnp    : relative depth of MediaPipe's metric world landmarks (x hand scale) after PnP onto the 2D joints, root
             refitted to the dense depth (median, inliers within 2 cm), palm 1.5 cm / fingers 2.5 cm
    hybrid : pnp, but joints whose own dense depth (+ skin offset) is within 3 cm of the pnp model use that depth
             (sigma 1.0 cm)
    """
    if mode == "zrel":
        return d["zw"] * (1.0 + d["zrel"] / fa), np.where(np.isin(np.arange(21), hl.PALM), 0.012, 0.025)
    zp = pnp_depths(d, Ka, scale)
    Dj = d["D"] + hl.SKIN_OFFSET
    if zp is None:
        return d["zw"] * (1.0 + d["zrel"] / fa), np.where(np.isin(np.arange(21), hl.PALM), 0.012, 0.025)
    rel = zp - zp[0]
    e = Dj - rel
    ok = np.isfinite(e)
    z0 = np.median(e[ok])
    inl = ok & (np.abs(e - z0) < 0.02)
    Z = (np.median(e[inl]) if inl.sum() >= 3 else z0) + rel
    sig = np.where(np.isin(np.arange(21), hl.PALM), 0.015, 0.025)
    if mode == "hybrid":
        use = np.isfinite(Dj) & (np.abs(Dj - Z) < 0.03)
        Z = np.where(use, Dj, Z)
        sig = np.where(use, 0.010, sig)
    return Z, sig


def palm_size(X):
    return np.mean([np.linalg.norm(X[a] - X[b]) for a, b in ((0, 5), (0, 17), (5, 17), (0, 9))])


# ----------------------------------------------------------------------------------------------- 5. optimisation
def solve_segment(obs, Ka, L, sig):
    """obs: list (len S) of None or dict(u, X0, zt (21,), zsig (21,), w, inimg (21,)). Returns X (S,21,3)."""
    S = len(obs)
    J = 21
    o_idx = [i for i, o in enumerate(obs) if o is not None]
    # initial guess: observed X0, linear interpolation across gaps
    X0 = np.full((S, J, 3), np.nan)
    for i in o_idx:
        X0[i] = obs[i]["X0"]
    tt = np.arange(S)
    for j in range(J):
        for c in range(3):
            X0[:, j, c] = np.interp(tt, o_idx, X0[o_idx, j, c])
    fx, fy, cx, cy = Ka[0, 0], Ka[1, 1], Ka[0, 2], Ka[1, 2]
    # observation arrays
    rp_t, rp_j, rp_u, rp_w = [], [], [], []
    dz_t, dz_j, dz_z, dz_s = [], [], [], []
    for i in o_idx:
        o = obs[i]
        for j in range(J):
            if o["inimg"][j]:
                rp_t.append(i)
                rp_j.append(j)
                rp_u.append(o["u"][j])
                rp_w.append(np.sqrt(o["w"]) / sig["px"])
            if np.isfinite(o["zt"][j]):
                dz_t.append(i)
                dz_j.append(j)
                dz_z.append(o["zt"][j])
                dz_s.append(o["zsig"][j])
    rp_t, rp_j, rp_u, rp_w = np.array(rp_t, int), np.array(rp_j, int), np.array(rp_u), np.array(rp_w)
    dz_t, dz_j, dz_z, dz_s = np.array(dz_t, int), np.array(dz_j, int), np.array(dz_z), np.array(dz_s)
    lsig = np.where(np.arange(len(E)) < NB, sig["bone"], sig["palm"])

    def fun(x):
        X = x.reshape(S, J, 3)
        Xr = X[rp_t, rp_j]
        z = np.maximum(Xr[:, 2], 1e-3)
        r1 = np.stack([(fx * Xr[:, 0] / z + cx - rp_u[:, 0]), (fy * Xr[:, 1] / z + cy - rp_u[:, 1])], 1) * rp_w[:, None]
        r2 = (X[dz_t, dz_j, 2] - dz_z) / dz_s
        r3 = ((np.linalg.norm(X[:, E[:, 1]] - X[:, E[:, 0]], axis=-1) - L) / lsig).ravel()
        r4 = ((X[2:] - 2 * X[1:-1] + X[:-2]) / sig["acc"]).ravel() if S >= 3 else np.zeros(0)
        return np.concatenate([r1.ravel(), r2, r3, r4])

    # sparsity
    n1, n2, n3 = 2 * len(rp_t), len(dz_t), S * len(E)
    n4 = 3 * J * (S - 2) if S >= 3 else 0
    A = lil_matrix((n1 + n2 + n3 + n4, S * J * 3), dtype=np.int8)
    vi = lambda t, j, c: (t * J + j) * 3 + c  # noqa: E731
    for k, (t, j) in enumerate(zip(rp_t, rp_j)):
        for c in range(3):
            A[2 * k, vi(t, j, c)] = 1
            A[2 * k + 1, vi(t, j, c)] = 1
    for k, (t, j) in enumerate(zip(dz_t, dz_j)):
        A[n1 + k, vi(t, j, 2)] = 1
    for t in range(S):
        for e, (a, b) in enumerate(E):
            r = n1 + n2 + t * len(E) + e
            for c in range(3):
                A[r, vi(t, a, c)] = 1
                A[r, vi(t, b, c)] = 1
    for t in range(S - 2):
        for j in range(J):
            for c in range(3):
                r = n1 + n2 + n3 + (t * J + j) * 3 + c
                for dt in range(3):
                    A[r, vi(t + dt, j, c)] = 1
    res = least_squares(fun, X0.ravel(), jac_sparsity=A.tocsr(), method="trf", loss="soft_l1", f_scale=2.0,
                        x_scale=0.01, tr_solver="lsmr", max_nfev=40, xtol=1e-6, ftol=1e-6)
    X = res.x.reshape(S, J, 3)
    # per observed frame median reprojection error
    rep = np.full(S, np.nan)
    for i in o_idx:
        o = obs[i]
        if o["inimg"].any():
            uv = hl.project(X[i], Ka)
            rep[i] = np.median(np.linalg.norm(uv - o["u"], axis=-1)[o["inimg"]])
    return X, rep, res


def lift_episode(args):
    ep_dir, opts = args
    import cv2
    cv2.setNumThreads(1)
    t_start = time.time()
    meta, Ka, Kr, T_a_rect = hl.load_meta(ep_dir)
    od = hl.out_dir(meta)
    out = f"{od}/hands_rect{opts.get('tag', '')}.npz"
    if os.path.exists(out) and not opts["overwrite"]:
        return f"skip {out}"
    det = np.load(f"{od}/det_mp_cam_a.npz")
    cands = {"auto": ("fs", "sgbm", ""), "fs": ("fs",), "sgbm": ("sgbm", "")}[opts.get("depth", "auto")]
    samp_p = next((f"{od}/depth_samples{'_' + c if c else ''}.npz" for c in cands
                   if os.path.exists(f"{od}/depth_samples{'_' + c if c else ''}.npz")), None)
    samp = np.load(samp_p) if samp_p else None
    depth_source = str(samp["source"]) if samp is not None else "none"
    T = meta["n_frames"]
    W, H = meta["images"]["cam_a"]["size_wh"]
    fa = Ka[0, 0]
    frames = merge_passes(det, None if samp is None else samp["win"])
    tracks = build_tracklets(frames)
    lab_stats = label_tracklets(tracks, W)
    stats = {"episode": meta["episode"], "split": meta["split"], "T": T, "depth_source": depth_source,
             "labels": lab_stats}
    # ---- per-side observation sequences with stereo root depth
    per = {0: [None] * T, 1: [None] * T}
    for tr in tracks:
        for t, d in zip(tr["t"], tr["d"]):
            per[tr["side"]][t] = d
    world_palm, stereo_palm = [], []
    n_far = 0
    for side in (0, 1):
        for t in range(T):
            d = per[side][t]
            if d is None:
                continue
            zw, ninl, D = root_depth(d, fa)
            d["zw"], d["ninl"], d["D"] = zw, ninl, D
            if np.isfinite(zw) and not (0.12 < zw < 1.0):
                per[side][t] = None
                n_far += 1
                continue
            world_palm.append(palm_size(d["world"]))
            if np.isfinite(zw):
                # hand-scale estimate: palm size of the stereo-placed hand. 'zrel' uses MediaPipe's landmark z;
                # 'pnp'/'hybrid' use the PnP world-landmark relative depth (the most consistent absolute palm size
                # across public episodes of the same demonstrator: 6.4-7.1 cm vs 5.8-6.8 (zrel) / 5.3-6.6 (dense))
                if opts.get("reldepth", "hybrid") == "zrel":
                    Z = zw * (1.0 + d["zrel"] / fa)
                else:
                    Z, _ = depth_targets(d, Ka, 1.0, "pnp", fa)
                d["X0"] = hl.backproject(d["u"], Z, Ka)
                stereo_palm.append(palm_size(d["X0"]))
    stats["n_dropped_depth_gate"] = n_far
    if len(stereo_palm) >= 5:
        scale = float(np.median(stereo_palm) / np.median(world_palm))
        scale_src = "stereo"
    else:
        scale, scale_src = 1.0, "mediapipe_prior"
    # lengths: MediaPipe world-landmark proportions x stereo scale
    wl = [np.linalg.norm(d["world"][E[:, 1]] - d["world"][E[:, 0]], axis=-1)
          for s in (0, 1) for d in per[s] if d is not None]
    L = scale * np.median(np.array(wl), 0) if wl else None
    raw_bl = [np.linalg.norm(d["X0"][E[:NB, 1]] - d["X0"][E[:NB, 0]], axis=-1)
              for s_ in (0, 1) for d in per[s_] if d is not None and "X0" in d]
    stats["raw_stereo_bone_cv_median"] = (float(np.median(np.std(raw_bl, 0) / np.mean(raw_bl, 0)))
                                          if len(raw_bl) > 2 else None)
    stats.update(hand_scale=scale, hand_scale_src=scale_src, n_stereo_frames=len(stereo_palm),
                 palm_size_stereo_median=float(np.median(stereo_palm)) if stereo_palm else None,
                 palm_size_mpworld_median=float(np.median(world_palm)) if world_palm else None)
    sig = dict(px=8.0, bone=0.004, palm=0.006, acc=0.008)
    names = {0: "left", 1: "right"}
    T_rect_a = np.linalg.inv(T_a_rect)
    outz = {}
    for side in (0, 1):
        nm = names[side]
        Xout = np.full((T, 21, 3), np.nan, np.float32)
        valid = np.zeros(T, bool)
        observed = np.zeros(T, bool)
        conf = np.zeros((T, 21), np.float32)
        dsrc = np.zeros(T, np.int8)
        uv_out = np.full((T, 21, 2), np.nan, np.float32)
        obs = [None] * T
        for t in range(T):
            d = per[side][t]
            if d is None:
                continue
            inimg = (d["u"][:, 0] >= 2) & (d["u"][:, 0] < W - 2) & (d["u"][:, 1] >= 2) & (d["u"][:, 1] < H - 2)
            if np.isfinite(d["zw"]):
                Z, zsig = depth_targets(d, Ka, scale, opts["reldepth"], fa)
                src = 1
            else:
                Z = pnp_depths(d, Ka, scale)
                if Z is None or not np.all(Z > 0.05):
                    continue
                zsig = np.full(21, 0.04)
                src = 2
            X0 = hl.backproject(d["u"], Z, Ka)
            obs[t] = dict(u=d["u"], X0=X0, zt=Z, zsig=zsig, w=float(np.clip(d["score"], 0.2, 1.0)) * (0.7 + 0.3 * (d["npass"] - 1)),
                          inimg=inimg, src=src)
        oi = [t for t in range(T) if obs[t] is not None]
        segs = []
        for t in oi:
            if segs and t - segs[-1][1] <= opts["max_gap"] + 1:
                segs[-1][1] = t
            else:
                segs.append([t, t])
        n_rej = 0
        rep_all = []
        for a, b in segs:
            seg_obs = obs[a:b + 1]
            if L is None:
                break
            X, rep, _ = solve_segment(seg_obs, Ka, L, sig)
            bad = [i for i in range(len(seg_obs)) if seg_obs[i] is not None and rep[i] > opts["rej_px"]]
            if bad and len(bad) < sum(o is not None for o in seg_obs):
                for i in bad:
                    seg_obs[i] = None
                n_rej += len(bad)
                # trim leading/trailing None after rejection
                idx = [i for i, o in enumerate(seg_obs) if o is not None]
                a2, b2 = a + idx[0], a + idx[-1]
                seg_obs = seg_obs[idx[0]:idx[-1] + 1]
                a, b = a2, b2
                X, rep, _ = solve_segment(seg_obs, Ka, L, sig)
            rep_all += [r for r in rep if np.isfinite(r)]
            Xr = hl.transform(T_rect_a, X)
            for i in range(len(seg_obs)):
                t = a + i
                Xout[t] = Xr[i]
                valid[t] = True
                uv_out[t] = hl.project(X[i], Ka)
                if seg_obs[i] is not None:
                    observed[t] = True
                    dsrc[t] = seg_obs[i]["src"]
                    c = seg_obs[i]["w"] * np.where(seg_obs[i]["inimg"], 1.0, 0.5) * (1.0 if dsrc[t] == 1 else 0.6)
                    conf[t] = np.clip(c * np.exp(-(rep[i] / 25.0) ** 2), 0, 1)
                else:
                    dsrc[t] = 3
            # gap frames: confidence decays with distance to the nearest observation
            for i in range(len(seg_obs)):
                t = a + i
                if not observed[t]:
                    prv = max([k for k in range(a, t) if observed[k]], default=None)
                    nxt = min([k for k in range(t + 1, b + 1) if observed[k]], default=None)
                    dmin = min(t - prv, nxt - t)
                    conf[t] = 0.5 ** (dmin / 2.0) * np.minimum(conf[prv], conf[nxt])
        Xout[~valid] = np.nan
        outz.update({nm: Xout, f"{nm}_valid": valid, f"{nm}_observed": observed, f"{nm}_conf": conf,
                     f"{nm}_depth_src": dsrc, f"{nm}_uv": uv_out})
        bl = np.linalg.norm(Xout[valid][:, E[:NB, 1]] - Xout[valid][:, E[:NB, 0]], axis=-1) if valid.any() else np.zeros((0, NB))
        stats[nm] = {
            "frames_valid": int(valid.sum()), "frames_observed": int(observed.sum()),
            "frames_stereo": int((dsrc == 1).sum()), "frames_pnp": int((dsrc == 2).sum()),
            "frames_gapfill": int((dsrc == 3).sum()), "segments": len(segs), "outlier_frames_rejected": n_rej,
            "reproj_px_median": float(np.median(rep_all)) if rep_all else None,
            "reproj_px_p90": float(np.percentile(rep_all, 90)) if rep_all else None,
            "bone_len_cv_median": float(np.median(bl.std(0) / bl.mean(0))) if len(bl) > 2 else None,
            "wrist_depth_rect_median": float(np.nanmedian(Xout[valid, 0, 2])) if valid.any() else None,
        }
    prov = {"detector": "MediaPipe HandLandmarker (Tasks 1.1.0, hand_landmarker.task float16, Apache-2.0), cam_a",
            "depth": depth_source, "lift": "stereo root depth + MediaPipe relative z; sparse robust temporal opt.",
            "opts": opts, "sig": sig, "dataset_revision": meta.get("dataset_revision")}
    np.savez_compressed(out, **outz, bone_lengths=np.asarray(L if L is not None else np.full(len(E), np.nan), np.float32),
                        len_edges=E, T_a_rect=T_a_rect, num_frames=np.array(T), episode_index=np.array(meta["episode"]),
                        split=np.array(meta["split"]), provenance=np.array(json.dumps(prov)))
    stats["seconds"] = round(time.time() - t_start, 1)
    stats["bone_lengths_m"] = None if L is None else np.round(L, 4).tolist()
    json.dump(stats, open(f"{od}/hands_rect{opts.get('tag', '')}.json", "w"), indent=1)
    s = stats
    return (f"{meta['split']}/episode_{meta['episode']:06d} depth={depth_source} scale={scale:.3f}({scale_src}) "
            f"L valid/obs/stereo={s['left']['frames_valid']}/{s['left']['frames_observed']}/{s['left']['frames_stereo']} "
            f"R={s['right']['frames_valid']}/{s['right']['frames_observed']}/{s['right']['frames_stereo']} T={T} "
            f"rep L/R={s['left']['reproj_px_median']}/{s['right']['reproj_px_median']} {s['seconds']}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--splits", default="public,evaluation")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max_gap", type=int, default=8, help="max gap (frames) filled by interpolation (0.4 s)")
    ap.add_argument("--rej_px", type=float, default=30.0)
    ap.add_argument("--reldepth", default="hybrid", choices=("zrel", "pnp", "hybrid"))
    ap.add_argument("--tag", default="", help="output hands_rect<tag>.npz (variants for tuning)")
    ap.add_argument("--depth", default="auto", choices=("auto", "fs", "sgbm"),
                    help="depth samples: auto = FoundationStereo if sampled, else SGBM")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    opts = {"max_gap": a.max_gap, "rej_px": a.rej_px, "overwrite": a.overwrite, "reldepth": a.reldepth, "tag": a.tag,
            "depth": a.depth}
    jobs = [(d, opts) for d in hl.episode_dirs(a.episodes, a.splits.split(","))]
    if a.workers <= 1:
        for j in jobs:
            print(lift_episode(j), flush=True)
        return
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for msg in pool.imap_unordered(lift_episode, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())
