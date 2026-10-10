"""Development evaluation of hands_rect<tag>.npz on PUBLIC episodes (no hand ground truth exists in dataset revision
5f68335: the public parquets carry only observation.objects + timestamps, so every metric here is a proxy).

Per episode and variant tag:
  coverage   valid / observed / stereo-supported frame fractions per side
  stereo     vs the independent RTMPose detections on the rectified LEFT/RIGHT views (stereo_check.py output
             stereo_tri_rtmpose.npz): 2D px error of the projected joints in each view, depth(B) - depth(A) on joints
             triangulated by B (median = bias, MAD = spread), 3D distance A-B
  temporal   joint jitter (median |second difference|, mm, observed consecutive frames), bone-length CV
  gt_objects (needs gt_cam_register.npz; frames with a valid registration): for frames where a GT object moves
             (> 5 cm/s or > 30 deg/s, i.e. someone must be holding it), the distance from the closest hand joint (any
             of 21, either hand) to the GT object's scan surface; fraction < 2 / 3 / 5 cm. Same for the DEV
             placeholder hands of sharpa_task (where a dev_gt task exists) for comparison, plus our-vs-placeholder
             wrist / fingertip distance during those frames; grasp rigidity = std (norm of per-axis std) of the
             grasping hand's palm centre in the moving object's frame per moving segment (>= 10 frames).

  python -I eval_hands.py --episodes all --tags "",_zrel,_pnp  -> HANDS/eval_public<...>.json + printed table
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handlib as hl  # noqa: E402

DEV_TASK = "/mnt/secondary/v2d/t3/tasks/dev_gt/human_motion_data/v2d_track3"
# gt_cam_register overlays inspected by eye on 2026-10-08 (cam_a, first/middle/last registered frame): only these
# public episodes register correctly throughout. ICP fitness is NOT a sufficient check: flat or single objects (dust pan
# eps 39/40, wooden pieces, white pot eps 41-45) slide onto the table plane with fitness 0.5-0.7 (ep 13 is wrong at f0).
REGISTRATION_OK = {12, 22, 26, 27, 31}


def pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return None if x.size == 0 else round(float(np.percentile(x, q)), 5)


def stereo_metrics(z, B, Kr):
    base = float(B["baseline"])
    f = Kr[0, 0]
    eL, eR, dz, d3 = [], [], [], []
    for s in hl.SIDES:
        X = z[s].astype(float)
        dl, dr = B[f"{s}_rectL"], B[f"{s}_rectR"]
        T = X.shape[0]
        for t in range(T):
            if not np.isfinite(X[t]).all():
                continue
            uvL = hl.project(X[t], Kr)
            uvR = hl.project(X[t] - [base, 0, 0], Kr)
            if np.isfinite(dl[t]).any():
                e = np.linalg.norm(dl[t] - uvL, axis=1)
                eL += e[np.isfinite(e)].tolist()
            if np.isfinite(dr[t]).any():
                e = np.linalg.norm(dr[t] - uvR, axis=1)
                eR += e[np.isfinite(e)].tolist()
            tri = B[s][t]
            ok = np.isfinite(tri).all(1)
            if ok.sum() >= 10:
                dz += (tri[ok, 2] - X[t, ok, 2]).tolist()
                d3 += np.linalg.norm(tri[ok] - X[t, ok], axis=1).tolist()
    dz = np.array(dz)
    return {"rectL_px_median": pct(eL, 50), "rectR_px_median": pct(eR, 50), "rectR_px_p90": pct(eR, 90),
            "depth_B_minus_A_median_m": pct(dz, 50),
            "depth_B_minus_A_MAD_m": None if not dz.size else round(float(np.median(np.abs(dz - np.median(dz)))), 5),
            "joint3d_A_B_median_m": pct(d3, 50), "n_joints": int(dz.size)}


def temporal_metrics(z):
    jit, cv = [], []
    for s in hl.SIDES:
        X = z[s].astype(float)
        ob = z[f"{s}_observed"]
        ok = ob[2:] & ob[1:-1] & ob[:-2]
        if ok.any():
            a = np.linalg.norm(X[2:] - 2 * X[1:-1] + X[:-2], axis=-1)[ok]
            jit += (a.ravel() * 1000).tolist()
        v = z[f"{s}_valid"]
        if v.sum() > 5:
            E = np.array(hl.BONES)
            bl = np.linalg.norm(X[v][:, E[:, 1]] - X[v][:, E[:, 0]], axis=-1)
            cv.append(float(np.median(bl.std(0) / bl.mean(0))))
    return {"jitter_mm_median": pct(jit, 50), "jitter_mm_p90": pct(jit, 90),
            "bone_cv_median": round(float(np.mean(cv)), 4) if cv else None}


def moving_frames(pose, fps=20.0, v_thr=0.05, w_thr=30.0):
    from scipy.spatial.transform import Rotation as R
    T, Bn = pose.shape[:2]
    mov = np.zeros((T, Bn), bool)
    for b in range(Bn):
        p = pose[:, b, :3]
        v = np.linalg.norm(np.gradient(p, axis=0), axis=-1) * fps
        r = R.from_quat(pose[:, b, [4, 5, 6, 3]])
        w = np.zeros(T)
        w[1:] = np.degrees((r[:-1].inv() * r[1:]).magnitude()) * fps
        from scipy.ndimage import median_filter
        mov[:, b] = median_filter((v > v_thr) | (w > w_thr), size=5)
    return mov


def placeholder_joints(ep):
    import pyarrow.dataset as pads
    d = f"{DEV_TASK}/loaded/sequence_id=episode_{ep:06d}/robot_name=sharpa_wave"
    if not os.path.isdir(d):
        return None
    m = json.load(open(f"{DEV_TASK}/manifests/episode_{ep:06d}.json"))
    Rm, tv = np.array(m["sim_from_input"]["R"]), np.array(m["sim_from_input"]["t"])
    t = pads.dataset(d, format="parquet").to_table(columns=[f"mano_{s}_joints" for s in hl.SIDES])
    out = {}
    for s in hl.SIDES:
        J = np.asarray(t[f"mano_{s}_joints"][0].as_py(), float)  # sim frame
        out[s] = (J - tv) @ Rm  # back to the input (mocap) frame
    return out


def gt_object_metrics(z, ep, reg):
    from scipy.spatial import cKDTree
    import gt_cam_register as G
    names, pose, vis = G.load_gt(ep)
    scans = [G.load_scan_points(n, 20000) for n in names]
    trees = [cKDTree(p) for p, _, _ in scans]
    W = reg["world_T_rect"]
    rv = reg["valid"]
    mov = moving_frames(pose)
    T = W.shape[0]
    ours = {s: z[s].astype(float) for s in hl.SIDES}
    ph = placeholder_joints(ep)
    d_ours, d_ph, d_wrist, d_tips, n_frames, n_nohand = [], [], [], [], 0, 0
    for t in range(T):
        if not rv[t] or not mov[t].any():
            continue
        n_frames += 1
        Wt = W[t]
        hands_m = [ours[s][t] @ Wt[:3, :3].T + Wt[:3, 3] for s in hl.SIDES if np.isfinite(ours[s][t]).all()]
        for b in np.flatnonzero(mov[t]):
            Rb = G.quat_R(pose[t, b, 3:])

            def dist(J):
                loc = (J - pose[t, b, :3]) @ Rb  # into the object frame
                return float(trees[b].query(loc)[0].min())
            if hands_m:
                d_ours.append(min(dist(J) for J in hands_m))
            else:
                n_nohand += 1
            if ph is not None:
                d_ph.append(min(dist(ph[s][t]) for s in hl.SIDES))
        if ph is not None and hands_m:
            for s in hl.SIDES:
                if np.isfinite(ours[s][t]).all():
                    J = ours[s][t] @ Wt[:3, :3].T + Wt[:3, 3]
                    d_wrist.append(float(np.linalg.norm(J[0] - ph[s][t][0])))
                    d_tips.append(float(np.linalg.norm(J[hl.TIPS] - ph[s][t][hl.TIPS], axis=1).mean()))

    # grasp rigidity: per moving segment (>= 10 frames, valid registration), the hand closest to the object on average;
    # std of its palm centre expressed in the object frame (a rigid grasp + exact 3D gives ~0; depth noise adds to it)
    rig = []
    for b in range(len(names)):
        m = mov[:, b] & rv
        t = 0
        while t < T:
            if not m[t]:
                t += 1
                continue
            e = t
            while e + 1 < T and m[e + 1]:
                e += 1
            if e - t + 1 >= 10:
                best = None
                for s_ in hl.SIDES:
                    P = []
                    for k in range(t, e + 1):
                        if np.isfinite(ours[s_][k]).all():
                            Wk = W[k]
                            Jm = ours[s_][k] @ Wk[:3, :3].T + Wk[:3, 3]
                            Rb = G.quat_R(pose[k, b, 3:])
                            P.append(((Jm[hl.PALM].mean(0) - pose[k, b, :3]) @ Rb,
                                      float(trees[b].query((Jm - pose[k, b, :3]) @ Rb)[0].min())))
                    if len(P) >= 8:
                        dm = np.mean([p[1] for p in P])
                        if best is None or dm < best[0]:
                            best = (dm, np.array([p[0] for p in P]))
                if best is not None and best[0] < 0.05:
                    rig.append(float(np.sqrt((best[1].std(0) ** 2).sum())))
            t = e + 1

    def summ(d):
        d = np.array(d)
        if not d.size:
            return None
        return {"median_m": round(float(np.median(d)), 4), "p75_m": round(float(np.percentile(d, 75)), 4),
                "lt2cm": round(float((d < 0.02).mean()), 3), "lt3cm": round(float((d < 0.03).mean()), 3),
                "lt5cm": round(float((d < 0.05).mean()), 3), "n": int(d.size)}
    return {"moving_frames_with_valid_registration": n_frames, "object_moving_no_hand_frames": n_nohand,
            "hand_to_moving_object_ours": summ(d_ours), "hand_to_moving_object_placeholder": summ(d_ph),
            "ours_vs_placeholder_wrist_median_m": pct(d_wrist, 50),
            "ours_vs_placeholder_tips_median_m": pct(d_tips, 50),
            "grasp_palm_std_in_object_frame_m": pct(rig, 50), "grasp_segments": len(rig),
            "registration_valid_frac": round(float(rv.mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--tags", default="")
    ap.add_argument("--out", default=f"{hl.HANDS}/eval_public.json")
    a = ap.parse_args()
    tags = a.tags.split(",")
    res = {}
    for ep_dir in hl.episode_dirs(a.episodes, ("public",)):
        meta, Ka, Kr, T_a_rect = hl.load_meta(ep_dir)
        od = hl.out_dir(meta)
        ep = meta["episode"]
        Bp = f"{od}/stereo_tri_rtmpose.npz"
        B = np.load(Bp) if os.path.exists(Bp) else None
        regp = f"{od}/gt_cam_register.npz"
        reg = np.load(regp) if os.path.exists(regp) and ep in REGISTRATION_OK else None
        for tag in tags:
            hp = f"{od}/hands_rect{tag}.npz"
            if not os.path.exists(hp):
                continue
            z = np.load(hp)
            T = int(z["num_frames"])
            r = {"T": T}
            for s in hl.SIDES:
                r[f"{s}_valid_frac"] = round(float(z[f"{s}_valid"].mean()), 3)
                r[f"{s}_observed_frac"] = round(float(z[f"{s}_observed"].mean()), 3)
                r[f"{s}_stereo_frac"] = round(float((z[f"{s}_depth_src"] == 1).mean()), 3)
            r.update(temporal_metrics(z))
            if B is not None:
                r["stereo"] = stereo_metrics(z, B, Kr)
            if reg is not None:
                r["gt_objects"] = gt_object_metrics(z, ep, reg)
            res.setdefault(tag or "hybrid", {})[ep] = r
            print(ep, tag or "hybrid", json.dumps(r), flush=True)
    # aggregate
    agg = {}
    for tag, eps in res.items():
        def col(f):
            return [f(r) for r in eps.values() if f(r) is not None]
        g = lambda r, k: (r.get("gt_objects") or {}).get(k)  # noqa: E731
        agg[tag] = {
            "episodes": len(eps),
            "valid_frac_mean": round(float(np.mean(col(lambda r: 0.5 * (r["left_valid_frac"] + r["right_valid_frac"])))), 3),
            "stereo_rectR_px_median": pct(col(lambda r: (r.get("stereo") or {}).get("rectR_px_median")), 50),
            "stereo_depth_bias_median_m": pct(col(lambda r: (r.get("stereo") or {}).get("depth_B_minus_A_median_m")), 50),
            "stereo_depth_MAD_median_m": pct(col(lambda r: (r.get("stereo") or {}).get("depth_B_minus_A_MAD_m")), 50),
            "jitter_mm_median": pct(col(lambda r: r.get("jitter_mm_median")), 50),
            "bone_cv_median": pct(col(lambda r: r.get("bone_cv_median")), 50),
            "hand_obj_ours_median_m": pct(col(lambda r: (g(r, "hand_to_moving_object_ours") or {}).get("median_m")), 50),
            "hand_obj_ours_lt3cm_mean": pct(col(lambda r: (g(r, "hand_to_moving_object_ours") or {}).get("lt3cm")), 50),
            "hand_obj_placeholder_median_m": pct(col(lambda r: (g(r, "hand_to_moving_object_placeholder") or {}).get("median_m")), 50),
            "grasp_palm_std_median_m": pct(col(lambda r: g(r, "grasp_palm_std_in_object_frame_m")), 50),
            "registered_episodes": len(col(lambda r: r.get("gt_objects"))),
        }
    out = {"per_episode": res, "aggregate": agg}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
