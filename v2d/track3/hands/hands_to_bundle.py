"""hands_rect.npz (rectified-left camera frame, per frame) -> t3_perception_v1 bundle hand keys (perception world).

The bundle (sharpa_task/bundle.py) wants hand_<side>_joints (N,21,3) in the SAME world as object_pose, joint order
= MANO/OpenPose 21 (== MediaPipe, so no re-ordering), plus hand_<side>_valid (N,). With no hand_<side>_joints_wxyz,
to_sharpa_task.py Kabsch-fits the template palm per frame (source 'bundle_keypoints+kabsch_template'), fills invalid
frames by linear interpolation (holding the ends), computes contacts and runs the Sharpa IK.

World: any npz with world_T_rect (T,4,4) + valid (T,) -- the cuVSLAM output vo_cuvslam.npz (world = rect-left at the
first tracked frame), or, DEV ONLY on public episodes, gt_cam_register.npz (world = mocap world of the public GT).

Library:  hands_in_world(hands_npz, world_T_rect, pose_valid, min_conf=0.0) -> {side: (J (T,21,3), valid (T,), conf)}
CLI:      python -I hands_to_bundle.py --bundle IN.npz --hands HANDS/<split>/episode_X/hands_rect.npz \\
              --poses VO/<split>/episode_X/vo_cuvslam.npz --out OUT.npz [--min_conf 0.05] [--sides left,right]
  --fill object (default): a hand that is invisible while in contact (< 3 cm) with an object that moves (> 2 cm)
  rides with that object (object-frame joints interpolated between the gap ends; hand_<side>_filled, conf 0.3).
  copies every array of IN.npz, adds/replaces hand_<side>_joints / _valid (+ hand_<side>_conf, ignored by the loader),
  drops any stale hand_<side>_joints_wxyz, appends to provenance, then validates with sharpa_task.bundle.load_bundle.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIDES = ("left", "right")


def hands_in_world(hands_npz, world_T_rect, pose_valid, min_conf=0.0, sides=SIDES):
    z = np.load(hands_npz, allow_pickle=False)
    T = int(z["num_frames"])
    if world_T_rect.shape != (T, 4, 4):
        raise ValueError(f"world_T_rect {world_T_rect.shape} != {(T, 4, 4)}")
    pv = np.asarray(pose_valid, bool) & np.isfinite(world_T_rect).all((1, 2))
    out = {}
    for s in sides:
        X = z[s].astype(np.float64)  # (T,21,3) rect
        v = z[f"{s}_valid"] & pv & np.isfinite(X).all((1, 2))
        conf = z[f"{s}_conf"].astype(np.float64)
        if min_conf > 0:
            v &= conf.mean(1) >= min_conf
        Rm = np.where(pv[:, None, None], world_T_rect[:, :3, :3], np.eye(3))
        tv = np.where(pv[:, None], world_T_rect[:, :3, 3], 0.0)
        J = np.einsum("tij,tkj->tki", Rm, X) + tv[:, None]
        J[~v] = np.nan
        out[s] = (J, v, np.where(v[:, None], conf, 0.0))
    return out, z


def _quat_R(q):
    from scipy.spatial.transform import Rotation as R
    return R.from_quat(np.asarray(q)[..., [1, 2, 3, 0]]).as_matrix()


def object_anchored_fill(J, v, obj_pose, obj_valid, verts, contact=0.03, min_move=0.02):
    """Fill invalid stretches of one hand (J (T,21,3) world, v (T,)) where the hand was in contact with a moving
    object: the hand rides with the object. For a gap (a, b) between valid frames, the object k in contact at BOTH ends
    (min joint-to-posed-vertex distance < contact) whose position moves > min_move during the gap carries the hand:
    joints in k's frame are linearly interpolated from frame a to frame b and re-posed with k's pose at each frame.
    Leading / trailing gaps use the single anchor (constant object-frame joints). Frames whose object pose is invalid
    stay unfilled. Returns (J, v, filled mask, per-gap log)."""
    from scipy.spatial import cKDTree
    T = J.shape[0]
    trees = [cKDTree(V) for V in verts]
    J, v = J.copy(), v.copy()
    filled = np.zeros(T, bool)
    log = []

    def local(t, k):
        return (J[t] - obj_pose[t, k, :3]) @ _quat_R(obj_pose[t, k, 3:])

    def contact_objs(t):
        out = []
        for k in range(len(verts)):
            if obj_valid[t, k] and trees[k].query(local(t, k))[0].min() < contact:
                out.append(k)
        return out

    vi = np.flatnonzero(v)
    if vi.size == 0:
        return J, v, filled, log
    gaps = [(int(a), int(b)) for a, b in zip(vi[:-1], vi[1:]) if b - a > 1]
    if vi[0] > 0:
        gaps.insert(0, (None, int(vi[0])))
    if vi[-1] < T - 1:
        gaps.append((int(vi[-1]), None))
    for a, b in gaps:
        ka = set(contact_objs(a)) if a is not None else None
        kb = set(contact_objs(b)) if b is not None else None
        cand = ka & kb if (ka is not None and kb is not None) else (ka if ka is not None else kb)
        lo, hi = (a + 1 if a is not None else 0), (b - 1 if b is not None else T - 1)
        best = None
        for k in sorted(cand):
            ok = obj_valid[lo:hi + 1, k]
            if not ok.any():
                continue
            P = obj_pose[lo:hi + 1, k, :3][ok]
            ref = obj_pose[a if a is not None else b, k, :3]
            move = float(np.linalg.norm(P - ref, axis=1).max())
            if move > min_move and (best is None or move > best[1]):
                best = (k, move)
        if best is None:
            continue
        k = best[0]
        La = local(a, k) if a is not None else None
        Lb = local(b, k) if b is not None else None
        n = 0
        for t in range(lo, hi + 1):
            if not obj_valid[t, k]:
                continue
            if La is not None and Lb is not None:
                w = (t - a) / (b - a)
                Lt = (1 - w) * La + w * Lb
            else:
                Lt = La if La is not None else Lb
            J[t] = Lt @ _quat_R(obj_pose[t, k, 3:]).T + obj_pose[t, k, :3]
            v[t] = filled[t] = True
            n += 1
        log.append({"gap": [a, b], "object": int(k), "object_move_m": round(best[1], 3), "frames_filled": n})
    return J, v, filled, log


def load_bundle_verts(b):
    import trimesh
    return [np.asarray(trimesh.load(str(p), force="mesh").vertices) for p in b["object_mesh_paths"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--hands", required=True)
    ap.add_argument("--poses", required=True, help="npz with world_T_rect + valid (vo_cuvslam.npz / gt_cam_register.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--sides", default="left,right")
    ap.add_argument("--fill", default="object", choices=("object", "none"),
                    help="object: hands invisible while in contact with a moving object ride with that object")
    a = ap.parse_args()
    b = dict(np.load(a.bundle, allow_pickle=False))
    pz = np.load(a.poses, allow_pickle=False)
    W, pv = pz["world_T_rect"].astype(np.float64), pz["valid"].astype(bool)
    hw, hz = hands_in_world(a.hands, W, pv, a.min_conf, tuple(a.sides.split(",")))
    N = int(b["num_frames"])
    if int(hz["num_frames"]) != N or int(hz["episode_index"]) != int(b["episode_index"]):
        raise SystemExit(f"episode/frames mismatch: hands ep {int(hz['episode_index'])} T={int(hz['num_frames'])} "
                         f"vs bundle ep {int(b['episode_index'])} N={N}")
    if "gt_cam_register" in os.path.basename(a.poses) and not bool(b.get("dev_only", False)):
        raise SystemExit("gt_cam_register poses (public GT) may only go into dev_only bundles")
    info = {}
    if a.fill == "object":
        ov = b["object_valid"].astype(bool) if "object_valid" in b else np.ones(b["object_pose"].shape[:2], bool)
        verts = load_bundle_verts(b)
    for s, (J, v, c) in hw.items():
        fill_log = []
        if a.fill == "object" and v.any():
            J, v, filled, fill_log = object_anchored_fill(J, v, b["object_pose"].astype(np.float64), ov, verts)
            c = np.where(filled[:, None], 0.3, c)
            b[f"hand_{s}_filled"] = filled
        b.pop(f"hand_{s}_joints_wxyz", None)
        if not v.any():
            for k in ("joints", "valid", "conf"):
                b.pop(f"hand_{s}_{k}", None)
            info[s] = "no valid frame -> omitted"
            continue
        b[f"hand_{s}_joints"] = J
        b[f"hand_{s}_valid"] = v
        b[f"hand_{s}_conf"] = c.astype(np.float32)
        info[s] = {"valid_frames": int(v.sum()), "of": N, "object_anchored_fill": fill_log}
    prov = json.loads(str(b["provenance"])) if "provenance" in b else {}
    prov["hands"] = {"source": "MANO-free MediaPipe 21-kp + stereo lift (track3/hands)", "hands_rect": a.hands,
                     "poses": a.poses, "min_conf": a.min_conf, "hands_provenance": json.loads(str(hz["provenance"])),
                     "per_side": info}
    b["provenance"] = np.array(json.dumps(prov))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **b)
    sys.path.insert(0, os.path.dirname(HERE))
    from sharpa_task.bundle import load_bundle
    bb = load_bundle(a.out)
    print(a.out, {s: int(h["valid"].sum()) for s, h in bb.hands.items()}, "of", bb.num_frames, "frames valid;",
          json.dumps({s: i.get("object_anchored_fill") if isinstance(i, dict) else i for s, i in info.items()}))


if __name__ == "__main__":
    main()
