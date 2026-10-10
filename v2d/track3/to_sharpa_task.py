#!/usr/bin/env python3
"""Track 3: per-episode perception bundle -> FlashCHORD floating-Sharpa training task.

    python to_sharpa_task.py BUNDLE.npz --out-root /mnt/secondary/v2d/t3/tasks/<variant> [--verify]

(re-execs itself with /mnt/secondary/v2d/envs/retarget/bin/python if robotic_grounding is not importable)

Input: one ``t3_perception_v1`` bundle (see sharpa_task/bundle.py): object world poses per video frame in
roster order, our own object meshes, optional MANO hand joints, optional table plane.

Output (all under <out-root>/human_motion_data/v2d_track3/):
  objects/episode_XXXXXX/<name>.obj|.urdf   budgeted mesh (== the mesh we submit) + single-link URDF
                                            (visual+collision = that mesh, mass/inertia from its hull,
                                            COM at the hull centroid)
  loaded/sequence_id=episode_XXXXXX/robot_name=sharpa_wave/*.parquet      MANO + objects (IK input)
  processed/sequence_id=episode_XXXXXX/robot_name=sharpa_wave/*.parquet   + Sharpa IK  (task.parquet)
  reconstructed_stage/episode_XXXXXX_sharpa_wave_support.usda             Cube support at the table plane
  manifests/episode_XXXXXX.json                                           everything needed downstream
and, for non-dev bundles, <out-root>/submit_meshes/episode_XXXXXX/<name>.obj (byte-identical copies) for
``tools/pack_submission.py --meshes``.

Contract with the evaluator / packer (checked here and by sharpa_task/verify_task.py):
  * fps = 20.0 and T = video frames; train with task.motion_speed=1.0 => the evaluator emits exactly N steps.
  * object_body_names in roster order (packer slot = body index; names must match the roster).
  * 'episode_XXXXXX' is the FIRST episode_<digits> match in the reference parquet path (harness regex).
  * the simulated mesh is exactly the submitted mesh (budgeted to the packer's 4096/4096 limits up front).
The sim world is a gravity-aligned re-expression of the perception world (z up, table horizontal at
z = table height > 0 because the training scene has a ground plane at z = 0). Track 3 scoring is invariant
to the world frame (one rigid frame-0 alignment per episode), so this changes nothing that is scored; the
transform is recorded in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

RETARGET_PY = "/mnt/secondary/v2d/envs/retarget/bin/python"
FLASH_PY = "/mnt/secondary/v2d/envs/flash_chord/bin/python"
KIT_PY = "/mnt/secondary/v2d/venv-kit/bin/python"
V2D_RG = "/mnt/secondary/v2d/video_to_data/robotic_grounding"
HERE = Path(__file__).resolve().parent


def _ensure_env() -> None:
    try:
        import robotic_grounding.retarget.data_logger  # noqa: F401
        import rtree  # noqa: F401
    except ImportError:
        if os.environ.get("_T3_REEXEC"):
            raise
        os.environ["_T3_REEXEC"] = "1"
        os.execv(RETARGET_PY, [RETARGET_PY, str(Path(__file__).resolve()), *sys.argv[1:]])


def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--table-height", default="auto", help="sim table height (m) or 'auto' (keep if in [0.3,2], else 0.70)")
    ap.add_argument("--no-recenter-xy", action="store_true", help="keep input x/y (default: frame-0 object centroid -> x=y=0)")
    ap.add_argument("--snap-mode", choices=("rest", "lift", "none"), default="rest",
                    help="rest: objects within --snap-tol above the table at frame 0 are lowered onto it (constant "
                         "shift) + lift; lift: only push penetrating frames up; none: report only")
    ap.add_argument("--snap-tol", type=float, default=0.02)
    ap.add_argument("--clearance", type=float, default=0.001, help="object bottom clearance above the support (m)")
    ap.add_argument("--hands", choices=("auto", "input", "placeholder"), default="auto",
                    help="auto: per side, bundle hands if present else the DEV placeholder")
    ap.add_argument("--contact-threshold", type=float, default=0.02)
    ap.add_argument("--support-thickness", type=float, default=0.05)
    ap.add_argument("--support-margin", type=float, default=0.25)
    ap.add_argument("--mano-to-robot-scale", type=float, default=1.2)
    ap.add_argument("--ik-extra", default="--smooth_reference", help="extra args for ego_recon_to_sharpa.py")
    ap.add_argument("--settle-static-start", action="store_true",
                    help="simulate the frame-0 scene (objects + support, training CoACD hulls, CPU MuJoCo, 1 s) and "
                         "move each object's initial static period onto its settled pose (blended out over "
                         "0.5 s once it starts moving). Removes reset 'pops' from hull overlap (e.g. lid on pot)")
    ap.add_argument("--settle-max-dp", type=float, default=0.04, help="refuse settling deltas larger than this (m)")
    ap.add_argument("--settle-max-rot", type=float, default=20.0, help="... or than this (deg)")
    ap.add_argument("--skip-ik", action="store_true")
    ap.add_argument("--verify", action="store_true", help="run sharpa_task/verify_task.py (flash_chord env) at the end")
    ap.add_argument("--verify-scene", action="store_true", help="also build the Newton scene on CPU (CoACD; slower)")
    return ap.parse_args()


def settle_static_start(args, manifest, b, pos, quat, radius, objs, center_xy, size_xy, table_z, in_json, warn):
    """Replace each resting object's initial static poses by its settled sim pose (in place on pos/quat)."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R, Slerp

    from sharpa_task import TASK_FPS
    from sharpa_task import hands as H

    N, B = pos.shape[:2]
    _, sig = H.motion_segments(pos, quat, radius, TASK_FPS)
    first_motion = [int(np.argmax(sig[:, k] > 0.05)) if (sig[:, k] > 0.05).any() else N for k in range(B)]
    min_static = int(round(0.5 * TASK_FPS))
    static = [fm < min_static for fm in first_motion]  # moving/held at frame 0 -> fixed in the settle sim
    th = args.support_thickness
    inp = {
        "objects": [{"name": o["name"], "obj": o["obj"], "urdf": o["urdf"]} for o in objs],
        "pos0": pos[0].tolist(), "quat0": quat[0].tolist(), "static": static,
        "boxes": [[[float(center_xy[0]), float(center_xy[1]), table_z - 0.5 * th], [float(size_xy[0]), float(size_xy[1]), th]]],
    }
    in_json.write_text(json.dumps(inp))
    out_json = in_json.with_name(in_json.name.replace("_in.json", "_out.json"))
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4",
           "FLASH_CHORD_CACHE_DIR": os.environ.get("FLASH_CHORD_CACHE_DIR", "/mnt/secondary/v2d/flash_cache"),
           "WARP_CACHE_PATH": os.environ.get("WARP_CACHE_PATH", "/mnt/secondary/v2d/warp_cache")}
    r = subprocess.run(["nice", "-n", "10", FLASH_PY, str(HERE / "sharpa_task" / "settle_check.py"), "--raw",
                        str(in_json), "--out", str(out_json)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"settle failed:\n{r.stderr[-3000:]}")
    rep = json.loads(out_json.read_text())
    blend = int(round(0.5 * TASK_FPS))
    applied = {}
    for k, name in enumerate(b.object_names):
        if static[k]:
            continue
        d = rep["drift_no_hands"][name]
        last = d[max(d, key=lambda s: float(s.rstrip("s")))]
        if last["dp_mm"] / 1000 > args.settle_max_dp or last["rot_deg"] > args.settle_max_rot:
            warn(f"{name}: settle drift {last} exceeds limits; NOT applied (pose/geometry likely inconsistent)")
            continue
        r0 = R.from_quat(quat[0, k], scalar_first=True)
        rf = R.from_quat(rep["final_wxyz"][k], scalar_first=True)
        DR = rf * r0.inv()
        Dt = np.asarray(rep["final_pos"][k]) - DR.apply(pos[0, k])
        s0 = first_motion[k]
        for t in range(N):
            w = 1.0 if t < s0 else max(0.0, 1.0 - (t - s0 + 1) / blend)
            if w <= 0.0:
                break
            p_full = DR.apply(pos[t, k]) + Dt
            q_t = R.from_quat(quat[t, k], scalar_first=True)
            q_full = DR * q_t
            pos[t, k] = (1 - w) * pos[t, k] + w * p_full
            quat[t, k] = Slerp([0.0, 1.0], R.concatenate([q_t, q_full]))([w])[0].as_quat(scalar_first=True)
        applied[name] = {"static_frames": s0, "blend_frames": blend, **last}
    manifest["settle"] = {"report": {k: v for k, v in rep.items() if k not in ("final_pos", "final_wxyz")},
                          "first_motion_frame": dict(zip(b.object_names, first_motion)), "applied": applied}
    print(f"[t3]   settle: applied {applied}; frame-0 hull contacts {rep['frame0_min_contact_dist_mm']}")


def main() -> int:
    _ensure_env()
    args = parse_args()
    sys.path.insert(0, str(HERE))
    import numpy as np
    import trimesh
    from robotic_grounding.retarget.data_logger import ManoSharpaData
    from robotic_grounding.retarget.naming import make_usd_safe
    from scipy.spatial.transform import Rotation as R

    from sharpa_task import DATASET_NAME, ROBOT_NAME, TASK_FPS, geom
    from sharpa_task import hands as H
    from sharpa_task.bundle import load_bundle
    from sharpa_task.checks import pair_penetration_report, support_gap_report
    from sharpa_task.usd import write_support_usda

    t_start = time.time()
    b = load_bundle(args.bundle)
    seq, N, B = b.sequence_id, b.num_frames, len(b.object_names)
    root = args.out_root.resolve()
    if re.search(r"episode_\d", str(root)):
        raise SystemExit(f"--out-root must not contain 'episode_<digit>' (harness takes the first match): {root}")
    ds_root = root / "human_motion_data" / DATASET_NAME
    obj_dir = ds_root / "objects" / seq
    loaded_root, processed_root = ds_root / "loaded", ds_root / "processed"
    stage_dir, manifest_dir = ds_root / "reconstructed_stage", ds_root / "manifests"
    for d in (obj_dir, loaded_root, processed_root, stage_dir, manifest_dir):
        d.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "sequence_id": seq,
        "episode_index": b.episode_index,
        "num_frames": N,
        "fps": TASK_FPS,
        "object_body_names": b.object_names,
        "bundle": str(b.path.resolve()),
        "bundle_sha256": _sha256(b.path),
        "bundle_provenance": b.provenance,
        "dev_only": b.dev_only,
        "warnings": [],
    }
    warn = manifest["warnings"].append
    print(f"[t3] {seq}: N={N} B={B} objects={b.object_names} dev_only={b.dev_only}")

    # ---------------------------------------------------------------- 1. meshes, URDFs, mass properties
    meshes, verts, objs = [], [], []
    for k, (name, src) in enumerate(zip(b.object_names, b.object_mesh_paths)):
        obj = obj_dir / f"{name}.obj"
        res = subprocess.run([KIT_PY, str(HERE / "sharpa_task" / "mesh_prep.py"), src, str(obj)],
                             capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "4"})
        if res.returncode != 0:
            raise SystemExit(f"mesh_prep failed for {name}: {res.stdout}\n{res.stderr}")
        info = json.loads(res.stdout.strip().splitlines()[-1])
        m = trimesh.load(obj, force="mesh", process=False)
        mp = geom.mass_properties(np.asarray(m.vertices), np.asarray(m.faces), name,
                                  None if not np.isfinite(b.object_mass[k]) else float(b.object_mass[k]))
        urdf = obj_dir / f"{name}.urdf"
        I = mp["inertia"]
        urdf.write_text(
            f"""<?xml version="1.0"?>
<!-- {seq} / {name}: generated by v2d/track3/to_sharpa_task.py. visual == collision == the submitted mesh;
     mass source: {mp['mass_source']}; COM = convex-hull centroid; inertia = solid hull scaled to mass. -->
<robot name="{make_usd_safe(name)}">
  <link name="{make_usd_safe(name)}">
    <inertial>
      <origin xyz="{mp['com'][0]:.9g} {mp['com'][1]:.9g} {mp['com'][2]:.9g}" rpy="0 0 0"/>
      <mass value="{mp['mass']:.6g}"/>
      <inertia ixx="{I[0,0]:.9g}" ixy="{I[0,1]:.9g}" ixz="{I[0,2]:.9g}" iyy="{I[1,1]:.9g}" iyz="{I[1,2]:.9g}" izz="{I[2,2]:.9g}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="./{obj.name}" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="./{obj.name}" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
"""
        )
        meshes.append(m)
        verts.append(np.asarray(m.vertices, np.float64))
        objs.append({
            "name": name, "source_mesh": src, "obj": str(obj), "urdf": str(urdf), "obj_sha256": info["obj_sha256"],
            "vertices": info["vertices"], "faces": info["faces"], "input_faces": info["input_faces"],
            "watertight": info["watertight"], "mass_kg": mp["mass"], "mass_source": mp["mass_source"],
            "com": mp["com"].tolist(), "inertia_diag": np.diag(I).tolist(), "radius": mp["radius"],
        })
        print(f"[t3]   {name}: {info['input_faces']}f -> {info['vertices']}v/{info['faces']}f, "
              f"mass {mp['mass']:.3f} kg ({mp['mass_source']}), radius {mp['radius']:.3f} m")
    manifest["objects"] = objs
    radius = np.array([o["radius"] for o in objs])

    # ---------------------------------------------------------------- 2. fill invalid frames
    pos = b.object_pos.copy()
    quat = b.object_quat.copy()
    filled = []
    for k in range(B):
        pos[:, k], quat[:, k], n_fill = geom.fill_invalid_poses(pos[:, k], quat[:, k], b.object_valid[:, k])
        filled.append(n_fill)
    quat = geom.hemisphere_continuous(quat, axis=0)
    manifest["filled_frames"] = dict(zip(b.object_names, filled))

    # ---------------------------------------------------------------- 3. gravity alignment + table height
    if b.table_plane is not None:
        n = b.table_plane[:3] / np.linalg.norm(b.table_plane[:3])
        d = float(b.table_plane[3] / np.linalg.norm(b.table_plane[:3]))
        if b.up is not None and float(n @ (b.up / np.linalg.norm(b.up))) < np.cos(np.radians(10)):
            warn("table normal and gravity-up differ by >10 deg; using the table normal")
        up_in = n
    else:
        up_in = (b.up / np.linalg.norm(b.up)) if b.up is not None else np.array([0.0, 0.0, 1.0])
        d = None
    Ra = geom.rotation_between(up_in, np.array([0.0, 0.0, 1.0]))
    pos, quat = geom.apply_rigid(Ra, np.zeros(3), pos, quat)
    zaxis = np.array([0.0, 0.0, 1.0])
    if d is not None:
        table_in = -d  # plane n.x + d = 0 -> height -d along n (n -> +z)
        table_src = "bundle_plane"
    else:
        bot = geom.bottom_heights(verts, pos, quat, zaxis)
        table_in = geom.estimate_table_height(bot, geom.speeds(pos, TASK_FPS))
        table_src = "estimated_from_object_bottoms"
    if args.table_height == "auto":
        table_z = table_in if 0.3 <= table_in <= 2.0 else 0.70
    else:
        table_z = float(args.table_height)
    t_align = np.array([0.0, 0.0, table_z - table_in])
    if not args.no_recenter_xy:
        c0 = np.mean([geom.world_vertices(verts[k], pos[0, k], quat[0, k]).mean(0) for k in range(B)], 0)
        t_align[:2] = -c0[:2]
    pos = pos + t_align
    manifest["sim_from_input"] = {"R": Ra.tolist(), "t": t_align.tolist(),
                                  "note": "x_sim = R x_in + t; q_sim = R * q_in"}
    manifest["table"] = {"z": table_z, "z_input_frame": table_in, "source": table_src}
    print(f"[t3]   table: {table_src} z_in={table_in:.4f} -> sim z={table_z:.4f}")

    # ---------------------------------------------------------------- 4. support snapping
    bot = geom.bottom_heights(verts, pos, quat, zaxis)
    gap0 = bot[0] - table_z
    snaps = {}
    if args.snap_mode == "rest":
        for k in range(B):
            if args.clearance < gap0[k] <= args.snap_tol:
                dz = -(gap0[k] - args.clearance)
                pos[:, k, 2] += dz
                snaps[b.object_names[k]] = {"constant_dz": float(dz)}
    if args.snap_mode in ("rest", "lift"):
        bot = geom.bottom_heights(verts, pos, quat, zaxis)
        lift = np.clip(table_z + args.clearance - bot, 0.0, None)
        pos[:, :, 2] += lift
        for k in range(B):
            if lift[:, k].max() > 0:
                s = snaps.setdefault(b.object_names[k], {})
                s.update(lift_frames=int((lift[:, k] > 0).sum()), lift_max=float(lift[:, k].max()),
                         lift_frame0=float(lift[0, k]))
                if lift[:, k].max() > args.snap_tol:
                    warn(f"{b.object_names[k]}: lifted up to {lift[:, k].max()*100:.1f} cm out of the table "
                         "(table plane or pose likely wrong)")
    manifest["snap"] = {"mode": args.snap_mode, "tol": args.snap_tol, "clearance": args.clearance,
                        "frame0_gap_before": dict(zip(b.object_names, gap0.tolist())), "applied": snaps}

    # ---------------------------------------------------------------- 4b. support box (axis-aligned Cube)
    cent = np.stack([[geom.world_vertices(verts[k], pos[t, k], quat[t, k]).mean(0) for k in range(B)]
                     for t in range(0, N, max(1, N // 50))]).reshape(-1, 3)
    lo = cent[:, :2].min(0) - radius.max() - args.support_margin
    hi = cent[:, :2].max(0) + radius.max() + args.support_margin
    size_xy = np.maximum(hi - lo, 0.8)
    center_xy = 0.5 * (lo + hi)

    # ---------------------------------------------------------------- 4c. optional: settle the static start
    if args.settle_static_start:
        settle_static_start(args, manifest, b, pos, quat, radius, objs, center_xy, size_xy, table_z,
                            manifest_dir / f"{seq}_settle_in.json", warn)
        quat = geom.hemisphere_continuous(geom.qnorm(quat), axis=0)

    # ---------------------------------------------------------------- 5. hands
    tmpl = H.load_template()
    hand_src, hand_data = {}, {}
    placeholder = None
    for side in H.SIDES:
        use_input = side in b.hands and args.hands in ("auto", "input")
        if args.hands == "input" and side not in b.hands:
            raise SystemExit(f"--hands input but the bundle has no {side} hand")
        if use_input:
            h = b.hands[side]
            J = h["joints"].copy()
            v = h["valid"]
            if not v.any():
                raise SystemExit(f"{side} hand has no valid frame")
            tt = np.arange(N, dtype=float)
            vi = np.flatnonzero(v).astype(float)
            for j in range(21):
                for c in range(3):
                    J[:, j, c] = np.interp(tt, vi, J[v, j, c])
            J = J @ Ra.T + t_align
            if h["quats"] is not None:
                Q = h["quats"].copy()
                for j in range(21):
                    _, Q[:, j], _ = geom.fill_invalid_poses(J[:, j], Q[:, j], v)
                Q = geom.apply_rigid(Ra, np.zeros(3), np.zeros((N, 21, 3)), Q)[1]
                src = "bundle_mano"
            else:
                Q = H.orientations_from_keypoints(J, tmpl, side)
                src = "bundle_keypoints+kabsch_template"
            hand_data[side] = {"joints": J, "quats": geom.qnorm(Q), "allowed": None}
            hand_src[side] = src
        else:
            if placeholder is None:
                placeholder = H.placeholder_hands(pos, quat, verts, radius, table_z, TASK_FPS, tmpl)
            p = placeholder[side]
            allowed = np.zeros((N, B), bool)
            if p["object"] is not None:
                allowed[:, p["object"]] = p["contact_mask"]
            hand_data[side] = {"joints": p["joints"], "quats": p["quats"], "allowed": allowed}
            hand_src[side] = "DEV_PLACEHOLDER"
    if "DEV_PLACEHOLDER" in hand_src.values():
        warn("placeholder hands in use: DEV ONLY (not a real hand estimate)")
        manifest["placeholder"] = placeholder["info"]
    manifest["hands"] = hand_src
    print(f"[t3]   hands: {hand_src}" + (f" placeholder={placeholder['info']}" if placeholder else ""))

    # ---------------------------------------------------------------- 6. contacts
    contacts = {}
    for side in H.SIDES:
        hd = hand_data[side]
        contacts[side] = H.compute_contacts(hd["joints"], pos, quat, meshes, hd["allowed"], args.contact_threshold)
    manifest["contact_frames"] = {s: contacts[s]["active_frames"] for s in H.SIDES}

    # ---------------------------------------------------------------- 7. loaded ManoSharpaData (IK input)
    object_name = "__".join(b.object_names)
    kw = dict(
        sequence_id=seq, raw_motion_file=str(b.path.resolve()), robot_name=ROBOT_NAME, fps=TASK_FPS,
        mano_flat_hand_mean=True, mano_center_idx=None, mano_to_robot_scale=float(args.mano_to_robot_scale),
        mano_right_betas=[0.0] * 10, mano_left_betas=[0.0] * 10, mano_link_names=list(H.MANO_LINK_NAMES),
        right_robot_finger_joint_names=[], right_robot_frame_names=[], right_robot_frame_task_names=[],
        left_robot_finger_joint_names=[], left_robot_frame_names=[], left_robot_frame_task_names=[],
        object_name=object_name, safe_object_name=make_usd_safe(object_name),
        object_body_names=list(b.object_names), safe_object_body_names=[make_usd_safe(x) for x in b.object_names],
        object_mesh_paths=[o["obj"] for o in objs], object_urdf_paths=[o["urdf"] for o in objs],
        object_mesh_radius=[float(r) for r in radius],
        object_articulation=[0.0] * N,
        object_root_axis_angle=R.from_quat(quat[:, 0], scalar_first=True).as_rotvec().tolist(),
        object_root_position=pos[:, 0].tolist(),
        object_body_position=pos.tolist(),
        object_body_wxyz=geom.qnorm(quat).tolist(),
    )
    for side in H.SIDES:
        J, Q, c = hand_data[side]["joints"], hand_data[side]["quats"], contacts[side]
        kw.update({
            f"mano_{side}_trans": J[:, 0].tolist(),
            f"mano_{side}_global_orient": R.from_quat(Q[:, 0], scalar_first=True).as_rotvec().tolist(),
            f"mano_{side}_finger_pose": np.zeros((N, 45)).tolist(),
            f"mano_{side}_joints": J.tolist(),
            f"mano_{side}_joints_wxyz": Q.tolist(),
            f"mano_{side}_fitting_err": [0.0] * N,
            f"mano_{side}_tips_distance": c["tips_distance"].tolist(),
            f"mano_{side}_link_contact_positions": c["link_contact_positions"].tolist(),
            f"mano_{side}_link_contact_normals": c["link_contact_normals"].tolist(),
            f"mano_{side}_object_contact_positions": c["object_contact_positions"].tolist(),
            f"mano_{side}_object_contact_normals": c["object_contact_normals"].tolist(),
            f"mano_{side}_object_contact_part_ids": c["object_contact_part_ids"].tolist(),
        })
    ManoSharpaData(**kw).save_to_parquet(root_path=str(loaded_root), partition_cols=["sequence_id", "robot_name"])
    loaded_dir = loaded_root / f"sequence_id={seq}" / f"robot_name={ROBOT_NAME}"
    print(f"[t3]   loaded parquet: {loaded_dir}")

    # ---------------------------------------------------------------- 8. support USDA
    usda = stage_dir / f"{seq}_{ROBOT_NAME}_support.usda"
    write_support_usda(usda, center_xy, size_xy, table_z, args.support_thickness, name=f"{seq}_table")
    manifest["support"] = {"usda": str(usda), "top_z": table_z, "center_xy": center_xy.tolist(),
                           "size_xy": size_xy.tolist(), "thickness": args.support_thickness}

    # ---------------------------------------------------------------- 9. frame-0 geometry checks (pre-IK)
    manifest["frame0_support"] = support_gap_report(verts, pos[0], quat[0], b.object_names, table_z,
                                                    center_xy, size_xy)
    manifest["frame0_pairs"] = pair_penetration_report(meshes, pos[0], quat[0], b.object_names)
    for name, r in manifest["frame0_support"].items():
        print(f"[t3]   frame0 {name}: bottom-support gap {r['gap_m']*1000:+.2f} mm (inside footprint: {r['inside_footprint']})")
    for pair, r in manifest["frame0_pairs"].items():
        print(f"[t3]   frame0 pair {pair}: min surface dist {r['min_dist_m']*1000:.1f} mm, "
              f"max pseudo-penetration {r['max_penetration_m']*1000:.1f} mm")

    # ---------------------------------------------------------------- 10. Sharpa IK (ego_recon_to_sharpa.py)
    processed_dir = processed_root / f"sequence_id={seq}" / f"robot_name={ROBOT_NAME}"
    if not args.skip_ik:
        log = manifest_dir / f"{seq}_ik.log"
        cmd = ["nice", "-n", "10", RETARGET_PY, "scripts/retarget/ego_recon_to_sharpa.py",
               "--input_dir", str(loaded_root), "--output_dir", str(processed_root), "--sequence_id", seq,
               "--save", "--device", "cpu", "--mano_to_robot_scale", str(args.mano_to_robot_scale),
               *args.ik_extra.split()]
        env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": ""}
        t0 = time.time()
        with open(log, "w") as fh:
            fh.write(" ".join(cmd) + "\n")
            fh.flush()
            r = subprocess.run(cmd, cwd=V2D_RG, stdout=fh, stderr=subprocess.STDOUT, env=env)
        if r.returncode != 0:
            raise SystemExit(f"IK failed ({r.returncode}); see {log}")
        manifest["ik"] = {"cmd": cmd, "seconds": round(time.time() - t0, 1), "log": str(log)}
        import pyarrow.dataset as pads

        t = pads.dataset(str(processed_dir), format="parquet").to_table(
            columns=["robot_right_frame_task_errors", "robot_left_frame_task_errors",
                     "robot_right_wrist_position", "robot_left_wrist_position"])
        for side in H.SIDES:
            err = np.asarray(t[f"robot_{side}_frame_task_errors"][0].as_py())
            wp_ = np.asarray(t[f"robot_{side}_wrist_position"][0].as_py())
            if wp_.shape[0] != N:
                raise SystemExit(f"IK produced {wp_.shape[0]} frames != {N}")
            manifest["ik"][f"{side}_task_error_mean"] = float(err.mean())
            manifest["ik"][f"{side}_task_error_p95"] = float(np.percentile(err, 95))
        print(f"[t3]   IK done in {manifest['ik']['seconds']} s: task err mean R/L "
              f"{manifest['ik']['right_task_error_mean']:.4f}/{manifest['ik']['left_task_error_mean']:.4f}")
    parquet_files = sorted(processed_dir.glob("*.parquet"))
    manifest["task_parquet"] = str(processed_dir)
    manifest["task_parquet_files"] = [str(p) for p in parquet_files]
    m = re.search(r"episode_(\d+)", str(processed_dir))
    if not m or int(m.group(1)) != b.episode_index:
        raise SystemExit(f"harness episode regex would read {m.group(0) if m else None} from {processed_dir}")
    manifest["harness_episode_index"] = int(m.group(1))

    # ---------------------------------------------------------------- 11. submit meshes (never for dev bundles)
    if not b.dev_only:
        sub = root / "submit_meshes" / seq
        sub.mkdir(parents=True, exist_ok=True)
        for o in objs:
            dst = sub / f"{o['name']}.obj"
            shutil.copyfile(o["obj"], dst)
            assert _sha256(dst) == o["obj_sha256"]
        manifest["submit_meshes"] = str(sub)
    manifest["train_overrides"] = [
        "experiment=sharpa_flash_sac", f"task.parquet='{processed_dir}'", "task.motion_speed=1.0",
    ]
    manifest["seconds"] = round(time.time() - t_start, 1)
    mpath = manifest_dir / f"{seq}.json"
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"[t3]   manifest: {mpath}  ({manifest['seconds']} s)")
    for w in manifest["warnings"]:
        print(f"[t3]   WARNING: {w}")

    if args.verify:
        cmd = [FLASH_PY, str(HERE / "sharpa_task" / "verify_task.py"), str(mpath)]
        if args.verify_scene:
            cmd.append("--scene")
        env = {**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"}
        return subprocess.run(cmd, env=env).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
