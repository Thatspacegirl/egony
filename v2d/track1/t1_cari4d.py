#!/usr/bin/env python3
"""Official CARI4D pipeline (nvidia-isaac/video_to_data @ a709404e, modules/v2d_cari4d, weights nvidia/cari4d_commercial
@ 1f7287ac + facebook/sam-3d-body-dinov3 @ 11aaa346) on one Track 1 / FORM-HOI val episode, on the host (no Docker),
plus the conversion of its three human/object estimates into Track 1 episode NPZs.

Env: /mnt/secondary/v2d/envs/t1-cari4d (source v2d_env.sh).  Weights: /mnt/secondary/v2d/weights/cari4d.

  prep   (CPU)  window clip + CARI4D mask H5 from our SAM3 masks (t1_masks.py, stride 1), and the object mesh:
                <out>/inputs/episode_X/episode_X.0.color.mp4  = video frames [s, e) of t1_items.window() (the scored span
                    - 60 / + 30 frames), re-encoded LOSSLESSLY (libx264 -qp 0, yuv420p, same as the source chroma)
                <out>/inputs/episode_X/episode_X_masks_k0.h5  = lib/pack_masks.py schema (v2d.cari4d.wild_masks.v1),
                    masks upsampled from scale 0.5 (bilinear, > 0.5)
                <out>/inputs/episode_X/object.glb            = the metric object mesh of --mesh-source
  run    (GPU)  stages 01-07 of lib/run_inference.py with its DEFAULT settings (MoGe-2 depth + intrinsics, export, SAM 3D
                Body init, depth alignment, FoundationPose register-then-track, CoCoNet, 300-step contact refinement over
                the full clip).  Stage 08 (videos) is skipped.  Every command is built by the toolkit's own
                run_inference(); we only point SAM3D_SOURCE_ROOT / FoundationPose at the host checkout.
  decode (CPU)  refined.pth -> three estimates, each converted EXACTLY (no fitting) to kit MHR params:
                  init    = bundle['in']          SAM 3D Body per frame (depth-aligned) + FoundationPose object track
                  coconet = bundle['pr_initial']  CoCoNet prediction
                  refined = bundle['pr']          after contact-guided refinement (the official output)
                CARI4D vertices = flip * MHRHead.mhr_forward(trans=0, ...)/100 + mhr_trans  (lib_mhr/mhr_layer.py);
                the head's own model params (204 = 136 pose + 68 scales) are the kit's pose/scales, with the kit root
                translation pose[:, :3] = 10 * flip * mhr_trans.  Checked against the MHRLayer vertices (decode.json).
                Object: pose_abs [T,4,4] maps the export's output_aligned.glb to camera space (metres).
                Writes <out>/episode_X/t1/{init,coconet,refined}/episode_X.npz (+ mesh) indexed by VIDEO frame; frames
                outside the window hold the nearest window value.

  python t1_cari4d.py prep   --split val --episode 3 --mesh-source trellis --out /mnt/secondary/v2d/t1/cari4d_trellis/val
  python t1_cari4d.py run    --split val --episode 3 --mesh-source trellis --out /mnt/secondary/v2d/t1/cari4d_trellis/val
  python t1_cari4d.py decode --split val --episode 3 --mesh-source trellis --out /mnt/secondary/v2d/t1/cari4d_trellis/val
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_items as I  # noqa: E402

M = Path("/mnt/secondary/v2d/video_to_data/reconstruction/modules")
WEIGHTS = Path("/mnt/secondary/v2d/weights/cari4d")
R = Path("/mnt/secondary/v2d/t1")
FLIP = np.array([1.0, -1.0, -1.0])
# mask source: "masks" (our SAM3, t1_masks.py; ALWAYS for Track 1) or "masks_fh" (FORM-HOI released masks, val only,
# t1_masks_formhoi.py).  Mesh dirs carry the same suffix (mesh_sam3do_fh, ...) so the two never mix.
MASKS = os.environ.get("T1_MASKS", "masks")
MS = MASKS[len("masks"):]
assert MASKS in ("masks", "masks_fh"), MASKS


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def atomic_json(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def mesh_of(source, split, ep):
    """Metric object mesh (metres) of a mesh source for an episode."""
    E = f"episode_{int(ep):06d}"
    cands = {"trellis": [R / f"mesh_trellis{MS}" / split / E / "mesh_0.ply"],
             "sam3do": [R / f"mesh_sam3do{MS}" / split / E / "mesh.glb"],
             "trellis2": [R / f"mesh_trellis2{MS}" / split / E / "mesh_0.ply"]}[source]
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(f"no {source} mesh for {split} ep {ep}: {cands}")


def unpack(bits, W):
    return np.unpackbits(bits, axis=-1, count=W).astype(bool)


# ------------------------------------------------------------------------------------------------ prep
def cmd_prep(a):
    import cv2
    import h5py
    it = I.item(a.split, a.episode)
    E = f"episode_{a.episode:06d}"
    if MASKS != "masks":
        assert a.split == "val", "FORM-HOI masks are a val-only input"
    Mk = np.load(R / MASKS / a.split / E / "masks.npz")
    frames = np.asarray(Mk["frames"])
    s, e = int(frames[0]), int(frames[-1]) + 1
    assert np.array_equal(frames, np.arange(s, e)), "CARI4D needs stride-1 masks (t1_masks.py --stride 1)"
    W = int(Mk["W"]); FW, FH = [int(x) for x in Mk["full_wh"]]
    od = Path(a.out) / "inputs" / E
    od.mkdir(parents=True, exist_ok=True)
    video = od / f"{E}.0.color.mp4"
    if not video.is_file():
        tmp = od / f".{E}.tmp.mp4"
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", it["video"], "-vf",
               f"select=between(n\\,{s}\\,{e - 1})", "-fps_mode", "passthrough", "-c:v", "libx264", "-qp", "0",
               "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", str(tmp)]
        subprocess.run(cmd, check=True)
        os.replace(tmp, video)
    # verify frame count and pixel identity of the first/last frames vs the source
    cap = cv2.VideoCapture(str(video)); n = 0; first = last = None
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if n == 0:
            first = im
        last = im; n += 1
    assert n == e - s, f"trimmed clip has {n} frames, expected {e - s}"
    src = cv2.VideoCapture(it["video"]); k = 0; sf = sl = None
    while k < e:
        ok, im = src.read()
        if not ok:
            break
        if k == s:
            sf = im
        if k == e - 1:
            sl = im
        k += 1
    dmax = max(int(np.abs(first.astype(int) - sf.astype(int)).max()), int(np.abs(last.astype(int) - sl.astype(int)).max()))
    h5 = od / f"{E}_masks_k0.h5"
    if not h5.is_file():
        tmp = od / f".{E}_masks_k0.tmp.h5"
        with h5py.File(tmp, "w") as f:
            f.attrs["schema"] = "v2d.cari4d.wild_masks.v1"
            f.attrs["sequence"] = E
            f.attrs["frame_count"] = n
            f.attrs["width"] = FW
            f.attrs["height"] = FH
            g = f.require_group(E)
            for i in range(n):
                for key, arr in (("person_mask", Mk["human"]), ("obj_rend_mask", Mk["object"])):
                    m = unpack(arr[i], W).astype(np.uint8) * 255
                    m = (cv2.resize(m, (FW, FH), interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8) * 255
                    g.create_dataset(f"{i:06d}-k0.{key}.png", data=m, compression="lzf", shuffle=True)
        os.replace(tmp, h5)
    info = {"split": a.split, "episode": a.episode, "sequence_id": it["sequence_id"], "source_video": it["video"],
            "window": [s, e], "T_video": it["T"], "frames": n, "first_last_frame_maxabs_diff": dmax,
            "video_sha256": sha256(video), "masks_sha256": sha256(h5)}
    atomic_json(info, od / "window.json")
    print(json.dumps(info), flush=True)
    return od


# ------------------------------------------------------------------------------------------------ run
def mesh_ref_frame(source, split, ep):
    """video frame the mesh was generated from (its least-occluded view)."""
    E = f"episode_{int(ep):06d}"
    j = {"trellis": R / f"mesh_trellis{MS}" / split / E / "trellis.json", "sam3do": R / f"mesh_sam3do{MS}" / split / E / "frame.json",
         "trellis2": R / f"mesh_trellis2{MS}" / split / E / "trellis2.json"}[source]
    d = json.load(open(j))
    return int(d["views"][0]["video_frame"]) if "views" in d else int(d["video_frame"])


def scale_hook(export_seq, aligned_depth, ref_name, mesh_source, fp_weights, out_json, a_samples=7, a_levels=3):
    """Metric scale of the generated mesh against CARI4D's human-aligned depth at the mesh's reference frame:
    the toolkit's FoundationPoseTracker.estimate_scale_grid_search (FP registration per candidate scale, mask IoU +
    depth consistency score), exactly as modules/v2d_pipelines/run_hand_masks.py does for SAM3D meshes
    (lo 0.5, hi 2.0, 5 registration iterations; 7 samples x 3 levels = the tracker's defaults since 2026-10-09 16:00,
    9 x 4 before: 156 s per episode), around an initial scale s0:
      all sources: s0 = (physical size of the masked silhouette at the median masked depth) / (mesh bounding diameter)
      (until 2026-10-09 14:50 sam3do used s0 = 1, i.e. its MoGe-2 point-map scale)
    The export's output_aligned.glb is rescaled IN PLACE about its centre (oriented-bounds origin) and the stage-02
    marker's output identities are refreshed so the toolkit's stage bookkeeping stays consistent."""
    import trimesh
    from prep.mhr_export_utils import camera_calibration, load_edex, read_depth_m, read_mask, read_rgb
    from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask
    from v2d.common.datatypes import Image as V2dImage
    from v2d.mesh.lib.mesh import Mesh
    from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker
    mesh_path = export_seq / "object_mesh" / "output_aligned.glb"
    K, _ = camera_calibration(load_edex(export_seq), 0)
    rgb = read_rgb(export_seq, 0, ref_name)
    depth = read_depth_m(export_seq, 0, ref_name, depth_root=aligned_depth).astype(np.float32)
    depth[(depth < 0.001) | (depth > 8.0)] = 0
    om = read_mask(export_seq, "object", 0, ref_name) > 0
    hm = read_mask(export_seq, "human", 0, ref_name) > 0
    H, W = depth.shape
    tm = trimesh.load(mesh_path, force="mesh", process=False)
    v = np.asarray(tm.vertices)
    diam = float(np.linalg.norm(v.max(0) - v.min(0)))
    ys, xs = np.nonzero(om)
    dz = depth[om]; dz = dz[dz > 0]
    zmed = float(np.median(dz)) if dz.size else float("nan")
    sil = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()) * zmed / K[0, 0]) if xs.size else float("nan")
    # s0 from the silhouette at CARI4D's human-aligned depth for EVERY source (2026-10-09: the sam3do mesh's own MoGe
    # scale was 0.44x GT on val ep 11, outside the 0.5-2x grid around s0 = 1; and MoGe's raw metric scale is not the
    # human-aligned scale of the CARI4D world anyway)
    s0 = sil / max(diam, 1e-6) if np.isfinite(sil) and sil > 0 else 1.0
    mesh = Mesh.load(str(mesh_path))
    mesh = Mesh(vertices=np.asarray(mesh.vertices) * s0, faces=mesh.faces, vertex_colors=mesh.vertex_colors,
                uv=mesh.uv, texture=mesh.texture)
    tr = FoundationPoseTracker(mesh, fp_weights, backend="nvlabs_pytorch")   # CARI4D's own stage 05 uses these weights
    intr = CameraIntrinsics(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), width=W, height=H)
    t0 = time.time()
    rel = tr.estimate_scale_grid_search(V2dImage(rgb), DepthImage(depth), Mask(om.astype(np.uint8)), intr, lo=0.5, hi=2.0,
                                        n_samples=a_samples, n_levels=a_levels, iou_weight=1.0, depth_weight=1.0, registration_iterations=5)
    s = float(s0 * rel)
    tm.vertices = v * s
    tmp = mesh_path.with_name(".output_aligned.scaled.tmp.glb")
    tm.export(tmp)
    os.replace(tmp, mesh_path)
    rep = {"ref_frame_name": ref_name, "mesh_source": mesh_source, "s0": s0, "grid": [a_samples, a_levels], "grid_rel": float(rel), "scale": s,
           "mesh_diam_before": diam, "mesh_extents_after_m": [float(x) for x in tm.extents], "silhouette_m": sil,
           "median_obj_depth_m": zmed, "obj_px": int(om.sum()), "obj_px_occluded_by_human": int((om & hm).sum()),
           "seconds": round(time.time() - t0, 1)}
    atomic_json(rep, out_json)
    print("[t1_cari4d] scale hook", json.dumps(rep), flush=True)
    del tr
    import torch
    torch.cuda.empty_cache()
    return rep


def cmd_run(a):
    """Stages 01-07 with the commands of lib/run_inference.py (copied verbatim, same defaults), plus the mesh-scale
    hook between 04 and 05 for generated meshes."""
    import v2d.cari4d.lib.run_inference as RI
    E = f"episode_{a.episode:06d}"
    od = Path(a.out) / "inputs" / E
    if not (od / "window.json").is_file():
        cmd_prep(a)
    info = json.load(open(od / "window.json"))
    video_path = (od / f"{E}.0.color.mp4").resolve(); mask_h5_path = (od / f"{E}_masks_k0.h5").resolve()
    src_mesh = mesh_of(a.mesh_source, a.split, a.episode)
    object_mesh_path = (od / ("object" + src_mesh.suffix)).resolve()
    if not object_mesh_path.is_file():
        import shutil
        shutil.copyfile(src_mesh, object_mesh_path)
    elif sha256(object_mesh_path) != sha256(src_mesh):
        raise SystemExit(f"{object_mesh_path} differs from the current {a.mesh_source} mesh {src_mesh}; use a new --out")
    info["mesh"] = str(src_mesh); info["mesh_sha256"] = sha256(src_mesh)
    weights_path = WEIGHTS.resolve(); output_dir = Path(a.out).resolve()
    sequence = E
    output_root = output_dir / sequence
    marker_root = output_root / ".stages"; marker_root.mkdir(parents=True, exist_ok=True)
    weights = RI._required_weights(weights_path)
    SAM3D_SOURCE_ROOT = M / "v2d_sam3d_body" / "lib"
    FPD = M / "v2d_foundation_pose" / "lib" / "FoundationPose"
    SOURCE_ROOT = RI.SOURCE_ROOT
    env = os.environ.copy()
    env.update({"PYTHONPATH": os.pathsep.join((str(SOURCE_ROOT), str(SAM3D_SOURCE_ROOT), str(FPD), env.get("PYTHONPATH", ""))), "HF_HOME": str(weights_path / "hf_home"), "TORCH_HOME": str(weights_path / "sam3d_body/torch_home"), "SAM3D_BODY_ROOT": str(SAM3D_SOURCE_ROOT), "MHR_ASSETS_ROOT": str(weights_path / "sam3d_body"), "FOUNDATIONPOSE_WEIGHTS_DIR": str(weights_path / "foundationpose/nvlabs_pytorch"), "OPENCV_IO_ENABLE_OPENEXR": "1", "PYOPENGL_PLATFORM": "egl", "HDF5_USE_FILE_LOCKING": "FALSE", "PYTHONUNBUFFERED": "1"})
    raw_depth = output_root / "depth/moge2/raw.npy"
    depth_intrinsics = output_root / "depth/moge2/intrinsics.pkl"
    depth_report = output_root / "depth/moge2/depth_report.json"
    export_root = output_root / "export"
    export_seq = export_root / sequence
    export_marker = export_seq / "wild_export.json"
    aligned_object_mesh = export_seq / "object_mesh/output_aligned.glb"
    mhr_init = output_root / "mhr_init" / f"{sequence}.pkl"
    sam3d_cache = output_root / "mhr_init" / f"{sequence}.sam3d.h5"
    aligned_depth = output_root / "depth/moge2/aligned.h5"
    foundationpose_init = output_root / "foundationpose" / f"{sequence}.pkl"
    coconet_bundle = output_root / "inference/coconet.pth"
    input_cache = output_root / "inference/materialized_inputs.h5"
    refined_bundle = output_root / "inference/refined.pth"
    refinement_checkpoint = output_root / "inference/refinement_checkpoint.pth"
    for path in (raw_depth, depth_intrinsics, depth_report, mhr_init, sam3d_cache, aligned_depth, foundationpose_init, coconet_bundle, input_cache, refined_bundle, refinement_checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    overwrite = False
    expected_frames = int(info["frames"])
    device = "cuda"
    timings = {}

    def stage(name, command, inputs, outputs):
        t0 = time.time()
        mk = marker_root / f"{name}.json"
        if mk.is_file() and all(p.is_file() and p.stat().st_size > 0 for p in outputs):
            # relaxed resume (2026-10-09): the toolkit's marker signature hashes the exact command, which changed for
            # benign reasons (stage-04 worker count, stage 05 through t1_fp_export.py).  A stage whose outputs are
            # byte-identical (path, size, mtime) to what its marker recorded is reused; per-variant output dirs and the
            # mesh sha256 check in cmd_run keep different inputs from ever sharing a directory.
            ex = json.loads(mk.read_text())
            if ex.get("outputs") == RI._output_identities(outputs):
                if ex.get("signature", {}).get("sha256") != RI._signature(command, inputs)["sha256"]:
                    print(f"CARI4D_STAGE_REUSED_RELAXED {name} (command/inputs signature differs, outputs unchanged)", flush=True)
                timings[name] = 0.0
                print(f"[t1_cari4d] {name} reused", flush=True)
                return {"name": name, "reused": True}
        r = RI._stage_run(name, command, inputs, outputs, marker_root, env, overwrite)
        timings[name] = round(time.time() - t0, 1)
        print(f"[t1_cari4d] {name} {timings[name]} s", flush=True)
        return r
    t_all = time.time()
    depth_command = [python, str(SOURCE_ROOT / "prep/mhr_wild_depth.py"), "--video", str(video_path), "--output-depth", str(raw_depth), "--output-intrinsics", str(depth_intrinsics), "--output-report", str(depth_report), "--device", device, "--batch-size", str(8), "--local-files-only", "--expected-frames", str(expected_frames)]
    stage("01_depth", depth_command, [video_path, weights["manifest"]], [raw_depth, depth_intrinsics, depth_report])
    if a.known_k:
        # ruling 2026-10-09: the published FORM-HOI intrinsics are a property of the physical camera (identical in all
        # March-June sessions; /mnt/secondary/v2d/t1/camera_intrinsics_formhoi.json).  Export K <- that K; MoGe-2's
        # per-frame depth is kept.  Stage-01 marker outputs refreshed (same bookkeeping as the mesh-scale hook).
        import joblib
        kj = json.load(open(R / "camera_intrinsics_formhoi.json"))["cameras"][I.item(a.split, a.episode)["camera"]]["K"]
        val = joblib.load(depth_intrinsics)
        if "moge_fx" not in val:
            val.update({"moge_fx": val["fx"], "moge_fy": val["fy"], "moge_cx": val["cx"], "moge_cy": val["cy"],
                        "fx": kj[0][0], "fy": kj[1][1], "cx": kj[0][2], "cy": kj[1][2], "camera_policy_t1": "formhoi_physical_camera_K"})
            tmp = depth_intrinsics.with_name(".intrinsics.knownK.tmp.pkl")
            joblib.dump(val, tmp); os.replace(tmp, depth_intrinsics)
            m01 = marker_root / "01_depth.json"
            mk = json.loads(m01.read_text()); mk["outputs"] = RI._output_identities([raw_depth, depth_intrinsics, depth_report])
            RI._atomic_json(m01, mk)
        print(f"[t1_cari4d] known K: fx {val['fx']:.2f} (MoGe-2 {val['moge_fx']:.2f})", flush=True)
    export_command = [python, str(SOURCE_ROOT / "prep/prepare_mhr_wild_export.py"), "--video", str(video_path), "--mask-h5", str(mask_h5_path), "--object-mesh", str(object_mesh_path), "--intrinsics-file", str(depth_intrinsics), "--output-root", str(export_root)]
    stage("02_export", export_command, [video_path, mask_h5_path, object_mesh_path, depth_intrinsics], [export_marker, aligned_object_mesh])
    sam3d_command = [python, str(SOURCE_ROOT / "prep/run_sam3d_mhr_export.py"), str(export_seq), "--out-file", str(mhr_init), "--camera-id", "0", "--depth-root", str(export_seq), "--sam3d-ckpt", str(weights["sam3d_checkpoint"]), "--mhr-path", str(weights["mhr_model"]), "--chunk-size", str(16), "--sam3d-cache", str(sam3d_cache), "--alignment-workers", str(1), "--refit-batch-size", str(512), "--human-prompt-mode", "sam2_bbox_mask", "--no-align-to-gt-depth"]
    stage("03_human_initialization", sam3d_command, [export_marker, weights["sam3d_checkpoint"], weights["mhr_model"]], [mhr_init, sam3d_cache])
    def alignment_cmd(workers):
        return [python, str(SOURCE_ROOT / "prep/align_depth_to_mhr_wild.py"), str(export_seq), "--raw-depth", str(raw_depth), "--depth-report", str(depth_report), "--mhr-init", str(mhr_init), "--output", str(aligned_depth), "--render-batch-size", str(4), "--encoding-workers", str(workers), "--write-batch-size", str(64)]
    alignment_command = alignment_cmd(a.encoding_workers)
    m04 = marker_root / "04_depth_alignment.json"
    if m04.is_file():
        # resume: the stage signature hashes the command, so reuse the worker count the existing marker was made with
        # (the default changed 4 -> 10 on 2026-10-09; the worker count does not change the output)
        sig = json.loads(m04.read_text()).get("signature")
        a04 = [export_marker, raw_depth, depth_report, mhr_init]
        for w in (a.encoding_workers, 4, 10, 16, 8):
            if raw_depth.is_file() and RI._signature(alignment_cmd(w), a04) == sig:
                alignment_command = alignment_cmd(w); break
    stage("04_depth_alignment", alignment_command, [export_marker, raw_depth, depth_report, mhr_init], [aligned_depth])
    # ---- mesh-scale hook (generated meshes only; once per run)
    hook_json = output_root / "t1_mesh_scale.json"
    if a.mesh_source != "given" and not hook_json.is_file():
        ref = mesh_ref_frame(a.mesh_source, a.split, a.episode) - int(info["window"][0])
        ref = int(np.clip(ref, 0, expected_frames - 1))
        sys.path.insert(0, str(SOURCE_ROOT)); sys.path.insert(0, str(FPD))
        os.environ.update({k: env[k] for k in ("HF_HOME", "TORCH_HOME", "MHR_ASSETS_ROOT", "FOUNDATIONPOSE_WEIGHTS_DIR")})
        scale_hook(export_seq, aligned_depth, f"{ref:06d}", a.mesh_source, str(WEIGHTS / "foundationpose"), hook_json)
        m02 = marker_root / "02_export.json"
        mk = json.loads(m02.read_text())
        mk["outputs"] = RI._output_identities([export_marker, aligned_object_mesh])
        RI._atomic_json(m02, mk)
    # stage 05 through t1_fp_export.py: the toolkit script + an inf -> NaN fix of one diagnostic (see its docstring)
    foundationpose_command = [python, str(HERE / "t1_fp_export.py"), str(export_seq), "--depth-root", str(aligned_depth), "--out-file", str(foundationpose_init), "--camera-id", "0", "--iteration", str(5), "--max-attempts", str(5), "--register-first-then-track", "--no-first-usable-frame-gt-rotation-oracle"]
    stage("05_object_pose", foundationpose_command, [export_marker, aligned_depth, weights["foundationpose_score"], weights["foundationpose_refine"]], [foundationpose_init])
    coconet_command = [python, str(SOURCE_ROOT / "tools/run_mhr_wild_inference.py"), str(export_seq), "--depth-h5", str(aligned_depth), "--mhr-init", str(mhr_init), "--foundationpose-file", str(foundationpose_init), "--config", str(weights["config"]), "--checkpoint", str(weights["checkpoint"]), "--output", str(coconet_bundle), "--stride", str(96), "--render-batch-size", str(32), "--crop-workers", str(a.crop_workers), "--crop-buffer-count", str(2), "--input-cache", str(input_cache), "--device", device, "--offline-supervision-contract"]
    stage("06_coconet", coconet_command, [export_marker, aligned_depth, mhr_init, foundationpose_init, weights["config"], weights["checkpoint"]], [coconet_bundle, input_cache])
    postopt_command = [python, "-m", "learning.training.mhr_opt_refineout", "--bundle", str(coconet_bundle), "--object-mesh", str(aligned_object_mesh), "--out", str(refined_bundle), "--mode", "smplh_parity", "--num-steps", str(300), "--batch-size", str(a.postopt_batch_size), "--device", device, "--postopt-checkpoint", str(refinement_checkpoint), "--mhr-assets-root", str(weights_path / "sam3d_body"), "--w-temporal", str(100.0), "--w-human-pose-prior", str(200.0), "--contact-activation-distance-m", str(0.05), "--report-every", str(10), "--diagnostics-every", str(500)]
    if a.optimize_object_rotation:
        postopt_command.append("--optimize-object-rotation")
    if a.stop_after == "06":
        rep = {"episode": a.episode, "split": a.split, "window": info["window"], "frames": info["frames"], "stop_after": "06",
               "mesh_source": a.mesh_source, "mesh": info["mesh"], "mesh_sha256": info["mesh_sha256"], "stage_seconds": timings,
               "seconds": round(time.time() - t_all, 1), "known_k": a.known_k, "coconet_sha256": sha256(coconet_bundle)}
        atomic_json(rep, output_root / "t1_run06.json")
        print(json.dumps(rep), flush=True)
        if not a.keep_raw_depth and raw_depth.is_file():
            raw_depth.unlink()
        return
    stage("07_refinement", postopt_command, [coconet_bundle, aligned_object_mesh, weights["mhr_model"]], [refined_bundle, refinement_checkpoint])
    rep = {"episode": a.episode, "split": a.split, "window": info["window"], "frames": info["frames"],
           "mesh_source": a.mesh_source, "mesh": info["mesh"], "mesh_sha256": info["mesh_sha256"], "stage_seconds": timings, "seconds": round(time.time() - t_all, 1),
           "postopt_batch_size": a.postopt_batch_size, "optimize_object_rotation": a.optimize_object_rotation,
           "known_k": a.known_k, "refined_sha256": sha256(refined_bundle)}
    atomic_json(rep, output_root / "t1_run.json")
    print(json.dumps(rep), flush=True)
    if not a.keep_raw_depth and raw_depth.is_file():
        # 3.5 MB/frame of float16 MoGe-2 depth, only read by stage 04; the run is complete (refined.pth written)
        raw_depth.unlink()
        print(f"[t1_cari4d] removed {raw_depth} (stage 01 would re-run on a resume)", flush=True)


# ------------------------------------------------------------------------------------------------ decode
def head_and_layer(device="cpu"):
    sys.path.insert(0, str(M / "v2d_cari4d" / "lib" / "cari4d"))
    from lib_mhr import MHRLayer
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=str(WEIGHTS / "sam3d_body"), device=device)
    return layer


def to_kit(layer, P, device="cpu", batch=128, check=0):
    """CARI4D MHR param dict (numpy [T,...]) -> kit pose [T,136], scales [T,68], shape [T,45] (+ optional check)."""
    import torch
    import roma
    from lib_mhr.rotations import rot6d_to_rotmat
    from sam_3d_body.models.modules.mhr_utils import compact_cont_to_model_params_body
    head = layer.backend._ensure_head(device)
    T = len(P["mhr_trans"])
    pose = np.zeros((T, 136)); scales = np.zeros((T, 68)); shape = np.zeros((T, 45)); err = []
    with torch.inference_mode():
        for s in range(0, T, batch):
            sl = slice(s, min(T, s + batch))
            g = lambda k, d: torch.as_tensor(np.asarray(P[k][sl]), dtype=torch.float32, device=device)  # noqa: E731
            n = sl.stop - sl.start
            grot = roma.rotmat_to_euler("ZYX", rot6d_to_rotmat(g("mhr_global_rot6d", 6)))
            body = compact_cont_to_model_params_body(g("mhr_body_pose_cont", 260))
            hand = g("mhr_hand", 108); sc = g("mhr_scale", 28); sh = g("mhr_shape", 45)
            zero3 = torch.zeros(n, 3, device=device)
            out = head.mhr_forward(global_trans=zero3, global_rot=grot, body_pose_params=body, hand_pose_params=hand,
                                   scale_params=sc, shape_params=sh, expr_params=torch.zeros(n, 72, device=device),
                                   return_keypoints=False, return_joint_coords=False, return_model_params=True,
                                   return_joint_rotations=False)
            verts, mp = out[0], out[-1]
            mp = mp.double().cpu().numpy()
            assert mp.shape[1] == 204, mp.shape
            assert np.abs(mp[:, :3]).max() < 1e-6
            tr = np.asarray(P["mhr_trans"][sl], np.float64)
            mp[:, :3] = 10.0 * tr * FLIP
            pose[sl] = mp[:, :136]; scales[sl] = mp[:, 136:]; shape[sl] = sh.double().cpu().numpy()
    return pose, scales, shape


def kit_vertices(pose, scales, shape):
    import torch
    import t1lib as L
    m = torch.jit.load(str(L.MHR_TS), map_location="cpu").eval()
    with torch.no_grad():
        v, _ = m(torch.as_tensor(shape, dtype=torch.float32), torch.as_tensor(np.concatenate([pose, scales], 1), dtype=torch.float32),
                 torch.zeros(len(pose), 72))
    return v.double().numpy() / 100.0 * FLIP


def cmd_decode(a):
    import torch
    import trimesh
    import t1lib as L
    E = f"episode_{a.episode:06d}"
    root = Path(a.out) / E
    info = json.load(open(Path(a.out) / "inputs" / E / "window.json"))
    s, e = info["window"]; Tv = int(info["T_video"])
    bundle = "coconet.pth" if a.bundle == "coconet" else "refined.pth"
    b = torch.load(root / "inference" / bundle, map_location="cpu", weights_only=False)
    names = [int(x) for x in b["frames"]]
    assert names == list(range(e - s)), (names[:3], e - s)
    mesh = root / "export" / E / "object_mesh" / "output_aligned.glb"
    layer = head_and_layer("cpu")
    rep = {"episode": a.episode, "window": [s, e], "T_video": Tv, "variants": {}}
    meta = json.load(open(root / "export" / E / "wild_export.json"))
    m2t = np.asarray(meta.get("object_mesh_to_training_transform", np.eye(4)), np.float64)
    todo = (("init", "in"), ("coconet", "pr_initial")) if a.bundle == "coconet" else (("init", "in"), ("coconet", "pr_initial"), ("refined", "pr"))
    for var, key in todo:
        P = {k: np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v) for k, v in b[key].items()
             if k.startswith("mhr_") or k == "pose_abs"}
        if "mhr_trans" not in P:
            print(f"{var}: no MHR params in bundle[{key!r}] ({sorted(b[key])[:12]})"); continue
        pose, scales, shape = to_kit(layer, P)
        # identity must be constant (CARI4D keeps shape/scale fixed); use the median and record the spread
        sc_std, sh_std = float(scales.std(0).max()), float(shape.std(0).max())
        scales_c, shape_c = np.median(scales, 0), np.median(shape, 0)
        # exactness check on a few frames: MHRLayer (CARI4D's decoder, expr 0) vs kit TorchScript forward of our params
        idx = np.linspace(0, len(pose) - 1, 4).round().astype(int)
        Pc = {k: torch.as_tensor(P[k][idx], dtype=torch.float32) for k in P if k != "pose_abs"}
        Pc["mhr_face"] = torch.zeros(len(idx), 72)
        with torch.inference_mode():
            v_c = layer.mhr_forward(Pc).vertices.double().numpy()
        v_k = kit_vertices(pose[idx], scales[idx], shape[idx])
        chk = float(np.linalg.norm(v_c - v_k, axis=-1).mean() * 1000)
        pa = np.asarray(P["pose_abs"], np.float64) @ m2t
        A = pa[:, :3, :3]; sdet = np.cbrt(np.linalg.det(A))
        u, _, vt = np.linalg.svd(A / sdet[:, None, None]); Rm = u @ vt
        # pad to the full video
        n = e - s
        def pad(x):
            out = np.empty((Tv,) + x.shape[1:], x.dtype)
            out[s:e] = x; out[:s] = x[0]; out[e:] = x[-1]
            return out
        epd = {"pose": pad(pose), "scales": scales_c, "shape": shape_c, "object_rotation": pad(Rm),
               "object_translation": pad(pa[:, :3, 3]), "object_scale": np.array(float(np.median(sdet)))}
        vd = root / "t1" / var
        vd.mkdir(parents=True, exist_ok=True)
        L.save_episode(vd / f"{E}.npz", epd)
        import shutil
        shutil.copyfile(mesh, vd / f"{E}_object.glb")
        rep["variants"][var] = {"layer_vs_kit_vertex_mm": chk, "scale_frame_std_max": sc_std, "shape_frame_std_max": sh_std,
                                "pose_abs_scale_median": float(np.median(sdet)), "pose_abs_scale_range": [float(sdet.min()), float(sdet.max())],
                                "n": n}
        print(var, json.dumps(rep["variants"][var]), flush=True)
    rep["mesh"] = str(mesh); rep["mesh_extents_m"] = [float(x) for x in trimesh.load(mesh, force="mesh").extents]
    tag = "_coconet" if a.bundle == "coconet" else ""
    np.savez(root / "t1" / f"decoded{tag}.npz", ok=1)
    atomic_json(rep, root / "t1" / f"decode{tag}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "run", "decode"])
    ap.add_argument("--split", required=True, choices=["val", "track1"])
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--mesh-source", default="trellis", choices=["trellis", "sam3do", "trellis2", "given"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--postopt-batch-size", type=int, default=0)
    ap.add_argument("--optimize-object-rotation", action="store_true")
    ap.add_argument("--known-k", action="store_true", help="export K = FORM-HOI physical camera K (ruling 2026-10-09)")
    ap.add_argument("--encoding-workers", type=int, default=10,
                    help="stage 04 (toolkit default 16).  4 made stage 04 CPU-bound: 373 s for 280 frames while holding the GPU lock")
    ap.add_argument("--keep-raw-depth", action="store_true", help="keep depth/moge2/raw.npy after a complete run")
    ap.add_argument("--stop-after", default="07", choices=["06", "07"],
                    help="run: 06 stops after CoCoNet (stage 07 = 300-step refinement, ~1.8 s/frame on the 3090)")
    ap.add_argument("--bundle", default="refined", choices=["refined", "coconet"],
                    help="decode: refined.pth (init/coconet/refined) or coconet.pth (init/coconet only, before stage 07)")
    ap.add_argument("--crop-workers", type=int, default=4, help="stage 06 (toolkit default 8)")
    a = ap.parse_args()
    {"prep": cmd_prep, "run": cmd_run, "decode": cmd_decode}[a.cmd](a)


if __name__ == "__main__":
    main()
