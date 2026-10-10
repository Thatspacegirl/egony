#!/usr/bin/env python3
"""CARI4D output (inference/refined.pth) -> Track 1 episode NPZ + object mesh.

Written against reconstruction/modules/v2d_cari4d (nvidia-isaac/video_to_data @ a709404e):
  * lib/run_inference.py           refined bundle = <out>/<seq>/inference/refined.pth (stage 07_refinement),
                                   aligned mesh   = <out>/<seq>/export/<seq>/object_mesh/output_aligned.glb
  * tools/render_mhr_wild_inference.py  the reference decode: MHRLayer.from_mhr_assets(MHR_ASSETS_ROOT)
                                   .mhr_forward(after["pr"][MHR_PARAM_DIMS keys]) -> vertices in CAMERA space
                                   (camera space == export world, asserted there), object = after["pr"]["pose_abs"]
                                   [T,4,4] applied to output_aligned.glb; bundle["frames"] = every video frame.
  * lib_mhr/mhr_layer.py           vertices = flip(1,-1,-1) * SAM3D-head MHR(...)/100 + mhr_trans  (metres) -- the
                                   same axis convention as the Track 1 GT (diag(1,-1,-1) @ MHR / 100); the frame is
                                   the camera instead of the rig world, which the scorer's frame-0 Sim(3) absorbs.

Stages (decode needs the CARI4D container + gated SAM 3D Body assets; fit/write run anywhere):
  decode : refined.pth -> MHR forward (CARI4D's own MHRLayer) -> lod1 vertices [T,18439,3] float32 + pose_abs
  fit    : kit tools/track1/mesh_to_mhr_params.convert (Meta MHR v1.0.1 TorchScript) -> pose[T,136], scales[68], shape[45]
           (one identity per episode by construction; check report vertex_error_mm)
  write  : episode_%06d.npz {pose, scales, shape, object_rotation, object_translation, object_scale} +
           episode_%06d_object.glb, optionally post-processed (smoothing.py, pen_refine.py)
  selftest: synthetic 'decoded' file from FORM-HOI GT params (TorchScript forward, random camera rigid transform)
           -> fit (CPU) -> write -> checks vertex error and object-pose round trip.

Example (inside the container; see README for the docker run line):
  python cari4d_to_t1.py decode --refined /out/episode_000003/inference/refined.pth \
      --export-seq /out/episode_000003/export/episode_000003 --mhr-assets-root /weights/sam3d_body \
      --out /work/episode_000003_decoded.npz
  python cari4d_to_t1.py fit --decoded /work/episode_000003_decoded.npz --mhr /weights/mhr_model.pt --device cuda \
      --precision float32 --model-batch 256 --out /work/episode_000003_params.npz
  python cari4d_to_t1.py write --decoded ... --params ... --episode 3 --out-dir /mnt/secondary/v2d/t1/npz \
      [--smooth-json smooth_cfg.json] [--pen] [--scored-frames-json frames.json | --first-scored 90]
  (simplest: write WITHOUT --smooth-json/--pen for every episode, then run postprocess.py on the directory with
   --sample <kit>/data/track_1_sample_submission.parquet, which takes the scored frames from the sample rows)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KIT = Path(os.environ.get("V2D_KIT", "/mnt/secondary/v2d/kit/v2d_submission_kit"))
META_MHR_SHA256 = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------- decode
def cmd_decode(a):
    import torch
    # CARI4D sources: lib/cari4d on PYTHONPATH (lib_mhr), SAM 3D Body lib for sam_3d_body.models.heads.mhr_head
    for p in (a.cari4d_src, a.sam3d_src):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    from lib_mhr import MHRLayer, MHR_PARAM_DIMS
    after = torch.load(a.refined, map_location="cpu", weights_only=False)   # our own pipeline output
    names = [str(x) for x in after["frames"]]
    frame_idx = np.array([int(n) for n in names])
    if not np.array_equal(frame_idx, np.arange(len(frame_idx))):
        raise ValueError(f"refined bundle frames are not 0..T-1 video frames: {names[:5]}...")
    pr = after["pr"]
    missing = [k for k in MHR_PARAM_DIMS if k not in pr]
    if missing:
        raise KeyError(f"refined bundle 'pr' lacks {missing}")
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=a.mhr_assets_root, device=a.device)
    T = len(names)
    verts = np.empty((T, 18439, 3), np.float32)
    with torch.inference_mode():
        for s in range(0, T, a.batch):
            sl = slice(s, min(T, s + a.batch))
            params = {k: torch.as_tensor(np.asarray(pr[k][sl]), device=a.device, dtype=torch.float) for k in MHR_PARAM_DIMS}
            if not a.keep_face:
                # Track 1 GT and the kit converter use expr = 0; face expression never moves the scored body/hand
                # roles, but a non-zero expression would add face-vertex residual to the mesh_to_mhr_params fit.
                params["mhr_face"] = torch.zeros_like(params["mhr_face"])
            v = layer.mhr_forward(params).vertices
            verts[sl] = v.detach().cpu().numpy().astype(np.float32)
    pose_abs = np.asarray(pr["pose_abs"], np.float64)
    export_meta = json.load(open(Path(a.export_seq) / "wild_export.json"))
    # run_mhr_wild_inference.py: poses live in the "training" frame and apply to the mesh AFTER
    # object_mesh.apply_transform(mesh_to_training) (lib_mhr/object_pose_frame.py). For wild exports the mesh is
    # output_aligned.glb, so mesh_to_training == I (asserted below by being explicit); compose anyway so the stored
    # pose always maps RAW output_aligned.glb vertices into camera space, like render_mhr_wild_inference.py does.
    m2t = np.asarray(export_meta.get("object_mesh_to_training_transform", np.eye(4)), np.float64)
    if not np.allclose(m2t, np.eye(4), atol=1e-6):
        print("note: object_mesh_to_training_transform is not identity; composing it into pose_abs")
    pose_abs = pose_abs @ m2t
    ident = {}
    ident["mhr_face_absmax"] = float(np.abs(np.asarray(pr["mhr_face"], np.float64)).max())
    ident["face_zeroed"] = not a.keep_face
    for key in ("mhr_shape", "mhr_scale"):
        x = np.asarray(pr[key], np.float64)
        ident[f"{key}_frame_std_max"] = float(x.std(0).max())
    bundled = Path(a.mhr_assets_root) / "checkpoints" / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"
    prov = {"refined": str(a.refined), "frames": T, "postopt": {k: after.get("postopt", {}).get(k) for k in ("mode", "optimized_parameters", "fixed_parameters")},
            "bundled_mhr_sha256": sha256(bundled) if bundled.exists() else None, "meta_mhr_sha256": META_MHR_SHA256,
            "object_mesh": str(Path(a.export_seq) / "object_mesh" / "output_aligned.glb"),
            "object_mesh_to_training_transform": m2t.tolist(),
            "object_pose_frame": export_meta.get("object_pose_frame"),
            "object_pose_storage_frame": export_meta.get("object_pose_storage_frame"), **ident}
    np.savez(a.out, vertices=verts, pose_abs=pose_abs, frame_index=frame_idx, provenance=json.dumps(prov))
    print(json.dumps(prov, indent=1))


# ------------------------------------------------------------------------------------------- fit
def cmd_fit(a):
    sys.path.insert(0, str(KIT / "tools" / "track1"))
    from mesh_to_mhr_params import convert
    d = np.load(a.decoded)
    verts = d["vertices"]
    if a.max_frames:
        verts = verts[: a.max_frames]
    out = convert(verts, a.mhr, a.device, a.keyframes, precision=a.precision, model_batch=a.model_batch,
                  log=lambda *x, **k: print(*x, **k, file=sys.stderr, flush=True))
    rep = out.pop("report")
    np.savez(a.out, **out, report=json.dumps(rep))
    print(json.dumps(rep["vertex_error_mm"]), "seconds", rep["seconds"])
    if rep["vertex_error_mm"]["mean"] > 1.0:
        print("WARNING: mean vertex error > 1 mm: input is off the MHR manifold (per-frame identity? wrong units/frame?)")


# ------------------------------------------------------------------------------------------- write
def episode_from(decoded, params, object_scale=1.0):
    pa = np.asarray(decoded["pose_abs"], np.float64)
    T = len(params["pose"])
    pa = pa[:T]
    A = pa[:, :3, :3]
    det = np.linalg.det(A)
    s_frame = np.cbrt(det)
    if np.abs(s_frame - 1).max() > 1e-3:
        print(f"note: pose_abs carries scale (median {np.median(s_frame):.4f}); factoring it into object_scale")
    s = float(np.median(s_frame)) * object_scale
    u, _, vt = np.linalg.svd(A / s_frame[:, None, None])
    R = u @ vt
    R[np.linalg.det(R) < 0] *= -1
    return {"pose": np.asarray(params["pose"], np.float64), "scales": np.asarray(params["scales"], np.float64),
            "shape": np.asarray(params["shape"], np.float64), "object_rotation": R,
            "object_translation": pa[:, :3, 3].copy(), "object_scale": np.array(s)}


def cmd_write(a):
    sys.path.insert(0, str(HERE))
    import t1lib as L
    dec = np.load(a.decoded)
    par = np.load(a.params)
    ep = episode_from(dec, par)
    prov = json.loads(str(dec["provenance"])) if "provenance" in dec.files else {}
    mesh = Path(a.mesh or prov.get("object_mesh", ""))
    if not mesh.is_file():
        raise FileNotFoundError(f"object mesh not found: {mesh}")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    report = {"episode": a.episode, "T": int(len(ep["pose"])), "mesh": str(mesh)}
    frames = (np.asarray(json.load(open(a.scored_frames_json)), int) if a.scored_frames_json
              else np.arange(len(ep["pose"])))
    first = int(a.first_scored) if a.first_scored is not None else int(frames[0])
    report["first_scored"] = first
    if a.smooth_json:
        import smoothing as SM
        cfg = SM.SmoothCfg(**json.load(open(a.smooth_json)))
        ep = SM.smooth_episode(ep, first, cfg)
        report["smoothing"] = SM.cfg_dict(cfg)
    if a.pen:
        # same defaults as postprocess.py: hand stage, then the object stage only if PEN is still > 0.0005 cm
        import pen_refine as P
        from v2dlb.mesh_budget import budget_mesh   # t1lib puts the kit on sys.path
        hands = P.HandPoints(L.MHR_TS)
        ep, prep = P.refine(ep, budget_mesh(mesh, 4096, 4096), frames, hands, margin=0.002, stages=("hand", "object"),
                            lam=100.0, object_if_pen_above=0.0005)
        report["pen_refine"] = prep
    L.save_episode(out / f"episode_{a.episode:06d}.npz", ep)
    shutil.copyfile(mesh, out / f"episode_{a.episode:06d}_object{mesh.suffix}")
    json.dump(report, open(out / f"episode_{a.episode:06d}_report.json", "w"), indent=1)
    print(json.dumps(report)[:2000])


# ------------------------------------------------------------------------------------------- selftest
def cmd_selftest(a):
    """GT params -> TorchScript lod1 vertices in a random camera frame (what decode produces) -> fit -> write."""
    import torch
    sys.path.insert(0, str(HERE))
    import t1lib as L
    from scipy.spatial.transform import Rotation
    gt = L.load_episode(a.gt_episode)
    sl = slice(a.start, a.start + a.frames)
    m = torch.jit.load(str(L.MHR_TS), map_location="cpu").eval()
    mp = np.concatenate([gt["pose"][sl], np.broadcast_to(gt["scales"], (a.frames, 68))], 1)
    with torch.no_grad():
        v, _ = m(torch.as_tensor(gt["shape"], dtype=torch.float32)[None].expand(a.frames, -1).contiguous(),
                 torch.as_tensor(mp, dtype=torch.float32), torch.zeros(a.frames, 72), True)
    flip = np.array([1.0, -1.0, -1.0])
    v_world = v.double().numpy() * flip / 100.0
    rng = np.random.default_rng(0)
    Rc = Rotation.random(random_state=1).as_matrix(); tc = rng.normal(0, 1.0, 3)          # world -> "camera"
    verts = (v_world @ Rc.T + tc).astype(np.float32)
    P = np.tile(np.eye(4), (a.frames, 1, 1))
    P[:, :3, :3] = Rc @ gt["object_rotation"][sl]; P[:, :3, 3] = gt["object_translation"][sl] @ Rc.T + tc
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    np.savez(work / "decoded.npz", vertices=verts, pose_abs=P, frame_index=np.arange(a.frames),
             provenance=json.dumps({"object_mesh": str(a.mesh), "synthetic": True}))
    t0 = time.time()
    fa = argparse.Namespace(decoded=str(work / "decoded.npz"), mhr=str(L.MHR_TS), device="cpu", keyframes=a.keyframes,
                            precision=a.precision, model_batch=a.model_batch, out=str(work / "params.npz"), max_frames=None)
    cmd_fit(fa)
    wa = argparse.Namespace(decoded=str(work / "decoded.npz"), params=str(work / "params.npz"), mesh=a.mesh, episode=0,
                            out_dir=str(work / "npz"), smooth_json=None, first_scored=None, pen=False, scored_frames_json=None)
    cmd_write(wa)
    ep = L.load_episode(work / "npz" / "episode_000000.npz")
    with torch.no_grad():
        mp2 = np.concatenate([ep["pose"], np.broadcast_to(ep["scales"], (a.frames, 68))], 1)
        v2, _ = m(torch.as_tensor(ep["shape"], dtype=torch.float32)[None].expand(a.frames, -1).contiguous(),
                  torch.as_tensor(mp2, dtype=torch.float32), torch.zeros(a.frames, 72), True)
    v2 = v2.double().numpy() * flip / 100.0
    err_mm = np.linalg.norm(v2 - verts, axis=-1).mean() * 1000
    obj_err = np.abs(ep["object_rotation"] - P[:, :3, :3]).max()
    res = {"frames": a.frames, "fit_seconds": round(time.time() - t0, 1), "refit_vertex_error_mm": float(err_mm),
           "object_rotation_roundtrip_maxabs": float(obj_err), "pass": bool(err_mm < 0.05 and obj_err < 1e-5)}
    print("SELFTEST", json.dumps(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--refined", required=True); d.add_argument("--export-seq", required=True)
    d.add_argument("--mhr-assets-root", default=os.environ.get("MHR_ASSETS_ROOT", "/weights/sam3d_body"))
    d.add_argument("--cari4d-src", default="/workspace/v2d_cari4d/lib/cari4d"); d.add_argument("--sam3d-src", default="/workspace/v2d_sam3d_body/lib")
    d.add_argument("--device", default="cuda"); d.add_argument("--batch", type=int, default=64); d.add_argument("--out", required=True)
    d.add_argument("--keep-face", action="store_true", help="decode with CARI4D's mhr_face instead of expr=0 (default zeroes it)")
    f = sub.add_parser("fit")
    f.add_argument("--decoded", required=True); f.add_argument("--mhr", required=True); f.add_argument("--out", required=True)
    f.add_argument("--device", default=None); f.add_argument("--precision", default="float32", choices=["float32", "float64"])
    f.add_argument("--model-batch", type=int, default=None); f.add_argument("--keyframes", type=int, default=None)
    f.add_argument("--max-frames", type=int, default=None)
    w = sub.add_parser("write")
    w.add_argument("--decoded", required=True); w.add_argument("--params", required=True); w.add_argument("--episode", type=int, required=True)
    w.add_argument("--out-dir", required=True); w.add_argument("--mesh", default=None)
    w.add_argument("--smooth-json", default=None); w.add_argument("--first-scored", type=int, default=None)
    w.add_argument("--pen", action="store_true"); w.add_argument("--scored-frames-json", default=None)
    t = sub.add_parser("selftest")
    t.add_argument("--gt-episode", default="/mnt/secondary/v2d/t1/formhoi_val/kitval/gt/episode_000000.npz")
    t.add_argument("--mesh", default="/mnt/secondary/v2d/t1/formhoi_val/kitval/gt/episode_000000_object.glb")
    t.add_argument("--start", type=int, default=90); t.add_argument("--frames", type=int, default=8)
    t.add_argument("--keyframes", type=int, default=2); t.add_argument("--precision", default="float32")
    t.add_argument("--model-batch", type=int, default=64); t.add_argument("--work", default="/mnt/secondary/v2d/t1/work/cari4d_selftest")
    a = ap.parse_args()
    {"decode": cmd_decode, "fit": cmd_fit, "write": cmd_write, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
