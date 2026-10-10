"""SAM 3D Objects (facebook/sam-3d-objects, SAM License) single-image object meshes from our masked cam_a frames,
grounded in metric scale by our stereo depth -- GPU (env t3-sam3do, source $E/v2d_env.sh).

Uses the toolkit wrapper v2d.sam3d.lib (InferencePipelinePointMap): the conditioning point map is OUR FoundationStereo
depth on the undistorted cam_a grid at 0.5 scale (depth_cam_a_s0.5, lag-aware), restricted to the (eroded) SAM3 object
mask, instead of MoGe.  The image is the full-resolution undistorted cam_a frame (2028x1520) with the SAM3 mask
upsampled into the alpha channel.  The model is loaded once for a whole job list.

Per job (split, episode, obj, frame[, seed]) it writes OUT/<split>/episode_X/<obj>/f<frame>_s<seed>.{glb,json,ply}:
  .glb   SAM3D mesh in its own canonical frame (vertex colours)
  .json  transform cam_a_T_obj (OpenCV, scale->rotate->translate, toolkit _to_opencv_transform), timings, peak VRAM,
         plus a depth-agreement check: median |z_mesh - z_depth| of the masked depth pixels against the mesh rendered
         with that pose (front-most surface) -- verifies the pose/scale convention on every job
  .ply   the same mesh in METRIC cam_a coordinates (frame `frame`), i.e. transform applied

  python -I t3_sam3do.py --jobs JOBS.json --out /mnt/secondary/v2d/t3/sam3do [--steps1 25 --steps2 25]
JOBS.json = [{"split": "public", "episode": 21, "obj": "blue_cup", "frame": 157, "seed": 0}, ...]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

T3 = "/mnt/secondary/v2d/t3"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_inputs(job, scale=0.5, erode=3):
    from t3_depth_warp import decode_inv_depth
    sp, e, obj, f = job["split"], f"episode_{int(job['episode']):06d}", job["obj"], int(job["frame"])
    meta = json.load(open(f"{T3}/frames/{sp}/{e}/meta.json"))
    rgb = cv2.cvtColor(cv2.imread(f"{T3}/frames/{sp}/{e}/cam_a/{f:06d}.jpg"), cv2.COLOR_BGR2RGB)
    H0, W0 = rgb.shape[:2]
    m_s = cv2.imread(f"{T3}/masks/{sp}/{e}/cam_a_s{scale:g}/{obj}/{f:06d}.png", 0)
    if m_s is None or not (m_s > 0).any():
        raise ValueError(f"empty mask {sp}/{e}/{obj}/{f}")
    m_s = m_s > 0
    m_full = cv2.resize(m_s.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
    d = decode_inv_depth(cv2.imread(f"{T3}/depth/{sp}/{e}/depth_cam_a_s{scale:g}/{f:06d}.png", cv2.IMREAD_UNCHANGED))
    K = np.array(meta["images"]["cam_a"]["K"], float)
    Ks = K.copy()
    Ks[:2] *= scale
    m_d = cv2.erode(m_s.astype(np.uint8), np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)) > 0
    if m_d.sum() < 50:
        m_d = m_s
    return rgb, m_full, d.astype(np.float32), Ks, m_d, K


def mesh_from_glb(scene):
    import trimesh
    if isinstance(scene, trimesh.Scene):
        return scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    return scene


def depth_agreement(Vc, F, depth, Ks, mask):
    """Rasterise the metric cam-frame mesh (z-buffer of vertices, splatted) and compare with the masked depth."""
    H, W = depth.shape
    z = Vc[:, 2]
    ok = z > 0.05
    u = np.round(Ks[0, 0] * Vc[ok, 0] / z[ok] + Ks[0, 2]).astype(int)
    v = np.round(Ks[1, 1] * Vc[ok, 1] / z[ok] + Ks[1, 2]).astype(int)
    zz = z[ok]
    zb = np.full((H, W), np.inf, np.float32)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    np.minimum.at(zb, (v[inb], u[inb]), zz[inb])
    zb = -cv2.dilate(-np.where(np.isfinite(zb), zb, 1e3).astype(np.float32), np.ones((5, 5), np.uint8))
    sel = mask & (depth > 0) & (zb < 1e2)
    if sel.sum() < 20:
        return dict(n=int(sel.sum()))
    dz = zb[sel] - depth[sel]
    rend = zb < 1e2
    iou = float((rend & mask).sum() / max((rend | mask).sum(), 1))
    return dict(n=int(sel.sum()), med_dz_mm=round(float(np.median(dz)) * 1000, 1),
                mad_dz_mm=round(float(np.median(np.abs(dz - np.median(dz)))) * 1000, 1), iou=round(iou, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", default=f"{T3}/sam3do")
    ap.add_argument("--weights", default="/mnt/secondary/v2d/weights/sam3d_objects")
    ap.add_argument("--steps1", type=int, default=None)
    ap.add_argument("--steps2", type=int, default=None)
    ap.add_argument("--mesh_postprocess", action="store_true")
    ap.add_argument("--layout_postprocess", action="store_true")
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()

    import torch
    import trimesh
    from v2d.sam3d.lib import image_to_mesh as I2M
    from v2d.common.datatypes import CameraIntrinsics

    jobs = json.load(open(a.jobs))
    t0 = time.time()
    pipe = I2M._get_pipeline(a.weights)
    print(f"pipeline loaded in {time.time() - t0:.1f}s, VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    for job in jobs:
        sp, e, obj, f = job["split"], f"episode_{int(job['episode']):06d}", job["obj"], int(job["frame"])
        seed = int(job.get("seed", 0))
        od = f"{a.out}/{sp}/{e}/{obj}"
        stem = f"{od}/f{f:06d}_s{seed}"
        if a.skip_existing and os.path.exists(stem + ".json"):
            continue
        os.makedirs(od, exist_ok=True)
        try:
            rgb, m_full, depth, Ks, m_d, K = load_inputs(job)
            intr = CameraIntrinsics(fx=float(Ks[0, 0]), fy=float(Ks[1, 1]), cx=float(Ks[0, 2]), cy=float(Ks[1, 2]),
                                    width=depth.shape[1], height=depth.shape[0])
            pm = I2M._depth_to_pointmap(depth, intr, mask=m_d)
            rgba = I2M._merge_mask_to_rgba(rgb, m_full)
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            out = pipe.run(rgba, None, seed, stage1_only=False, with_mesh_postprocess=a.mesh_postprocess,
                           with_texture_baking=False, with_layout_postprocess=a.layout_postprocess,
                           use_vertex_color=True, stage1_inference_steps=a.steps1, stage2_inference_steps=a.steps2,
                           pointmap=pm)
            dt = time.time() - t1
            tr = I2M._to_opencv_transform(out["rotation"][0].tolist(), out["translation"][0].tolist(),
                                          out["scale"][0].tolist())
            Mt = tr.to_matrix()
            I2M._export_mesh(out["glb"], stem + ".glb.tmp.glb")
            os.replace(stem + ".glb.tmp.glb", stem + ".glb")
            mesh = mesh_from_glb(trimesh.load(stem + ".glb"))
            V = np.asarray(mesh.vertices)
            Vc = V @ Mt[:3, :3].T + Mt[:3, 3]
            mc = trimesh.Trimesh(Vc, np.asarray(mesh.faces), vertex_colors=getattr(mesh.visual, "vertex_colors", None),
                                 process=False)
            mc.export(stem + ".tmp.ply")
            os.replace(stem + ".tmp.ply", stem + ".ply")
            chk = depth_agreement(Vc, np.asarray(mesh.faces), depth, Ks, m_d)
            info = dict(job=job, seconds=round(dt, 1), peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                        cam_a_T_obj=Mt.tolist(), transform=tr.to_dict(), n_verts=len(V), n_faces=len(mesh.faces),
                        extent_cm=(np.ptp(Vc, 0) * 100).round(1).tolist(), depth_check=chk,
                        steps=[a.steps1, a.steps2], mesh_postprocess=a.mesh_postprocess,
                        layout_postprocess=a.layout_postprocess)
            json.dump(info, open(stem + ".json.tmp", "w"), indent=1)
            os.replace(stem + ".json.tmp", stem + ".json")
            print(json.dumps(dict(sp=sp, ep=int(job["episode"]), obj=obj, frame=f, seed=seed, s=info["seconds"],
                                  vram=info["peak_vram_gb"], ext=info["extent_cm"], chk=chk)), flush=True)
        except Exception as ex:  # keep going with the other jobs
            import traceback
            traceback.print_exc()
            print(json.dumps(dict(sp=sp, ep=int(job["episode"]), obj=obj, frame=f, error=str(ex)[:300])), flush=True)
        torch.cuda.empty_cache()
    print(f"done {len(jobs)} jobs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
