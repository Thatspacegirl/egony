"""Object mesh by masked TSDF fusion (Open3D, CPU) -- the Docker-free alternative to BundleSDF for Track 3.

Fuses masked rectified-left depth into one TSDF given per-frame camera poses in the frame the mesh should live in:
  * stage 1 (static pre-grasp window): poses = world_T_rect from VO (t3_vo_cuvslam.py) -> mesh in the VO world frame;
  * stage 2 (whole clip, object moving): poses = obj_T_rect = inv(world_T_obj) @ world_T_rect, with world_T_obj from
    FoundationPose tracking -> mesh in the object body frame (all views, incl. in-hand ones).
Post-processing keeps the largest connected component (hand/table fragments inflate the RMS radius used by the CD-O
scale normalisation) and applies light Taubin smoothing.

Fusion:   python -I t3_tsdf_fuse.py --ep_dir FRAMES/<split>/episode_X --depth_dir DEPTH/<split>/episode_X/depth_rect \
              --mask_dir MASKS/<split>/episode_X/left_s1/<object> --poses vo_cuvslam.npz:world_T_rect --frames 0-40 \
              --out mesh.ply [--voxel 0.002]
Self-test (CPU only, single frame, identity pose): SGBM depth + GroundingDINO/SAM2 mask from the t3-seg venv
          python -I t3_tsdf_fuse.py --ep_dir ... --self_test 0 --prompt "blue cup" --out /tmp/x.ply
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t3_depth_warp import decode_inv_depth  # noqa: E402

SEG_SNIPPET = """
import sys, cv2, numpy as np, torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
img_path, prompt, out = sys.argv[1:4]
W = '/mnt/secondary/v2d/weights'
img = Image.fromarray(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
with torch.inference_mode():
    d = W + '/grounding_dino/grounding-dino-base'
    p = AutoProcessor.from_pretrained(d); g = AutoModelForZeroShotObjectDetection.from_pretrained(d)
    i = p(images=img, text=prompt + '.', return_tensors='pt'); o = g(**i)
    r = p.post_process_grounded_object_detection(o, i.input_ids, threshold=0.3, text_threshold=0.25,
                                                 target_sizes=[img.size[::-1]])[0]
    b = r['boxes'][int(r['scores'].argmax())].numpy()
    print('GDINO box', b.round().tolist(), 'score', round(float(r['scores'].max()), 3))
    s = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',
                                      W + '/sam2/sam2.1-hiera-large/sam2.1_hiera_large.pt', device='cpu'))
    s.set_image(np.asarray(img)); m, _, _ = s.predict(box=b[None], multimask_output=False)
cv2.imwrite(out, (m[0] > 0).astype(np.uint8) * 255)
"""


def static_frames_v2(mask_dir, hand_dir, depth_dir, P, K, n, contact_px=20, min_px=200, tol=0.015, dev=0.03,
                     max_frames=60):
    """Frames in which the object rests in its INITIAL pose, by world-frame consistency instead of only the
    contiguous hand-free run at the clip start:
      c_f = world median of the object's cleaned masked depth (VO pose P[f]); reference = median c_f over the first
      5 hand-free visible frames.  If no hand-free frame ever deviates by > dev from it, the object never moved (e.g.
      the dish rack) and every hand-free frame within tol is used; otherwise only those before the first deviation.
    Falls back to the final pose (reference = last 5 hand-free frames, frames after the last deviation) when fewer
    than 3 initial frames exist.  Returns (frames, info)."""
    k = np.ones((2 * contact_px + 1, 2 * contact_px + 1), np.uint8)
    C = np.full((n, 3), np.nan)
    free = np.zeros(n, bool)
    for f in range(n):
        m = cv2.imread(f"{mask_dir}/{f:06d}.png", 0)
        if m is None or (m > 0).sum() < min_px or not np.isfinite(P[f]).all():
            continue
        h = cv2.imread(f"{hand_dir}/{f:06d}.png", 0) if hand_dir else None
        if h is not None and h.any() and (cv2.dilate((h > 0).astype(np.uint8), k) > 0)[m > 0].any():
            continue
        d = decode_inv_depth(cv2.imread(f"{depth_dir}/{f:06d}.png", cv2.IMREAD_UNCHANGED))
        mc = clean_mask_depth(d, m > 0, 5)
        if mc.sum() < min_px:
            continue
        v, u = np.nonzero(mc)
        z = d[v, u]
        X = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
        C[f] = np.median(X @ P[f][:3, :3].T + P[f][:3, 3], 0)
        free[f] = True
    idx = np.nonzero(free)[0]
    info = dict(n_free=int(len(idx)))
    if len(idx) < 3:
        return [], dict(info, window="none")

    def pick(order, name):
        ref = np.median(C[order[:5]], 0)
        dist = np.linalg.norm(C[order] - ref, axis=1)
        moved = np.nonzero(dist > dev)[0]
        if len(moved) == 0:
            sel = order[dist < tol]
            return sel, dict(window=name + "_never_moved")
        sel = order[:moved[0]]
        sel = sel[np.linalg.norm(C[sel] - ref, axis=1) < tol]
        return sel, dict(window=name, first_deviation=int(order[moved[0]]))
    sel, w = pick(idx, "initial")
    if len(sel) < 3:
        sel, w = pick(idx[::-1], "final")
        sel = np.sort(sel)
    info.update(w, n_static=int(len(sel)))
    sel = list(sel[:: max(1, int(np.ceil(len(sel) / max_frames)))])
    info["n_used"] = len(sel)
    return [int(f) for f in sel], info


def parse_frames(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), (int(b) if b else int(a)) + 1))
    return [f for f in out if f < n]


def static_window(mask_dir, hand_dir, n, contact_px=20, min_px=200, stride=2, max_frames=60, min_pre=8):
    """Frames where the object is visible and the (dilated) hand mask does not touch it, taken from the longest
    contiguous no-contact run that starts at the clip start (pre-grasp) or ends at the clip end (post-release); the
    object is static in the world there, so VO poses alone register the views.  Returns (frames, info)."""
    k = np.ones((2 * contact_px + 1, 2 * contact_px + 1), np.uint8)
    vis, touch = np.zeros(n, bool), np.zeros(n, bool)
    for f in range(n):
        m = cv2.imread(f"{mask_dir}/{f:06d}.png", 0)
        if m is None:
            continue
        m = m > 0
        vis[f] = m.sum() >= min_px
        h = cv2.imread(f"{hand_dir}/{f:06d}.png", 0) if hand_dir else None
        if h is not None and h.any():
            if h.shape != m.shape:
                h = cv2.resize(h, m.shape[::-1], interpolation=cv2.INTER_NEAREST)
            touch[f] = (cv2.dilate((h > 0).astype(np.uint8), k) > 0)[m].any()
    if contact_px > 6 and touch[:3].any() and vis[:3].any():
        # a hand hovering next to the object from the very first frame (public 24): retry with a tighter dilation so a
        # real pre-grasp window can still be found
        return static_window(mask_dir, hand_dir, n, contact_px // 2, min_px, stride, max_frames, min_pre)
    first = int(np.argmax(touch)) if touch.any() else n           # first contact
    last = int(n - 1 - np.argmax(touch[::-1])) if touch.any() else -1  # last contact
    pre = [f for f in range(0, first) if vis[f]]
    post = [f for f in range(last + 1, n) if vis[f]]
    # prefer the pre-grasp window (initial resting pose; after release objects are often stacked / slotted / inside
    # another object) whenever it has >= min_pre frames
    win, name = (pre, "pre_grasp") if (len(pre) >= min_pre or len(pre) >= len(post)) else (post, "post_release")
    sel = win[::max(stride, int(np.ceil(len(win) / max_frames)))] if win else []  # spread over the whole window
    info = dict(window=name, first_contact=first, last_contact=last, n_pre=len(pre), n_post=len(post),
                n_used=len(sel), stride=stride, contact_px=contact_px)
    info["_pre"], info["_post"] = pre, post
    return sel, info


def windows_agree(depths_a, masks_a, poses_a, depths_b, masks_b, poses_b, K, size, voxel=0.003):
    """Did the object stay put between two static windows?  Fuse each window, ICP (point-to-plane, identity init,
    1 cm) -> (agree, info); agree = fitness >= 0.5 and the correction is < 3 cm and < 5 deg (VO drift between the
    windows); info["T"] maps window b onto window a and is applied to b's camera poses before the joint fusion."""
    import open3d as o3d
    ma = fuse(depths_a, masks_a, poses_a, K, size, voxel=voxel)
    mb = fuse(depths_b, masks_b, poses_b, K, size, voxel=voxel)
    if len(ma.vertices) < 100 or len(mb.vertices) < 100:
        return False, dict(reason="empty")
    pa = ma.sample_points_uniformly(5000)
    pb = mb.sample_points_uniformly(5000)
    for p in (pa, pb):
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    r = o3d.pipelines.registration.registration_icp(pb, pa, 0.01, np.eye(4),
                                                    o3d.pipelines.registration.TransformationEstimationPointToPlane())
    T = r.transformation
    dt = float(np.linalg.norm(T[:3, 3]))
    dr = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
    ok = r.fitness >= 0.5 and dt < 0.03 and dr < 5
    return ok, dict(fitness=round(float(r.fitness), 3), shift_cm=round(dt * 100, 2), rot_deg=round(dr, 2), T=T)


def clean_mask_depth(d, m, erode_px=3, disc=0.03, gate=0.35):
    """Masked depth without silhouette bleed: erode, drop pixels at depth discontinuities (5x5 max-min > disc m: the
    cam_a->left mask transfer and SAM3 edges let background/foreground pixels in at the border), keep the largest
    4-connected blob plus blobs >= 2 % of it at a similar depth (cup interiors behind the rim), gate |z - median| < gate."""
    m = m.astype(np.uint8)
    if erode_px:
        m = cv2.erode(m, np.ones((erode_px, erode_px), np.uint8))
    dz = np.where(d > 0, d, 0).astype(np.float32)
    k = np.ones((5, 5), np.uint8)
    rng = cv2.dilate(dz, k) - cv2.erode(np.where(d > 0, d, 99).astype(np.float32), k)
    m = (m > 0) & (d > 0) & (rng < disc)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
    if n > 2:  # keep the largest blob and every blob >= 2 % of it whose median depth is near the largest's
        areas = st[1:, cv2.CC_STAT_AREA]
        big = 1 + int(np.argmax(areas))
        zb = np.median(d[lab == big])
        keep = [i for i in range(1, n) if i == big or (areas[i - 1] >= 0.02 * areas.max() and
                                                         abs(np.median(d[lab == i]) - zb) < gate)]
        m = np.isin(lab, keep)
    if m.any():
        m &= np.abs(d - np.median(d[m])) < gate
    return m


def fuse(depths, masks, poses, K, size, voxel=0.002, trunc=0.008, max_depth=2.0, erode_px=3):
    import open3d as o3d
    W, H = size
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    for d, m, T in zip(depths, masks, poses):
        m = clean_mask_depth(d, m, erode_px)  # stay off the silhouette, where stereo depth bleeds
        dd = np.where(m, d, 0).astype(np.float32)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((H, W, 3), np.uint8)), o3d.geometry.Image(dd), depth_scale=1.0,
            depth_trunc=max_depth, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(T))  # Open3D takes the extrinsic cam_T_world
    return vol.extract_triangle_mesh()


def clean(mesh, smooth=5):
    clusters, n_tri, _ = mesh.cluster_connected_triangles()
    clusters, n_tri = np.asarray(clusters), np.asarray(n_tri)
    if len(n_tri):
        mesh.remove_triangles_by_mask(clusters != int(np.argmax(n_tri)))
        mesh.remove_unreferenced_vertices()
    if smooth:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth)
    mesh.compute_vertex_normals()
    return mesh


def vertex_colors_cam_a(Vw, ep_dir, meta, P, frames, depth_dir=None, scale=0.5, tol=0.015, max_views=20):
    """Median cam_a colour per vertex over the fusion views where it projects in-bounds and (if depth_dir)
    agrees with the observed cam_a depth within tol.  Vw: vertices in the --poses frame; P: frame_T_rect."""
    Ka = np.array(meta["images"]["cam_a"]["K"], float)
    W0, H0 = meta["images"]["cam_a"]["size_wh"]
    K = Ka.copy()
    K[:2] *= scale
    W, H = int(round(W0 * scale)), int(round(H0 * scale))
    T_a_rect = np.array(meta["transforms"]["T_a_rect"])
    from t3_sync import load_lag
    lag = load_lag(meta["split"], f"episode_{meta['episode']:06d}", meta["n_frames"])
    cols = []
    step = max(1, len(frames) // max_views)
    for f in frames[::step]:
        a_T_w = T_a_rect @ np.linalg.inv(P[f])
        f_a = int(np.clip(f + lag[f], 0, meta["n_frames"] - 1))  # cam_a frame showing stereo frame f
        X = Vw @ a_T_w[:3, :3].T + a_T_w[:3, 3]
        z = X[:, 2]
        u = np.round(X[:, 0] / z * K[0, 0] + K[0, 2]).astype(int)
        v = np.round(X[:, 1] / z * K[1, 1] + K[1, 2]).astype(int)
        ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        im = cv2.resize(cv2.imread(f"{ep_dir}/cam_a/{f_a:06d}.jpg"), (W, H), interpolation=cv2.INTER_AREA)
        if depth_dir and os.path.exists(f"{depth_dir}/{f_a:06d}.png"):
            d = decode_inv_depth(cv2.imread(f"{depth_dir}/{f_a:06d}.png", cv2.IMREAD_UNCHANGED))
            dd = np.zeros_like(z)
            dd[ok] = d[v[ok], u[ok]]
            ok &= (dd > 0) & (np.abs(dd - z) < tol)
        c = np.full((len(Vw), 3), np.nan)
        c[ok] = im[v[ok], u[ok], ::-1] / 255.0
        cols.append(c)
    C = np.nanmedian(np.stack(cols), 0) if cols else np.full((len(Vw), 3), np.nan)
    miss = ~np.isfinite(C).all(1)
    C[miss] = np.nanmedian(C[~miss], 0) if (~miss).any() else 0.5
    print(f"vertex colours: {100 * (1 - miss.mean()):.0f}% of vertices observed in cam_a")
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_dir", required=True)
    ap.add_argument("--depth_dir")
    ap.add_argument("--mask_dir")
    ap.add_argument("--poses", help="npz[:key] holding [T,4,4] rect-left camera poses in the output frame")
    ap.add_argument("--frames", default="", help="e.g. 0-40,60 ; or 'auto' = static window (needs --hand_dir)")
    ap.add_argument("--hand_dir", default=None, help="hand masks (same grid) for --frames auto")
    ap.add_argument("--contact_px", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out")
    ap.add_argument("--voxel", type=float, default=0.002)
    ap.add_argument("--erode", type=int, default=3, help="mask erosion (px) before fusion")
    ap.add_argument("--perception", default=None, help="stage 2: perception.npz (t3_assemble.py) -> object-frame poses")
    ap.add_argument("--obj", default=None, help="stage 2: object name in perception.npz")
    ap.add_argument("--conf_min", type=float, default=0.7, help="stage 2 --frames conf: min FoundationPose mask IoU")
    ap.add_argument("--max_views", type=int, default=60)
    ap.add_argument("--merge_windows", type=int, default=1, help="auto: fuse pre+post windows when they agree")
    ap.add_argument("--no_center", action="store_true", help="keep the mesh in the --poses frame")
    ap.add_argument("--color_cam_a", action="store_true", help="vertex colours from undistorted cam_a frames")
    ap.add_argument("--color_depth_dir", default=None, help="depth_cam_a_s0.5 dir for the colour visibility test")
    ap.add_argument("--self_test", type=int, default=None)
    ap.add_argument("--prompt", default="blue cup")
    ap.add_argument("--seg_python", default="/mnt/secondary/v2d/envs/t3-seg/bin/python")
    a = ap.parse_args()
    import open3d as o3d
    meta = json.load(open(f"{a.ep_dir}/meta.json"))
    K = np.array(meta["images"]["left"]["K"])
    size = tuple(meta["images"]["left"]["size_wh"])
    if a.self_test is not None:
        f = a.self_test
        L = cv2.imread(f"{a.ep_dir}/left/{f:06d}.jpg", 0)
        R = cv2.imread(f"{a.ep_dir}/right/{f:06d}.jpg", 0)
        sg = cv2.StereoSGBM_create(0, 160, 5, P1=200, P2=800, uniquenessRatio=10, speckleWindowSize=100,
                                   speckleRange=2)
        disp = sg.compute(L, R).astype(np.float32) / 16
        d = np.where(disp > 2, meta["stereo"]["fx"] * meta["stereo"]["baseline_m"] / np.maximum(disp, 1e-3), 0)
        out = a.out or "/tmp/tsdf_selftest.ply"
        mask_png = out.rsplit(".", 1)[0] + "_mask.png"
        subprocess.run([a.seg_python, "-I", "-c", SEG_SNIPPET, f"{a.ep_dir}/left/{f:06d}.jpg", a.prompt, mask_png],
                       check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4"))
        m = cv2.imread(mask_png, 0) > 0
        mesh = clean(fuse([d], [m], [np.eye(4)], K, size, voxel=a.voxel))
        v = np.asarray(mesh.vertices)
        print(f"mask {m.sum()} px, median masked depth {np.median(d[m & (d > 0)]):.3f} m; mesh {len(v)} verts "
              f"{len(mesh.triangles)} tris; extent (x,y,z) = {np.round((v.max(0) - v.min(0)) * 100, 1)} cm")
    else:
        world_T_obj_ref = None
        if a.perception:  # stage 2: rect camera in the OBJECT body frame from tracked world poses
            z = np.load(a.perception)
            b = [str(x) for x in z["object_names"]].index(a.obj)
            q = z["object_pose"][:, b]  # x y z qw qx qy qz
            Wo = np.tile(np.eye(4), (len(q), 1, 1))
            Wo[:, :3, :3] = Rotation.from_quat(np.c_[q[:, 4:7], q[:, 3]]).as_matrix()
            Wo[:, :3, 3] = q[:, :3]
            # object pose at the instant of STEREO frame f: perception in the cam_a time base is indexed at f + lag[f]
            N_ = len(Wo)
            tb = str(z["time_base"]) if "time_base" in z else "cam_a"
            lag = z["a_lag"] if "a_lag" in z else np.zeros(N_, int)
            idx = np.clip(np.arange(N_) + lag, 0, N_ - 1) if tb == "cam_a" else np.arange(N_)
            Wo = Wo[idx]
            P = np.linalg.inv(Wo) @ z["world_T_rect"]
            conf = np.nan_to_num(z["object_conf"][idx, b], nan=0.0)
            ok = z["object_valid"][idx, b] & (conf >= a.conf_min)
            if a.frames in ("conf", "auto"):
                cand = np.nonzero(ok)[0]
                frames = [int(f) for f in cand[::max(a.stride, int(np.ceil(len(cand) / a.max_views)))]]
                info = dict(window="conf", conf_min=a.conf_min, n_ok=int(ok.sum()), n_used=len(frames))
                print("stage-2 frames:", info, flush=True)
            ref = int(np.argmax(np.where(ok, conf, -1)))
            world_T_obj_ref = (ref, Wo[ref])
        else:
            path, _, key = a.poses.partition(":")
            P = np.load(path)[key or "world_T_rect"]
        if a.frames in ("auto", "conf") and a.perception:
            pass
        elif a.frames == "auto2":
            frames, info = static_frames_v2(a.mask_dir, a.hand_dir, a.depth_dir, P, K, meta["n_frames"], a.contact_px,
                                            max_frames=a.max_views)
            print("static frames v2:", info, flush=True)
            if len(frames) < 3:
                raise SystemExit(f"too few static frames ({len(frames)})")
        elif a.frames == "auto":
            frames, info = static_window(a.mask_dir, a.hand_dir, meta["n_frames"], a.contact_px, stride=a.stride,
                                         max_frames=a.max_views)
            pre, post = info.pop("_pre"), info.pop("_post")
            if a.merge_windows and len(pre) >= 3 and len(post) >= 3:
                # objects that end where they started (the dish rack receives a cup / pot but never moves): fuse BOTH
                # windows when their separate reconstructions agree in the VO world
                def load(fr):
                    fr = [f for f in fr[:: max(1, len(fr) // 15)] if np.isfinite(P[f]).all()]
                    return ([decode_inv_depth(cv2.imread(f"{a.depth_dir}/{f:06d}.png", cv2.IMREAD_UNCHANGED)) for f in fr],
                            [cv2.imread(f"{a.mask_dir}/{f:06d}.png", 0) > 0 for f in fr], [P[f] for f in fr])
                da, ma_, pa = load(pre)
                db, mb_, pb = load(post)
                ok, ag = windows_agree(da, ma_, pa, db, mb_, pb, K, size)
                Tab = ag.pop("T", np.eye(4))
                info["windows_agree"] = dict(ag, merged=bool(ok))
                if ok:
                    P = P.copy()
                    for f in post:
                        P[f] = Tab @ P[f]  # post-release views registered onto the pre-grasp reconstruction
                    both = pre + post
                    frames = both[:: max(a.stride, int(np.ceil(len(both) / a.max_views)))]
                    info.update(window="pre+post", n_used=len(frames))
            print("static window:", info, flush=True)
            if len(frames) < 3:
                raise SystemExit(f"static window too short ({len(frames)} frames); fuse with FoundationPose poses instead")
        else:
            frames = parse_frames(a.frames, meta["n_frames"])
            info = dict(window="manual")
        frames = [f for f in frames if np.isfinite(P[f]).all()]
        depths = [decode_inv_depth(cv2.imread(f"{a.depth_dir}/{f:06d}.png", cv2.IMREAD_UNCHANGED)) for f in frames]
        masks = []
        for f in frames:
            m = cv2.imread(f"{a.mask_dir}/{f:06d}.png", 0)
            if m.shape[::-1] != size:
                m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
            masks.append(m > 0)
        mesh = clean(fuse(depths, masks, [P[f] for f in frames], K, size, voxel=a.voxel, erode_px=a.erode))
        out = a.out
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        frames_for_json = frames
        v = np.asarray(mesh.vertices)
        if len(v) == 0:
            raise SystemExit("empty mesh")
        print(f"fused {len(frames)} frames {frames[:3]}..{frames[-1:]} -> {len(v)} verts, "
              f"extent {np.round((v.max(0) - v.min(0)) * 100, 1)} cm")
        # body frame: bounding-box centre at the origin (FoundationPose centres on the bbox centre too), axes = the
        # output frame of --poses (VO world for stage 1).  frame_T_obj maps body -> pose frame.
        c = (v.max(0) + v.min(0)) / 2 if not a.no_center else np.zeros(3)
        frame_T_obj = np.eye(4)
        frame_T_obj[:3, 3] = c
        mesh.translate(-c)
        if world_T_obj_ref is not None:  # express the new body frame in the VO world at a well-tracked frame
            info["ref_frame"] = world_T_obj_ref[0]
            frame_T_obj = world_T_obj_ref[1] @ frame_T_obj
            frames_for_json = [world_T_obj_ref[0]]
        if a.color_cam_a:
            col = vertex_colors_cam_a(np.asarray(mesh.vertices) + c, a.ep_dir, meta, P, list(frames),
                                      a.color_depth_dir)
            mesh.vertex_colors = o3d.utility.Vector3dVector(col)
        json.dump(dict(frame_T_obj=frame_T_obj.tolist(), frames=[int(f) for f in frames_for_json], info=info,
                       fused_frames=[int(f) for f in frames], perception=a.perception,
                       poses=a.poses, voxel=a.voxel, erode=a.erode, n_verts=int(len(v)),
                       extent_cm=((v.max(0) - v.min(0)) * 100).round(2).tolist()),
                  open(out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    o3d.io.write_triangle_mesh(out, mesh)
    print("wrote", out)


if __name__ == "__main__":
    main()
