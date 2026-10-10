"""Score candidate object meshes against OUR multi-view observations of one episode (no GT; usable on evaluation).

For one (split, episode, object) and a list of candidate meshes, each with a pose in the VO world, the script
  1. picks up to --n_views cam_a frames of the stage-1 static window (object static in the VO world, hand-free),
  2. back-projects the masked FoundationStereo depth of those frames into the VO world (observed points),
  3. refines each candidate's world pose with point-to-plane ICP (observed points -> candidate surface, rigid; optional
     scale search), so every candidate is compared at its best placement,
  4. scores it: silhouette IoU against the SAM3 masks in every view (pixels under the dilated hand mask ignored),
     median |rendered z - observed z| on the overlap, and the observed->candidate distance (median / p90).
Writes OUT.json with the per-candidate scores and the refined world_T_mesh, so the best candidate can be used as the
episode's mesh + FoundationPose initialisation (t3_mesh_adopt.py).

Candidate spec (JSON list): [{"name": "tsdf", "mesh": ".../final/.../obj.ply", "json": "...obj.json"} (pose from the
stage-1 JSON frame_T_obj), {"name": "sam3do_f157", "mesh": ".../f000157_s0.ply", "cam_a_frame": 157} (mesh already
in metric cam_a coordinates of that frame)].

  /mnt/secondary/v2d/envs/t3-fpose/bin/python -I t3_mesh_select.py --split public --episode 21 --obj blue_cup \
      --cands CANDS.json --out OUT.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t3_depth_warp import decode_inv_depth  # noqa: E402
from t3_sync import load_lag, stereo_index_for_cam_a  # noqa: E402

R0 = "/mnt/secondary/v2d/t3"


def raster(Vc, F, K, H, W):
    """Silhouette (bool HxW) and z-buffer (float, inf = empty) of a mesh in camera coordinates."""
    z = Vc[:, 2]
    good = z > 0.05
    uv = np.stack([K[0, 0] * Vc[:, 0] / np.maximum(z, 1e-6) + K[0, 2],
                   K[1, 1] * Vc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]], 1)
    tri_ok = good[F].all(1)
    tris = np.round(uv[F[tri_ok]] * 4).astype(np.int32)  # 2 bits of sub-pixel precision
    sil = np.zeros((H, W), np.uint8)
    if len(tris):
        cv2.fillPoly(sil, list(tris), 1, lineType=cv2.LINE_8, shift=2)
    zb = np.full((H, W), np.inf, np.float32)
    u = np.round(uv[good, 0]).astype(int)
    v = np.round(uv[good, 1]).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    np.minimum.at(zb, (v[inb], u[inb]), z[good][inb].astype(np.float32))
    zf = np.where(np.isfinite(zb), zb, 1e3).astype(np.float32)
    zf = -cv2.dilate(-zf, np.ones((5, 5), np.uint8))
    zb = np.where((zf < 1e2) & (sil > 0), zf, np.inf)
    return sil > 0, zb


class Episode:
    def __init__(self, split, ep, obj, scale=0.5):
        self.sp, self.e, self.obj = split, f"episode_{ep:06d}", obj
        self.meta = json.load(open(f"{R0}/frames/{split}/{self.e}/meta.json"))
        n = self.meta["n_frames"]
        K = np.array(self.meta["images"]["cam_a"]["K"], float)
        self.K = K.copy()
        self.K[:2] *= scale
        W0, H0 = self.meta["images"]["cam_a"]["size_wh"]
        self.W, self.H = int(round(W0 * scale)), int(round(H0 * scale))
        vo = f"{R0}/vo_masked/{split}/{self.e}/vo_cuvslam.npz"
        if not os.path.exists(vo):
            vo = f"{R0}/vo/{split}/{self.e}/vo_cuvslam.npz"
        self.vo = vo
        Wr = np.load(vo)["world_T_rect"]
        lag = load_lag(split, self.e, n)
        self.src = stereo_index_for_cam_a(lag) if lag.any() else np.arange(n)
        self.lag = lag
        T_a_rect = np.array(self.meta["transforms"]["T_a_rect"])
        self.world_T_cam = Wr[self.src] @ np.linalg.inv(T_a_rect)  # cam_a frame j
        self.n = n
        self.scale = scale

    def mask(self, name, j):
        m = cv2.imread(f"{R0}/masks/{self.sp}/{self.e}/cam_a_s{self.scale:g}/{name}/{j:06d}.png", 0)
        return np.zeros((self.H, self.W), bool) if m is None else m > 0

    def depth(self, j):
        return decode_inv_depth(cv2.imread(f"{R0}/depth/{self.sp}/{self.e}/depth_cam_a_s{self.scale:g}/{j:06d}.png",
                                           cv2.IMREAD_UNCHANGED))

    def window_cam_frames(self, stage1_json, n_views):
        s1 = json.load(open(stage1_json))
        fr = list(s1["frames"])
        info = s1.get("info", {})
        if info.get("window") == "pre+post":  # frame_T_obj is the pre-grasp pose; post views were registered onto it
            fr = [f for f in fr if f < info.get("first_contact", 10 ** 9)] or fr
        win = sorted({int(f + self.lag[f]) for f in fr if f < len(self.lag)})
        win = [j for j in win if j < self.n and np.isfinite(self.world_T_cam[j]).all() and self.mask(self.obj, j).sum() > 100]
        if len(win) > n_views:
            win = [win[int(round(i))] for i in np.linspace(0, len(win) - 1, n_views)]
        return win

    def observed(self, frames, erode=2, hand_px=10, max_pts=8000, seed=0):
        pts = []
        ker = np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)
        hk = np.ones((2 * hand_px + 1, 2 * hand_px + 1), np.uint8)
        for j in frames:
            m = cv2.erode(self.mask(self.obj, j).astype(np.uint8), ker) > 0
            m &= ~(cv2.dilate(self.mask("hand", j).astype(np.uint8), hk) > 0)
            d = self.depth(j)
            v, u = np.nonzero(m & (d > 0.05) & (d < 2.0))
            z = d[v, u]
            P = np.stack([(u - self.K[0, 2]) * z / self.K[0, 0], (v - self.K[1, 2]) * z / self.K[1, 1], z], 1)
            Tw = self.world_T_cam[j]
            pts.append(P @ Tw[:3, :3].T + Tw[:3, 3])
        P = np.concatenate(pts) if pts else np.zeros((0, 3))
        if len(P) > max_pts:
            P = P[np.random.default_rng(seed).choice(len(P), max_pts, replace=False)]
        return P


def icp_obs_to_mesh(obs, V, F, max_ds=(0.04, 0.02, 0.01), iters=30):
    """Rigid correction C (4x4, applied to the mesh in world coordinates) so that C @ mesh fits the observed points
    (point-to-plane ICP observed -> mesh surface, coarse to fine).  Returns C and the observed->mesh distances."""
    import open3d as o3d
    import trimesh
    from scipy.spatial import cKDTree
    tgt = trimesh.Trimesh(V, F, process=False)
    S, fi = trimesh.sample.sample_surface(tgt, 30000, seed=0)
    pt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(S))
    pt.normals = o3d.utility.Vector3dVector(tgt.face_normals[fi])
    ps = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obs))
    T = np.eye(4)
    for md in max_ds:
        r = o3d.pipelines.registration.registration_icp(
            ps, pt, md, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
        T = r.transformation  # obs -> mesh
    moved = obs @ T[:3, :3].T + T[:3, 3]
    dist = cKDTree(S).query(moved)[0]
    return np.linalg.inv(T), dist


def fit_candidate(ep, frames, obs, Vw, F, scales=(1.0,), rot_deg=(0.0,), score_views_n=4):
    """Best similarity placement A (4x4: rotation perturbation + scale s about the mesh centroid, then rigid ICP on the
    observed points), chosen by mean silhouette IoU over the window views.  The search runs on a decimated copy and
    a subset of views; the winner is re-scored on all views with the full mesh."""
    import itertools
    from scipy.spatial.transform import Rotation as Rot
    c = Vw.mean(0)
    Vd, Fd = Vw, F
    if len(F) > 3000:
        import trimesh
        from t3_mesh_complete import decimate
        md = decimate(trimesh.Trimesh(Vw, F, process=False), 3000)
        Vd, Fd = np.asarray(md.vertices), np.asarray(md.faces)
    sub = frames[:: max(1, len(frames) // score_views_n)][:score_views_n]
    eul = [e for e in itertools.product(rot_deg, repeat=3) if sum(x != 0 for x in e) <= 1]  # one axis at a time
    rots = [Rot.from_euler("xyz", e, degrees=True).as_matrix() for e in eul]
    best = None

    def search(scale_list):
        nonlocal best
        for s in scale_list:
            for Rp in rots:
                Sm = np.eye(4)
                Sm[:3, :3] = s * Rp
                Sm[:3, 3] = c - Sm[:3, :3] @ c
                Vs = Vd @ Sm[:3, :3].T + Sm[:3, 3]
                C, dist = icp_obs_to_mesh(obs, Vs, Fd, max_ds=(0.04, 0.02, 0.01), iters=20)
                V2 = Vs @ C[:3, :3].T + C[:3, 3]
                iou, _, _ = score_views(ep, V2, Fd, sub)
                key = (round(iou, 3), -float(np.median(dist)))
                if best is None or key > best[0]:
                    best = (key, C @ Sm, s)
    search(scales)
    # 2026-10-09: SAM 3D layouts can be off by > 40 %: when the winner sits on the grid boundary, keep extending the
    # grid in that direction (step 0.1, up to 2.2 / down to 0.5) while it keeps winning
    if len(scales) > 1:
        step = round(float(scales[1] - scales[0]), 3)
        while best[2] == max(scales) and max(scales) + step <= 2.2 + 1e-9:
            scales = list(scales) + [round(max(scales) + step, 3)]
            search(scales[-1:])
        while best[2] == min(scales) and min(scales) - step >= 0.5 - 1e-9:
            scales = [round(min(scales) - step, 3)] + list(scales)
            search(scales[:1])
    A, s = best[1], best[2]
    V2 = Vw @ A[:3, :3].T + A[:3, 3]
    C, dist = icp_obs_to_mesh(obs, V2, F, max_ds=(0.01, 0.006))
    A = C @ A
    V2 = Vw @ A[:3, :3].T + A[:3, 3]
    iou, iou_min, dz = score_views(ep, V2, F, frames)
    return A, s, dist, iou, iou_min, dz


def score_views(ep, Vw, F, frames, hand_px=10):
    ious, dzs = [], []
    hk = np.ones((2 * hand_px + 1, 2 * hand_px + 1), np.uint8)
    for j in frames:
        Tc = np.linalg.inv(ep.world_T_cam[j])
        Vc = Vw @ Tc[:3, :3].T + Tc[:3, 3]
        sil, zb = raster(Vc, F, ep.K, ep.H, ep.W)
        m = ep.mask(ep.obj, j)
        hand = cv2.dilate(ep.mask("hand", j).astype(np.uint8), hk) > 0
        care = ~hand
        inter = (sil & m & care).sum()
        uni = ((sil | m) & care).sum()
        ious.append(inter / max(uni, 1))
        d = ep.depth(j)
        sel = m & care & np.isfinite(zb) & (d > 0.05)
        sel = cv2.erode(sel.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if sel.sum() > 20:
            dzs.append(np.median(np.abs(zb[sel] - d[sel])))
    return float(np.mean(ious)), float(np.min(ious)), (float(np.median(dzs)) if dzs else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--stage1_json", default=None, help="window source (default: meshes/stage1/<...>/<obj>.json)")
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--scale_search", action="store_true")
    ap.add_argument("--rot_search", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import trimesh
    ep = Episode(a.split, a.episode, a.obj)
    s1j = a.stage1_json or f"{R0}/meshes/stage1/{a.split}/{ep.e}/{a.obj}.json"
    frames = ep.window_cam_frames(s1j, a.n_views)
    obs = ep.observed(frames)
    res = dict(split=a.split, episode=a.episode, obj=a.obj, views=frames, n_obs=len(obs), cands=[])
    for c in json.load(open(a.cands)):
        m = trimesh.load(c["mesh"], force="mesh", process=False)
        V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
        if "cam_a_frame" in c:
            Tw = ep.world_T_cam[int(c["cam_a_frame"])]
        else:
            Tw = np.array(json.load(open(c["json"]))["frame_T_obj"])
        Vw = V @ Tw[:3, :3].T + Tw[:3, 3]
        iou0, _, dz0 = score_views(ep, Vw, F, frames)
        gen = "cam_a_frame" in c  # generated candidate: pose/scale from a single image, search around it
        scales = np.round(np.arange(0.8, 1.451, 0.1), 3) if (a.scale_search and gen) else (1.0,)
        rots = (-30.0, -15.0, 0.0, 15.0, 30.0) if (a.rot_search and gen) else (0.0,)
        A, s, dist, iou, iou_min, dz = fit_candidate(ep, frames, obs, Vw, F, scales, rots)
        Vw2 = Vw @ A[:3, :3].T + A[:3, 3]
        r = dict(name=c["name"], mesh=c["mesh"], iou_init=round(iou0, 4), dz_init_mm=round(dz0 * 1000, 2),
                 iou=round(iou, 4), iou_min=round(iou_min, 4), dz_mm=round(dz * 1000, 2), scale=float(s),
                 obs_med_mm=round(float(np.median(dist)) * 1000, 2), obs_p90_mm=round(float(np.percentile(dist, 90)) * 1000, 2),
                 world_T_mesh_sim=(A @ Tw).tolist(), extent_cm=(np.ptp(Vw2, 0) * 100).round(1).tolist())
        res["cands"].append(r)
        print(json.dumps({k: r[k] for k in ("name", "iou_init", "iou", "iou_min", "dz_mm", "obs_med_mm", "obs_p90_mm", "scale")}),
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out + ".tmp", "w"), indent=1)
    os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
