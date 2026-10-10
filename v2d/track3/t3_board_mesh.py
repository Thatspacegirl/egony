"""Flat boards lying on the table (wooden_piece_*): rebuild the mesh as an extruded footprint + the observed marker balls.

Why: the official scorer registers our mesh onto the scan by ICP; a symmetric slab registers 180-deg flipped half of
the time (t3_reg_check.py, public 23/24), and the flipped anchor rotates the whole episode.  The only asymmetric
features of the boards are the through-slot (from one long edge), the I-shaped end notches and the mocap balls on ONE
face, so they must be in the mesh at the observed places.  Everything comes from OUR observations of the episode's
static pre-grasp window (VO world, FoundationStereo depth, SAM3 masks, undistorted cam_a colour), nothing from scans:
  1. table plane (RANSAC on depth outside all masks) and board top plane (RANSAC on the board's points); the board
     must lie flat (normals within --max_tilt deg), else the script exits without output;
  2. footprint on a 1 mm grid of the top plane = majority vote over the window views of the SAM3 mask pixels
     back-projected onto the top plane, minus the slot: thin dark line inside the footprint in the full-res cam_a
     image (black-hat), also voted over views;
  3. balls = clusters of board points 4-30 mm above the top plane; each becomes a sphere (radius from the cluster
     width, clipped to 5-9 mm) on a stalk;
  4. occupancy volume (1 mm) = footprint x [0, top height] + balls -> marching cubes -> mesh, bbox-centred, decimated to
     <= 4000 faces; JSON = stage-1 JSON with the new frame_T_obj (VO world) so FoundationPose/assembly use it as is.

  /mnt/secondary/v2d/envs/t3-fpose/bin/python -I t3_board_mesh.py --split public --episode 23 --obj wooden_piece_1 \
      [--stage board] [--debug_png P]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t3_mesh_select import Episode  # noqa: E402

R0 = "/mnt/secondary/v2d/t3"


def ransac_plane(P, thr=0.004, iters=400, seed=0):
    rng = np.random.default_rng(seed)
    best, bn = None, -1
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < thr
        if inl.sum() > bn:
            bn, best = inl.sum(), inl
    Q = P[best]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][2]
    return n, -n @ c, best


def static_prefix(ep, obj, step=1, max_frames=120, tol=0.004, shift=0.012):
    pts0 = None
    out = []
    for j in range(0, min(ep.n, max_frames), step):
        P = ep.observed([j], erode=3, hand_px=5, max_pts=4000)
        if len(P) < 200:
            break
        if pts0 is None:
            n0, d0, _ = ransac_plane(P, thr=0.003, iters=200)
            c0 = P.mean(0)
            pts0 = P
        dist = np.abs(P @ n0 + d0)
        cs = P.mean(0) - c0
        cs -= (cs @ n0) * n0
        print(f"  static_prefix f{j}: plane dist {np.median(dist) * 1000:.1f} mm, shift {np.linalg.norm(cs) * 1000:.1f} mm", flush=True)
        if np.median(dist) > tol or np.linalg.norm(cs) > shift:
            break
        out.append(j)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--stage", default="board")
    ap.add_argument("--stage1_json", default=None)
    ap.add_argument("--n_views", type=int, default=10)
    ap.add_argument("--grid_mm", type=float, default=1.0)
    ap.add_argument("--max_tilt", type=float, default=15.0)
    ap.add_argument("--no_slot", action="store_true")
    ap.add_argument("--ball_r", type=float, default=0.007)
    ap.add_argument("--ball_h", type=float, default=0.012, help="ball centre height above the board top (m)")
    ap.add_argument("--ball_min_pts", type=int, default=8)
    ap.add_argument("--debug_png", default=None)
    a = ap.parse_args()
    import trimesh
    from skimage.measure import marching_cubes
    from t3_mesh_complete import decimate

    ep = Episode(a.split, a.episode, a.obj)
    s1j = a.stage1_json or f"{R0}/meshes/stage1/{a.split}/{ep.e}/{a.obj}.json"
    s1 = json.load(open(s1j))
    if s1.get("info", {}).get("window") in ("pre_grasp", "pre+post"):
        views = ep.window_cam_frames(s1j, a.n_views)
    else:
        # no hand-free pre-grasp window (a hand hovers next to the board from frame 0): take the prefix of frames in
        # which the board's own depth points stay on the frame-0 board plane in the VO world (static board)
        views = static_prefix(ep, a.obj)
        print(f"stage-1 window is {s1.get('info', {}).get('window')}; static prefix: {views[:3]}..{views[-1:]} "
              f"({len(views)} frames)", flush=True)
        if len(views) < 3:
            raise SystemExit("no static prefix: skip (keep the old mesh)")
        if len(views) > a.n_views:
            views = [views[int(round(i))] for i in np.linspace(0, len(views) - 1, a.n_views)]
    objs = ep.meta["objects"]
    # --- table plane (world) from the depth outside every object/hand mask
    TP = []
    for j in views[:4]:
        d = ep.depth(j)
        ex = np.zeros_like(d, bool)
        for o in list(objs) + ["hand"]:
            ex |= cv2.dilate(ep.mask(o, j).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        v, u = np.nonzero((d > 0.25) & (d < 1.0) & ~ex)
        sel = np.random.default_rng(0).choice(len(u), min(len(u), 15000), replace=False)
        u, v = u[sel], v[sel]
        z = d[v, u]
        P = np.stack([(u - ep.K[0, 2]) * z / ep.K[0, 0], (v - ep.K[1, 2]) * z / ep.K[1, 1], z], 1)
        Tw = ep.world_T_cam[j]
        TP.append(P @ Tw[:3, :3].T + Tw[:3, 3])
    nt, dt, _ = ransac_plane(np.concatenate(TP), thr=0.006)
    cam_c = ep.world_T_cam[views[0]][:3, 3]
    if nt @ cam_c + dt < 0:
        nt, dt = -nt, -dt
    # --- board points and top plane
    obs = ep.observed(views, erode=2, max_pts=30000)
    h = obs @ nt + dt
    cand = obs[(h > 0.004) & (h < 0.03)]
    nb, db, inl = ransac_plane(cand, thr=0.0025)
    if nb @ nt < 0:
        nb, db = -nb, -db
    tilt = float(np.degrees(np.arccos(np.clip(nb @ nt, -1, 1))))
    top = cand[inl]
    h_top = float(np.median(top @ nt + dt))
    print(f"table/top tilt {tilt:.1f} deg, top height {h_top * 1000:.1f} mm, {len(top)} top points", flush=True)
    if tilt > a.max_tilt:
        raise SystemExit("board not lying flat: skip")
    # 2D frame on the table plane: x = major axis of the top points, z = table normal (up)
    c0 = top.mean(0) - (top.mean(0) @ nt + dt) * nt
    Q = top - c0
    Q2 = Q - np.outer(Q @ nt, nt)
    ev = np.linalg.svd(Q2 - Q2.mean(0), full_matrices=False)[2]
    ex_ = ev[0] - (ev[0] @ nt) * nt
    ex_ /= np.linalg.norm(ex_)
    ey_ = np.cross(nt, ex_)
    W_T_B = np.eye(4)  # board/table frame -> world
    W_T_B[:3, 0], W_T_B[:3, 1], W_T_B[:3, 2], W_T_B[:3, 3] = ex_, ey_, nt, c0
    B_T_W = np.linalg.inv(W_T_B)
    g = a.grid_mm / 1000.0
    half = 0.2  # 40 cm grid
    ng = int(2 * half / g)

    def to_grid(xy):
        return np.floor((xy + half) / g).astype(int)

    # --- footprint votes from masks (back-projected onto the top plane), and slot votes from the colour image
    votes = np.zeros((ng, ng), np.int32)
    slot_votes = np.zeros((ng, ng), np.int32)
    seen = np.zeros((ng, ng), np.int32)
    gy, gx = np.mgrid[0:ng, 0:ng]
    cell = np.stack([(gx + 0.5) * g - half, (gy + 0.5) * g - half, np.full(gx.shape, h_top)], -1).reshape(-1, 3)
    cell_w = cell @ W_T_B[:3, :3].T + W_T_B[:3, 3]
    for j in views:
        Tc = np.linalg.inv(ep.world_T_cam[j])
        pc = cell_w @ Tc[:3, :3].T + Tc[:3, 3]
        uu = ep.K[0, 0] * pc[:, 0] / pc[:, 2] + ep.K[0, 2]
        vv = ep.K[1, 1] * pc[:, 1] / pc[:, 2] + ep.K[1, 2]
        ok = (uu >= 0) & (uu < ep.W - 1) & (vv >= 0) & (vv < ep.H - 1) & (pc[:, 2] > 0.1)
        m = ep.mask(a.obj, j)
        hand = cv2.dilate(ep.mask("hand", j).astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
        ui, vi = np.round(uu[ok]).astype(int), np.round(vv[ok]).astype(int)
        good = ~hand[vi, ui]
        idx = np.nonzero(ok)[0][good]
        mv = m[vi[good], ui[good]]
        seen.flat[idx] += 1
        votes.flat[idx[mv]] += 1
        if not a.no_slot:
            # thin dark structure inside the board in the full-res image: grey black-hat (closing - image)
            img = cv2.imread(f"{R0}/frames/{ep.sp}/{ep.e}/cam_a/{j:06d}.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
            bh = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) - img
            mf = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            mf = cv2.erode(mf.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            dark = (bh > 25) & mf
            uf, vf = np.round(uu[ok] / ep.scale).astype(int), np.round(vv[ok] / ep.scale).astype(int)
            uf, vf = np.clip(uf, 0, img.shape[1] - 1), np.clip(vf, 0, img.shape[0] - 1)
            slot_votes.flat[idx] += dark[vf[good], uf[good]]
    foot = (votes >= 0.5 * np.maximum(seen, 1)) & (seen >= max(2, len(views) // 3))
    foot = cv2.morphologyEx(foot.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    n_lab, lab, st, _ = cv2.connectedComponentsWithStats(foot.astype(np.uint8))
    if n_lab < 2:
        raise SystemExit("empty footprint")
    foot = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    # fill enclosed holes (< 6 cm2; e.g. under the marker balls, whose silhouettes vote inconsistently)
    nh, labh, sth, _ = cv2.connectedComponentsWithStats((~foot).astype(np.uint8), connectivity=4)
    for i in range(1, nh):
        x0, y0, w0, h0, ar = sth[i]
        if x0 > 0 and y0 > 0 and x0 + w0 < ng and y0 + h0 < ng and ar * g * g < 6e-4:
            foot |= labh == i
    slot = np.zeros_like(foot)
    if not a.no_slot:
        slot = (slot_votes >= 0.4 * np.maximum(seen, 1)) & foot
        # keep elongated thin components only (the slot: >= 3 cm long, <= 1.2 cm wide)
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(slot.astype(np.uint8))
        keep = np.zeros_like(slot)
        for i in range(1, n2):
            comp = lab2 == i
            ys, xs = np.nonzero(comp)
            P2 = np.stack([xs, ys], 1).astype(float)
            if len(P2) < 20:
                continue
            s = np.linalg.svd(P2 - P2.mean(0), compute_uv=False) / np.sqrt(len(P2))
            length, width = 4 * s[0] * g, 4 * s[1] * g
            if length >= 0.03 and width <= 0.012:
                keep |= comp
        slot = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        foot &= ~slot
    print(f"footprint {foot.sum() * g * g * 1e4:.1f} cm2, slot {slot.sum() * g * g * 1e4:.1f} cm2", flush=True)
    # --- balls: clusters of board points >= 5 mm above the top plane (FoundationStereo flattens the ~14 mm spheres on
    # their ~5 mm pedestals to 5-12 mm bumps, so only their (x, y) is measured; the sphere is modelled with
    # --ball_r and --ball_h = centre height above the board top, read off the public videos, not the scans)
    obs0 = ep.observed(views, erode=0, max_pts=200000)
    Bp = obs0 @ B_T_W[:3, :3].T + B_T_W[:3, 3]
    hb = Bp[:, 2] - h_top
    inside = np.zeros(len(Bp), bool)
    gi = to_grid(Bp[:, :2])
    okg = (gi >= 0).all(1) & (gi < ng).all(1)
    fd = cv2.dilate(foot.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    inside[okg] = fd[gi[okg, 1], gi[okg, 0]]
    bp = Bp[(hb > 0.005) & (hb < 0.04) & inside]
    balls = []
    if len(bp) >= 5:
        import open3d as o3d
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.c_[bp[:, :2], np.zeros(len(bp))]))
        labs = np.asarray(pc.cluster_dbscan(eps=0.008, min_points=5))
        for l in range(labs.max() + 1 if len(labs) else 0):
            C = bp[labs == l]
            if len(C) < max(5, a.ball_min_pts):
                continue
            xy = np.median(C[:, :2], 0)
            balls.append(dict(xy=xy.tolist(), top=h_top + a.ball_h + a.ball_r, r=a.ball_r, n=len(C),
                              bump_mm=float(np.percentile(C[:, 2] - h_top, 90) * 1000)))
        # one ball per 3 cm neighbourhood (FoundationStereo splits / smears a sphere into several bumps)
        balls.sort(key=lambda b: -b["n"])
        merged = []
        for b in balls:
            if all(np.hypot(b["xy"][0] - m["xy"][0], b["xy"][1] - m["xy"][1]) > 0.03 for m in merged):
                merged.append(b)
        balls = merged
    print(f"balls (x cm, y cm, n pts, bump p90 mm): {[(round(b['xy'][0] * 100, 1), round(b['xy'][1] * 100, 1), b['n'], round(b['bump_mm'], 1)) for b in balls]}", flush=True)
    # --- occupancy volume -> marching cubes
    nz = int(np.ceil((h_top + 0.035) / g)) + 2
    occ = np.zeros((ng, ng, nz), bool)  # [y, x, z]
    kz = int(round(h_top / g))
    occ[:, :, 1:kz + 1] = foot[:, :, None]
    zc = (np.arange(nz) - 0.5) * g
    for b in balls:
        cz = b["top"] - b["r"]
        cx, cy = b["xy"]
        X = (np.arange(ng) + 0.5) * g - half
        dx = (X[None, :, None] - cx) ** 2 + (X[:, None, None] - cy) ** 2
        occ |= dx + (zc[None, None, :] - cz) ** 2 <= b["r"] ** 2
        occ |= (dx <= (0.35 * b["r"]) ** 2) & (zc[None, None, :] >= 0) & (zc[None, None, :] <= cz)
    vol = np.pad(occ, 1).astype(np.float32)
    verts, faces, _, _ = marching_cubes(vol, 0.5)
    verts = verts - 1
    Vb = np.stack([(verts[:, 1] + 0.5) * g - half, (verts[:, 0] + 0.5) * g - half, (verts[:, 2] - 0.5) * g], 1)
    mesh = trimesh.Trimesh(Vb, faces[:, ::-1], process=True)
    if mesh.volume < 0:
        mesh.invert()
    trimesh.smoothing.filter_taubin(mesh, iterations=5)
    mesh = decimate(mesh, 4000)
    V = np.asarray(mesh.vertices)
    ctr = (V.max(0) + V.min(0)) / 2
    mesh.vertices = V - ctr
    Tc = np.eye(4)
    Tc[:3, 3] = ctr
    out = dict(s1)
    out["frame_T_obj"] = (W_T_B @ Tc).tolist()
    if s1.get("info", {}).get("window") not in ("pre_grasp", "pre+post"):
        # the mesh pose refers to the static prefix: make it the window FoundationPose initialises from (stereo frames)
        out["frames"] = sorted({int(j - ep.lag[j]) for j in views if j - ep.lag[j] >= 0})
        out["info"] = dict(s1.get("info", {}), window="static_prefix", stage1_window=s1.get("info", {}).get("window"))
    out["board"] = dict(views=views, tilt_deg=tilt, top_height_mm=h_top * 1000, footprint_cm2=float(foot.sum() * g * g * 1e4),
                        slot_cm2=float(slot.sum() * g * g * 1e4), balls=balls, grid_mm=a.grid_mm)
    od = f"{R0}/meshes/{a.stage}/{a.split}/{ep.e}"
    os.makedirs(od, exist_ok=True)
    mesh.export(f"{od}/{a.obj}.tmp.ply")
    os.replace(f"{od}/{a.obj}.tmp.ply", f"{od}/{a.obj}.ply")
    json.dump(out, open(f"{od}/{a.obj}.json.tmp", "w"), indent=1)
    os.replace(f"{od}/{a.obj}.json.tmp", f"{od}/{a.obj}.json")
    print(json.dumps(dict(out=f"{od}/{a.obj}.ply", faces=len(mesh.faces), extent_cm=(np.ptp(mesh.vertices, 0) * 100).round(1).tolist())))
    if a.debug_png:
        dbg = np.dstack([foot * 255, slot * 255, (votes * 255 // max(votes.max(), 1))]).astype(np.uint8)
        for b in balls:
            gi = to_grid(np.array(b["xy"]))
            cv2.circle(dbg, (int(gi[0]), int(gi[1])), int(b["r"] / g), (0, 0, 255), 1)
        cv2.imwrite(a.debug_png, dbg)


if __name__ == "__main__":
    main()
