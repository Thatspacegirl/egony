"""Shape completion of stage-1 meshes for objects RESTING ON THE TABLE in the static window (CPU, no GT).

The table plane is estimated from the stereo depth of the window frames (objects + hands masked out) and moved into
the mesh body frame with VO and the stage-1 frame_T_obj, so "down" and the support height are known.
Modes:
  revolve  axisymmetric objects (cups): axis = table normal through the circle fitted to the horizontal projection
           of the fused points; profile r(h) = robust (70th pct) radius per 4 mm height bin from the table up to the
           top; mesh = outer wall + inner wall (r - wall) + bottom disk + inner floor + rim.  Marker balls are lost.
  revolve_keep  revolve the body (points within 1.25 x the 60th-pct radius) and keep the observed faces outside the
           revolved profile + 12 mm (pot handle, marker balls).
  extrude  flat objects lying on the table (boards, spoon): the observed top surface + its projection onto the table
           plane + side walls along the open boundary = a closed 2.5-D solid.
  none     copy.
Output: the completed mesh re-centred on its bounding-box centre, OUT.json = stage-1 JSON with frame_T_obj updated
(frame_T_obj_new = frame_T_obj_stage1 @ T(c)), decimated to <= 4000 faces (vertex colours kept).

  python -I t3_mesh_complete.py --split public --episode 21 --obj blue_cup --mode revolve \
      --in_root /mnt/secondary/v2d/t3/meshes/stage1 --out_root /mnt/secondary/v2d/t3/meshes/stage1c
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t3_assemble import table_plane_rect  # noqa: E402
from t3_depth_warp import decode_inv_depth  # noqa: E402

R0 = "/mnt/secondary/v2d/t3"


def table_in_body(split, e, frames, frame_T_obj, vo_path, n_use=5):
    """table plane (n, d) in the mesh body frame, n pointing up (away from the table, towards the camera)."""
    meta = json.load(open(f"{R0}/frames/{split}/{e}/meta.json"))
    K = np.array(meta["images"]["left"]["K"])
    W = np.load(vo_path)["world_T_rect"]
    obj_T_world = np.linalg.inv(frame_T_obj)
    planes = []
    for f in frames[:: max(1, len(frames) // n_use)][:n_use]:
        d = decode_inv_depth(cv2.imread(f"{R0}/depth/{split}/{e}/depth_rect/{f:06d}.png", cv2.IMREAD_UNCHANGED))
        excl = np.zeros(d.shape, bool)
        for s in meta["objects"] + ["hand"]:
            p = f"{R0}/masks/{split}/{e}/left_s1/{s}/{f:06d}.png"
            if os.path.exists(p):
                excl |= cv2.dilate(cv2.imread(p, 0), np.ones((25, 25))) > 0
        pl, frac = table_plane_rect(d, K, excl)
        if pl is None or frac < 0.2:
            continue
        B = obj_T_world @ W[f]  # body_T_rect
        n = B[:3, :3] @ pl[:3]
        p = B[:3, :3] @ (-pl[3] * pl[:3]) + B[:3, 3]
        planes.append(np.r_[n, -n @ p])
    if not planes:
        return None
    P = np.array(planes)
    n = P[:, :3].mean(0)
    n /= np.linalg.norm(n)
    return n, float(np.median(P[:, 3]))


def basis(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def fit_circle(xy):
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c[:2], np.sqrt(max(c[2] + c[0] ** 2 + c[1] ** 2, 1e-8))


def revolve(V, n, dtab, wall=0.004, nseg=96, bin_h=0.004):
    u, v = basis(n)
    h = V @ n + dtab  # height above the table
    xy = np.c_[V @ u, V @ v]
    keep = h > 0.005
    c, r0 = fit_circle(xy[keep])
    # refine the centre robustly (trimmed)
    for _ in range(3):
        rr = np.linalg.norm(xy[keep] - c, axis=1)
        good = np.abs(rr - np.median(rr)) < 2.5 * np.median(np.abs(rr - np.median(rr))) + 1e-3
        c, r0 = fit_circle(xy[keep][good])
    rr = np.linalg.norm(xy - c, axis=1)
    top = np.percentile(h, 99.5)
    hs = np.arange(0, top + bin_h, bin_h)
    prof = []
    for h0 in hs:
        s = (h >= h0 - bin_h) & (h < h0 + bin_h)
        prof.append(np.percentile(rr[s], 70) if s.sum() >= 5 else np.nan)
    prof = np.array(prof)
    ok = np.isfinite(prof)
    prof = np.interp(hs, hs[ok], prof[ok])
    prof = np.convolve(np.pad(prof, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    th = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    centre3 = c[0] * u + c[1] * v - dtab * n  # point on the table below the axis

    def ring(r, hh):
        return centre3 + hh * n + r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
    verts, faces = [], []

    def add_strip(R1, R2, flip=False):
        o = len(verts) * nseg
        verts.extend([R1, R2])
        for i in range(nseg):
            j = (i + 1) % nseg
            a, b, c_, d = o + i, o + j, o + nseg + j, o + nseg + i
            faces.extend([[a, b, c_], [a, c_, d]] if not flip else [[a, c_, b], [a, d, c_]])
    rings_out = [ring(prof[k], hs[k]) for k in range(len(hs))]
    rin = np.maximum(prof - wall, 0.002)
    floor = min(0.006, top / 3)
    rings_in = [ring(rin[k], max(hs[k], floor)) for k in range(len(hs))]
    for k in range(len(hs) - 1):
        add_strip(rings_out[k], rings_out[k + 1])
        add_strip(rings_in[k], rings_in[k + 1], flip=True)
    add_strip(rings_out[-1], rings_in[-1])  # rim
    V2 = np.concatenate(verts) if verts else np.zeros((0, 3))
    F = np.array(faces)
    # bottom disk + inner floor (fans)
    nb = len(V2)
    bot_c, in_c = centre3, centre3 + floor * n
    V2 = np.vstack([V2, bot_c, in_c, rings_out[0], rings_in[0]])
    i0 = nb + 2
    i1 = i0 + nseg
    fb = [[nb, i0 + (i + 1) % nseg, i0 + i] for i in range(nseg)]
    fi = [[nb + 1, i1 + i, i1 + (i + 1) % nseg] for i in range(nseg)]
    F = np.vstack([F, fb, fi])
    return trimesh.Trimesh(V2, F, process=True), dict(radius_base=float(prof[0]), radius_top=float(prof[-1]),
                                                     height=float(top), centre_xy=c.tolist(),
                                                     profile_h=hs.tolist(), profile_r=prof.tolist())


def keep_outside(mesh, rev_info, n, dtab, margin=0.012):
    """faces of the observed mesh lying outside the revolved profile (+margin): handles, marker balls."""
    V = np.asarray(mesh.vertices)
    u, v = basis(n)
    c = np.array(rev_info["centre_xy"])
    h = V @ n + dtab
    rr = np.linalg.norm(np.c_[V @ u, V @ v] - c, axis=1)
    hs, prof = np.array(rev_info["profile_h"]), np.array(rev_info["profile_r"])
    out = rr > np.interp(h, hs, prof) + margin
    fsel = out[np.asarray(mesh.faces)].all(1)
    sub = mesh.submesh([np.nonzero(fsel)[0]], append=True) if fsel.any() else None
    return sub, int(fsel.sum())


def mirror(mesh, n, dtab, split, e, s1, vo_path, n_pts=12000, n_views=6, lam=2.0, free_tol=0.015):
    """Bilateral-symmetry completion for an object resting on the table: the symmetry plane is vertical (contains the
    table normal n).  Search azimuth (2 deg) x offset (2 mm); score = fraction of mirrored samples within 3 mm of the
    observed surface (structures crossing the plane: rims, spouts, handles) - lam x fraction landing in OBSERVED FREE
    SPACE (in front of the measured depth by > free_tol in any of n_views window frames).  Returns the union of the
    observed mesh and its mirror image."""
    from scipy.spatial import cKDTree
    P = np.asarray(mesh.sample(n_pts))
    tree = cKDTree(P)
    u, v = basis(n)
    meta = json.load(open(f"{R0}/frames/{split}/{e}/meta.json"))
    K = np.array(meta["images"]["left"]["K"])
    Wd, Hd = meta["images"]["left"]["size_wh"]
    Wr = np.load(vo_path)["world_T_rect"]
    F = np.array(s1["frame_T_obj"])
    fr = s1["frames"][:: max(1, len(s1["frames"]) // n_views)][:n_views]
    views = []
    for f in fr:
        d = decode_inv_depth(cv2.imread(f"{R0}/depth/{split}/{e}/depth_rect/{f:06d}.png", cv2.IMREAD_UNCHANGED))
        views.append((np.linalg.inv(Wr[f]) @ F, d))  # rect_T_body

    def free_violation(Q):
        bad = np.zeros(len(Q), bool)
        for T, d in views:
            X = Q @ T[:3, :3].T + T[:3, 3]
            z = X[:, 2]
            ok = z > 0.1
            uu = np.round(X[:, 0] / np.where(ok, z, 1) * K[0, 0] + K[0, 2]).astype(int)
            vv = np.round(X[:, 1] / np.where(ok, z, 1) * K[1, 1] + K[1, 2]).astype(int)
            ok &= (uu >= 0) & (uu < Wd) & (vv >= 0) & (vv < Hd)
            dd = np.zeros(len(Q))
            dd[ok] = d[vv[ok], uu[ok]]
            bad |= ok & (dd > 0) & (z < dd - free_tol)
        return bad.mean()
    sub = P[:: max(1, len(P) // 3000)]
    best = None
    for th in np.radians(np.arange(0, 180, 2)):
        m = np.cos(th) * u + np.sin(th) * v
        proj = sub @ m
        for s0 in np.arange(np.percentile(proj, 10), np.percentile(proj, 90), 0.002):
            Q = sub - 2 * (sub @ m - s0)[:, None] * m
            ov = float((tree.query(Q, distance_upper_bound=0.003)[0] < 0.003).mean())
            if best is not None and ov < best[0] - 0.05:
                continue
            vio = free_violation(Q)
            sc = ov - lam * vio
            if best is None or sc > best[0]:
                best = (sc, th, s0, ov, vio)
    sc, th, s0, ov, vio = best
    m = np.cos(th) * u + np.sin(th) * v
    V = np.asarray(mesh.vertices)
    Vm = V - 2 * (V @ m - s0)[:, None] * m
    mm = trimesh.Trimesh(Vm, np.asarray(mesh.faces)[:, ::-1], process=False)
    if getattr(mesh.visual, "vertex_colors", None) is not None:
        mm.visual.vertex_colors = np.asarray(mesh.visual.vertex_colors)
    out = trimesh.util.concatenate([mesh, mm])
    return out, dict(plane_normal=m.tolist(), plane_offset=float(s0), score=float(sc), overlap=ov, free_violation=vio)


def extrude(mesh, n, dtab):
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    h = V @ n + dtab
    Vb = V - h[:, None] * n  # projection onto the table plane
    nv = len(V)
    Fb = F[:, ::-1] + nv
    # open boundary edges (used by exactly one face)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    eu, cnt = np.unique(e, axis=0, return_counts=True)
    be = eu[cnt == 1]
    Fs = np.concatenate([np.c_[be[:, 0], be[:, 1], be[:, 1] + nv], np.c_[be[:, 0], be[:, 1] + nv, be[:, 0] + nv]])
    m = trimesh.Trimesh(np.vstack([V, Vb]), np.vstack([F, Fb, Fs]), process=True)
    trimesh.repair.fix_normals(m)
    return m, dict(n_boundary_edges=int(len(be)), max_height=float(h.max()), min_height=float(h.min()))


def decimate(m, max_faces=4000):
    """Open3D quadric decimation to <= max_faces (keeps vertex colours) so the packer's budget_mesh(4096, 4096) is a
    no-op instead of a fast_simplification run that can stall on non-manifold extrusions."""
    import open3d as o3d
    m = trimesh.Trimesh(m.vertices, m.faces, vertex_colors=getattr(m.visual, "vertex_colors", None), process=True)
    m.remove_unreferenced_vertices()
    if len(m.faces) <= max_faces and len(m.vertices) <= max_faces:
        return m
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    vc = getattr(m.visual, "vertex_colors", None)
    if vc is not None and len(vc) == len(m.vertices):
        om.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vc)[:, :3] / 255.0)
    om = om.simplify_quadric_decimation(target_number_of_triangles=max_faces)
    om.remove_degenerate_triangles()
    om.remove_unreferenced_vertices()
    cols = (np.asarray(om.vertex_colors) * 255).astype(np.uint8) if om.has_vertex_colors() else None
    return trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), vertex_colors=cols, process=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--mode", choices=["revolve", "revolve_keep", "extrude", "mirror", "mirror_extrude", "none"], required=True)
    ap.add_argument("--in_root", default=f"{R0}/meshes/stage1")
    ap.add_argument("--out_root", default=f"{R0}/meshes/stage1c")
    ap.add_argument("--vo", default=None)
    ap.add_argument("--max_faces", type=int, default=4000)
    ap.add_argument("--extrude_max_h", type=float, default=0.05, help="extrude only if the mesh top is below this")
    a = ap.parse_args()
    e = f"episode_{a.episode:06d}"
    src = f"{a.in_root}/{a.split}/{e}/{a.obj}"
    s1 = json.load(open(src + ".json"))
    mesh = trimesh.load(src + ".ply", force="mesh")
    vo = a.vo or s1["poses"].split(":")[0]
    info = dict(mode=a.mode)
    if a.mode != "none":
        tb = table_in_body(a.split, e, s1.get("fused_frames", s1["frames"]), np.array(s1["frame_T_obj"]), vo)
        if tb is None:
            raise SystemExit("no table plane")
        n, dtab = tb
        hV = np.asarray(mesh.vertices) @ n + dtab
        info.update(table_normal_body=n.tolist(), table_d=dtab, mesh_height_range=[float(hV.min()), float(hV.max())])
        if a.mode in ("revolve", "revolve_keep"):
            body = np.asarray(mesh.vertices)
            if a.mode == "revolve_keep":  # profile from the body only: drop points far outside the robust radius
                u_, v_ = basis(n)
                xy = np.c_[body @ u_, body @ v_]
                c0, _ = fit_circle(xy)
                rr0 = np.linalg.norm(xy - c0, axis=1)
                body = body[rr0 < np.percentile(rr0, 60) * 1.25]
            out, i2 = revolve(body, n, dtab)
            if a.mode == "revolve_keep":
                extra, nf = keep_outside(mesh, i2, n, dtab)
                if extra is not None:
                    out = trimesh.util.concatenate([out, extra])
                i2["kept_outside_faces"] = nf
            i2.pop("profile_h"), i2.pop("profile_r")
        elif a.mode == "mirror":
            out, i2 = mirror(mesh, n, dtab, a.split, e, s1, vo)
        elif a.mode == "mirror_extrude":  # mirror about the vertical symmetry plane, then close the underside
            mm, i2 = mirror(mesh, n, dtab, a.split, e, s1, vo)
            if hV.max() <= a.extrude_max_h:
                out, i3 = extrude(mm, n, dtab)
                i2.update(i3)
            else:
                out = mm
        elif hV.max() <= a.extrude_max_h:
            out, i2 = extrude(mesh, n, dtab)
        else:  # not lying flat in the window (e.g. boards slotted upright) -> leave it
            out, i2 = mesh, dict(skipped=f"top {hV.max():.3f} m above the table > {a.extrude_max_h}")
        info.update(i2)
    else:
        out = mesh
    if a.mode != "none" and getattr(mesh.visual, "vertex_colors", None) is not None and len(mesh.vertices):
        from scipy.spatial import cKDTree  # colours of the nearest observed vertex (FoundationPose renders them)
        _, nn = cKDTree(np.asarray(mesh.vertices)).query(np.asarray(out.vertices))
        out.visual.vertex_colors = np.asarray(mesh.visual.vertex_colors)[nn]
    out = decimate(out, a.max_faces)
    # keep the bounding-box centre at the origin (FoundationPose centres on it; its mask-IoU render assumes it)
    c = (np.asarray(out.vertices).max(0) + np.asarray(out.vertices).min(0)) / 2
    out.vertices = np.asarray(out.vertices) - c
    Tc = np.eye(4)
    Tc[:3, 3] = c
    s1["frame_T_obj"] = (np.array(s1["frame_T_obj"]) @ Tc).tolist()
    info["recentred_by_m"] = c.tolist()
    od = f"{a.out_root}/{a.split}/{e}"
    os.makedirs(od, exist_ok=True)
    out.export(f"{od}/{a.obj}.ply")
    s1["completion"] = info
    json.dump(s1, open(f"{od}/{a.obj}.json", "w"), indent=1)
    print(a.obj, json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, (list, np.ndarray)) else v) for k, v in info.items()}),
          "extent cm", np.round(np.ptp(out.vertices, 0) * 100, 1).tolist())


if __name__ == "__main__":
    main()
