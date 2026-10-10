"""MANO-joint hand trajectories for the ManoSharpaData 'loaded' stage (IK input).

Three sources, in order of preference:
  1. perception MANO joints + per-joint orientations (``hand_<side>_joints`` / ``_joints_wxyz``),
  2. perception 21 keypoints only -> per-joint orientations from a Kabsch fit of the template palm,
  3. DEV PLACEHOLDER (no hand estimate yet): a template hand that approaches each moving object,
     rides rigidly with it while it moves, and retreats to a parking pose otherwise.

The template is one open and one grasping MANO hand per side, taken from the released
ego_recon tissue-box reference (HaMeR MANO, Apache-2.0, robotic_grounding assets), expressed in the
MANO wrist frame (joint 0 pose = ``mano_<side>_joints_wxyz[t, 0]``; fingers ~ -x, palm ~ -y).

Contacts follow the v2d_task_library_loader layout (16 MANO links, world frame, inward object normals,
part ids 1..B, 0 = none) but use link-centre proximity instead of MANO skin vertices (no MANO model).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from .geom import frame_from_two_dirs, qnorm, rot_from_wxyz, wxyz_from_rot

SIDES = ("right", "left")
MANO_LINK_NAMES = [
    "link_palm", "link_thumb1", "link_thumb2", "link_thumb3", "link_index1", "link_index2", "link_index3",
    "link_middle1", "link_middle2", "link_middle3", "link_ring1", "link_ring2", "link_ring3",
    "link_pinky1", "link_pinky2", "link_pinky3",
]  # fmt: skip
MANO_HAND_LINKS = [
    [0, 1, 5, 9, 13, 17], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12],
    [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20],
]  # fmt: skip  (== robotic_grounding.retarget.params.MANO_HAND_LINKS)
TIP_JOINTS = [4, 8, 12, 16, 20]
PALM_JOINTS = [0, 5, 9, 13, 17]
KABSCH_JOINTS = [0, 1, 5, 9, 13, 17]
TEMPLATE_PATH = Path(__file__).with_name("hand_template.npz")
TISSUE_PARQUET = (
    "/mnt/secondary/v2d/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/"
    "human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave"
)
TEMPLATE_FRAMES = {"right": {"open": 0, "grasp": 110}, "left": {"open": 0, "grasp": 99}}


# ----------------------------------------------------------------------------- template
@dataclass
class HandTemplate:
    joints: dict  # side -> kind -> (21,3) wrist-local positions
    quats: dict  # side -> kind -> (21,4) wrist-local wxyz

    def palm_axes(self, side: str, kind: str = "grasp") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(finger dir, palm normal, palm centre) in the wrist frame."""
        J = self.joints[side][kind]
        f = J[9] / np.linalg.norm(J[9])
        n = np.cross(J[5], J[17]) if side == "right" else np.cross(J[17], J[5])
        return f, n / np.linalg.norm(n), J[PALM_JOINTS].mean(0)


def build_template(parquet: str = TISSUE_PARQUET, out: Path = TEMPLATE_PATH) -> Path:
    import pyarrow.dataset as ds

    t = ds.dataset(parquet, format="parquet").to_table()
    arrays = {}
    for side in SIDES:
        J_all = np.asarray(t[f"mano_{side}_joints"][0].as_py(), np.float64)
        Q_all = np.asarray(t[f"mano_{side}_joints_wxyz"][0].as_py(), np.float64)
        for kind, fr in TEMPLATE_FRAMES[side].items():
            r0 = rot_from_wxyz(Q_all[fr, 0])
            arrays[f"{side}_{kind}_joints"] = (J_all[fr] - J_all[fr, 0]) @ r0.as_matrix()
            arrays[f"{side}_{kind}_quats"] = wxyz_from_rot(r0.inv() * rot_from_wxyz(Q_all[fr]))
    np.savez(out, source=np.array(parquet), frames=np.array(str(TEMPLATE_FRAMES)), **arrays)
    return out


def load_template(path: Path = TEMPLATE_PATH) -> HandTemplate:
    if not Path(path).exists():
        build_template(out=Path(path))
    z = np.load(path)
    J = {s: {k: z[f"{s}_{k}_joints"] for k in ("open", "grasp")} for s in SIDES}
    Q = {s: {k: z[f"{s}_{k}_quats"] for k in ("open", "grasp")} for s in SIDES}
    return HandTemplate(J, Q)


def pose_hand(template: HandTemplate, side: str, wrist_pos, wrist_rot: R, blend: float) -> tuple[np.ndarray, np.ndarray]:
    """World joints (21,3) and joint quats (21,4) for a wrist pose and open(0)->grasp(1) blend."""
    Jo, Jg = template.joints[side]["open"], template.joints[side]["grasp"]
    Qo, Qg = template.quats[side]["open"], template.quats[side]["grasp"]
    Jl = (1.0 - blend) * Jo + blend * Jg
    if blend <= 0.0:
        Ql = rot_from_wxyz(Qo)
    elif blend >= 1.0:
        Ql = rot_from_wxyz(Qg)
    else:
        Ql = R.from_rotvec(
            np.stack([Slerp([0.0, 1.0], rot_from_wxyz(np.stack([Qo[k], Qg[k]])))([blend]).as_rotvec()[0] for k in range(21)])
        )
    joints = wrist_pos + Jl @ wrist_rot.as_matrix().T
    quats = wxyz_from_rot(wrist_rot * Ql)
    return joints, quats


# ----------------------------------------------------------------------------- keypoints -> orientations
def orientations_from_keypoints(joints: np.ndarray, template: HandTemplate, side: str) -> np.ndarray:
    """Per-joint wxyz (T,21,4) for keypoint-only hands: Kabsch-fit the template palm to each frame."""
    Jl = template.joints[side]["grasp"][KABSCH_JOINTS]
    Ql = rot_from_wxyz(template.quats[side]["grasp"])
    out = np.empty(joints.shape[:2] + (4,))
    for t in range(joints.shape[0]):
        X = joints[t, KABSCH_JOINTS] - joints[t, 0]
        H = Jl.T @ X
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        Rw = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        out[t] = wxyz_from_rot(R.from_matrix(Rw) * Ql)
    return out


# ----------------------------------------------------------------------------- placeholder
def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for t, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = t
        elif not m and start is not None:
            out.append((start, t - 1))
            start = None
    return out


def motion_segments(obj_pos, obj_quat, radius, fps, thresh=0.05, merge_gap_s=1.0, min_len_s=0.2, pre_s=0.6, post_s=0.4):
    """Per-body list of [start, end] frame segments where the object is being manipulated."""
    from scipy.ndimage import median_filter

    T, B = obj_pos.shape[:2]
    lin = np.linalg.norm(np.gradient(obj_pos, axis=0), axis=-1) * fps
    ang = np.zeros((T, B))
    for b in range(B):
        r = rot_from_wxyz(obj_quat[:, b])
        d = (r[:-1].inv() * r[1:]).magnitude() * fps
        ang[1:, b] += 0.5 * d
        ang[:-1, b] += 0.5 * d
    sig = median_filter(lin + ang * np.asarray(radius)[None, :], size=(5, 1), mode="nearest")
    segs = []
    for b in range(B):
        raw = _segments(sig[:, b] > thresh)
        merged = []
        for s, e in raw:
            if merged and s - merged[-1][1] <= merge_gap_s * fps:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        merged = [(s, e) for s, e in merged if e - s + 1 >= min_len_s * fps]
        dil = [(max(0, s - int(round(pre_s * fps))), min(T - 1, e + int(round(post_s * fps)))) for s, e in merged]
        out = []
        for s, e in dil:
            if out and s <= out[-1][1] + 1:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        segs.append(out)
    return segs, sig


def placeholder_hands(
    obj_pos: np.ndarray,
    obj_quat: np.ndarray,
    verts_body: list[np.ndarray],
    radius: np.ndarray,
    table_z: float,
    fps: float,
    template: HandTemplate,
    up=np.array([0.0, 0.0, 1.0]),
    transition_s: float = 0.5,
) -> dict:
    """DEV-ONLY synthetic two-hand MANO trajectory (see module doc). Returns per-side arrays + info."""
    T, B = obj_pos.shape[:2]
    up = np.asarray(up, float)
    segs, sig = motion_segments(obj_pos, obj_quat, radius, fps)
    tr = max(1, int(round(transition_s * fps)))
    # a retreat and the next approach must not overlap: merge segments closer than 2 transitions
    for b in range(B):
        merged = []
        for s, e in segs[b]:
            if merged and s - merged[-1][1] <= 2 * tr + 1:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        segs[b] = merged
    amount = np.array([sum(sig[s : e + 1, b].sum() for s, e in segs[b]) for b in range(B)])
    order = [int(b) for b in np.argsort(-amount) if segs[b]]
    assign = {"right": order[0] if len(order) > 0 else None, "left": order[1] if len(order) > 1 else None}
    lateral = {"right": np.array([0.0, -1.0, 0.0]), "left": np.array([0.0, 1.0, 0.0])}
    world_c0 = np.mean([verts_body[b].mean(0) @ rot_from_wxyz(obj_quat[0, b]).as_matrix().T + obj_pos[0, b] for b in range(B)], 0)
    out, info = {}, {"assignment": assign, "segments": {}}
    for side in SIDES:
        f_l, n_l, pc_l = template.palm_axes(side)
        local_frame = frame_from_two_dirs(f_l, n_l)
        lat = lateral[side]
        # parking pose: beside the objects, 25 cm above the table, palm down, fingers toward the objects
        park_p = world_c0 + 0.35 * lat
        park_p = park_p + (table_z + 0.25 - park_p @ up) * up
        park_R = R.from_matrix(frame_from_two_dirs(-lat, -up) @ local_frame.T)
        park_wrist = park_p - park_R.apply(pc_l)
        wrist_p = np.repeat(park_wrist[None], T, 0)
        wrist_r = [park_R] * T
        blend = np.zeros(T)
        contact_mask = np.zeros(T, bool)
        b = assign[side]
        seg_list = segs[b] if b is not None else []
        info["segments"][side] = [(int(s), int(e)) for s, e in seg_list]
        for s, e in seg_list:
            Rb0 = rot_from_wxyz(obj_quat[s, b])
            V = verts_body[b] @ Rb0.as_matrix().T + obj_pos[s, b]
            c = V.mean(0)
            a = up + 0.5 * lat
            a /= np.linalg.norm(a)
            reach = float(((V - c) @ a).max())
            palm_target = c + a * (reach + 0.02)
            # fingers along the object's dominant axis (projected orthogonal to the approach)
            ext = np.ptp(verts_body[b], axis=0)
            major = Rb0.apply(np.eye(3)[int(np.argmax(ext))])
            f_des = major - a * (major @ a)
            if np.linalg.norm(f_des) < 0.3:
                f_des = -lat - a * (-lat @ a)
            if f_des @ (-lat) < 0:
                f_des = -f_des
            grasp_R = R.from_matrix(frame_from_two_dirs(f_des, -a) @ local_frame.T)
            grasp_wrist = palm_target - grasp_R.apply(pc_l)
            # hand pose in the object frame, held fixed while attached
            rel_R = Rb0.inv() * grasp_R
            rel_p = Rb0.inv().apply(grasp_wrist - obj_pos[s, b])
            for t in range(s, e + 1):
                Rt = rot_from_wxyz(obj_quat[t, b])
                wrist_r[t] = Rt * rel_R
                wrist_p[t] = obj_pos[t, b] + Rt.apply(rel_p)
                blend[t] = 1.0
                contact_mask[t] = True
            # approach / retreat transitions (smoothstep), clipped to the episode
            for t0, t1, start_pose, end_pose in (
                (max(0, s - tr), s, (park_wrist, park_R), (wrist_p[s].copy(), wrist_r[s])),
                (e, min(T - 1, e + tr), (wrist_p[e].copy(), wrist_r[e]), (park_wrist, park_R)),
            ):
                if t1 - t0 < 1:
                    continue
                sl = Slerp([0.0, 1.0], R.concatenate([start_pose[1], end_pose[1]]))
                for t in range(t0, t1 + 1):
                    if (t == s and t0 < s) or (t == e and t1 > e):
                        continue  # keep the attached pose on the segment boundary
                    x = (t - t0) / (t1 - t0)
                    w = x * x * (3.0 - 2.0 * x)
                    wrist_p[t] = (1 - w) * start_pose[0] + w * end_pose[0]
                    wrist_r[t] = sl([w])[0]
                    blend[t] = w if t0 < s and t <= s else 1.0 - w
        joints = np.empty((T, 21, 3))
        quats = np.empty((T, 21, 4))
        for t in range(T):
            joints[t], quats[t] = pose_hand(template, side, wrist_p[t], wrist_r[t], float(np.clip(blend[t], 0, 1)))
        out[side] = {"joints": joints, "quats": qnorm(quats), "contact_mask": contact_mask, "object": b}
    out["info"] = info
    return out


# ----------------------------------------------------------------------------- contacts
def _closest_near(mesh, pts: np.ndarray, near: float, chunk: int = 2048):
    """Exact closest surface points for query points within ``near`` (+ longest edge) of a vertex.

    Far points get the nearest-vertex distance (an upper bound >= near, so they never count as contact)
    and zero normals. Keeps trimesh's candidate search bounded: querying points far from the mesh makes
    every triangle a candidate (several GB for a few thousand points).
    """
    import trimesh
    from scipy.spatial import cKDTree

    vtx = np.asarray(mesh.vertices)
    vd, vi = cKDTree(vtx).query(pts)
    edge = float(mesh.edges_unique_length.max()) if len(mesh.edges_unique) else 0.0
    cp = vtx[vi].copy()
    dist = vd.copy()
    nrm = np.zeros_like(pts)
    sel = np.flatnonzero(vd <= near + edge)
    for s in range(0, len(sel), chunk):
        idx = sel[s : s + chunk]
        c, d, tri = trimesh.proximity.closest_point(mesh, pts[idx])
        cp[idx], dist[idx], nrm[idx] = c, d, mesh.face_normals[tri]
    return cp, dist, nrm


def compute_contacts(
    joints: np.ndarray,
    obj_pos: np.ndarray,
    obj_quat: np.ndarray,
    meshes: list,
    allowed: np.ndarray | None = None,
    threshold: float = 0.02,
    finger_radius: float = 0.009,
) -> dict:
    """ManoSharpaData contact fields for one hand.

    joints (T,21,3) world; meshes = body-frame trimesh.Trimesh per body; allowed (T,B) bool mask of
    bodies that may be in contact (None = all). A MANO link is in contact with the nearest allowed body
    when its centre lies within ``threshold`` of that body's surface (link centres sit ~1 cm inside the
    skin, so 2 cm ~ the loader's 1 cm skin-vertex threshold).
    """
    import trimesh

    T = joints.shape[0]
    B = obj_pos.shape[1]
    K = len(MANO_HAND_LINKS)
    lc = np.stack([joints[:, idx].mean(1) for idx in MANO_HAND_LINKS], 1)  # (T,16,3)
    tips = joints[:, TIP_JOINTS]  # (T,5,3)
    best_d = np.full((T, K), np.inf)
    best_cp = np.zeros((T, K, 3))
    best_n = np.zeros((T, K, 3))
    best_b = np.full((T, K), -1)
    tip_d = np.full((T, 5), np.inf)
    for b in range(B):
        Rm = rot_from_wxyz(obj_quat[:, b]).as_matrix()  # (T,3,3)
        loc = np.einsum("tji,tkj->tki", Rm, lc - obj_pos[:, b, None]).reshape(-1, 3)  # R^T (x - p)
        cp, dist, nrm = _closest_near(meshes[b], loc, threshold)
        cp = cp.reshape(T, K, 3)
        dist = dist.reshape(T, K)
        nrm = -nrm.reshape(T, K, 3)  # inward (loader convention)
        if allowed is not None:
            dist = np.where(allowed[:, b, None], dist, np.inf)
        better = dist < best_d
        best_d = np.where(better, dist, best_d)
        best_cp = np.where(better[..., None], np.einsum("tij,tkj->tki", Rm, cp) + obj_pos[:, b, None], best_cp)
        best_n = np.where(better[..., None], np.einsum("tij,tkj->tki", Rm, nrm), best_n)
        best_b = np.where(better, b, best_b)
        tl = np.einsum("tji,tkj->tki", Rm, tips - obj_pos[:, b, None]).reshape(-1, 3)
        _, td, _ = _closest_near(meshes[b], tl, 0.05)
        tip_d = np.minimum(tip_d, td.reshape(T, 5))
    active = best_d < threshold
    part = np.where(active, best_b + 1, 0).astype(np.int32)
    obj_cp = np.where(active[..., None], best_cp, 0.0)
    obj_n = np.where(active[..., None], best_n, 0.0)
    to_obj = best_cp - lc
    dn = np.linalg.norm(to_obj, axis=-1, keepdims=True).clip(min=1e-9)
    link_n = np.where(active[..., None], to_obj / dn, 0.0)
    link_p = np.where(active[..., None], lc + to_obj * np.clip(1.0 - finger_radius / dn, 0.0, 1.0), 0.0)
    return {
        "object_contact_positions": obj_cp,
        "object_contact_normals": obj_n,
        "object_contact_part_ids": part,
        "link_contact_positions": link_p,
        "link_contact_normals": link_n,
        "tips_distance": np.where(np.isfinite(tip_d), tip_d, 1.0),
        "active_frames": int(active.any(1).sum()),
    }
