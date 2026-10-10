#!/usr/bin/env python3
"""Build a kit-format MHR body asset (the npz the Track 1 metric's `_load_body` reads) from Meta's
Apache-2.0 MHR v1.0.1 TorchScript model, so local scoring runs the kit's OWN `_MHRBody` forward.

The organizers omit their 603-vertex asset (and its vertex roles) from the participant kit. We rebuild
the same arrays from mhr_model.pt and choose roles as follows (APPROXIMATION of the hidden
track1_vertex_indices.json, documented):
  left_hand / right_hand : 256 + 256 vertices = deterministic farthest-point samples of the 2318
                           wrist/finger-dominant vertices per hand. This is exactly CARI4D's
                           "mhr-hand-surface-contact-v1" spec (reconstruction/modules/v2d_cari4d/lib/cari4d/
                           lib_mhr/hand_surface_contact.py), which is likely what the host used.
  body                   : 64 deterministic farthest-point samples of the remaining (non-hand) rest vertices.
  alignment              : the same FPS continued to 91 points (64 body + 27), so the unique vertex count is
                           91 + 512 = 603, matching the size of the hidden asset.

usage: mhr_asset.py --mhr mhr_model.pt --out mhr_asset_603.npz [--check-params mhr_params_mv.pt]
"""
import argparse, json
import numpy as np
import torch


def fps(points, n):
    """deterministic FPS identical to CARI4D hand_surface_contact.deterministic_farthest_point_indices"""
    points = np.asarray(points, np.float64)
    first = int(np.argmax(np.sum((points - points.mean(0)) ** 2, axis=1)))
    sel = np.empty(n, np.int64); sel[0] = first
    d2 = np.sum((points - points[first]) ** 2, axis=1)
    for i in range(1, n):
        sel[i] = int(np.argmax(d2)); d2 = np.minimum(d2, np.sum((points - points[sel[i]]) ** 2, axis=1))
    assert len(np.unique(sel)) == n
    return sel


def hand_vertices(m):
    """per hand (left, right): the 2318 vertices whose dominant skinning joint is the wrist or a finger joint
    (CARI4D hand_surface_contact._hand_joint_ids). `m` = loaded TorchScript MHR model."""
    names = [str(n) for n in m.get_joint_names()]
    lbs_idx, lbs_w = m.get_lbsw()
    lbs_idx = lbs_idx.to(torch.int64).numpy(); lbs_w = lbs_w.float().numpy()
    dom = lbs_idx[np.arange(len(lbs_idx)), np.argmax(lbs_w, 1)]
    out = []
    for p in ("l", "r"):
        ids = {i for i, n in enumerate(names) if n == f"{p}_wrist" or n.startswith(tuple(f"{p}_{f}" for f in ("thumb", "index", "middle", "ring", "pinky")))}
        assert len(ids) == 23, (p, len(ids))
        sel = np.flatnonzero(np.isin(dom, list(ids)))
        assert len(sel) == 2318, len(sel)
        out.append(sel)
    return out


def build(mhr_path, n_body=64, n_align_extra=27, n_hand=256):
    m = torch.jit.load(mhr_path, map_location="cpu").eval()
    ct = m.character_torch
    names = [str(n) for n in m.get_joint_names()]
    lbs_idx, lbs_w = m.get_lbsw()
    lbs_idx = lbs_idx.to(torch.int64).numpy(); lbs_w = lbs_w.float().numpy()
    dom = lbs_idx[np.arange(len(lbs_idx)), np.argmax(lbs_w, 1)]
    with torch.no_grad():
        rest = m(torch.zeros(1, 45), torch.zeros(1, 204), torch.zeros(1, 72), True)[0][0].double().numpy() / 100.0
    roles, hand_all = {}, []
    for side, key in (("left", "left_hand"), ("right", "right_hand")):
        p = "l" if side == "left" else "r"
        ids = {i for i, n in enumerate(names) if n == f"{p}_wrist" or n.startswith(tuple(f"{p}_{f}" for f in ("thumb", "index", "middle", "ring", "pinky")))}
        assert len(ids) == 23, (side, len(ids))
        sel = np.flatnonzero(np.isin(dom, list(ids)))
        assert len(sel) == 2318, len(sel)
        hand_all.append(sel)
        roles[key] = sel[fps(rest[sel], n_hand)]
    nonhand = np.setdiff1d(np.arange(len(rest)), np.concatenate(hand_all))
    order = nonhand[fps(rest[nonhand], n_body + n_align_extra)]
    roles["body"] = order[:n_body]
    roles["alignment"] = order
    keep = np.unique(np.concatenate(list(roles.values())))
    remap = -np.ones(len(rest), np.int64); remap[keep] = np.arange(len(keep))

    f32 = lambda t: t.detach().cpu().numpy().astype(np.float32)
    sk = ct.skeleton; lbs = ct.linear_blend_skinning; bs = ct.blend_shape
    pt = f32(ct.parameter_transform.parameter_transform)            # [889, 249]; cols 204: are identity -> fed zeros
    vi = lbs.vert_indices_flattened.numpy(); ji = lbs.skin_indices_flattened.numpy().astype(np.int64)
    wv = lbs.skin_weights_flattened.numpy().astype(np.float64)
    skin = np.zeros((len(rest), 127)); np.add.at(skin, (vi, ji), wv)
    pc = m.pose_correctives_model.pose_dirs_predictor
    sl, li = getattr(pc, "0"), getattr(pc, "2")
    w1_idx = sl.sparse_indices.numpy(); w1_val = f32(sl.sparse_weight)
    w2 = li.weight.detach().numpy()                                    # [V*3, 3000]
    rows = (keep[:, None] * 3 + np.arange(3)[None]).reshape(-1)
    w2k = w2[rows].astype(np.float32)
    import scipy.sparse as sp
    w2c = sp.csr_matrix(w2k)
    asset = {
        "joint_parents": sk.joint_parents.numpy().astype(np.int64),
        "joint_offsets": f32(sk.joint_translation_offsets),
        "joint_prerotations": f32(sk.joint_prerotations),
        "parameter_transform": pt[:, :204],
        "inverse_bind_pose": f32(lbs.inverse_bind_pose),
        "base_shape": f32(bs.base_shape)[keep],
        "shape_vectors": f32(bs.shape_vectors)[:, keep],
        "skin_dense": skin[keep].astype(np.float32),
        "corr_w1_idx": w1_idx, "corr_w1_val": w1_val, "corr_w1_shape": np.array([3000, 750]),
        "corr_w2_data": w2c.data, "corr_w2_indices": w2c.indices, "corr_w2_indptr": w2c.indptr,
        "corr_w2_shape": np.array(w2k.shape),
    }
    for k, v in roles.items():
        asset[f"role_{k}"] = remap[v]
    meta = {"n_keep": int(len(keep)), "roles": {k: int(len(v)) for k, v in roles.items()},
            "hand_vertices_all": [h.tolist() for h in hand_all]}
    return asset, keep, roles, meta


def check(asset_path, keep, mhr_path, params_path, kit):
    """kit _MHRBody forward on the built asset vs TorchScript forward, on GT params."""
    import sys, importlib.util
    spec = importlib.util.spec_from_file_location("cdh", f"{kit}/metric_code/track_1/CD-H.py")
    cdh = importlib.util.module_from_spec(spec); spec.loader.exec_module(cdh)
    body = cdh._load_body(open(asset_path, "rb").read())
    gp = torch.load(params_path, map_location="cpu", weights_only=True)
    mp = gp["mhr_model_params"].double().numpy()[::25]; sh = gp["shape_params"][0].double().numpy()
    v_kit, j_kit = body(mp, sh)
    m = torch.jit.load(mhr_path, map_location="cpu").eval()
    with torch.no_grad():
        v, s = m(torch.as_tensor(sh, dtype=torch.float32)[None].expand(len(mp), -1).contiguous(),
                 torch.as_tensor(mp, dtype=torch.float32), torch.zeros(len(mp), 72), True)
    flip = np.array([1., -1., -1.])
    v_ts = v.double().numpy()[:, keep] / 100 * flip; j_ts = s[..., :3].double().numpy() / 100 * flip
    return {"frames": int(len(mp)), "max_vertex_diff_mm": float(np.abs(v_kit - v_ts).max() * 1000),
            "max_joint_diff_mm": float(np.abs(j_kit - j_ts).max() * 1000)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhr", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--check-params"); ap.add_argument("--kit", default="/mnt/secondary/v2d/kit/v2d_submission_kit")
    a = ap.parse_args()
    torch.set_num_threads(4)
    asset, keep, roles, meta = build(a.mhr)
    np.savez_compressed(a.out, **asset)
    np.save(a.out.replace(".npz", "_keep.npy"), keep)
    json.dump(meta, open(a.out.replace(".npz", "_roles.json"), "w"))
    print(json.dumps({k: v for k, v in meta.items() if k != "hand_vertices_all"}))
    if a.check_params:
        print(json.dumps(check(a.out, keep, a.mhr, a.check_params, a.kit)))
