#!/usr/bin/env python3
"""Batch local Track 1 scorer on FORM-HOI pseudo-GT, through the kit's exact submission path.

  prediction NPZ (+ mesh)  --kit tools/pack_reconstruction.py track1-->  submission.parquet
  FORM-HOI GT NPZ (+ mesh) --same packer------------------------------->  solution.parquet (x,y,z -> ref_x,ref_y,ref_z)
  per-episode metrics = the kit metric modules' own _to_frames / _episode_metrics (CD-H.py) and
  _episode_penetration (PEN.py), exactly the loop their score() runs; --kit-score additionally calls each of
  the five modules' score() and asserts the means agree.

Body: kit `_MHRBody` on a 603-vertex asset rebuilt from Meta MHR v1.0.1 (mhr_asset.py). Vertex ROLES are an
approximation of the hidden track1_vertex_indices.json (hands = CARI4D hand-surface FPS spec), so absolute
CD-H/PEN differ slightly from Kaggle; use for relative comparisons.

subcommands
  build-val --selection selection.json [--root /mnt/secondary/v2d/t1/formhoi_val] [--out <root>/kitval]
      writes <out>/gt/episode_%06d.npz + episode_%06d_object.glb (symlink), sample.parquet, solution.parquet,
      val_index.json (episode -> sequence, camera, T, scored frames).
  score --pred <dir with episode_%06d.npz + episode_%06d_object.*> [--val <root>/kitval] [--episodes 0,3,..]
        [--kit-score] [--json out.json]

Python API (fast path, identical numbers, no parquet): FastScorer(val_dir).score_episode(ep, pred_dict, mesh_vf)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1lib as L  # noqa: E402

PER_FRAME_ROWS = {"mhr_pose": 46, "object_rotation": 3, "object_translation": 1}
PER_EPISODE_ROWS = {"mhr_scales": 23, "mhr_shape": 15, "object_scale": 1, "object_mesh_vertices": 4096, "object_mesh_faces": 4096}
METRICS = ("cd_h_cm", "cd_o_cm", "acc_h_cm", "acc_o_cm", "interpenetration_cm")
SHORT = {"cd_h_cm": "CD-H", "cd_o_cm": "CD-O", "acc_h_cm": "ACC-H", "acc_o_cm": "ACC-O", "interpenetration_cm": "PEN"}


def ms():
    from v2dlb import mhr_submission
    return mhr_submission


def build_sample(index: list[dict]) -> pd.DataFrame:
    order = ms().array_order()
    rows = []
    for e in index:
        ep = e["episode"]
        for f in e["frames"]:
            for name, n in PER_FRAME_ROWS.items():
                a = order.index(name)
                rows += [ms().row_id(1, ep, int(f), a, p) for p in range(n)]
        for name, n in PER_EPISODE_ROWS.items():
            a = order.index(name)
            rows += [ms().row_id(1, ep, L.EPISODE_FRAME, a, p) for p in range(n)]
    return pd.DataFrame({"row_id": rows})


def kit_pack(episodes_dir: Path, sample: Path, out: Path, log: Path | None = None):
    cmd = [sys.executable, "-I", str(L.KIT / "tools" / "pack_reconstruction.py"), "track1", "--episodes", str(episodes_dir),
           "--sample", str(sample), "--commit", L.FAKE_COMMIT, "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(L.KIT))
    if log:
        Path(log).write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"pack_reconstruction failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return out


def keys_from_row_id(df: pd.DataFrame) -> pd.DataFrame:
    order = ms().array_order()
    p = df["row_id"].str.split("_", expand=True)
    return pd.DataFrame({"row_id": df["row_id"], "episode_index": p[1].astype(int), "frame_index": p[2].astype(int),
                         "array_name": p[3].astype(int).map(lambda i: order[i]), "point_index": p[4].astype(int)})


def cmd_build_val(a):
    root = Path(a.root); out = Path(a.out or root / "kitval")
    (out / "gt").mkdir(parents=True, exist_ok=True)
    sel = json.load(open(a.selection))["sequences"]
    index = []
    for r in sel:
        seq_dir = root / r["sequence_id"]
        if not (seq_dir / "mhr_params_mv.pt").exists():
            print("skip (not fetched)", r["sequence_id"]); continue
        ep = int(r["val_index"])
        gt = L.load_formhoi_gt(seq_dir)
        T = len(gt["pose"])
        frames = L.formhoi_scored_frames(seq_dir, T)
        if len(frames) < L.MIN_STRETCH:
            print("skip (no scored stretch)", r["sequence_id"]); continue
        L.save_episode(out / "gt" / f"episode_{ep:06d}.npz", gt)
        link = out / "gt" / f"episode_{ep:06d}_object.glb"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(seq_dir / "object_mesh__output_aligned.glb")
        st = L.stretches(frames)
        index.append({"episode": ep, "sequence_id": r["sequence_id"], "object": r["object"], "camera": r["camera"], "T": T,
                      "n_scored": int(len(frames)), "first_scored": int(frames[0]), "n_stretches": len(st),
                      "frames": [int(f) for f in frames]})
        print(f"ep {ep:2d} {r['sequence_id']:70s} T={T} scored={len(frames)} stretches={len(st)} first={frames[0]}")
    json.dump(index, open(out / "val_index.json", "w"))
    build_sample(index).to_parquet(out / "sample.parquet", index=False)
    kit_pack(out / "gt", out / "sample.parquet", out / "gt_packed.parquet", out / "gt_pack.log")
    g = pd.read_parquet(out / "gt_packed.parquet")
    sol = keys_from_row_id(g).assign(ref_x=g["x"].to_numpy(), ref_y=g["y"].to_numpy(), ref_z=g["z"].to_numpy(), Usage="Public")
    sol.to_parquet(out / "solution.parquet", index=False)
    # cache budgeted GT meshes + reference arrays for the fast path
    print(f"wrote {out}: {len(index)} episodes, {len(sol):,} solution rows")


def per_episode_metrics(solution: pd.DataFrame, submission: pd.DataFrame, episodes=None) -> dict:
    cdh, pen = L.metric_module("CD-H"), L.metric_module("PEN")
    sol = solution.drop(columns=["Usage"], errors="ignore").copy()
    sub = submission.copy()
    sol["row_id"] = sol["row_id"].astype(str); sub["row_id"] = sub["row_id"].astype(str)
    merged = sol.merge(sub, on="row_id", how="left", validate="one_to_one")
    if merged[["x", "y", "z"]].isna().any().any():
        raise ValueError("submission is missing rows")
    out = {}
    for episode, block in merged.groupby("episode_index", sort=True):
        if episodes is not None and int(episode) not in episodes:
            continue
        block = block.sort_values(["array_name", "frame_index", "point_index"])
        pred, frames, scene = cdh._to_frames(block, ["x", "y", "z"], int(episode), cdh._BODY)
        ref, _, _ = cdh._to_frames(block, ["ref_x", "ref_y", "ref_z"], int(episode), cdh._BODY)
        m = cdh._episode_metrics(pred, ref, frames, None)
        m["interpenetration_cm"] = pen._episode_penetration(pred, scene, ref)
        out[int(episode)] = {k: float(m[k]) for k in METRICS}
    return out


def kit_full_score(solution: pd.DataFrame, submission: pd.DataFrame) -> dict:
    res = {}
    for name, key in (("CD-H", "cd_h_cm"), ("CD-O", "cd_o_cm"), ("ACC-H", "acc_h_cm"), ("ACC-O", "acc_o_cm"), ("PEN", "interpenetration_cm")):
        res[key] = float(L.metric_module(name).score(solution.copy(), submission.copy(), "row_id"))
    return res


def summarize(per_ep: dict) -> dict:
    return {k: float(np.mean([v[k] for v in per_ep.values()])) for k in METRICS}


def cmd_score(a):
    val = Path(a.val)
    pred = Path(a.pred)
    index = json.load(open(val / "val_index.json"))
    eps = None if not a.episodes else {int(x) for x in a.episodes.split(",")}
    sample = val / "sample.parquet"
    sol = pd.read_parquet(val / "solution.parquet")
    if eps is not None:
        keep = sol["episode_index"].isin(eps)
        sol = sol[keep].reset_index(drop=True)
        sample = pred / "_sample_subset.parquet"
        sol[["row_id"]].to_parquet(sample, index=False)
    t0 = time.time()
    sub_path = pred / "submission_local.parquet"
    kit_pack(pred, sample, sub_path, pred / "pack.log")
    sub = pd.read_parquet(sub_path)
    per = per_episode_metrics(sol, sub, eps)
    res = {"mean": summarize(per), "per_episode": per, "n_episodes": len(per), "seconds": round(time.time() - t0, 1)}
    if a.kit_score:
        full = kit_full_score(sol, sub)
        res["kit_score"] = full
        res["kit_score_matches"] = bool(all(abs(full[k] - res["mean"][k]) <= 1e-9 * max(1.0, abs(full[k])) for k in METRICS))
    seq = {e["episode"]: e["sequence_id"] for e in index}
    for ep, m in per.items():
        print(f"ep {ep:2d} {seq[ep][:60]:60s} " + " ".join(f"{SHORT[k]} {m[k]:.4f}" for k in METRICS))
    print("MEAN " + " ".join(f"{SHORT[k]} {res['mean'][k]:.5f}" for k in METRICS))
    if a.kit_score:
        print("KIT score() " + " ".join(f"{SHORT[k]} {res['kit_score'][k]:.5f}" for k in METRICS), "matches:", res["kit_score_matches"])
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


# ------------------------------------------------------------------------------------- fast path
class FastScorer:
    """Same arrays as the kit's _to_frames, built directly from NPZ dicts (no parquet round trip).
    Verified identical to the parquet path by `selftest` (max |diff| ~1e-12)."""

    def __init__(self, val_dir=None):
        self.val = Path(val_dir or L.VAL_ROOT / "kitval")
        self.index = {e["episode"]: e for e in json.load(open(self.val / "val_index.json"))}
        self.cdh, self.pen = L.metric_module("CD-H"), L.metric_module("PEN")
        self.body = self.cdh._BODY
        self._ref = {}

    @staticmethod
    def budget(mesh_path):
        from v2dlb.mesh_budget import budget_mesh
        return budget_mesh(mesh_path, 4096, 4096)

    def arrays(self, ep, d, mesh_vf):
        frames = np.asarray(self.index[ep]["frames"])
        cdh = self.cdh
        pose = np.asarray(d["pose"], np.float64)[frames]
        scales = np.asarray(d["scales"], np.float64).reshape(68)
        params = np.concatenate([pose, np.broadcast_to(scales, (len(pose), 68))], 1)
        V, J = self.body(params, np.asarray(d["shape"], np.float64).reshape(45))
        mv, mf = mesh_vf
        R = cdh._rotations(np.asarray(d["object_rotation"], np.float64)[frames])
        t = np.asarray(d["object_translation"], np.float64)[frames]
        s = float(np.asarray(d["object_scale"]).reshape(()))
        local = cdh._sample_mesh_surface(mv, mf, len(self.body.roles["body"]), int(ep))
        arrays = {"mhr_vertices": V[:, self.body.roles["alignment"]],
                  "mhr_joints": J[:, list(cdh.MHR_TABLE3_BODY_JOINT_INDICES)],
                  "human_surface_points": V[:, self.body.roles["body"]],
                  "object_surface_points": s * np.einsum("tij,pj->tpi", R, local) + t[:, None],
                  "object_translation": t}
        scene = {"hands": V[:, np.concatenate([self.body.roles["left_hand"], self.body.roles["right_hand"]])],
                 "mesh_vertices": mv, "mesh_faces": mf, "rotation": R, "translation": t, "scale": s}
        return arrays, frames, scene

    def ref(self, ep):
        if ep not in self._ref:
            d = L.load_episode(self.val / "gt" / f"episode_{ep:06d}.npz")
            vf = self.budget(self.val / "gt" / f"episode_{ep:06d}_object.glb")
            self._ref[ep] = (d, vf, self.arrays(ep, d, vf)[0])
        return self._ref[ep]

    def score_episode(self, ep, d, mesh_vf, pen=True):
        _, _, ref = self.ref(ep)
        pred, frames, scene = self.arrays(ep, d, mesh_vf)
        m = self.cdh._episode_metrics(pred, ref, frames, None)
        out = {k: float(m[k]) for k in METRICS if k in m}
        if pen:
            out["interpenetration_cm"] = float(self.pen._episode_penetration(pred, scene, ref))
        return out


def cmd_selftest(a):
    """GT vs GT through the full kit path (expect 0 on all 5), and fast path == parquet path on a perturbed pred."""
    val = Path(a.val)
    fs = FastScorer(val)
    eps = sorted(fs.index)[: a.n]
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for ep in eps:
        d = L.load_episode(val / "gt" / f"episode_{ep:06d}.npz")
        d["pose"] = d["pose"] + rng.normal(0, 0.01, d["pose"].shape)
        d["object_translation"] = d["object_translation"] + rng.normal(0, 0.01, d["object_translation"].shape)
        L.save_episode(tmp / f"episode_{ep:06d}.npz", d)
        link = tmp / f"episode_{ep:06d}_object.glb"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.realpath(val / "gt" / f"episode_{ep:06d}_object.glb"))
    sol = pd.read_parquet(val / "solution.parquet")
    sol = sol[sol.episode_index.isin(eps)].reset_index(drop=True)
    sol[["row_id"]].to_parquet(tmp / "_sample.parquet", index=False)
    kit_pack(tmp, tmp / "_sample.parquet", tmp / "sub.parquet")
    sub = pd.read_parquet(tmp / "sub.parquet")
    per = per_episode_metrics(sol, sub)
    gtsub = sol[["row_id"]].assign(x=sol.ref_x, y=sol.ref_y, z=sol.ref_z, code_commit_url=L.FAKE_COMMIT)
    gt_full = kit_full_score(sol, gtsub)
    print("GT vs GT, kit score() on", len(eps), "episodes:", json.dumps(gt_full))
    worst = 0.0
    for ep in eps:
        d = L.load_episode(tmp / f"episode_{ep:06d}.npz")
        f = fs.score_episode(ep, d, fs.budget(tmp / f"episode_{ep:06d}_object.glb"))
        diff = max(abs(f[k] - per[ep][k]) for k in METRICS)
        worst = max(worst, diff)
        print(f"ep {ep}: parquet {json.dumps({SHORT[k]: round(per[ep][k], 6) for k in METRICS})}  fast-vs-parquet max|diff| {diff:.2e}")
    # PEN is not a comparison metric: GT-vs-GT gives the pseudo-GT's OWN hand/object penetration (> 0).
    ok = worst < 1e-8 and all(abs(gt_full[k]) < 1e-9 for k in METRICS if k != "interpenetration_cm")
    print("SELFTEST", "PASS" if ok else "FAIL", f"(CD-H/CD-O/ACC-H/ACC-O == 0; GT own PEN {gt_full['interpenetration_cm']:.4f} cm;"
          f" fast-vs-parquet worst diff {worst:.2e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-val"); b.add_argument("--selection", required=True); b.add_argument("--root", default=str(L.VAL_ROOT)); b.add_argument("--out")
    s = sub.add_parser("score"); s.add_argument("--pred", required=True); s.add_argument("--val", default=str(L.VAL_ROOT / "kitval"))
    s.add_argument("--episodes"); s.add_argument("--kit-score", action="store_true"); s.add_argument("--json")
    t = sub.add_parser("selftest"); t.add_argument("--val", default=str(L.VAL_ROOT / "kitval")); t.add_argument("--n", type=int, default=3)
    t.add_argument("--tmp", default="/mnt/secondary/v2d/t1/work/selftest")
    a = ap.parse_args()
    {"build-val": cmd_build_val, "score": cmd_score, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
