#!/usr/bin/env python3
"""Tune the SO(3)/frame-0-protected Whittaker smoother on FORM-HOI GT + synthetic monocular-like noise.

Split: episodes 0-12 tune, 13-24 held-out report. Objective (rank-oriented; the public leaderboard gaps are
~0.01-0.03 cm on ACC and ~0.25 cm on CD-H):
    J_H = ACC-H / 0.01 + CD-H / 0.25          (human components)
    J_O = ACC-O / 0.01 + CD-O / 0.5           (object components)
Stages: joint human lambda sweep x anchor mode -> coordinate descent on (root rot, root trans, body) ->
robust on/off -> object (rot, trans) coordinate descent. Also reports the naive Euler-angle Whittaker
(the failure mode) and the bias cost on clean GT.

usage: tune_smoother.py --presets cari4d_like,raw_sam3d_like --out /mnt/secondary/v2d/t1/work/tune_smoother.json
       tune_smoother.py --presets cari4d_real --pred-dir <dir of real val predictions> --out ...   (real model output)
       tune_smoother.py --presets cari4d_like --eval-cfg smooth_cari4d_like.json --out ...      (held-out report only)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1lib as L  # noqa: E402
import score_t1 as S  # noqa: E402
import noise_model as N  # noqa: E402
import smoothing as SM  # noqa: E402

GRID = [0, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5]
TUNE = list(range(0, 13))
TEST = list(range(13, 25))
KEYS = ("cd_h_cm", "cd_o_cm", "acc_h_cm", "acc_o_cm")


class Bench:
    """preset = noise_model preset applied to FORM-HOI GT, or (pred_dir given) REAL predictions: a directory of
    episode_%06d.npz + episode_%06d_object.<ext> for the val episodes (e.g. CARI4D run on the FORM-HOI val videos
    and converted with cari4d_to_t1.py); then each prediction is scored with its OWN mesh, as the scorer does."""

    def __init__(self, preset, seed_base=1000, pred_dir=None):
        self.fs = S.FastScorer()
        self.sel = {r["val_index"]: r for r in json.load(open(L.VAL_ROOT / "selection.json"))["sequences"]}
        self.noisy, self.vf = {}, {}
        for ep in sorted(self.fs.index):
            d, vf, _ = self.fs.ref(ep)
            if pred_dir:
                pd_ = Path(pred_dir)
                src = pd_ / f"episode_{ep:06d}.npz"
                if not src.exists():
                    continue
                self.noisy[ep] = L.load_episode(src)
                mesh = next(pd_ / f"episode_{ep:06d}_object{x}" for x in (".glb", ".obj", ".ply", ".stl", ".off")
                            if (pd_ / f"episode_{ep:06d}_object{x}").exists())
                self.vf[ep] = self.fs.budget(mesh)
            else:
                self.noisy[ep] = N.corrupt(d, L.VAL_ROOT / self.sel[ep]["sequence_id"], self.sel[ep]["camera"], N.PRESETS[preset], seed=seed_base + ep)
                self.vf[ep] = vf

    def run(self, cfg: SM.SmoothCfg | None, eps, naive_euler=False):
        ms = []
        for ep in eps:
            if ep not in self.noisy:
                continue
            vf = self.vf[ep]
            x = self.noisy[ep]
            if cfg is not None:
                x = naive(x, cfg) if naive_euler else SM.smooth_episode(x, self.fs.index[ep]["first_scored"], cfg)
            ms.append(self.fs.score_episode(ep, x, vf, pen=False))
        return {k: float(np.mean([m[k] for m in ms])) for k in KEYS}


def naive(ep, cfg):
    """the failure mode: Whittaker directly on Euler angles (root + object as Euler)."""
    from scipy.spatial.transform import Rotation
    out = {k: np.array(v, copy=True) for k, v in ep.items()}
    p = out["pose"]
    p[:, 3:6] = SM.whittaker(p[:, 3:6], cfg.lam_root_rot)
    p[:, 0:3] = SM.whittaker(p[:, 0:3], cfg.lam_root_trans)
    p[:, SM.BODY_COLS] = SM.whittaker(p[:, SM.BODY_COLS], cfg.lam_body)
    eo = Rotation.from_matrix(out["object_rotation"]).as_euler("xyz")
    out["object_rotation"] = Rotation.from_euler("xyz", SM.whittaker(eo, cfg.lam_obj_rot)).as_matrix()
    out["object_translation"] = SM.whittaker(out["object_translation"], cfg.lam_obj_trans)
    return out


def JH(m):
    return m["acc_h_cm"] / 0.01 + m["cd_h_cm"] / 0.25


def JO(m):
    return m["acc_o_cm"] / 0.01 + m["cd_o_cm"] / 0.5


def fmt(m):
    return " ".join(f"{k.split('_')[0].upper()}-{k.split('_')[1].upper()} {m[k]:.4f}" for k in KEYS)


def tune(preset, log, pred_dir=None):
    t0 = time.time()
    B = Bench(preset, pred_dir=pred_dir)
    res = {"preset": preset, "noise": asdict(N.PRESETS[preset]) if not pred_dir else {"pred_dir": str(pred_dir)}}
    res["raw_tune"] = B.run(None, TUNE)
    res["raw_test"] = B.run(None, TEST)
    log(f"[{preset}] unsmoothed tune: {fmt(res['raw_tune'])}")
    base = SM.SmoothCfg(lam_obj_rot=1e3, lam_obj_trans=1e3, lam_hand=1e2, robust=None)
    # 1) joint human sweep x anchor
    sweep = []
    for anchor in ("none", "light", "raw"):
        for lam in GRID[1:]:
            cfg = replace(base, lam_root_rot=lam, lam_root_trans=lam, lam_body=min(lam, 1e3), anchor=anchor)
            m = B.run(cfg, TUNE)
            sweep.append({"anchor": anchor, "lam": lam, **m, "JH": JH(m)})
            log(f"  joint lam={lam:g} body={min(lam, 1e3):g} anchor={anchor:5s} {fmt(m)} JH={JH(m):.2f}")
    res["joint_sweep"] = sweep
    best = min(sweep, key=lambda r: r["JH"])
    cfg = replace(base, lam_root_rot=best["lam"], lam_root_trans=best["lam"], lam_body=min(best["lam"], 1e3), anchor=best["anchor"])
    # 2) coordinate descent on human components
    cd_log = []
    for _pass in range(2):
        for field in ("lam_root_rot", "lam_root_trans", "lam_body"):
            scores = []
            for lam in GRID:
                c = replace(cfg, **{field: lam})
                m = B.run(c, TUNE)
                scores.append((JH(m), lam, m))
                cd_log.append({"pass": _pass, "field": field, "lam": lam, **m, "JH": JH(m)})
            j, lam, m = min(scores, key=lambda s: s[0])
            cfg = replace(cfg, **{field: lam})
            log(f"  CD pass{_pass} {field} -> {lam:g}  {fmt(m)} JH={j:.2f}")
    res["coord_descent_human"] = cd_log
    # light-anchor strength
    for la in (3.0, 10.0, 30.0, 100.0):
        if cfg.anchor != "light":
            break
        m = B.run(replace(cfg, lam_anchor=la), TUNE)
        log(f"  lam_anchor={la:g} {fmt(m)} JH={JH(m):.2f}")
        res.setdefault("anchor_strength", []).append({"lam_anchor": la, **m, "JH": JH(m)})
    if "anchor_strength" in res:
        cfg = replace(cfg, lam_anchor=min(res["anchor_strength"], key=lambda r: r["JH"])["lam_anchor"])
    # robust
    mr = B.run(replace(cfg, robust="huber"), TUNE)
    m0 = B.run(cfg, TUNE)
    log(f"  robust huber: {fmt(mr)} JH={JH(mr):.2f} vs {JH(m0):.2f}")
    if JH(mr) < JH(m0):
        cfg = replace(cfg, robust="huber")
    # 3) object
    for _pass in range(2):
        for field in ("lam_obj_trans", "lam_obj_rot"):
            scores = []
            for lam in GRID:
                m = B.run(replace(cfg, **{field: lam}), TUNE)
                scores.append((JO(m), lam, m))
            j, lam, m = min(scores, key=lambda s: s[0])
            cfg = replace(cfg, **{field: lam})
            log(f"  CD pass{_pass} {field} -> {lam:g}  {fmt(m)} JO={j:.2f}")
    return final_eval(preset, cfg, B, res, log)


def final_eval(preset, cfg, B, res, log):
    """held-out (episodes 13-24) report for a fixed cfg; anchor variants are REPORTED only, never selected here."""
    t0 = time.time()
    res["best_cfg"] = asdict(cfg)
    res["best_tune"] = B.run(cfg, TUNE)
    res["best_test"] = B.run(cfg, TEST)
    res["best_test_no_anchor"] = B.run(replace(cfg, anchor="none"), TEST)
    res["naive_euler_test"] = B.run(cfg, TEST, naive_euler=True)
    res["anchor_variants_test"] = {}
    for anc, la in (("none", 10.0), ("light", 30.0), ("light", 100.0), ("raw", 10.0)):
        m = B.run(replace(cfg, anchor=anc, lam_anchor=la), TEST)
        res["anchor_variants_test"][f"{anc}{la:g}" if anc == "light" else anc] = m
        log(f"[{preset}] held-out anchor={anc} lam_anchor={la:g}: {fmt(m)}")
    res["no_robust_test"] = B.run(replace(cfg, robust=None), TEST)
    log(f"[{preset}] held-out no robust : {fmt(res['no_robust_test'])}")
    log(f"[{preset}] BEST {cfg.tag()}")
    log(f"[{preset}] held-out unsmoothed : {fmt(res['raw_test'])}")
    log(f"[{preset}] held-out smoothed   : {fmt(res['best_test'])}")
    log(f"[{preset}] held-out no anchor  : {fmt(res['best_test_no_anchor'])}")
    log(f"[{preset}] held-out naive Euler: {fmt(res['naive_euler_test'])}")
    # bias on clean GT with the same cfg
    C = Bench("clean")
    res["clean_test"] = C.run(cfg, TEST)
    res["clean_test_no_anchor"] = C.run(replace(cfg, anchor="none"), TEST)
    log(f"[{preset}] clean GT smoothed (bias cost): {fmt(res['clean_test'])}; no anchor: {fmt(res['clean_test_no_anchor'])}")
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="cari4d_like,raw_sam3d_like")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-cfg", help="skip tuning; report a fixed SmoothCfg JSON")
    ap.add_argument("--pred-dir", help="tune on REAL predictions for the val episodes instead of synthetic noise "
                                       "(--presets is then just a label; use e.g. --presets cari4d_like)")
    a = ap.parse_args()
    out = {}

    def log(s):
        print(s, flush=True)

    for p in a.presets.split(","):
        if a.eval_cfg:
            B = Bench(p, pred_dir=a.pred_dir)
            res = {"preset": p, "noise": asdict(N.PRESETS[p]) if not a.pred_dir else {"pred_dir": a.pred_dir},
                   "raw_tune": B.run(None, TUNE), "raw_test": B.run(None, TEST)}
            out[p] = final_eval(p, SM.SmoothCfg(**json.load(open(a.eval_cfg))), B, res, log)
        else:
            out[p] = tune(p, log, pred_dir=a.pred_dir)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
