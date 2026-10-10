#!/usr/bin/env python3
"""Evaluate pen_refine.py on FORM-HOI val: clean GT and (noisy GT -> tuned smoother), before/after.

Reports per condition: kit PEN (local roles), PEN on the hand-point SUPERSET (all 2x2318 hand vertices),
CD-O / ACC-O / CD-H / ACC-H change, contact fraction (closest hand vertex within 1.5 cm) before/after,
object offset size and runtime.

usage: eval_pen.py --smooth-json best_cfg.json --episodes 13-24 --out /mnt/secondary/v2d/t1/work/eval_pen.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1lib as L  # noqa: E402
import score_t1 as S  # noqa: E402
import noise_model as N  # noqa: E402
import smoothing as SM  # noqa: E402
import pen_refine as P  # noqa: E402


def parse_eps(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def superset_pen(full_hands, ep, vf, frames):
    mq = P.MeshQuery(*vf)
    H = full_hands(ep["pose"], ep["scales"], ep["shape"])
    return P.penetration_cm(H, ep["object_rotation"], ep["object_translation"], float(ep["object_scale"]), mq, frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth-json", required=True)
    ap.add_argument("--preset", default="cari4d_like")
    ap.add_argument("--episodes", default="13-24")
    ap.add_argument("--margin", type=float, default=0.002)
    ap.add_argument("--n-per-hand", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = SM.SmoothCfg(**json.load(open(a.smooth_json)))
    fs = S.FastScorer()
    sel = {r["val_index"]: r for r in json.load(open(L.VAL_ROOT / "selection.json"))["sequences"]}
    hands = P.HandPoints(L.MHR_TS, n_per_hand=a.n_per_hand)
    full = P.HandPoints(L.MHR_TS, n_per_hand=2318)
    res = {"cfg": {"margin": a.margin, "n_per_hand": a.n_per_hand, "smooth": SM.cfg_dict(cfg), "preset": a.preset}, "episodes": {}}
    for ep_i in parse_eps(a.episodes):
        gt, vf, _ = fs.ref(ep_i)
        frames = np.asarray(fs.index[ep_i]["frames"])
        noisy = N.corrupt(gt, L.VAL_ROOT / sel[ep_i]["sequence_id"], sel[ep_i]["camera"], N.PRESETS[a.preset], seed=1000 + ep_i)
        cond = {"clean_gt": gt, f"{a.preset}+smooth": SM.smooth_episode(noisy, int(frames[0]), cfg)}
        res["episodes"][ep_i] = {}
        for name, x in cond.items():
            t0 = time.time()
            before = fs.score_episode(ep_i, x, vf, pen=True)
            sup_b = float("nan")                      # superset only checked after refinement (cost)
            y, rep = P.refine(x, vf, frames, hands, margin=a.margin)
            after = fs.score_episode(ep_i, y, vf, pen=True)
            sup_a = superset_pen(full, y, vf, frames)
            r = {"before": before, "after": after, "superset_pen_before": sup_b, "superset_pen_after": sup_a,
                 "refine": rep, "seconds": round(time.time() - t0, 1)}
            res["episodes"][ep_i][name] = r
            print(f"ep {ep_i:2d} {name:24s} PEN {before['interpenetration_cm']:.4f}->{after['interpenetration_cm']:.5f} "
                  f"superset {sup_b:.4f}->{sup_a:.5f} CD-O {before['cd_o_cm']:.3f}->{after['cd_o_cm']:.3f} "
                  f"ACC-O {before['acc_o_cm']:.4f}->{after['acc_o_cm']:.4f} CD-H {before['cd_h_cm']:.3f}->{after['cd_h_cm']:.3f} "
                  f"contact {rep['contact_before']:.3f}->{rep['contact_after']:.3f} maxoff {rep['max_offset_cm']:.2f}cm "
                  f"still {rep['still_inside_frames']} {r['seconds']}s", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
    # summary
    summ = {}
    for name in next(iter(res["episodes"].values())):
        rows = [v[name] for v in res["episodes"].values()]
        summ[name] = {f"{k}_{w}": float(np.mean([r[w][k] for r in rows])) for k in S.METRICS for w in ("before", "after")}
        summ[name]["superset_pen_after"] = float(np.mean([r["superset_pen_after"] for r in rows]))
        summ[name]["contact_before"] = float(np.mean([r["refine"]["contact_before"] for r in rows]))
        summ[name]["contact_after"] = float(np.mean([r["refine"]["contact_after"] for r in rows]))
        summ[name]["mean_offset_cm"] = float(np.mean([r["refine"]["mean_offset_cm"] for r in rows]))
        summ[name]["still_inside_frames"] = int(sum(r["refine"]["still_inside_frames"] for r in rows))
    res["summary"] = summ
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
