#!/usr/bin/env python3
"""Summarise end-to-end kit-scored runs (score_t1.py score --json) per split.

usage: summarize_e2e.py name=path.json [name=path.json ...] [--tune 0-12] [--test 13-24] [--md]
Prints mean metrics on the tuning split, the held-out split and all episodes, plus deltas vs the first run.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

METRICS = ("cd_h_cm", "cd_o_cm", "acc_h_cm", "acc_o_cm", "interpenetration_cm")
SHORT = ("CD-H", "CD-O", "ACC-H", "ACC-O", "PEN")


def rng(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def mean(per_ep, eps):
    eps = [e for e in eps if str(e) in per_ep]
    return {m: float(np.mean([per_ep[str(e)][m] for e in eps])) for m in METRICS}, len(eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tune", default="0-12")
    ap.add_argument("--test", default="13-24")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    runs = [(r.split("=", 1)[0], json.load(open(r.split("=", 1)[1]))["per_episode"]) for r in a.runs]
    splits = [("tune " + a.tune, rng(a.tune)), ("held-out " + a.test, rng(a.test)),
              ("all", sorted({int(e) for _, p in runs for e in p}))]
    for sname, eps in splits:
        base = None
        if a.md:
            print(f"\n**{sname}**\n\n| run | n | " + " | ".join(SHORT) + " |\n|---|---|" + "---|" * len(SHORT))
        else:
            print(f"== {sname}")
        for name, per in runs:
            m, n = mean(per, eps)
            base = base or m
            if a.md:
                print(f"| {name} | {n} | " + " | ".join(f"{m[k]:.4f}" for k in METRICS) + " |")
            else:
                d = " ".join(f"{s} {m[k]:8.4f} ({m[k] - base[k]:+.4f})" for s, k in zip(SHORT, METRICS))
                print(f"  {name:<22} n={n:2d}  {d}")


if __name__ == "__main__":
    main()
