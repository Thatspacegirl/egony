#!/usr/bin/env python3
"""Write a synthetic prediction directory (episode_%06d.npz + episode_%06d_object.glb) = FORM-HOI GT corrupted by
noise_model.PRESETS[preset], for end-to-end tests of postprocess.py + score_t1.py. The object mesh is the GT mesh
(symlink): this tests post-processing and scoring, not mesh reconstruction.

usage: make_noisy_preds.py --preset cari4d_like --out /mnt/secondary/v2d/t1/work/preds_cari4d_like
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1lib as L  # noqa: E402
import noise_model as N  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="cari4d_like")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", default=str(L.VAL_ROOT / "kitval"))
    ap.add_argument("--seed-base", type=int, default=1000)
    a = ap.parse_args()
    val, out = Path(a.val), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = {r["val_index"]: r for r in json.load(open(L.VAL_ROOT / "selection.json"))["sequences"]}
    for e in json.load(open(val / "val_index.json")):
        ep = e["episode"]
        gt = L.load_episode(val / "gt" / f"episode_{ep:06d}.npz")
        nz = N.corrupt(gt, L.VAL_ROOT / sel[ep]["sequence_id"], sel[ep]["camera"], N.PRESETS[a.preset], seed=a.seed_base + ep)
        L.save_episode(out / f"episode_{ep:06d}.npz", nz)
        link = out / f"episode_{ep:06d}_object.glb"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.realpath(val / "gt" / f"episode_{ep:06d}_object.glb"))
    print("wrote", out)


if __name__ == "__main__":
    main()
