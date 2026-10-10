#!/usr/bin/env python3
"""Pack 30 Track 1 episode NPZs (+ meshes) into the Kaggle parquet with the KIT's own packer, then validate it
against the competition sample submission (local only; nothing is uploaded).

  t1_pack.py --episodes /mnt/secondary/v2d/t1/predictions/<run> --out /mnt/secondary/v2d/t1/predictions/<run>.parquet

Validation: same row ids as data/track_1_sample_submission.parquet (set and count), finite x/y/z, one commit URL,
and the kit metric modules can decode every episode: scoring the submission against ITSELF must give
CD-H = CD-O = ACC-H = ACC-O = 0 (PEN = our own penetration, reported).  The commit URL is a placeholder built from
the local repo HEAD (https://github.com/local/egony-v2d/commit/<sha>); a real upload needs the public repo's URL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1lib as L  # noqa: E402
import score_t1 as S  # noqa: E402

SAMPLE = L.KIT / "data" / "track_1_sample_submission.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", default=None)
    a = ap.parse_args()
    commit = a.commit
    if commit is None:
        sha = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        commit = f"https://github.com/local/egony-v2d/commit/{sha}"
    out = Path(a.out)
    tmp = out.with_name(out.stem + ".tmp.parquet")
    cmd = [sys.executable, "-I", str(L.KIT / "tools" / "pack_reconstruction.py"), "track1", "--episodes", a.episodes,
           "--sample", str(SAMPLE), "--commit", commit, "--out", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(L.KIT))
    print(r.stdout[-3000:], r.stderr[-3000:])
    if r.returncode != 0:
        raise SystemExit("packer failed")
    sub = pd.read_parquet(tmp)
    smp = pd.read_parquet(SAMPLE, columns=["row_id"])
    checks = {"rows": len(sub), "sample_rows": len(smp),
              "same_row_ids": bool(set(sub.row_id.astype(str)) == set(smp.row_id.astype(str))),
              "finite": bool(np.isfinite(sub[["x", "y", "z"]].to_numpy()).all()),
              "one_commit_url": bool(sub.code_commit_url.nunique() == 1), "commit_url": commit}
    # self-score through the kit metric code: decodes every episode exactly as the scorer does
    sol = S.keys_from_row_id(sub[["row_id"]].astype({"row_id": str})).assign(
        ref_x=sub.x.to_numpy(), ref_y=sub.y.to_numpy(), ref_z=sub.z.to_numpy(), Usage="Public")
    per = S.per_episode_metrics(sol, sub)
    mean = S.summarize(per)
    checks["self_score"] = mean
    checks["self_score_ok"] = bool(all(abs(mean[k]) < 1e-9 for k in ("cd_h_cm", "cd_o_cm", "acc_h_cm", "acc_o_cm")))
    checks["n_episodes"] = len(per)
    ok = checks["rows"] == checks["sample_rows"] and checks["same_row_ids"] and checks["finite"] and \
        checks["one_commit_url"] and checks["self_score_ok"] and len(per) == 30
    checks["VALID"] = bool(ok)
    tmp.replace(out)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    checks["sha256"] = h
    json.dump(checks, open(out.with_suffix(".validation.json"), "w"), indent=1)
    print(json.dumps(checks, indent=1))
    if not ok:
        raise SystemExit("VALIDATION FAILED")


if __name__ == "__main__":
    main()
