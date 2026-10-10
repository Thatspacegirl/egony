#!/usr/bin/env python3
"""Range-fetch selected members of FORM-HOI tars (nvidia/form-hoi, CC-BY-4.0) without downloading whole tars.

For each sequence in selection.json it walks the tar headers with HTTP range reads (HfFileSystem), extracts
only the wanted members (exact member-path match), verifies each one against the release manifest's sha256
(manifest/files.parquet), and writes them flat into <out>/<sequence_id>/ ('/' -> '__').

Default members: the one exo camera video chosen in selection.json, that camera's human/object mask H5,
GT (mhr_params_mv.pt, poses.npy, pose_valid_mask.npy, interaction_trim.json, failure_segments.json),
object_mesh/output_aligned.glb (+ symmetry), hoi_metadata.yaml, ground_plane.json and edex (calibration;
validation-only use: NEVER feed FORM-HOI calibration or meshes into Track 1 inference).

usage: formhoi_fetch.py --selection selection.json --files-manifest files.parquet --out /mnt/secondary/v2d/t1/formhoi_val
"""
import argparse, hashlib, json, os, sys, time
import pandas as pd
from huggingface_hub import HfFileSystem

GT_MEMBERS = ["mhr_params_mv.pt", "poses.npy", "pose_valid_mask.npy", "interaction_trim.json",
              "failure_segments.json", "object_mesh/output_aligned.glb", "object_mesh/output_symmetry.json",
              "hoi_metadata.yaml", "ground_plane.json", "edex"]


def walk_and_fetch(fs, repo_path, seq, want, outdir):
    got = {}
    with fs.open(repo_path, "rb", block_size=1 << 16) as f:
        off, pax_name = 0, None
        while len(got) < len(want):
            f.seek(off); h = f.read(512)
            if len(h) < 512 or h == b"\0" * 512:
                break
            name = h[0:100].rstrip(b"\0").decode()
            prefix = h[345:500].rstrip(b"\0").decode()
            if prefix:
                name = prefix + "/" + name
            size = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
            typ = h[156:157]
            data_off = off + 512
            if typ in (b"x", b"g"):
                f.seek(data_off); pax = f.read(size).decode(errors="replace")
                for line in pax.splitlines():
                    parts = line.split(" ", 1)
                    if len(parts) == 2 and parts[1].startswith("path="):
                        pax_name = parts[1][5:]
            else:
                full = pax_name or name; pax_name = None
                base = full.split("/", 1)[1] if full.startswith(seq + "/") else full
                if base in want and base not in got:
                    f.seek(data_off); data = f.read(size)
                    dst = os.path.join(outdir, base.replace("/", "__"))
                    with open(dst + ".part", "wb") as w:
                        w.write(data)
                    os.replace(dst + ".part", dst)
                    got[base] = (dst, size, hashlib.sha256(data).hexdigest())
            off = data_off + ((size + 511) // 512) * 512
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--files-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--token-file", default=os.path.expanduser("~/.cache/huggingface/token"))
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--revision", default="c63db107e84c7f74bb4929ef643b67b5c8bcc00e",
                    help="nvidia/form-hoi commit used for the 2026-10-08 val set (members are sha256-checked anyway)")
    a = ap.parse_args()
    sel = json.load(open(a.selection))["sequences"]
    man = pd.read_parquet(a.files_manifest)
    token = open(a.token_file).read().strip() if os.path.exists(a.token_file) else None
    fs = HfFileSystem(token=token)
    report = []
    for r in sel:
        seq, cam = r["sequence_id"], r["camera"]
        want = set(GT_MEMBERS + [f"videos/{cam}.mp4"])
        if not a.no_masks:
            want |= {f"human_masks/{cam}.h5", f"object_masks/{cam}.h5"}
        m = man[man.sequence_id == seq].set_index("member_path")
        want = {w for w in want if w in m.index}
        outdir = os.path.join(a.out, seq)
        os.makedirs(outdir, exist_ok=True)
        # skip members already present and verified
        todo = set()
        for w in want:
            dst = os.path.join(outdir, w.replace("/", "__"))
            if not (os.path.exists(dst) and os.path.getsize(dst) == int(m.loc[w, "size"])):
                todo.add(w)
        t0 = time.time()
        repo_path = f"datasets/nvidia/form-hoi@{a.revision}/" + m["archive_path"].iloc[0]   # archive_path = data/<seq>.tar
        got = walk_and_fetch(fs, repo_path, seq, todo, outdir) if todo else {}
        bad = [w for w, (_, size, sha) in got.items() if sha != m.loc[w, "sha256"] or size != int(m.loc[w, "size"])]
        missing = sorted(todo - set(got))
        mb = sum(v[1] for v in got.values()) / 1e6
        status = "ok" if not bad and not missing else "FAIL"
        print(f"{status} {seq} cam={cam} fetched={len(got)} ({mb:.1f} MB) skipped={len(want) - len(todo)} "
              f"bad_sha={bad} missing={missing} {time.time() - t0:.1f}s", flush=True)
        report.append({"sequence_id": seq, "status": status, "bad_sha": bad, "missing": missing})
    json.dump(report, open(os.path.join(a.out, "fetch_report.json"), "w"), indent=1)
    sys.exit(0 if all(x["status"] == "ok" for x in report) else 1)


if __name__ == "__main__":
    main()
