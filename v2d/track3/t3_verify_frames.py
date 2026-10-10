#!/usr/bin/env python
"""Independent on-disk check of /mnt/secondary/v2d/t3/frames (does not trust preprocess logs).

For every episode folder: #files in left/right/cam_a == meta.n_frames == parquet rows (public)
or == sample-submission rows and ffprobe frame count (evaluation); first/last JPEG decode with the
expected size; optional fresh SIFT |dy| check on a few rectified pairs.
Also normalises meta.json['dataset_revision'] to the full 40-char hash (--fix_revision).
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np, pandas as pd, cv2

REV = "5f68335f3acc802033d1e80728c1633197521de8"

def ffcount(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", p],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out else -1

def dy_check(ep, frames):
    sift = cv2.SIFT_create(4000)
    bf = cv2.BFMatcher()
    dys = []
    for f in frames:
        L = cv2.imread(f"{ep}/left/{f:06d}.jpg", 0); R = cv2.imread(f"{ep}/right/{f:06d}.jpg", 0)
        k1, d1 = sift.detectAndCompute(L, None); k2, d2 = sift.detectAndCompute(R, None)
        if d1 is None or d2 is None: continue
        for m, n in bf.knnMatch(d1, d2, k=2):
            if m.distance < 0.7 * n.distance:
                p1 = k1[m.queryIdx].pt; p2 = k2[m.trainIdx].pt
                if abs(p1[1] - p2[1]) < 8: dys.append(p1[1] - p2[1])
    dys = np.abs(np.array(dys))
    return float(np.median(dys)) if len(dys) else float("nan"), len(dys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/mnt/secondary/v2d/t3/frames")
    ap.add_argument("--ds", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--dy_frames", type=int, default=3)
    ap.add_argument("--fix_revision", action="store_true")
    a = ap.parse_args()
    cv2.setNumThreads(2)
    rid = pd.read_parquet(f"{a.kit}/data/track_3_sample_submission.parquet", columns=["row_id"]).row_id
    parts = rid.str.extract(r"e(\d+)/w\d+/f(\d+)/o\d+").astype(int)   # row_id = track_3/eEEEEEE/wWWW/fFFFFFF/oK
    sub_rows = (parts.groupby(0)[1].max() + 1).to_dict()                  # frames per eval episode
    bad = 0; rows = []
    for split in ["public", "evaluation"]:
        for ep in sorted(glob.glob(f"{a.frames}/{split}/episode_*")):
            e = int(ep[-6:]); m = json.load(open(f"{ep}/meta.json"))
            n = {d: len(os.listdir(f"{ep}/{d}")) for d in ("left", "right", "cam_a")}
            if split == "public":
                ref = len(pd.read_parquet(glob.glob(f"{a.ds}/public/data/*/episode_{e:06d}.parquet")[0]))
            else:
                ref = int(sub_rows.get(e, -1))
            vids = {c: ffcount(glob.glob(f"{a.ds}/{split}/videos/*/observation.images.ego_cam_{c}/episode_{e:06d}.mp4")[0]) for c in "abc"}
            sizes_ok = True
            for d, wh in (("left", (1280, 800)), ("right", (1280, 800)), ("cam_a", (2028, 1520))):
                for f in (0, m["n_frames"] - 1):
                    im = cv2.imread(f"{ep}/{d}/{f:06d}.jpg", cv2.IMREAD_UNCHANGED)
                    sizes_ok &= im is not None and (im.shape[1], im.shape[0]) == wh
            fr = np.linspace(0, m["n_frames"] - 1, a.dy_frames).astype(int)
            med, nm = dy_check(ep, fr)
            ok = (len(set(n.values()) | {m["n_frames"], ref} | set(vids.values())) == 1) and sizes_ok and med < 0.6
            bad += not ok
            if a.fix_revision and m.get("dataset_revision") != REV:
                m["dataset_revision"] = REV
                json.dump(m, open(f"{ep}/meta.json", "w"), indent=1)
            rows.append(dict(split=split, ep=e, files=n["left"], files_r=n["right"], files_a=n["cam_a"], meta=m["n_frames"],
                             ref=ref, ffprobe=vids, jpeg_ok=sizes_ok, med_abs_dy=round(med, 3), n_matches=nm, ok=ok))
            print(rows[-1], flush=True)
    meds = [r["med_abs_dy"] for r in rows]
    print(f"SUMMARY episodes={len(rows)} bad={bad} total_frames_public={sum(r['meta'] for r in rows if r['split']=='public')} "
          f"total_frames_eval={sum(r['meta'] for r in rows if r['split']=='evaluation')} med|dy| min/mean/max="
          f"{min(meds):.3f}/{np.mean(meds):.3f}/{max(meds):.3f}")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
