"""Temporal clean-up of SAM3 object masks (CPU, in place; originals kept in <slot>_raw/ once).

SAM3 occasionally attaches far-away blobs (people / chairs at the image border) to a tracked instance, e.g. public 23
wooden_piece_1 after the boards are slotted together.  Per object slot, frame by frame (forward):
  * connected components (8-conn., >= --min_px); the reference is the last non-empty cleaned mask within
    --memory frames, dilated by --radius px;
  * keep every component that touches the reference; if none does, keep the largest component only when its centroid
    is within --jump of the reference centroid (fraction of the image width), else the frame becomes empty (lost);
  * at the clip start keep the largest component; after a loss longer than --memory, the largest component only if
    it is within 2 x --jump of the last known centroid.
Hand masks are not touched.  Writes masks_clean.json (per slot: frames changed, frames emptied).

  python -I t3_mask_clean.py --split public --episode 23 [--cam cam_a_s0.5]
"""
import argparse
import json
import os
import shutil

import cv2
import numpy as np

R0 = "/mnt/secondary/v2d/t3"


def clean_slot(paths, min_px=30, radius=25, memory=15, jump=0.15):
    ref, ref_f, ref_c = None, -999, None
    changed, emptied = 0, 0
    out = []
    for f, p in enumerate(paths):
        m = cv2.imread(p, 0) > 0
        H, W = m.shape
        n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        comps = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px]
        new = np.zeros_like(m)
        if comps:
            if ref is not None and f - ref_f <= memory:
                touch = [i for i in comps if (ref & (lab == i)).any()]
                if touch:
                    new = np.isin(lab, touch)
                else:
                    big = max(comps, key=lambda i: st[i, cv2.CC_STAT_AREA])
                    if np.hypot(*(cen[big] - ref_c)) < jump * W:
                        new = lab == big
            else:  # clip start: largest; after a long loss: largest only if it is near the last known place
                big = max(comps, key=lambda i: st[i, cv2.CC_STAT_AREA])
                if ref_c is None or np.hypot(*(cen[big] - ref_c)) < 2 * jump * W:
                    new = lab == big
        if new.any():
            ref = cv2.dilate(new.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0
            ys, xs = np.nonzero(new)
            ref_f, ref_c = f, np.array([xs.mean(), ys.mean()])
        changed += int((new != m).any())
        emptied += int(m.any() and not new.any())
        out.append(new)
    return out, changed, emptied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--cam", default="cam_a_s0.5")
    a = ap.parse_args()
    e = f"episode_{a.episode:06d}"
    meta = json.load(open(f"{R0}/frames/{a.split}/{e}/meta.json"))
    md = f"{R0}/masks/{a.split}/{e}/{a.cam}"
    rep = {}
    for s in meta["objects"]:
        raw = f"{md}/{s}_raw"
        if not os.path.isdir(raw):
            shutil.copytree(f"{md}/{s}", raw)
        paths = [f"{raw}/{f:06d}.png" for f in range(meta["n_frames"])]
        masks, ch, em = clean_slot(paths)
        for f, m in enumerate(masks):
            cv2.imwrite(f"{md}/{s}/{f:06d}.png", m.astype(np.uint8) * 255)
        rep[s] = dict(frames_changed=ch, frames_emptied=em, coverage=float(np.mean([m.any() for m in masks])))
    json.dump(rep, open(f"{md}/masks_clean.json", "w"), indent=1)
    print(md, rep)


if __name__ == "__main__":
    main()
