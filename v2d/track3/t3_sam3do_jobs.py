"""Job list for t3_sam3do.py: for every (episode, object) pick K cam_a frames of the stage-1 static window (object
static and hand-free): the frame with the largest SAM3 mask plus frames spread over the window (different head poses).
Frames whose object mask touches the dilated hand mask or the image border are skipped when possible.

  python -I t3_sam3do_jobs.py --split public --episodes 0 2 ... --k 3 --seeds 0 --out JOBS.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
R0 = "/mnt/secondary/v2d/t3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--episodes", type=int, nargs="+")
    ap.add_argument("--objs", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from t3_mesh_select import Episode
    jobs = []
    for epn in a.episodes:
        e = f"episode_{epn:06d}"
        objs = json.load(open(f"{R0}/frames/{a.split}/{e}/meta.json"))["objects"]
        for obj in objs:
            if a.objs and obj not in a.objs:
                continue
            ep = Episode(a.split, epn, obj)
            win = ep.window_cam_frames(f"{R0}/meshes/stage1/{a.split}/{e}/{obj}.json", 1000)
            if not win:
                print(f"no window frames for {a.split}/{e}/{obj}")
                continue
            hk = np.ones((21, 21), np.uint8)
            info = []
            for j in win:
                m = ep.mask(obj, j)
                hand = cv2.dilate(ep.mask("hand", j).astype(np.uint8), hk) > 0
                border = m[:3].any() or m[-3:].any() or m[:, :3].any() or m[:, -3:].any()
                info.append((j, int(m.sum()), bool((m & hand).any()), bool(border)))
            good = [x for x in info if not x[2] and not x[3]] or info
            pick = [max(good, key=lambda x: x[1])[0]]
            for q in np.linspace(0, len(good) - 1, a.k + 1):
                if len(pick) >= a.k:
                    break
                j = good[int(round(q))][0]
                if all(abs(j - p) >= 4 for p in pick):
                    pick.append(j)
            for j in pick:
                for s in a.seeds:
                    jobs.append(dict(split=a.split, episode=epn, obj=obj, frame=int(j), seed=s))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(jobs, open(a.out, "w"), indent=0)
    print(f"{len(jobs)} jobs -> {a.out}")


if __name__ == "__main__":
    main()
