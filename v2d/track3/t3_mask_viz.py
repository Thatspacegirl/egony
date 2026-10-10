"""Overlay sheet of written SAM3 video masks (all roster slots + hand) on sampled frames, to LOOK at.

  python -I t3_mask_viz.py --split evaluation --episode 3 --cam cam_a_s0.5 [--n 8] --out sheet.jpg
"""
import argparse
import json
import os

import cv2
import numpy as np

R0 = "/mnt/secondary/v2d/t3"
COL = [(0, 255, 0), (255, 0, 255), (0, 200, 255), (255, 128, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--cam", default="cam_a_s0.5")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    e = f"episode_{a.episode:06d}"
    meta = json.load(open(f"{R0}/frames/{a.split}/{e}/meta.json"))
    md = f"{R0}/masks/{a.split}/{e}/{a.cam}"
    src = "cam_a" if a.cam.startswith("cam_a") else "left"
    n = meta["n_frames"]
    fr = a.frames or np.linspace(0, n - 1, a.n).astype(int).tolist()
    tiles = []
    for f in fr:
        m0 = cv2.imread(f"{md}/{meta['objects'][0]}/{f:06d}.png", 0)
        im = cv2.imread(f"{R0}/frames/{a.split}/{e}/{src}/{f:06d}.jpg")
        im = cv2.resize(im, m0.shape[::-1], interpolation=cv2.INTER_AREA)
        for k, s in enumerate(meta["objects"] + ["hand"]):
            p = f"{md}/{s}/{f:06d}.png"
            if not os.path.exists(p):
                continue
            m = cv2.imread(p, 0) > 0
            col = np.array(COL[k] if s != "hand" else (60, 60, 255), np.uint8)
            im[m] = (0.5 * im[m] + 0.5 * col).astype(np.uint8)
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, cnt, -1, col.tolist(), 2)
            if m.any() and s != "hand":
                ys, xs = np.nonzero(m)
                cv2.putText(im, s, (int(xs.mean()) - 40, int(ys.mean())), 0, 0.6, (255, 255, 255), 2)
        cv2.putText(im, f"{a.split[:4]}{a.episode} f{f}", (8, 28), 0, 0.9, (0, 255, 255), 2)
        tiles.append(cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(a.out, json.load(open(f"{md}/masks.json")).get("coverage"))


if __name__ == "__main__":
    main()
