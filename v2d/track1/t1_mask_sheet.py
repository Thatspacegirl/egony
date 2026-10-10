#!/usr/bin/env python3
"""QA contact sheet of an episode's masks: 6 frames across the window (first scored frame included), actor = red,
object = green, at the masks' scale.  One JPEG per episode, plus an all-episodes overview (first scored frame).

  python -I t1_mask_sheet.py --split track1 --episodes 0-29 [--masks-root /mnt/secondary/v2d/t1/masks]
     -> <masks>/<split>/episode_X/sheet.jpg and <masks>/<split>/overview_f0.jpg       (env t1-gen: cv2; CPU)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_items as I  # noqa: E402


def parse_eps(s):
    out = []
    for p in s.split(","):
        if "-" in p:
            a, b = p.split("-"); out += list(range(int(a), int(b) + 1))
        elif p:
            out.append(int(p))
    return out


def frames_at(video, idx, scale):
    import cv2
    want = sorted(set(int(i) for i in idx)); got = {}
    cap = cv2.VideoCapture(str(video)); n = 0
    while n <= want[-1]:
        ok, im = cap.read()
        if not ok:
            break
        if n in want:
            got[n] = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        n += 1
    return got


def overlay(im, hm, om):
    out = im.astype(np.float32)
    out[hm] = 0.5 * out[hm] + 0.5 * np.array([0, 0, 255])
    out[om] = 0.4 * out[om] + 0.6 * np.array([0, 255, 0])
    return out.astype(np.uint8)


def main():
    import cv2
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="track1"); ap.add_argument("--episodes", default="0-29")
    ap.add_argument("--masks-root", default="/mnt/secondary/v2d/t1/masks")
    a = ap.parse_args()
    tiles = []
    for ep in parse_eps(a.episodes):
        d = Path(a.masks_root) / a.split / f"episode_{ep:06d}"
        if not (d / "masks.npz").exists():
            continue
        M = np.load(d / "masks.npz"); it = I.item(a.split, ep)
        W = int(M["W"]); fr = list(M["frames"]); sc = float(M["scale"])
        pick = sorted(set([fr.index(it["f0"]) if it["f0"] in fr else 0] + list(np.linspace(0, len(fr) - 1, 5).round().astype(int))))
        ims = frames_at(it["video"], [fr[i] for i in pick], sc)
        row = []
        for i in pick:
            im = ims[fr[i]]
            hm = np.unpackbits(M["human"][i], axis=-1, count=W).astype(bool)
            om = np.unpackbits(M["object"][i], axis=-1, count=W).astype(bool)
            t = overlay(im, hm, om)
            cv2.putText(t, f"ep{ep} f{fr[i]}{' F0' if fr[i] == it['f0'] else ''}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            row.append(cv2.resize(t, (384, 288)))
            if fr[i] == it["f0"]:
                tiles.append(row[-1])
        sheet = np.concatenate([np.concatenate(row[:3], 1), np.concatenate((row + [np.zeros_like(row[0])] * 6)[3:6], 1)], 0)
        cv2.imwrite(str(d / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("wrote", d / "sheet.jpg")
    if tiles:
        cols = 5
        tiles += [np.zeros_like(tiles[0])] * ((-len(tiles)) % cols)
        grid = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range(len(tiles) // cols)], 0)
        p = Path(a.masks_root) / a.split / "overview_f0.jpg"
        cv2.imwrite(str(p), grid, [cv2.IMWRITE_JPEG_QUALITY, 80]); print("wrote", p)


if __name__ == "__main__":
    main()
