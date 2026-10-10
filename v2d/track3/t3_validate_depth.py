"""Integrity check of FoundationStereo depth PNGs after the 2026-10-09 RAM-fault crash (CPU only, read-only).

Per frame of depth_rect (and depth_cam_a_s0.5): decodes the PNG (zlib/CRC errors -> fail), checks dtype/shape, recomputes
the median depth and compares it with the value t3_stereo_depth.py logged at write time (depth_rect.json
median_depth_m), counts isolated single-pixel outliers (|z - median3x3(z)| > 25 % with smooth neighbours; a flipped
high byte of a uint16 inverse depth makes exactly such a spike, FoundationStereo output never does) and the fraction
of pixels whose depth jumps > 15 % w.r.t. the previous frame.  Prints one summary line per episode and every flagged
frame; exit code 1 if any frame is flagged.

  python -I t3_validate_depth.py --split evaluation --episodes 25 28 30 ... [--ref_split public --ref_episodes 12 21]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

R0 = "/mnt/secondary/v2d/t3"


def decode(p):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None or im.dtype != np.uint16:
        return None
    z = 65535.0 / np.maximum(im.astype(np.float32), 1.0) - 1.0
    z[im == 65535] = 0.0
    return z


def spikes(z):
    v = z > 0
    med = cv2.medianBlur(z.astype(np.float32), 3)
    # neighbourhood smoothness: 3x3 max-min of the median-filtered map
    k = np.ones((3, 3), np.uint8)
    rng = cv2.dilate(med, k) - cv2.erode(med, k)
    s = v & (med > 0) & (np.abs(z - med) > 0.25 * med) & (rng < 0.05 * med)
    return int(s.sum())


def check_dir(d, n_expect, logged_median=None):
    rows = []
    prev = None
    for i in range(n_expect):
        p = f"{d}/{i:06d}.png"
        if not os.path.exists(p):
            rows.append(dict(i=i, err="missing"))
            prev = None
            continue
        z = decode(p)
        if z is None:
            rows.append(dict(i=i, err="decode"))
            prev = None
            continue
        v = z > 0
        vm = (z > 0.1) & (z < 3)  # same selection as t3_stereo_depth.write()
        r = dict(i=i, valid=float(v.mean()), med=float(np.median(z[vm])) if vm.any() else 0.0, spikes=spikes(z))
        if logged_median is not None and i < len(logged_median):
            r["dmed"] = abs(r["med"] - logged_median[i])
        if prev is not None:
            both = v & (prev > 0)
            r["jump"] = float((np.abs(z[both] - prev[both]) > 0.15 * prev[both]).mean()) if both.any() else 0.0
        rows.append(r)
        prev = z
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--subdirs", nargs="+", default=["depth_rect", "depth_cam_a_s0.5"])
    ap.add_argument("--max_spikes", type=int, default=50)
    ap.add_argument("--max_jump", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    bad = 0
    allres = {}
    for ep in a.episodes:
        e = f"episode_{ep:06d}"
        n = json.load(open(f"{R0}/frames/{a.split}/{e}/meta.json"))["n_frames"]
        js = f"{R0}/depth/{a.split}/{e}/depth_rect.json"
        logged = json.load(open(js)).get("median_depth_m") if os.path.exists(js) else None
        for sd in a.subdirs:
            rows = check_dir(f"{R0}/depth/{a.split}/{e}/{sd}", n, logged if sd == "depth_rect" else None)
            allres[f"{e}/{sd}"] = rows
            errs = [r for r in rows if "err" in r]
            ok = [r for r in rows if "err" not in r]
            sp = np.array([r["spikes"] for r in ok]) if ok else np.zeros(1)
            jm = np.array([r.get("jump", 0) for r in ok]) if ok else np.zeros(1)
            dm = np.array([r["dmed"] for r in ok if "dmed" in r])
            # spikes are only meaningful on depth_rect (the forward-warped cam_a grid has natural 1-px holes)
            flag = [r for r in ok if (sd == "depth_rect" and r["spikes"] > a.max_spikes) or r.get("jump", 0) > a.max_jump
                    or r.get("dmed", 0) > 0.002]
            print(f"{a.split} {e} {sd}: {len(ok)}/{n} decoded, errors {len(errs)}, spikes p50/max "
                  f"{np.median(sp):.0f}/{sp.max():.0f}, jump p50/max {np.median(jm):.3f}/{jm.max():.3f}"
                  + (f", |median - logged| max {dm.max() * 1000:.2f} mm" if len(dm) else ""), flush=True)
            for r in errs + flag:
                print("   FLAG", r, flush=True)
            bad += len(errs) + len(flag)
    if a.out:
        tmp = a.out + ".tmp"
        json.dump(allres, open(tmp, "w"))
        os.replace(tmp, a.out)
    print(f"flagged frames: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
