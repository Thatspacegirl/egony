"""DEV-ONLY metric-scale check of FoundationStereo depth on PUBLIC episodes (uses public scans + public GT poses).

(1) Wooden boards (public eps 23, 24): per static frame (no hand contact), the board's masked depth points are
    plane-fitted and the 2-D minimum-area rectangle of the in-plane points gives length x width; compared with the
    public scan's extents (PCA axes z = length, y = width).  Mask edge effects bias both a little low; the ratio
    length/scan_length is the depth scale estimate.
(2) Inter-object distance per untouched frame: distance between the two boards' plane-centroids vs the GT distance
    (same frame) between the two scan origins (surface centroids; for flat boards the visible-top centroid sits ~half the
    thickness above it for BOTH boards, which cancels in the distance when both lie flat).
Never run on evaluation episodes (refuses).

  python -I t3_scale_check.py --episodes 23 24 [--frames_every 5] [--out J.json]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t3_depth_warp import decode_inv_depth  # noqa: E402

R0 = "/mnt/secondary/v2d/t3"
DS = os.path.expanduser("~/TestingGrounds/egony/video_to_data_challenge/track_3")


def board_points(dep, m, K):
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5))) > 0
    v, u = np.nonzero(m & (dep > 0.2) & (dep < 2))
    z = dep[v, u]
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)


def rect_dims(X, m_full, dep, K):
    """plane fit on interior points, then project the FULL mask's pixels as rays onto that plane (edges come from
    the mask, not from bleeding depth) and take the min-area rectangle."""
    c = np.median(X, 0)
    for _ in range(3):
        U, S, Vt = np.linalg.svd(X - c, full_matrices=False)
        n = Vt[2]
        r = np.abs((X - c) @ n)
        keep = r < max(0.004, 3 * np.median(r))
        c = X[keep].mean(0)
        X = X[keep]
    U, S, Vt = np.linalg.svd(X - c, full_matrices=False)
    n, e1, e2 = Vt[2], Vt[0], Vt[1]
    cnt, _ = cv2.findContours(m_full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnt, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    rays = np.stack([(cnt[:, 0] - K[0, 2]) / K[0, 0], (cnt[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(cnt))], 1)
    t = (n @ c) / (rays @ n)
    P = rays * t[:, None]
    q = np.stack([(P - c) @ e1, (P - c) @ e2], 1).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(q)
    return max(w, h), min(w, h), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=[23, 24])
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pyarrow.parquet as pq
    import trimesh
    res = []
    scan = {}
    for nm in ["wooden_piece_1", "wooden_piece_2"]:
        mm = trimesh.load(f"{DS}/public/mesh/{nm}/{nm}_visual.glb", force="mesh")
        V = np.asarray(mm.vertices)
        board = V[np.abs(V[:, 0] - np.median(V[:, 0])) < 0.012]  # exclude the marker balls (they stick out in x)
        scan[nm] = dict(length=float(np.ptp(board[:, 2])), width=float(np.ptp(board[:, 1])))
    print("scan board dims (m):", scan)
    for ep in a.episodes:
        e = f"episode_{ep:06d}"
        if not os.path.exists(f"{DS}/public/data/chunk-000/{e}.parquet"):
            raise SystemExit(f"{ep} is not a public episode")
        meta = json.load(open(f"{R0}/frames/public/{e}/meta.json"))
        K = np.array(meta["images"]["left"]["K"])
        objs = meta["objects"]
        t = pq.read_table(f"{DS}/public/data/chunk-000/{e}.parquet").to_pandas()
        gtp = np.stack([np.array([np.asarray(o["pose"])[:3] for o in row]) for row in t["observation.objects"]])
        md = f"{R0}/masks/public/{e}/left_s1"
        per = {o: [] for o in objs}
        dist = []
        for f in range(0, meta["n_frames"], a.every):
            dep = decode_inv_depth(cv2.imread(f"{R0}/depth/public/{e}/depth_rect/{f:06d}.png", cv2.IMREAD_UNCHANGED))
            h = cv2.imread(f"{md}/hand/{f:06d}.png", 0)
            hd = cv2.dilate((h > 0).astype(np.uint8), np.ones((41, 41))) > 0 if h is not None else None
            cs = {}
            for o in objs:
                m = cv2.imread(f"{md}/{o}/{f:06d}.png", 0) > 0
                if m.sum() < 2000 or (hd is not None and hd[m].any()):
                    continue
                X = board_points(dep, m, K)
                if len(X) < 1000:
                    continue
                L, Wd, c = rect_dims(X, m, dep, K)
                per[o].append((f, L, Wd))
                cs[o] = c
            if len(cs) == 2:  # GT distance at the SAME frame (both boards lie flat when not touched)
                dist.append((np.linalg.norm(cs[objs[0]] - cs[objs[1]]), np.linalg.norm(gtp[f, 0] - gtp[f, 1])))
        r = dict(episode=ep)
        for o in objs:
            if per[o]:
                A = np.array(per[o])
                r[o] = dict(n=len(A), length_cm=float(np.median(A[:, 1]) * 100), width_cm=float(np.median(A[:, 2]) * 100),
                            scan_length_cm=scan[o]["length"] * 100, scan_width_cm=scan[o]["width"] * 100,
                            ratio_length=float(np.median(A[:, 1]) / scan[o]["length"]),
                            ratio_width=float(np.median(A[:, 2]) / scan[o]["width"]))
        if dist:
            D = np.array(dist)
            r["interobj"] = dict(n=len(D), ours_cm=float(np.median(D[:, 0]) * 100), gt_cm=float(np.median(D[:, 1]) * 100),
                                 ratio_median=float(np.median(D[:, 0] / D[:, 1])), ratio_p10_p90=np.percentile(D[:, 0] / D[:, 1], [10, 90]).round(4).tolist())
        print(json.dumps(r), flush=True)
        res.append(r)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
