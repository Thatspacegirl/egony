#!/usr/bin/env python3
"""Do the Track 1 September-2026 sessions keep the physical cameras' intrinsics?  (CPU, env t1-gen: cv2.)

For each physical camera (front/back/left/right *_stereo_camera_left), the first frame of every Track 1 video and
of every FORM-HOI val video of that camera is matched (SIFT + RANSAC homography) against ONE Mar-Jun FORM-HOI
reference video of the same camera.  With the published FORM-HOI K (camera_intrinsics_formhoi.json, ruling
2026-10-09), a re-mount that only ROTATES the camera gives H = K R K^-1, i.e. M = K^-1 H K is a rotation:
its singular values (det-normalised) are 1.  A focal change by a factor a gives singular values ~ (a, a, 1/a^2)^(1/3)-like
spread; we report sv and the best focal ratio a* = argmin |K^-1 diag(1/a,1/a,1)... | via a 1-D search on the
orthogonality residual of diag(1/a,1/a,1) K^-1 H K.  Bystanders/actor move -> RANSAC on background matches.
Output: JSON to stdout (and --json).  Uses only Track 1 videos + FORM-HOI val videos (no GT, no extrinsics).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_items as I  # noqa: E402


def frame0(path, idx=0):
    cap = cv2.VideoCapture(str(path))
    for _ in range(idx + 1):
        ok, im = cap.read()
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def homog(a, b, sift):
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [x for x, y in m if x.distance < 0.7 * y.distance]
    pa = np.float32([ka[g.queryIdx].pt for g in good]); pb = np.float32([kb[g.trainIdx].pt for g in good])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 2.0)
    inl = inl.ravel().astype(bool)
    err = np.linalg.norm(cv2.perspectiveTransform(pa[inl][None], H)[0] - pb[inl], axis=1)
    return H, int(inl.sum()), len(good), float(np.median(err)), pa[inl], pb[inl]


def analyse(H, K):
    M = np.linalg.inv(K) @ H @ K
    M = M / np.cbrt(np.linalg.det(M))
    sv = np.linalg.svd(M, compute_uv=False)
    best = None
    for a in np.linspace(0.85, 1.15, 301):
        Ma = np.diag([1 / a, 1 / a, 1.0]) @ M     # H = K diag(a,a,1) R K^-1  (target focal = a * f)
        Ma = Ma / np.cbrt(np.linalg.det(Ma))
        r = float(np.linalg.norm(Ma @ Ma.T - np.eye(3)))
        if best is None or r < best[1]:
            best = (float(a), r)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))))
    shift = float(np.linalg.norm(H[:2, 2]))
    return {"sv": [round(float(x), 5) for x in sv], "focal_ratio_best": round(best[0], 4), "orth_resid_best": round(best[1], 5),
            "orth_resid_a1": round(float(np.linalg.norm((M @ M.T) - np.eye(3))), 5), "rot_deg": round(ang, 3)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    Kj = json.load(open("/mnt/secondary/v2d/t1/camera_intrinsics_formhoi.json"))["cameras"]
    sift = cv2.SIFT_create(4000)
    val = I.items("val"); t1 = I.items("track1")
    out = {}
    for cam in sorted(Kj):
        K = np.asarray(Kj[cam]["K"], float)
        refs = [v for v in val if v["camera"] == cam]
        if not refs:
            continue
        ref = refs[0]; R0 = frame0(ref["video"])
        rows = []
        for it in refs[1:] + [t for t in t1 if t["camera"] == cam]:
            try:
                H, ni, ng, med, _, _ = homog(R0, frame0(it["video"]), sift)
                r = {"split": it["split"], "ep": it["episode"], "seq": it["sequence_id"][:19], "inliers": ni, "good": ng,
                     "reproj_med_px": round(med, 3), **analyse(H, K)}
            except Exception as e:  # noqa: BLE001
                r = {"split": it["split"], "ep": it["episode"], "err": repr(e)}
            rows.append(r); print(cam, json.dumps(r), flush=True)
        out[cam] = {"ref": ref["sequence_id"], "rows": rows}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
