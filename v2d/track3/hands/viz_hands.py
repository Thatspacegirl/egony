"""Visual check of hands_rect.npz: per selected frame, cam_a (final joints projected), rect-LEFT and rect-RIGHT
(3D joints projected with K_rect; right = shifted by the baseline). The stereo views are not used by the 2D fit, so
alignment there checks the 3D depth (right view parallax ~ 120 px at 0.45 m from cam_a).

  python -I viz_hands.py public/episode_000012 20,50,80,110,140,170 OUT.jpg
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handlib as hl  # noqa: E402

C = hl.BONES + [(5, 9), (9, 13), (13, 17)]


def draw(img, uv, col, th):
    for a, b in C:
        if np.isfinite(uv[[a, b]]).all():
            cv2.line(img, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), col, th, cv2.LINE_AA)
    for j in hl.TIPS:
        if np.isfinite(uv[j]).all():
            cv2.circle(img, tuple(np.round(uv[j]).astype(int)), th + 2, (0, 255, 0), -1)


def main():
    ep, frames, out = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
    ep_dir = f"{hl.FRAMES}/{ep}"
    meta, Ka, Kr, T_a_rect = hl.load_meta(ep_dir)
    z = np.load(f"{hl.out_dir(meta)}/hands_rect.npz")
    B = meta["stereo"]["baseline_m"]
    rows = []
    for f in frames:
        A = cv2.imread(f"{ep_dir}/cam_a/{f:06d}.jpg")
        L = cv2.cvtColor(cv2.imread(f"{ep_dir}/left/{f:06d}.jpg", 0), cv2.COLOR_GRAY2BGR)
        R = cv2.cvtColor(cv2.imread(f"{ep_dir}/right/{f:06d}.jpg", 0), cv2.COLOR_GRAY2BGR)
        for side, col in (("left", (0, 0, 255)), ("right", (255, 0, 0))):
            if not z[f"{side}_valid"][f]:
                continue
            X = z[side][f].astype(float)
            draw(A, hl.project(hl.transform(T_a_rect, X), Ka), col, 4)
            draw(L, hl.project(X, Kr), col, 2)
            draw(R, hl.project(X - np.array([B, 0, 0]), Kr), col, 2)
            tag = f"{side[0].upper()} z={X[0, 2]:.2f} {'obs' if z[side + '_observed'][f] else 'fill'}"
            cv2.putText(A, tag, (30 if side == "left" else 1100, 1480), cv2.FONT_HERSHEY_SIMPLEX, 2, col, 5)
        cv2.putText(A, f"f{f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)
        a = cv2.resize(A, (640, 480))
        # crop the stereo views to their lower part (where the hands are) at 2x for visibility
        l = cv2.resize(L[400:800, 160:1120], (640, 267))
        r = cv2.resize(R[400:800, 160:1120], (640, 267))
        st = np.vstack([l, r, np.zeros((480 - 534 if 534 < 480 else 0, 640, 3), np.uint8)])[:480]
        if st.shape[0] < 480:
            st = np.vstack([st, np.zeros((480 - st.shape[0], 640, 3), np.uint8)])
        rows.append(np.hstack([a, st]))
    cv2.imwrite(out, np.vstack(rows))


if __name__ == "__main__":
    main()
