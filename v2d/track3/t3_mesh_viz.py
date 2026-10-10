"""Quick orthographic views (3 PCA axes, depth-shaded point splats) of one or more meshes side by side -> PNG.

  python -I t3_mesh_viz.py --meshes A.ply B.glb --labels ours scan --out views.png [--pca]
--pca puts every mesh in its own PCA frame (major axis vertical) so shapes compare without registration.
"""
import argparse

import cv2
import numpy as np
import trimesh


def views(V, size=260, pad=12, extent=None):
    out = []
    ext = extent or np.ptp(V, 0).max()
    s = (size - 2 * pad) / max(ext, 1e-6)
    c = (V.max(0) + V.min(0)) / 2
    for ax_u, ax_v, ax_d in [(0, 2, 1), (1, 2, 0), (0, 1, 2)]:
        img = np.zeros((size, size, 3), np.uint8)
        u = ((V[:, ax_u] - c[ax_u]) * s + size / 2).astype(int)
        v = (size / 2 - (V[:, ax_v] - c[ax_v]) * s).astype(int)
        d = V[:, ax_d]
        o = np.argsort(d)  # far first, near drawn last
        sh = ((d - d.min()) / max(np.ptp(d), 1e-6) * 200 + 55).astype(np.uint8)
        ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
        for i in o[ok[o]]:
            img[v[i], u[i]] = (sh[i], sh[i], sh[i])
        img = cv2.dilate(img, np.ones((2, 2), np.uint8))
        out.append(img)
    return np.hstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca", action="store_true")
    ap.add_argument("--n", type=int, default=40000)
    a = ap.parse_args()
    Vs = []
    for p in a.meshes:
        m = trimesh.load(p, force="mesh")
        V = np.asarray(m.sample(a.n)) if len(m.faces) else np.asarray(m.vertices)
        if a.pca:
            V = V - V.mean(0)
            U, S, Wt = np.linalg.svd(V, full_matrices=False)
            V = V @ Wt[[2, 1, 0]].T  # minor->x, mid->y, major->z (vertical)
        Vs.append(V)
    ext = max(np.ptp(V, 0).max() for V in Vs)
    rows = []
    for i, V in enumerate(Vs):
        r = views(V, extent=ext)
        lab = (a.labels[i] if a.labels and i < len(a.labels) else a.meshes[i].split("/")[-1])
        cv2.putText(r, f"{lab}  {np.round(np.ptp(V, 0) * 100, 1).tolist()} cm", (6, 18), 0, 0.5, (0, 255, 255), 1)
        rows.append(r)
    cv2.imwrite(a.out, np.vstack(rows))
    print(a.out)


if __name__ == "__main__":
    main()
