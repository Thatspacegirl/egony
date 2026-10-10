"""Re-express FoundationPose tracks (fpose npz) in the body frame of a different mesh of the same object WITHOUT
re-running FoundationPose (CPU; the npz-level twin of t3_rebody.py).  Both meshes carry the VO-world pose of their
body during the same static window (stage JSON `frame_T_obj`), so the fixed body change is
    old_T_new = inv(frame_T_old) @ frame_T_new,   cam_T_new(t) = cam_T_old(t) @ old_T_new.
Used as a GPU-free interim when a new mesh policy is frozen before FoundationPose can be re-run with it.

  python -I t3_rebody_npz.py --split evaluation --fpose_in /mnt/secondary/v2d/t3/fpose \
      --new_stage select_k3 --out /mnt/secondary/v2d/t3/fpose_rebody
Objects without a mesh in --new_stage are symlinked unchanged.  The old mesh/JSON is the one recorded in the npz.
"""
import argparse
import glob
import json
import os

import numpy as np

R0 = "/mnt/secondary/v2d/t3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--fpose_in", required=True)
    ap.add_argument("--new_stage", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for d in sorted(glob.glob(f"{a.fpose_in}/{a.split}/episode_*")):
        e = os.path.basename(d)
        od = f"{a.out}/{a.split}/{e}"
        os.makedirs(od, exist_ok=True)
        for p in sorted(glob.glob(f"{d}/*.npz")):
            obj = os.path.basename(p)[:-4]
            dst = f"{od}/{obj}.npz"
            if os.path.lexists(dst):
                os.remove(dst)
            new = f"{R0}/meshes/{a.new_stage}/{a.split}/{e}/{obj}"
            if not os.path.exists(new + ".ply"):
                os.symlink(os.path.realpath(p), dst)
                continue
            z = dict(np.load(p, allow_pickle=True))
            old_mesh = str(z["mesh"])
            Fo = np.array(json.load(open(old_mesh[:-4] + ".json"))["frame_T_obj"])
            Fn = np.array(json.load(open(new + ".json"))["frame_T_obj"])
            D = np.linalg.inv(Fo) @ Fn
            P = z["cam_T_obj"]
            ok = np.isfinite(P).all((1, 2))
            P2 = P.copy()
            P2[ok] = P[ok] @ D
            z["cam_T_obj"] = P2
            z["mesh"] = np.array(new + ".ply")
            z["rebody_from"] = np.array(old_mesh)
            z["rebody_old_T_new"] = D
            for k in ("cam_T_obj_ref", "cam_T_obj_f0"):
                if k in z:
                    Q = z[k]
                    okq = np.isfinite(Q).all((1, 2))
                    Q2 = Q.copy()
                    Q2[okq] = Q[okq] @ D
                    z[k] = Q2
            np.savez_compressed(f"{od}/{obj}.tmp.npz", **z)
            os.replace(f"{od}/{obj}.tmp.npz", dst)
            ang = np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
            print(f"{a.split} {e} {obj}: rebody {old_mesh.split('/meshes/')[1]} -> {a.new_stage}  "
                  f"(body change {ang:.1f} deg, {np.linalg.norm(D[:3, 3]) * 100:.1f} cm)", flush=True)


if __name__ == "__main__":
    main()
