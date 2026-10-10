"""Build candidate lists (TSDF final mesh + every SAM 3D Objects output) for each (episode, object) of a split and
print one t3_mesh_select.py command per object (feed to xargs -P).

  python3 t3_select_all.py --split public --sam3do_root /mnt/secondary/v2d/t3/sam3do/k3 --out_root /mnt/secondary/v2d/t3/select/k3 \
      | xargs -P 3 -I{} bash -c {}
"""
import argparse
import glob
import json
import os

R0 = "/mnt/secondary/v2d/t3"
T3 = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--sam3do_root", default=f"{R0}/sam3do/k3")
    ap.add_argument("--tsdf_stage", default="final")
    ap.add_argument("--out_root", default=f"{R0}/select/k3")
    a = ap.parse_args()
    eps = a.episodes or sorted(int(os.path.basename(p)[8:]) for p in glob.glob(f"{R0}/frames/{a.split}/episode_*"))
    for ep in eps:
        e = f"episode_{ep:06d}"
        for obj in json.load(open(f"{R0}/frames/{a.split}/{e}/meta.json"))["objects"]:
            cands = []
            t = f"{R0}/meshes/{a.tsdf_stage}/{a.split}/{e}/{obj}"
            if os.path.exists(t + ".ply"):
                cands.append(dict(name=f"tsdf_{a.tsdf_stage}", mesh=t + ".ply", json=t + ".json"))
            for p in sorted(glob.glob(f"{a.sam3do_root}/{a.split}/{e}/{obj}/f*_s*.ply")):
                st = os.path.basename(p)[:-4]
                j = json.load(open(p[:-4] + ".json"))
                cands.append(dict(name=f"s3d_{st}", mesh=p, cam_a_frame=int(j["job"]["frame"])))
            if len(cands) < 1:
                continue
            od = f"{a.out_root}/{a.split}/{e}"
            os.makedirs(od, exist_ok=True)
            cj = f"{od}/{obj}.cands.json"
            json.dump(cands, open(cj, "w"), indent=0)
            out = f"{od}/{obj}.json"
            if os.path.exists(out):
                continue
            print(f"OMP_NUM_THREADS=2 nice -n 10 /mnt/secondary/v2d/envs/t3-fpose/bin/python -I {T3}/t3_mesh_select.py "
                  f"--split {a.split} --episode {ep} --obj {obj} --cands {cj} --scale_search --rot_search --out {out} "
                  f"> {od}/{obj}.log 2>&1")


if __name__ == "__main__":
    main()
