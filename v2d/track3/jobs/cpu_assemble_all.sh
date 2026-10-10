#!/bin/bash
# Assemble the per-episode perception output (t3_assemble.py, frozen options: --clamp_static, stereo time base) for one
# split from one FoundationPose output root, then validate every perception.npz and write a sha256 manifest.
#   FR=/mnt/secondary/v2d/t3/fpose_sel OUT=/mnt/secondary/v2d/t3/perception [DEV=/mnt/.../devscore/pred_X] \
#     bash cpu_assemble_all.sh public|evaluation [EP ...]
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3; sp=$1; shift
FR=${FR:-/mnt/secondary/v2d/t3/fpose}; OUT=${OUT:-/mnt/secondary/v2d/t3/perception}
PY=/mnt/secondary/v2d/scratch/venv/bin/python
if [ $# -gt 0 ]; then EPS="$@"; else EPS=$(ls /mnt/secondary/v2d/t3/frames/$sp | sed 's/episode_0*//; s/^$/0/'); fi
for ep in $EPS; do
  e=episode_$(printf %06d $ep)
  FD=$FR/$sp/$e; [ -d $FD ] || FD=/mnt/secondary/v2d/t3/fpose/$sp/$e   # per-episode fallback to the v1 run
  OMP_NUM_THREADS=2 nice -n 10 $PY -I $T/t3_assemble.py --split $sp --episode $ep --fpose_dir $FD --out $OUT \
     --clamp_static ${DEV:+--devscore $DEV} > $OUT/assemble_${sp}_$e.log 2>&1 || echo "assemble FAILED $sp $e"
done
$PY -I - "$OUT" "$sp" <<'PY'
import glob, hashlib, json, os, sys
import numpy as np
out, sp = sys.argv[1], sys.argv[2]
man, bad = {}, 0
for p in sorted(glob.glob(f"{out}/{sp}/episode_*/perception.npz")):
    z = np.load(p, allow_pickle=True)
    N, B = z["object_pose"].shape[:2]
    ok = (np.isfinite(z["object_pose"]).all() and z["object_valid"].shape == (N, B) and len(z["object_names"]) == B
          and all(os.path.exists(str(m)) for m in z["object_mesh_paths"]) and np.isfinite(z["table_plane"]).all()
          and np.allclose(np.linalg.norm(z["object_pose"][..., 3:], axis=-1), 1, atol=1e-4))
    bad += not ok
    man[p] = dict(sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(), frames=int(N), objects=list(map(str, z["object_names"])),
                  meshes=list(map(str, z["object_mesh_paths"])), valid_frac=float(z["object_valid"].mean()), ok=bool(ok))
    print(("OK  " if ok else "BAD ") + p, N, list(map(str, z["object_names"])), f"valid {z['object_valid'].mean():.3f}")
tmp = f"{out}/{sp}_manifest.json.tmp"
json.dump(man, open(tmp, "w"), indent=1)
os.replace(tmp, f"{out}/{sp}_manifest.json")
print(f"{len(man)} perception.npz, {bad} bad -> {out}/{sp}_manifest.json")
PY
