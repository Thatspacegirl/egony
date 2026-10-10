#!/bin/bash
# Stage-1 shape completion for every object of the given episodes -> meshes/stage1c (mode by object name; tuned on
# PUBLIC episodes, applied unchanged to evaluation): cups revolve, flat boards / spoon mirror+extrude, others mirror
# about the vertical bilateral-symmetry plane (t3_mesh_complete.py).
#   bash complete_all.sh EP [EP ...]
T=$(cd "$(dirname "$0")" && pwd); F=/mnt/secondary/v2d/t3/frames
for ep in "$@"; do
  d=$(ls -d $F/*/episode_$(printf %06d $ep)); sp=$(basename $(dirname $d))
  for obj in $(python3 -c "import json;print(' '.join(json.load(open('$d/meta.json'))['objects']))"); do
    # public dev CD-O (stage-1 -> completed): cups 1.41-1.83 -> 0.32-0.60 (revolve); boards 0.31-0.33 -> 0.20-0.21
    # (mirror_extrude); rack 0.61/1.21 -> 0.50/1.01, sweeper 0.42 -> 0.35, dust pan 0.38 -> 0.25, pitcher 0.97/0.94 ->
    # 0.70/1.00, pot 1.81/1.06 -> 1.60/1.09, lid 0.28 -> 0.27 (mirror)
    case $obj in
      blue_cup|beige_cup) mode=${CUP_MODE:-revolve} ;;
      wooden_piece_1|wooden_piece_2|wooden_spoon) mode=${FLAT_MODE:-mirror_extrude} ;;
      white_pot) mode=${POT_MODE:-revolve_keep} ;;   # public 0/2/26/27/41: mirror 1.46 mean -> revolve_keep 1.14
      *) mode=${OTHER_MODE:-mirror} ;;
    esac
    OMP_NUM_THREADS=4 nice -n 10 /mnt/secondary/v2d/envs/t3-fpose/bin/python -I $T/t3_mesh_complete.py --split $sp \
      --episode $ep --obj $obj --mode $mode >> /mnt/secondary/v2d/t3/logs/cpu_complete.log 2>&1 || echo "complete FAILED $sp $ep $obj $mode"
  done
done
