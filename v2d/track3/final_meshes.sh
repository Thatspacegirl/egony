#!/bin/bash
# Final per-episode meshes -> /mnt/secondary/v2d/t3/meshes/final/<split>/episode_X/<obj>.{ply,json}
# Policy (chosen on PUBLIC CD-O, applied unchanged to evaluation; fusion never mixes splits):
#   cross-episode fusion (t3_xep_fuse.py) + one shared completion (t3_xep_complete.py) for objects seen in >= 2 episodes
#   of the split:  blue_cup / beige_cup (revolve), mini_sweeper / water_pitcher / plastic_dish_rack (mirror)
#     public dev: cups 0.42-0.61 -> 0.30; sweeper 0.35/0.87 -> 0.32; pitcher 0.76/1.05 -> 0.83; rack 0.50-0.63 -> 0.55-0.57
#   everything else: per-episode stage1c (complete_all.sh): pot revolve_keep, lid / dust pan mirror, boards / spoon
#     mirror_extrude (xep made the dust pan worse: 0.21-0.26 -> 0.31; the pot's handle yaw is ambiguous in xep).
#   bash final_meshes.sh SPLIT EP [EP ...]      (all episodes of the split that should take part)
T=$(cd "$(dirname "$0")" && pwd); M=/mnt/secondary/v2d/t3/meshes; F=/mnt/secondary/v2d/t3/frames
PY=/mnt/secondary/v2d/envs/t3-fpose/bin/python; export OMP_NUM_THREADS=4
sp=$1; shift
declare -A EPS_OF
for ep in "$@"; do
  for obj in $(python3 -c "import json;print(' '.join(json.load(open('$F/$sp/episode_$(printf %06d $ep)/meta.json'))['objects']))"); do
    EPS_OF[$obj]="${EPS_OF[$obj]} $ep"
  done
done
for ep in "$@"; do  # default: per-episode completed meshes
  e=episode_$(printf %06d $ep); mkdir -p $M/final/$sp/$e
  for f in $M/stage1c/$sp/$e/*.ply; do o=$(basename $f .ply); cp $f $M/final/$sp/$e/; cp $M/stage1c/$sp/$e/$o.json $M/final/$sp/$e/; done
done
for obj in "${!EPS_OF[@]}"; do
  case $obj in blue_cup|beige_cup) mode=revolve ;; mini_sweeper|water_pitcher|plastic_dish_rack) mode=mirror ;; *) continue ;; esac
  n=$(echo ${EPS_OF[$obj]} | wc -w); [ $n -lt 2 ] && continue
  rm -rf $M/xep_final/$sp/*/$obj.* $M/xep_finalc/$sp/*/$obj.*
  timeout -k 30 900 nice -n 10 $PY -I $T/t3_xep_fuse.py --split $sp --obj $obj --episodes ${EPS_OF[$obj]} --in_root $M/stage1 \
     --out_root $M/xep_final >> /mnt/secondary/v2d/t3/logs/cpu_final_meshes.log 2>&1 || { echo "xep FAILED $obj"; continue; }
  timeout -k 30 900 nice -n 10 $PY -I $T/t3_xep_complete.py --split $sp --obj $obj --mode $mode --in_root $M/xep_final \
     --out_root $M/xep_finalc >> /mnt/secondary/v2d/t3/logs/cpu_final_meshes.log 2>&1 || { echo "xep complete FAILED $obj"; continue; }
  for d in $M/xep_finalc/$sp/episode_*; do
    [ -f $d/$obj.ply ] && cp $d/$obj.ply $d/$obj.json $M/final/$sp/$(basename $d)/
  done
  echo "$sp $obj: fused over${EPS_OF[$obj]} ($mode)"
done
