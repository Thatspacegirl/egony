#!/bin/bash
# CPU chain per episode once FoundationStereo depth (depth_rect) + VO + cam_a SAM3 masks exist:
#   t3_mask_clean.py (once) -> t3_sync.py (cam_a <-> stereo frame lag) -> warp (lag-aware depth_cam_a_s0.5) -> masks_left_cpu -> tsdf1
#   bash cpu_chain.sh EP [EP ...]        (episode numbers; split resolved from frames/)   logs: /mnt/secondary/v2d/t3/logs/cpu_*.log
T=$(cd "$(dirname "$0")" && pwd)
F=/mnt/secondary/v2d/t3/frames; L=/mnt/secondary/v2d/t3/logs
PY=/mnt/secondary/v2d/envs/t3-fpose/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
free -g | awk '/Mem:/ {if ($7 < 10) {print "available RAM " $7 " GB < 10 GB, aborting"; exit 1}}' || exit 1
for ep in "$@"; do
  d=$(ls -d $F/*/episode_$(printf %06d $ep)); sp=$(basename $(dirname $d)); e=$(basename $d)
  n=$(python3 -c "import json;print(json.load(open('$d/meta.json'))['n_frames'])")
  nd=$(ls /mnt/secondary/v2d/t3/depth/$sp/$e/depth_rect 2>/dev/null | wc -l)
  nm=$(ls /mnt/secondary/v2d/t3/masks/$sp/$e/cam_a_s0.5/hand 2>/dev/null | wc -l)
  if [ "$nd" -lt "$n" ] || [ "$nm" -lt "$n" ]; then echo "$sp/$e: depth $nd masks $nm of $n -- skip"; continue; fi
  [ -f /mnt/secondary/v2d/t3/sync/$sp/$e.json ] && grep -q lag_per_frame /mnt/secondary/v2d/t3/sync/$sp/$e.json || \
    nice -n 10 $PY -I $T/t3_sync.py --eps $sp:$ep >> $L/cpu_sync.log 2>&1
  [ -f /mnt/secondary/v2d/t3/masks/$sp/$e/cam_a_s0.5/masks_clean.json ] || \
    nice -n 10 $PY -I $T/t3_mask_clean.py --split $sp --episode $ep >> $L/cpu_mask_clean.log 2>&1
  nice -n 10 $PY -I $T/t3_depth_warp.py --ep_dir $d --depth_dir /mnt/secondary/v2d/t3/depth/$sp/$e --scale 0.5 >> $L/cpu_warp.log 2>&1
  nice -n 10 $PY -I $T/t3_masks_to_left.py --split $sp --episode $ep >> $L/cpu_masks_left.log 2>&1
  bash $T/gpu_queue.sh tsdf1 $ep > /dev/null 2>&1
  echo "$sp/$e done: $(grep -c "wrote /mnt/secondary/v2d/t3/meshes/stage1/$sp/$e" $L/cpu_tsdf1.log) meshes logged"
done
