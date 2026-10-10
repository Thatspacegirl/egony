#!/bin/bash
# CPU follower for evaluation chunk B: as soon as an episode's FoundationStereo depth is complete (depth_rect.json is
# written after its last frame), run the CPU chain (sync -> mask clean -> warp -> masks_to_left -> tsdf1) and the
# frozen completion policy (complete_all.sh).
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3; F=/mnt/secondary/v2d/t3/frames; D=/mnt/secondary/v2d/t3/depth/evaluation
for ep in ${EPS:-25 28 30 32 33 35 36 37 38}; do
  e=episode_$(printf %06d $ep)
  n=$(python3 -c "import json;print(json.load(open('$F/evaluation/$e/meta.json'))['n_frames'])")
  until [ -f $D/$e/depth_rect.json ] && [ "$(ls $D/$e/depth_rect | wc -l)" -ge "$n" ]; do sleep 30; done
  bash $T/cpu_chain.sh $ep
  bash $T/complete_all.sh $ep
  echo "$e chain+complete done $(date -Is)"
done
echo ALL_DONE
