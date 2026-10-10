#!/bin/bash
# 2026-10-09 post-crash: recompute the CPU chain of the evaluation episodes whose outputs were written in the crash
# window (sync -> mask clean -> lag-aware warp -> masks_to_left -> tsdf1 -> completion), all deterministic, then compare
# sha256 with checks/recompute_1009/before_evalB_chain.json (t3_hashdirs.py).  Resumable: an episode is skipped when
# checks/recompute_1009/done_<ep> exists.   EPS="25 28 ..." bash cpu_recompute_evalB.sh
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3; L=/mnt/secondary/v2d/t3/logs; C=/mnt/secondary/v2d/t3/checks/recompute_1009
PY=/mnt/secondary/v2d/envs/t3-fpose/bin/python; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for ep in ${EPS:-25 28 30 32 33 35 37 38}; do
  [ -f $C/done_$ep ] && { echo "skip $ep (done)"; continue; }
  until free -g | awk '/Mem:/ {exit !($7 >= 14)}'; do echo "waiting for RAM $(date +%T)"; sleep 60; done
  nice -n 10 $PY -I $T/t3_sync.py --eps evaluation:$ep >> $L/cpu_sync.log 2>&1 || echo "sync FAILED $ep"
  nice -n 10 $PY -I $T/t3_mask_clean.py --split evaluation --episode $ep >> $L/cpu_mask_clean.log 2>&1 || echo "mask_clean FAILED $ep"
  bash $T/cpu_chain.sh $ep
  bash $T/complete_all.sh $ep
  touch $C/done_$ep
  echo "evaluation $ep recomputed $(date -Is)"
done
echo RECOMPUTE_DONE
