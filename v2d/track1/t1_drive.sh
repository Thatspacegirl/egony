#!/bin/bash
# Run t1_gpu_queue.sh in successive high-priority lock holds until every listed episode has its output.
#   t1_drive.sh <stage> <split> <budget_min_per_hold> <max_holds> <ep...>    (run in the background; logs in logs/)
set -u
STAGE=$1; SPLIT=$2; BUDGET=$3; MAXH=$4; shift 4
D=$(cd "$(dirname "$0")" && pwd)
LOGD=/mnt/secondary/v2d/t1/logs; mkdir -p $LOGD
TAG=${STAGE}_${SPLIT}_$(date +%m%d_%H%M%S)
for h in $(seq 1 $MAXH); do
  echo "[drive] $(date +%T) hold $h: waiting for the GPU lock"
  /mnt/secondary/v2d/bin/gpu_hi.sh --raw bash $D/t1_gpu_queue.sh $STAGE $SPLIT $BUDGET "$@" > $LOGD/q_${TAG}_h$h.log 2>&1
  echo "[drive] $(date +%T) hold $h rc=$?"; grep "^\[q\]" $LOGD/q_${TAG}_h$h.log | tail -4
  grep -q "^\[q\] remaining 0" $LOGD/q_${TAG}_h$h.log && { echo "[drive] all done"; exit 0; }
  # a hold that made no progress at all (every episode failed) -> stop, do not loop
  grep -q "rc=0" $LOGD/q_${TAG}_h$h.log || { echo "[drive] no successful episode in hold $h; stopping"; exit 1; }
done
echo "[drive] not finished after $MAXH holds"
