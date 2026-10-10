#!/bin/bash
# Queue the human stage under the shared GPU lock in short holds (<= BUDGET min of new work per hold).
# Usage: t1_human_driver.sh <split> <budget_min> <max_holds> <episodes...>
set -u
SPLIT=$1; BUDGET=$2; MAXH=$3; shift 3
D=$(cd "$(dirname "$0")" && pwd)
LOGD=/mnt/secondary/v2d/t1/logs; mkdir -p $LOGD
done_all() {
  for E in "$@"; do
    O=/mnt/secondary/v2d/t1/human/$SPLIT/$(printf episode_%06d $E)
    ls $O/gemx/*/sam3db_mhr.npz >/dev/null 2>&1 && ls $O/gemx/*/gemx_dump.npz >/dev/null 2>&1 || return 1
  done
  return 0
}
for h in $(seq 1 $MAXH); do
  done_all "$@" && { echo "[driver] all done"; exit 0; }
  echo "[driver] $(date +%T) waiting for lock (hold $h)"
  /mnt/secondary/so101_r2s/gpu_run.sh $LOGD/human_${SPLIT}_hold$h.log -- $D/t1_human_job.sh $SPLIT $BUDGET "$@"
  echo "[driver] $(date +%T) hold $h rc=$?"
  tail -3 $LOGD/human_${SPLIT}_hold$h.log
done
done_all "$@" && echo "[driver] all done" || echo "[driver] not finished after $MAXH holds"
