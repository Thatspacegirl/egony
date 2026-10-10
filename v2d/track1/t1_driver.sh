#!/bin/bash
# Queue a Track 1 GPU job under the shared lock in short holds (<= BUDGET min of new work per hold), repeating
# until every episode has its outputs (or MAXH holds).  The job scripts skip finished stages.
# Usage: t1_driver.sh <human|object> <split> <budget_min> <max_holds> <episodes...>
set -u
JOB=$1; SPLIT=$2; BUDGET=$3; MAXH=$4; shift 4
D=$(cd "$(dirname "$0")" && pwd)
R=/mnt/secondary/v2d/t1; LOGD=$R/logs; mkdir -p $LOGD
done_all() {
  for E in "$@"; do
    EP=$(printf episode_%06d $E)
    if [ $JOB = human ]; then
      ls $R/human/$SPLIT/$EP/gemx/*/sam3db_mhr.npz >/dev/null 2>&1 && ls $R/human/$SPLIT/$EP/gemx/*/gemx_dump.npz >/dev/null 2>&1 || return 1
    else
      [ -f $R/fpose/$SPLIT/$EP/fpose.npz ] || return 1
    fi
  done
  return 0
}
for h in $(seq 1 $MAXH); do
  done_all "$@" && { echo "[driver] all done"; exit 0; }
  echo "[driver] $(date +%T) waiting for lock (hold $h)"
  /mnt/secondary/so101_r2s/gpu_run.sh $LOGD/${JOB}_${SPLIT}_hold$h.log -- $D/t1_${JOB}_job.sh $SPLIT $BUDGET "$@"
  echo "[driver] $(date +%T) hold $h rc=$?"
  grep -E "^\[(job|obj)\]" $LOGD/${JOB}_${SPLIT}_hold$h.log | tail -4
done
done_all "$@" && echo "[driver] all done" || echo "[driver] not finished after $MAXH holds"
