#!/bin/bash
# GPU job (run under the shared lock): per episode SAM3 masks -> MoGe-2 depth (window, stride 2) -> TRELLIS mesh
# -> FoundationPose scale + track.  Skips stages whose outputs exist.  Stops starting new episodes after BUDGET min.
# Usage: t1_object_job.sh <val|track1> <budget_min> <episode> [episode ...]
set -u
SPLIT=$1; BUDGET=$2; shift 2
D=$(cd "$(dirname "$0")" && pwd)
R=/mnt/secondary/v2d/t1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 HF_HOME=/mnt/secondary/caches/huggingface
T0=$(date +%s)
info() { /mnt/secondary/v2d/scratch/venv/bin/python -c "
import sys; sys.path.insert(0,'$D'); import t1_items as I
it=I.item('$SPLIT',$1); s,e=I.window(it); print(it['video'], s, e)"; }
for E in "$@"; do
  el=$(( ($(date +%s) - T0) / 60 ))
  if [ $el -ge $BUDGET ]; then echo "[obj] budget reached ($el min); stop before ep $E"; break; fi
  avail=$(free -g | awk '/Mem:/{print $7}')
  if [ "$avail" -lt 10 ]; then echo "[obj] only ${avail} GB RAM available; stop before ep $E"; break; fi
  EP=$(printf episode_%06d $E)
  read V WS WE < <(info $E)
  MK=$R/masks/$SPLIT/$EP; DP=$R/depth/$SPLIT/$EP; ME=$R/mesh/$SPLIT/$EP; FP=$R/fpose/$SPLIT/$EP; KJ=$R/human/$SPLIT/$EP/moge/K.json
  echo "[obj] $(date +%T) ep $E window $WS-$WE (elapsed $el min, avail ${avail} GB)"
  if [ ! -f $MK/masks.npz ]; then
    /mnt/secondary/v2d/envs/t3-seg/bin/python -I $D/t1_masks.py --split $SPLIT --episode $E --out $MK || { echo "[obj] masks failed ep $E"; continue; }
  fi
  if [ ! -f $KJ ]; then
    ( source /mnt/secondary/v2d/envs/t1-gen/v2d_env.sh
      /mnt/secondary/v2d/envs/t1-gen/bin/python $D/t1_moge.py focal --video $V --out $(dirname $KJ) --n 24 ) || { echo "[obj] focal failed ep $E"; continue; }
  fi
  if [ ! -f $DP/depth_s0.5.npy ]; then
    ( source /mnt/secondary/v2d/envs/t1-gen/v2d_env.sh
      /mnt/secondary/v2d/envs/t1-gen/bin/python $D/t1_moge.py depth --video $V --out $DP --K $KJ --start $WS --end $WE --stride 2 --scale 0.5 ) || { echo "[obj] depth failed ep $E"; continue; }
  fi
  if [ ! -f $ME/mesh_0.ply ]; then
    ( source /mnt/secondary/v2d/envs/t1-gen/v2d_env.sh
      /mnt/secondary/v2d/envs/t1-gen/bin/python $D/t1_trellis.py --split $SPLIT --episode $E --masks $MK --out $ME ) || { echo "[obj] trellis failed ep $E"; continue; }
  fi
  if [ ! -f $FP/fpose.npz ]; then
    /mnt/secondary/v2d/envs/t3-fpose/bin/python -I $D/t1_fpose.py --split $SPLIT --episode $E --masks $MK --depth $DP --mesh $ME --out $FP || { echo "[obj] fpose failed ep $E"; continue; }
  fi
  echo "[obj] $(date +%T) ep $E done"
done
echo "[obj] finished $(date +%T), elapsed $(( ($(date +%s) - T0) / 60 )) min"
