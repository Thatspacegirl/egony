#!/bin/bash
# Track 1 TEST predictions from a CARI4D run directory (CPU only; run detached, ONE heavy CPU job):
#   setsid nohup bash t1_final.sh <cari4d_dir> <variant:init|coconet|refined> <name> [smooth_json] > LOG 2>&1 < /dev/null &
#   e.g. t1_final.sh /mnt/secondary/v2d/t1/cari4d_sam3do_knownK refined cari4d_knownK_refined
# 1. decode every Track 1 episode whose refined.pth exists (exact MHR conversion; skips decoded ones)
# 2. collect <variant> NPZ + mesh into predictions/<name>_raw/ (copies, sha256 listed)
# 3. postprocess.py: SO(3) smoothing (smooth_json, default smooth_default.json) + PEN refinement (hand + gated object
#    stage) on the scored frames of the kit sample submission -> predictions/<name>/
# 4. t1_pack.py: kit packer -> predictions/<name>.parquet + .validation.json (row ids == sample, finite, self-score 0)
# Requires all 30 episodes for step 4; steps 1-3 run on whatever exists.
set -u
CD=$1; VAR=$2; NAME=$3; SJ=${4:-}
D=$(cd "$(dirname "$0")" && pwd)
P=/mnt/secondary/v2d/t1/predictions; mkdir -p $P
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
PY="nice -n 10 /mnt/secondary/v2d/scratch/venv/bin/python -I -W ignore"
MESHSRC=$(basename $CD | sed -E 's/^cari4d_([a-z0-9]+).*/\1/')
SAMPLE=/mnt/secondary/v2d/kit/v2d_submission_kit/data/track_1_sample_submission.parquet
RAW=$P/${NAME}_raw; OUT=$P/$NAME; mkdir -p $RAW
n=0
for E in $(seq 0 29); do
  EP=$(printf episode_%06d $E); R=$CD/track1/$EP
  if [ -f $R/inference/refined.pth ]; then BUN=refined; MK=$R/t1/decoded.npz
  elif [ -f $R/t1_run06.json ] && [ "$VAR" != refined ]; then BUN=coconet; MK=$R/t1/decoded_coconet.npz
  else echo "[final] $EP: no CARI4D output for variant $VAR"; continue; fi
  if [ ! -f $MK ]; then
    ( source /mnt/secondary/v2d/envs/t1-cari4d/v2d_env.sh && nice -n 10 /mnt/secondary/v2d/envs/t1-cari4d/bin/python \
      $D/t1_cari4d.py decode --split track1 --episode $E --mesh-source $MESHSRC --out $CD/track1 --bundle $BUN ) > $R/t1_decode.log 2>&1 \
      || { echo "[final] decode $EP failed: $(tail -2 $R/t1_decode.log)"; continue; }
  fi
  cp -f $R/t1/$VAR/$EP.npz $RAW/$EP.npz && cp -f $R/t1/$VAR/${EP}_object.glb $RAW/${EP}_object.glb && n=$((n+1))
done
echo "[final] $(date +%T) collected $n episodes in $RAW"
( cd $RAW && sha256sum episode_* > SHA256SUMS )
ARGS="--in $RAW --out $OUT --sample $SAMPLE"
[ -n "$SJ" ] && ARGS="$ARGS --smooth-json $SJ"
/mnt/secondary/so101_r2s/guard.sh $P/${NAME}_post.log -- $PY $D/postprocess.py $ARGS
echo "[final] $(date +%T) postprocess rc=$? (log $P/${NAME}_post.log)"
( cd $OUT && sha256sum episode_*.npz episode_*_object.* > SHA256SUMS 2>/dev/null )
if [ $(ls $OUT/episode_*.npz 2>/dev/null | wc -l) -eq 30 ]; then
  $PY $D/t1_pack.py --episodes $OUT --out $P/$NAME.parquet > $P/${NAME}_pack.log 2>&1
  echo "[final] $(date +%T) pack rc=$?"; tail -15 $P/${NAME}_pack.log
else
  echo "[final] only $(ls $OUT/episode_*.npz 2>/dev/null | wc -l)/30 episodes post-processed; not packing"
fi
