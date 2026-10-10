#!/bin/bash
# DEV ONLY (public split): assemble perception for the given public episodes into a t3_devscore.py prediction dir
# and score them with the official Track 3 metric code.
#   PRED=/mnt/secondary/v2d/t3/devscore/pred_X  ASM_EXTRA="--clamp_static --smooth 1.5 --time_base cam_a" \
#   FPOSE_DIR_ROOT=/mnt/secondary/v2d/t3/fpose  bash t3_dev_eval.sh 12 21 ...
T=$(cd "$(dirname "$0")" && pwd)
PRED=${PRED:?set PRED}; FR=${FPOSE_DIR_ROOT:-/mnt/secondary/v2d/t3/fpose}
PY=/mnt/secondary/v2d/scratch/venv/bin/python  # needs pyarrow for the parquet
mkdir -p $PRED
for ep in "$@"; do
  e=episode_$(printf %06d $ep)
  OMP_NUM_THREADS=4 nice -n 10 $PY -I $T/t3_assemble.py --split public --episode $ep --fpose_dir $FR/public/$e \
     --out ${PERC_OUT:-$PRED/perception} --devscore $PRED $ASM_EXTRA > $PRED/assemble_$e.log 2>&1 || echo "assemble FAILED $e"
done
cd $T && OMP_NUM_THREADS=4 nice -n 10 /mnt/secondary/v2d/scratch/venv/bin/python -I t3_devscore.py \
   /mnt/secondary/v2d/kit/v2d_submission_kit ~/TestingGrounds/egony/video_to_data_challenge/track_3 --pred $PRED \
   --episodes "$@" ${CDO:+--cdo} ${PER_EP:+--per_episode} 2>&1 | grep -v Warn | tail -${TAILN:-3} | tee $PRED/score.txt
