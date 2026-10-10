#!/bin/bash
# GPU job (run under the shared lock):  MoGe-2 focal estimate -> GEM-X + SAM-3D-Body(K) per episode.
# Usage: t1_human_job.sh <val|track1> <budget_min> <episode> [episode ...]
#   stops starting new episodes once <budget_min> minutes have passed (keeps each lock hold short);
#   skips episodes whose outputs already exist.  Outputs: /mnt/secondary/v2d/t1/human/<split>/episode_%06d/
set -u
SPLIT=$1; BUDGET=$2; shift 2
D=$(cd "$(dirname "$0")" && pwd)
ROOT=/mnt/secondary/v2d/t1/human/$SPLIT
DS=/home/asubuntudesktop/TestingGrounds/egony/video_to_data_challenge/track_1
VAL=/mnt/secondary/v2d/t1/formhoi_val
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
T0=$(date +%s)
video_of() {
  if [ "$SPLIT" = val ]; then
    python3 -c "import json,sys;s=json.load(open('$VAL/selection.json'))['sequences'];r=[x for x in s if int(x['val_index'])==$1][0];print('$VAL/'+r['sequence_id']+'/videos__'+r['camera']+'.mp4')"
  else
    printf "%s/videos/chunk-000/observation.images.exo_camera/episode_%06d.mp4\n" $DS $1
  fi
}
for E in "$@"; do
  now=$(date +%s); el=$(( (now - T0) / 60 ))
  if [ $el -ge $BUDGET ]; then echo "[job] budget reached ($el min); stop before ep $E"; break; fi
  avail=$(free -g | awk '/Mem:/{print $7}')
  if [ "$avail" -lt 10 ]; then echo "[job] only ${avail} GB RAM available; stop before ep $E"; break; fi
  O=$ROOT/$(printf episode_%06d $E); mkdir -p $O
  V=$(video_of $E)
  echo "[job] $(date +%T) ep $E video $V (elapsed $el min, avail ${avail} GB)"
  if [ "${T1_KNOWN_K:-1}" = 1 ]; then
    # physical-camera K (FORM-HOI intrinsics; ruling 2026-10-09; t1_cam_check.py shows it holds for the Sept sessions too)
    KV=$(/mnt/secondary/v2d/scratch/venv/bin/python -c "import sys,json;sys.path.insert(0,'$D');import t1_items as I
K=json.load(open('/mnt/secondary/v2d/t1/camera_intrinsics_formhoi.json'))['cameras'][I.item('$SPLIT',$E)['camera']]['K'];print(f'{K[0][0]:.3f},{K[1][1]:.3f},{K[0][2]:.3f},{K[1][2]:.3f}')")
  elif [ ! -f $O/moge/K.json ]; then
    ( source /mnt/secondary/v2d/envs/t1-gen/v2d_env.sh
      /mnt/secondary/v2d/envs/t1-gen/bin/python $D/t1_moge.py focal --video $V --out $O/moge --n 24 ) || { echo "[job] moge failed ep $E"; continue; }
  fi
  [ "${T1_KNOWN_K:-1}" = 1 ] || KV=$(python3 -c "import json;K=json.load(open('$O/moge/K.json'))['K'];print(f'{K[0][0]:.3f},{K[1][1]:.3f},{K[0][2]:.3f},{K[1][2]:.3f}')")
  STEM=$(basename $V .mp4)
  if [ ! -f $O/gemx/$STEM/sam3db_mhr.npz ] || [ ! -f $O/gemx/$STEM/gemx_dump.npz ]; then
    ( source /mnt/secondary/v2d/envs/gemx/v2d_env.sh
      /mnt/secondary/v2d/envs/gemx/bin/python $D/t1_gemx.py --video $V --out $O/gemx --K-values $KV ) || { echo "[job] gemx failed ep $E"; continue; }
  fi
  echo "[job] $(date +%T) ep $E done"
done
echo "[job] finished $(date +%T), elapsed $(( ($(date +%s) - T0) / 60 )) min"
