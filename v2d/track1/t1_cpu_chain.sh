#!/bin/bash
# CPU follow-up of CARI4D runs: decode (exact MHR conversion, env t1-cari4d, CUDA hidden) for every episode whose
# refined.pth exists and is not decoded yet, then score init/coconet/refined (raw + default smoothing, --diag) on the
# val episodes given.  Idempotent.  Run detached:
#   setsid nohup bash t1_cpu_chain.sh <cari4d_variant_dir> <split> <eps...> > LOG 2>&1 < /dev/null &
#   e.g. t1_cpu_chain.sh /mnt/secondary/v2d/t1/cari4d_sam3do val 11 5
set -u
VD=$1; SPLIT=$2; shift 2
D=$(cd "$(dirname "$0")" && pwd)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
NAME=$(basename $VD)
MESHSRC=$(echo $NAME | sed -E 's/^cari4d_([a-z0-9]+).*/\1/')
done_eps=()
for E in "$@"; do
  EP=$(printf episode_%06d $E)
  R=$VD/$SPLIT/$EP
  if [ -f $R/inference/refined.pth ]; then BUN=refined; MK=$R/t1/decoded.npz
  elif [ -f $R/t1_run06.json ]; then BUN=coconet; MK=$R/t1/decoded_coconet.npz
  else echo "[chain] $EP: no CARI4D output"; continue; fi
  if [ ! -f $MK ]; then
    avail=$(free -g | awk '/Mem:/{print $7}'); [ "$avail" -lt 10 ] && { echo "[chain] only $avail GB free; stop"; break; }
    echo "[chain] $(date +%T) decode $EP"
    ( source /mnt/secondary/v2d/envs/t1-cari4d/v2d_env.sh && nice -n 10 /mnt/secondary/v2d/envs/t1-cari4d/bin/python \
      $D/t1_cari4d.py decode --split $SPLIT --episode $E --mesh-source $MESHSRC --out $VD/$SPLIT --bundle $BUN ) > $R/t1_decode.log 2>&1 \
      || { echo "[chain] decode $EP failed: $(tail -2 $R/t1_decode.log)"; continue; }
    tail -4 $R/t1_decode.log
  fi
  done_eps+=($E)
done
[ ${#done_eps[@]} -eq 0 ] && { echo "[chain] nothing decoded"; exit 0; }
[ "$SPLIT" = val ] || exit 0
EPS=$(IFS=,; echo "${done_eps[*]}")
G="$VD/val/{E}/t1"
echo "[chain] $(date +%T) score eps $EPS"
nice -n 10 /mnt/secondary/v2d/scratch/venv/bin/python -I -W ignore $D/t1_val.py \
  --src "$NAME:init=$G/init/{E}.npz" --src "$NAME:coconet=$G/coconet/{E}.npz" --src "$NAME:refined=$G/refined/{E}.npz" \
  --smooth none,default --episodes $EPS --diag --json $VD/val/score_$(date +%m%d_%H%M).json
echo "[chain] $(date +%T) done"
