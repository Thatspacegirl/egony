#!/bin/bash
# Track 1 GPU queue: ONE lock hold (launch through /mnt/secondary/v2d/bin/gpu_hi.sh --raw), several episodes of one
# stage, each episode under guard.sh (kernel memory-corruption watchdog).  Skips episodes whose outputs exist.
# Stops starting new episodes after BUDGET minutes or when < 10 GB RAM is available, so a hold stays < ~1.5 h.
#
#   gpu_hi.sh --raw bash t1_gpu_queue.sh <stage> <split> <budget_min> <ep> [ep ...]  > LOG 2>&1
#
# stages:
#   masks    SAM3 actor+object masks on the processing window, stride 1, scale 0.5     (env t3-seg)
#   trellis  TRELLIS-image-large object mesh from the best masked frame                (env t1-gen)
#   sam3do   SAM-3D-Objects object mesh (+ MoGe point map scale)                       (env t1-sam3do)
#   trellis2 TRELLIS.2-4B object mesh from the best masked frame (512 pipeline)          (env t1-trellis2)
#   cari4d   official CARI4D pipeline (stages 01-07, no render) on the window clip      (env t1-cari4d)
#   decode   CARI4D refined/coconet/init bundles -> lod1 vertices -> kit MHR params      (env t1-cari4d)
#   gemx     MoGe-2 focal + GEM-X / SAM-3D-Body(K)                                      (envs t1-gen, gemx)
set -u
STAGE=$1; SPLIT=$2; BUDGET=$3; shift 3
D=$(cd "$(dirname "$0")" && pwd)
R=/mnt/secondary/v2d/t1
G=/mnt/secondary/so101_r2s/guard.sh
L=$R/logs/gpu; mkdir -p $L
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 HF_HOME=/mnt/secondary/caches/huggingface
MESHSRC=${T1_MESH:-sam3do}
export T1_MASKS=${T1_MASKS:-masks}   # masks (our SAM3; Track 1 always) | masks_fh (FORM-HOI released masks, VAL ONLY)
MS=${T1_MASKS#masks}                 # suffix of mesh dirs made from those masks
[ "$T1_MASKS" = masks ] || [ "$SPLIT" = val ] || { echo "T1_MASKS=$T1_MASKS is val-only"; exit 2; }
TAG=${T1_TAG:-}          # output dir suffix for a variant, e.g. T1_TAG=_knownK T1_CARI_EXTRA="--known-k"
EXTRA=${T1_CARI_EXTRA:-}
TMO=""; [ "${T1_EP_TIMEOUT:-0}" -gt 0 ] && TMO="timeout -k 60 ${T1_EP_TIMEOUT}m"   # per-episode wall limit (minutes)
out_of() {  # the file whose existence marks stage $STAGE done for episode $1
  local EP=$(printf episode_%06d $1)
  case $STAGE in
    masks) echo $R/$T1_MASKS/$SPLIT/$EP/masks.npz ;;
    trellis) echo $R/mesh_trellis$MS/$SPLIT/$EP/mesh_0.ply ;;
    sam3do) echo $R/mesh_sam3do$MS/$SPLIT/$EP/mesh.glb ;;
    trellis2) echo $R/mesh_trellis2$MS/$SPLIT/$EP/mesh_0.ply ;;
    cari4d) echo $R/cari4d_$MESHSRC$TAG/$SPLIT/$EP/inference/refined.pth ;;
    decode) echo $R/cari4d_$MESHSRC$TAG/$SPLIT/$EP/t1/decoded.npz ;;
    gemx) ls $R/human/$SPLIT/$EP/gemx/*/sam3db_mhr.npz 2>/dev/null | head -1 || true ;;
  esac
}
T0=$(date +%s)
echo "[q] $(date -Is) start stage=$STAGE split=$SPLIT budget=${BUDGET}min eps=$*"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
for E in "$@"; do
  el=$(( ($(date +%s) - T0) / 60 ))
  if [ $el -ge $BUDGET ]; then echo "[q] budget reached ($el min); stop before ep $E"; break; fi
  avail=$(free -g | awk '/Mem:/{print $7}')
  if [ "$avail" -lt 10 ]; then echo "[q] only ${avail} GB RAM available; stop before ep $E"; break; fi
  EP=$(printf episode_%06d $E)
  F=$(out_of $E); if [ -n "$F" ] && [ -f "$F" ]; then echo "[q] ep $E $STAGE done already"; continue; fi
  case $STAGE in sam3do|trellis|trellis2|cari4d) [ -f $R/$T1_MASKS/$SPLIT/$EP/masks.npz ] || { echo "[q] ep $E: no masks yet, skip $STAGE"; continue; } ;; esac
  case $STAGE in cari4d) [ -f "$(case $MESHSRC in sam3do) echo $R/mesh_sam3do$MS/$SPLIT/$EP/mesh.glb;; trellis) echo $R/mesh_trellis$MS/$SPLIT/$EP/mesh_0.ply;; trellis2) echo $R/mesh_trellis2$MS/$SPLIT/$EP/mesh_0.ply;; *) echo /nonexistent;; esac)" ] || [ "$MESHSRC" = given ] || { echo "[q] ep $E: no $MESHSRC mesh yet, skip cari4d"; continue; } ;; esac
  LOG=$L/${STAGE}${TAG}_${SPLIT}_${EP}.log
  t1=$(date +%s)
  case $STAGE in
    masks)
      O=$R/masks/$SPLIT/$EP; [ -f $O/masks.npz ] && { echo "[q] ep $E masks exist"; continue; }
      $G $LOG -- $TMO nice -n 5 /mnt/secondary/v2d/envs/t3-seg/bin/python -I $D/t1_masks.py --split $SPLIT --episode $E \
        --out $O --stride 1 --scale 0.5 --max-rss-gb 12 ;;
    trellis)
      O=$R/mesh_trellis$MS/$SPLIT/$EP
      $G $LOG -- $TMO bash -c "source /mnt/secondary/v2d/envs/t1-gen/v2d_env.sh && nice -n 5 /mnt/secondary/v2d/envs/t1-gen/bin/python \
        $D/t1_trellis.py --split $SPLIT --episode $E --masks $R/$T1_MASKS/$SPLIT/$EP --out $O --n-views 1" ;;
    trellis2)
      O=$R/mesh_trellis2$MS/$SPLIT/$EP
      $G $LOG -- $TMO bash -c "source /mnt/secondary/v2d/envs/t1-trellis2/v2d_env.sh && nice -n 5 /mnt/secondary/v2d/envs/t1-trellis2/bin/python \
        $D/t1_trellis2.py --split $SPLIT --episode $E --masks $R/$T1_MASKS/$SPLIT/$EP --out $O ${T1_TRELLIS2_EXTRA:-}" ;;
    sam3do)
      O=$R/mesh_sam3do$MS/$SPLIT/$EP
      VID=$(/mnt/secondary/v2d/scratch/venv/bin/python -c "import sys;sys.path.insert(0,'$D');import t1_items as I;print(I.item('$SPLIT',$E)['video'])")
      $G $LOG -- $TMO bash -c "set -e; source /mnt/secondary/v2d/envs/t1-gen/v2d_env.sh
        [ -f $O/frame.json ] || nice -n 5 /mnt/secondary/v2d/envs/t1-gen/bin/python $D/t1_sam3do.py prep --split $SPLIT --episode $E --masks $R/$T1_MASKS/$SPLIT/$EP --out $O
        F=\$(python3 -c \"import json;print(json.load(open('$O/frame.json'))['video_frame'])\")
        [ -f $O/moge/points.npy ] || nice -n 5 /mnt/secondary/v2d/envs/t1-gen/bin/python $D/t1_moge.py points --video $VID --frame \$F --out $O/moge
        source /mnt/secondary/v2d/envs/t1-sam3do/v2d_env.sh
        nice -n 5 /mnt/secondary/v2d/envs/t1-sam3do/bin/python $D/t1_sam3do.py gen --split $SPLIT --episode $E --out $O" ;;
    cari4d)
      O=$R/cari4d_$MESHSRC$TAG/$SPLIT
      $G $LOG -- $TMO bash -c "source /mnt/secondary/v2d/envs/t1-cari4d/v2d_env.sh && nice -n 5 /mnt/secondary/v2d/envs/t1-cari4d/bin/python \
        $D/t1_cari4d.py run --split $SPLIT --episode $E --mesh-source $MESHSRC --out $O $EXTRA" ;;
    decode)
      $G $LOG -- $TMO bash -c "source /mnt/secondary/v2d/envs/t1-cari4d/v2d_env.sh && nice -n 5 /mnt/secondary/v2d/envs/t1-cari4d/bin/python \
        $D/t1_cari4d.py decode --split $SPLIT --episode $E --mesh-source $MESHSRC --out $R/cari4d_$MESHSRC$TAG/$SPLIT" ;;
    gemx)
      $G $LOG -- $TMO bash $D/t1_human_job.sh $SPLIT 1000 $E ;;
    *) echo "unknown stage $STAGE"; exit 2 ;;
  esac
  rc=$?
  echo "[q] $(date +%T) ep $E stage $STAGE rc=$rc $(( $(date +%s) - t1 ))s  $(grep -v '^\[guard\]' $LOG | tail -1 | cut -c1-200)"
done
left=0; for E in "$@"; do F=$(out_of $E); { [ -n "$F" ] && [ -f "$F" ]; } || left=$((left+1)); done
echo "[q] remaining $left"
echo "[q] $(date -Is) finished, elapsed $(( ($(date +%s) - T0) / 60 )) min"
