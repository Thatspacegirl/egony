#!/usr/bin/env bash
# Train one Track 3 floating-Sharpa FlashSAC policy on the RTX 3090 (GPU job).
#
#   train_3090.sh TASK RUN_DIR [extra hydra overrides...]
#     TASK    = converter manifest (.../manifests/episode_XXXXXX.json) or the processed task dir
#               (.../processed/sequence_id=episode_XXXXXX/robot_name=sharpa_wave)
#     RUN_DIR = checkpoints + logs (policy_<steps>.safetensors / state_<steps>.safetensors)
#   env: SEED (default 42), FORCE=1 skips the free-GPU gate, DRY=1 only prints the composed config.
#
# Preset: configs/experiment/sharpa_flash_sac_3090.yaml (motion_speed 1.0, 2048 worlds, upc 4, 2M fp16 replay).
# Smoke:  train_3090.sh TASK RUN training.total_environment_steps=2048000 \
#             training.checkpoint.save_interval_environment_steps=1024000
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ $# -ge 2 ] || { sed -n 2,14p "$0"; exit 2; }
TASK="$1"; RUN_DIR="$(realpath -m "$2")"; shift 2
# shellcheck disable=SC1091
source "$HOME/TestingGrounds/egony/v2d/envs/env.sh" flash_chord >/dev/null
if [[ "$TASK" == *.json ]]; then
  TASK="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_parquet"])' "$TASK")"
fi
[ -d "$TASK" ] || { echo "task dir not found: $TASK" >&2; exit 2; }
[[ "$TASK" =~ episode_[0-9]{6} ]] || { echo "task path lacks episode_XXXXXX (harness regex): $TASK" >&2; exit 2; }

CMD=(python "$V2D_FC/scripts/train_flash_sac.py" --config-dir "$HERE/configs" experiment=sharpa_flash_sac_3090
     "task.parquet='$TASK'" "reset.seed=${SEED:-42}" logging.mode=disabled
     "output_dir=$RUN_DIR" "hydra.run.dir=$RUN_DIR/hydra" "$@")
if [ "${DRY:-0}" = 1 ]; then
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "${CMD[@]}" --cfg job --resolve
  exit 0
fi

avail=$(free -g | awk '/^Mem:/{print $7}')
[ "$avail" -ge 10 ] || { echo "only ${avail} GB RAM available (<10); not starting" >&2; exit 3; }
gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${FORCE:-0}" != 1 ] && [ "$gpu_free" -lt 20000 ]; then
  echo "GPU has ${gpu_free} MiB free (<20000): another job owns it. FORCE=1 to override." >&2; exit 4
fi
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.txt"; echo >> "$RUN_DIR/command.txt"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$RUN_DIR/gpu_before.txt"
# GPU memory/utilisation trace every 30 s for the measurement the preset still needs (fits 24 GB?)
( while sleep 30; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader; done ) \
  > "$RUN_DIR/gpu_trace.csv" 2>/dev/null &
TRACE=$!
trap 'kill $TRACE 2>/dev/null || true' EXIT
/mnt/secondary/so101_r2s/guard.sh "$RUN_DIR/train.log" -- nice -n 5 "${CMD[@]}"
