#!/usr/bin/env bash
# Evaluate trained dev runs exactly like the kit (world_count=1, explicit reset, VOC off) and dev-score them
# against PUBLIC ground truth (GPU job for the rollouts; scoring is CPU).
#
#   eval_dev.sh MANIFEST_DIR OUT_DIR RUN_DIR [RUN_DIR...]
#     each RUN_DIR must be a train_3090.sh output for a PUBLIC episode; its newest policy_*.safetensors is used
#     (CKPT_STEP=<n> picks policy_<n>.safetensors instead).
# Writes OUT_DIR/results (evaluator json+parquet), OUT_DIR/pred_policy + OUT_DIR/pred_reference (dev-scorer
# layout) and OUT_DIR/devscore_{policy,reference}.txt.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ $# -ge 3 ] || { sed -n 2,10p "$0"; exit 2; }
MAN="$(realpath "$1")"; OUT="$(realpath -m "$2")"; shift 2
# shellcheck disable=SC1091
source "$HOME/TestingGrounds/egony/v2d/envs/env.sh" flash_chord >/dev/null
KIT=/mnt/secondary/v2d/kit/v2d_submission_kit
DS=/home/asubuntudesktop/TestingGrounds/egony/video_to_data_challenge/track_3
KITPY=/mnt/secondary/v2d/venv-kit/bin/python
SCORER="$HOME/TestingGrounds/egony/v2d/track3/t3_devscore.py"   # repo copy (imports devscore.py beside it)
[ -f "$SCORER" ] || SCORER=/mnt/secondary/v2d/scratch/track3_perception/t3_devscore.py
mkdir -p "$OUT/results"
python - "$OUT/checkpoints.json" "$@" <<'EOF'
import json, os, re, sys
from pathlib import Path
out, runs = sys.argv[1], sys.argv[2:]
mapping = {}
for run in runs:
    cmd = Path(run, "command.txt").read_text()
    ep = int(re.search(r"episode_(\d{6})", cmd).group(1))
    step = os.environ.get("CKPT_STEP")
    pols = sorted(Path(run).glob("policy_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]))
    pick = Path(run, f"policy_{step}.safetensors") if step else pols[-1]
    mapping[str(ep)] = str(pick.resolve())
json.dump(mapping, open(out, "w"), indent=1)
print(json.dumps(mapping, indent=1))
EOF
/mnt/secondary/so101_r2s/guard.sh "$OUT/eval.log" -- python "$KIT/examples/run_policy_evaluation.py" \
  --checkpoints "$OUT/checkpoints.json" --results "$OUT/results" --world-count 1 --runtime-root "$V2D_FC"
EPS=$(python -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1])).keys()))' "$OUT/checkpoints.json")
python "$HERE/rollout_to_devscore.py" "$OUT/results" "$MAN" "$OUT/pred_policy"
python "$HERE/rollout_to_devscore.py" "$OUT/results" "$MAN" "$OUT/pred_reference" --which reference
for k in policy reference; do
  OMP_NUM_THREADS=2 nice -n 10 "$KITPY" -I "$SCORER" "$KIT" "$DS" --pred "$OUT/pred_$k" --episodes $EPS \
    | tee "$OUT/devscore_$k.txt"
done
