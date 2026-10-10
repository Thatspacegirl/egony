#!/bin/bash
# 2026-10-09 batch C: FINAL evaluation FoundationPose run with the frozen policy meshes ("board select_k3 final") and
# the FoundationPose flags chosen on PUBLIC from batch A (read from jobs/C_FLAGS at start; may be empty) -> fpose_final/
# Also re-runs public with exactly the same flags into fpose_final/public when jobs/C_PUBLIC=1 (so the public dev score
# of the frozen pipeline comes from the same code path).
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3
EV="1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38"
PUB="0 2 11 12 13 18 21 22 23 24 26 27 31 39 40 41 42 43 44 45"
FLAGS=$(cat $T/jobs/C_FLAGS 2>/dev/null)
echo "batchC start $(date -Is) flags='$FLAGS'"; free -g | head -2
FPOSE_STAGE="board select_k3 final" FPOSE_OUT=fpose_final FPOSE_EXTRA="$FLAGS" timeout 3000 bash $T/gpu_queue.sh fpose $EV
echo "fpose_final evaluation rc=$? $(date -Is)"
if [ "$(cat $T/jobs/C_PUBLIC 2>/dev/null)" = 1 ]; then
  FPOSE_STAGE="board select_k3 final" FPOSE_OUT=fpose_final FPOSE_EXTRA="$FLAGS" timeout 2400 bash $T/gpu_queue.sh fpose $PUB
  echo "fpose_final public rc=$? $(date -Is)"
fi
echo "batchC end $(date -Is)"
