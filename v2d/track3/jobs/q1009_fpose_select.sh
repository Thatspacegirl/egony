#!/bin/bash
# FoundationPose with the selected meshes: stage search order board -> select -> final (FPOSE_STAGE), output fpose_sel.
#   bash q1009_fpose_select.sh public|evaluation|both      (run under gpu_hi.sh --raw)
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3
PUB="0 2 11 12 13 18 21 22 23 24 26 27 31 39 40 41 42 43 44 45"
EV="1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38"
case ${1:-both} in public) EPS=$PUB ;; evaluation) EPS=$EV ;; *) EPS="$PUB $EV" ;; esac
echo "fpose_select start $(date -Is) ${FPOSE_EXTRA:-}"
FPOSE_STAGE="board select final" FPOSE_OUT=${FPOSE_OUT:-fpose_sel} FPOSE_EXTRA="${FPOSE_EXTRA:-}" bash $T/gpu_queue.sh fpose $EPS
echo "fpose_select end rc=$? $(date -Is)"
