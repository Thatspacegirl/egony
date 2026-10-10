#!/bin/bash
# 2026-10-09 batch A (REWRITTEN 15:17 while still queued, to need only ONE more queue round for both splits).
# Meshes = frozen policy ("board select_k3 final"; select_k3 = SAM 3D pots).  One gpu_hi.sh --raw hold, ~60 min:
#  1) plain FoundationPose for the select_k3 objects only (SAM 3D pots + eval 19/30 sweeper broken-mesh fallbacks) (FPOSE_REQUIRE_STAGE) -> fpose_sel/{public,evaluation}
#     (all other objects keep fpose/ v1, identical settings and meshes)
#  2) --rereg_iou 0.3 (re-register when rendered/mask IoU drops; kept only if IoU improves) for ALL objects of both
#     splits -> fpose_rr/{public,evaluation}.  Plain vs rr is decided on PUBLIC, evaluation takes the same variant.
#  3) DEV probe: --rereg_iou 0.3 --anchor0 on public 11 42 27 -> fpose_rra0/
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3
PUB="0 2 11 12 13 18 21 22 23 24 26 27 31 39 40 41 42 43 44 45"
EV="1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38"
MS="board select_k3 final"
echo "batchA start $(date -Is)"; free -g | head -2; cat /mnt/secondary/v2d/t3/meshes/select_k3/READY /mnt/secondary/v2d/t3/meshes/select_k3/EVAL_APPLIED
FPOSE_STAGE="$MS" FPOSE_REQUIRE_STAGE=select_k3 FPOSE_OUT=fpose_sel timeout 900 bash $T/gpu_queue.sh fpose 0 2 26 27 41 42 43 44 45 1 3 5 6 8 19 28 30
echo "fpose_sel pots rc=$? $(date -Is)"
FPOSE_STAGE="$MS" FPOSE_OUT=fpose_rr FPOSE_EXTRA="--rereg_iou 0.3" timeout 1800 bash $T/gpu_queue.sh fpose $PUB
echo "fpose_rr public rc=$? $(date -Is)"
FPOSE_STAGE="$MS" FPOSE_OUT=fpose_rr FPOSE_EXTRA="--rereg_iou 0.3" timeout 1800 bash $T/gpu_queue.sh fpose $EV
echo "fpose_rr evaluation rc=$? $(date -Is)"
FPOSE_STAGE="$MS" FPOSE_OUT=fpose_rra0 FPOSE_EXTRA="--rereg_iou 0.3 --anchor0" timeout 900 bash $T/gpu_queue.sh fpose 11 42 27
echo "fpose_rra0 probe rc=$? $(date -Is)"
echo "batchA end $(date -Is)"
