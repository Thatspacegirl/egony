#!/bin/bash
# 2026-10-09 batch B (one gpu_hi.sh --raw lock hold, target < 70 min), after jobs/cpu_eval_prep2.sh (recomputed eval
# chain + final meshes + job list):
#  1) SAM 3D Objects candidates for all 20 EVALUATION episodes (frozen public settings: --k 3 --seeds 0)
#  2) FoundationPose on all 20 EVALUATION episodes with the frozen v1 meshes (FPOSE_STAGE "board final") -> fpose/
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3
G=/mnt/secondary/so101_r2s/guard.sh; L=/mnt/secondary/v2d/t3/logs
E=/mnt/secondary/v2d/envs/t3-sam3do
EV="1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38"
echo "batchB start $(date -Is)"; free -g | head -2
$G $L/gpu_sam3do_evaluation_k3.log -- bash -c "source $E/v2d_env.sh && cd $T && timeout 2700 $E/bin/python -I t3_sam3do.py --jobs /mnt/secondary/v2d/t3/sam3do/jobs/evaluation_k3.json --out /mnt/secondary/v2d/t3/sam3do/k3 --skip_existing"; echo "sam3do evaluation k3 rc=$? $(date -Is)"
FPOSE_STAGE="board final" timeout 2700 bash $T/gpu_queue.sh fpose $EV; echo "fpose eval v1 rc=$? $(date -Is)"
echo "batchB end $(date -Is)"
# --- appended 12:45 while B was running (moved here from batch A so the public SAM3D candidates do not wait another
#     queue round; hold stays < 80 min): SAM3D public k3, FoundationPose public 40, boards 23/24 -> fpose_board
echo "batchB part2 start $(date -Is)"
$G $L/gpu_sam3do_public_k3.log -- bash -c "source $E/v2d_env.sh && cd $T && timeout 1800 $E/bin/python -I t3_sam3do.py --jobs /mnt/secondary/v2d/t3/sam3do/jobs/public_k3.json --out /mnt/secondary/v2d/t3/sam3do/k3 --skip_existing"; echo "sam3do public k3 rc=$? $(date -Is)"
bash $T/gpu_queue.sh fpose 40; echo "fpose pub40 rc=$? $(date -Is)"
FPOSE_STAGE="board final" FPOSE_OUT=fpose_board bash $T/gpu_queue.sh fpose 23 24; echo "fpose board 23 24 rc=$? $(date -Is)"
echo "batchB part2 end $(date -Is)"
