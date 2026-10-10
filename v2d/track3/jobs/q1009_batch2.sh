#!/bin/bash
# 2026-10-09 batch 2 (one gpu_hi.sh --raw lock hold, ~60 min):
#  1) SAM 3D Objects on all 20 PUBLIC episodes, 3 static-window frames per object (94 jobs, ~20 s each)
#  2) SAM 3D Objects with layout post-optimisation on the 4 smoke jobs (pose/scale check)
#  3) FoundationPose with the frame-0 anchor track (--anchor0) on public chunk A -> fpose_a0 (dev comparison)
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3
G=/mnt/secondary/so101_r2s/guard.sh; L=/mnt/secondary/v2d/t3/logs
E=/mnt/secondary/v2d/envs/t3-sam3do
echo "batch2 start $(date -Is)"
$G $L/gpu_sam3do_public_k3.log -- bash -c "source $E/v2d_env.sh && cd $T && timeout 3600 $E/bin/python -I t3_sam3do.py --jobs /mnt/secondary/v2d/t3/sam3do/jobs/public_k3.json --out /mnt/secondary/v2d/t3/sam3do/k3 --skip_existing"; echo "sam3do public k3 rc=$? $(date -Is)"
$G $L/gpu_sam3do_smoke_layout.log -- bash -c "source $E/v2d_env.sh && cd $T && timeout 900 $E/bin/python -I t3_sam3do.py --jobs /mnt/secondary/v2d/t3/sam3do/jobs/smoke.json --out /mnt/secondary/v2d/t3/sam3do/smoke_layout --layout_postprocess"; echo "sam3do smoke layout rc=$? $(date -Is)"
FPOSE_OUT=fpose_a0 FPOSE_EXTRA="--anchor0" bash $T/gpu_queue.sh fpose 0 2 11 12 13 18 21 22 23 24 26; echo "fpose a0 pubA rc=$? $(date -Is)"
echo "batch2 end $(date -Is)"
