#!/bin/bash
# 2026-10-09 batch 1 (one gpu_hi.sh --raw lock hold, ~75 min):
#  0) SAM 3D Objects smoke test (4 public objects; VRAM/time/pose-convention check), guarded, 20 min timeout
#  1) FoundationStereo depth for evaluation chunk B (killed on 10-08 before it started)
#  2) FoundationPose for public chunk B with the final meshes (same settings as public chunk A on 10-08 16:35:
#     FPOSE_STAGE default "final stage1c stage1", no FPOSE_EXTRA)
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3
G=/mnt/secondary/so101_r2s/guard.sh; L=/mnt/secondary/v2d/t3/logs
echo "batch1 start $(date -Is)"
$G $L/gpu_sam3do_smoke.log -- bash -c "source /mnt/secondary/v2d/envs/t3-sam3do/v2d_env.sh && cd $T && timeout 1200 /mnt/secondary/v2d/envs/t3-sam3do/bin/python -I t3_sam3do.py --jobs /mnt/secondary/v2d/t3/sam3do/jobs/smoke.json --out /mnt/secondary/v2d/t3/sam3do/smoke"; echo "sam3do smoke rc=$? $(date -Is)"
bash $T/gpu_queue.sh fs_all 25 28 30 32 33 35 36 37 38; echo "fs_all evalB rc=$? $(date -Is)"
bash $T/gpu_queue.sh fpose 27 31 39 40 41 42 43 44 45; echo "fpose pubB rc=$? $(date -Is)"
echo "batch1 end $(date -Is)"
