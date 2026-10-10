#!/bin/bash
# 2026-10-09 batch 3 (~40 min): SAM 3D Objects candidates for all 20 EVALUATION episodes with the settings frozen on
# public (t3_sam3do_jobs.py --k 3 --seeds 0: 3 static-window frames per object; stereo-depth point map; no post-opt).
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3
G=/mnt/secondary/so101_r2s/guard.sh; L=/mnt/secondary/v2d/t3/logs
E=/mnt/secondary/v2d/envs/t3-sam3do
echo "batch3 start $(date -Is)"
$G $L/gpu_sam3do_evaluation_k3.log -- bash -c "source $E/v2d_env.sh && cd $T && timeout 4200 $E/bin/python -I t3_sam3do.py --jobs /mnt/secondary/v2d/t3/sam3do/jobs/evaluation_k3.json --out /mnt/secondary/v2d/t3/sam3do/k3 --skip_existing"; echo "sam3do evaluation k3 rc=$? $(date -Is)"
echo "batch3 end $(date -Is)"
