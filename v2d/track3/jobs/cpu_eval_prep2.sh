#!/bin/bash
# 2026-10-09 post-crash follower: after cpu_recompute_evalB.sh (eval 25-38 except 36) finishes, recompute eval 36 the
# same way, rebuild the evaluation final meshes (final_meshes.sh, frozen policy) and the SAM3D evaluation job list,
# then hash everything (checks/recompute_1009/after_eval_prep2.json).  Ends with EVAL_PREP2_DONE.
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3; C=/mnt/secondary/v2d/t3/checks/recompute_1009; L=/mnt/secondary/v2d/t3/logs
until grep -q RECOMPUTE_DONE $L/cpu_recompute_evalB.log 2>/dev/null; do sleep 30; done
EPS=36 bash $T/jobs/cpu_recompute_evalB.sh
EV="1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38"
until free -g | awk '/Mem:/ {exit !($7 >= 14)}'; do echo "waiting for RAM $(date +%T)"; sleep 60; done
bash $T/final_meshes.sh evaluation $EV
OMP_NUM_THREADS=4 nice -n 10 /mnt/secondary/v2d/envs/t3-fpose/bin/python -I $T/t3_sam3do_jobs.py --split evaluation \
   --episodes $EV --k 3 --out /mnt/secondary/v2d/t3/sam3do/jobs/evaluation_k3.json
ls /mnt/secondary/v2d/t3/meshes/final/evaluation/*/*.ply | wc -l
python3 $T/t3_hashdirs.py /mnt/secondary/v2d/t3 $C/after_eval_prep2.json meshes/stage1/evaluation meshes/stage1c/evaluation \
   meshes/final/evaluation meshes/xep_final/evaluation meshes/xep_finalc/evaluation sam3do/jobs/evaluation_k3.json
echo "EVAL_PREP2_DONE $(date -Is)"
