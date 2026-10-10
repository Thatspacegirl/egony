#!/bin/bash
# After all 20 evaluation episodes have stage-1 + completed meshes: build the final TSDF meshes (cross-episode fusion
# inside the evaluation split only, frozen policy of final_meshes.sh) and the SAM 3D Objects job list (same frame
# rule as public: t3_sam3do_jobs.py --k 3 --seeds 0).
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3
EV="1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38"
bash $T/final_meshes.sh evaluation $EV
OMP_NUM_THREADS=4 nice -n 10 /mnt/secondary/v2d/envs/t3-fpose/bin/python -I $T/t3_sam3do_jobs.py --split evaluation \
   --episodes $EV --k 3 --out /mnt/secondary/v2d/t3/sam3do/jobs/evaluation_k3.json
ls /mnt/secondary/v2d/t3/meshes/final/evaluation/*/*.ply | wc -l
echo PREP_DONE
