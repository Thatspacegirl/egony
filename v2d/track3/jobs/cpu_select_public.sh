#!/bin/bash
# After SAM 3D public candidates exist: CD-O of every candidate (dev) + multi-view selection of every (episode, object)
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3
OMP_NUM_THREADS=4 nice -n 10 /mnt/secondary/v2d/scratch/venv/bin/python -I $T/t3_mesh_eval.py --root /mnt/secondary/v2d/t3/sam3do/k3 \
   --candidates --out /mnt/secondary/v2d/t3/checks/mesh_eval_sam3do_k3_public.json > /mnt/secondary/v2d/t3/logs/cpu_cdo_sam3do_k3.log 2>&1 &
python3 $T/t3_select_all.py --split public --sam3do_root /mnt/secondary/v2d/t3/sam3do/k3 --out_root /mnt/secondary/v2d/t3/select/k3 \
   | xargs -P 3 -I{} bash -c {}
wait
echo SELECT_DONE
