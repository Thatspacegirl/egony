#!/bin/bash
# 2026-10-09 13:40: re-run the label-free selection after extending t3_mesh_select's scale search (23/127 SAM 3D
# candidates had hit the 1.4 grid boundary).  Public first (+ DEV CD-O of candidates and TSDF finals + rule report),
# then evaluation.  3 parallel x 1 thread.  Markers: SELECT_PUB_DONE, SELECT_EVAL2_DONE.
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3; L=/mnt/secondary/v2d/t3/logs; C=/mnt/secondary/v2d/t3/checks
S=/mnt/secondary/v2d/t3/select/k3
rm -rf $S/public
python3 $T/t3_select_all.py --split public --sam3do_root /mnt/secondary/v2d/t3/sam3do/k3 --out_root $S \
   | sed 's/OMP_NUM_THREADS=2/OMP_NUM_THREADS=1/' | xargs -P 3 -I{} bash -c {}
echo "SELECT_PUB_DONE $(date -Is) $(ls $S/public/*/*.json | grep -vc cands)"
OMP_NUM_THREADS=3 nice -n 10 /mnt/secondary/v2d/scratch/venv/bin/python -I $T/t3_mesh_eval.py --root /mnt/secondary/v2d/t3/sam3do/k3 \
   --candidates --out $C/mesh_eval_sam3do_k3_public.json > $L/cpu_cdo_sam3do_k3.log 2>&1
OMP_NUM_THREADS=3 nice -n 10 /mnt/secondary/v2d/scratch/venv/bin/python -I $T/t3_mesh_eval.py --root /mnt/secondary/v2d/t3/meshes/final \
   --out $C/mesh_eval_final_public_1009b.json > $L/cpu_cdo_final_1009b.log 2>&1
for rule in iou iou_dz; do for g in 0.0 0.02 0.05; do
  echo "== rule $rule min_gain $g"
  python3 $T/t3_select_report.py --select_root $S --cdo $C/mesh_eval_final_public_1009b.json \
     $C/mesh_eval_sam3do_k3_public.json --rule $rule --min_gain $g
done; done > $C/select_report_k3_public.txt 2>&1
echo "REPORT_DONE $(date -Is)"
rm -rf $S/evaluation
python3 $T/t3_select_all.py --split evaluation --sam3do_root /mnt/secondary/v2d/t3/sam3do/k3 --out_root $S \
   | sed 's/OMP_NUM_THREADS=2/OMP_NUM_THREADS=1/' | xargs -P 3 -I{} bash -c {}
echo "SELECT_EVAL2_DONE $(date -Is) $(ls $S/evaluation/*/*.json | grep -vc cands)"
