#!/bin/bash
# 2026-10-09 (post-crash): once batch A's SAM 3D public run has finished: label-free multi-view selection of every
# public (episode, object) (t3_mesh_select.py via t3_select_all.py, 2 parallel x 2 threads), DEV CD-O of every
# candidate + of the TSDF final meshes (recomputed; the 01:00 JSON is in the crash window), and the rule-vs-oracle
# report.  Ends with SELECT2_DONE.
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3; L=/mnt/secondary/v2d/t3/logs; C=/mnt/secondary/v2d/t3/checks
until grep -q "sam3do public k3 rc=" $L/q1009b_B.out 2>/dev/null; do sleep 60; done
grep "sam3do public k3 rc=" $L/q1009b_B.out
rm -rf /mnt/secondary/v2d/t3/select/k3/public
python3 $T/t3_select_all.py --split public --sam3do_root /mnt/secondary/v2d/t3/sam3do/k3 --out_root /mnt/secondary/v2d/t3/select/k3 \
   | xargs -P 2 -I{} bash -c {}
echo "select done $(date -Is)"
OMP_NUM_THREADS=4 nice -n 10 /mnt/secondary/v2d/scratch/venv/bin/python -I $T/t3_mesh_eval.py --root /mnt/secondary/v2d/t3/sam3do/k3 \
   --candidates --out $C/mesh_eval_sam3do_k3_public.json > $L/cpu_cdo_sam3do_k3.log 2>&1
OMP_NUM_THREADS=4 nice -n 10 /mnt/secondary/v2d/scratch/venv/bin/python -I $T/t3_mesh_eval.py --root /mnt/secondary/v2d/t3/meshes/final \
   --out $C/mesh_eval_final_public_1009b.json > $L/cpu_cdo_final_1009b.log 2>&1
for rule in iou iou_dz; do for g in 0.0 0.02; do
  echo "== rule $rule min_gain $g"
  python3 $T/t3_select_report.py --select_root /mnt/secondary/v2d/t3/select/k3 --cdo $C/mesh_eval_final_public_1009b.json \
     $C/mesh_eval_sam3do_k3_public.json --rule $rule --min_gain $g
done; done > $C/select_report_k3_public.txt 2>&1
echo "SELECT2_DONE $(date -Is)"
