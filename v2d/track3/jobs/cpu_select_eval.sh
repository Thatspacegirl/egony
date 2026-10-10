#!/bin/bash
# After batch B's SAM 3D evaluation run: label-free multi-view selection scores for every evaluation (episode, object)
# (t3_mesh_select.py, the same settings as public; no GT involved).  The policy itself is applied later with
# t3_policy_apply.py once it is frozen on public.  Ends with SELECT_EVAL_DONE.
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3; L=/mnt/secondary/v2d/t3/logs
until grep -q "sam3do evaluation k3 rc=" $L/q1009b_B.out 2>/dev/null; do sleep 60; done
grep "sam3do evaluation k3 rc=" $L/q1009b_B.out
rm -rf /mnt/secondary/v2d/t3/select/k3/evaluation
python3 $T/t3_select_all.py --split evaluation --sam3do_root /mnt/secondary/v2d/t3/sam3do/k3 --out_root /mnt/secondary/v2d/t3/select/k3 \
   | xargs -P 2 -I{} bash -c {}
ls /mnt/secondary/v2d/t3/select/k3/evaluation/*/*.json | grep -vc cands
echo "SELECT_EVAL_DONE $(date -Is)"
