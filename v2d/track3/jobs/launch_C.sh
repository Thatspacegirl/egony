#!/bin/bash
# queue batch C only after batch A has ended (so C never overtakes A in the non-FIFO lock race) and the flags are frozen
L=/mnt/secondary/v2d/t3/logs; J=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3/jobs
until grep -q "batchA end" $L/q1009b_A.out 2>/dev/null && [ -f $J/C_FLAGS ]; do sleep 60; done
echo "queue C $(date -Is) flags='$(cat $J/C_FLAGS)'"
/mnt/secondary/v2d/bin/gpu_hi.sh --raw bash $J/q1009b_C.sh > $L/q1009b_C.out 2>&1
echo "C exited rc=$? $(date -Is)"
