#!/bin/bash
# waits for the eval CPU prep, then queues batch B under gpu_hi.sh (detached launcher; pid in logs/launch_B.pid)
L=/mnt/secondary/v2d/t3/logs
until grep -q EVAL_PREP2_DONE $L/cpu_eval_prep2.log 2>/dev/null; do sleep 60; done
echo "queue B $(date -Is)"
/mnt/secondary/v2d/bin/gpu_hi.sh --raw bash /home/asubuntudesktop/TestingGrounds/egony/v2d/track3/jobs/q1009b_B.sh > $L/q1009b_B.out 2>&1
echo "B exited rc=$? $(date -Is)"
