#!/bin/bash
# MANO-free hand pipeline, all stages, CPU only (no GPU lock needed). Re-runnable: stages skip finished outputs
# unless OVERWRITE=1. Usage:  bash run_hands.sh [EPISODES=all] ; logs in /mnt/secondary/v2d/t3/logs/hands_*.log
#   detect  MediaPipe HandLandmarker fwd+bwd VIDEO passes on cam_a          -> det_mp_cam_a.npz
#   sample  stereo depth windows (FoundationStereo if complete, else SGBM)  -> depth_samples_{fs,sgbm}.npz
#   lift    tracks/handedness/3D/temporal optimisation (frozen settings)    -> hands_rect.npz (+ .json)
# Dev-only extras (public split): stereo_check.py, gt_cam_register.py, eval_hands.py (see README.md).
set -euo pipefail
EPS=${1:-all}
H=$(cd "$(dirname "$0")" && pwd)
P=/mnt/secondary/v2d/envs/t3-hands/bin/python
L=/mnt/secondary/v2d/t3/logs
OW=${OVERWRITE:+--overwrite}
avail=$(free -g | awk '/^Mem:/{print $7}')
if [ "$avail" -lt 10 ]; then echo "only ${avail} GB RAM available (< 10); refusing"; exit 1; fi
export OMP_NUM_THREADS=2
nice -n 10 $P -I $H/detect_mp.py --episodes $EPS --workers 3 $OW >> $L/hands_detect_mp.log 2>&1
nice -n 10 $P -I $H/sample_depth.py --episodes $EPS --workers 3 $OW >> $L/hands_sample_depth.log 2>&1
OMP_NUM_THREADS=1 nice -n 10 $P -I $H/lift_hands.py --episodes $EPS --workers 3 --overwrite >> $L/hands_lift.log 2>&1
grep -h -E "^(public|evaluation)/" $L/hands_lift.log | tail -n 40
