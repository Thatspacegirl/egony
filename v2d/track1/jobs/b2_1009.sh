#!/bin/bash
# Track 1 GPU batch 2 (2026-10-09).  ONE high-priority lock hold (budget ~85 min):
#   setsid nohup /mnt/secondary/v2d/bin/gpu_hi.sh --raw bash jobs/b2_1009.sh > /mnt/secondary/v2d/t1/logs/b2_1009.log 2>&1 < /dev/null &
# SAM3 masks (stride 1, scale 0.5, window = scored span -60/+30) for the Track 1 TEST episodes, then SAM-3D-Objects
# meshes for whichever of them have masks.  Idempotent (finished episodes are skipped), so the same script can be
# re-queued until everything is done.
D=/home/asubuntudesktop/TestingGrounds/egony/v2d/track1
Q="bash $D/t1_gpu_queue.sh"
T0=$(date +%s); B=${B_BUDGET:-85}
left() { echo $(( B - ($(date +%s) - T0) / 60 )); }
echo "[b2] start $(date -Is)"; free -g | head -2
# one episode per object first (meshes can then be made per object), then the rest
ORDER="0 3 6 9 12 15 18 21 24 27 1 4 7 10 13 16 19 22 25 28 2 5 8 11 14 17 20 23 26 29"
L=$(left); T1_EP_TIMEOUT=14 $Q masks track1 $(( L - 20 )) $ORDER
L=$(left); [ $L -gt 5 ] && T1_EP_TIMEOUT=8 $Q sam3do track1 $(( L - 5 )) $ORDER
echo "[b2] end $(date -Is) elapsed $(( ($(date +%s) - T0) / 60 )) min"
