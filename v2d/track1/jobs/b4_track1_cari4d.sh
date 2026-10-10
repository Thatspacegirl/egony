#!/bin/bash
# Track 1 TEST production: official CARI4D 01-06 (stop after CoCoNet), physical-camera K, SAM-3D-Objects mesh
# (scale hook), our SAM3 masks.  ONE high-priority lock hold per invocation (budget ~85 min); idempotent, re-queue
# until `[q] remaining 0`:
#   setsid nohup /mnt/secondary/v2d/bin/gpu_hi.sh --raw bash jobs/b4_track1_cari4d.sh > /mnt/secondary/v2d/t1/logs/b4_<tag>.log 2>&1 < /dev/null &
# Missing masks/meshes are made first (same stages as b2), so this script alone can finish Track 1.
D=/home/asubuntudesktop/TestingGrounds/egony/v2d/track1
Q="bash $D/t1_gq.sh"
T0=$(date +%s); B=${B_BUDGET:-85}
left() { echo $(( B - ($(date +%s) - T0) / 60 )); }
echo "[b4] start $(date -Is)"; free -g | head -2
ORDER="0 3 6 9 12 15 18 21 24 27 1 4 7 10 13 16 19 22 25 28 2 5 8 11 14 17 20 23 26 29"
L=$(left); T1_EP_TIMEOUT=14 $Q masks track1 $(( L - 70 > 0 ? L - 70 : 1 )) $ORDER
L=$(left); T1_EP_TIMEOUT=6 $Q sam3do track1 $(( L - 60 > 0 ? L - 60 : 1 )) $ORDER
L=$(left); [ $L -gt 15 ] && T1_EP_TIMEOUT=$(( L - 3 < 25 ? L - 3 : 25 )) T1_STOP=06 T1_MESH=sam3do T1_TAG=_knownK T1_CARI_EXTRA=--known-k \
  $Q cari4d track1 $(( L - 12 )) $ORDER
echo "[b4] end $(date -Is) elapsed $(( ($(date +%s) - T0) / 60 )) min"
