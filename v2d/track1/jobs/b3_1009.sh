#!/bin/bash
# Track 1 GPU batch 3 (2026-10-09).  ONE high-priority lock hold (budget ~85 min), queue script t1_gq.sh:
#   setsid nohup /mnt/secondary/v2d/bin/gpu_hi.sh --raw bash jobs/b3_1009.sh > /mnt/secondary/v2d/t1/logs/b3_1009.log 2>&1 < /dev/null &
# Val (FORM-HOI masks, T1_MASKS=masks_fh), CARI4D with the physical-camera K, stopping after CoCoNet (T1_STOP=06:
# on val ep 5 the 300-step refinement changed nothing after our smoothing -- CD-H 4.849 vs 4.849, CD-O 9.90 vs 9.93 --
# and costs ~1.8 s/frame, more than stages 01-06 together):
#   1. SAM-3D-Objects meshes for the next tune episodes; TRELLIS.2 meshes on 11, 5 (first GPU test of t1-trellis2)
#   2. CARI4D known-K on 11 and 5 (pairs with the MoGe-K runs of b1: the K decision)
#   3. CARI4D known-K on further tune episodes while the budget lasts
# Idempotent; re-queue to continue.
D=/home/asubuntudesktop/TestingGrounds/egony/v2d/track1
Q="bash $D/t1_gq.sh"
T0=$(date +%s); B=${B_BUDGET:-85}
left() { echo $(( B - ($(date +%s) - T0) / 60 )); }
echo "[b3] start $(date -Is)"; free -g | head -2
export T1_MASKS=masks_fh
T1_EP_TIMEOUT=6 $Q sam3do val 14 0 1 3 4 6 7 8
T1_EP_TIMEOUT=5 $Q trellis2 val 8 11 5
L=$(left); [ $L -gt 20 ] && T1_EP_TIMEOUT=22 T1_STOP=06 T1_MESH=sam3do T1_TAG=_fh_knownK T1_CARI_EXTRA=--known-k $Q cari4d val $(( L - 15 )) 11 5
L=$(left); [ $L -gt 18 ] && T1_EP_TIMEOUT=$(( L - 3 < 25 ? L - 3 : 25 )) T1_STOP=06 T1_MESH=sam3do T1_TAG=_fh_knownK T1_CARI_EXTRA=--known-k $Q cari4d val $(( L - 15 )) 1 7 6 3 0 4 8
echo "[b3] end $(date -Is) elapsed $(( ($(date +%s) - T0) / 60 )) min"
