#!/bin/bash
# Track 1 GPU batch 1 (2026-10-09, third agent).  ONE high-priority lock hold:
#   setsid nohup /mnt/secondary/v2d/bin/gpu_hi.sh --raw bash jobs/b1_1009.sh > /mnt/secondary/v2d/t1/logs/b1_1009.log 2>&1 < /dev/null &
# Budget ~85 min.  End-to-end smoke of every GPU stage on two short val TUNE episodes (11: 280-frame window,
# 5: 500 frames).  Val runs use FORM-HOI's released masks (T1_MASKS=masks_fh, t1_masks_formhoi.py; val only) to save
# SAM3 GPU time; SAM3 itself is smoke-tested on ep 11 (its masks are compared with FORM-HOI's on CPU).
#   1. SAM3 masks val 11 (our Track 1 mask path)
#   2. SAM-3D-Objects meshes (fh masks) val 11, 5
#   3. official CARI4D 01-07, MoGe-2 K (the toolkit default) on 11, 5       -> cari4d_sam3do_fh/val
#   4. same with the physical-camera K (--known-k) on 11, 5                  -> cari4d_sam3do_fh_knownK/val
#   5. GEM-X + SAM-3D-Body(K) human (known K) on 11, 5                        -> human/val
#   6. TRELLIS.2 (512) and TRELLIS v1 meshes on 11, 5 (mesh-source comparison; first GPU run of env t1-trellis2)
#   7. SAM-3D-Objects meshes (fh masks) for the other tune episodes (inputs of the next CARI4D batch)
# Every step is idempotent (finished episodes are skipped; CARI4D stages resume from their markers).
D=/home/asubuntudesktop/TestingGrounds/egony/v2d/track1
Q="bash $D/t1_gpu_queue.sh"
T0=$(date +%s); B=${B1_BUDGET:-85}
left() { echo $(( B - ($(date +%s) - T0) / 60 )); }
echo "[b1] start $(date -Is)"; free -g | head -2
T1_EP_TIMEOUT=8 $Q masks val 10 11
export T1_MASKS=masks_fh
T1_EP_TIMEOUT=10 $Q sam3do val 15 11 5
L=$(left); [ $L -gt 25 ] && T1_EP_TIMEOUT=25 T1_MESH=sam3do T1_TAG=_fh $Q cari4d val $(( L - 25 )) 11 5
L=$(left); [ $L -gt 25 ] && T1_EP_TIMEOUT=20 T1_MESH=sam3do T1_TAG=_fh_knownK T1_CARI_EXTRA=--known-k $Q cari4d val $(( L - 20 )) 11 5
L=$(left); [ $L -gt 12 ] && T1_EP_TIMEOUT=8 $Q gemx val $(( L - 12 )) 11 5
L=$(left); [ $L -gt 10 ] && T1_EP_TIMEOUT=8 $Q sam3do val 10 11 5
L=$(left); [ $L -gt 25 ] && T1_EP_TIMEOUT=25 T1_MESH=sam3do T1_TAG=_fh $Q cari4d val $(( L - 15 )) 11 5
L=$(left); [ $L -gt 15 ] && T1_EP_TIMEOUT=15 T1_MESH=sam3do T1_TAG=_fh $Q cari4d val $(( L - 12 )) 11 5
L=$(left); [ $L -gt 12 ] && T1_EP_TIMEOUT=15 T1_MESH=sam3do T1_TAG=_fh $Q cari4d val $(( L - 10 )) 11 5
L=$(left); [ $L -gt 8 ] && T1_EP_TIMEOUT=$(( L - 3 )) T1_MESH=sam3do T1_TAG=_fh $Q cari4d val $(( L - 5 )) 11 5
L=$(left); [ $L -gt 6 ] && T1_EP_TIMEOUT=6 $Q trellis2 val $(( L - 5 )) 11
echo "[b1] end $(date -Is) elapsed $(( ($(date +%s) - T0) / 60 )) min"
