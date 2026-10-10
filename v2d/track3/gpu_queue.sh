#!/bin/bash
# Track 3 GPU perception stages.  Run each under the shared lock: flock /mnt/secondary/so101_r2s/gpu.lock bash gpu_queue.sh <stage>
#   bash gpu_queue.sh <stage> [episodes...]      every stage is wrapped by guard.sh, logs in /mnt/secondary/v2d/t3/logs/
# Stages, in order:
#   fs_smoke    FoundationStereo TRT engine build (Python API) + depth for public ep 12 and eval ep 1 (compare with SGBM)
#   fs_all      FoundationStereo depth for all 40 episodes (skips frames already written)
#   vo          cuVSLAM stereo VO for all 40 episodes (unmasked)
#   masks_cam_a SAM3 video masks on undistorted cam_a at 0.5 scale (+ hand) for all 40 episodes
#   masks_left  SAM3 video masks on the rectified left grid (+ hand)  [superseded by masks_left_cpu]
#   masks_left_cpu (CPU) transfer the cam_a masks to the rect-left grid through the depth (after warp)
#   vo_masked   cuVSLAM VO again with object+hand masks
#   warp        (CPU) FoundationStereo depth_rect -> undistorted cam_a grid at 0.5 scale (depth_cam_a_s0.5), all 40
#   tsdf1       (CPU) stage-1 object meshes: masked TSDF over the automatic static window (pre-grasp or post-release,
#               hand-contact test on left_s1 masks) with VO poses (vo_masked if present, else vo)
#   fpose EP    FoundationPose (NVLabs PyTorch backend) per object for the given episodes; needs stage-1 meshes in
#               /mnt/secondary/v2d/t3/meshes/stage1/<split>/episode_X/<object>.ply (stage tsdf1). Extra CLI flags via
#               FPOSE_EXTRA, e.g. FPOSE_EXTRA="--mask_depth --reregister_iou_thresh 0.3" (tune on PUBLIC eps only)
set -e
T=$(cd "$(dirname "$0")" && pwd)
G=/mnt/secondary/so101_r2s/guard.sh
L=/mnt/secondary/v2d/t3/logs
F=/mnt/secondary/v2d/t3/frames
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
free -g | awk '/Mem:/ {if ($7 < 10) {print "available RAM " $7 " GB < 10 GB, aborting"; exit 1}}'
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
stage=$1; shift || true
case $stage in
  fs_smoke)
    source $T/envs/t3-geom.env
    $G $L/gpu_fs_smoke.log -- python -I $T/t3_stereo_depth.py --frames $F --out /mnt/secondary/v2d/t3/depth --episodes 12 1 ;;
  fs_all)
    # optional episode list (keep each lock hold < 1.5 h); FS_EXTRA e.g. "--fp16"
    source $T/envs/t3-geom.env
    $G $L/gpu_fs_all_$(date +%H%M).log -- python -I $T/t3_stereo_depth.py --frames $F --out /mnt/secondary/v2d/t3/depth \
      ${1:+--episodes "$@"} $FS_EXTRA ;;
  vo|vo_masked)
    # optional episode list.  NOTE: tsdf1/fpose/assemble prefer vo_masked when it exists -- stage-1 meshes, FoundationPose
    # init and assembly must all use the SAME VO world, so (re)run tsdf1+fpose after vo_masked for those episodes.
    source $T/envs/t3-geom.env
    EPS=$(if [ $# -gt 0 ]; then for x in "$@"; do ls -d $F/*/episode_$(printf %06d $x); done; else ls -d $F/*/episode_*; fi)
    for d in $EPS; do
      args=(--ep_dir $d --out /mnt/secondary/v2d/t3/vo)
      if [ $stage = vo_masked ]; then
        sp=$(basename $(dirname $d)); ep=$(basename $d); md=/mnt/secondary/v2d/t3/masks/$sp/$ep/left_s1
        args=(--ep_dir $d --out /mnt/secondary/v2d/t3/vo_masked --mask_dir $(ls -d $md/*/ | tr '\n' ',' | sed 's/,$//'))
      fi
      $G $L/gpu_${stage}_$(basename $d).log -- python -I $T/t3_vo_cuvslam.py "${args[@]}" || echo "VO FAILED $d"
    done ;;
  masks_cam_a)
    $G $L/gpu_masks_cam_a.log -- /mnt/secondary/v2d/envs/t3-seg/bin/python -I $T/t3_masks_sam3.py --frames $F \
      --out /mnt/secondary/v2d/t3/masks --cam cam_a --scale 0.5 --hands ${1:+--episodes "$@"} ;;
  masks_left)
    $G $L/gpu_masks_left.log -- /mnt/secondary/v2d/envs/t3-seg/bin/python -I $T/t3_masks_sam3.py --frames $F \
      --out /mnt/secondary/v2d/t3/masks --cam left --scale 1 --hands ${1:+--episodes "$@"} ;;
  masks_left_cpu)
    # (CPU) cam_a SAM3 masks -> rect-left grid through the stereo depth (t3_masks_to_left.py); needs `warp` first.
    # Replaces the GPU masks_left stage (same instance ids in both views, no second SAM3 run).
    EPS=$(if [ $# -gt 0 ]; then for x in "$@"; do ls -d $F/*/episode_$(printf %06d $x); done; else ls -d $F/*/episode_*; fi)
    for d in $EPS; do
      sp=$(basename $(dirname $d)); e=$(basename $d)
      nice -n 10 /mnt/secondary/v2d/envs/t3-fpose/bin/python -I $T/t3_masks_to_left.py --split $sp --episode $((10#${e#episode_})) \
        >> $L/cpu_masks_left.log 2>&1 || echo "masks_left_cpu FAILED $sp/$e" >> $L/cpu_masks_left.log
    done ;;
  warp)
    EPS=$(if [ $# -gt 0 ]; then for x in "$@"; do ls -d $F/*/episode_$(printf %06d $x); done; else ls -d $F/*/episode_*; fi)
    for d in $EPS; do
      sp=$(basename $(dirname $d)); e=$(basename $d)
      nice -n 10 /mnt/secondary/v2d/envs/t3-fpose/bin/python -I $T/t3_depth_warp.py --ep_dir $d \
        --depth_dir /mnt/secondary/v2d/t3/depth/$sp/$e --scale 0.5 >> $L/cpu_warp.log 2>&1
    done ;;
  tsdf1)
    # TSDF1_EXTRA e.g. "--voxel 0.0025 --erode 5"; meshes get cam_a vertex colours (needs the warp stage)
    EPS=$(if [ $# -gt 0 ]; then for x in "$@"; do ls -d $F/*/episode_$(printf %06d $x); done; else ls -d $F/*/episode_*; fi)
    for d in $EPS; do
      sp=$(basename $(dirname $d)); e=$(basename $d); M=/mnt/secondary/v2d/t3/masks/$sp/$e/left_s1
      V=/mnt/secondary/v2d/t3/vo_masked/$sp/$e/vo_cuvslam.npz; [ -f $V ] || V=/mnt/secondary/v2d/t3/vo/$sp/$e/vo_cuvslam.npz
      for obj in $(python3 -c "import json;print(' '.join(json.load(open('$d/meta.json'))['objects']))"); do
        nice -n 10 /mnt/secondary/v2d/envs/t3-fpose/bin/python -I $T/t3_tsdf_fuse.py --ep_dir $d \
          --depth_dir /mnt/secondary/v2d/t3/depth/$sp/$e/depth_rect --mask_dir $M/$obj --hand_dir $M/hand \
          --poses $V:world_T_rect --frames auto --out /mnt/secondary/v2d/t3/meshes/stage1/$sp/$e/$obj.ply \
          --color_cam_a --color_depth_dir /mnt/secondary/v2d/t3/depth/$sp/$e/depth_cam_a_s0.5 $TSDF1_EXTRA \
          >> $L/cpu_tsdf1.log 2>&1 || echo "tsdf1 FAILED $sp/$e/$obj" >> $L/cpu_tsdf1.log
      done
    done ;;
  fpose)
    # our driver (t3_fpose_track.py): reference pose from the stage-1 mesh (static window, VO world) composed with VO
    # -> no global registration; tracks forward + backward.  Args: episode numbers (split resolved from frames/).
    #   FPOSE_STAGE="final stage1c stage1" (mesh dirs, first hit wins; final = final_meshes.sh), FPOSE_OUT=fpose, FPOSE_EXTRA="--mask_depth --rereg_iou 0.3"
    #   FPOSE_REQUIRE_STAGE=select_k3 -> skip objects without a mesh in that stage
    PY=/mnt/secondary/v2d/envs/t3-fpose/bin/python
    MS=${FPOSE_STAGE:-final stage1c stage1}; FO=${FPOSE_OUT:-fpose}
    for ep in "$@"; do
      for d in $F/*/episode_$(printf %06d $ep); do
        sp=$(basename $(dirname $d)); e=$(basename $d)
        V=/mnt/secondary/v2d/t3/vo_masked/$sp/$e/vo_cuvslam.npz; [ -f $V ] || V=/mnt/secondary/v2d/t3/vo/$sp/$e/vo_cuvslam.npz
        for obj in $(python3 -c "import json;print(' '.join(json.load(open('$d/meta.json'))['objects']))"); do
          M=""; for st in $MS; do [ -z "$M" ] && [ -f /mnt/secondary/v2d/t3/meshes/$st/$sp/$e/$obj.ply ] && M=/mnt/secondary/v2d/t3/meshes/$st/$sp/$e/$obj; done
          [ -z "$M" ] && { echo "fpose: no mesh for $sp/$e/$obj in $MS"; continue; }
          # FPOSE_REQUIRE_STAGE=<stage>: only objects that have a mesh in that stage (e.g. the adopted SAM 3D meshes)
          [ -n "${FPOSE_REQUIRE_STAGE:-}" ] && [ ! -f /mnt/secondary/v2d/t3/meshes/$FPOSE_REQUIRE_STAGE/$sp/$e/$obj.ply ] && continue
          $G $L/gpu_${FO}_${sp}_${e}_${obj}.log -- env PYTHONPATH=/mnt/secondary/v2d/video_to_data/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose \
            $PY -I $T/t3_fpose_track.py --ep_dir $d --obj $obj \
            --depth_dir /mnt/secondary/v2d/t3/depth/$sp/$e/depth_cam_a_s0.5 \
            --mask_dir /mnt/secondary/v2d/t3/masks/$sp/$e/cam_a_s0.5/$obj \
            --mesh $M.ply --stage1_json $M.json --vo $V \
            --out /mnt/secondary/v2d/t3/$FO/$sp/$e/$obj.npz $FPOSE_EXTRA || echo "fpose FAILED $sp/$e/$obj"
        done
      done
    done ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac
