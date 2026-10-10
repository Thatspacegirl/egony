# V2D Track 3: perception preprocessing and Docker-free GPU runtime

Data: HF `nvidia/video_to_data_challenge` at the **pinned revision 5f68335f3acc802033d1e80728c1633197521de8**, in `~/TestingGrounds/egony/video_to_data_challenge/track_3`.
Toolkit: `nvidia-isaac/video_to_data` @ a709404e, in `/mnt/secondary/v2d/video_to_data`.
Everything in this directory has been run CPU-only unless marked GPU.

## 1. Preprocessing (done for all 40 episodes)

`t3_preprocess.py` decodes the three ego streams of every public and evaluation episode. It then:

- rectifies the mono stereo pair with the verified convention: left = `ego_cam_c`, right = `ego_cam_b`, `stereo_pairs.left_to_right`, `cv2.stereoRectify(alpha=0)`;
- applies a **per-episode vertical refinement** to the right map, `dy ~ a + b·x + c·y` fitted from SIFT matches;
- undistorts `ego_cam_a` to a square-pixel pinhole (f=1040 px, calibration principal point, 100% valid pixels, 88.6° HFOV).

```
OUT=/mnt/secondary/v2d/t3/frames
OMP_NUM_THREADS=2 nice -n 10 /mnt/secondary/v2d/venv-kit/bin/python -I t3_preprocess.py \
   --ds ~/TestingGrounds/egony/video_to_data_challenge/track_3 --kit /mnt/secondary/v2d/kit/v2d_submission_kit \
   --out $OUT --workers 3            # 40 episodes in ~6 min, 12 GB of JPEG q95
```

Per-episode layout `OUT/{public,evaluation}/episode_XXXXXX/`:

| path | content |
|---|---|
| `left/000000.jpg` | rectified ego_cam_c (stereo LEFT), 1280×800 gray, `K_rect` (fx=fy=477.49, cx=612.24, cy=391.18) |
| `right/000000.jpg` | rectified ego_cam_b (stereo RIGHT), same K, baseline 0.07547 m |
| `cam_a/000000.jpg` | ego_cam_a colour, undistorted pinhole 2028×1520, `Ka_undist` (f=1040, cx=994.80, cy=767.54) |
| `meta.json` | n_frames, timestamps, objects (slot order = scorer slot order), all K's, `T_a_rect`, `T_a_c`, `T_rect_c`, `T_b_c`, full calibration, right-map dy correction, verification report |
| `fs_calibration.json` | `{fx,fy,cx,cy,baseline}` for FoundationStereo |
| `cam_a_intrinsics.json`, `left_intrinsics.json` | v2d `CameraIntrinsics` JSON (FoundationPose / BundleSDF CLIs) |
| `cam_K.txt`, `left_K.txt` | 3×3 K |

Transforms follow `T_dst_src` (X_dst = T·X_src, metres). `rect` is the rectified-left camera frame, which is also the depth frame, so `X_a = T_a_rect · X_rect`.

**Verification** (`OUT/preprocess_report.json`):

- Frame counts: all 40 OK. Public: decoded = ffprobe = parquet rows (5,246 frames). Eval: decoded = ffprobe = `track_3_sample_submission.parquet` (5,181 frames), all three streams equal.
- Rectification: median |dy| of SIFT matches with the factory calibration is 0.88-2.16 px (episode-dependent: ~1 px or ~2 px sessions). After the per-episode correction it is **0.32-0.41 px** (mean 0.35); 80-86% of matches have |dy| < 1 px and ≥98.4% have positive disparity.
- cam_a transform: `t3_check_cam_a.py` warps SGBM stereo points into cam_a with `T_a_rect`. NCC is 0.909-0.959, against 0.69-0.78 for identity and 0.57-0.67 for the inverse, so the exported transform is correct.
- Independent re-check (2026-10-08 10:30, `t3_verify_frames.py`, reads only the files on disk): all 40 episodes have
  #left = #right = #cam_a = meta.n_frames = parquet rows (public) / sample-submission frames (eval) = ffprobe packets of
  all three streams (public 5,246 frames, eval 5,181); first/last JPEGs decode at 1280x800 / 2028x1520; fresh SIFT
  median |dy| on 3 frames per episode is 0.327-0.453 px (mean 0.365). Log: `/mnt/secondary/v2d/t3/logs/verify_frames.log`.

  ```
  python -I t3_verify_frames.py --ds $DS --kit $KIT [--fix_revision]   # exit code 1 if any episode fails
  ```
- `t3_depth_warp.py --test_sgbm` gives a rect-depth to cam_a depth overlay (`/mnt/secondary/v2d/t3/checks/depth_warp_sgbm_*.png`) whose edges align with the image.

## 2. GPU perception runtime without Docker (host venvs on /mnt/secondary)

Reproduce everything with `envs/setup_envs.sh`; `envs/build_fpose_exts.sh` builds the CUDA extensions.

| env | purpose | key pins | CPU-only verification |
|---|---|---|---|
| `envs/t3-geom` (py3.12) | (a) FoundationStereo; (d) cuVSLAM VO | tensorrt-cu12 10.7.0.post1, cuda-python 12.6, onnxruntime-gpu 1.22 (TRT/CUDA/CPU EPs), cuvslam 17.0.0+cu12, v2d_foundation_stereo lib | imports OK (`source envs/t3-geom.env`); ONNX = opset 17 standard ops only, I/O 1×3×576×960 → disparity |
| `envs/t3-seg` (py3.12) | (b) masks | torch 2.8.0+cu128, transformers 5.19 (GroundingDINO, SAM3), sam2 @ 2b90b9f (pure-python), kernels | GDINO-base + SAM2.1-L on a real frame (2.8 GB RSS); SAM3 image (4.7 GB) and SAM3 **video** API (3-frame smoke) |
| `envs/t3-fpose` (py3.11) | (c) FoundationPose; (e) TSDF | torch 2.5.1+cu124 (same as the module's Docker base), kaolin 0.18.0, nvdiffrast 0.4.0, pytorch3d 0.7.9, mycpp, mycuda (sm_86), open3d 0.19, tensorrt 10.7 | extensions built with conda nvcc 12.4.131 (`envs/cuda124-toolchain`); TSDF self-test |

Weights (`/mnt/secondary/v2d/weights/`):

- `foundationstereo/deployable_foundationstereo_small_576x960_v2.0.onnx` (909,161,154 B, public NGC TAO)
- `foundationpose/nvlabs_pytorch/{2023-10-28-18-33-37,2024-01-11-20-02-45}` (NVLabs PyTorch refiner and scorer, public Google Drive)
- `sam2/sam2.1-hiera-{large,base-plus}`
- `grounding_dino/grounding-dino-{base,tiny}`
- `sam3/sam3` (gated, already granted to this account)
- `bundlesdf/roma/{roma_outdoor,dinov2_vitl14_pretrain}.pth`
- `cuvslam_wheels/`

Not downloaded: the TAO *commercial* FoundationPose ONNX (`--accept_nvidia_model_eula` needs the user to accept the NVIDIA Open Model License). Use `--backend nvlabs_pytorch` instead.

Decisions:

- (a) **FoundationStereo**: TensorRT 10.7 from pip. The pip wheel has no `trtexec`, so `t3_stereo_depth.py` builds the engine with the TRT Python API under the versioned filename the module expects. ORT-CUDA is the fallback (`--backend ort`). The module's own `image_list_to_depth` crashes on grayscale JPEGs, which is why our runner exists. A CPU forward pass needs >12 GB RSS (stopped by the watchdog), so the numerical check is GPU job 1.
- (b) **SAM3 first, GroundingDINO + SAM2.1 as fallback.** On eval ep 3 frame 0, SAM3 segments the white pot *including its handle*; GDINO's box clips the handle, so SAM2's mask misses it. Prompt map: `OBJECT_PROMPTS` in `t3_masks_sam3.py` (the two wooden pieces share a prompt and are split left-to-right on first appearance; verify).
- (c) **FoundationPose (NVLabs PyTorch backend)**, using the module's `run_video_to_poses` CLI on cam_a with `--target_width 1014 --target_height 760`, `depth_cam_a_s0.5` depth and SAM3 masks. All extensions were compiled for sm_86 on the host; no Docker needed.
- (d) **cuVSLAM 17 pip wheel** (`t3_vo_cuvslam.py`). It requires a GPU (`use_gpu=False` → `ValueError: cfg.use_gpu must be enabled`). The rig is the rectified pair (`rectified_stereo_camera=True`), and hand/object masks can be passed. ORB-SLAM3 is not needed. Cross-check: the static dish rack / table plane must stay fixed in the VO world.
- (e) **Masked TSDF fusion** (`t3_tsdf_fuse.py`, Open3D, CPU): stage 1 uses VO poses on the static pre-grasp window; stage 2 uses FoundationPose object poses on the whole clip. Fragments are cleaned so they do not shrink the CD-O scale. **BundleSDF was not built**: it needs OpenCV-CUDA + PCL 1.10 + yaml-cpp C++ builds. Estimated Docker image is ~18-22 GB (cuda:12.4.1-devel ~7 GB + OpenCV-CUDA/PCL ~3 GB + torch 2.6/kaolin/pytorch3d ~6-8 GB). SAM3D-objects is still gated (403).

## 3. Local dev scorer

`t3_devscore.py`, `devscore.py` and `make_gt_pred.py` are copied from the Phase-1 scratch. They use the official `metric_code/track_3/*.py`, unmodified.

```
KIT=/mnt/secondary/v2d/kit/v2d_submission_kit; DS=~/TestingGrounds/egony/video_to_data_challenge/track_3
python -I make_gt_pred.py $KIT $DS /mnt/secondary/v2d/t3/devscore/pred_gt_identity --identity   # GT vs GT
python -I t3_devscore.py $KIT $DS --pred /mnt/secondary/v2d/t3/devscore/pred_gt_identity --cdo
#   AUC=1.0000 SP-SR=1.0000 MP-SR=1.0000 RPE=0.0000 MPPE=0.0000 CD-O=0.0000   (all 20 public episodes, 7m20s)
python -I make_gt_pred.py $KIT $DS /mnt/secondary/v2d/t3/devscore/pred_gt_random                # random frames + 1.15x mesh
#   same perfect scores
# negative control (2026-10-08 re-run, eps 11+12): GT + N(0, 3 cm) position noise
#   GT vs GT:  AUC=1.0000 SP-SR=1.0000 MP-SR=1.0000 RPE=0.0000 MPPE=0.0000 CD-O=0.0000   (65 s)
#   GT+noise:  AUC=0.6799 SP-SR=1.0000 MP-SR=0.0000 RPE=3.2525 MPPE=3.5647
```

Depth PNG convention (all our tools): v2d `DepthImage`, `pixel = 65535/(depth_m+1)`, **no data = 65535** (decodes to
0 m). Pixel 0 would decode to +inf in `DepthImage.from_array` and pass FoundationPose's `depth > 0.001` validity
tests. `t3_depth_warp.encode_inv_depth` wrote holes as 0 until 2026-10-08 10:35; this is fixed and round-trip tested
against `DepthImage.load`. No depth had been produced with the old encoder yet.

## 4. GPU queue (`gpu_queue.sh <stage>`, guarded, logs in `/mnt/secondary/v2d/t3/logs/`)

Stages, in order: `fs_smoke`, `fs_all`, `vo`, `masks_cam_a`, `masks_left`, `vo_masked`, `warp` (CPU), `tsdf1` (CPU),
`fpose EP...`. The script refuses to start if available RAM is below 10 GB.

- `tsdf1` runs `t3_tsdf_fuse.py --frames auto --hand_dir .../left_s1/hand`. It picks the longest no-hand-contact run
  (dilated hand mask 20 px vs object mask) touching the clip start (pre-grasp) or end (post-release), stride 2, ≤60
  views, so the object is static in the VO world. Selection and fusion are CPU-tested on synthetic hand masks plus SGBM
  depth (ep 12 blue cup gives an 8×9×5 cm partial shell from one view).
- `fpose` picks the reference frame automatically: the first frame whose mask area is ≥50% of the clip's 95th-percentile
  area, so a frame-0 occlusion is skipped. `run_video_to_poses` tracks forward and backward from there. Extra flags
  go through `FPOSE_EXTRA` and are tuned on public episodes only.

The module CLI `v2d.foundation_pose.lib.run_video_to_poses` takes `--video_path cam_a/%06d.jpg` directly. OpenCV image-sequence seeking (FFMPEG backend, t3-fpose opencv 4.11) was checked to be frame-exact on eval ep 3 and ep 37 (seek 0/5/77/149/150: each best-matching frame is the requested one). Pixels differ from `cv2.imread` by a mean of 0.75 grey levels because of FFmpeg MJPEG chroma upsampling.

1. FoundationStereo engine + depth (smoke test on ep 12, compare with SGBM, then all 40)
2. cuVSLAM VO, all 40
3. SAM3 masks on cam_a s0.5 (+hands), then on the left grid (for VO masks and TSDF)
4. Depth → cam_a warp (CPU) and stage-1 TSDF meshes (CPU)
5. FoundationPose track → stage-2 TSDF → re-track + EKF
6. Score public episodes with `t3_devscore.py`

## 5. Policy side: perception bundle → FlashCHORD floating-Sharpa task (CPU-verified)

`to_sharpa_task.py` turns one per-episode perception bundle (`t3_perception_v1` .npz: object world poses per video
frame in roster order, our meshes, optional MANO/keypoint hands, optional table plane) into a FlashCHORD task:
ManoSharpaData parquet at fps 20 with T = video frames, one URDF per object (visual = collision = the budgeted mesh we
submit, plausible mass, COM at the hull centroid), a Cube support USDA at the table plane, then Sharpa IK with
`scripts/retarget/ego_recon_to_sharpa.py`. Train with `task.motion_speed=1.0`. Details, contract checks, dev results
and the 3090 preset/launchers: `sharpa_task/README.md`.

```bash
/mnt/secondary/v2d/envs/retarget/bin/python to_sharpa_task.py BUNDLE.npz --out-root /mnt/secondary/v2d/t3/tasks/<variant> --verify
```

## 6. Hands without MANO (`hands/`, CPU only; details and all numbers: `hands/README.md`)

MANO pickles are registration-gated, so HaMeR cannot run. The 21-keypoint hands come from MediaPipe HandLandmarker
(Apache-2.0) on undistorted cam_a: forward and backward VIDEO passes, tracklet-level left/right voting, and stereo
depth (FoundationStereo when an episode's depth is complete, else SGBM). The relative depth is the hybrid of dense
per-joint depth and the PnP world-landmark shape. A sparse robust temporal optimisation then applies bone/palm
lengths, smoothness, outlier rejection and filling of gaps up to 0.4 s. Output for all 40 episodes:
`/mnt/secondary/v2d/t3/hands/{public,evaluation}/episode_X/hands_rect.npz`, holding left/right (T,21,3) in the
rectified-left frame plus valid/observed/conf/depth_src.

```bash
bash hands/run_hands.sh all                                    # detect -> sample depth -> lift (frozen settings)
/mnt/secondary/v2d/envs/t3-hands/bin/python -I hands/hands_to_bundle.py --bundle IN.npz \
   --hands /mnt/secondary/v2d/t3/hands/<split>/episode_X/hands_rect.npz --poses /mnt/secondary/v2d/t3/vo/<split>/episode_X/vo_cuvslam.npz --out OUT.npz
```

- `hands_to_bundle.py` writes `hand_{left,right}_joints`/`_valid`, which `to_sharpa_task.py` turns into
  `bundle_keypoints+kabsch_template`. With `--fill object`, a hand that is out of view while carrying a moving object
  rides with that object.
- There is no hand GT in rev 5f68335, so the evaluation uses proxies on the public split:
  - an independent RTMPose triangulation on the rect pair: depth MAD 1.10 cm, bias -0.27 cm;
  - GT objects, using a camera pose from ICP of the public scans against stereo, verified on 5 episodes: the closest hand joint is a median 2.0 mm from the moving object's surface, against 6.0 mm for the DEV placeholder.
- End to end on public eps 12 and 22, `to_sharpa_task.py --verify` passes 15/15 with IK error 2.9-4.0 cm.


## 7. Object perception run on the GPU (2026-10-08) -- status, findings, output format

Everything below was run under the shared GPU lock (`flock /mnt/secondary/so101_r2s/gpu.lock bash gpu_queue.sh <stage>`).
CPU steps per episode: `bash cpu_chain.sh EP...` (mask clean-up -> cam_a/stereo sync -> lag-aware warp -> mask transfer
to the left grid -> stage-1 TSDF), then `bash complete_all.sh EP...`, then `gpu_queue.sh fpose EP...`, then
`t3_assemble.py`.  Logs: `/mnt/secondary/v2d/t3/logs/{gpu,cpu}_*.log`; visual checks: `/mnt/secondary/v2d/t3/checks/`.

### 7.1 FoundationStereo (stage fs_smoke / fs_all)
- The TensorRT 10.7 build of the TAO FP32 ONNX needs a **~14.4 GB builder workspace** (cost-volume Myelin node; with
  the old 6 GB limit: `Could not find any implementation for node {ForeignNode[/cost_agg/...]}`) -> `--workspace_gb 18`
  (default now).  Build: 833 s, ~10.5 GB host RSS (one-off).  Engine:
  `/mnt/secondary/v2d/weights/foundationstereo/deployable_foundationstereo_small_576x960_v2.0_10_7_0_post1_sm_8_6.engine`.
- Speed: **1.0-1.06 s/frame** (pure inference; reading/writing is overlapped by a thread pool) -> ~2.9 GPU-hours for
  all 10,427 frames, run in 4-5 chunks of <= 45 min.  An FP16 build was stopped after 27 min at 13.7 GB RSS (and TRT
  warned about FP16 layer-norm overflow); not used.
- vs OpenCV SGBM on the same rectified pair (`t3_depth_check.py`, public 12 and eval 1, 4 frames each): median
  FS/SGBM depth ratio 0.996-0.998, 78-83 % of pixels within 5 % (SGBM is the noisy one; `checks/fs_vs_sgbm_*.png`).
- **Metric scale (dev, public only)**: `t3_scale_check.py` (wooden boards, public 23): inter-board distance
  27.67 cm vs GT 27.89 cm (ratio 0.998); board width 13.91 / 13.72 cm vs scan 13.87 cm (1.003 / 0.989).  Pitcher
  stage-1 mesh major axis 41.3 cm vs scan 41.3 cm.  => no depth-scale correction (bias < 1 %).

### 7.2 cuVSLAM VO (stage vo), all 40 episodes
- 100 % of frames tracked in all 40 episodes, 104-147 fps.  `t3_vo_check.py`: no jumps (max step 0.20-1.21 cm and
  0.49-2.74 deg per frame; 0 frames > 3 cm / 5 deg).  Head motion per episode: 1.6-8.7 cm, 3.4-25 deg.
- Pose convention verified by depth reprojection (public 21): `inv(W[f]) @ W[0]` maps frame-0 depth onto frame f with
  4.8-6.6 mm median |dz| (vs 14.8 mm identity, 32.5 mm for the inverse convention).
- Static-world check: the table plane (RANSAC on FS depth < 1 m, objects/hands masked) mapped into the VO world on eval
  1 (112 frames): offset std 1.8 mm, spread 11 mm, normal spread 3.0 deg while the head moves -> VO metric and
  drift-free at the mm level.  The public parquet has NO camera/device trajectory (only object poses), so there is no
  direct GT comparison for VO.
- `vo_masked` (cuVSLAM with object+hand masks) was not run: the unmasked VO already passes the checks, and switching
  the VO world would require re-running tsdf1 + fpose (`gpu_queue.sh` now warns about this).

### 7.3 cam_a <-> stereo time offset (NEW finding, affects every consumer of cam_a + depth/VO)
The three ego streams have equal frame counts, but **cam_a frame i+k shows the scene of stereo frame i, k = 1 or 2**
(found because SAM3 cam_a masks transferred into the left view were shifted during head motion).
`t3_sync.py` warps the LEFT image into cam_a with the stereo depth and finds the k in [-2, 5] maximising gradient NCC
against cam_a frames, per frame with head rotation > 0.4 deg/frame, then a Viterbi smoother gives a piecewise-constant
per-frame lag.  Public results: k = 1 for 0, 2, 11, 12, 13, 18, 23, 24, 26 and eval 1; k = 2 for 21; 22 switches
2 -> 1 at frame 63 (a dropped frame in one stream).  Typical NCC curve (public 21): k=0 0.278, k=1 0.374, **k=2 0.484**,
k=3 0.344.  Output: `/mnt/secondary/v2d/t3/sync/<split>/episode_X.json` (`lag_per_frame`, `changes`, NCC curves).
Used by `t3_depth_warp.py` (depth_cam_a frame j = warp of stereo frame j - lag), `t3_masks_to_left.py`,
`t3_tsdf_fuse.py` (vertex colours), `t3_fpose_track.py` (VO at the right instant) and `t3_assemble.py`.
The hands pipeline (MediaPipe on cam_a + stereo depth) should apply the same lag.

### 7.4 SAM3 masks (stage masks_cam_a) + CPU transfer to the left grid
- Prompt map chosen from `t3_prompt_check.py` overlays (13 episodes x 3 frames, `checks/prompts/`): white_pot
  "white pot" (keeps the handle on eval 3 f0/f102/f205), white_pot_lid "pot lid", wooden_spoon "wooden spoon",
  blue_cup/beige_cup "blue cup"/"beige cup", plastic_dish_rack "white dish rack", water_pitcher **"white jug"**
  ("watering can" missed it on public 21 f0), mini_sweeper "hand brush", mini_dust_pan **"yellow dustpan"**,
  wooden_piece_1/2 "wooden block".
- **wooden_piece_1 vs _2**: the two public scans differ only by their mocap balls (wooden_piece_1 has 4 balls,
  wooden_piece_2 has 3, `checks/overview/wooden_scans.png`).  In public 23, 24 and eval 25 frame 0 the 4-ball board is
  the LEFT one (`checks/overview/wooden_crops.jpg`), so the two "wooden block" instances are assigned left -> right
  on the first frame that shows both.
- Memory fix: `init_video_session(video=frames)` preprocesses the whole clip in float32 (12 MB/frame) -> 10 GB host
  RSS abort on a 325-frame clip / CUDA OOM on the GPU.  Now frames are preprocessed in chunks of 16 and stored as bf16
  on the CPU via the session's `add_new_frame` (~2.4 frames/s, 5.5 GB RSS peak).
- Hand mask = union of all "hand" instances.  Lost instances are re-acquired from new ids of the same prompt near the
  last position (`switches` in masks.json).  `t3_mask_clean.py` then removes far-away blobs SAM3 attaches to a track
  (public 23: wooden_piece_1 picked up a chair at the image border after the boards were slotted) -- originals in
  `<slot>_raw/`.
- Left-grid masks are NOT a second SAM3 run: `t3_masks_to_left.py` transfers the cam_a masks through the stereo depth
  (left point -> cam_a pixel of frame i+lag, accepted if it lies on the surface cam_a sees, |dz| < 3 cm), 7x7 close.
  Same instance ids in both views.

### 7.5 Stage-1 meshes, completion, cross-episode fusion (CPU)
- `t3_tsdf_fuse.py --frames auto`: hand-free window (dilated hand mask 20 px vs object mask on the left grid);
  the pre-grasp window is preferred whenever it has >= 8 frames (after release objects are often stacked, slotted or
  inside another object).  If both a pre- and a post-release window exist, each is fused separately and ICP'd
  (`windows_agree`): fitness >= 0.5 and a correction < 3 cm / 5 deg -> the post views are registered onto the pre
  reconstruction and both are fused (objects that never move, e.g. the dish rack; public racks 12/13/26:
  CD-O 0.50/1.02/2.44 -> 0.55/0.58/0.58 after mirroring).
- Masked depth is cleaned per frame (`clean_mask_depth`: erosion, 5x5 depth-discontinuity rejection, blobs at a
  consistent depth) -- without it background seen past the cup rim fused into a 30 cm sheet.
- Body frame = bounding-box centre at the origin; `<obj>.json` stores `frame_T_obj` (VO-world pose of the body during
  the window), the window stereo frames and the VO file.  Vertex colours from cam_a (lag-aware, depth-tested).
- Completion (`t3_mesh_complete.py`, policy in `complete_all.sh`, tuned on PUBLIC CD-O and frozen):
  `revolve` for cups (axis = table normal through a fitted circle, 70th-pct radius profile, inner wall, bottom),
  `revolve_keep` for the pot (revolved body + the observed faces outside it: handle, balls; public pots
  0/2/26/27/41: mirror mean 1.46 -> 1.14), `mirror` (vertical bilateral-symmetry plane: azimuth x offset search maximising self-overlap minus a penalty for
  mirrored points landing in observed free space) for lid, rack, pitcher, sweeper, dust pan, and
  `mirror_extrude` (mirror, then close the underside down to the table) for wooden boards and the spoon.  Output is
  decimated to <= 4000 faces so the packer's `budget_mesh(4096, 4096)` is a no-op (an un-decimated extrusion made
  fast_simplification stall: "mesh simplification cannot satisfy the budget").
  Public CD-O, stage-1 -> completed: cups 1.41-1.83 -> 0.42-0.61, boards 0.31-0.33 -> 0.20-0.21, sweeper 0.42 -> 0.35,
  dust pan 0.38 -> 0.25, pitcher 0.97/0.94 -> 0.73/1.05, pot 1.81/1.06 -> 1.60/1.07, lid 0.28 -> 0.27.
- `t3_xep_fuse.py` (cross-episode fusion of one object within ONE split: up-constrained yaw search + point-to-plane
  ICP, one TSDF over all accepted episodes, per-episode `frame_T_obj` re-expressed in the shared frame) +
  `t3_xep_complete.py` (complete the fused mesh once, share it) -> `final_meshes.sh SPLIT EP...` builds
  `meshes/final/<split>/episode_X/<obj>.{ply,json}`: fused for cups (revolve; public blue cup x5: 0.42-0.61 -> 0.30),
  sweeper (0.35/0.87 -> 0.32), pitcher (0.76/1.05 -> 0.83), rack (0.50-0.63 -> 0.55-0.57); per-episode for the rest
  (fusion made the dust pan worse, 0.21-0.26 -> 0.31; the pot handle's yaw is ambiguous in the fusion -> two handles).
  Fusion never mixes splits (evaluation meshes come only from evaluation videos).
- `t3_mesh_eval.py --root MESH_ROOT` = official CD-O core vs the public scans (dev only); `t3_mesh_viz.py` = 3-view
  splat images (`checks/mesh_*.png`).

### 7.6 FoundationPose tracking (stage fpose, GPU) -- `t3_fpose_track.py`
- NVLabs PyTorch backend via the module's `FoundationPoseTracker`, undistorted cam_a at 0.5 scale (1014x760, K/2)
  with the lag-aware `depth_cam_a_s0.5` and the cleaned SAM3 masks; ~7-9 frames/s.
- Mesh search order `FPOSE_STAGE="stage1c stage1"` (completed mesh first).  **No global registration**: the
  reference pose is the stage-1/completed `frame_T_obj` (VO world, static window) composed with VO at the matching
  stereo instant, refined with `track_one(iteration=10)` on the window frame with the largest mask; then tracked
  forward and backward (`track_one`, 5 iterations).  Output `fpose/<split>/episode_X/<obj>.npz`: `cam_T_obj[T,4,4]`
  (cam_a frame, mesh coordinates), `iou[T]` (rendered mesh vs SAM3 mask), `mask_px[T]`, `ref_frame`, `init`, `mesh`.
- Meshes must be bbox-centred: the module's `_mask_iou` renders the CENTRED mesh with the caller-frame pose, so an
  off-centre mesh gets a biased IoU (public 21 revolved cup: constant 0.54 before re-centring).

### 7.7 Per-episode perception output (for the bundle assembler) -- `t3_assemble.py`
`/mnt/secondary/v2d/t3/perception/<split>/episode_X/perception.npz` (+ `perception.json` summary):

| key | shape | meaning |
|---|---|---|
| `episode_index`, `num_frames`, `split` | | |
| `object_names` | (B,) | roster / scorer slot order (meta.json `objects` = `track_3_evaluation_objects.json` for eval) |
| `object_mesh_paths` | (B,) | the mesh the poses refer to (`meshes/stage1c/...ply`, bbox-centred, <= 4000 faces, vertex colours) |
| `object_pose` | (N,B,7) | world_T_object `[x,y,z,qw,qx,qy,qz]`, world = VO world (rect-left camera at stereo frame 0), metres |
| `object_valid` | (N,B) | VO and object pose available (gaps are interpolated / held, never NaN) |
| `object_conf` | (N,B) | FoundationPose rendered-vs-mask IoU where the measurement was trusted, else NaN |
| `table_plane`, `up` | (4,), (3,) | `[nx,ny,nz,d]`, n.x + d = 0, n up (towards the camera), from frame-0 depth (< 1 m, objects/hands masked) |
| `world_T_rect` | (N,4,4) | VO, stereo time base |
| `world_T_cam_a`, `cam_a_T_obj` | (N,4,4), (N,B,4,4) | cam_a time base |
| `a_lag`, `time_base` | (N,), str | per-frame cam_a lag (t3_sync.py); which stream frame index i of `object_pose` refers to |

Same keys as `sharpa_task/bundle.py` (`t3_perception_v1`) minus `schema`/hands, so assembling a bundle is a copy:
`object_names`, `object_mesh_paths`, `object_pose`, `object_valid`, `table_plane`, `up`.
Options (frozen values in 7.8): `--clamp_static` (one robust pose per hand-free run: frame 0 and the rest periods
become exact constants), `--smooth SIGMA` (IoU-weighted Gaussian smoothing), `--time_base cam_a|stereo|shift<k>`,
`--min_mask_px`, `--min_iou` (measurements below are dropped and interpolated).  `--devscore DIR` also writes the
t3_devscore.py layout (PUBLIC only); `t3_dev_eval.sh EP...` = assemble + official scorer; `t3_dev_diag.py` = per-slot
error breakdown (scorer-aligned vs oracle-aligned error, registration rotation, displacement).

### 7.8 Public dev scores, 2026-10-09 (resumed run; GPU jobs via `/mnt/secondary/v2d/bin/gpu_hi.sh`, queues in `jobs/`)
FoundationPose on public chunk A (eps 0 2 11 12 13 18 21 22 23 24 26; final meshes; run 10-08 16:35-16:46, all 22
npz validated: finite, full length) scored with the official metric code (`PER_EP=1 t3_dev_eval.sh`):

| assembly | AUC | SP-SR | MP-SR | RPE | MPPE | CD-O |
|---|---|---|---|---|---|---|
| base (cam_a time base) | 0.6001 | 0.6364 | 0.4545 | 6.908 | 9.266 | 0.5926 |
| `--clamp_static` | 0.5992 | 0.6364 | 0.4545 | 6.958 | 9.253 | |
| `--clamp_static --smooth 1.5` | 0.6014 | 0.6364 | 0.4545 | 6.919 | 9.236 | |
| `--smooth 1.5` | 0.6048 | 0.6364 | 0.4545 | 6.860 | 9.198 | |
| `--clamp_static --time_base stereo` | **0.6072** | 0.6364 | **0.5455** | **6.761** | **9.175** | |

`--time_base stereo` is better in 9/11 episodes (e.g. 13: AUC 0.926 -> 0.945, RPE 1.64 -> 1.14) -> the GT follows the
stereo clock; it is now the `t3_assemble.py` default.  CD-O of the final meshes over all 20 public episodes (33
pairs): **0.658 cm** (`checks/mesh_eval_final_public_1009.json`; pots 0.67-1.53, ep 24 wooden_piece_1 2.67).

Per episode the score is bimodal: 12/13/18/22 have AUC 0.88-0.98, while 11 (0.11), 23 (0.04), 24 (0.03) and 26 (0.50)
are catastrophic.  `t3_reg_check.py` (new, DEV) recovers the TRUE mesh->scan body rotation from the trajectories
(world alignment by Umeyama on the surface-centroid tracks of all objects, then Q = mean R_o^T R_W^T R_g) and compares
it with the rotation the official scorer's registration picks for our mesh:

- **boards 23/24 and the pot in 26: the scorer's registration of our mesh is flipped by ~180 deg** (reg_err 118-180 deg).
  Because the episode alignment uses the frame-0 orientation of the first non-symmetric object, a flipped anchor
  rotates the whole scene (MPPE 31-33 cm).  For the boards the chamfer at the true and the flipped alignment is
  identical to 0.001 cm (our board mesh is symmetric), i.e. a coin flip.  Even the scan itself only loses 0.16-0.21 cm
  of CD under a 180-deg flip (pot 1.7-2.7, rack 1.0-1.8, dust pan 0.8-1.5, sweeper 0.16 about one axis), so boards
  can only register reliably if the slot, the notches and the one-sided mocap balls are reconstructed.  The pot in 26
  prefers the flip because of its shape error (CD 1.207 flipped vs 1.249 true).
- ep 11 blue cup: not a registration problem but a tracking failure (the cup is static in GT; our track starts at the
  post-release reference frame 133 where the beige cup is nested in it, then drifts 14 cm during the occlusion).
- anchor registration errors of 8-14 deg (pot in 0/2) displace the other objects by 2 sin(err/2) x distance (lid f0 3 cm).

## 8. Generated meshes (SAM 3D Objects), board meshes, candidate selection (2026-10-09)

### 8.1 SAM 3D Objects host env (`envs/build_sam3do_env.sh`, env `/mnt/secondary/v2d/envs/t3-sam3do`)
Copy of t3-fpose (torch 2.5.1+cu124, kaolin 0.18, pytorch3d 0.7.9, nvdiffrast) + the inference subset of the
toolkit's `modules/v2d_sam3d/docker/Dockerfile`: gsplat 1.4.0 (pt25cu124 wheel), spconv/cumm cu124, lightning 2.3.3,
MoGe @ a8c3734 with its utils3d pin 3913c65, `setuptools<70` (lightning needs pkg_resources), sam-3d-objects
(`envs/src/sam-3d-objects` @ f91db411, `--no-deps`), v2d_common + v2d_sam3d lib editable.  No flash-attn (SDPA on the
3090).  `source /mnt/secondary/v2d/envs/t3-sam3do/v2d_env.sh` (TORCH_HOME = `/mnt/secondary/v2d/t3/sam3do/torch_home`:
DINOv2 hub code @ 7764ea0f + links to the reg4 checkpoints).  Weights: `/mnt/secondary/v2d/weights/sam3d_objects`
(downloaded by the Track 1 agent with `t1_download_weights.py`, layout of `v2d_sam3d/lib/download_weights.py`).
Licences: SAM 3D Objects checkpoints + code = **SAM License** (Meta, 2025-11-19, gated facebook/sam-3d-objects);
MoGe v1 ViT-L weights (Ruicheng/moge-vitl) = MIT; DINOv2 = Apache-2.0.
On the 3090: model load 62 s (13.7 GB VRAM resident), **14-22 s per object, peak 18.5-19.3 GB VRAM**.

`t3_sam3do.py --jobs J.json` (one model load per job list): full-resolution undistorted cam_a frame + the SAM3 mask
(alpha), conditioning point map = OUR FoundationStereo depth (`depth_cam_a_s0.5`, eroded object mask, PyTorch3D camera
convention via the toolkit's `_depth_to_pointmap`) instead of MoGe.  Output `sam3do/<run>/<split>/episode_X/<obj>/
f<frame>_s<seed>.{glb,json,ply}` (.ply = metric cam_a coordinates of that frame).  The SSI layout is only approximate:
the depth check in the JSON shows the mesh 1-5 cm in front of the observed surface and 10-25 % too small
(t3_mesh_select.py fixes the placement).  `t3_sam3do_jobs.py`: K=3 frames per object from the stage-1 static window
(largest unoccluded mask + spread), pre-grasp part only for merged windows.

Smoke test (public, 1 frame each) CD-O vs scan: pot ep0 **0.863** (TSDF final 1.324), cup ep21 0.662 (0.291: wrong
height/diameter from a top-down view), pitcher ep21 1.222 (0.707), board ep23 0.196 (0.189).  The SAM3D pot has the
long handle that every single-pot TSDF mesh (public 41/43/44/45: 18 x 15 x 15 cm) lost, and its scorer registration
has a **0.31 cm basin margin** (next-best basin > 30 deg away) vs 0.02 cm for the TSDF pot (`t3_reg_margin.py`).

### 8.2 Board meshes (`t3_board_mesh.py`, stage `meshes/board/`)
For `wooden_piece_*` lying flat in the static window: table plane + board top plane (RANSAC), footprint on a 1 mm grid
of the top plane = majority vote of the SAM3 mask back-projected from every window view (I-shaped notches come out),
through-slot = thin dark black-hat structure of the full-res cam_a image voted over views (kept if >= 3 cm long and
<= 1.2 cm wide), marker balls = clusters of board points >= 5 mm above the top (stereo flattens the ~14 mm balls to
6-14 mm bumps, so only their x/y is measured; modelled as r = 7 mm spheres centred 12 mm above the top on a stalk -
constants read off the videos), occupancy -> marching cubes, <= 4000 faces.  If the stage-1 window is not pre-grasp
(public 24 wooden_piece_1: a hand hovers next to it from frame 0) the window is the "static prefix": frames whose board
points stay on the frame-0 board plane (< 4 mm), frames 0-4 there; the JSON then carries `frames` = that prefix so
FoundationPose initialises from it.  Public 23 (FoundationPose track unchanged, poses re-expressed with
`t3_rebody.py`): scorer registration error 179.6/179.3 deg -> **1.7/0.7 deg**, AUC 0.035 -> **0.997**, MPPE 33.1 ->
0.58 cm, RPE 20.2 -> 0.88 cm, CD-O 0.189/0.212 -> 0.158/0.151.  Seed stability (`t3_reg_seeds.py`, 10 other scorer
seeds): new wooden_piece_1 100 % / wooden_piece_2 90 % pick the same (correct) basin; the old TSDF board picked the
wrong basin with 80 % of seeds.  Public 24 needs a FoundationPose re-run (its old wooden_piece_1 track came from the
fused interlocked post-release mesh, CD-O 2.67).  Evaluation 25: both boards pre-grasp, 4 / 2 balls found.

## 9. Post-crash resume (2026-10-09, 3rd agent from 10:02 EDT) -- validation, recompute, new tools

The machine crashed at 01:44 EDT (RAM-fault GPF, soft lockups from 02:36).  Everything written 01:00-02:50 was
validated or recomputed before use (hash lists in `/mnt/secondary/v2d/t3/checks/recompute_1009/`):

- **FoundationStereo depth** (`t3_validate_depth.py`, read-only): every PNG of all 40 episodes decodes; the per-frame
  median depth recomputed from the file equals the value logged by `t3_stereo_depth.py` at write time to <= 0.08 mm
  (no post-write corruption); no single-pixel spike bursts.  It does flag a few **whole-frame FoundationStereo failures**
  (eval 36 f347 = flat ~0.5 m, f57 upper half; public 22 f18, public 0 f187/f194) -- the input images are fine and
  the same happens in the pre-crash public data, so it is FS behaviour, not RAM corruption; left as is (consistent
  between public dev and evaluation).
- **Public FoundationPose npz** (incl. 27/31/39/41-45 written 01:39-01:47): finite, full length, orthonormal, and the
  IoU median/p10 recomputed from each npz equals the value logged in-process -> kept.  Public 40 (killed) re-run.
- **Evaluation CPU chain 25-38** recomputed (`jobs/cpu_recompute_evalB.sh`, `jobs/cpu_eval_prep2.sh`): sync JSONs,
  cleaned cam_a masks, left-grid masks and lag-aware depth warps are **byte-identical** to the crash-window files;
  stage-1 / stage-1c / final meshes differ only through known nondeterminism (Open3D ICP in merged windows and the
  unseeded `trimesh.sample` in `t3_mesh_complete.mirror`), so the recomputed files replace them.  The regenerated
  SAM 3D evaluation job list is identical.  `t3_hashdirs.py ROOT OUT.json PATH...` = the hashing tool.
- **Board meshes** regenerated (`jobs/cpu_regen_boards.sh`): all 6 ply byte-identical -> `meshes/board/*/VERIFIED`.
- `t3_assemble.py` now writes `perception.npz` / `.json` atomically (tmp + rename).

New tools: `t3_policy_apply.py` (apply a label-free mesh-source rule to `t3_mesh_select.py` results of a split and
adopt the SAM 3D picks into a stage dir; boards untouched), `jobs/cpu_assemble_all.sh` (assemble + validate +
sha256 manifest `perception/<split>_manifest.json`).

### 9.1 Mesh-source policy (decided on PUBLIC 2026-10-09 14:15, applied unchanged to evaluation)
SAM 3D Objects candidates (`t3_sam3do.py`, 3 static-window frames per object, stereo point map) were generated for
all 40 episodes (public 94 + evaluation 104 jobs, 0 errors, ~14 s/job).  `t3_mesh_select.py` places every candidate
against OUR window observations (ICP + scale search, now extended past the 0.8/1.4 grid ends while the boundary keeps
winning: 23/127 candidates had hit 1.4) and scores view IoU / depth residual -- no GT.  DEV CD-O on public
(`checks/select_report_k3_public.txt`, 33 pairs): TSDF finals 0.658 cm, SAM 3D chosen by view IoU 0.659, oracle 0.495.
SAM 3D wins clearly only for the **white pot** (9/9 episodes, mean 1.121 -> 0.627 cm with rule `iou_dz`: the long
handle that single-view TSDF loses); it is worse for the lid (0.285 -> 1.093), cups (0.29 -> 0.59), dust pan,
sweeper and pitcher, equal for the rack.  Frozen policy (`meshes/select_k3/READY`):

| object | mesh source |
|---|---|
| white_pot | SAM 3D candidate with the best `iou - 0.002*dz_mm` (`t3_policy_apply.py --classes white_pot --rule iou_dz`), adopted with its fitted similarity (scale baked in) |
| wooden_piece_1/2 | `meshes/board` (t3_board_mesh.py; CD-O 0.166 vs TSDF 0.189-2.672) |
| everything else, incl. wooden_spoon (no public scan, so no evidence for a switch) | `meshes/final` (TSDF + completion / cross-episode fusion) |

**Broken-mesh fallback** (`--broken_iou 0.5`, added 15:50 after the evaluation v1 FoundationPose run lost the eval
19/30 sweepers from the first frame; label-free, applied to both splits, changes nothing on public): if the TSDF
final mesh's view IoU against our own window observations is < 0.5, use (1) the episode's own stage1c mesh when the
final is a cross-episode fusion and stage1c scores >= 0.5 and +0.15 better (eval 19 sweeper: fused 0.38 -> own 0.795;
the fused mesh was misplaced), else (2) the best SAM 3D candidate if its IoU >= 0.6 (eval 30 sweeper: TSDF 0.30 ->
SAM 3D 0.84).  Public: only ep 12 blue cup triggers the check (0.24, occluded in the rack) and neither fallback
qualifies.

FoundationPose mesh order: `FPOSE_STAGE="board select_k3 final"`.  Public mesh CD-O with this policy: **0.440 cm**.

### 9.2 Public dev scores per stage, 2026-10-09 (official metric code via `t3_devscore.py`, all 20 public episodes)
Assembly always `t3_assemble.py --clamp_static` (stereo time base).  Leader (public LB, SoySauceChicken):
AUC 0.765 / SP-SR 0.8 / MP-SR 0.5 / RPE 3.853 / MPPE 3.002 / CD-O 0.583.

| stage | pred dir (`devscore/`) | AUC | SP-SR | MP-SR | RPE | MPPE | CD-O |
|---|---|---|---|---|---|---|---|
| v1: TSDF finals, FoundationPose plain | pred_1009c_v1 | 0.5807 | 0.55 | 0.50 | 4.472 | 7.825 | 0.658 |
| + board meshes, FP re-run on 23/24 (`fpose_board`) | pred_1009c_v1b | 0.6767 | 0.65 | 0.60 | 2.418 | 4.687 | (0.578) |
| + SAM 3D pots, FP tracks re-bodied (`t3_rebody.py`, estimate) | pred_1009c_pot_rebody | 0.7182 | 0.70 | 0.60 | 2.377 | 4.360 | 0.440 |

(0.578 = mesh CD-O with boards, from `t3_mesh_eval.py`.)  Per-episode lists: `<pred>/score.txt` or `--per_episode`.
