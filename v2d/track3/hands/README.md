# Track 3 hands without MANO: 21-keypoint 3D hand trajectories for all 40 episodes

MANO pickles are not on this machine (registration-gated), so HaMeR/WiLoR cannot run. This directory builds
MANO-free 3D hand keypoints from openly licensed 2D detectors plus our stereo, in the rectified-left (depth)
camera frame, and converts them into the `t3_perception_v1` bundle's hand keys (`sharpa_task/bundle.py`). CPU only:
no GPU lock is needed. Env: `/mnt/secondary/v2d/envs/t3-hands` (py3.12: mediapipe 1.1.0, onnxruntime 1.30,
opencv 5.0, scipy, pyarrow, open3d 0.19, trimesh).

Keypoint order (21) = MediaPipe = OpenPose = the HaMeR/MANO 21-joint order used by `sharpa_task/hands.py`
(`MANO_HAND_LINKS`, tips 4/8/12/16/20): 0 wrist | 1-4 thumb | 5-8 index | 9-12 middle | 13-16 ring | 17-20 pinky.
No re-ordering is needed anywhere.

## Pipeline (`run_hands.sh [EPISODES]`, frozen settings = the script defaults)

| stage | script | output (`/mnt/secondary/v2d/t3/hands/{public,evaluation}/episode_X/`) |
|---|---|---|
| 1 detect | `detect_mp.py` | `det_mp_cam_a.npz`: MediaPipe HandLandmarker (Tasks, `hand_landmarker.task` float16, Apache-2.0, `/mnt/secondary/v2d/weights/mediapipe/`, sha256 fbc2a300...) on undistorted cam_a colour, VIDEO mode, two passes (forward and time-reversed), 2 hands, conf 0.3 |
| 2 sample | `sample_depth.py` | `depth_samples_{fs,sgbm}.npz`: rect-left depth warped to the cam_a grid at scale 0.5 (z-buffered, `t3_depth_warp.warp_depth`), 9x9 window per keypoint. FoundationStereo when the episode's `depth_rect` is COMPLETE, else SGBM (3WAY, 192 disp, LR check); never mixed within an episode |
| 3 lift | `lift_hands.py` | `hands_rect.npz` + `hands_rect.json` (stats) |
| convert | `hands_to_bundle.py` | bundle with `hand_{left,right}_joints` / `_valid` / `_conf` in the bundle world |

`lift_hands.py`:
1. **Pass merge.** The two passes are merged per frame by Hungarian matching on 2D joints, averaging the matched pairs and dropping duplicates.
2. **Tracklets and handedness.** Detections are linked into tracklets (gaps of up to 4 frames). Each tracklet gets one label, left or right, by voting: summed MediaPipe handedness score plus a weak image-side prior. Tracklets that co-occur must get different labels; the weaker one is flipped or its conflicting frames are dropped. MediaPipe's per-frame labels are wrong in some ego frames (e.g. ep 12 f20), and the track-level vote fixes them.
3. **Depth targets.**
   - Each joint's dense surface depth plus a skin offset (5-12 mm, joint centre behind the visible skin) gives a joint depth.
   - The relative depth comes from MediaPipe's metric world landmarks, PnP-fitted onto the 2D joints. The root is refitted robustly to the dense depth (median, then inliers within 2 cm).
   - `--reldepth hybrid` (default): a joint whose own dense depth agrees within 3 cm uses it directly (sigma 1 cm). The others keep the PnP shape (palm 1.5 cm, fingers 2.5 cm).
   - With no stereo support (hand outside the rectified FOV), PnP of the scaled world landmarks is used (sigma 4 cm).
   - A detection with wrist depth outside 0.12-1.0 m is dropped (bystanders, garbage).
4. **Hand scale.** s = median stereo palm size / median MediaPipe world-landmark palm size. Bone and palm-edge lengths are s x the median MediaPipe proportions, shared by both hands. Measured wrist-to-middle-MCP: median 7.9 cm on public, 7.8 cm on eval (CV 6-7 % across episodes).
5. **Temporal optimisation.** One sparse robust least-squares problem per hand segment (`scipy.optimize.least_squares` with trf, lsmr and soft-L1). Terms:
   - cam_a reprojection, sigma 8 px;
   - depth targets;
   - 20 bones (4 mm) and 5 palm edges (6 mm);
   - second differences, 8 mm.

   Segments split at gaps larger than 8 frames (0.4 s); shorter gaps are filled and marked valid but not observed. Frames with a median reprojection error above 30 px after the first solve are rejected as outliers, and the segment is re-solved.

### `hands_rect.npz` (frame = rectified-LEFT camera = depth frame, metres; `X_a = T_a_rect X_rect`)

| key | shape | meaning |
|---|---|---|
| `left`, `right` | (T,21,3) f4 | joints, NaN where not valid |
| `{side}_valid` | (T,) bool | observed or gap-filled |
| `{side}_observed` | (T,) bool | a detection was used in this frame |
| `{side}_conf` | (T,21) f4 | 0..1 heuristic: handedness score x passes x in-image x stereo support x reprojection fit; it halves every 2 frames into a gap |
| `{side}_depth_src` | (T,) i1 | 0 none, 1 stereo, 2 PnP size prior, 3 gap fill |
| `{side}_uv` | (T,21,2) f4 | final joints projected into cam_a (px) |
| `bone_lengths`, `len_edges` | (25,), (25,2) | the lengths used: 20 bones + 5 palm edges |
| `T_a_rect`, `num_frames`, `episode_index`, `split`, `provenance` | | |

### Into the perception bundle

```bash
P=/mnt/secondary/v2d/envs/t3-hands/bin/python; H=~/TestingGrounds/egony/v2d/track3/hands
$P -I $H/hands_to_bundle.py --bundle IN.npz --hands /mnt/secondary/v2d/t3/hands/evaluation/episode_X/hands_rect.npz \
   --poses /mnt/secondary/v2d/t3/vo/evaluation/episode_X/vo_cuvslam.npz --out OUT.npz
```

- The world is any npz with `world_T_rect` (T,4,4) and `valid`. That is the cuVSLAM output, or, for DEV/public only, `gt_cam_register.npz`. The script refuses to put GT-registration poses into a non-`dev_only` bundle.
- The perception bundle writer can instead call `hands_to_bundle.hands_in_world(hands_npz, world_T_rect, valid)`.
- No `joints_wxyz` is written, so `to_sharpa_task.py` Kabsch-fits the template palm (`bundle_keypoints+kabsch_template`) and linearly interpolates invalid frames, holding the ends.
- Both sides are written whenever they have a valid frame. A side with none is omitted, and the converter's `--hands auto` would then use the DEV placeholder for it. Use `--hands input` for eval bundles to forbid that.

## Evaluation on the PUBLIC split (dev only)

**There is no hand ground truth in dataset revision 5f68335.** The public parquets hold only `observation.objects`
(world_T_object) plus timestamps; there are no MANO, wrist or joint columns. Older revisions were not read. All
numbers below are proxies:

* `stereo_check.py` (independent detector and views). RTMPose-m hand5 (OpenMMLab, Apache-2.0, ONNX, `rtmpose_hand.py`) runs top-down on the rectified LEFT and RIGHT gray views, boxed by our projection, and its detections are triangulated per joint. Those views are never used for 2D by the lift. MediaPipe on upscaled gray crops recalled only 12/60 and 8/60 hands, so RTMPose is used instead.
* `gt_cam_register.py` (DEV, public GT). Camera pose in the mocap world, from point-to-plane ICP of the GT-posed public scans against stereo depth (static objects only, our hands masked). It then gives:
  - hand-to-object distance while a GT object moves (someone must be holding it);
  - grasp rigidity in the object frame;
  - a comparison with the DEV placeholder (`sharpa_task/hands.placeholder_hands`, read from `tasks/dev_gt` for eps 0/12/41).

  Registration is reliable only with large, distinctive static geometry (dish rack, pitcher). Rotationally symmetric pot+lid, sweeper+dustpan and the wooden pieces mostly fail; see `eval_public.json` for per-episode `registration_valid_frac`.
* `eval_hands.py` aggregates both, writing `/mnt/secondary/v2d/t3/hands/eval_public.json` (frozen hybrid + `_p2` / `_z2` variants) and `eval_public_variants_stereo.json` (all variants, before curation).
* ICP fitness alone is NOT a valid registration check. The flat dust pan (eps 39/40) and the white pot (ep 43) slid onto the table plane at fitness 0.5-0.7. Only overlays verified by eye count (`REGISTRATION_OK`).

### Results (2026-10-08; all 20 public episodes; SGBM depth except ep 12 = FoundationStereo; `eval_public.json`)

Lifting variants (A = our 3D, B = independent RTMPose triangulation on the rect pair, joints where B exists).
Hand-to-moving-object distances come from the 5 episodes whose GT registration was verified by eye
(`REGISTRATION_OK = {12, 22, 26, 27, 31}`, see `eval_hands.py`):

| variant (`--reldepth`) | depth bias B-A | **depth MAD vs B** | rect-R px vs B | jitter | hand-to-moving-object median / share < 3 cm (5 eps) |
|---|---|---|---|---|---|
| **hybrid (frozen default)** | -0.27 cm | **1.10 cm** | 8.5 | 2.2 mm | **2.0 mm** / 100 % (episode median) |
| pnp | +0.03 cm | 1.35 cm | 8.6 | 2.0 mm | 2.7 mm / 99 % |
| zrel (MediaPipe landmark z) | -0.22 cm | 1.67 cm | 8.4 | 1.9 mm | 4.7 mm / 100 % |
| DEV placeholder (`sharpa_task`) | - | - | - | - | 6.0 mm (ep 12 only; by construction) |

- Coverage: hands are valid in 92.5 % of frames (both sides, public, frame-weighted); 98.4 % of valid frames have stereo
  support.
- Hand-to-moving-object, frozen hybrid, per episode:

  | episode | median | share < 2 cm | events |
  |---|---|---|---|
  | 12 | 0.9 mm | 100 % | 68 |
  | 27 | 2.0 mm | 97 % | 131 |
  | 31 | 1.8 mm | 100 % | 144 |
  | 22 | 2.3 mm | 64 % | 103 |
  | 26 | 15 cm | 33 % | 12 registered moving frames |

  The ep 22 misses are frames f46-f92, where the right hand pours with the pitcher OUTSIDE the cam_a field of view (checked on f64/f80). `hands_to_bundle.py --fill object` makes that hand ride with the pitcher: 45 frames filled, and 91.7 % of ep 22 moving-object events then have a hand within 3 cm (median 1.6 mm).
- Ours vs the DEV placeholder (ep 12, moving frames): wrist 34 cm apart, fingertips 18.5 cm apart (median). The placeholder approaches from above and parks the idle hand 35 cm to the side, while the real left hand helps hold the cup.
- Grasp rigidity: the grasping hand's palm centre moves 1.3 cm (median over episodes) in the held object's frame during a manipulation segment. This includes real regrasps.
- Choice of 3D source:
  - Independent per-joint triangulation (method B, RTMPose on the rect pair) is much noisier than dense stereo plus the hand-shape prior (method A). B's raw per-frame bone-length CV is 0.21, and it recovers only 148 of 176 hands on ep 12.
  - Most of the 8-12 px residual in the stereo views is inter-detector disagreement, not 3D error. RTMPose and MediaPipe differ by 17.9 px on cam_a (f=1040), which is 8.2 px at f=477, and RTMPose's wrist sits about 35 px further down the forearm.
  - The hybrid lift is frozen. On ep 12, FoundationStereo and SGBM give the same result: depth MAD vs B 0.95 / 0.96 cm, hand scale 0.875 / 0.879.

Per episode (frozen hybrid):

| ep | valid L/R | stereo-supported L/R | rect-R px vs B | depth bias B-A (cm) | depth MAD (cm) | jitter (mm) |
|---|---|---|---|---|---|---|
| 0 | 0.77/0.72 | 0.76/0.68 | 9.5 | -0.26 | 0.94 | 2.8 |
| 2 | 0.98/0.88 | 0.96/0.85 | 9.9 | +0.03 | 1.24 | 2.4 |
| 11 | 1.00/0.78 | 0.98/0.78 | 9.8 | -0.42 | 1.22 | 1.6 |
| 12 | 1.00/0.94 | 0.97/0.93 | 12.1 | -0.44 | 0.95 | 3.2 |
| 13 | 1.00/0.78 | 0.96/0.76 | 8.9 | -0.18 | 1.31 | 1.9 |
| 18 | 1.00/1.00 | 0.99/1.00 | 5.8 | +0.05 | 1.32 | 1.8 |
| 21 | 1.00/0.75 | 0.92/0.71 | 14.8 | -0.32 | 1.29 | 3.4 |
| 22 | 1.00/0.68 | 0.95/0.68 | 8.3 | -0.36 | 1.23 | 3.0 |
| 23 | 1.00/0.83 | 1.00/0.82 | 6.3 | -0.19 | 1.03 | 1.5 |
| 24 | 0.99/0.91 | 0.98/0.91 | 8.6 | +0.26 | 1.47 | 2.4 |
| 26 | 1.00/0.70 | 1.00/0.70 | 5.2 | -0.60 | 0.84 | 1.6 |
| 27 | 1.00/1.00 | 1.00/1.00 | 3.9 | -0.54 | 0.67 | 1.2 |
| 31 | 1.00/1.00 | 1.00/1.00 | 6.6 | +0.22 | 1.56 | 1.4 |
| 39 | 1.00/0.89 | 1.00/0.86 | 6.5 | -0.56 | 0.95 | 2.1 |
| 40 | 1.00/1.00 | 1.00/1.00 | 8.7 | +0.01 | 1.10 | 2.0 |
| 41 | 0.77/1.00 | 0.70/1.00 | 8.9 | -0.38 | 1.00 | 3.2 |
| 42 | 1.00/1.00 | 0.97/0.95 | 10.3 | -0.15 | 0.97 | 3.4 |
| 43 | 0.93/0.67 | 0.93/0.67 | 6.9 | -0.26 | 0.74 | 2.3 |
| 44 | 1.00/1.00 | 1.00/1.00 | 7.8 | -0.27 | 1.66 | 2.1 |
| 45 | 1.00/1.00 | 1.00/1.00 | 7.5 | -0.28 | 1.10 | 2.7 |

Evaluation split (frozen settings, never tuned on): 20/20 episodes lifted. Hand scale is 0.77-1.05 and wrist-to-middle-MCP 7.2-8.7 cm (median 7.8). Median cam_a reprojection is 1.7-4.6 px. Log: `/mnt/secondary/v2d/t3/logs/hands_lift_evaluation.log`.

End to end, through `to_sharpa_task.py --verify` on dev bundles of GT objects plus OUR hands, put in the mocap world by `gt_cam_register.npz` (`/mnt/secondary/v2d/t3/tasks/dev_ourhands`):

- Ep 12: 15/15 checks pass, with hands from `bundle_keypoints+kabsch_template`.
  - IK frame-task error R/L is 3.61 / 3.72 cm (final hands; an earlier lift gave 3.67 / 4.05 cm). For comparison, the template placeholder gives 2.2 cm and keypoint-only MANO hands in the frame test give 2.32 / 3.27 cm.
  - Contact frames are R 106 / L 25 for our hands, against R 118 / L 0 for the placeholder.
- Ep 22 (blue cup + water pitcher, with `--fill object`, 45 right-hand frames riding the pitcher): 15/15 checks pass. IK frame-task error R/L is 3.97 / 2.88 cm, with contact frames R 119 / L 138. Log: `/mnt/secondary/v2d/t3/logs/hands_convert_dev_ourhands_ep22.log`.

## Commands (as run 2026-10-08)

```bash
P=/mnt/secondary/v2d/envs/t3-hands/bin/python; cd ~/TestingGrounds/egony/v2d/track3/hands
OMP_NUM_THREADS=2 nice -n 10 $P -I detect_mp.py --episodes all --workers 3          # 40 eps, ~7 min
OMP_NUM_THREADS=3 nice -n 10 $P -I sample_depth.py --episodes all --workers 3       # SGBM ~0.5 s/frame under load
OMP_NUM_THREADS=1 nice -n 10 $P -I lift_hands.py --episodes all --workers 3 --overwrite   # ~10-30 s/episode
# dev only (public):
$P -I stereo_check.py --splits public --stride 3 --workers 2
$P -I gt_cam_register.py --stride 2 --workers 3
$P -I eval_hands.py --tags "" --out /mnt/secondary/v2d/t3/hands/eval_public.json
$P -I viz_hands.py evaluation/episode_000030 40,130,220 /tmp/x.jpg     # cam_a | rect-L | rect-R overlay
```

When FoundationStereo depth is complete for more episodes, run `sample_depth.py` again (it writes
`depth_samples_fs.npz` next to the SGBM file) and then `lift_hands.py --overwrite` (`--depth auto` prefers FS).
On ep 12, FS and SGBM lifts agree: hand scale 0.875 / 0.879, depth MAD vs B 0.95 / 0.96 cm.

## Known limitations / next steps

- MediaPipe 2D is weakest on motion blur and on fingers hidden behind a held object. Possible improvement: fuse RTMPose-hand on cam_a, boxed by the tracks; it is already wired in `rtmpose_hand.py`.
- `conf` is heuristic. Hands out of view and not touching a moving object (gaps over 0.4 s) are interpolated or held by the converter; there is no reach or park model. `--fill object` only covers hands carrying an object.
- Hand size is estimated per episode, with no same-demonstrator assumption. A global size prior would cut the 6-7 % size CV if one person recorded all episodes; the same white watch is seen in public and eval.
- GT-anchored metrics cover only the registrable public episodes.
