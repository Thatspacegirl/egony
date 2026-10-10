# V2D Track 1: local validation harness and model-agnostic post-processing (CPU)

Everything here runs on CPU, scores through the kit's own code, and does not touch Kaggle.
Python: `/mnt/secondary/v2d/scratch/venv/bin/python -I` (numpy, scipy, pandas, torch-cpu, numba, trimesh, fast_simplification, huggingface_hub).
Set `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""` and use `nice -n 10`.

Large data lives on `/mnt/secondary/v2d/t1/`:

| path | content |
|---|---|
| `formhoi_val/<sequence_id>/` | 25 FORM-HOI sequences: one exo video, its human/object mask H5, GT `mhr_params_mv.pt`, `poses.npy`, `pose_valid_mask.npy`, `interaction_trim.json`, `failure_segments.json`, `object_mesh__output_aligned.glb`, `edex`, `ground_plane.json` (1.5 GB) |
| `formhoi_val/selection.json` | the exact list (copy: `formhoi_val_selection.json` here) |
| `formhoi_val/kitval/` | GT in kit NPZ form (`gt/episode_%06d.npz` + mesh link), `sample.parquet`, `solution.parquet` (kit-packed GT as `ref_x/y/z`), `val_index.json` (scored frames) |
| `assets/mhr_asset_603.npz` | kit-format MHR body asset rebuilt from Meta MHR v1.0.1 (`mhr_asset.py`) |
| `work/` | tuning / evaluation logs and JSON (see "Results") |

## Integrity

* Dataset revision used for Track 1/2 metadata: `5f68335` (local clone). FORM-HOI (`nvidia/form-hoi`, CC-BY-4.0, external data) pinned at commit `c63db107e84c7f74bb4929ef643b67b5c8bcc00e`; each member is sha256-checked against the release manifest.
* `formhoi_select.py` **asserts** that none of the 30 Track 1 sequence ids and none of the 30 Track 2 tier-1 sequence ids are selected, and that no selected sequence name contains a Track 1 object token (hula_hoop, big_red_bowl, iron, white_desk, black_pan, foam_grass_block, paint_roller, pink_foam_roll, short_wood_stool, white_laptop_cart). Note: `g1_box` (val episode 13) is the same physical object as Track 2's `big_black_box`; the sequence itself is not a Track 2 sequence and this set is only used for Track 1.
* FORM-HOI meshes and `edex` calibration are used ONLY to score/corrupt FORM-HOI GT locally. They must never feed Track 1 inference.

## 2026-10-09: full pipeline with the newly accessible gated models

Status/results of this phase are in "Results (real predictions)" at the end; this section is the how-to.

**Stages** (GPU stages run through `/mnt/secondary/v2d/bin/gpu_hi.sh --raw bash t1_gpu_queue.sh <stage> <split> <budget> <eps>`,
or `t1_drive.sh` which repeats holds until done; per-episode logs in `/mnt/secondary/v2d/t1/logs/gpu/`):

| stage | script | env | output (`/mnt/secondary/v2d/t1/...`) |
|---|---|---|---|
| masks | `t1_masks.py --stride 1` (SAM3 text prompts "person" + object sentence + object noun; object = best-scoring chain, actor = person chain touching the object most; 240-frame SAM3 sessions linked by mask IoU) | t3-seg | `masks/<split>/episode_X/masks.npz` (window = scored span -60/+30 frames, scale 0.5) |
| sam3do | `t1_sam3do.py prep/gen` + `t1_moge.py points` (SAM 3D Objects via the toolkit's `v2d.sam3d.lib.image_to_mesh`, MoGe-2 metric point map of the least-occluded frame restricted to the object mask, R*s baked) | t1-gen, t1-sam3do | `mesh_sam3do/<split>/episode_X/mesh.glb` |
| trellis | `t1_trellis.py` (TRELLIS-image-large on the same kind of frame; unit-cube mesh) | t1-gen | `mesh_trellis/<split>/episode_X/mesh_0.ply` |
| cari4d | `t1_cari4d.py run` = official CARI4D stages 01-07 (commands copied verbatim from `lib/run_inference.py`, default settings) on a lossless window clip + our masks (`prep`), plus a mesh-scale hook between stages 04 and 05 (toolkit `FoundationPoseTracker.estimate_scale_grid_search` against CARI4D's human-aligned depth at the mesh's reference frame, lo 0.5 hi 2 x9 x4 levels, as in `v2d_pipelines/run_hand_masks.py`); optional `--known-k` | t1-cari4d | `cari4d_<mesh>[tag]/<split>/episode_X/` |
| decode (CPU) | `t1_cari4d.py decode`: refined bundle -> three variants init (SAM 3D Body + FoundationPose), coconet, refined; EXACT param conversion (MHRHead model params, root translation 10*flip*mhr_trans; checked 0.0003 mm vs CARI4D's MHRLayer) | t1-cari4d | `.../episode_X/t1/{init,coconet,refined}/episode_X.npz` (video-frame indexed, padded) |
| score (CPU) | `t1_val.py --src name=glob --smooth none,default` | scratch venv | tables per split (tune 0-12 / held-out 13-24) |

**Val-only mask shortcut** (2026-10-09 third agent; GPU time is shared by three tracks + training): val CARI4D /
mesh runs use FORM-HOI's RELEASED per-frame masks converted to our masks.npz schema (`t1_masks_formhoi.py` ->
`masks_fh/val/`, 25 episodes in 50 s CPU) instead of SAM3; select with `T1_MASKS=masks_fh` (t1_gpu_queue.sh /
t1_cari4d.py), mesh dirs then get the suffix `_fh` (`mesh_sam3do_fh`), CARI4D runs use `T1_TAG=_fh...`.
Track 1 ALWAYS uses our SAM3 masks (the queue refuses masks_fh for split track1).  Our SAM3 masks matched these at
IoU 0.957 / 0.969 on val ep 18; val ep 11 gets both mask sources to measure the effect on the metrics.

**Envs** (built 2026-10-09, no Docker): `t1-cari4d` = copy of t3-fpose + CARI4D Dockerfile packages
(`v2d/envs/build_t1cari4d_env.sh`); `t1-sam3do` = copy of t3-fpose + v2d_sam3d Dockerfile packages, MoGe-1 @ a8c3734 with
its utils3d @ 3913c65 (`v2d/envs/build_t1sam3do_env.sh`); `t1-gen` = MoGe-2 + TRELLIS (`v2d/envs/build_t1gen_env.sh`).
`t1-trellis2` (built 2026-10-09 11:05-11:27 by the third agent, `v2d/envs/build_t1trellis2_env.sh`): torch 2.6.0+cu124,
xformers 0.0.29.post3 (attention backend, no flash-attn), TRELLIS.2 @ 75fbf01 via PYTHONPATH, compiled o-voxel (TRELLIS.2
repo), CuMesh @ 12289e1 and FlexGEMM (JeffreyXiang) for sm_86; stage `trellis2` (`t1_trellis2.py`, 512 pipeline,
our mask as alpha so RMBG-2.0 is loaded but unused, vertex colours = queried base colour).  On a CPU-only import
triton refuses to start ("0 active drivers"), so the env is first exercised in GPU batch 1.

**Weights** (`t1_download_weights.py`): `/mnt/secondary/v2d/weights/cari4d` (toolkit layout; cari4d_commercial step200000
sha256 78ff5cb8... verified), `/mnt/secondary/v2d/weights/sam3d_objects`, TRELLIS.2/DINOv3/RMBG-2.0 in the HF cache (downloaded,
env not built).

**Models and licences** (all used for inference only):

| model | licence |
|---|---|
| nvidia/cari4d_commercial @ 1f7287ac (CoCoNet) | NVIDIA Open Model Agreement |
| facebook/sam-3d-body-dinov3 @ 11aaa346 (CARI4D stage 03; also inside nvidia/GEM-X) | SAM License |
| facebook/sam-3d-objects (code facebookresearch/sam-3d-objects @ f91db41) | SAM License |
| facebook/sam3 | SAM License |
| Ruicheng/moge-2-vitl-normal @ b135031b, Ruicheng/moge-vitl | MIT |
| microsoft/TRELLIS-image-large | MIT |
| FoundationPose nvlabs_pytorch weights (toolkit download) | NVIDIA Source Code License (FoundationPose) |
| nvidia/GEM-X | NVIDIA Open Model License |
| DINOv2 (torch hub, CARI4D / SAM3D deps) | Apache-2.0 |
| microsoft/TRELLIS.2-4B (+ its sparse-structure decoder from microsoft/TRELLIS-image-large), facebook/dinov3-vitl16-pretrain-lvd1689m, briaai/RMBG-2.0 | MIT, DINOv3 License, Bria RMBG-2.0 licence (CC BY-NC 4.0 for the open weights; loaded by the TRELLIS.2 pipeline but never run: we pass our own alpha mask) |
| Meta MHR v1.0.1 (`mhr_model.pt`, sha256 352e271a..., identical to sam-3d-body-dinov3/assets/mhr_model.pt) | Apache-2.0 |

**Camera intrinsics** (ruling 2026-10-09): the published FORM-HOI intrinsics are identical in all 25 val sessions
(March-June 2026) per physical camera (front 856.64, back 846.02, left 846.90, right 840.57 px;
`/mnt/secondary/v2d/t1/camera_intrinsics_formhoi.json`) and may be used as a property of the physical camera
(`t1_cari4d.py run --known-k`).  FORM-HOI extrinsics / world registration are never used.

**Same K for the September 2026 Track 1 sessions** (`t1_cam_check.py`, 2026-10-09 third agent; 18 of the 30 Track 1
episodes are 2026-09-10/11 sessions, FORM-HOI has none after June): the first frame of every Track 1 video and every
FORM-HOI val video was matched (SIFT + RANSAC homography, sub-pixel: median reprojection 0.2-0.66 px) against one
FORM-HOI val video of the same physical camera.  With the FORM-HOI K, M = K^-1 H K is a pure rotation in every case:
Mar-Jun sessions rotate 0-1 deg, the September re-mount rotates front 4.9-5.0, right 6.2, back 2.4-2.6, left 0.4 deg;
the best focal ratio is 0.996-1.003 (orthogonality residual 0.0005-0.009).  So the physical-camera K holds (to
+-0.3 %) for all 30 Track 1 episodes; only the (unused) extrinsics changed.  JSON: `/mnt/secondary/v2d/t1/work/cam_check_1009.json`.

## Files

| file | what |
|---|---|
| `formhoi_select.py` | deterministic selection (seed 20261008), 25 objects outside Track 1/2 object lists |
| `formhoi_fetch.py` | range-reads individual tar members (HTTP range via `HfFileSystem`, no 1.1 GB tar downloads), sha256-verified |
| `mhr_asset.py` | rebuilds the hidden 603-vertex kit body asset from Meta MHR v1.0.1; kit `_MHRBody` forward == TorchScript to < 0.001 mm (42 frames checked). Vertex ROLES are an approximation (hands = CARI4D hand-surface FPS spec) |
| `t1lib.py` | loads the kit metric modules by path with our body asset; FORM-HOI GT -> kit episode dict; scored-frame rule (contact span minus failure segments and invalid poses, stretches >= 8) |
| `score_t1.py` | **batch local scorer**: `build-val`, `score`, `selftest`, plus `FastScorer` (same arrays without the parquet round trip, identical numbers) |
| `smoothing.py` | **SO(3)-correct, frame-0-protectable Whittaker smoother** for root rotation/translation, body, hands, object rotation/translation |
| `noise_model.py` | monocular-like corruption of GT in the validation camera frame (iid + low-frequency drift + glitches + depth-scale error); presets `cari4d_like` (calibrated to the CARI4D baseline row), `raw_sam3d_like` |
| `tune_smoother.py` | coordinate-descent tuning on val episodes 0-12, held-out report on 13-24; `--pred-dir` tunes on REAL predictions instead of synthetic noise |
| `pen_refine.py` | **penetration removal that keeps contact**: hand-side (finger + wrist params) optimisation through a differentiable MHR subset forward, then a gated smooth object-offset stage (runs only if PEN after the hand stage > 0.0005 cm, and is kept only if it at least halves that PEN; otherwise reverted) |
| `eval_pen.py` | PEN refinement before/after on clean GT and noisy+smoothed GT |
| `postprocess.py` | applies smoothing + PEN refinement to any directory of episode NPZs (scored frames from the competition sample parquet or `val_index.json`) |
| `cari4d_to_t1.py` | **CARI4D converter**: `decode` (refined.pth -> CARI4D MHRLayer forward -> lod1 vertices, in the CARI4D container; object pose composed with the export's `object_mesh_to_training_transform` so it maps raw `output_aligned.glb` to camera space), `fit` (kit `tools/track1/mesh_to_mhr_params.convert`), `write` (episode NPZ + mesh, optional smoothing + PEN with the `postprocess.py` defaults), `selftest` (CPU) |
| `t1_masks_sam3.py` | **SAM3 human + object masks -> CARI4D mask H5** (replaces the interactive SAM2 step; `lib/pack_masks.py` schema; prompts "person" + metadata sentence + object noun). CPU smoke test on val ep 18 (2 frames, scale 0.5): IoU vs FORM-HOI masks human 0.957, object 0.969 (the sentence prompt alone found no object at scale 0.25; the noun did) |
| `summarize_e2e.py` | tables of kit-scored runs per split (tune 0-12 / held-out 13-24 / all) |
| `make_noisy_preds.py` | writes a synthetic prediction directory (corrupted GT + GT mesh) for end-to-end tests |
| `smooth_default.json` = `smooth_cari4d_like.json` | tuned smoother config for CARI4D-like input (default of `postprocess.py`) |
| `smooth_raw_sam3d_like.json` | tuned config for noisier per-frame input |

## Commands

```bash
PY="nice -n 10 /mnt/secondary/v2d/scratch/venv/bin/python -I -W ignore"; D=~/TestingGrounds/egony/v2d/track1
# 1) validation set (already done; fetch is idempotent and skips verified members)
$PY $D/formhoi_select.py --archives /mnt/secondary/v2d/scratch/track1/formhoi_meta/manifest/archives.parquet \
   --t1-meta ~/TestingGrounds/egony/video_to_data_challenge/track_1/meta/episodes_metadata.jsonl \
   --t2-meta ~/TestingGrounds/egony/video_to_data_challenge/track_2/tier_1_multiview_caption/meta/episodes_metadata.jsonl \
   --n 25 --exclude 2026-04-07_13-27-49_big_fish_hook_rotate_01 --out /mnt/secondary/v2d/t1/formhoi_val/selection.json
$PY $D/formhoi_fetch.py --selection /mnt/secondary/v2d/t1/formhoi_val/selection.json \
   --files-manifest /mnt/secondary/v2d/scratch/track1/formhoi_meta/manifest/files.parquet --out /mnt/secondary/v2d/t1/formhoi_val
$PY $D/mhr_asset.py --mhr /mnt/secondary/v2d/scratch/track1/mhr_assets/mhr_model.pt --out /mnt/secondary/v2d/t1/assets/mhr_asset_603.npz
$PY $D/score_t1.py build-val --selection /mnt/secondary/v2d/t1/formhoi_val/selection.json
# 2) sanity: GT vs GT through the kit packer + kit score(); fast path == parquet path
$PY $D/score_t1.py selftest --n 3
# 3) score any prediction dir (episode_%06d.npz + episode_%06d_object.glb, val episode numbering)
$PY $D/score_t1.py score --pred <dir> --kit-score --json <dir>/score.json
# 4) post-process any model output, then score it
$PY $D/postprocess.py --in <dir> --out <dir_post> --val-index /mnt/secondary/v2d/t1/formhoi_val/kitval/val_index.json [--no-pen]
#    for Track 1 itself: --sample /mnt/secondary/v2d/kit/v2d_submission_kit/data/track_1_sample_submission.parquet
# 5) re-tune on real predictions once CARI4D / SAM 3D Body runs on the val videos
$PY $D/tune_smoother.py --presets cari4d_real --pred-dir <dir> --out /mnt/secondary/v2d/t1/work/tune_real.json
# 6) CARI4D converter selftest (CPU) and real use (see cari4d_to_t1.py docstring)
$PY $D/cari4d_to_t1.py selftest --frames 8 --keyframes 2
# 7) masks for CARI4D (GPU; env t3-seg). CPU smoke: add --cpu --max-frames 2 --episodes 18
/mnt/secondary/v2d/envs/t3-seg/bin/python -I $D/t1_masks_sam3.py --formhoi-val --compare-formhoi --out /mnt/secondary/v2d/t1/masks/formhoi_val
/mnt/secondary/v2d/envs/t3-seg/bin/python -I $D/t1_masks_sam3.py --track1 --out /mnt/secondary/v2d/t1/masks/track1
# 8) per-split tables
$PY $D/summarize_e2e.py raw=work/e2e_raw_score.json smooth=work/e2e_smooth_score.json smooth+pen=work/e2e_smooth_pen_score.json
```

The `episode_%06d` numbers of a val prediction directory are the `val_index` values of `selection.json`
(`kitval/val_index.json` maps them to sequence ids and cameras). The video to run a model on is
`formhoi_val/<sequence_id>/videos__<camera>.mp4`.

## GPU queue (not run: the RTX 3090 belongs to another job in this phase)

Frame counts: Track 1 = 30 videos / 16,563 frames; FORM-HOI val = 25 videos / 20,177 frames (13,713 scored).
HF access checked 2026-10-08 with the local token: **gated, no access** for `nvidia/cari4d_commercial`,
`facebook/sam-3d-body-dinov3`, `facebook/sam-3d-objects`; OK for `facebook/sam3`, `Ruicheng/moge-2-vitl-normal`,
`microsoft/TRELLIS-image-large`.

| # | job | gated? | est. GPU h | command / note |
|---|---|---|---|---|
| 1 | SAM3 masks, val + Track 1 (36.7k frames, scale 0.5, bf16) | no | 1.5 | `t1_masks_sam3.py --formhoi-val --compare-formhoi ...` then `--track1` (env t3-seg, `--max-rss-gb 12`) |
| 2 | object meshes (10 Track 1 objects + 25 val objects) | SAM-3D-Objects: yes; TRELLIS fallback: no | 1-2 (+ env build) | from the best SAM3-masked frame per episode; no host env/script yet. Track 2 meshes are forbidden for Track 1 |
| 3 | CARI4D full pipeline, val 25 + Track 1 30 episodes | yes (cari4d_commercial + sam-3d-body-dinov3) | 12-20 | `python -m v2d.cari4d.docker.run_inference --video_path <masks>/episode_X/episode_X.0.color.mp4 --mask_h5_path <masks>/episode_X/episode_X_masks_k0.h5 --object_mesh_path <mesh> --weights_path /mnt/secondary/v2d/weights/cari4d --output_dir /mnt/secondary/v2d/t1/cari4d/<split> --expected_frames <T> --postopt_batch_size 256`; needs the ~25-30 GB CARI4D Docker image (root disk) |
| 4 | CARI4D -> Track 1 NPZ: `decode` (container) + `fit` (kit mesh_to_mhr_params, float32, GPU) | decode: yes (SAM 3D Body `model.ckpt` holds the MHR head buffers) | 2.5-3.5 (0.2-0.35 s/frame) | `cari4d_to_t1.py decode/fit/write` (docstring), then `postprocess.py` and `score_t1.py score`. CPU fit measured: 923 s for 8 frames, so GPU only |
| 5 | SAM 3D Body single-view baseline (no CARI4D) | yes (sam-3d-body-dinov3) | 2.5 | a raw per-frame baseline for `smooth_raw_sam3d_like.json`; CARI4D stage 3 runs it anyway |

After 3-4 on the val set: re-tune on real predictions (`tune_smoother.py --pred-dir`, CPU) and re-check the PEN gate.

## Notes on the metric (from the kit code)

* CD-H/CD-O/ACC-H/ACC-O compare against the reference after ONE Sim(3) fitted on the first scored frame's
  alignment vertices; PEN is computed on the submission alone (own hand points vs own mesh, scaled by the Sim(3)
  scale), so "GT vs GT" gives the pseudo-GT's own penetration, not 0.
* Root and object rotations must be smoothed on SO(3): Whittaker on the raw Euler angles (they wrap at +-pi)
  is catastrophic (held-out CD-H 40.6 vs 13.5).

## Results

All numbers: local kit metric code on the 25 FORM-HOI val episodes (tune split = val episodes 0-12, held-out =
13-24), synthetic "cari4d_like" predictions = FORM-HOI GT corrupted by `noise_model.py` (calibrated so the
unsmoothed set scores like the public CARI4D baseline row), GT mesh as the object mesh. Logs/JSON in
`/mnt/secondary/v2d/t1/work/`. Absolute values are not Kaggle values (different sequences, approximate vertex roles).

**Sanity.** `score_t1.py selftest --n 3`: GT vs GT through the kit packer + each kit module's `score()`:
CD-H 8.7e-14, CD-O 4.9e-14, ACC-H 4.6e-15, ACC-O 3.4e-15 cm; PEN 0.0067 cm (= the pseudo-GT's own
hand/object penetration; PEN is computed on the submission alone). Fast path == parquet path (max diff 0).
End-to-end runs assert our per-episode means == the kit `score()` output (`kit_score_matches: true`).

**Smoother** (`smooth_default.json` = `smooth_cari4d_like.json`: Whittaker lambda root-rot 1000 (SO(3)),
root-trans 30, body 30, hands 100, object-rot 100 (SO(3)), object-trans 10, anchor none, Huber IRLS), tuned by
coordinate descent on episodes 0-12 only:

| held-out 13-24 (cari4d_like) | CD-H | CD-O | ACC-H | ACC-O |
|---|---|---|---|---|
| unsmoothed | 18.149 | 24.675 | 2.156 | 0.730 |
| smoothed (tuned) | **13.471** | **14.141** | **0.134** | **0.050** |
| same but Whittaker on raw Euler angles | 40.636 | 64.136 | 0.459 | 0.058 |
| same without Huber | 13.599 | 14.291 | 0.138 | 0.056 |
| frame-0 anchor 'light' lam 30 instead of none | 13.645 | 14.771 | 0.134 | 0.050 |
| clean GT through the same smoother | 1.840 | 1.922 | 0.132 | 0.046 |

Tune split 0-12: 13.157/15.763/1.981/0.702 -> 11.919/13.707/0.146/0.067. The held-out CD gains come mostly
from ONE glitch at the alignment frame that the robust smoother repairs (ep 14: CD-H 68.9 -> 14.6, CD-O 131.6 -> 9.8);
without ep 14 the held-out change is CD-H 13.538 -> 13.369, CD-O 14.950 -> 14.535, ACC-H 2.125 -> 0.135,
ACC-O 0.733 -> 0.052. So: ACC gains are large and robust; CD gains are small except where a glitch hits frame 0.
Noisier "raw_sam3d_like" input (ACC-H 6.1 raw): its own tuned config (`smooth_raw_sam3d_like.json`) gives
13.107/13.794/0.140/0.055 held-out, but the cari4d_like config transfers better on CD (12.839/13.413/0.148/0.074)
and its no-anchor variant is best (12.847/13.470/0.140/0.055): **frame-0 anchoring helps only on clean input**
(clean GT CD-H 1.08 with anchor vs 2.34 without); with noisy input it pins frame-0 noise into the Sim(3).

**PEN refinement** (`pen_refine.py`, margin 2 mm, hand stage then gated object stage):
* hand stage only, held-out, `eval_pen.json`: clean GT PEN 0.1210 -> 0.00025 cm (CD-H +0.013, CD-O +0.002,
  ACC +0.000); smoothed cari4d_like PEN 0.0692 -> 0.0054 cm, CD-H +0.0003, CD-O/ACC-H/ACC-O unchanged.
* object stage alone is costly (clean GT ep 24: CD-O +7.3, ACC-O +1.8), hence gated (PEN after hand > 0.0005 cm)
  and accepted only if it at least halves that PEN. Without the acceptance check val ep 0 got WORSE
  (0.00079 -> 0.00147 cm, 1.3 cm mean object shift: two-sided grasp); with it the offset is reverted.
* end-to-end, held-out 13-24, smoothed cari4d_like -> `postprocess.py --no-smooth` (defaults) -> kit packer +
  kit metrics (`work/e2e_smooth_pen_score_heldout.json`; hand-only = same run with the input object translation
  restored, `work/e2e_smooth_penhand_score_heldout.json`):

| held-out 13-24 | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|
| smoothed | 13.4714 | 14.1413 | 0.13389 | 0.05034 | 0.06921 |
| + PEN hand stage | 13.4717 | 14.1413 | 0.13389 | 0.05034 | 0.00535 |
| + gated object stage (default) | 13.4717 | 14.2435 | 0.13389 | 0.05220 | **0.00023** |

  Object stage accepted on ep 18 (palm inside the air purifier: PEN 0.058 -> 0, CD-O +1.35, ACC-O +0.021) and
  ep 21 (0.0035 -> 0, CD-O -0.12); ran and was REJECTED on ep 14 and 16 (it would have raised PEN). Hand-stage
  contact fraction (hand points within 1.5 cm of the surface) unchanged in every episode; the largest single
  finger-parameter change was 1.3-1.8 rad (fingers lifted out of the surface), which the metrics do not see but a
  viewer might. Public leaderboard PEN best is 0.00058, so the object stage is what gets below it locally.
* the PEN settings (margin 2 mm, gate 0.0005 cm, object lambda 100, accept ratio 0.5) were chosen by looking at
  held-out episodes 18/21, so episodes 0-12 are the cleaner check for them (`work/e2e_smooth_pen_score.json`,
  `work/e2e_smooth_penhand_score.json`, all 25 episodes, kit `score()` matches):

| | split | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|---|
| smoothed | 0-12 | 11.9192 | 13.7071 | 0.14624 | 0.06741 | 0.35112 |
| + hand stage | 0-12 | 11.9186 | 13.7066 | 0.14624 | 0.06741 | 0.06598 |
| + gated object stage | 0-12 | 11.9186 | 14.0243 | 0.14624 | 0.07131 | **0.00019** |
| smoothed | all 25 | 12.6643 | 13.9155 | 0.14028 | 0.05921 | 0.21582 |
| + hand stage | all 25 | 12.6641 | 13.9153 | 0.14028 | 0.05921 | 0.03690 |
| + gated object stage | all 25 | 12.6641 | 14.1295 | 0.14028 | 0.06215 | **0.00019** |

  Object stage accepted on 7/25 episodes (4, 5, 8, 9, 12, 18, 21: the ones where the palm, not the fingers, is
  inside: yoga ball ep 9 PEN 2.61 -> hand 0.65 -> 0, giant box ep 12 1.35 -> 0.20 -> 0); CD-O cost per accepted
  episode -0.12 .. +1.70 cm. Runtime (4 threads, machine load 20-70): 10 s - 10 min per episode, ep 12 16 min, worst 38 min
  (ep 9, 894 frames, 766 penetrating). Full log: `work/post_cari4d_like_smooth_pen/episode_*_post.json`.

**Bottom line for a real submission** (any model: CARI4D, SAM 3D Body, ...): `postprocess.py` with defaults
(smooth_default.json + PEN hand + gated object) is a CPU-only, model-agnostic pass that on this validation set takes
ACC-H 2.06 -> 0.140, ACC-O 0.72 -> 0.062, PEN 0.216 -> 0.0002 and does not hurt CD-H/CD-O (-2.9 / -5.9 cm here,
mostly from repairing frame-0 glitches). The CD metrics then depend on the upstream model (frame-0 pose accuracy and
metric scale), which post-processing cannot fix. Re-tune the smoother with `tune_smoother.py --pred-dir` once real
predictions on the val videos exist; the synthetic noise model is calibrated only on the public CARI4D row.

## Results (real predictions), 2026-10-09 third agent

Val = FORM-HOI val episodes; CARI4D runs on val use FORM-HOI's released masks (`masks_fh`, see "Val-only mask
shortcut"), SAM-3D-Objects meshes rescaled by the scale hook.  Only val TUNE episodes 5 and 11 so far (GPU batch b1);
absolute values are far below the leaderboard because these two clips are short and easy -- compare rows, not
against the leaderboard.

| val 5 + 11 (mean) | CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|---|
| CARI4D init (SAM 3D Body + FoundationPose), raw | 6.90 | 46.1 | 1.66 | 3.62 | |
| CARI4D CoCoNet, raw | 5.56 | 10.88 | 1.49 | 2.10 | |
| CARI4D CoCoNet + default smoothing | 5.34 | 10.62 | 0.128 | 0.136 | |
| CARI4D CoCoNet + smoothing + PEN (`postprocess.py` defaults) | 5.34 | 10.62 | 0.128 | 0.136 | 0.0006 |
| GEM-X route `s3` (SAM-3D-Body(K) per frame, t1_human.py) + smoothing | 6.92 | - | 0.136 | - | |
| GEM-X route `s3gx` (GEM-X depth) + smoothing | 20.10 | - | 0.139 | - | |
| hybrid CoCoNet root + SAM-3D-Body articulation, smoothed | 7.38 | 11.22 | 0.132 | 0.136 | |

* Ep 5 alone, refinement (stage 07) vs CoCoNet, both + default smoothing: CD-H 4.8487 vs 4.8491, CD-O 9.90 vs 9.93,
  ACC-H 0.120 vs 0.114, ACC-O 0.097 vs 0.104.  Stage 07 costs ~1.8 s/frame (more than stages 01-06), so the next
  batches stop after CoCoNet (`T1_STOP=06`, `t1_cari4d.py run --stop-after 06`, decode `--bundle coconet`).
* Mesh shape (`t1_mesh_shape.py`): SAM-3D-Objects shapes are good (similarity-aligned Chamfer 0.40 / 0.97 cm on
  vase / crate) but their own MoGe-based size is 0.38x GT; after the CARI4D scale hook (silhouette s0 + FoundationPose
  grid) 1.09x / 1.20x GT.
* SAM3 vs FORM-HOI masks on val 11: IoU human 0.974, object 0.886 (median 0.928).
