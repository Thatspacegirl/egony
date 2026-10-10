# Track 3 policy side: perception bundle → FlashCHORD floating-Sharpa task

Entry point: `../to_sharpa_task.py`. If `robotic_grounding` is not importable, it re-execs itself with
`/mnt/secondary/v2d/envs/retarget/bin/python`. Everything here is CPU-only except `train_3090.sh` and
`eval_dev.sh`.

```
perception bundle (.npz, t3_perception_v1)  ──to_sharpa_task.py──▶  <out-root>/human_motion_data/v2d_track3/
   object poses (N,B) in roster order          objects/episode_X/<name>.obj|.urdf   (sim mesh == submit mesh)
   our meshes, table plane, MANO (optional)    loaded/…/sequence_id=episode_X/robot_name=sharpa_wave   (MANO+objects)
                                               processed/…/sequence_id=episode_X/robot_name=sharpa_wave (+Sharpa IK) = task.parquet
                                               reconstructed_stage/episode_X_sharpa_wave_support.usda
                                               manifests/episode_X.json (+ _ik.log, _verify.json)
                                             <out-root>/submit_meshes/episode_X/<name>.obj   (non-dev bundles only)
```

## Files

| file | env | what |
| --- | --- | --- |
| `../to_sharpa_task.py` | retarget | the converter (steps 1-11 in the file) |
| `bundle.py` | any | input contract `t3_perception_v1` + roster/episode guards |
| `mesh_prep.py` | **kit venv** | runs the packer's own `v2dlb.mesh_budget.budget_mesh` (4096/4096), strips the padding, writes OBJ, then re-budgets that OBJ to prove the packer will not change it |
| `geom.py` | any | quaternions, gap filling, alignment, mass properties, table estimation |
| `hands.py` | retarget | MANO template (from the released tissue-box HaMeR reference), DEV placeholder hands, keypoint→orientation Kabsch, 16-link contacts |
| `checks.py`, `usd.py` | any | frame-0 support/pair checks, Cube support USDA writer |
| `verify_task.py` | flash_chord | loads the task with FlashCHORD itself (frames, names, URDFs, support lookup, regex) + Hydra compose + optional CPU Newton scene build |
| `settle_check.py` | flash_chord | frame-0 physics with the exact training CoACD hulls (Newton loader → same cache key) in CPU MuJoCo, objects + support only: frame-0 contact depths and drift after 0.25 s / 1 s with no hands |
| `make_dev_inputs.py` | any | **DEV ONLY**: bundles from PUBLIC GT poses + public scans (refuses eval episodes, marks `dev_only`) |
| `make_frame_test_bundles.py` | any | DEV TEST: re-expresses a dev bundle in a tilted/offset "VO" world with a table plane and input MANO or keypoint hands (exercises the eval-side code paths; the score must be unchanged) |
| `rollout_to_devscore.py` | any | evaluator rollouts (or `--task-reference`) → layout of `scratch/track3_perception/t3_devscore.py` |
| `configs/experiment/sharpa_flash_sac_3090.yaml` | — | 3090 preset (used via `--config-dir`) |
| `replay_budget.py` | flash_chord (CPU) | builds the real RLEnv on the Warp CPU device (1 world) and evaluates FlashCHORD's `replay_memory()` for the preset and the recipe → `manifests/<ep>_replay_budget.json` |
| `train_3090.sh`, `eval_dev.sh` | flash_chord (GPU) | guarded launchers with RAM/GPU gates |

## Contract checks (why each exists)

* `fps=20.0`, `T = N video frames`, and training with `task.motion_speed=1.0`. The evaluator then emits
  exactly N steps. The recipe's 0.5 gives about 2N steps, and the packer silently keeps the first half.
  `verify_task.py` prints both counts.
* `object_body_names` = roster order from `data/track_3_evaluation_objects.json`. This is enforced in
  `bundle.py` for eval episodes. The packer slot is the body index.
* The first `episode_(\d+)` match in the task path is the episode. The harness provenance uses
  `re.search` on the path, so `--out-root` must not contain `episode_<digit>`.
* One rigid URDF per body. `<visual>` and `<collision>` point to the same budgeted OBJ. The evaluator
  needs a visible mesh, and the packer re-budgets the same file without changing it.
  Mass: name prior (`geom.MASS_PRIOR_KG`, plausible household values) → else hull volume × 250 kg/m³,
  clipped to 0.03–1.5 kg. COM is the convex-hull centroid. Inertia is the solid hull scaled to the
  mass. Newton honours URDF `<inertial>` (mass, COM, full tensor).
* Sim frame: z-up and table horizontal. FlashCHORD supports only axis-aligned Cube/Cylinder supports,
  and the training scene has a ground plane at z=0, so the table must sit above it (`auto` keeps the
  input height if it is in [0.3, 2] m, otherwise uses 0.70). The x/y is re-centred on the frame-0
  objects unless `--no-recenter-xy`. The scorer is invariant to the world frame (one rigid frame-0
  alignment), and the transform is stored in the manifest (`sim_from_input`).
* Support snapping (`--snap-mode rest`): an object whose frame-0 bottom is within 2 cm above the table
  is lowered onto it (constant shift). Any frame that penetrates the table is lifted to 1 mm
  clearance. Large lifts raise a warning (the plane or the pose is likely wrong).

## Dev test (public split only)

```bash
P=/mnt/secondary/v2d/envs/retarget/bin/python; T=~/TestingGrounds/egony/v2d/track3
$P $T/sharpa_task/make_dev_inputs.py --episodes 12 41 0 --out /mnt/secondary/v2d/t3/dev_bundles
for e in 000012 000041 000000; do
  $P $T/to_sharpa_task.py /mnt/secondary/v2d/t3/dev_bundles/dev_gt_episode_$e.npz \
     --out-root /mnt/secondary/v2d/t3/tasks/dev_gt --no-recenter-xy --verify      # + --verify-scene (CoACD, slow)
done
# converter-only error vs GT with the official metric code (reference trajectory, no policy):
$P $T/sharpa_task/rollout_to_devscore.py - /mnt/secondary/v2d/t3/tasks/dev_gt/human_motion_data/v2d_track3/manifests \
   /mnt/secondary/v2d/t3/devscore/pred_taskref --task-reference
/mnt/secondary/v2d/venv-kit/bin/python -I /mnt/secondary/v2d/scratch/track3_perception/t3_devscore.py \
   /mnt/secondary/v2d/kit/v2d_submission_kit ~/TestingGrounds/egony/video_to_data_challenge/track_3 \
   --pred /mnt/secondary/v2d/t3/devscore/pred_taskref --episodes 12 41 0 --cdo
```

### Measured on 2026-10-08 (CPU; public ep 0 pot+lid / 12 cup+rack / 41 pot)

| | ep0 (373 f) | ep12 (181 f) | ep41 (299 f) |
| --- | --- | --- | --- |
| converter wall / peak RSS (incl. IK) | 47 s / 0.97 GB | 50 s / 1.6 GB | 38 s / 1.6 GB |
| IK frame-task error mean R/L | 2.25 / 2.25 cm | 2.23 / 2.19 cm | 2.22 / 2.16 cm (tissue example: 2.3 cm) |
| verify_task.py (+ CPU Newton scene build) | 18/18 | 20/20 | 12/12 |
| task reference vs GT (official scorer) AUC / RPE / MPPE | 1.000 / 0.04 / 0.02 cm | 1.000 / 0.20 / 0.11 cm | 1.000 / 0 / 0.06 cm |
| settle (no hands, 1 s), plain | lid **+25.3 mm / 5.3°** (hulls overlap 9.6 mm), pot 0.8 mm / 0.9° | cup 1.7 mm / 1.0°, rack 0.7 mm / 1.4° | 0.7 mm / 0.8° |
| settle, `--settle-static-start` | lid 0.03 mm / 0.3°, pot 0 | 0 | 0 |
| settled reference vs GT AUC / RPE / MPPE | 0.986 / 0.45 / 0.35 cm | 1.000 / 0.25 / 0.43 cm | 1.000 / 0 / 0.21 cm |

Re-run with the final converter code on 2026-10-08 10:26-10:29 (after the agent crash): ep12 20/20
(incl. CPU Newton scene build: spawn pose |dp|=0, mass/COM/inertia as authored, 91+58 shapes incl. CoACD hulls),
ep41 12/12, ep0 15/15; task reference vs GT unchanged (AUC 1.000, RPE 0.0809, MPPE 0.0627 over the 3 episodes;
log `/mnt/secondary/v2d/t3/logs/rerun_dev_gt.log`, `devscore_taskref_dev_gt_rerun.log`).

Frame-invariance test (`make_frame_test_bundles.py`): ep12 re-expressed in a world tilted 19° with
70° yaw and the table at z=-0.08. The table was re-placed at 0.70 and xy recentred. Input MANO hands
gave the same IK error as above. Keypoint-only hands gave 2.32/3.27 cm. All 15/15 checks passed, and
the official scorer gave AUC 0.9998, RPE 0.22, MPPE 0.11 cm (the residual comes from 3 deliberately
dropped cup frames that were then interpolated).

**Stacked or nested objects.** CoACD hulls (threshold 0.05) of a lid resting in a pot overlap by
about 1 cm where the meshes only touch. With the evaluator's explicit reset the lid then pops 25 mm.
`--settle-static-start` moves each object's initial static period onto its simulated rest pose (CPU
MuJoCo, exact training hulls). This costs ≤0.45 cm RPE / 0.43 cm MPPE against GT on the reference
alone and removes the pop. It is opt-in until a GPU A/B on rollouts confirms it. Candidates: eval ep1
(pot+lid), plus any episode whose `settle` report shows mm-level hull overlap (spoon in pot, cup in
rack). `checks.pair_penetration_report` (surface-based) gives rim false positives of about 20 mm on
nested objects even after settling; use `settle_check.py` for the real answer.

The dev bundles use public GT poses and public scans as stand-ins. They are never packed: no
`submit_meshes`, and `dev_only` is set. Hands are a placeholder (`hands.placeholder_hands`) until
perception provides MANO.

## Hands

* MANO from perception (`hand_<side>_joints` + `_joints_wxyz`, world frame): this is what HaMeR /
  WiLoR FK gives (`ego_recon_loader` re-poses camera→world).
* Keypoints only: orientations come from a Kabsch fit of the template palm.
* Neither (dev): a template hand approaches each moving object (motion segments from object speed,
  dilated 0.6 s before / 0.4 s after), rides rigidly with it, then retreats to a parking pose 35 cm to
  the side and 25 cm above the table. The right hand takes the most-moved object and the left hand the
  second. This exists to exercise the plumbing only.
* MANO for the eval videos is blocked on CPU here. HaMeR/WiLoR need MANO_RIGHT/LEFT.pkl (registration
  at mano.is.tue.mpg.de; not on this machine) plus GPU detectors/backbones. The E2E path
  (`run_ego_reconstruction.py --hand_tracking hamer`) is GPU + Docker.
* **MANO-free keypoint hands for all 40 episodes now exist** (`../hands/`, 2026-10-08):
  `hands_rect.npz` holds the rectified-left frame. `hands/hands_to_bundle.py` puts them into a bundle's world (VO
  `world_T_rect`) as `hand_<side>_joints` / `_valid`, so the converter takes the keypoint path. Use `--hands input` for
  eval bundles so that a missing side never silently falls back to the placeholder. Dev tasks with real hands:
  `/mnt/secondary/v2d/t3/tasks/dev_ourhands` (eps 12, 22).

## GPU (3090) usage

```bash
S=~/TestingGrounds/egony/v2d/track3/sharpa_task; M=/mnt/secondary/v2d/t3/tasks/dev_gt/human_motion_data/v2d_track3/manifests
DRY=1 $S/train_3090.sh $M/episode_000012.json /tmp/x            # print composed config (CPU)
$S/train_3090.sh $M/episode_000012.json /mnt/secondary/v2d/t3/runs/smoke_ep12 \
   training.total_environment_steps=2048000 training.checkpoint.save_interval_environment_steps=1024000
$S/train_3090.sh $M/episode_000012.json /mnt/secondary/v2d/t3/runs/pilot_ep12      # full 250M-step recipe
$S/eval_dev.sh $M /mnt/secondary/v2d/t3/eval/pilot /mnt/secondary/v2d/t3/runs/pilot_ep12 /mnt/secondary/v2d/t3/runs/pilot_ep41
```

Replay memory is exact, not estimated (`replay_budget.py`, CPU, FlashCHORD's own `replay_memory()`):

| task | obs / act / objective terms / critic ctx | 3090 preset (2048 w, 2M, fp16) | recipe (4096 w, 4M, fp32) |
| --- | --- | --- | --- |
| ep12, 2 objects | 722 / 56 / 11 / 16 | **6.42 GiB** (3286 B/transition) | 24.12 GiB (6174 B) |
| ep41, 1 object | 504 / 56 / 11 / 10 | **4.62 GiB** | 17.12 GiB |

Every eval episode has 1 or 2 roster objects (16 with 2, 4 with 1), so the preset needs at most 6.42 GiB of replay;
the canonical recipe's 24.1 GiB of replay alone does not fit the 24 GB card. What is still unmeasured is the
simulator state for 2048 worlds + networks + XLA workspaces. `train_3090.sh` logs `gpu_trace.csv`; read it
after the smoke run before scheduling the pilots (fall back to `scene.world_count=1024
training.updates_per_collection=2.0` if the smoke OOMs).
