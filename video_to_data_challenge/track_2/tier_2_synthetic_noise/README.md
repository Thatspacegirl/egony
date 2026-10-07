# Track 2 — Tier 2: synthetic noise

This tier provides synthetically perturbed human and object trajectories while
preserving identity parameters, visibility, episode timing, videos, and meshes.
Object poses remain **`world_T_object` in the source OpenCV world frame**.

| | |
|---|---|
| Episodes | 30 |
| Frames | 22,990 (440–1,003 per episode, mean 766) |
| Rate | 30 fps nominal |
| Tasks | 30 |
| Cameras | 1 exocentric stream, 30 clips |
| Objects | 29, each with a textured mesh |

## Layout

```
track_2/tier_2_synthetic_noise/
├── data/chunk-000/episode_0000NN.parquet    human + object pose, one row per frame
├── videos/chunk-000/observation.images.exo_camera/episode_0000NN.mp4
├── mesh/<object>/<object>.glb               textured mesh
├── support_surface/<sequence>_support.usda  static G1 support geometry
└── meta/
    ├── info.json                            schema, counts, tasks, objects
    ├── episodes_metadata.jsonl              episode → sequence, task, object, mesh
    ├── episodes.jsonl                       episode lengths and tasks
    ├── episodes_stats.jsonl                 per-episode feature statistics
    └── tasks.jsonl                          task descriptions
```

## Pose data

Each parquet has **13 columns**, one row per video frame:

```
observation.human.pose                   fixed_size_list<float32, 231>
observation.human.translation            fixed_size_list<float32, 3>
observation.human.identity_coeffs        fixed_size_list<float32, 45>
observation.human.scale_params           fixed_size_list<float32, 68>
observation.human.bone_length_flexibles  fixed_size_list<float32, 6>
observation.object.pose                  fixed_size_list<float32, 7>
observation.object.visible               bool
timestamp  frame_index  episode_index  index  task_index  next.done
```

Each episode tracks exactly one object. Its name is stored in the parquet's schema
metadata under `objects` and in `meta/episodes_metadata.jsonl` with the source sequence,
task, and mesh path.

An object's name **is** its mesh folder: `white_desk` →
`mesh/white_desk/white_desk.glb`.

### Conventions

- **`observation.human.pose` contains 77 local rotation vectors in radians**, flattened
  in the joint order declared by `meta/info.json`.
- Human translation is in metres. Identity, scale, and flexible-bone values are the
  corresponding SOMA-X body-model parameters.
- **`observation.object.pose` is `[x, y, z, qw, qx, qy, qz]`** — translation in metres,
  **quaternion w-first**.
- **Object poses are `world_T_object`** in the source OpenCV world frame. Each parquet
  declares this in schema metadata under `pose_convention`.
- When `observation.object.visible` is false the object pose is all zeros — a zero
  quaternion, not a rotation. **Filter on visibility before using a pose.**

## Cameras

| Stream | Resolution | Kind |
|---|---|---|
| `exo_camera` | 1536×1152 | colour, static third-person |

Video is **frame-exact with the parquets** — row *i* corresponds to frame *i* in the
clip, verified across all 30 episodes. Each file is a byte-identical copy of its selected
source H.264 clip; no concatenation or re-encoding was performed.

The supplied export has **no camera intrinsics or extrinsics**, so meshes cannot be
projected into the images from this dataset alone. The source calibration sequence
identifier is preserved per episode, but it is not a calibration matrix.

## Ground plane

Every episode declares `ground_plane/<sequence_id>.json` in
`meta/episodes_metadata.jsonl`. The sidecar contains normalized `[a, b, c, d]`
coefficients for `ax + by + cz + d = 0` in the same source OpenCV world frame
as `observation.object.pose`. SOMA-to-G1 retargeting transforms this fitted
plane through its first-frame anchor. Do not replace it with a horizontal
world-plane estimate when reproducing a retargeted training reference.

## Support surface

Every episode declares `support_surface/<sequence_id>_support.usda` in
`meta/episodes_metadata.jsonl`. These static collision stages use metres and Z-up in
the released G1 retargeted world. Place the declared file at
`<motion-root>/whole_body/reconstructed_stage/<sequence_id>_support.usda` alongside
the G1 retargeted motion dataset before grounding or training. Ground-only episodes
have a valid empty stage at the same path.
