# Track 3 Public Release

See the [dataset card](v2d-challenge-track3.md) for intended uses, ownership, and
ethical considerations.

## License

The Track 3 dataset and all accompanying assets, including videos, trajectories,
metadata, textured 3D meshes, and URDF object descriptions, are licensed under
[Creative Commons Attribution 4.0 International (CC BY-4.0)](https://creativecommons.org/licenses/by/4.0/).
See the [full license terms](https://creativecommons.org/licenses/by/4.0/legalcode).

## Overview

Human manipulation of everyday objects, captured with a Vicon motion-capture system and
two OAK camera rigs. Every tracked object has a textured 3D scan, and object poses are
expressed **in that scan's own frame**, so placing a mesh at a logged pose puts it where
the real object was.

| | |
|---|---|
| Episodes | 20 |
| Frames | 5,246 (mean 262) |
| Rate | 20.25 fps nominal |
| Tasks | 10 |
| Cameras | 3 ego streams, 60 clips |
| Objects | 10, each with a textured mesh |

## Layout

```
track_3/
└── public/
    ├── data/ and videos/                    trajectories and three ego streams
    ├── mesh/ and urdf/                      object assets
    └── meta/                                metadata and calibration

```

## Pose data

Each parquet has **7 columns**, one row per video frame:

```
observation.objects   list<struct{name: string,
                                  pose: fixed_size_list<float32, 7>,
                                  visible: bool}>
timestamp  capture_time  frame_index  episode_index  index  task_index
```

`observation.objects` holds only the objects that episode tracks (1–2, occasionally more).
**Membership is constant within an episode** and sorted by name, so `objects[i]` is the
same object on every frame and no per-frame branching is needed. The episode's object list
is also in the parquet's schema metadata under `objects`.

An object's `name` **is** its mesh folder: `white_pot` → `mesh/white_pot/white_pot.glb`.

### Conventions

- **`pose` is `[x, y, z, qw, qx, qy, qz]`** — translation in metres, **quaternion w-first**.
- **Poses are `world_T_object`**, the frame of the scanned mesh, not the mocap rigid body.
  Each parquet declares this in schema metadata under `pose_convention`.
- **x and y are centred per episode** on the mean of that episode's visible object
  positions. **z is not centred**, so height above the table stays physical (~0.61 m
  resting). The offset is baked into the values and is not stored, so episodes do not
  share a common origin.
- When `visible` is false the pose is all zeros — a zero quaternion, not a rotation.
  **Filter on `visible` before using a pose.**

## Cameras

| Stream | Resolution | Kind |
|---|---|---|
| `ego_cam_a` | 2028×1520 | colour, head-mounted |
| `ego_cam_b`, `ego_cam_c` | 1280×800 | mono stereo pair, 75 mm baseline |

Video is **frame-exact with the parquets** — row *i* corresponds to frame *i* in every
clip, verified across all 60 ego clips.

Only the ego streams are distributed in this repository. The exocentric clips were
intentionally moved to a separate backup. `meta/camera_calibration.json` retains
calibration for both originally captured rigs, including the backed-up exo streams.
It has **no camera-to-mocap-world extrinsic**, so meshes cannot be projected into the
images without solving that first.
