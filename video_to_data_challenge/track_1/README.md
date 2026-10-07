# Track 1

Video-only single-view human-object interaction sequences for object tracking. Track 1
contains 30 selected episodes from the FORM-HOI multiview recordings. Each episode provides
one static third-person RGB video, its physical camera view, the target object identifier
and prompt, and the original action description.

| | |
|---|---|
| Episodes | 30 |
| Frames | 16,563 (360–877 per episode, mean 552) |
| Rate | 30 fps |
| Tasks | 22 original action descriptions |
| Cameras | 1 stream per episode, 30 clips, 4 distinct physical cameras |
| Target objects | 10 |

## Layout

```
track_1/
├── data/chunk-000/episode_0000NN.parquet    LeRobot frame indexes only
├── videos/chunk-000/observation.images.exo_camera/episode_0000NN.mp4
└── meta/
    ├── info.json                            LeRobot v2.1 schema and counts
    ├── episodes.jsonl                       episode lengths and action descriptions
    ├── episodes_metadata.jsonl              sequence, camera, and target-object metadata
    ├── episodes_stats.jsonl                 video and index statistics
    └── tasks.jsonl                          original action descriptions
```

## Per-episode metadata

Each record in `meta/episodes_metadata.jsonl` contains only:

- `episode_index`
- `sequence_id`
- `camera`
- `object`
- `object_prompt`
- `video_key`

The video path is determined from `meta/info.json` using `episode_index` and `video_key`.
The `object` and `object_prompt` fields identify the object to track. The corresponding
action description is resolved through the episode's `task_index` and
`meta/tasks.jsonl`.

## Video

Videos are 1536×1152 H.264 (`yuv420p`) at 30 fps with no audio. They are byte-identical
copies of the selected source videos and were not re-encoded.

| Camera | Episodes |
|---|---:|
| `back_stereo_camera_left` | 5 |
| `front_stereo_camera_left` | 9 |
| `left_stereo_camera_left` | 11 |
| `right_stereo_camera_left` | 5 |

## LeRobot v2.1 indexes

The per-episode Parquet files contain only the structural columns required to align video
frames with LeRobot episode and task metadata:

- `timestamp`
- `frame_index`
- `episode_index`
- `index`
- `task_index`
- `next.done`

No human pose, body reconstruction, object pose, visibility annotation, or object mesh is
included in the current revision.

Episode indexes follow the 30-sequence selection order. Each of the 10 target objects has three episodes.

RGB statistics for retained videos are preserved. For new videos, channel statistics use
uniformly sampled frames (sample count `min(N, max(100, min(10000, floor(N**0.75))))`),
area-downsampled to 154×116 and normalized to [0, 1]. This downsampling is used only
for statistics; the distributed videos retain their original bytes and resolution.
