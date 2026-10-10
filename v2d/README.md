# NVIDIA Isaac Video-to-Data challenge: Track 1 and Track 3 pipelines

Work in progress for the 2026 V2D challenge. This directory holds the code for Track 1 (monocular 4D human-object
reconstruction) and Track 3 (egocentric stereo video to Sharpa Wave hand policy). Track 2 is not published here.

Nothing here has been submitted to Kaggle. The numbers below come from the organisers' own scoring code run locally on
data that has ground truth, so they are development numbers, not leaderboard entries.

## What is better than the public leaderboard (snapshot 2026-10-09)

| Track | Metric | Ours (local dev) | Best on the board | Note |
|---|---|---|---|---|
| 1 | CD-H (human Chamfer, cm, lower is better) | 8.3 | 14.2 | 5 FORM-HOI validation episodes |
| 3 | MP-SR (higher is better) | 0.60 | 0.50 | 20 public episodes |
| 3 | RPE (lower is better) | 2.38 | 3.85 | 20 public episodes |

## What is not better yet

| Track | Metric | Ours (local dev) | Best on the board |
|---|---|---|---|
| 1 | CD-O (object Chamfer, cm) | 40.1 (about 20 without episode 1, which fails at 122) | 22.1 |
| 1 | ACC-H / ACC-O | 0.156 / 0.301 | 0.139 / 0.106 |
| 1 | PEN | 0.0009 on one real episode (0.0002 on simulated input) | 0.00058 |
| 3 | AUC / SP-SR | 0.718 / 0.70 | 0.765 / 0.80 |
| 3 | MPPE (cm) | 4.36 | 3.00 |
| 3 | CD-O | 0.440 | 0.112 (second team's value; the AUC leader has 0.583) |

Caveats that matter:

- Track 1 uses 5 validation episodes of the FORM-HOI dataset; the smoothing was tuned on episodes of the same set.
- The Track 3 numbers score the object trajectories our perception recovers, not rollouts of a trained policy. They
  show what a perfect policy copying the recovered trajectories would reach. No Track 3 policy has been trained yet.
- Track 3 mesh choices were made on the public split and then applied unchanged to the evaluation split.
- The public leaderboards have 2 to 7 teams.

## Layout

- `track1/` : SAM 3 masks, SAM 3D Objects / TRELLIS meshes, CARI4D, exact MHR conversion, model-agnostic smoothing and
  penetration cleanup, local scorer. See `track1/README.md`.
- `track3/` : stereo depth, visual odometry, SAM 3 masks, mesh selection, FoundationPose tracking, perception bundle and
  Sharpa task builder, local dev scorer. See `track3/README.md`.

Paths in the scripts are absolute paths of the machine they were developed on (`/mnt/secondary/v2d/...`).

## Not included

Model weights, challenge data (get it from `nvidia/video_to_data_challenge` on Hugging Face), generated meshes,
predictions, and anything covered by the MANO or SMPL-X licences. The MANO converter in `track3/hands/mano_clean.py`
is code only and needs your own licensed MANO files.
