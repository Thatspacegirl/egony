# Track 2: trajectory variants

Track 2 contains the same 30 human-object interaction episodes in two challenge tiers. Both tiers use the same episode order, 22,990-frame timeline, tasks, exocentric RGB observation, fitted ground planes, and G1 support surfaces.

| Tier | Directory | Human trajectory | Object trajectory and mesh |
|---|---|---|---|
| Tier 1: multiview caption | [`tier_1_multiview_caption/`](tier_1_multiview_caption/) | Original SOMA-X reconstruction | Original pose and aligned mesh |
| Tier 2: synthetic noise | [`tier_2_synthetic_noise/`](tier_2_synthetic_noise/) | Synthetically perturbed human trajectory | Synthetically perturbed object trajectory; original mesh |

Each directory is a complete LeRobot v2.1 dataset with its own `README.md`, `meta/`, `data/`, `videos/`, and `mesh/`. Select one tier directory as the dataset root. Do not concatenate tiers: episode and global frame indices intentionally overlap so results can be compared frame by frame.

The `observation.images.exo_camera` video stream is byte-identical across tiers and is the monocular input participants should use for reconstruction. “Multiview caption” is the challenge tier name; this staged participant release exposes only the selected exocentric stream. Reconstruction results are not published; participants generate them with their own method before retargeting and grounding.
