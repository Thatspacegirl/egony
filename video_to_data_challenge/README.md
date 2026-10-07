---
license: cc-by-4.0
pretty_name: Video to Data (V2D) Challenge Dataset
tags:
  - robotics
  - robot-learning
  - manipulation
  - human-object-interaction
  - motion-capture
  - video
  - 4d-reconstruction
  - reinforcement-learning
  - egocentric-video
---

# Video to Data (V2D) Challenge Dataset

## Dataset Description

The **Video to Data (V2D) Challenge Dataset** is an NVIDIA-developed benchmark for studying the complete path from human demonstration video to physics-grounded robot behavior. It supports three coupled challenge tracks over shared manipulation tasks: **4D human-object interaction reconstruction**, **robotic grounding**, and **end-to-end egocentric transfer**.

- [Challenge website](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/)
- [Starter toolkit](https://github.com/nvidia-isaac/video_to_data)
- [Dataset repository](https://huggingface.co/datasets/nvidia/video_to_data_challenge)
- Contact: [v2d_challenge@nvidia.com](mailto:v2d_challenge@nvidia.com)

## Challenge Tracks

| Track | Input | Goal | Evaluation summary |
| --- | --- | --- | --- |
| **Track 1: Reconstruction** | Monocular third-person RGB video | Recover the human, object pose, and object geometry as a metric 4D human-object interaction scene in a consistent world frame | Reconstruction accuracy and physical plausibility relative to the multi-view reference |
| **Track 2: Robotic Grounding** | Third-person 4D human-object interaction trajectories at different input-noise tiers | Retarget the demonstration and learn an executable policy for the robotic embodiment and simulator | Object-tracking performance at each input tier |
| **Track 3: Egocentric** | Egocentric human demonstration video | Produce an executable robot policy through either an explicit reconstruct-and-retarget pipeline or an implicit end-to-end method | Final results produced with the official evaluation script |

### Track 1: Reconstruction

Track 1 evaluates monocular 4D human-object interaction reconstruction under challenging conditions including occlusion, bimanual coordination, and long-horizon manipulation.

Participants reconstruct:

- the human body and hands;
- object pose trajectories;
- object geometry; and
- metric scale.

Track 1 is evaluated along two equally weighted axes:

1. **Accuracy**
   - Chamfer distance to the multi-view human mesh
   - Chamfer distance to the multi-view object mesh
2. **Physical plausibility**
   - Human-joint acceleration error
   - Object acceleration error
   - Contact penetration error

### Track 2: Robotic Grounding

Track 2 measures how upstream reconstruction quality affects human-to-robot transfer and downstream policy learning.

The dataset provides three input tiers:

1. **Tier 1 — Clean multi-view capture:** an upper-bound input for upstream reconstruction.
2. **Tier 2 — Synthetic corruption:** trajectories with jitter, dropout, and contact errors sampled from Track 1 error distributions.
3. **Tier 3 — Off-the-shelf reconstruction:** trajectories produced by current reconstruction methods.

Each tier is scored separately. Metrics include **AUC**, **SP-SR**, **MP-SR**, and **MPPE**, as defined by the challenge evaluation protocol.

### Track 3: Egocentric

Track 3 evaluates the full pipeline from egocentric human video to robot behavior. The track is method-agnostic: participants may use an explicit reconstruction-and-retargeting pipeline, an end-to-end model, a pretrained vision-language-action model, a world-action model, or a hybrid approach.

The NVIDIA-collected data includes human-object manipulation recordings and associated assets made available for development and evaluation. Depending on the released split, these assets may include:

- egocentric videos;
- motion-capture trajectories;
- sequence metadata;
- textured 3D object meshes; and
- URDF object descriptions.

Use `eval_e2e.py` from the starter toolkit to package the required reconstructions and recorded policy evaluations.

## Download

Install the Hugging Face Hub client:

```bash
python -m pip install -U "huggingface_hub"
```

Download the complete dataset repository while preserving its file structure:

```bash
hf download nvidia/video_to_data_challenge \
  --repo-type dataset \
  --local-dir ./video_to_data_challenge
```

If authentication is requested, first run:

```bash
hf auth login
```

The same operation can be performed from Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/video_to_data_challenge",
    repo_type="dataset",
    local_dir="./video_to_data_challenge",
)
```

Large assets are stored using Hugging Face's large-file infrastructure. Make sure sufficient disk space is available before downloading the complete repository.

## Support

For challenge or dataset questions, contact [v2d_challenge@nvidia.com](mailto:v2d_challenge@nvidia.com). For software issues, use the issue tracker in the [Video to Data repository](https://github.com/nvidia-isaac/video_to_data/issues).
