"""Track 3 policy-side helpers: perception bundle -> FlashCHORD floating-Sharpa task.

Entry point: ``../to_sharpa_task.py``. See README.md in this directory.
"""

BUNDLE_SCHEMA = "t3_perception_v1"
DATASET_NAME = "v2d_track3"
ROBOT_NAME = "sharpa_wave"
TASK_FPS = 20.0  # eval videos are 20/1 fps; the packer scores rollout step i as video frame i
