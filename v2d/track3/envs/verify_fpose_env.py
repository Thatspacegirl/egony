"""CPU-only import/functional check of the host FoundationPose stack (t3-fpose venv).

  CUDA_VISIBLE_DEVICES= /mnt/secondary/v2d/envs/t3-fpose/bin/python -I envs/verify_fpose_env.py
"""
import glob
import importlib
import os
import sys

FP = "/mnt/secondary/v2d/video_to_data/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose"
sys.path.insert(0, FP)
import numpy as np  # noqa: E402
import torch  # noqa: E402

print("torch", torch.__version__, "| built for CUDA", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
for m in ["torchvision", "pytorch3d", "kaolin", "open3d", "trimesh", "tensorrt", "warp", "kornia", "nvdiffrast"]:
    mod = importlib.import_module(m)
    print(f"  {m:12s} {getattr(mod, '__version__', '?')}")
import pytorch3d._C  # noqa: E402,F401  compiled CUDA/C++ ops
from pytorch3d.transforms import so3_exp_map  # noqa: E402
print("  pytorch3d._C:", os.path.basename(pytorch3d._C.__file__), "| so3_exp_map:",
      so3_exp_map(torch.zeros(1, 3)).shape)
import nvdiffrast.torch as dr  # noqa: E402
import _nvdiffrast_c  # noqa: E402  compiled at install time (nvdiffrast 0.4.0), sm_86
print("  nvdiffrast compiled plugin:", os.path.basename(_nvdiffrast_c.__file__),
      "| RasterizeCudaContext:", hasattr(dr, "RasterizeCudaContext"))
# mycpp: pose clustering used by FoundationPose.register -- run it for real on CPU
import mycpp.build.mycpp as mycpp  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
R = Rotation.random(200, random_state=0).as_matrix()
poses = np.tile(np.eye(4), (200, 1, 1)).astype(np.float32)
poses[:, :3, :3] = R
out = mycpp.cluster_poses(30, 99999, poses, np.eye(4, dtype=np.float32)[None])
print(f"  mycpp.cluster_poses: 200 random rotations -> {len(out)} clusters (30 deg)")
from bundlesdf.mycuda import common  # noqa: E402,F401
sys.path.insert(0, f"{FP}/bundlesdf/mycuda")  # what `pip install -e mycuda` does in the Docker image
import gridencoder  # noqa: E402,F401  (BundleSDF NeRF only; built in place next to common)
print("  mycuda common + gridencoder import OK")
# FoundationPose python stack (Utils pulls pytorch3d renderer, nvdiffrast, kaolin, open3d, warp)
import Utils  # noqa: E402
print("  FoundationPose Utils: mycpp", Utils.mycpp is not None, "| mycuda common", Utils.common is not None)
import estimater  # noqa: E402,F401
from v2d.foundation_pose.lib import foundation_pose_tracker, run_video_to_poses, run_ekf_smoothing, backends  # noqa
print("  v2d.foundation_pose.lib imports OK; backends:", backends.SUPPORTED_BACKENDS)
from learning.training.predict_score import ScorePredictor  # noqa: E402,F401
from learning.training.predict_pose_refine import PoseRefinePredictor  # noqa: E402,F401
w = "/mnt/secondary/v2d/weights/foundationpose/nvlabs_pytorch"
for run in ["2023-10-28-18-33-37", "2024-01-11-20-02-45"]:
    sd = torch.load(f"{w}/{run}/model_best.pth", map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    print(f"  NVLabs weights {run}: {len(sd)} tensors, {sum(v.numel() for v in sd.values()) / 1e6:.1f} M params")
print("FPOSE_ENV_OK")
