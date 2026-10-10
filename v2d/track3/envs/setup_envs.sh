#!/bin/bash
# Reproduce the Track 3 perception host venvs WITHOUT Docker (exact commands used on 2026-10-08).
# Everything lands on /mnt/secondary; nothing needs sudo or a GPU to install.
set -ex
export UV_CACHE_DIR=/mnt/secondary/uv-cache
UV=~/.local/bin/uv
M=/mnt/secondary/v2d/video_to_data/reconstruction/modules          # nvidia-isaac/video_to_data @ a709404e
E=/mnt/secondary/v2d/envs
W=/mnt/secondary/v2d/weights
mkdir -p $E/src $W

# ---------------------------------------------------------------- (a)+(d) t3-geom: stereo depth + VO (py3.12, CUDA 12)
$UV venv $E/t3-geom --python 3.12
$UV pip install -p $E/t3-geom/bin/python "tensorrt-cu12==10.7.0.post1" "tensorrt-cu12-libs==10.7.0.post1" \
  "tensorrt-cu12-bindings==10.7.0.post1" "cuda-python==12.6.*" onnx "onnxruntime-gpu[cuda,cudnn]==1.22.0" "numpy<2.3" \
  pillow scipy h5py pyyaml tqdm opencv-python-headless -e "$M/v2d_common[io]" -e $M/v2d_mv -e $M/v2d_foundation_stereo/lib \
  $W/cuvslam_wheels/cuvslam-17.0.0+cu12-cp312-abi3-manylinux_2_39_x86_64.whl \
  nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12
# runtime: source envs/t3-geom.env (puts the pip CUDA/TensorRT libs on LD_LIBRARY_PATH for cuvslam)

# ---------------------------------------------------------------- (b) t3-seg: GroundingDINO(HF) + SAM2.1 + SAM3 (py3.12)
$UV venv $E/t3-seg --python 3.12
$UV pip install -p $E/t3-seg/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" \
  "transformers>=4.57" accelerate huggingface_hub safetensors "numpy<2.3" pillow opencv-python-headless scipy tqdm \
  hydra-core iopath einops timm pycocotools setuptools wheel kernels
[ -d $E/src/sam2 ] || git clone https://github.com/facebookresearch/sam2.git $E/src/sam2      # @ 2b90b9f
SAM2_BUILD_CUDA=0 $UV pip install -p $E/t3-seg/bin/python --no-build-isolation --no-deps -e $E/src/sam2
$UV pip install -p $E/t3-seg/bin/python -e "$M/v2d_common[io]" -e $M/v2d_mv opencv-python-headless flask
$UV pip install -p $E/t3-seg/bin/python --no-deps -e $M/v2d_sam2/lib   # decord has no cp312 wheel; we feed JPEG dirs

# ---------------------------------------------------------------- (c) t3-fpose: FoundationPose (py3.11, torch 2.5.1 cu124 = module Docker base)
CONDA_PKGS_DIRS=/mnt/secondary/conda-pkgs ~/miniconda3/bin/conda create -y -p $E/cuda124-toolchain --override-channels \
  -c nvidia/label/cuda-12.4.1 -c conda-forge cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-nvtx-dev cuda-cccl \
  cuda-nvrtc-dev cuda-profiler-api "eigen=3.4" libboost-devel "pybind11>=2.10"
ln -sfn lib $E/cuda124-toolchain/lib64
$UV venv $E/t3-fpose --python 3.11
$UV pip install -p $E/t3-fpose/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124 "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" \
  "numpy==1.26.4" scipy joblib scikit-learn ruamel.yaml trimesh pyyaml "opencv-python-headless<4.12" \
  "opencv-contrib-python-headless<4.12" imageio "open3d==0.19.0" transformations "warp-lang<1.9" einops kornia pyrender \
  scikit-image meshcat webdataset omegaconf pypng roma seaborn h5py fast_simplification gdown pandas psutil ninja \
  setuptools wheel "pybind11>=2.10" iopath fvcore "tensorrt-cu12==10.7.0.post1" "tensorrt-cu12-libs==10.7.0.post1" \
  "tensorrt-cu12-bindings==10.7.0.post1" "cuda-python==12.6.*" pydantic pytransform3d tqdm av
$UV pip install -p $E/t3-fpose/bin/python --no-deps "kaolin==0.18.0" \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
$UV pip install -p $E/t3-fpose/bin/python usd-core comm ipycanvas ipyevents jupyter_client pygltflib dataclasses-json flask
[ -d $E/src/nvdiffrast ] || git clone https://github.com/NVlabs/nvdiffrast $E/src/nvdiffrast              # v0.4.0
[ -d $E/src/pytorch3d ] || git clone --branch v0.7.9 --depth 1 https://github.com/facebookresearch/pytorch3d $E/src/pytorch3d
bash "$(dirname "$0")/build_fpose_exts.sh"      # nvdiffrast, pytorch3d, mycpp, mycuda for sm_86 (MAX_JOBS=4, ~40 min CPU)
$UV pip install -p $E/t3-fpose/bin/python --no-deps -e "$M/v2d_common" -e "$M/v2d_mv" -e $M/v2d_mesh/lib \
  -e $M/v2d_foundation_pose/lib
