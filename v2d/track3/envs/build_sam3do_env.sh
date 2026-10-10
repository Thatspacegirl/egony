#!/bin/bash
# Track 3: host venv for SAM 3D Objects (facebook/sam-3d-objects) WITHOUT Docker, following the toolkit's
# modules/v2d_sam3d/docker/Dockerfile (inference-only subset; the repo's full requirements.txt is a training env).
# Base = a full copy of t3-fpose (python 3.11, torch 2.5.1+cu124, kaolin 0.18, nvdiffrast 0.4, pytorch3d 0.7.9 built for
# sm_86), then: gsplat (prebuilt pt25cu124 wheel), spconv/cumm cu124, lightning, MoGe @ a8c3734 (pin of the Dockerfile),
# sam-3d-objects (--no-deps, from /mnt/secondary/v2d/envs/src/sam-3d-objects), v2d_common + v2d_sam3d lib (editable).
# flash-attn is NOT installed: sam3d only selects it on A100/H100/H200; the 3090 uses torch SDPA.
#   bash build_sam3do_env.sh            (CPU only, ~10 min, no sudo)
set -ex
export UV_CACHE_DIR=/mnt/secondary/uv-cache
UV=~/.local/bin/uv
SRC=/mnt/secondary/v2d/envs/t3-fpose
E=/mnt/secondary/v2d/envs/t3-sam3do
M=/mnt/secondary/v2d/video_to_data/reconstruction/modules
S3D=/mnt/secondary/v2d/envs/src/sam-3d-objects      # git f91db411 (2026-06-02)
[ -d $E ] || cp -a $SRC $E
sed -i "s|$SRC|$E|g" $E/bin/activate* $E/pyvenv.cfg 2>/dev/null || true
PY=$E/bin/python
$UV pip install -p $PY --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124 "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" \
  loguru timm astor easydict "networkx==3.2.1" "spconv-cu124==2.3.8" cumm-cu124 "lightning==2.3.3" xatlas pyvista \
  pymeshfix igraph imageio omegaconf hydra-core seaborn huggingface_hub roma einops optree fvcore jsonlines \
  "pyrender==0.1.45" "pyglet==2.1.15" plyfile "setuptools<70"
$UV pip install -p $PY "gsplat==1.4.0" --index-url https://docs.gsplat.studio/whl/pt25cu124 --extra-index-url https://pypi.org/simple \
  || $UV pip install -p $PY "gsplat==1.4.0"
$UV pip install -p $PY --no-deps "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"  # pin of MoGe a8c3734
$UV pip install -p $PY --no-deps "moge @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"
$UV pip install -p $PY --no-deps $S3D
$UV pip install -p $PY --no-deps -e "$M/v2d_common" -e "$M/v2d_sam3d/lib"
cat > $E/v2d_env.sh <<EOF
# source me: Track 3 SAM 3D Objects host env
export HF_HOME=/mnt/secondary/caches/huggingface TORCH_HOME=/mnt/secondary/v2d/t3/sam3do/torch_home
export HYDRA_FULL_ERROR=1 LIDRA_SKIP_INIT=1 PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export CUDA_HOME=/mnt/secondary/v2d/envs/cuda124-toolchain CONDA_PREFIX=/mnt/secondary/v2d/envs/cuda124-toolchain
export WARP_CACHE_PATH=/mnt/secondary/v2d/warp_cache
export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=\${MKL_NUM_THREADS:-4}
EOF
source $E/v2d_env.sh
CUDA_VISIBLE_DEVICES="" $PY -c "
import torch, kaolin, pytorch3d, spconv, gsplat, moge, utils3d, lightning, hydra, omegaconf
import sam3d_objects
print('torch', torch.__version__, 'gsplat', gsplat.__version__, 'spconv', spconv.__version__)
"
$UV pip freeze -p $PY > $E/freeze.txt
echo BUILD_DONE
