#!/bin/bash
# Build FoundationPose's CUDA/C++ extensions for the host venv t3-fpose (no Docker, no sudo, no GPU needed).
# Toolchain: conda env cuda124-toolchain (nvcc 12.4.131 + eigen 3.4 + boost + pybind11), system gcc 13.3.
set -ex
E=/mnt/secondary/v2d/envs/t3-fpose
C=/mnt/secondary/v2d/envs/cuda124-toolchain
S=/mnt/secondary/v2d/envs/src
FP=/mnt/secondary/v2d/video_to_data/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose
export UV_CACHE_DIR=/mnt/secondary/uv-cache
export CUDA_HOME=$C PATH=$C/bin:$E/bin:$PATH LIBRARY_PATH=$C/lib CPATH=$C/include:$C/include/eigen3
export TORCH_CUDA_ARCH_LIST="8.6" FORCE_CUDA=1 MAX_JOBS=${MAX_JOBS:-4} CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
UV=~/.local/bin/uv
# 1) nvdiffrast (NVlabs, v0.4.0) -- compiles its CUDA rasterizer extension at install time
$UV pip install -p $E/bin/python --no-build-isolation --no-deps $S/nvdiffrast
# 2) pytorch3d v0.7.9 from source
$UV pip install -p $E/bin/python --no-build-isolation --no-deps $S/pytorch3d
# 3) mycpp (pose clustering, used by FoundationPose.register) -- Boost>=1.89 has no boost_system lib -> drop it
rm -rf $S/fp_mycpp && cp -r $FP/mycpp $S/fp_mycpp
sed -i 's/COMPONENTS system program_options/COMPONENTS program_options/' $S/fp_mycpp/CMakeLists.txt
mkdir -p $S/fp_mycpp/build && cd $S/fp_mycpp/build
cmake .. -DPYTHON_EXECUTABLE=$E/bin/python -DCMAKE_PREFIX_PATH=$C -Dpybind11_DIR=$($E/bin/python -c "import pybind11;print(pybind11.get_cmake_dir())") -DBoost_ROOT=$C -DEigen3_DIR=$C/share/eigen3/cmake
make -j4
mkdir -p $FP/mycpp/build && cp -v $S/fp_mycpp/build/mycpp*.so $FP/mycpp/build/
# 4) mycuda (common + gridencoder), built in place so `from bundlesdf.mycuda import common` works (*.so is gitignored)
cd $FP/bundlesdf/mycuda && $E/bin/python setup.py build_ext --inplace -j4
ls -la $FP/bundlesdf/mycuda/*.so $FP/mycpp/build/*.so
echo BUILD_DONE
