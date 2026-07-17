#!/usr/bin/env bash
#
# Configure + build the Chat app inside the Mila build container.
#
# Source is read from the bind-mounted repo at /mila; build artifacts (the C++23
# module BMIs, objects, and the ChatApp binary) are written to /build, a
# container-local volume. Keeping the build tree OFF the bind mount matters: the
# module build is metadata-heavy BMI I/O, which is slow across the host->container
# filesystem boundary (the same reason the WSL build uses an ext4 clone, not /mnt).
#
# The flag set mirrors the validated CI toolchain path (clang-21 modules, gcc-15
# nvcc host, CUDA 13.3) and trims the build to the Chat target: no tests, samples,
# profiling, docs, or Python binding -- none are on the path to a running Chat.
set -euo pipefail

# CUDA arch for targets that do not pin their own. NOTE: the Mila library target
# hard-sets CUDA_ARCHITECTURES "75;80;86;89;90" (a portable fat binary for
# find_package consumers), and a target property overrides this global -- so the
# library always builds all five arches regardless of MILA_CUDA_ARCH. sm_89 (Ada /
# RTX 4070) is in that set, so the result runs on the 4070; MILA_CUDA_ARCH only
# affects any non-pinning target. Trimming the library to one arch for a faster
# container build is a follow-up (an opt-in override in Mila/CMakeLists.txt).
: "${MILA_CUDA_ARCH:=89}"
: "${MILA_BUILD_TYPE:=Release}"

# Build parallelism is a MEMORY limit, not a core-count one. The heaviest module
# TUs (OperationTraits + the CUDA op .ixx units instantiate the full
# device x precision x quantization dispatch table) each cost several GB to compile;
# running $(nproc) of them at once starves the Docker VM's RAM -> swap thrash -> the
# engine goes unresponsive (a false "hang" right at that frontier). Cap conservatively
# and let a RAM-rich host raise it: MILA_BUILD_JOBS=8 (or bump the WSL2 VM's memory
# in %USERPROFILE%\.wslconfig).
: "${MILA_BUILD_JOBS:=4}"

SRC=/mila
BUILD=/build

cmake -S "${SRC}" -B "${BUILD}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${MILA_BUILD_TYPE}" \
    -DCMAKE_C_COMPILER=clang-21 \
    -DCMAKE_CXX_COMPILER=clang++-21 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCUDAToolkit_ROOT=/usr/local/cuda \
    -DCMAKE_CUDA_HOST_COMPILER=gcc-15 \
    -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler" \
    -DCMAKE_CUDA_ARCHITECTURES="${MILA_CUDA_ARCH}" \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DMILA_ENABLE_ADAPTORS=ON \
    -DMILA_ENABLE_CUDA=ON \
    -DMILA_ENABLE_TESTING=OFF \
    -DMILA_ENABLE_SAMPLES=OFF \
    -DMILA_ENABLE_PROFILING=OFF \
    -DMILA_ENABLE_DOCS=OFF \
    -DMILA_ENABLE_PYTHON_BINDINGS=OFF \
    -DMILA_INSTALL=OFF

cmake --build "${BUILD}" --target ChatApp -- -j "${MILA_BUILD_JOBS}"

echo "Built ${BUILD}/ChatApp (arch sm_${MILA_CUDA_ARCH}, ${MILA_BUILD_TYPE}, -j ${MILA_BUILD_JOBS}). Run it with: mila-chat"
