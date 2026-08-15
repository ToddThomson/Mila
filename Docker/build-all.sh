#!/usr/bin/env bash
#
# Configure + build the FULL user-facing Mila product set inside the build container.
#
# This is the "give me everything" companion to build-chat.sh. Where build-chat.sh
# trims to just the mila-chat target (the fast path for someone who only wants to run
# Chat), this builds the whole set a user might want from the known-good environment:
#   - the Mila library
#   - the samples (MNIST, Bard) -- read and run them to learn the API
#   - the Chat adaptor (mila-chat)
#   - the Python binding (mila.pyd / Mila.Bindings) -- stand up the Inference Server
#
# PURPOSE: end-user convenience -- a completely known build environment that produces
# the full product in one step. It is deliberately NOT a maintainer portability or
# test gate: cross-compiler portability and the CUDA test suite are owned by CI and
# the WSL build (see RELEASING.md, "Cross-platform build policy"). Tests are therefore
# OFF by default here; a user who wants to sanity-check their build can opt in with
# MILA_ENABLE_TESTING=ON.
#
# Source is read from the bind-mounted repo at /mila; build artifacts (the C++23
# module BMIs, objects, and binaries) go to /build, a container-local volume -- kept
# OFF the bind mount because the module build is metadata-heavy BMI I/O that is slow
# across the host->container filesystem boundary (same reason build-chat.sh does it).
#
# The flag set mirrors the validated CI/WSL toolchain (clang-21 modules, gcc-15 nvcc
# host, CUDA 13.3).
set -euo pipefail

# CUDA arch(es) to build for. Default `native`: CMake detects the arch of the GPU(s)
# actually present at configure time (the container reserves the host GPUs -- see
# docker-compose.yml), so the build matches THIS machine's hardware, and a host with
# multiple cards of different archs builds for all of them. Passed to BOTH the global
# CMAKE_CUDA_ARCHITECTURES and the library's MILA_LIBRARY_CUDA_ARCHITECTURES override,
# so the library compiles only these arch(es) instead of its default 5-arch portable
# fat binary -- the dominant build-time saving. Override for a specific arch,
# cross-building, or a GPU-less configure: MILA_CUDA_ARCH=89.
: "${MILA_CUDA_ARCH:=native}"
: "${MILA_BUILD_TYPE:=Release}"

# Build parallelism is a MEMORY limit, not a core-count one: the heaviest module TUs
# (OperationTraits + the CUDA op .ixx units instantiate the full
# device x precision x quantization dispatch table) each cost several GB to compile,
# so $(nproc) of them at once starves the Docker VM's RAM. Cap conservatively; a
# RAM-rich host can raise it (or bump the WSL2 VM memory in %USERPROFILE%\.wslconfig).
: "${MILA_BUILD_JOBS:=4}"

# Optional opt-ins (default off -- this is a convenience build, not a test/packaging gate):
: "${MILA_ENABLE_TESTING:=OFF}"     # ON to also build the GTest suite (runnable with ctest)
: "${MILA_ENABLE_PYTHON_BINDINGS:=ON}"  # the binding is part of the full product; OFF to skip pybind fetch

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
    -DMILA_LIBRARY_CUDA_ARCHITECTURES="${MILA_CUDA_ARCH}" \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DMILA_ENABLE_CUDA=ON \
    -DMILA_ENABLE_ADAPTORS=ON \
    -DMILA_ENABLE_SAMPLES=ON \
    -DMILA_ENABLE_PYTHON_BINDINGS="${MILA_ENABLE_PYTHON_BINDINGS}" \
    -DMILA_ENABLE_TESTING="${MILA_ENABLE_TESTING}" \
    -DMILA_ENABLE_PROFILING=OFF \
    -DMILA_ENABLE_DOCS=OFF \
    -DMILA_INSTALL=OFF

# No --target: build everything the configure enabled.
cmake --build "${BUILD}" -- -j "${MILA_BUILD_JOBS}"

echo "Built the full Mila product set in ${BUILD} (arch ${MILA_CUDA_ARCH}, ${MILA_BUILD_TYPE}, -j ${MILA_BUILD_JOBS})."
echo "Run Chat with: mila-chat"
if [ "${MILA_ENABLE_TESTING}" = "ON" ]; then
    echo "Run the test suite with: ctest --test-dir ${BUILD} --output-on-failure"
fi
