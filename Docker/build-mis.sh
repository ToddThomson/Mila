#!/usr/bin/env bash
#
# Configure + build the Mila Inference Server (MIS) inside the build container.
#
# MIS is the Python "wire adaptor": a FastAPI server that imports the `mila` CPython
# binding and exposes it over HTTP under an OpenAI / Anthropic / Mila-native protocol,
# so a foreign harness (Codex CLI, Claude Code CLI, ...) can drive Mila as its brain.
#
# It has two halves, like the MIS README describes -- but the container collapses the
# painful one:
#   Half 1  the `mila` binding: a version-locked CPython extension built by the MilaPy
#           CMake target. Its POST_BUILD step (Mila/Bindings/CMakeLists.txt) stages the
#           built .so into Mila/Bindings/Package, which the venv installs editable -- so
#           a later rebuild is picked up with no reinstall.
#   Half 2  the Python server deps (fastapi/uvicorn/pydantic), installed into a venv.
# On the host these must be reconciled by hand (an isolated 3.13 venv, PATH-shadow
# hazards, ModuleNotFoundError). In the container there is exactly ONE Python -- the
# binding is built against it and MIS runs under it -- so the version lock that
# dominates the MIS README is satisfied by construction.
#
# Build tree -> /build (container-local volume, off the bind mount) like build-chat.sh;
# the venv also lives on /build so both survive `run --rm`.
set -euo pipefail

: "${MILA_CUDA_ARCH:=native}"   # see build-chat.sh for the native/override rationale
: "${MILA_BUILD_TYPE:=Release}"
: "${MILA_BUILD_JOBS:=4}"

SRC=/mila
BUILD=/build
VENV=/build/mis-venv

# --- Half 1: build the mila binding (library + MilaPy target only) ---
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
    -DMILA_ENABLE_PYTHON_BINDINGS=ON \
    -DMILA_ENABLE_ADAPTORS=OFF \
    -DMILA_ENABLE_SAMPLES=OFF \
    -DMILA_ENABLE_TESTING=OFF \
    -DMILA_ENABLE_PROFILING=OFF \
    -DMILA_ENABLE_DOCS=OFF \
    -DMILA_INSTALL=OFF

cmake --build "${BUILD}" --target MilaPy -- -j "${MILA_BUILD_JOBS}"

# --- Half 2: the Python server dependencies, in a venv on the /build volume ---
# The venv is created from the container's python3 -- the SAME interpreter MilaPy was
# built against, so `import mila` matches by construction. Deps are installed EXPLICITLY
# (mirroring the runtime deps in Server/pyproject.toml) rather than `pip install -e .`,
# so whatever Python minor Ubuntu 26.04 ships does not trip the pyproject's
# Windows-oriented ">=3.13,<3.14" requires-python pin (that pin exists to match the
# committed cp313 Windows binding; in the container the binding is freshly built to fit).
if [ ! -d "${VENV}" ]; then
    python3 -m venv "${VENV}"
fi
"${VENV}/bin/pip" install --upgrade pip
"${VENV}/bin/pip" install fastapi "uvicorn[standard]" pydantic pydantic-settings

# The runtime, editable off the package tree MilaPy stages into -- NOT from PyPI. A
# published wheel is cp313/x86_64 and would not match this container's interpreter, and
# the whole point of building here is to serve the binding just built. --no-deps because
# the CUDA libraries come from the container's toolkit, not from NVIDIA's wheels.
"${VENV}/bin/pip" install --no-deps -e "${SRC}/Mila/Bindings/Package"

echo "MIS built: mila binding (arch ${MILA_CUDA_ARCH}) + server venv at ${VENV}."
echo "Run it with: mila-mis"
