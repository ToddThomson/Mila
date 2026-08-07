#!/usr/bin/env bash
#
# Package the `mila-llm` Linux wheel for PyPI, inside the wheel container.
#
# Configure and build come from the linux-wheel CMake preset -- one source of truth,
# so selecting "Linux Wheel" in VS 2026 against the Wheel Container and running this
# script produce the same binary. This script re-runs configure+build (a no-op when
# VS already did it) and then does the part CMake does not: package and repair.
set -euo pipefail

: "${MILA_BUILD_JOBS:=4}"

SRC=/mila
BUILD=/build/linux-wheel
STAGE=/build/package
OUT="${SRC}/out/wheel"

cmake --preset linux-wheel
cmake --build "${BUILD}" -- -j "${MILA_BUILD_JOBS}"

# Package from a COPY, never from the bind-mounted tree. MilaPy's POST_BUILD stages the
# extension into Mila/Bindings/Package/src/mila on the host, where a Windows build has
# already left its _mila*.pyd -- and package-data globs both *.pyd and *.so, so
# packaging in place would put the Windows extension inside the Linux wheel. Copying
# also means this never deletes anything from the developer's tree.
rm -rf "${STAGE}"
cp -r "${SRC}/Mila/Bindings/Package" "${STAGE}"
find "${STAGE}/src/mila" -name '_mila*.pyd' -delete

# Clear THIS platform's wheels only, never the whole directory: the Windows wheel lands
# here too and both are published from one glob. A previous run's wheel carries a
# different version, and leaving one behind is how a stale build gets uploaded alongside
# the intended one -- which cannot be undone on PyPI.
mkdir -p "${OUT}"
rm -f "${OUT}"/mila_llm-*linux*.whl
pip wheel --no-deps --no-build-isolation -w "${OUT}" "${STAGE}"

# auditwheel sets the manylinux tag from the glibc actually linked. The CUDA libraries
# are EXCLUDED because they arrive from the nvidia-cublas / nvidia-curand wheels the
# package depends on; vendoring them instead would add ~400 MB and defeat that design.
auditwheel repair "${OUT}"/mila_llm-*-linux_x86_64.whl \
    --exclude libcublas.so.13 \
    --exclude libcublasLt.so.13 \
    --exclude libcurand.so.10 \
    --exclude libcudart.so.13 \
    -w "${OUT}"

rm -f "${OUT}"/mila_llm-*-linux_x86_64.whl

echo "Wheel written to ${OUT} (visible on the host under out/wheel):"
ls -1 "${OUT}"
