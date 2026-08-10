#!/usr/bin/env bash
#
# Package the `mila-llm` Linux wheels for PyPI, inside the wheel container.
#
# Configure and build come from the linux-wheel CMake preset -- one source of truth,
# so selecting "Linux Wheel" in VS 2026 against the Wheel Container and running this
# script produce the same binary. This script re-runs configure+build (a no-op when
# VS already did it) and then does the part CMake does not: package and repair.
#
# One wheel per interpreter: pybind11 cannot use Py_LIMITED_API, so there is no abi3
# build covering a range. INTERPRETERS must match requires-python in pyproject.toml --
# metadata promising an interpreter with no wheel behind it turns a clear "requires a
# different Python" into "no matching distribution".
set -euo pipefail

: "${MILA_BUILD_JOBS:=4}"
: "${INTERPRETERS:=3.12 3.13}"

SRC=/mila
BUILD=/build/linux-wheel
STAGE=/build/package
OUT="${SRC}/out/wheel"

# Clear THIS platform's wheels ONCE, before the loop, and never the whole directory: the
# Windows wheel lands here too and both are published from one glob. A previous run's
# wheel carries a different version, and leaving one behind is how a stale build gets
# uploaded alongside the intended one -- which cannot be undone on PyPI. Inside the loop
# this would delete the wheel the previous interpreter just built.
mkdir -p "${OUT}"
rm -f "${OUT}"/mila_llm-*linux*.whl

for version in ${INTERPRETERS}; do
    python="/usr/bin/python${version}"

    if [ ! -x "${python}" ]; then
        echo "Python ${version} not found at ${python}. Add it to Dockerfile.wheel, or drop" >&2
        echo "it from INTERPRETERS and from requires-python." >&2
        exit 1
    fi

    echo
    echo "=== Python ${version} -- ${python} ==="

    # PYBIND11_FINDPYTHON=NEW and Python_EXECUTABLE are what actually select the
    # interpreter. pybind11 v3 resolves through FindPython, so Python3_EXECUTABLE alone
    # steers only the rest of the project: a tree configured with Python3_EXECUTABLE=3.12
    # was measured building a cp313 extension. That mismatch is silent and ships -- the
    # wheel tag comes from the packaging venv below, so the archive would be tagged for
    # one interpreter and contain an extension for another, and `import mila` then fails
    # on a machine that has only the tagged one. All three are passed so nothing resolves
    # independently of the others.
    # --fresh because FindPython caches the include and library paths it resolved, and
    # those do not move when the executable does: reconfiguring a tree built for one
    # interpreter against another fails with "Could NOT find Python (missing:
    # Development.Module)" while reporting the NEW version as found. One build tree, wiped
    # per interpreter, rather than a tree each -- a wheel build is a release-time operation,
    # so predictability is worth more than incrementality here.
    cmake --preset linux-wheel --fresh \
        -DPYBIND11_FINDPYTHON=NEW \
        -DPython_EXECUTABLE="${python}" \
        -DPython3_EXECUTABLE="${python}"
    cmake --build "${BUILD}" -- -j "${MILA_BUILD_JOBS}"

    # Package from a COPY, never from the bind-mounted tree. MilaPy's POST_BUILD stages
    # into Mila/Bindings/Package/src/mila on the host, which accumulates: a Windows build
    # leaves its _mila*.pyd and every interpreter leaves its own .so. package-data globs
    # both, so packaging in place would put every extension ever built into every wheel.
    # Copying also means this never deletes anything from the developer's tree.
    rm -rf "${STAGE}"
    cp -r "${SRC}/Mila/Bindings/Package" "${STAGE}"

    # Take the extension from THIS build rather than trusting what POST_BUILD left behind:
    # strip every staged extension, then copy in the one just produced. Deterministic
    # regardless of what earlier runs deposited in the tree.
    find "${STAGE}/src/mila" -name '_mila*' -delete

    # Named by the interpreter's OWN suffix, never "the .so in the build directory":
    # --fresh clears the CMake cache but not build outputs, so a previous interpreter's
    # extension is still sitting there. Matching the tag exactly is what stops a stale
    # artifact being packaged as this one.
    suffix=$("${python}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
    built="${BUILD}/Mila/Bindings/_mila${suffix}"

    if [ ! -f "${built}" ]; then
        echo "expected ${built} from the Python ${version} build, and it is not there" >&2
        exit 1
    fi

    cp "${built}" "${STAGE}/src/mila/"

    # One packaging venv PER interpreter, because pip wheel takes the wheel's cp tag from
    # the interpreter running it. 24.04 marks the system interpreter externally-managed
    # (PEP 668), so a plain pip install into it fails by design. setuptools is named
    # explicitly: since 3.12 ensurepip no longer seeds it into a venv, and pyproject.toml
    # declares it as the build backend.
    venv="/build/wheel-venv-${version}"

    if [ ! -x "${venv}/bin/python" ]; then
        "${python}" -m venv "${venv}"
        "${venv}/bin/pip" install --no-cache-dir --upgrade pip "setuptools>=77" build wheel
    fi

    "${venv}/bin/pip" wheel --no-deps --no-build-isolation -w "${OUT}" "${STAGE}"

    # auditwheel sets the manylinux tag from the glibc actually linked. The CUDA libraries
    # are EXCLUDED because they arrive from the nvidia-cublas / nvidia-curand wheels the
    # package depends on; vendoring them instead would add ~400 MB and defeat that design.
    # Run per interpreter so the unrepaired linux_x86_64 wheel never outlives its loop
    # iteration and get repaired twice.
    auditwheel repair "${OUT}"/mila_llm-*-linux_x86_64.whl \
        --exclude libcublas.so.13 \
        --exclude libcublasLt.so.13 \
        --exclude libcurand.so.10 \
        --exclude libcudart.so.13 \
        -w "${OUT}"

    rm -f "${OUT}"/mila_llm-*-linux_x86_64.whl
    rm -rf "${STAGE}"
done

echo
echo "Wheels written to ${OUT} (visible on the host under out/wheel):"
ls -1 "${OUT}"
