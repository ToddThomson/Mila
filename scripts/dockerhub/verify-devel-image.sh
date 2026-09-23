#!/usr/bin/env bash
# Run the website's Docker tab (#p-docker) against a LOCAL -devel image, before anything is published.
#
# verify-image.sh covers -runtime and never touches this image, which is how a -devel whose `mila`
# symlink pointed at nothing reached a published tag: it built, every automated check passed, and
# the tab's step 2 was a command-not-found for every reader. This walks the tab's three steps:
#
#   1. land in a shell in a built tree
#   2. mila install gemma-4-12b-it-fp4
#   3. cd ~/myapp && cmake --build build && ./build/myapp "Why is the sky blue?"
#
# When the #p-docker copy in Web/layouts/index.html changes, this script changes with it.
#
# THE STORE IS KEPT BETWEEN RUNS, unlike verify-image.sh's. The tab installs Gemma 4 12B, 6.8 GB,
# and a fresh download per run would dominate a release. A fresh-volume download through the same
# `mila install` code is already proven by verify-image.sh; what only this image can get wrong is
# that the command exists and runs, and a store that already holds the model still exercises that.
#
#   MILA_IMAGE=mila-release:<commit>-devel scripts/dockerhub/verify-devel-image.sh
set -euo pipefail

# See verify-image.sh: nothing here passes a host path to docker, so disabling Git Bash's path
# rewriting is safe. No-op off Windows.
export MSYS_NO_PATHCONV=1

: "${MILA_IMAGE:=mila-llm:0.20.0-devel}"
: "${MILA_VERIFY_STORE:=mila-verify-devel-store}"

# Hard-coded in the sample main.cpp that ~/myapp is built from; the tab installs the same name.
readonly MODEL="gemma-4-12b-it-fp4"
readonly PROMPT="Why is the sky blue?"

fail() { echo; echo "FAILED: $*" >&2; exit 1; }

docker image inspect "${MILA_IMAGE}" >/dev/null 2>&1 || fail "no local image ${MILA_IMAGE}."

echo "Image  : ${MILA_IMAGE}"
echo "Model  : ${MODEL}"
echo "Store  : ${MILA_VERIFY_STORE} (kept between runs)"
echo

# ---------------------------------------------------------------------------
# Step 1 -- the shell the tab promises, in a built tree, with `mila` on PATH resolving to a real
# executable. `ln -sf` succeeds against a missing target, so a dangling symlink passes `command -v`;
# resolving it is the part that has failed before.
# ---------------------------------------------------------------------------
echo "== 1. a shell in a built tree ============================================"
docker run --rm "${MILA_IMAGE}" bash -lc '
    set -e
    test "$(pwd)" = /src || { echo "landed in $(pwd), not /src"; exit 1; }
    command -v mila >/dev/null || { echo "mila is not on PATH"; exit 1; }
    target="$(readlink -f "$(command -v mila)")"
    test -x "${target}" || { echo "mila resolves to ${target}, which is not an executable"; exit 1; }
    test -x /opt/mila/build/mila-chat || { echo "no built mila-chat in /opt/mila/build"; exit 1; }
    test -f /root/myapp/build/build.ninja || { echo "~/myapp is not configured"; exit 1; }
    echo "ok -- /src, mila -> ${target}, built tree and configured ~/myapp present"
' || fail "the image is not the environment the tab describes"
echo

# ---------------------------------------------------------------------------
# Steps 2 and 3 in ONE container, because the tab runs them in one: myapp's build output lives in
# the container's filesystem and would not survive into a second `docker run`.
# ---------------------------------------------------------------------------
echo "== 2. mila install ${MODEL} ============================================="
echo "== 3. build and run ~/myapp (the first build compiles Mila; expect ~10 min) =="
output="$( docker run --rm --gpus all -v "${MILA_VERIFY_STORE}:/models" "${MILA_IMAGE}" bash -lc "
    set -e
    mila install ${MODEL}
    cd ~/myapp
    cmake --build build
    ./build/myapp '${PROMPT}'
" 2>&1 )" || fail "the tab's steps did not complete:
$( echo "${output}" | tail -40 | sed 's/^/    /' )"

echo "${output}" | tail -25 | sed 's/^/    /'
echo

echo "== assertions ============================================================"
echo "${output}" | grep -q '^Mila ' || fail "myapp did not print its Mila version line"
echo "  ok   myapp linked Mila and started"
echo "${output}" | grep -q '^\[stop\]$' || fail "myapp did not finish on a stop token"
echo "  ok   generated an answer and stopped on a stop token"

echo
echo "PASS -- the Docker tab's three steps work against this image."
