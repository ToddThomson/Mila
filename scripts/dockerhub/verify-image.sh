#!/usr/bin/env bash
# Run the website's "Try it" sequence against a LOCAL image, before anything is published.
#
# The commands below are the ones the #evaluate band in Web/layouts/index.html tells a reader to
# type, reproduced with exactly two substitutions: the image reference (a local, namespace-free
# tag that cannot be pushed) and the store volume (a throwaway -- see below). When the site copy
# changes this script has to change with it; that coupling IS the point. A published image whose
# quick start does not work is worse than no published image.
#
#   scripts/dockerhub/verify-image.sh
#   MILA_IMAGE=mila-llm:local scripts/dockerhub/verify-image.sh
#   MILA_KEEP_VOLUME=1 scripts/dockerhub/verify-image.sh   # leave the store behind to poke at
set -euo pipefail

# Git Bash rewrites POSIX-looking arguments into Windows paths before handing them to a native
# binary, which turns `-v vol:/models` into `-v vol:C:/Program Files/Git/models`. Nothing here
# passes a HOST path to docker -- only a volume name and container-side paths -- so disabling the
# rewrite is safe. Do NOT copy this into a script that passes a build context: there the rewrite
# is the thing making `/d/Repos/Mila` reach docker as `D:\Repos\Mila`. No-op off Windows.
export MSYS_NO_PATHCONV=1

: "${MILA_IMAGE:=mila-llm:0.20.0-beta.3-runtime}"
: "${MILA_VERIFY_MODEL:=Llama-3.2-3B-Instruct-fp4}"
: "${MILA_VERIFY_PROMPT:=Why is the sky blue?}"
: "${MILA_KEEP_VOLUME:=0}"

# A FRESH volume per run, NOT the site's named `mila-store`. Reusing a populated store would skip
# the download, so `install` would report success without doing anything -- which is the exact
# failure this script exists to catch. The cost is re-downloading the model on every run.
volume="mila-verify-$$"

cleanup() {
    if [ "${MILA_KEEP_VOLUME}" = "1" ]; then
        echo
        echo "Store volume kept: ${volume}"
    else
        docker volume rm "${volume}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

fail() { echo; echo "FAILED: $*" >&2; exit 1; }

if ! docker image inspect "${MILA_IMAGE}" >/dev/null 2>&1; then
    fail "no local image ${MILA_IMAGE}. Build one first:
    MILA_RUNTIME_IMAGE_TAG=${MILA_IMAGE} MILA_CLEAN_BUILD=1 scripts/dockerhub/build-runtime-image.sh"
fi

echo "Image  : ${MILA_IMAGE}"
echo "Model  : ${MILA_VERIFY_MODEL}"
echo "Volume : ${volume} (fresh, empty store)"
echo

# ---------------------------------------------------------------------------
# Step 1 -- the site's first command.
# ---------------------------------------------------------------------------
echo "== 1. install =============================================================="
docker run --rm --gpus all -v "${volume}:/models" \
    "${MILA_IMAGE}" install "${MILA_VERIFY_MODEL}" \
    || fail "install exited non-zero"

# ---------------------------------------------------------------------------
# Step 2 -- the site's second command, as close as an unattended run can get.
#
# It NAMES the model, because that is the correct command and not a concession: Chat switches
# models at runtime, so a `chat` that inferred one from a store holding exactly one would change
# meaning the moment a second was installed. Only the interactivity is replaced here (-p for one
# shot, JSON so the outcome is assertable rather than eyeballed).
# ---------------------------------------------------------------------------
echo
echo "== 2. chat --model ${MILA_VERIFY_MODEL} (what the site tells a reader to type) ====="
payload="$( docker run --rm --gpus all -v "${volume}:/models" \
    "${MILA_IMAGE}" chat --model "${MILA_VERIFY_MODEL}" \
    -p "${MILA_VERIFY_PROMPT}" --output-format json 2>&1 )" \
    || fail "the image did not generate:
$( echo "${payload}" | sed 's/^/    /' )"
echo "ok -- generated an answer"

# ---------------------------------------------------------------------------
# Step 3 -- bare `chat` must fail WELL. Nobody is told to run this, but the image's default CMD
# is `chat`, so `docker run <image>` with no arguments lands here. The requirement is a named
# instruction, not a crash and not a guess at which model was meant.
# ---------------------------------------------------------------------------
echo
echo "== 3. bare chat, no model named -- must refuse clearly ====================="
unnamed_ok=1

if unnamed="$( docker run --rm --gpus all -v "${volume}:/models" \
        "${MILA_IMAGE}" chat -p "${MILA_VERIFY_PROMPT}" --output-format json 2>&1 )"; then
    unnamed_ok=0
    echo "  UNEXPECTED -- it produced an answer without being told which model to load."
    echo "  A store with one model today has two tomorrow; this command would change meaning."
elif echo "${unnamed}" | grep -q -- "--model"; then
    echo "  ok -- refused, and named the flag that fixes it"
else
    unnamed_ok=0
    echo "  refused, but without pointing at --model:"
    echo "${unnamed}" | sed 's/^/      /'
fi

# ---------------------------------------------------------------------------
# Assertions. The payload is nlohmann's dump(2), so the scalar fields are one per line and
# grep is enough; no jq or host python is assumed. Chat.ixx emitOneShotJson owns this shape.
# ---------------------------------------------------------------------------
echo
echo "== assertions =============================================================="

check() {
    if echo "${payload}" | grep -q "$2"; then
        echo "  ok   $1"
    else
        echo "  FAIL $1"
        echo "${payload}" | sed 's/^/       /'
        exit 1
    fi
}

check "response is JSON with content"     '"content"'
check "model is ${MILA_VERIFY_MODEL}"     "\"model\": \"${MILA_VERIFY_MODEL}\""
check "stopped on a stop token"           '"finish_reason": "stop"'

if echo "${payload}" | grep -q '"tokens_generated": 0'; then
    fail "the model generated zero tokens"
fi
echo "  ok   generated a non-empty answer"

# ---------------------------------------------------------------------------
# The answer itself, because a coherence problem is not something an assertion catches.
# ---------------------------------------------------------------------------
echo
echo "== the answer ============================================================="
echo "${payload}" | sed -n '/"content"/,/",$/p' | sed 's/^/  /'
echo
echo "== summary ================================================================"
echo "${payload}" | grep -E '"(model|context_length|context_source|tokens_generated|rounds|finish_reason)"' | sed 's/^/  /'

echo
if [ "${unnamed_ok}" -eq 1 ]; then
    echo "PASS -- the image works, and the site's two commands are what makes it work."
else
    echo "PASS WITH A DEFECT -- the site's two commands work, but bare 'chat' does not refuse"
    echo "cleanly. That path is reachable as the image's default CMD, so it needs to say what"
    echo "is missing rather than crash or guess."
    exit 2
fi
