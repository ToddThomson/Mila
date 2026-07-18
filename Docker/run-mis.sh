#!/usr/bin/env bash
#
# Run the Mila Inference Server (MIS) inside the container.
#
# Serves the mila binding over HTTP so a foreign harness (Codex CLI, Claude Code CLI,
# ...) can drive Mila as its model brain. Build it first with: mila-build-mis
#
# Defaults are container-friendly and all overridable via MILA_* env (see the MIS README
# for the full table). Model + tokenizer default to the Gemma 4 12B artifacts under the
# bind-mounted /mila/Data/Models, and are EXPORTED here so they win over the committed
# Server/.env (whose Windows paths are invalid in the container -- env vars take
# precedence over .env in pydantic-settings). The tuned params in that .env
# (MILA_CONTEXT_LENGTH, generation defaults) still apply.
#
# Port: 6452 -- a distinctive, collision-unlikely default ("MILA" on a phone keypad),
# not the crowded generic-HTTP 8000. Host is 0.0.0.0 so the published port is reachable
# from outside the container.
set -euo pipefail

VENV=/build/mis-venv
SERVER=/mila/Adaptors/Inference/Server

if [ ! -x "${VENV}/bin/python" ]; then
    echo "MIS is not built. Build it first with: mila-build-mis" >&2
    exit 1
fi

# MilaPy's POST_BUILD copies the built extension next to main.py; on Linux that is a
# mila.cpython-*-linux-gnu.so. Its absence means the binding was not (re)built.
if ! ls "${SERVER}"/mila*.so >/dev/null 2>&1; then
    echo "The mila binding (.so) is not next to main.py. (Re)build with: mila-build-mis" >&2
    exit 1
fi

export MILA_HOST="${MILA_HOST:-0.0.0.0}"
export MILA_PORT="${MILA_PORT:-6452}"
export MILA_PROTOCOL="${MILA_PROTOCOL:-openai}"
export MILA_MODEL_FAMILY="${MILA_MODEL_FAMILY:-gemma}"
export MILA_MODEL_PATH="${MILA_MODEL_PATH:-/mila/Data/Models/gemma/gemma4_12b_it_bf16.bin}"
export MILA_TOKENIZER_PATH="${MILA_TOKENIZER_PATH:-/mila/Data/Models/gemma/gemma_tokenizer.bin}"

if [ ! -f "${MILA_MODEL_PATH}" ]; then
    echo "Model file not found: ${MILA_MODEL_PATH}" >&2
    echo "Put the weights under Data/Models on the host, or set MILA_MODEL_PATH." >&2
    exit 1
fi

echo "Starting MIS: protocol=${MILA_PROTOCOL} family=${MILA_MODEL_FAMILY} on ${MILA_HOST}:${MILA_PORT}"

# config.py loads Server/.env relative to the CWD, so run from the server directory.
cd "${SERVER}"
exec "${VENV}/bin/python" main.py
