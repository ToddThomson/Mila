#!/usr/bin/env bash
#
# Run the Mila Inference Server (MIS) inside the container.
#
# Serves the mila binding over HTTP so a foreign harness (Codex CLI, Claude Code CLI,
# ...) can drive Mila as its model brain. Build it first with: mila-build-mis
#
# Defaults are container-friendly and all overridable via MILA_* env (see the MIS README
# for the full table). The model is a NAME in the local Mila store, which lives on the
# bind mount (MILA_CACHE_DIR) so it survives `run --rm` and is the same store the host
# sees. MIS never downloads: install the model first -- from the host, or in the container
# with the chat harness. The tuned params in Server/.env (MILA_CONTEXT_LENGTH, generation
# defaults) still apply; env vars set here win over it in pydantic-settings.
#
# Port: 6452 -- a distinctive, collision-unlikely default ("MILA" on a phone keypad),
# not the crowded generic-HTTP 8000. Host is 0.0.0.0 so the published port is reachable
# from outside the container.
set -euo pipefail

VENV=/build/mis-venv
SERVER=/mila/Mila/Adaptors/Inference/Server

if [ ! -x "${VENV}/bin/python" ]; then
    echo "MIS is not built. Build it first with: mila-build-mis" >&2
    exit 1
fi

# The venv installs the mila package in editable mode off Mila/Bindings/Package, whose
# extension MilaPy's POST_BUILD stages. An unbuilt binding shows up as an import failure
# rather than a missing file, so ask the interpreter rather than the filesystem.
if ! "${VENV}/bin/python" -c "import mila" >/dev/null 2>&1; then
    echo "The mila binding does not import. (Re)build with: mila-build-mis" >&2
    exit 1
fi

export MILA_HOST="${MILA_HOST:-0.0.0.0}"
export MILA_PORT="${MILA_PORT:-6452}"
export MILA_PROTOCOL="${MILA_PROTOCOL:-openai}"
export MILA_MODEL="${MILA_MODEL:-gemma-4-12b-it-fp4}"

if ! "${VENV}/bin/python" -c "
import mila, sys
store = mila.ModelStore()
if store.locate('${MILA_MODEL}') is None:
    print('Not installed in ' + store.root + ': ${MILA_MODEL}', file=sys.stderr)
    print('Installed: ' + (', '.join(m.name for m in store.list()) or 'nothing'), file=sys.stderr)
    sys.exit(1)
"; then
    echo "MIS loads only what is already installed; it never downloads." >&2
    echo "Install it with the chat harness (/install ${MILA_MODEL}) or set MILA_MODEL." >&2
    exit 1
fi

echo "Starting MIS: protocol=${MILA_PROTOCOL} model=${MILA_MODEL} on ${MILA_HOST}:${MILA_PORT}"

# config.py loads Server/.env relative to the CWD, so run from the server directory.
cd "${SERVER}"
exec "${VENV}/bin/python" main.py
