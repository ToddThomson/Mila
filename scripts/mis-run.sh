#!/usr/bin/env bash
# Run the Mila Inference Server in the container, publishing its port to the host so a
# harness on the host (or WSL) can drive it. Override port/protocol via env, e.g.:
#   MILA_PORT=6452 MILA_PROTOCOL=anthropic scripts/mis-run.sh
cd "$(dirname "$0")/../Docker"
PORT="${MILA_PORT:-6452}"
docker compose run --rm \
    --publish "${PORT}:${PORT}" \
    -e MILA_PORT="${PORT}" \
    -e MILA_PROTOCOL="${MILA_PROTOCOL:-openai}" \
    mila-dev mila-mis
