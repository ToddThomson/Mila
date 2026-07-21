#!/usr/bin/env bash
# Build the Mila Inference Server (mila binding + Python server venv) in the container.
cd "$(dirname "$0")/../Docker"
docker compose run --rm mila-dev mila-build-mis
