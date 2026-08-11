#!/bin/bash
# Build the mila-llm Linux wheel for PyPI. Writes to out/wheel.
cd "$(dirname "$0")/../Docker"
docker compose -f docker-compose.wheel.yml build
docker compose -f docker-compose.wheel.yml run --rm mila-wheel mila-build-wheel
