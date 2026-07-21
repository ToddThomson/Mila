#!/usr/bin/env bash
# Configure + build the full Mila product set (library, samples, Chat, Python binding)
# in the Mila container (writes to the mila-build volume).
cd "$(dirname "$0")/../Docker"
docker compose run --rm mila-dev mila-build-all
