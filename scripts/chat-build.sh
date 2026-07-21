#!/usr/bin/env bash
# Configure + build the Chat app in the Mila container (writes to the mila-build volume).
cd "$(dirname "$0")/../Docker"
docker compose run --rm mila-dev mila-build-chat
