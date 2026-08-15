#!/usr/bin/env bash
# Run the built Chat app in the Mila container (needs GPU + weights under Data/Models).
# Any arguments are forwarded to mila-chat (e.g. --settings Data/other.json, --help).
cd "$(dirname "$0")/../Docker"
docker compose run --rm mila-dev mila-chat "$@"
