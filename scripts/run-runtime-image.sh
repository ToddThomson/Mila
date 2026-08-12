#!/usr/bin/env bash
# Run the locally built Mila image the way a user of the published one would.
#
# The store lives in a named docker volume rather than the repo, because that is what a
# user without a checkout has -- and it is what keeps a ~6 GB model install from being
# discarded by --rm. Chat opens on an empty store: install with `/install <name>`.
#
#   scripts/run-runtime-image.sh            # Chat (default)
#   scripts/run-runtime-image.sh serve      # MIS, port published to the host
set -euo pipefail

: "${MILA_RUNTIME_IMAGE_TAG:=mila-llm:local}"

publish=()
if [ "${1:-}" = "serve" ]; then
    publish=(--publish 6452:6452)
fi

exec docker run --rm -it --gpus all \
    -v mila-store:/models \
    "${publish[@]}" \
    "${MILA_RUNTIME_IMAGE_TAG}" "$@"
