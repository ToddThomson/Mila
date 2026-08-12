#!/usr/bin/env bash
# Build the PUBLISHED Mila image (Docker/Dockerfile.runtime), tagged for local testing.
# The build context is the repo root -- .dockerignore keeps Data/Models, out/ and .git out.
# Override the GPU compatibility list with MILA_IMAGE_CUDA_ARCHITECTURES=89 for a fast
# single-arch test build. Publishing to a registry is a separate step.
set -euo pipefail

# MILA_CLEAN_BUILD=1 discards the cached build tree first. A publish build must set it:
# `--no-cache` invalidates layers but leaves BuildKit cache mounts intact, so without it an image
# can be assembled from objects left by a different source tree.
: "${MILA_RUNTIME_IMAGE_TAG:=mila-llm:local}"
: "${MILA_IMAGE_CUDA_ARCHITECTURES:=89;90;120}"
: "${MILA_CLEAN_BUILD:=0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker build \
    -f "${REPO_ROOT}/Docker/Dockerfile.runtime" \
    --build-arg MILA_IMAGE_CUDA_ARCHITECTURES="${MILA_IMAGE_CUDA_ARCHITECTURES}" \
    --build-arg MILA_CLEAN_BUILD="${MILA_CLEAN_BUILD}" \
    -t "${MILA_RUNTIME_IMAGE_TAG}" \
    "${REPO_ROOT}"
