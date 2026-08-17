#!/usr/bin/env bash
# Build and publish the Mila images to Docker Hub, from a pushed release tag.
#
#   scripts/dockerhub/publish-image.sh v0.20.0-beta.3           # build + verify gates, NO push
#   scripts/dockerhub/publish-image.sh v0.20.0-beta.3 --push    # ...then push, after confirming
#
# Building is safe and repeatable; pushing is neither. So the default is build-only and --push is
# a deliberate second decision -- the same shape as publish-site.yml, where running the workflow IS
# the decision to publish. A pushed tag cannot be withdrawn, only superseded.
#
# THE TAG IS THE SOURCE OF TRUTH FOR THE IMAGE TAG. It is never typed twice and never derived from
# the working tree: a published -devel image ships /src, so a user reads and edits that source, and
# if it came from anywhere but a public tag there is nothing to reproduce against.
#
# Credentials are never handled here. Run `docker login` yourself first; this script only checks
# that you did.
set -euo pipefail

readonly REPOSITORY="toddthomson/mila-llm"

# The published GPU compatibility list, and the authority for it. Every published model is FP4 or
# FP8, which need SM 8.9+, so Turing and Ampere kernels would advertise cards that cannot run
# anything in the store; 120 is Blackwell, which the library's own portable list omits and which is
# the generation people are buying. `native` -- the default in the local build scripts -- is wrong
# twice here: it does not resolve on a GPU-less builder, and the image is pulled by hardware the
# builder never saw.
readonly ARCHITECTURES="89;90;120"

# No `latest`. A bare `docker run toddthomson/mila-llm` resolves to it, so pointing it at a
# pre-release makes the beta the default for everyone who does not read the tag list. It starts
# existing at the first unsuffixed release and tracks the runtime variant of the newest one.
readonly TARGETS=("runtime" "devel")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

die() { echo "publish-image: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
[ "$#" -ge 1 ] || die "usage: $(basename "$0") <release-tag> [--push]"

tag="$1"
shift
push=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --push) push=1 ;;
        *) die "unknown option '$1'" ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Gates. Every one of these has to pass before anything is built, because a wrong image that
# builds successfully is the expensive failure -- two have already shipped silently from stale
# cache mounts alone.
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"

git rev-parse --verify "refs/tags/${tag}" >/dev/null 2>&1 \
    || die "no such tag '${tag}'. Tag the release first."

# On the REMOTE, not just locally: a local-only tag is not something a user can fetch, so an image
# built from it has nothing public behind it.
git ls-remote --exit-code --tags origin "refs/tags/${tag}" >/dev/null 2>&1 \
    || die "tag '${tag}' is not pushed to origin. A publishable image must come from a pushed tag."

tag_sha="$(git rev-parse "${tag}^{commit}")"
head_sha="$(git rev-parse HEAD)"
[ "${tag_sha}" = "${head_sha}" ] \
    || die "HEAD is not ${tag}.
    HEAD  ${head_sha}
    ${tag}  ${tag_sha}
  Check out the tag before publishing from it."

[ -z "$(git status --porcelain)" ] \
    || die "working tree is dirty. A published image must be reproducible from the tag alone."

# The image version is the tag with its leading v removed. Release tags carry no +build metadata
# (v0.20.0-beta.2 is the precedent), which is why no sanitising is needed here -- OCI forbids '+'
# and the tag never contains one. Asserted rather than assumed.
version="${tag#v}"
case "${version}" in
    *+*) die "tag '${tag}' carries build metadata; OCI image tags cannot contain '+'." ;;
esac

# Version.txt at this commit names the same release. It carries the +build counter the tag drops.
file_version="$(tr -d '[:space:]' < Version.txt)"
case "${file_version}" in
    "${version}"|"${version}"+*) ;;
    *) die "Version.txt says '${file_version}', which is not ${version}." ;;
esac

if [ "${push}" -eq 1 ] && ! grep -q 'index.docker.io' "${HOME}/.docker/config.json" 2>/dev/null; then
    die "no Docker Hub credentials found. Run 'docker login' first; this script never handles them."
fi

# ---------------------------------------------------------------------------
# Build. Both targets come from one Dockerfile and share the builder stage, so the compile happens
# once. MILA_CLEAN_BUILD is forced, never inherited: --no-cache invalidates layers but leaves
# BuildKit cache mounts intact, and an image assembled from another tree's objects is exactly the
# failure this whole gate list exists to prevent.
# ---------------------------------------------------------------------------
echo "Repository   ${REPOSITORY}"
echo "Tag          ${tag}  ->  ${version}"
echo "Revision     ${head_sha}"
echo "Targets      ${TARGETS[*]}"
echo "Architectures ${ARCHITECTURES}"
echo

built=()

for target in "${TARGETS[@]}"; do
    image="${REPOSITORY}:${version}-${target}"
    echo "== building ${image} ============================================="

    docker build \
        -f "${REPO_ROOT}/Docker/Dockerfile.runtime" \
        --target "${target}" \
        --build-arg MILA_IMAGE_CUDA_ARCHITECTURES="${ARCHITECTURES}" \
        --build-arg MILA_CLEAN_BUILD=1 \
        --label "org.opencontainers.image.title=Mila" \
        --label "org.opencontainers.image.description=A C++23 and CUDA reference implementation of LLM inference" \
        --label "org.opencontainers.image.version=${version}" \
        --label "org.opencontainers.image.revision=${head_sha}" \
        --label "org.opencontainers.image.source=https://github.com/ToddThomson/Mila" \
        --label "org.opencontainers.image.licenses=MIT" \
        --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        -t "${image}" \
        "${REPO_ROOT}"

    built+=("${image}")
done

echo
echo "== built ==========================================================="
for image in "${built[@]}"; do
    printf "  %-52s %s\n" "${image}" \
        "$(docker image inspect "${image}" --format '{{.Size}}' | awk '{printf "%.2f GB content", $1/1000000000}')"
done

# ---------------------------------------------------------------------------
# Push. Irreversible and outward-facing, so it asks -- and asks for the version rather than a
# y/n, because a reflexive "y" is not a decision.
# ---------------------------------------------------------------------------
if [ "${push}" -eq 0 ]; then
    echo
    echo "Not pushed (no --push). These would go to Docker Hub:"
    printf '  %s\n' "${built[@]}"
    exit 0
fi

echo
echo "About to PUSH the images above to Docker Hub. This cannot be undone -- a published tag can"
echo "only be superseded, never withdrawn."
printf "Type the version (%s) to confirm: " "${version}"
read -r confirmation

[ "${confirmation}" = "${version}" ] || die "not confirmed; nothing pushed."

for image in "${built[@]}"; do
    echo "== pushing ${image} =============================================="
    docker push "${image}"
done

echo
echo "Published:"
printf '  %s\n' "${built[@]}"
echo
echo "Next: pull each on a GPU host and run the website's quick start against it --"
echo "  MILA_IMAGE=${REPOSITORY}:${version}-runtime scripts/dockerhub/verify-image.sh"
echo "CI cannot do this: a hosted runner has no GPU, so nothing here has been executed."
