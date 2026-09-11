#!/usr/bin/env bash
#
# Run the built Chat app inside the Mila container.
#
# The POST_BUILD step copies Prompts/ and chat.json next to the binary in /build, and
# mila-chat finds them itself: executable_directory() reads /proc/self/exe on Linux rather
# than falling through to the working directory. The cd below is therefore redundant and is
# kept only until a container run confirms it -- see Specifications/ChatConfiguration.md
# sections 2 and 8.
#
# Models come from the local Mila store at MILA_CACHE_DIR (/mila/Data/Models/Store, set
# in the image), which the bind-mounted repo supplies and MIS shares. Chat opens on an
# empty store: install the default with `/install gemma-4-12b-it-fp4` at the prompt.
# Any argument (e.g. --settings other.json, --system-prompt tools, --help) is forwarded on.
set -euo pipefail

BUILD=/build

if [ ! -x "${BUILD}/mila-chat" ]; then
    echo "mila-chat is not built. Build it first with: mila-build-chat" >&2
    exit 1
fi

cd "${BUILD}"
exec "${BUILD}/mila-chat" "$@"
