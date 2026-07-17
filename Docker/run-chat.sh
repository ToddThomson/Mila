#!/usr/bin/env bash
#
# Run the built Chat app inside the Mila container.
#
# ChatApp resolves its session config (Data/session.json) and system prompt against
# the current working directory on Linux -- executable_directory() there is just
# current_path(), since the GetModuleFileName path is Windows-only. The POST_BUILD
# step copies Data/ next to the binary in /build, so we cd there before exec.
#
# Model weights resolve through the compiled-in MODELS_DIR (/mila/Data/Models), which
# the bind-mounted repo supplies. The default model (gemma-12b) needs:
#   /mila/Data/Models/gemma/gemma4_12b_it_bf16.bin
#   /mila/Data/Models/gemma/gemma_tokenizer.bin
# Any argument (e.g. --config Data/other.json, --help) is forwarded to ChatApp.
set -euo pipefail

BUILD=/build

if [ ! -x "${BUILD}/ChatApp" ]; then
    echo "ChatApp is not built. Build it first with: mila-build-chat" >&2
    exit 1
fi

cd "${BUILD}"
exec "${BUILD}/ChatApp" "$@"
