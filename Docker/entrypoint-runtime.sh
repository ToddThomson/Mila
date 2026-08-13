#!/usr/bin/env bash
#
# Entry point for the PUBLISHED Mila image (Docker/Dockerfile.runtime).
#
# The image ships two interfaces onto one library, so the first argument selects which
# one runs and everything after it is forwarded untouched:
#   chat            the Chat harness -- a human at a prompt. The default.
#   serve           the Mila Inference Server -- a foreign harness over HTTP.
#   install <name>  install a published model into the store, without a REPL.
#   <anything else> exec'd as given, so `docker run ... bash` still works.
#
# `install` exists because chat and serve both REFUSE to download: loading never pulls, so
# without a non-interactive verb the only way to get a model into a fresh volume is to run
# the image in chat mode and type at a prompt -- which is not something a quick start, a
# Dockerfile or a CI job can do.
#
# This is deliberately not the Docker/run-*.sh pair: those assume a bind-mounted checkout
# and a build tree on a volume, neither of which exists here.
set -euo pipefail

APP_DIR=/opt/mila

case "${1:-chat}" in
    chat)
        shift || true
        # Chat resolves Data/session.json against the working directory, which the image
        # sets to APP_DIR. Staying here is a correctness requirement, not tidiness.
        cd "${APP_DIR}"
        exec "${APP_DIR}/ChatApp" "$@"
        ;;
    serve)
        shift || true
        # MIS is installed, not run in place, so there is no server tree to enter. Staying
        # in APP_DIR is what makes a mounted /opt/mila/.env the one MIS reads; without one
        # it starts on its own defaults, which is the intended behaviour for the image.
        cd "${APP_DIR}"
        exec "${APP_DIR}/venv/bin/mila-server" "$@"
        ;;
    install)
        shift || true

        if [ "$#" -eq 0 ]; then
            echo "usage: install <model-name> [<model-name> ...]" >&2
            echo "Names are what the model store lists; see https://huggingface.co/mila-llm" >&2
            exit 2
        fi

        cd "${APP_DIR}"

        # Through the binding rather than a CLI: the store tooling (ExportArtifact) is a
        # build artifact this image does not ship, while the binding is already installed
        # for MIS. ModelStore.pull is the same call the Python quick start makes.
        exec "${APP_DIR}/venv/bin/python" - "$@" <<'PYTHON'
import sys

import mila

mila.initialize("warning")

store = mila.ModelStore()
owner = mila.default_hub_owner()

for name in sys.argv[1:]:
    print(f"Installing {name} from {owner} ...", flush=True)
    store.pull(name, owner)
    print(f"Installed {name}.", flush=True)
PYTHON
        ;;
    *)
        exec "$@"
        ;;
esac
