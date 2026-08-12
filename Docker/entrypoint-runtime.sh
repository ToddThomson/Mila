#!/usr/bin/env bash
#
# Entry point for the PUBLISHED Mila image (Docker/Dockerfile.runtime).
#
# The image ships two interfaces onto one library, so the first argument selects which
# one runs and everything after it is forwarded untouched:
#   chat            the Chat harness -- a human at a prompt. The default.
#   serve           the Mila Inference Server -- a foreign harness over HTTP.
#   <anything else> exec'd as given, so `docker run ... bash` still works.
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
        # MIS reads its .env from the working directory, so the server tree is the CWD.
        cd "${APP_DIR}/server"
        exec "${APP_DIR}/venv/bin/python" main.py "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
