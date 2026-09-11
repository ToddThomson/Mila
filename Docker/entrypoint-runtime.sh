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

# The ENTRYPOINT of the `runtime` tag, installed as /usr/local/bin/mila-entrypoint. It is NOT
# the `mila` command: that name belongs to the Mila/Tools/Cli binary in both tags, which reaches
# the store directly and carries no chat verb. The verbs here are the IMAGE's interface, and the
# image is where they belong -- it knows where it put the binaries, so it needs none of the
# discovery a native front door would. MILA_APP_DIR lets the devel tag point at its build tree,
# and the venv is addressed separately because devel's lives beside that tree, not inside it.
APP_DIR="${MILA_APP_DIR:-/opt/mila}"
VENV_DIR="${MILA_VENV_DIR:-${APP_DIR}/venv}"

case "${1:-chat}" in
    chat)
        shift || true
        # No cd: Chat finds its config and prompts beside the binary, from any working
        # directory. That was a correctness requirement until it read /proc/self/exe.
        exec "${APP_DIR}/mila-chat" "$@"
        ;;
    serve)
        shift || true
        # MIS is installed, not run in place, so there is no server tree to enter. Staying
        # in APP_DIR is what makes a mounted /opt/mila/.env the one MIS reads; without one
        # it starts on its own defaults, which is the intended behaviour for the image.
        cd "${APP_DIR}"
        exec "${VENV_DIR}/bin/mila-server" "$@"
        ;;
    install)
        shift || true

        if [ "$#" -eq 0 ]; then
            echo "usage: install <model-name> [<model-name> ...]" >&2
            echo "Names are what the model store lists; see https://huggingface.co/mila-llm" >&2
            exit 2
        fi

        # Through the C++ CLI, not the binding. Both reach the same store, but only this one
        # reports download progress: it drives Distribution's (received, total) callback, while
        # the binding's pull() exposes no progress parameter -- so the Python route printed
        # "Installing <name> ..." and then nothing at all for several GB, which on the first
        # command of the published quick start reads as a hang. It also takes Python out of the
        # install path entirely, which is what this CLI was built for.
        exec "${APP_DIR}/mila" install "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
