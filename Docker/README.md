# Mila in Docker — build the library, run the Chat app

A single CUDA `-devel` container that **builds** the Mila C++23 module library and
**runs** the Gemma 4 Chat app on the GPU. The `-devel` image already carries the CUDA
runtime libraries Chat links (cudart / cuBLAS / cuBLASLt / cuRAND / nvtx), so one image
covers both build and run for the interactive workflow. A slim multi-stage *runtime*
image for headless deployment is a separate, later deliverable (see `ROADMAP.md`).

## Toolchain (pinned)

Mirrors the validated CI / WSL matrix, so the container and CI stay comparable:

| | |
|---|---|
| Base image | `nvidia/cuda:13.3.0-devel-ubuntu26.04` |
| C++23 modules | clang-21 |
| nvcc host compiler | gcc-15 (module-free `.cu` files) |
| Build system | CMake 4.2.3 + Ninja + ccache |

CUDA 13.3 (not 13.0) is required on Ubuntu 26.04 / glibc 2.43. No cuDNN is installed —
`USE_CUDNN` is unset and no cuDNN image exists for 26.04.

## Prerequisites (host)

- Docker with the **NVIDIA Container Toolkit** (GPU passthrough). On Windows this is
  Docker Desktop on the WSL 2 backend with an NVIDIA driver.
- An NVIDIA GPU. The build defaults to **`sm_89` (Ada / RTX 4070)** — override
  `MILA_CUDA_ARCH` for another card (e.g. `120` for Blackwell / RTX 5060 Ti).
- **Nothing else.** Models are pulled into the local store at first use (`/install
  gemma-4-12b-it-fp4` at the Chat prompt), which the image points at
  `Data/Models/Store` on the bind mount — so the download survives `run --rm` and the
  host shares it. Weights are **never baked into the image**.

## Quick start

From the repo root:

```bash
# 1. Build the image (one time, or after changing the Dockerfile)
docker compose -f Docker/docker-compose.yml build

# 2. Build the Chat app (writes to the mila-build volume; incremental on re-runs)
docker compose -f Docker/docker-compose.yml run --rm mila-dev mila-build-chat

# 3. Run Chat (interactive; loads Gemma 4 12B FP4 onto the GPU)
docker compose -f Docker/docker-compose.yml run --rm mila-dev mila-chat
```

Want more than Chat? `mila-build-all` builds the full product set (library, samples,
Chat, Python binding) in the same known environment:

```bash
docker compose -f Docker/docker-compose.yml run --rm mila-dev mila-build-all
```

Convenience wrappers under `scripts/` do the same (`.sh` and `.ps1`):

```
scripts/build-docker.{sh,ps1}   # docker compose build
scripts/chat-build.{sh,ps1}     # mila-build-chat
scripts/all-build.{sh,ps1}      # mila-build-all
scripts/chat-run.{sh,ps1}       # mila-chat   (args forwarded, e.g. --help)
scripts/mis-build.{sh,ps1}      # mila-build-mis
scripts/mis-run.{sh,ps1}        # mila-mis     (publishes the port to the host)
```

## How it fits together

- **Source** is bind-mounted at `/mila`. The Chat build compiles `MODELS_DIR` in as the
  absolute path `/mila/Data/Models`, so weights on the host resolve with no extra config.
- **Build artifacts** go to `/build`, a container-local named volume (`mila-build`) kept
  *off* the bind mount — the C++23 module BMI I/O is metadata-heavy and slow across the
  host↔container filesystem boundary. `ccache` persists in the `mila-ccache` volume. Both
  survive `run --rm`, so rebuilds are incremental.
- **`mila-build-chat`** configures + builds only the `ChatApp` target (no tests, samples,
  profiling, docs, or Python binding) — the fast path when you just want to run Chat.
  **`mila-chat`** `cd`s into `/build` (where the POST_BUILD step copies `Data/`) and runs
  `ChatApp`; arguments are forwarded.
- **`mila-build-all`** configures + builds the full user-facing product set — library,
  samples, Chat, and the Python binding — for someone who wants more than Chat from the
  known environment. It builds for `MILA_CUDA_ARCH` (default `native` — CMake detects the
  arch of the GPU(s) present, so a multi-card host builds for all of them) instead of the
  library's portable five, which is the main build-time saving; set `MILA_CUDA_ARCH=89` to
  target a specific arch. Tests are off by default (this is a convenience build, not a
  portability/test gate — those are owned by CI and the WSL build; see the repo-root
  `RELEASING.md`); set `MILA_ENABLE_TESTING=ON` to also build the GTest suite.
- **`mila-build-mis`** builds the `mila` Python binding (the `MilaPy` target — no Chat,
  samples, or tests) and a Python venv at `/build/mis-venv` with the MIS server deps.
  **`mila-mis`** runs the server. See the next section.

## Driving Mila from a harness (MIS)

The **Mila Inference Server** is the HTTP *wire adaptor*: it serves the `mila` binding under
an OpenAI / Anthropic / Mila-native protocol, so a foreign harness (Codex CLI, Claude Code
CLI, …) can use Mila as its model brain. The container is the easy path — on the host, MIS
means reconciling a version-locked binding against an isolated venv; in the container
there is one Python, so the binding and server always match.

```bash
# 1. Build the binding + server venv (once, or after changing the binding)
docker compose -f Docker/docker-compose.yml run --rm mila-dev mila-build-mis

# 2. Run the server, publishing its port to the host (default 6452)
docker compose -f Docker/docker-compose.yml run --rm --publish 6452:6452 \
    -e MILA_PORT=6452 mila-dev mila-mis
```

The host wrappers `scripts/mis-build.{sh,ps1}` and `scripts/mis-run.{sh,ps1}` do the same
(the run wrapper handles `--publish` for you). Then point a harness at
`http://localhost:6452` — e.g. an OpenAI-compatible client at `http://localhost:6452/v1`,
or Claude Code at the Anthropic `/v1/messages` path (launch with `MILA_PROTOCOL=anthropic`).

- **Port:** `6452` by default (`MILA_PORT`) — distinctive and collision-unlikely ("MILA" on
  a phone keypad), deliberately not the crowded generic-HTTP `8000`.
- **Protocol:** `MILA_PROTOCOL` selects one adapter per launch (`openai` default, or
  `anthropic` / `mila`). Only that protocol's routes are registered.
- **Model:** a NAME in the local Mila store (`MILA_MODEL`, default `gemma-4-12b-it-fp4`),
  not a path. The store is `MILA_CACHE_DIR=/mila/Data/Models/Store`, set in the image so
  Chat and MIS share one — and it is on the bind mount, so it survives `run --rm` and the
  host sees the same models. **MIS never downloads**: install the model first, or it
  refuses to start and lists what is installed. `mila-mis` exports these so they win over
  the committed `Server/.env`; the tuned `MILA_CONTEXT_LENGTH` and generation defaults in
  that `.env` still apply.

See `Mila/Adaptors/Inference/Server/README.md` for the full protocol/endpoint and
configuration reference.

## VS Code Dev Container

`.devcontainer/devcontainer.json` layers `docker-compose.dev.yml` (which keeps the
container alive with `sleep infinity`) over this file. Open the folder in a container and
build with `mila-build-chat` from the integrated terminal.

## Troubleshooting

- **`Not installed in /mila/Data/Models/Store: <name>`** — nothing has been installed into
  the shared store yet, or you're running from a different repo checkout than the mount.
  Install with the chat harness's `/install <name>`; MIS itself never downloads.
- **No GPU in the container / CUDA init fails** — the NVIDIA Container Toolkit isn't active.
  Verify with `docker compose -f Docker/docker-compose.yml run --rm mila-dev nvidia-smi`.
  If `run` doesn't attach the GPU on your Docker version, use `up -d` + `exec` instead.
- **`no kernel image is available for execution`** — the binary was built for a different
  arch than the GPU. Rebuild with `MILA_CUDA_ARCH=<your arch>` (pass it via the compose
  `environment:` or `-e`).
