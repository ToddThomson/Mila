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
- **Model weights**, converted offline (`Mila/Tools/Converters`) and present under the
  repo's `Data/Models/`. The bind mount exposes them to the container; they are **never
  baked into the image**. The default model (`gemma-12b`) needs:
  - `Data/Models/gemma/gemma4_12b_it_bf16.bin`
  - `Data/Models/gemma/gemma_tokenizer.bin`

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

Convenience wrappers under `scripts/` do the same (`.sh` and `.ps1`):

```
scripts/build-docker.{sh,ps1}   # docker compose build
scripts/chat-build.{sh,ps1}     # mila-build-chat
scripts/chat-run.{sh,ps1}       # mila-chat   (args forwarded, e.g. --help)
```

## How it fits together

- **Source** is bind-mounted at `/mila`. The Chat build compiles `MODELS_DIR` in as the
  absolute path `/mila/Data/Models`, so weights on the host resolve with no extra config.
- **Build artifacts** go to `/build`, a container-local named volume (`mila-build`) kept
  *off* the bind mount — the C++23 module BMI I/O is metadata-heavy and slow across the
  host↔container filesystem boundary. `ccache` persists in the `mila-ccache` volume. Both
  survive `run --rm`, so rebuilds are incremental.
- **`mila-build-chat`** configures + builds only the `ChatApp` target (no tests, samples,
  profiling, docs, or Python binding). **`mila-chat`** `cd`s into `/build` (where the
  POST_BUILD step copies `Data/`) and runs `ChatApp`; arguments are forwarded.

## VS Code Dev Container

`.devcontainer/devcontainer.json` layers `docker-compose.dev.yml` (which keeps the
container alive with `sleep infinity`) over this file. Open the folder in a container and
build with `mila-build-chat` from the integrated terminal.

## Troubleshooting

- **`Model file not found: /mila/Data/Models/gemma/...`** — the weights aren't on the host
  under `Data/Models/`, or you're running from a different repo checkout than the mount.
- **No GPU in the container / CUDA init fails** — the NVIDIA Container Toolkit isn't active.
  Verify with `docker compose -f Docker/docker-compose.yml run --rm mila-dev nvidia-smi`.
  If `run` doesn't attach the GPU on your Docker version, use `up -d` + `exec` instead.
- **`no kernel image is available for execution`** — the binary was built for a different
  arch than the GPU. Rebuild with `MILA_CUDA_ARCH=<your arch>` (pass it via the compose
  `environment:` or `-e`).
