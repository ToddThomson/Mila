# Getting Started with Mila

This guide takes you from a fresh clone to building Mila, getting model weights ready, and
running inference. Section 7 shows how to consume Mila as a dependency in your own project;
if you want to contribute changes, Section 8 adds the fork-and-pull-request workflow on top.

**Who this is for.** Sections 1–6 apply to anyone building and running Mila. Section 7 is for
building your own application against Mila. Contributors are a superset — they do everything a
user does, then follow Section 8 for coding standards and the PR process. If you only want to
read about what Mila is and does, start with the [README](README.md).

Mila is a C++23 module-based library for open LLMs (CUDA/CPU inference and training), currently in late alpha
(feature-frozen, hardening toward the v0.20 first production release). Pre-1.0, breaking
changes are still expected — backward compatibility is not yet a goal.

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Visual Studio | 2026 18.6.2 or newer | Windows path — "Desktop development with C++" workload; earlier 2026 builds have a C++23 module regression |
| VS Code | 1.122 or newer | Linux / WSL path — with the WSL, C/C++, and CMake Tools extensions |
| Git | 2.x or newer | Git for Windows (validated on 2.54.0) / distro `git`. Required at configure time — CPM fetches dependencies via `git`. GitHub Desktop is optional |
| Docker | Docker Desktop (WSL2) or native `docker-ce` in WSL | Optional — only for the Docker dev-container path (Section 4); GPU access also needs the NVIDIA Container Toolkit |
| CUDA Toolkit | 13.0 or newer | Required for the CUDA backend |
| CMake | 4.0 or newer | Bundled with recent Visual Studio |
| Ninja | latest | Required for fast C++23 module incremental builds |
| GTest | 1.17.0 | Fetched by the build |
| Doxygen | latest | Optional — only needed to build the API docs (`-DMILA_ENABLE_DOCS=ON`) |
| Graphviz (dot) | latest | Optional — renders the call/caller graphs in the generated docs |
| C++ Standard | C++23 | Modules, deducing-this, concepts |
| Python | 3.10+ | Only needed to convert model weights (Section 5); validated on 3.14.5 |

Mila requires CUDA Toolkit 13.0 or newer. It is CI-tested on 13.0 and developed on 13.3;
newer 13.x releases are expected to work but are not exhaustively validated.

**Supported C++ compilers** — Mila's C++23 modules require a recent compiler:

| Compiler | Minimum | Notes |
|---|---|---|
| MSVC (Visual Studio) | 2026 18.6.2 | Primary Windows compiler |
| Clang | 19 | CI-validated on Linux |
| GCC | 15.3 | 15.2 and earlier **cannot** compile the modules; on Ubuntu 26.04 install the `gcc-16` package |

In the CUDA build, the compiler above compiles the C++23 module units; nvcc uses a separate
host compiler for `.cu` files (which contain no modules), so an older GCC is fine for that role.

A CUDA-capable NVIDIA GPU is needed to run the CUDA inference paths. The library builds
without a GPU, but the validated inference targets (Llama, GPT-2) run on CUDA. BF16
compute and FP8/FP4 quantization require an Ada Lovelace (SM 8.9) or newer GPU for full
Tensor Core support.

> **Prefer not to install the toolchain by hand?** Section 4 covers the Docker / dev
> container path, which gives you a reproducible Linux build environment (it still builds
> from source, and still needs a GPU for inference).

---

## 2. Get the code

Clone the repository:

```bash
git clone https://github.com/toddthomson/mila.git
cd mila
```

That is all you need to build and run Mila. **If you intend to contribute**, use the
fork-and-pull-request workflow in [Section 8](#8-contributing) instead — fork first, then
clone your fork.

---

## 3. Build and test

The repository ships CMake presets in `CMakePresets.json`. The output directory is always
`out/build/<preset-name>`. Ninja is the generator on every platform — MSBuild does not
handle C++23 modules reliably. Tests are opt-in: configure with `-DMILA_ENABLE_TESTING=ON`
(it defaults to `OFF`) or `ctest` will find nothing to run.

The first configure fetches dependencies through CPM (GoogleTest, nlohmann_json, miniz,
CUTLASS, and others), which runs `git clone` under the hood — so **`git` must be installed
and on `PATH`**, and network access is required, even though you already cloned the repo.

### Windows (Visual Studio)

Use **Visual Studio 2026 18.6.2 or newer** — earlier 2026 builds have a regression that
breaks the C++23 module build; 18.6.2 fixed it.

1. Launch Visual Studio and choose **Open a local folder**, then select the cloned repo.
2. Visual Studio detects `CMakeLists.txt` and configures automatically.
3. Pick a preset (`x64-debug` or `x64-release`) from the configuration dropdown.
4. Build with **F7** (or **Build > Build All**).

### Linux / WSL (VS Code)

Two C++ compilers are supported on Linux: **Clang ≥ 19** (CI-validated on Ubuntu 24.04) and
**GCC ≥ 15.3** (validated with GCC 16 on Ubuntu 26.04). GCC 15.2 and earlier cannot compile
the C++23 modules, so on an older distro use Clang. The steps below target a recent Ubuntu
(24.04 or 26.04) with CUDA through WSL.

1. **Set up WSL** (skip on native Linux). From an elevated PowerShell on Windows:
   ```powershell
   wsl --install                    # installs the latest Ubuntu LTS (e.g. 26.04)
   # or pin a release:  wsl --install -d Ubuntu-24.04
   ```
   Reboot if prompted, then open the Ubuntu shell to finish user setup. Check the release
   with `lsb_release -a`.

2. **Install the CUDA toolkit inside WSL.** Do **not** install a Linux display driver — the
   Windows NVIDIA driver provides the GPU through WSL. NVIDIA's `wsl-ubuntu` repo is
   version-agnostic, so it works regardless of the Ubuntu release. For CUDA 13.3:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
   sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/13.3.0/local_installers/cuda-repo-wsl-ubuntu-13-3-local_13.3.0-1_amd64.deb
   sudo dpkg -i cuda-repo-wsl-ubuntu-13-3-local_13.3.0-1_amd64.deb
   sudo cp /var/cuda-repo-wsl-ubuntu-13-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-13-3
   ```
   The toolkit does **not** add itself to `PATH` — that is a manual post-install step:
   ```bash
   echo 'export PATH=/usr/local/cuda-13.3/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   nvcc --version | grep release        # expect: release 13.3
   ```

3. **Install the build tools and a C++ compiler.**
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential ninja-build git wget ca-certificates cmake
   gcc --version                        # need GCC >= 15.3 (see the compiler matrix above)
   ```
   `build-essential` installs the distro's default GCC. Mila needs **GCC ≥ 15.3** — Ubuntu
   26.04 currently ships 15.2, which is too old, so install a newer one and select it at
   configure time:
   ```bash
   sudo apt-get install -y gcc-16 g++-16     # then configure with -DCMAKE_CXX_COMPILER=g++-16
   ```
   Prefer Clang? `sudo apt-get install -y clang clangd` (≥ 19). Ubuntu 26.04 provides
   CMake ≥ 4.0 via apt; on older distros use Kitware's APT repo or the official tarball.
   Add `doxygen graphviz` only if you plan to build the docs.

4. **Install VS Code (latest) with the WSL workflow.** Install
   [VS Code](https://code.visualstudio.com/) on Windows, add the **WSL**, **C/C++**, and
   **CMake Tools** extensions, then from the WSL shell open the repo:
   ```bash
   cd ~/mila && code .
   ```
   VS Code reattaches inside WSL (the status bar shows **WSL: Ubuntu**). Configure and build
   from the CMake Tools panel, or use the command line below.

### Command line (any platform)

```bash
# Configure (Ninja, Release, with tests enabled)
cmake -S . -B out/build/x64-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DMILA_ENABLE_TESTING=ON

# Build
cmake --build out/build/x64-release

# Run the full test suite
ctest --test-dir out/build/x64-release
```

On Linux, point CMake at your chosen compiler and the CUDA toolkit explicitly. GCC example
(Ubuntu 26.04 + GCC 16 + CUDA 13.3):

```bash
cmake -S . -B out/build/linux-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-16 -DCMAKE_CXX_COMPILER=g++-16 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler" \
  -DCMAKE_CXX_STANDARD=23 -DMILA_ENABLE_TESTING=ON
```

For the CI-validated Clang path, swap in `-DCMAKE_C_COMPILER=clang-19
-DCMAKE_CXX_COMPILER=clang++-19` and set `-DCMAKE_CUDA_HOST_COMPILER=gcc-14` (nvcc's
host compiler for the `.cu` files -- do not put `-ccbin` in `CMAKE_CUDA_FLAGS`, that
conflicts with the one CMake emits from `CMAKE_CUDA_HOST_COMPILER`).

Notes:
- Set `-DCMAKE_CUDA_ARCHITECTURES` to your GPU's arch (`89` = Ada). `native` fails on GPUs
  CUDA 13 has dropped (e.g. Maxwell `sm_50`), and building a single arch is far faster than
  the default multi-arch list.
- `--allow-unsupported-compiler` lets nvcc proceed with a host GCC newer than it officially lists.
- Add `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` to feed clangd in VS Code, and build with
  `cmake --build <dir> -- -k 0` to collect every error in one pass.

Run a single test binary directly, for example:

```bash
./out/build/x64-release/Mila/Tests/Dnn/Components/Activations/Gelu/GeluTests
```

Available presets: `x64-debug`, `x64-release`, `x64-profile` (RelWithDebInfo with device
line info for Nsight), `x86-debug`, `x86-release`, plus `linux-debug` and `macos-debug`
for remote / WSL targets.

> **Building the API documentation (optional).** Doc generation is opt-in via the
> `MILA_ENABLE_DOCS` option (default `OFF`), so a normal library/test build needs neither
> Doxygen nor Graphviz. To build the docs, install both, then configure with the option on
> and build the `docs` target:
>
> ```bash
> cmake -S . -B out/build/x64-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DMILA_ENABLE_DOCS=ON
> cmake --build out/build/x64-release --target docs
> ```
>
> Output lands in `<build-dir>/docs`. Note: when `MILA_ENABLE_DOCS=ON`, Doxygen becomes a
> hard configure-time requirement (the `Docs` target uses `find_package(Doxygen REQUIRED)`).

---

## 4. Build with Docker / dev container

If you do not want to install the CUDA/Clang/CMake toolchain locally, the development
container provides a reproducible Linux build environment (CUDA 13.0, Clang 19, CMake 4.x,
Ninja) — handy from WSL. It mounts the repo at `/mila` with GPU access. Note this **still
builds Mila from source** inside the container; it removes toolchain setup, not the build.
(A pull-and-run published image is planned for beta — see the note at the end of this section.)

**Prerequisites for this path:**

- A Docker engine reachable from your shell — either **Docker Desktop for Windows** with
  WSL2 integration enabled, or **Docker Engine installed natively inside the WSL distro**
  (`docker-ce`). Docker Desktop is free for personal, small-business, education, and
  open-source use but requires a paid license for larger commercial use.
- For GPU access: the **NVIDIA driver on Windows** (the same one the bare-metal WSL path
  uses) and the **NVIDIA Container Toolkit**. Docker Desktop largely sets this up; with
  native in-WSL Docker you install `nvidia-container-toolkit` in the distro yourself.

```bash
docker compose -f Docker/docker-compose.yml run --rm mila-dev

# Inside the container:
cmake -S . -B out/build/linux-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DMILA_ENABLE_TESTING=ON
cmake --build out/build/linux-release
ctest --test-dir out/build/linux-release
```

VS Code users can **Reopen in Container** — `.devcontainer/` wires up the compose service,
GPU access, the repo mount, and the C/C++ / CMake Tools / clangd extensions.

Model weights are converted offline on the host (Section 5); the repo bind mount makes the
converted `.bin` files available inside the container automatically.

> A slim, published runtime image — `docker run … mila` for users who only want to run
> inference without building — is planned for the beta release. See [ROADMAP.md](ROADMAP.md).

---

## 5. Get model weights (required for inference)

**Model weight files are not stored in git.** Everything under `Data/Models/` is gitignored
(covered by the global `*.bin` rule), so a fresh clone has the directory scaffold but no
weights. You generate them locally by converting HuggingFace checkpoints with the Python
converters in `Mila/Tools/Converters/`.

> Quantized variants (FP8, FP4) are produced by Mila at model load time — you only ever
> convert and store the **BF16** source files.

### 5a. Set up the converter environment

Run once from the `Mila/Tools/Converters/` directory:

```powershell
cd Mila/Tools/Converters
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Conversion runs on CPU — a GPU PyTorch wheel is not required. Always run the scripts from
the `Converters/` root so `common.py` is on the Python path.

Use Python 3.10 or newer (validated on 3.14.5). PyTorch and Transformers publish wheels per
Python minor version and can lag brand-new releases — if `pip install` cannot find a torch
wheel for your interpreter, create the venv with a slightly older minor (e.g. 3.12).

### 5b. GPT-2 (ungated, easiest first target)

```powershell
python Gpt2/convert_weights.py --model gpt2 --output ../../../Data/Models/Gpt2/gpt2_small_fp32.bin
```

### 5c. Llama (gated — requires HuggingFace auth)

Llama checkpoints are gated. Accept Meta's license on the model page, then authenticate:

```powershell
hf auth login
```

Convert the tokenizer once (shared across all Llama 3.x variants), then the weights you want.
The chat CLI's default model is **Llama 3.1 8B FP4**, which needs `llama31_8b_instruct_bf16.bin`:

```powershell
# Tokenizer (shared)
python Llama/convert_tokenizer.py --model meta-llama/Llama-3.2-3B-Instruct --output ../../../Data/Models/Llama/llama_tokenizer.bin

# Smallest model — good for a first run
python Llama/convert_weights.py --model meta-llama/Llama-3.2-1B-Instruct --output ../../../Data/Models/Llama/llama32_1b_instruct_bf16.bin

# Chat CLI default (large — ~16 GB host RAM to convert)
python Llama/convert_weights.py --model meta-llama/Llama-3.1-8B-Instruct --output ../../../Data/Models/Llama/llama31_8b_instruct_bf16.bin
```

The chat app resolves weights relative to the repo's `Data/Models/` directory, so place
the `.bin` files there (under `Gpt2/` and `Llama/`) and they will be found automatically.
Model files are large — make sure you have adequate disk space before converting.

See [Mila/Tools/Converters/README.md](Mila/Tools/Converters/README.md) for the full option
tables and per-model notes.

---

## 6. Run inference (Chat CLI)

The chat sample builds as the `ChatApp` target. Its executable is written to the build root
(e.g. `out/build/x64-release/ChatApp.exe`), with a `Data/` folder copied alongside it.

```bash
./out/build/x64-release/ChatApp.exe
```

From Visual Studio, set **ChatApp** as the startup item and run. Once you have the smaller
weights converted, switch models at runtime:

```
/model llama-1b        # Llama 3.2 1B
/model llama-3b        # Llama 3.2 3B
/model llama-8b fp4    # Llama 3.1 8B, FP4 quantized (the default)
/model gpt2
```

Model aliases: `gpt2`, `llama-1b`, `llama-3b`, `llama-8b`, with `-fp32` variants. The
`llama-8b` alias uses the `llama31` family prefix; 1B/3B use `llama32`.

The MNIST training sample (`Samples/MNIST`) is another good way to exercise a full
forward + backward + AdamW loop.

---

## 7. Consume Mila in your own project (FetchContent)

To build your own application against Mila, pull it in with **FetchContent** — the supported way
to depend on Mila. Mila compiles once, in your project's own toolchain (no install step, no
prebuilt/recompiled ABI split); this is the same mechanism Mila uses for its own dependencies
(googletest, CUTLASS, nlohmann).

```cmake
cmake_minimum_required(VERSION 4.0)
project(MyApp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)   # your toolchain recompiles Mila's module units

include(FetchContent)
FetchContent_Declare(
    Mila
    GIT_REPOSITORY https://github.com/ToddThomson/Mila.git
    GIT_TAG        v0.20.0           # pin to a published release tag
)
FetchContent_MakeAvailable(Mila)

add_executable(my_app main.cpp)      # main.cpp does: import Mila;
target_link_libraries(my_app PRIVATE Mila::Mila)

# Clang consumers only (MSVC auto-configures module consumption):
# target_compile_options(my_app PRIVATE -fno-implicit-modules -fno-implicit-module-maps)
```

`GIT_TAG`, `URL` (a release archive), and `SOURCE_DIR` (a local working tree) are interchangeable
in `FetchContent_Declare`. Because C++23 module BMIs are not portable, your toolchain recompiles
Mila's module units from source — inherent to consuming any module library, not specific to Mila.

A complete, copy-paste starting point is in
[`Mila/Samples/QuickStart`](Mila/Samples/QuickStart/README.md).

> **`find_package(Mila)`?** Parked in favor of FetchContent: a module library is a source
> distribution, so `find_package`'s prebuilt-binary benefit is void while its install layout is
> pure maintenance surface. It remains on disk, opt-in only.

---

## 8. Contributing

Everything above gets you building and running Mila. Contributing adds the
fork-and-pull-request workflow on top.

### Fork and branch

Mila uses a fork-and-pull-request workflow. All contributions target the `dev` branch —
never `master`.

```bash
# Fork https://github.com/toddthomson/mila on GitHub, then clone your fork
git clone https://github.com/<your-username>/mila.git
cd mila

# Add the canonical repository as "upstream" to stay in sync
git remote add upstream https://github.com/toddthomson/mila.git

# Create a feature branch from dev
git fetch upstream
git checkout -b my-feature upstream/dev
```

Keep your branch current by rebasing on `upstream/dev` before opening a PR:

```bash
git fetch upstream
git rebase upstream/dev
```

### Coding standards

Code style is defined authoritatively in [CLAUDE.md](CLAUDE.md) (the "Code Style" and
"C++ Module Conventions" sections) and the process is in [CONTRIBUTING.md](CONTRIBUTING.md).
Key points:

- **No abbreviations in identifiers** — `Quantization` not `Quant`, `Index` not `Idx`,
  `TWeightQuantization` not `TWeightQuant`. Established acronyms (`Kv`, `Gqa`, `Mha`,
  `Mlp`, `Lpe`, `Bpe`) are allowed.
- C++23 modules: `.ixx` for interface units; module names mirror the directory structure
  (e.g. `Dnn.Components.Linear`). Backend specializations live in `:Cuda` / `:Cpu` partitions.
- Single-space formatting (no column alignment). Blank line before control-flow blocks and
  before a final `return`; no blank line for early-return guard clauses.
- Comments explain **why**, never restate the code. ASCII only in comments.

New work should follow the compile-time dispatch pattern: components resolve their operation
type through `OperationTraits<OperationType, TDeviceType, TPrecision, TPolicy>`. `Linear`
([Mila/Src/Dnn/Components/Linear/Linear.ixx](Mila/Src/Dnn/Components/Linear/Linear.ixx)) is
the reference implementation. Do not add new string-keyed `OperationRegistry`/`*Registrar`
classes — that path is being removed. See
[Mila/Specifications/OperationDispatch.md](Mila/Specifications/OperationDispatch.md).

### Tests are required

Every new component must include **forward and backward** pass tests, with CPU/CUDA
equivalence where applicable. Tests live under `Mila/Tests/Dnn/` and mirror the `Src/Dnn`
tree. Run the suite with `ctest` before opening a PR.

Good first contributions, per the README: CPU reference implementations, additional test
coverage, and new encoding strategies under `Mila/Src/Dnn/Components/Encodings/`.

### Open the pull request

1. Make focused commits with clear messages.
2. Ensure the full build is clean and `ctest` passes.
3. Push your branch to your fork.
4. Open a PR **targeting `dev`** and fill out the PR template.
5. A maintainer reviews and approves before merge.

---

## 9. Where to go next

- [ROADMAP.md](ROADMAP.md) — current alpha status (Alpha.5) and the full task breakdown.
- [CLAUDE.md](CLAUDE.md) — architecture overview, type axes, dispatch, and code style.
- `Mila/Specifications/` — design documents:
  [OperationDispatch.md](Mila/Specifications/OperationDispatch.md),
  [Quantization.V2.md](Mila/Specifications/Quantization.V2.md), and the planned-feature
  specs (PromptCaching, TokenSampling, ToolCalling).
- API reference: https://toddthomson.github.io/Mila (regenerated on every push to master).

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Chat app reports a missing `.bin` file | Weights not converted yet — see Section 5. They are not in git. |
| `hf auth login` fails or model 403s | You have not accepted Meta's license on the HuggingFace model page. |
| Module / incremental build errors with MSBuild | Use the **Ninja** generator — MSBuild does not handle C++23 modules well. |
| Out-of-memory converting Llama 3.1 8B | Conversion needs ~16 GB host RAM; convert in BF16 (the default). |
| FP8/FP4 produce garbage or fail | Requires an SM 8.9+ (Ada Lovelace or newer) GPU. |
