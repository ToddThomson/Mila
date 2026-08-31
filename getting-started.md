# Getting Started with Mila

This guide takes you from a fresh clone to building Mila, getting model weights ready, and
running inference. Section 7 shows how to consume Mila as a dependency in your own project;
if you want to contribute changes, Section 8 adds the fork-and-pull-request workflow on top.

**Who this is for.** Sections 1–6 apply to anyone building and running Mila. Section 7 is for
building your own application against Mila. Contributors are a superset — they do everything a
user does, then follow Section 8 for coding standards and the PR process. If you only want to
read about what Mila is and does, start with the [README](README.md).

Mila is a C++23 module-based library for open LLMs (CUDA/CPU inference and training), currently in public beta
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
| Doxygen | latest | Optional — only needed to build the API docs (`MILA_ENABLE_DOCS`, `ON` by default) |
| C++ Standard | C++23 | Modules, deducing-this, concepts |
| Python | 3.10+ | Only needed to convert a checkpoint Mila does not publish (Section 5b); validated on 3.14.5 |

Mila requires CUDA Toolkit 13.0 or newer. It is CI-tested on 13.0 and developed on 13.3;
newer 13.x releases are expected to work but are not exhaustively validated.

**Supported C++ compilers** — Mila's C++23 modules require a recent compiler:

| Compiler | Minimum | Notes |
|---|---|---|
| MSVC (Visual Studio) | 2026 18.6.2 | Primary Windows compiler |
| Clang | 19 | The Linux path CI and the container build |
| GCC | 16 | Alternative on Linux; 15.2 and earlier **cannot** compile the modules, and 15.3 is untested |

**Two compilers are involved in a Linux CUDA build, and only one of them is bound by that
table.** The compiler above compiles the C++23 module units. nvcc uses a separate host compiler
for the `.cu` files, which contain no modules — CI and the dev container pair clang-21 with
gcc-15 in that role, and its version is nvcc's business rather than the module floor's.

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
handle C++23 modules reliably. `MILA_ENABLE_TESTING` is `ON` for a clone of this repository and
`OFF` when Mila is embedded in another project, so `ctest` finds the suite without a flag.

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

**Clang is the Linux path** — clang-21 is what CI compiles and what the dev container runs, with
gcc-15 as nvcc's host compiler for the `.cu` files. Clang 19 is the floor.

GCC can compile the module units instead, and there the floor is **GCC 16**: 15.2 and earlier
cannot, and 15.3 has never been built. That is a different question from nvcc's host GCC, which
the container pins at 15.

The steps below target a recent Ubuntu (24.04 or 26.04) with CUDA through WSL.

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
   sudo apt-get install -y build-essential ninja-build git wget ca-certificates cmake libssl-dev
   sudo apt-get install -y clang-21     # compiles the module units; see the matrix above
   ```
   `libssl-dev` is curl's, not Mila's: `MILA_ENABLE_LIBCURL` defaults ON, and on Linux the
   vendored libcurl takes TLS from the system OpenSSL (`CURL_USE_SCHANNEL` is Windows-only),
   so its `find_package(OpenSSL)` fails the whole configure without it. Omit it only if you
   also configure with `-DMILA_ENABLE_LIBCURL=OFF`, which builds a Mila that can still list,
   install, locate and load models but cannot pull one without a caller-supplied transport.
   `build-essential` installs the distro's default GCC, which is what nvcc uses as its host
   compiler for the `.cu` files. Ubuntu 26.04 ships GCC 15.2, and that is fine in this role —
   the module floor does not apply to it.

   To compile the module units with GCC instead of Clang, install **GCC 16** and select it at
   configure time; 15.2 cannot compile them:
   ```bash
   sudo apt-get install -y gcc-16 g++-16     # then configure with -DCMAKE_CXX_COMPILER=g++-16
   ```
   Ubuntu 26.04 provides CMake ≥ 4.0 via apt; on older distros use Kitware's APT repo or the
   official tarball. Add `doxygen` only if you plan to build the docs.

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

On Linux, point CMake at both compilers and the CUDA toolkit explicitly. This is what CI and the
dev container run — clang-21 for the module units, gcc-15 as nvcc's host, CUDA 13.3:

```bash
cmake -S . -B out/build/linux-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-21 -DCMAKE_CXX_COMPILER=clang++-21 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=gcc-15 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler" \
  -DCMAKE_CXX_STANDARD=23 -DMILA_ENABLE_TESTING=ON
```

Do not put `-ccbin` in `CMAKE_CUDA_FLAGS` — it conflicts with the one CMake emits from
`CMAKE_CUDA_HOST_COMPILER`.

To compile the module units with GCC instead, swap in `-DCMAKE_C_COMPILER=gcc-16
-DCMAKE_CXX_COMPILER=g++-16` and drop `CMAKE_CUDA_HOST_COMPILER`.

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

Available presets, from `CMakePresets.json`:

| Platform | Presets |
|---|---|
| Windows | `x64-debug`, `x64-release`, `x64-release-ada`, `x64-release-blackwell`, `x64-profile` (RelWithDebInfo with device line info for Nsight), `x64-validate`, `x64-coverage`, `x64-debug-cpu-only`, `x64-debug-no-libcurl` |
| Linux / WSL | `linux-clang-debug`, `linux-clang-release`, `linux-clang-cpu-debug`, `linux-clang-cpu-release` |
| Packaging | `x64-wheel`, `linux-wheel`, `x64-release-cpm-gate` |

> **Building the API documentation.** `MILA_ENABLE_DOCS` is `ON` by default. Doxygen is not a
> hard requirement — without it the configure prints a warning and offers no `docs` target, and
> the library still builds. Graphviz is not needed either; the Doxyfile disables the call graphs.
> With Doxygen installed:
>
> ```bash
> cmake --build out/build/x64-release --target docs
> ```
>
> Output lands in `<build-dir>/docs`.

---

## 4. Build with Docker / dev container

If you do not want to install the CUDA/Clang/CMake toolchain locally, the development
container provides a reproducible Linux build environment (CUDA 13.3, clang-21, gcc-15 as nvcc's
host, CMake 4.2.3, Ninja) — handy from WSL. It mounts the repo at `/mila` with GPU access. Note this **still
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

The image sets `MILA_CACHE_DIR=/mila/Data/Models/Store`, which sits on the repo bind mount, so
a model installed (Section 5) from either side is the same store — install it once.

> A slim, published runtime image — `docker run … mila` for users who only want to run
> inference without building — is planned for the v0.20 release. See [ROADMAP.md](ROADMAP.md).

---

## 5. Get a model

**Model files are not stored in git**, so a fresh clone has no weights. There are two ways to
get one, and the first is the normal one.

### 5a. Install a published model

Mila publishes pre-quantized models under [`mila-llm`](https://huggingface.co/mila-llm). They
are ungated — no account, no access request, no token. From the chat harness:

```
/models --online          # what is published
/install gemma-4-12b-it-fp4
/models                   # what is installed, and what each costs in VRAM
```

Published today: `gemma-4-12b-it-fp4` (~6.3 GB, the chat default),
`Llama-3.2-3B-Instruct-fp4`, and `Llama-3.1-8B-Instruct-fp4`. The download lands in the local
store — `MILA_CACHE_DIR`, else the platform user cache — which Chat, the inference server and
the Python binding all share, so a model installed once is loadable by all of them.

The same thing from Python, with no C++ build involved:

```python
import mila
mila.initialize("warning")
mila.ModelStore().pull("gemma-4-12b-it-fp4", mila.default_hub_owner())
```

**Pull and load are separate verbs.** Loading never downloads, so a multi-gigabyte transfer
can never begin inside a chat prompt or an inference request — an uninstalled name is an
error, not a surprise.

### 5b. Convert a checkpoint (for families Mila does not publish)

Everything below is the fallback path: a family whose licence prevents republication, a
variant Mila has not published, or your own fine-tune. It needs a PyTorch environment,
HuggingFace authentication for a gated family, and enough disk for the source checkpoint.
Skip it entirely if 5a gave you what you need.

> Quantized variants (FP8, FP4) are produced by Mila at model load time — you only ever
> convert and store the **BF16** source files.

#### Set up the converter environment

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

#### GPT-2 (ungated)

```powershell
python Gpt2/convert_weights.py --model gpt2 --output ../../../Data/Models/Gpt2/gpt2_small_fp32.bin
```

#### Llama (upstream is gated — requires HuggingFace auth)

Meta's own repositories are gated, so converting from them means accepting the licence on the
model page and authenticating. (Mila's published Llama artifacts are not gated — 5a needs
none of this.)

```powershell
hf auth login
```

Convert the tokenizer once (shared across all Llama 3.x variants), then the weights you want:

```powershell
# Tokenizer (shared)
python Llama/convert_tokenizer.py --model meta-llama/Llama-3.2-3B-Instruct --output ../../../Data/Models/Llama/llama_tokenizer.bin

# Smallest model — good for a first run
python Llama/convert_weights.py --model meta-llama/Llama-3.2-1B-Instruct --output ../../../Data/Models/Llama/llama32_1b_instruct_bf16.bin

# Larger (~16 GB host RAM to convert)
python Llama/convert_weights.py --model meta-llama/Llama-3.1-8B-Instruct --output ../../../Data/Models/Llama/llama31_8b_instruct_bf16.bin
```

Model files are large — make sure you have adequate disk space before converting.

#### Put the converted model in the store

A loose `.bin` is not a model Mila can name. `ExportArtifact` turns one into a store entry —
export to safetensors, wrap it in a package with a manifest, then install:

```powershell
ExportArtifact Data/Models/Llama/llama32_1b_instruct_bf16.bin out/llama32-1b.safetensors
ExportArtifact --package out/llama32-1b-package --weights out/llama32-1b.safetensors --instruct
ExportArtifact --install out/llama32-1b-package
```

Every file is hashed as it is adopted, so a blob is trusted only after its digest matches the
manifest. Afterwards the model is listed by `/models` and loaded by name exactly like a
published one — that is the point of one manifest describing every model, whatever its origin.

> `--instruct` is not implied by the model's name. Omitting it writes `instruct: false` and
> every consumer then applies the wrong prompt template.

See [Mila/Tools/Converters/README.md](Mila/Tools/Converters/README.md) for the full option
tables and per-model notes.

---

## 6. Run inference (Chat CLI)

The chat sample builds as the `mila-chat` target. Its executable is written to the build root
(e.g. `out/build/x64-release/mila-chat.exe`), with a `Data/` folder copied alongside it.

```bash
./out/build/x64-release/mila-chat.exe
```

From Visual Studio, set **mila-chat** as the startup item and run. Chat opens on an empty store,
so a first run with nothing installed still reaches `/install`.

A model is named, not aliased — what `/models` shows is what you type:

```
/models                            # installed, with what each costs in VRAM
/model Llama-3.2-3B-Instruct-fp4   # switch (clears history)
/model                             # current model and quantization
/help
```

The default is `gemma-4-12b-it-fp4`.

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
    GIT_TAG        v0.20.0-beta.2    # pin to a published release tag
)
FetchContent_MakeAvailable(Mila)

add_executable(my_app main.cpp)      # main.cpp does: import Mila; -- after its #includes
target_link_libraries(my_app PRIVATE Mila::Mila)

# Clang consumers only (MSVC auto-configures module consumption):
# target_compile_options(my_app PRIVATE -fno-implicit-modules -fno-implicit-module-maps)
```

`GIT_TAG`, `URL` (a release archive), and `SOURCE_DIR` (a local working tree) are interchangeable
in `FetchContent_Declare`. Because C++23 module BMIs are not portable, your toolchain recompiles
Mila's module units from source — inherent to consuming any module library, not specific to Mila.

**Budget for the first build**: your project builds all of Mila, every CUDA kernel included, so
set `-DCMAKE_CUDA_ARCHITECTURES` to your own GPU's arch (`89` for Ada) as in Section 3 — the
default multi-arch list compiles every kernel several times over. Rebuilds are incremental.

A complete, copy-paste starting point is in
[`Mila/Samples/QuickStart/Cpp`](Mila/Samples/QuickStart/Cpp/README.md) — a standalone project that
consumes Mila exactly like this and then loads a model and streams a reply, so the `CMakeLists.txt`
above is shown doing real work rather than printing a version.

**Two rules for the translation unit that imports Mila**, both from an MSVC modules defect and both
temporary. Put `import Mila;` **after** your `#include`s — importing first stops the build with
C1116. Include `<sstream>` among them if you instantiate a model, and read input with `std::fgets`
rather than `std::getline` or `std::cin.getline`. The QuickStart sample above shows both, commented
where they appear.

> **`find_package(Mila)`?** Not supported, and removed in 0.20.0-beta.3. A module library is a
> source distribution, so `find_package`'s prebuilt-binary benefit is void while its install
> layout costs real maintenance. FetchContent is the one supported path.

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

- [ROADMAP.md](ROADMAP.md) — the release narrative and trajectory; [BACKLOG.md](BACKLOG.md) for the task breakdown.
- [CLAUDE.md](CLAUDE.md) — architecture overview, type axes, dispatch, and code style.
- `Mila/Specifications/` — design documents:
  [OperationDispatch.md](Mila/Specifications/OperationDispatch.md),
  [Quantization.md](Mila/Specifications/Quantization.md), and the planned-feature
  specs (PromptCaching, TokenSampling, ToolCalling).
- API reference: https://mila.toddt.me/api/ (regenerated on every push to master).

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Chat reports a model is not installed | Nothing in the store yet — `/models --online` then `/install <name>`, see Section 5a. Weights are not in git. |
| `hf auth login` fails or model 403s | You have not accepted Meta's license on the HuggingFace model page. |
| Module / incremental build errors with MSBuild | Use the **Ninja** generator — MSBuild does not handle C++23 modules well. |
| Out-of-memory converting Llama 3.1 8B | Conversion needs ~16 GB host RAM; convert in BF16 (the default). |
| FP8/FP4 produce garbage or fail | Requires an SM 8.9+ (Ada Lovelace or newer) GPU. |
