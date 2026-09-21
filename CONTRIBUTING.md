# Contributing to Mila

Thank you for your interest in contributing to Mila! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

We expect all contributors to be respectful and constructive. By participating in this project, you agree to maintain a welcoming, inclusive, and harassment-free environment for everyone.

### No self-promotion or spam

Mila's Issues and Discussions are for the project itself — its code, design, and use. Unsolicited promotion is not welcome and will be removed without notice, including:

- Advertising a product, service, paid offering, or hosting/GPU provider.
- Comments whose main purpose is to drive traffic to an external link, especially generic praise followed by a promotional pitch.
- Recruiting, soliciting, or affiliate/referral links.

Mentioning a relevant open-source tool or a personal project in genuine, on-topic context is fine. The line is intent: are you contributing to the conversation, or using it as a billboard? Maintainers may hide, delete, or lock content and block accounts at their discretion. To reduce drive-by spam, Discussion and Issue participation may be rate-limited or restricted to existing contributors during high-noise periods.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Add the original repository as a remote named "upstream"
4. Create a branch from `upstream/dev` for your feature or bug fix
5. Make your changes and commit them
6. Push to your fork and open a pull request against `dev`

For a full walkthrough — build, model weight conversion, running inference, and your
first PR — see [getting-started.md](getting-started.md).

## Development Environment

### Required Components
* A C++23 compiler with module support: **MSVC** (Visual Studio 2026 18.6.2+), **Clang 19+**, or **GCC 16+**. GCC 15.2 and earlier cannot compile Mila's C++23 modules, and 15.3 has never been built; on Ubuntu 26.04 install the `gcc-16` package. (In CUDA builds the C++ compiler handles the modules; nvcc uses a separate host compiler for `.cu` files, where an older GCC is acceptable.)
* Git 2.x or newer, on `PATH` (validated on 2.54.0; used to clone, and required at CMake configure time — CPM fetches dependencies via `git clone`). GitHub Desktop is an optional convenience
* NVIDIA CUDA Toolkit 13.3 (the version CI builds with) — **only for GPU builds**; see below
* CMake 4.0 or newer
* Ninja (required for fast C++23 module incremental builds)
* GTest 1.17.0 for unit testing
* Doxygen and Graphviz (optional — only when building the API docs with `-DMILA_ENABLE_DOCS=ON`)

### Contributing CUDA and kernel work

This is where Mila most needs help, and where a contribution moves the library furthest. New
operations, faster kernels, a quantization format, an attention variant, support for an
architecture nobody here owns — all welcome.

**What you can check yourself, without model weights.** Every component has CPU/CUDA equivalence
tests named `*_MatchesReference` (see `Mila/Tests/Dnn/Components/`). They build a small tensor,
run both paths and compare. A new kernel should arrive with one, and that test is the single most
useful thing in a kernel PR — it is what lets someone else trust the change without owning your
card. Cases self-skip with `CUDA device not available`, so the same suite stays green for
contributors without a GPU.

**Model-level parity tests need weights**, which are a download rather than a barrier:
`mila models --online` lists what is published and `mila install <name>` fetches one.

**Architectures.** Mila supports **SM 8.0 and newer — Ampere onward**. The library builds for
`80;86;89;90;120`, and the published wheels and images carry the same list. That floor is drawn by
the kernels rather than chosen: the FP4 GEMM and both GQA flash prefill paths refuse anything
below it. For fast iteration on one card, `x64-release-ada` (sm_89) and `x64-release-blackwell`
(sm_120) pin a single architecture, or pass `-DMILA_LIBRARY_CUDA_ARCHITECTURES=<arch>` for yours.

One trap worth knowing: a binary built for the full list still runs on a newer card by JIT from
older PTX, and at the prompt that is indistinguishable from a native build. If you are making a
claim about a specific architecture, pin it and say which preset you used.

**Measurements.** If your change is about speed, three things turn a number into something
reviewable: build the baseline and the change from the same tree one after the other; say whether
the weights were resident in GPU memory or streaming from disk, because the ranking changes
between them; and run it more than once.

**Hardware the project can check on** is Ada (sm_89) and Blackwell (sm_120). If your change targets
anything else — Ampere, Hopper, a datacentre part — say so in the PR. It can still land, but we
should both know what went verified and what did not. **Pull Request Process** below has the rest
of what happens after you open it.

### No NVIDIA GPU? You can still build and contribute

Mila builds without CUDA. Configure with `MILA_ENABLE_CUDA=OFF` — preset **`x64-debug-cpu-only`**
on Windows, **`linux-clang-cpu-release`** on Linux — and you get the library, the unit tests and a
working `ctest` run with no Toolkit and no card installed.

There is real work there too. The CPU operations in `OperationTraits.Cpu.ixx` are the clearest
example: self-contained, with CUDA counterparts to check against, and their absence is what stops
Mila running anywhere but an NVIDIA machine.

Run the CPU-only build and its tests before you open a pull request, exactly as a contributor with
a card runs `x64-validate` — the expectation is the fullest validation your hardware allows, not a
lighter one.

The single thing that leaves uncovered is whether your change still compiles with CUDA enabled.
CI answers that one question on the PR, which is why it exists; nobody expects you to own the
hardware. It answers nothing else, so say in the PR that you have no GPU — that tells the
maintainer what is left to check on real cards.

### Build Instructions

#### Using Visual Studio

1. **Prerequisites**
   - Visual Studio 2026 18.6.2 or newer with "Desktop development with C++" workload (earlier 2026 builds have a C++23 module regression that breaks the build)
   - CUDA Toolkit 13.3
   - CMake 4.0 or newer (included with Visual Studio)

2. **Open the Project**
   - Launch Visual Studio
   - Select "Open a local folder" and navigate to your cloned Mila repository
   - Visual Studio will automatically detect the CMakeLists.txt file

3. **Configure Project**
   - Visual Studio will automatically generate CMake cache
   - To customize build settings, right-click on CMakeLists.txt and select "CMake Settings for MilaProject"
   - Under "Configuration type", select "Release" for optimal performance

4. **Build the Project**
   - Right-click on CMakeLists.txt and select "Build All"
   - Alternatively, use the Build menu or press F7

5. **Run Tests**
   - In the Solution Explorer, expand the "Tests" folder
   - Right-click on a test project and select "Run Tests"

#### Using Visual Studio Code

1. **Prerequisites**
   - Visual Studio Code 1.122 or newer
   - C/C++ extension
   - CMake Tools extension
   - CUDA Toolkit 13.3
   - CMake 4.0 or newer

2. **Open the Project**
   - Launch VS Code
   - Open the folder containing your cloned Mila repository
   - VS Code should detect the CMake project automatically

3. **Configure Project**
   - Press Ctrl+Shift+P to open the command palette
   - Type "CMake: Configure" and select it
   - Choose your preferred generator (Ninja is recommended for faster builds)
   - Select the build variant (Debug/Release)

4. **Build the Project**
   - Press Ctrl+Shift+P to open the command palette
   - Type "CMake: Build" and select it, or use the build button in the status bar

5. **Run Tests**
   - Press Ctrl+Shift+P to open the command palette
   - Type "CMake: Run Tests" and select it
   - Alternatively, use the Test Explorer extension to browse and run tests

## Coding Standards

These are summarized here; the authoritative reference is the "Code Style" and
"C++ Module Conventions" sections of [CLAUDE.md](CLAUDE.md).

### Naming Conventions
* **No abbreviations in identifiers.** Spell every name out in full: `Quantization` not
  `Quant`, `Parameter` not `Param`, `Context` not `Ctx`, `Index` not `Idx`,
  `Implementation` not `Impl`. This applies to template parameters too:
  `TWeightQuantization` not `TWeightQuant`. The only exceptions are established acronyms:
  `Kv`, `Gqa`, `Mha`, `Mlp`, `Lpe`, `Bpe`.
* Use `PascalCase` for class, struct, and enum names
* Use `camelCase` for method and variable names
* Use `UPPER_CASE` for constants and macros
* Private member variables are suffixed with an underscore (e.g., `weight_scales_`)
* Template parameters use the `TFoo` convention (e.g., `TDeviceType`, `TComputePrecision`)

### Code Formatting

Formatting is encoded in [`.editorconfig`](.editorconfig) and [`.clang-format`](.clang-format)
at the repository root. Visual Studio 2026 applies `.editorconfig` natively; for VS Code and
other editors, `.clang-format` is a best-effort match (see its header for the rules it cannot
express). The rules below are the human-reviewed conventions, some of which no formatter can
enforce (notably full-word identifiers and the blank-line placement rules):

* Use 4 spaces for indentation (no tabs)
* Opening braces go on their **own line** (Allman style), for namespaces, classes,
  functions, and control statements
* Single-space formatting throughout — do **not** align consecutive lines with extra spaces
* Blank line before control-flow blocks (`if`, `for`, `while`, `switch`) and after the
  closing brace of a block
* No blank line between `} else {` or `} catch {`
* Blank line before a final `return`; no blank line for early-return guard clauses
* Each file should end with a newline

### C++ Specific Guidelines
* Use modern C++23 features where appropriate
* Source files use `.ixx` for module interface units and module partitions. Module names
  mirror the directory structure (e.g., `Dnn.Components.Linear`). Backend specializations
  live in `:Cuda` / `:Cpu` partitions. There are no header guards or `#pragma once` — this
  is a module-based codebase.
* Resolve operations at compile time via `OperationTraits` rather than runtime registries.
  See [Mila/Specifications/OperationDispatch.md](Mila/Specifications/OperationDispatch.md).
* Prefer `nullptr` to `NULL` or `0`
* Use smart pointers instead of raw pointers when possible
* Use `const` whenever applicable
* Avoid using exceptions in performance-critical code paths

### Documentation
* Use Doxygen-style comments for classes, functions, and non-trivial code sections
* Keep file-level Doxygen brief — one to three sentences. Detail belongs on the symbol.
* Comments explain **why** or state a non-obvious contract — never restate what the code does
* Use ASCII only in code comments (no Unicode symbols or emojis)
* Document parameters, return values, exceptions, and any side effects

## Pull Request Process

**Validate locally first. That is the gate, and it is the same one the maintainer uses on his own
commits.** Build the **`x64-validate`** preset and run `ctest`:

```
ctest --test-dir out/build/x64-validate --output-on-failure
```

Without an NVIDIA GPU, run the fullest configuration your hardware allows —
`x64-debug-cpu-only`, or `linux-clang-cpu-release` — and say so in the PR. Cases needing a device
or model weights skip themselves, so a clean run there is a real result.

Then:

1. Ensure your code follows the coding standards and passes all tests
2. Update the documentation, including the README.md if necessary
3. Include relevant tests for your changes:
   - Unit tests for new features
   - Regression tests for bug fixes
   - A `*_MatchesReference` test for a new kernel or operation
4. Title your PR with a concise description of the changes
5. Fill out the PR template, including what you built and what you ran it on
6. Your PR should target the `dev` branch

### What happens to your pull request

1. **CI runs**, in a few minutes to an hour. It is a pre-check, not the review — see below for the
   narrow question it answers.
2. **A maintainer reads the PR**, and will ask about anything unclear before spending a build on it.
3. **The branch is pulled into the maintainer's own environment**, built, and run on real hardware —
   currently Ada (sm_89) and Blackwell (sm_120). For a CUDA change this is where correctness is
   actually established, because nothing else in the process can establish it.
4. **It merges when it meets the bar**, which may take a round or two of changes.

Step 3 is the slow one, and it is a person on real cards rather than a queue. Two things make it
quick: a `*_MatchesReference` test, which turns validation into a build and a `ctest` instead of
reading your kernel and inventing the test; and an accurate account in the PR of what you ran, so
the maintainer knows what is already covered and what is not.

## Testing Requirements

### Unit Tests
* All new features must have corresponding unit tests
* Follow existing test patterns in the codebase (see the `Mila/Tests/Dnn/` directory, which mirrors the `Mila/Src/Dnn/` tree)
* Include tests for both CPU and CUDA implementations where applicable
* Include tests for edge cases and error conditions
* Test numerical stability for floating-point operations

### Test Patterns
For modules, include tests for:
1. Basic functionality (forward pass, parameter counts)
2. Edge cases (minimal dimensions, boundary conditions)
3. Numerical stability (large and small input values)
4. CPU/CUDA equivalence where applicable
5. Training/inference mode behavior

### Running the tests
Tests are off in the default build (lean library + samples). The **`x64-validate`** preset enables
the unit tests plus the packaging gates; see **Pull Request Process** above for the command and for
what to run when you have no GPU.

See [RELEASING.md](RELEASING.md) for the full version, validation, and tagging flow (including
the post-tag CPM release-access smoke test).

### What CI answers, and what it does not

Two workflows run on a PR against `dev`:

* **Mila Gates** — about a minute. Dependency notices, version consistency, and Doxygen (doc
  warnings are errors).
* **Mila CI** — compiles the whole tree under clang-21 on Linux with CUDA enabled, and runs the
  CPU test suite.

**No kernel is executed.** Hosted runners have no GPU, so there is a CUDA *compile* path and a CPU
*test* path and nothing in between. A green tick means your change builds on a machine that is not
yours; it says nothing about numerical correctness on a device.

So CI is not a substitute for validating locally, and it is not the review. It exists for one
narrow reason: clang on Linux rejects things MSVC quietly resolves, and if you have no CUDA
Toolkit it is the only thing that can tell you the CUDA tree still compiles. That is its whole job.

If it fails for something that looks like Mila rather than your change, say so — that is our bug
to fix, not yours.

## Documentation

* Update Doxygen comments in code for API changes
* For significant changes, update the relevant design document under `Mila/Specifications/`
* Document any new dependencies or system requirements

## Issue Reporting

When reporting issues, please include:
1. Description of the issue
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details (OS, compiler version, CUDA version)
6. Any relevant logs or screenshots

## License

By contributing to Mila, you agree that your contributions will be licensed under the project's [MIT License](License.md).

---

Thank you for contributing to Mila!
