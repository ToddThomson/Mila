# Notice — third-party material

Mila is MIT licensed — see [License.md](License.md). This file records everything else: material in
this repository that Mila did not write, and the third-party code the build fetches.

**This is the only place Mila records third-party licensing.** [ATTRIBUTIONS.md](ATTRIBUTIONS.md) is
a separate document about intellectual debt — the research and ideas that shaped the implementation —
and carries no licensing meaning.

## Vendored files

Files in this repository that are not ours. They are kept **unmodified**; if one is ever changed, say
so in the Modified column and in the file itself.

| Path | Origin | License | Modified |
|---|---|---|---|
| `Lib/Deps/cpm/CPM.cmake` | [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) — Lars Melchior and contributors | MIT — full text in the file's own header | No |
| `Apps/Server/tests/reference/gemma4_12b_chat_template.jinja` | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) — retrieved 2026-07-16, SHA-256 `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4` | Distributed under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) | No |

The Gemma chat template is **test-only**: it is the oracle for Mila's Gemma 4 prompt construction
(`Apps/Server/tests/test_reference_parity.py`, which documents why it exists and
how to refresh it). The Mila Inference Server builds prompts natively and never runs Jinja at
serving time.

## Dependencies fetched at build time

Not present in this repository — CPM and FetchContent clone them at configure time, so each arrives
with its own license attached. Listed here so this file answers "what does Mila depend on, and under
what terms?" in one place.

| Dependency | Version | License | When | In Mila Assistant |
|---|---|---|---|---|
| [nlohmann/json](https://github.com/nlohmann/json) | 3.12.0 | MIT | Always | Windows and macOS |
| [miniz](https://github.com/richgel999/miniz) | 3.1.2 | MIT | Always | Windows and macOS |
| [curl](https://github.com/curl/curl) | curl-8_22_0 | [curl](https://curl.se/docs/copyright.html) (SPDX `curl`) | `MILA_ENABLE_NETWORKING`, not on macOS | Windows |
| [pybind11](https://github.com/pybind/pybind11) | v3.1.0 | BSD-3-Clause | `MILA_ENABLE_PYTHON_BINDINGS` | No |
| [metal-cpp](https://github.com/apple/metal-cpp) | release/metal-cpp_macOS27.0.1_iOS27.0.1 | Apache-2.0 | `MILA_ENABLE_METAL` | macOS |
| [googletest](https://github.com/google/googletest) | v1.17.0 | BSD-3-Clause | Tests | No |

This table is checked mechanically against the build — `Publishing/dependencies/check_pins.py
--verify-notice` fails if a pin and its row disagree, or if a fetched dependency has no row.
Change a pin and this table in the same commit.

Licenses are as declared by each project; a build fetches each from its own repository rather than
from this one.

Mila is no longer distributed only as source. The `mila-llm` wheels on PyPI and the container images
on Docker Hub are **binary distributions that link this material** — the wheels carry nlohmann/json,
miniz and pybind11, and the container images additionally carry curl. Each carries the
licence texts itself, copied from the fetched sources at their pinned versions: a wheel in its
`.dist-info/licenses/`, an image in `/usr/share/doc/mila/`. A wheel build refuses to run without them
(`Lib/Bindings/Python/Package/setup.py`), and `Publishing/dockerhub/verify-image.sh` checks the image. The
`0.20.0b3` wheels and the images published alongside them predate this and carry only Mila's own
licence.

## What Mila Assistant carries

Mila Assistant's installer is a binary distribution too. Its session carries Mila's own licence beside the
program, and in `Licences/<name>/` the text of each dependency marked in the table above (`Licences/nlohmann/json/`)
and of what it carries beside them:

| Name | What | Platform | Terms | Text in `Licences/` |
|---|---|---|---|---|
| `CUDA` | NVIDIA's cuBLASLt and cuRAND libraries | Windows | NVIDIA CUDA Toolkit EULA; both are distributable under its Attachment A | `CUDA/EULA.txt` |
| `LLVM` | LLVM's libc++, libc++abi and libunwind, linked into the session | macOS | Apache-2.0 with LLVM Exceptions | `LLVM/LICENSE.TXT` |
| `MXC` | Microsoft's MXC library, which runs commands in a container | Windows | MIT | `MXC/LICENSE.md` |
| Visual C++ runtime | Microsoft's runtime libraries, beside the session from Visual Studio's redistributable folder | Windows | Visual Studio's licence terms, *Distributable Code* | None; Visual Studio gives no text with the files |
| Electron and Chromium | The window | Windows and macOS | MIT; Chromium's are many | Not in `Licences/`: `LICENSE.electron.txt` and `LICENSES.chromium.html` at the app's root on Windows and Linux, where electron-builder leaves them, and in its `Resources` on a Mac, where the package puts them back after electron-builder removes them |

The session's install refuses to run without each of those texts, and the window's packaging refuses a session staged
without them, and a package without Electron's and Chromium's. The window lists every one under *Licences* in *About
Mila Assistant*, with Mila's own. `check_pins.py --verify-notice` holds both tables to the texts the session's install names
(`Apps/Assistant/Session/CMakeLists.txt`, `mila_session_licence`).
