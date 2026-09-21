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
| `Cmake/CPM.cmake` | [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) — Lars Melchior and contributors | MIT — full text in the file's own header | No |
| `Mila/Adaptors/Inference/Server/tests/reference/gemma4_12b_chat_template.jinja` | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) — retrieved 2026-07-16, SHA-256 `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4` | Distributed under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) | No |

The Gemma chat template is **test-only**: it is the oracle for Mila's Gemma 4 prompt construction
(`Mila/Adaptors/Inference/Server/tests/test_reference_parity.py`, which documents why it exists and
how to refresh it). The Mila Inference Server builds prompts natively and never runs Jinja at
serving time.

## Dependencies fetched at build time

Not present in this repository — CPM and FetchContent clone them at configure time, so each arrives
with its own license attached. Listed here so this file answers "what does Mila depend on, and under
what terms?" in one place.

| Dependency | Version | License | When |
|---|---|---|---|
| [nlohmann/json](https://github.com/nlohmann/json) | 3.12.0 | MIT | Always |
| [miniz](https://github.com/richgel999/miniz) | 3.1.2 | MIT | Always |
| [curl](https://github.com/curl/curl) | curl-8_22_0 | [curl](https://curl.se/docs/copyright.html) (SPDX `curl`) | `MILA_ENABLE_LIBCURL` |
| [pybind11](https://github.com/pybind/pybind11) | v3.1.0 | BSD-3-Clause | `MILA_ENABLE_PYTHON_BINDINGS` |
| [googletest](https://github.com/google/googletest) | v1.17.0 | BSD-3-Clause | Tests |

This table is checked mechanically against the build — `scripts/dependencies/check_pins.py
--verify-notice` fails if a pin and its row disagree, or if a fetched dependency has no row.
Change a pin and this table in the same commit.

Licenses are as declared by each project; a build fetches each from its own repository rather than
from this one.

Mila is no longer distributed only as source. The `mila-llm` wheels on PyPI and the container images
on Docker Hub are **binary distributions that link this material** — the wheels carry nlohmann/json,
miniz and pybind11, and the container images additionally carry curl. Each carries the
licence texts itself, copied from the fetched sources at their pinned versions: a wheel in its
`.dist-info/licenses/`, an image in `/usr/share/doc/mila/`. A wheel build refuses to run without them
(`Mila/Bindings/Package/setup.py`), and `scripts/dockerhub/verify-image.sh` checks the image. The
`0.20.0b3` wheels and the images published alongside them predate this and carry only Mila's own
licence.
