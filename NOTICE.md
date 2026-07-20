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
| [miniz](https://github.com/richgel999/miniz) | `master` | MIT | Always |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | v4.5.1 | BSD-3-Clause | CUDA builds |
| [pybind11](https://github.com/pybind/pybind11) | v3.0.4 | BSD-3-Clause | `MILA_ENABLE_PYTHON_BINDINGS` |
| [googletest](https://github.com/google/googletest) | v1.17.0 | BSD-3-Clause | Tests |

Licenses are as declared by each project. Mila is distributed as source, so these arrive from their
own repositories rather than from this one. **A binary distribution that links them would need to
carry their notices** — that decision is open (see BACKLOG, *Project Hygiene & Contributor
Readiness*).
