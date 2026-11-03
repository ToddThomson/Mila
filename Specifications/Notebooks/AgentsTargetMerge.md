# Merging MilaAgents into Mila: measurements

Evidence for whether `Mila::Agents` needs a build target and an import of its own, against `Direction.md` decision 2,
which gives compile cost and Clang's source-location ceiling as the reasons and records no measurement of either.
The measurements themselves changed nothing in the repository's build; Todd then agreed the merge, the same day (`Direction.md`
decision 2). Scripts and logs: `D:\Claude\agents-merge`.

## 2026-11-03: MSVC

Gates written before the runs (`GATES.md` there). Trees were `git archive` of `+114`, in scratch. MSVC 14.51, Release, five
CUDA architectures, MXC on, on an i5-13600K with nothing else running.

- **base:** today, `Mila` and `MilaAgents`.
- **merged:** one target, with `MilaAgents` an interface name for `Mila`, and `import Mila.Agents;` still apart.
- **folded:** merged, with `export import Mila.Agents;` in `Mila.ixx`, so `import Mila;` brings agents.

| Question | base | merged / folded | Reading |
|---|---|---|---|
| Builds, no warnings: library, `mila-chat`, `mila-session`, a models-only program (the QuickStart's `main.cpp`) | yes | yes | Passes |
| Clean build of those, wall clock | 131 s, 133 s | 140 s | One merged run; the target alone below is the better figure |
| Clean build of the `Mila` target alone, wall clock, two rounds each | 93.7 s, 93.3 s | 98.5 s, 98.3 s | **About 5 s, or 5%**, for a program that uses no agents |
| Agents modules, summed compile time (38 objects) | 56 s | 68 s | From one log each, under different parallel load |
| Models-only `.exe` | 38,997,504 | 38,998,528 in both | **1 KB.** Folding pulls in nothing through module initializers |
| `mila-chat.exe` | 41,552,896 | 41,542,144 | 11 KB smaller |
| `mila-session.exe` | 41,713,152 | 41,701,888 | 11 KB smaller |
| `Mila.lib` | 165.8 MB, with `MilaAgents.lib` 41.7 MB | 206.6 MB | The sum |
| Compile `import Mila;` and an empty `main`, five runs, median | 0.08 s, 0.02 GiB | 0.08 s, 0.02 GiB | No cost |
| Compile the QuickStart's `main.cpp` | 9.51 s (9.48-9.69), 0.42 GiB | 9.53 s (9.47-9.68), 0.42 GiB | **No cost**, within the spread |
| Compile `mila-session`'s `main.cpp` | 2.50 s, 0.26 GiB | 2.49 s, 0.26 GiB | No cost |

The importer rows compare base with folded, not base plus an added `import Mila.Agents;` as the gates first said, since
folded is the thing being decided.

**Libcurl off and the Python binding on, as the wheels build, folded:** configures, builds with no warnings, and links
`_mila.cp314-win_amd64.pyd` at 41.0 MB. Its size against a binding with agents off was not measured.

**What merging removes:** a `Dnn` module importing an `Agents` module compiles once they share a target
(`Dnn.PromptPrefixReuse` importing `Agents.ToolNeeds`, exit 0, the edit checked). So the rule needs a check to replace the
target. An import that closed a cycle would still fail.

**One defect the first merged run found:** `/bigobj` public on `Mila` reaches `nvcc` on the `.cu` files, which fails. It has to
be held to C++ (`$<$<COMPILE_LANGUAGE:CXX>:/bigobj>`).

**Not measured on MSVC:** `x64-profile` (RelWithDebInfo sections, C1128); the full suite; Docker images; a wheel's
packaging.

## 2026-11-03: Clang

Clang 22 in `mila-llm:local-devel` (Ubuntu 26.04; `mila-dev:latest` has only Clang 21), one CUDA architecture, `-j 4`, MXC off.
Today's tree built clean (178 s for the library, `mila-chat` and the models-only program), then the same build merged and
folded in place. Source locations are Clang's own `-print-stats`, against the 31-bit limit of 2,147,483,648 bytes. Each compile
ran once.

| Translation unit | Today, loaded + local | Folded, loaded + local | Share of the limit, folded |
|---|---|---|---|
| Empty, `import Mila;` | 463,313,703 | 464,292,458 | 21.6% |
| Models-only `main.cpp` | 469,638,473 | 470,617,228 | 21.9% |
| `mila-chat` `main.cpp` | 474,043,465 | 474,044,395 | 22.1% |
| `Chat.Session.ixx`, which imports both | 464,966,831 | 464,967,761 | 21.7% |

- **Agents adds 978,755 bytes of source locations to `import Mila;`, 0.05% of the limit.** The ceiling is not a reason to keep
  the import apart. `import std` is what opened the headroom: the same importer had reached 99% before it.
- **The method checks out:** an empty unit importing `Mila` and `Mila.Agents` on today's tree loads 464,270,363 bytes, and
  `import Mila;` alone on the folded tree 464,271,293, 930 bytes apart. That unit's compile exited 1 in both stages, after the
  statistics printed, with a module unit's command used on a plain source; the cause was not read. `Chat.Session.ixx`
  imports both for real and exits 0.
- **Compile time:** models-only `main.cpp` 25.1 s and 24.9 s; `Chat.Session.ixx` 41.5 s and 41.0 s. Single runs: no difference
  shown.
- **Models-only executable:** 15,276,504 and 15,295,368 bytes, 18,864 bytes or 0.12% larger, within the 1% gate. Unlike MSVC it
  did grow; why was not read.
- **Builds, on `Agents.GrantFileSystem.Posix` and the absent container:** every target, with the same 11 warnings in both
  stages, none of them new.

`mila-session` is Windows only, so it was not built on Clang.

**Reading, with MSVC:** neither reason decision 2 names holds. The separate target and import cost nothing to give up beyond
about 5 s on a clean library build.

**Not measured here either:** a wheel's manylinux link.

## 2026-11-03: the change itself

The merge in the working tree, with the `Mila.Agents` aggregate deleted and its modules listed in `Mila.ixx`.

| Gate | Result |
|---|---|
| `x64-claude-verify`, all targets, MSVC | Builds, 0 warnings, 377 s |
| `MilaTests` with MXC's library | 2435 pass, 12 skip, 7 fail: the 4 of `BACKLOG.md` and the 3 Llama date failures of `Untriaged.md` |
| `AgentsHtmlTextTests`, `ChatRichTextTests`, `DesktopSessionTests` | 10, 33 and 14 pass |
| `check_layers.py` against three deliberate breaks: `Dnn` importing `Agents`, `Distribution` importing `Dnn`, an unlisted folder | All three fail it; the tree as it stands passes |
| CUDA off, MSVC, `x64-debug-cpu-only`: library and every test target | Builds, 0 warnings. `MilaTests` 1279 pass, 15 skip, the same 3 date failures; the other three binaries pass |
| CUDA off, `mila-store` | Fails, and fails the same on a clean `+114`: `Untriaged.md` |
| FetchContent consumer of the working tree | Builds, 202 s |
| Clang 22, five architectures: library, `mila-chat`, `MilaTests`, the two standalone test binaries | Builds, 603 s. No warning from agents code or its tests. Agents tests that need no model 71 pass, `AgentsHtmlTextTests` 10, `ChatRichTextTests` 33 |

**Not run:** `x64-profile` (RelWithDebInfo sections), a second GPU, a wheel's packaging, the Docker images, the Mila Desktop drives.
