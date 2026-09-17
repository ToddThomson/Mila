# Mila — Backlog

**Work committed to the release in flight, and nothing else.** Narrative and success criteria are in
[ROADMAP.md](ROADMAP.md); everything upstream of the commitment is in
[`Mila/Issues/`](Mila/Issues/README.md). Completed work is in the git history.

**Admission:** name the ROADMAP success criterion that fails if this never ships. If you cannot,
it belongs in `Mila/Issues/`.

Each `###` bucket is a v0.20 theme, its name matching the ROADMAP section — the only join.

**Entry shape**, one level deeper than the same shape in `Mila/Issues/`:

```markdown
#### Llama throws away its long-context scaling factor

`open` · `llama`

The load path reads `rope_scaling` from the model metadata and discards it — the
`.withRoPEScalingFactor()` call at `Llama.ixx:703` is commented out, for a reason recorded as
unclear. 3.1 8B cannot reach the context length it advertises.
```

The **heading states the problem** and has to read cold: no term that exists only inside Mila.
The **metadata line** carries status — `open`, `in progress`, `done` — then area tags from
[Tags.md](Mila/Issues/Tags.md); status never appears in the prose. The **body** carries whatever
detail the work needs and ends in an anchor, unless the finding is an absence, in which case say so
rather than inventing a location.

**The gate is the entry count, and it only goes down.** A release in flight burns down, so an
addition is paired with a removal or it is a deliberate admission that scope grew. Past roughly
forty entries this is a wishlist, not a release.

**Done means deleted**, in the same commit as the work — `done` is a working-tree marker and is
never committed.

---

## Current release (v0.20.0)

### Observability

#### `observe()` documents a path pattern that does not work the way it says

`open` · `observability` · `docs`

The Doxygen on the public `CompositeComponent::observe` teaches `"qwen.blk_*"` as "every block, but
not their children", and offers `"qwen.blk_*.*"` for the children (`CompositeComponent.ixx:405-406`).
Both are false — `*` matches dots, so the two patterns select the same set. Measured:
`"*.tf_layer_*"` selected 816 components on a 48-layer Gemma 4 12B. Either `*` stops at a dot or the
examples describe what it actually does; `Observability.md` §11 carries the same claim.

---

### API Documentation

#### `Component` documents a compute contract it does not declare

`open` · `api` · `docs`

`Component.ixx:132-133` and `:728` teach that `forward()` requires `build()` and that `backward()`
requires `isTrainingMode()`. Neither the base class, `CompositeComponent`, nor `Network` declares
those methods, so the prose describes a contract a reader cannot find. Correct it to name the
concrete methods it means — this is the class every component derives from.

#### Nothing checks Doxygen when doc drift is introduced

`open` · `docs` · `ci`

A break from a `Src/**` or `README.md` change is caught only by `publish-site.yml`, which is now
manual — so nothing exercises Doxygen between publishes at all. Seventy-five errors once
accumulated unseen and then blocked the site. A non-deploying check in `build-pipeline.yml` needs
neither CUDA nor CMake.

---

### Packaging & Distribution

#### No Ampere or Turing card has ever run Mila, and the published lists assume one answer

`open` · `binding` · `build`

The published-artifact architecture list is `80;86;89;90;120` on the reasoning that SM 8.0 is the
floor Mila's kernels draw — the FP4 GEMM gates on `major >= 8` (`CudaLinearOp.ixx:661`) and both
GQA flash prefill paths throw below it (`Gqa.Flash.Fa2.cu:513`, `Gqa.Flash.Wmma.cu:632`). That is
what the code is written for; it is not what anyone has observed, because the dev box has only sm_89
and sm_120. A rented A10G or A100 hour would settle whether an RTX 30-series card really runs a
published FP4 model, and whether Turing's non-WMMA fallback (`cuda_fp4a16_gemm`, dispatched at
`CudaLinearOp.ixx:882`) is reachable at all or is dead code behind those throws — every bound model
uses GQA, and the published list starts at 80, so today it compiles for nobody.

#### CUTLASS is fetched on every CUDA build and nothing includes it — decide at the rc.1 tag

`open` · `build` · `ci`

No translation unit in the tree carries a cutlass header; `Mila/CMakeLists.txt:77` wires the
include directory and that is the whole of its use. It arrived in `e75938f3` alongside the WMMA
FP4 GEMM work, which then hand-wrote `CudaW4A16Gemm.Wmma.cu` and never used it. The cost is
**233 MB of the 753 MB `_deps` total**, cloned on every `MILA_HAS_CUDA` configure — every
developer, every CI run, every container and wheel build, and every FetchContent consumer.
`getting-started.md:85` and `Mila/Samples/QuickStart/Cpp/README.md:57` both name it to
first-time readers. It has already shaped the build around itself: `linux-wheel`'s `binaryDir`
is `/build` because its clone step fails across a Windows bind mount, and both the CI cache and
`drive_cpm.cmake` exist partly to avoid re-cloning it.

**Decision 2026-09-12: kept through rc.1, bumped to `v4.8.0dev`, and REMOVED AT THE rc.1 TAG IF
STILL UNUSED.** Keeping it is not inertia — it is the only credible route to the two things
cuBLASLt cannot express: the MoE grouped (Ptr-Array) GEMM, and a block-scaled GEMM that consumes
scale factors natively and so absorbs the `dequantize_fp4_to_fp8` and `apply_per_token_scales`
passes measured at 22-53% of prefill. Both are post-v0.20 and unscheduled, which is exactly why
this is a gate rather than a commitment. Removal touches nine files: the CPM block,
`Mila/CMakeLists.txt:77`, `NOTICE.md`, two `CMakePresets.json` descriptions, the CI cache
comment, `drive_cpm.cmake`, `drive_fetchcontent.cmake`, and the two user-facing docs above.

#### miniz serves an archive stack nothing reaches

`open` · `build` · `architecture`

`CPMAddPackage(NAME miniz GITHUB_REPOSITORY richgel999/miniz GIT_TAG 3.1.2)` at
`CMakeLists.txt:228`. This theme's criterion requires every vendored dependency in a published
binary to be current or pinned with the reason written down, and **removal is what satisfies
it**, at the production release the criterion binds.

The pin arrived incidentally — the tag went from `master` to `3.1.2` while `NOTICE.md` was being
brought back into agreement with the build, so the entry no longer turns on miniz being the one
unpinned dependency. That was the weaker half of the case anyway; the action is unchanged.

**Pinning it first was considered and declined** (Todd, 2026-09-09) as inconsequential, and the
reasoning is worth keeping so it is not re-proposed: unlike an ambient CUDA toolkit, a floating
miniz cannot silently change a shipped artifact. A different revision either fails the build loudly
or yields a serializer no code path reaches. Pinning was never an interim step toward removal and
is not one now — it bought nothing, which is exactly what "inconsequential" predicted.

Remove it for the production release: delete
`ZipSerializer.ixx` (the only `ArchiveSerializer` implementation and the only file naming miniz),
`GptModel::fromCheckpoint` / `saveCheckpoint` (`GptModel.ixx:177`, `:231`), the `Mila.ixx:309`
re-export, and the `PUBLIC` link plus the two `INTERFACE` include directories at
`Mila/CMakeLists.txt:962` and `:971-973` — which is what currently makes every consumer's
`import Mila;` recompile a module that includes `<miniz.h>`.

`ModelArchive` and the 24 public pure-virtual `save_` implementations **stay** and go inert; giving
them a safetensors backend is the Vnext entry "Saving never followed loading to safetensors", and
attempting it here would be a core `Mila/Src` change under the freeze. The cost to accept
deliberately: roughly 46 test usages across 10 files construct a `ZipSerializer` to exercise the
component save/load round trip, and that coverage goes with it until the migration restores it.

#### The Docker Hub Overview page is authored in a browser with no source in the repo

`open` · `docs` · `distribution`

It is what container search shows, and it carries the container-distribution message. Hand-editing
it in the browser is exactly how the HuggingFace organization card came to need a rewrite.
[[project_four_channel_roles]]

#### The container build is not yet reproducible from a clean tree

`in progress` · `build` · `ci`

Validated on a clang-21 + gcc-15 host at CUDA 13.3. Remaining: build against the bind-mounted tree,
and have CI build `FROM` the image rather than apt-installing its dependencies again.

---

### Consumer & Contributor Surface

#### There is no guided reading path through the source

`open` · `docs`

Mila's positioning is the stack you can read, and nothing shows a reader where to start. One token's
journey — embed, attend, sample, decode — through the real source, followable by a strong C++
developer unaided. No anchor: the finding is an absence.

#### A consumer's path budget is about thirty characters, spent by one seven-level include

`open` · `build`

`ElementwiseActivation.cu:21` includes an 86-character `../../../../../../../` path, and MSVC applies
MAX_PATH to the unresolved string, so a FetchContent consumer with a source root deeper than about 97
characters gets C1083 on a header that exists. It tipped the CPM gate over at beta.3 (264 against
260). Two other five-plus-level relative includes exist in `Mila/Src`; an include directory and
short includes close all three.

---

### Model Distribution

#### The published model cards tell users to run `/install` and `/models`, which Chat does not have

`open` · `distribution` · `docs`

Chat's commands are `/model install <name>`, `/model load <name>` and `/model list`. The card
sources in the repository are correct; the live copies on huggingface.co only change when a model is
re-published, and they are what a new user reads *before* they have Mila at all. Fold the card
refresh into the next publish of each: `Mila/Tools/ExportArtifact/ModelCards/`.

#### A mistyped model name is reported as an authentication failure

`open` · `distribution`

`HuggingFaceHub.ixx:283` maps every 401 to "no valid HuggingFace token", and HuggingFace hides
repository existence from strangers. So an authenticated caller gets a 404 and the right message,
while a new user gets sent to obtain a token they never needed — and a typo is the likeliest failure
on the evaluation path. Invisible to anyone who has run `huggingface-cli login`. When no token was
sent and the owner is `mila-llm`, lead with the name being wrong.

#### The getting-started walkthrough ends in a download and no conversation

`open` · `distribution` · `docs`

It names `gpt2-small`, which installs and then cannot be used from Chat: Chat refuses base models by
design, and `/model list` says so in the row — but only *after* 623 MB has transferred. Either the
getting-started paths name an instruct model, or `/model install` says so before the transfer starts.

#### `gemma-4-12b-it-fp4` has two manifests and they no longer match

`open` · `gemma` · `distribution`

The package directory carries the current one; `ModelCards/gemma-4-12b-it-fp4/mila.json` is the
pre-package copy. Two sources of truth for a published model, and publishing from the stale one is a
live risk. One has to go, and the card directory's `publish.json` flow goes with it.

#### Nothing checks that published wheels and images carry the notices they owe

`open` · `docs` · `distribution`

`NOTICE.md:39` no longer pretends Mila ships only as source, and its table is gated, but whether each
published file actually carries the notices is untested: the wheels bundle nlohmann/json, miniz,
CUTLASS and pybind11, and the container images add curl. Both wheel presets are now
`MILA_ENABLE_LIBCURL=OFF`, so a wheel built today has no curl — establish whether the published
`0.20.0b3` wheels predate that change, and whether the wheel and image builds embed `NOTICE.md` at
all. The licence texts are read at source, not from `NOTICE.md`.

---

### Product Family — Adaptor Validation

#### Gemma loses its own reasoning between tool calls in a turn

`open` · `gemma` · `adaptors`

Google's multi-turn rule is to strip thoughts from *prior* turns and keep the current turn's.
`extractAnswer` (`Gemma.Protocol.ixx:1288`) removes every channel span from a response rather than a
leading run, so a model working through a multi-step tool sequence starts each step without the
reasoning that led to it.

#### MIS tool calling is not yet validated across the full set of Gemma 4 flows

`in progress` · `gemma` · `adaptors`

Codex and Claude Code CLI round-trips are live, and the native grammar is reconciled to Google's
canonical template and pinned by an oracle. Three gaps remain: N sequential distinct tool calls
within one turn, channel-content parser polish, and Codex-CLI re-validation against the reconciled
grammar rather than the one it was first driven on.

#### Qwen refuses prompt-prefix reuse and never says so

`open` · `qwen` · `adaptors`

`QwenDeltaNetBlock::rewindKvCache` always returns false — correctly, since a recurrent state is a
lossy summary and cannot be rewound — and `QwenTransformer::rewindKvCache` ANDs that into a refusal
for the whole stack. A server that reuses prefixes has to read this as a property of the model and
plan around it, not discover it as a failed retry. Chat is exempt: it re-prefills every turn. The
per-block mechanism exists (`snapshotState`/`restoreState`); a whole-model policy does not.

#### `mila serve <args>` loses every argument on Windows

`open` · `adaptors` · `build`

`runProgram` (`Cli.ixx:100`) hands a concatenated string to `std::system`, so `cmd.exe` strips the
outer quotes of the whole command line and nothing survives; the code returned is the shell's rather
than the server's. Launch with an argument vector — `CreateProcessW` or `posix_spawn` — behind a
CMake-selected module partition, since module code carries no `#ifdef`.
