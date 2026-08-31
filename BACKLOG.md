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

### Models

#### Llama throws away its long-context scaling factor

`open` · `llama`

The load path reads `rope_scaling` from the model metadata and discards it — the
`.withRoPEScalingFactor()` call at `Llama.ixx:703` is commented out, for a reason recorded as
unclear. 3.1 8B is a shipped model that cannot reach the context length it advertises.

#### Tool calling has never been run end-to-end on Llama

`open` · `llama` · `adaptors`

Gemma 4 is validated through both adaptors. Llama 3.2 3B and 3.1 8B are named in this release's
success criteria and in the product definition, and neither has been exercised — so the claim is
made on two models nobody has driven through a tool call.

#### GPT-2 and Llama 3 tokenize by approximation on every build and platform

`open` · `tokenizer` · `gpt` · `llama`

`\p{L}` and `\p{N}` (`BpePreTokenizationMode.ixx:33`, `:57`) compile in no standard `std::regex`, so
`BpeTokenizer.ixx:344` throws and silently falls back to an ASCII scanner — in every build, the
published Linux container included. No parity test catches it, which raises a second question worth
settling in the same pass: whether the fixtures are English-only, because if they are, the site's
tokenizer parity claim is untested. The fix is PCRE2/RE2 or a hand-written Unicode scanner.

#### The Qwen oracle is gated on an agreement no correct implementation can reach

`open` · `qwen` · `docs`

`Qwen3.8.md` §8 asks the 16 GiB oracle to agree token-for-token across architectures. BF16, FP8 and
FP4 all fork between Ada and Blackwell, at a token index set by the prompt rather than by the
precision, while each card stays deterministic run to run — floating-point non-associativity, not a
defect. The release's Qwen quality claim rests on this gate, so restate it as teacher-forced;
perplexity never samples.

---

### Observability

#### `observe()` documents a path pattern that does not work the way it says

`open` · `observability` · `docs`

The Doxygen on the public `CompositeComponent::observe` teaches `"qwen.blk_*"` as "every block, but
not their children", and offers `"qwen.blk_*.*"` for the children (`CompositeComponent.ixx:405-406`).
Both are false — `*` matches dots, so the two patterns select the same set. Measured:
`"*.tf_layer_*"` selected 816 components on a 48-layer Gemma 4 12B. Either `*` stops at a dot or the
examples describe what it actually does; `Observability.md` §11 carries the same claim.

---

### Test Suite Revival

#### The authored component, tensor and tokenizer suites are not yet green on the current API

`in progress` · `architecture`

The suite was largely commented out during the inference-era refactors and is being re-aligned
rather than rewritten. Concrete component classes are re-enabled and build-green.
`SoftmaxCrossEntropy` is parked until loss moves onto the device, and three backward-numeric cases
are skipped pending the kernel defects below.

#### The quantization and Llama inference paths were built during the test drought and have no coverage

`in progress` · `quantization` · `llama`

`OperationTraits` dispatch is now covered. What remains is the load-time quantization white-box —
`PerChannelFp8`, `PerGroupFp4` and the decode matvec kernels, which are the one legitimate op-layer
test because they cannot be reached through the public component — and the Llama path.

#### Llama silently runs past its context limit and no test says so

`open` · `llama`

`LlamaModel.ixx:336` carries the guard and nothing exercises it. Where GPT-2 crashes, Llama walks
off the end of the KV cache instead, so nobody reports it and absence of reports is not evidence.
The template is `Tests/Dnn/Models/GptModel.Cuda.cpp`: a weightless checkpoint at a small deployment
context.

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

#### PyPI advertises a Linux wheel that does not exist

`open` · `binding` · `ci`

`pyproject.toml:37` declares `POSIX :: Linux` while the only published file is `win_amd64`, and
release metadata is immutable once uploaded. Linux is clean-room proven under `python:3.13-slim`;
Windows has never had a clean-room run and cannot get one locally, since Windows 11 Home has neither
Containers nor Hyper-V. Both resolve only through a release cycle, so the matrix needs
`wheel-cleanroom.yml` running on `master`.

#### The published wheel stops one GPU generation short

`open` · `binding` · `build`

The `x64-wheel` and `linux-wheel` presets pin `75;80;86;89;90` (`CMakePresets.json:183-184`,
`:214-215`) while the library default already carries `120` (`Mila/CMakeLists.txt:24`). A
`pip install mila-llm` on an RTX 50-series card therefore JITs from sm_90 PTX at first launch.
Adding `120` costs one more CUDA compile per wheel; the alternative is saying so on the PyPI page.

#### The Docker runtime image has never had a publish build

`in progress` · `build` · `ci`

The image builds and all three entrypoint verbs are verified in a container: `install` pulled into a
fresh volume, `chat` listed that store, and `serve` bound 6452 and answered a real
`/v1/chat/completions` from a read-only mount of the host store. What has never been built is a
*publishable* image — verification used single-arch `89`, where a published one needs `89;90;120`
and `MILA_CLEAN_BUILD=1`, since `--no-cache` leaves BuildKit cache mounts intact and has already
produced two silently wrong images in one day. The website's devel cost figures come from that
build, via `docker manifest inspect` and `docker images`.

#### Every container build path defaults to an architecture a published image cannot use

`open` · `build` · `ci`

`Docker/build-chat.sh:25` defaults `MILA_CUDA_ARCH=native` and passes it to both
`CMAKE_CUDA_ARCHITECTURES` and `MILA_LIBRARY_CUDA_ARCHITECTURES`, so an image carries kernels only
for the GPU that happened to build it — and `native` does not resolve at all on the GPU-less builder
a publish runs on. The publish pipeline has to set the portable list explicitly.

#### The runtime image ships a binding that cannot import, and the gate calls it fine

`in progress` · `build` · `binding`

`site-packages/mila/` holds only `__init__.py`, so `install` and `serve` both die on
`ImportError: No module named 'mila._mila'`. The extension reaches the image only as a `POST_BUILD`
side-effect into the source tree, which a cache-warm compile never re-runs. Install from
`/build/python/mila`, where the build actually writes it.

#### The shared-library gate passes when the file it checks is missing

`open` · `build` · `ci`

`Dockerfile.runtime`'s runtime stage greps `ldd` output for `"not found"`. An unmatched glob makes
the shell hand `ldd` a literal pattern, `ldd` answers `"No such file or directory"`, and the grep
finds nothing — so the gate printed "Shared library check passed" over a missing extension. Assert
the file exists first, then check its NEEDED entries.

#### `build-mis.sh` installs a package the container's Python is too new for

`open` · `build` · `binding`

`Docker/build-mis.sh:76` runs `pip install --no-deps -e Mila/Bindings/Package` under Python 3.14,
and `mila-llm`'s `requires-python` is `>=3.12,<3.14`; `--no-deps` does not suppress that check. The
script's own comment shows the ceiling was handled for the server dependencies and missed for the
package. Verify in a container, then add `--ignore-requires-python` as the runtime image already
does.

#### The container tag scheme is undecided, including whether a pre-release gets `latest`

`open` · `ci` · `docs`

RELEASING covers dropping `+build` from the version (OCI forbids `+`) and nothing else. `latest` is
what a bare `docker run toddthomson/mila-llm` resolves to, so pointing it at a beta makes the beta
the default for everyone who does not read the tag list. The repository name is settled:
`toddthomson/mila-llm`.

#### The Docker Hub Overview page is authored in a browser with no source in the repo

`open` · `docs` · `distribution`

It is what container search shows, and it carries the container-distribution message. Hand-editing
it in the browser is exactly how the HuggingFace organization card came to need a rewrite.
[[project_four_channel_roles]]

#### curl is pinned three years behind in what the container ships

`open` · `build`

`CMakeLists.txt:266` holds 8.11.1 under a `REVIEW:` marker naming 8.21 as current. A vendored
TLS-adjacent dependency is the one pin where staleness carries a security cost rather than a
maintenance one, and the container links it even though both wheel presets no longer do. Bump it, or
record why 8.11.1 stands. Settle **after** the `NOTICE.md` entry below, which establishes where curl
actually ships.

#### The container build is not yet reproducible from a clean tree

`in progress` · `build` · `ci`

Validated on a clang-21 + gcc-15 host at CUDA 13.3. Remaining: build against the bind-mounted tree,
and have CI build `FROM` the image rather than apt-installing its dependencies again.

---

### Consumer & Contributor Surface

#### `import Mila;` breaks the standard library in the consumer's translation unit

`open` · `api` · `build`

FetchContent is the one supported C++ consumption path, and it works only with workarounds. Three
failures show up in a real consumer and vanish without the import: stream **input** fails on an
undefined `basic_istream::sentry`; instantiating a model needs `<sstream>` included **before** the
import, because virtual `Component::toString()` compiles in through the vtable; and putting the
import before the includes is fatal (C1116). `Samples/QuickStart/Cpp/main.cpp` carries two of these
workarounds today. [[project_import_mila_breaks_std]]

#### Linux/clang is not yet a first-class platform

`in progress` · `build` · `ci`

WSL is green, CI compiles under clang-21, and the container builds and runs Gemma 4 FP4. The GCC 16
second compiler oracle and the broadened compiler matrix move to Future rather than blocking this.

#### A missing dispatch specialization still reads as a cascade in places

`in progress` · `architecture` · `api`

A missing `(Op, Device, Precision)` combination should read as one sentence, not a wall of
constraint failures. The core landed; the optional named kernel concepts and the
`OperationDispatch.md` §12 reconcile remain.

#### There is no guided reading path through the source

`open` · `docs`

Mila's positioning is the stack you can read, and nothing shows a reader where to start. One token's
journey — embed, attend, sample, decode — through the real source, followable by a strong C++
developer unaided. No anchor: the finding is an absence.

#### The QuickStart Python samples still call weights an "artifact"

`in progress` · `docs` · `binding`

A user has a model, Mila has that model's weights, CI has artifacts. The ten model cards, Chat, the
pybind layer and MIS are converted; QuickStart Python is the published surface still outstanding —
the website's Get Started tabs link straight to it. The maintainer docs and the `Mila/Src` tail are
deferred. Must **not** change: `tool_bridge.py:84` and `:455`.
[[project_artifact_vocabulary_rule]]

---

### Model Distribution

#### The published model cards still tell users to run `/install`

`open` · `distribution` · `docs`

The card sources in the repository are correct; the live copies on huggingface.co only change when a
model is re-published. Those copies are what a new user reads *before* they have Mila at all, so the
first instruction they follow is the wrong one. Fold the card refresh into the next publish.

#### `--instruct` is missing from the packaging tool's option list, and its absence is silent

`open` · `distribution`

The flag is parsed (`ExportArtifact.cpp:142`) but absent from the `--package` option list
(`:42-56`), so leaving it off writes `instruct: false` into the manifest with no warning — which
changes the prompt template every consumer of that model applies. Document it, and consider refusing
a model whose name says instruct while its manifest says otherwise.

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
design, and `/models` says so in the row — but only *after* 623 MB has transferred. Either the
getting-started paths name an instruct model, or `/install` says so before the transfer starts.

#### `gemma-4-12b-it-fp4` has two manifests and they no longer match

`open` · `gemma` · `distribution`

The package directory carries the current one; `ModelCards/gemma-4-12b-it-fp4/mila.json` is the
pre-package copy. Two sources of truth for the flagship model, and publishing from the stale one is a
live risk. One has to go, and the card directory's `publish.json` flow goes with it.

#### `NOTICE.md` omits curl, and may no longer need to

`open` · `docs` · `distribution`

The note at `:33` treats notice-carrying as open for "a binary distribution that links them", but
both wheel presets are now `MILA_ENABLE_LIBCURL=OFF`, so a wheel built today contains no curl at all
— while the container still does. Establish whether the published wheel predates that change; the
answer decides whether this is an obligation or a non-issue. The note also points at a bucket that no
longer exists, which needs fixing either way.

#### The README promises FP8 and BF16 deployments nobody can reach

`open` · `docs` · `distribution`

`applyRequestedQuantization` refuses to reload pre-quantized weights as anything else, so after the
FP4-only publishing decision every published model is FP4 at runtime. The FP8 rows at
`README.md:163,165` are converter-only capabilities presented as deployment options. Say so, or the
table describes a path that does not exist.

#### `prune()` deletes every model on a store that predates records

`open` · `distribution`

Every pre-record blob is by definition unreferenced, so the first sweep on an upgraded store reclaims
all of it — 6.33 GB in the case observed. A documented command destroying a user's models is not a
sweep, it is data loss. Blobs-with-zero-records is a recognizable state and should be reported rather
than silently collected.

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
