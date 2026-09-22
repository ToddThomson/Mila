# Vnext

**The seed corpus for the next release's backlog.** When the release in flight goes to production,
this is what `BACKLOG.md` is rewritten from, so an item here carries a real intention to do it in
that cycle. That is the difference from [`Future.md`](Future.md), which carries zero commitment:
"we mean to do this next" belongs here, "someday, if the hardware or the reason arrives" belongs
there.

A shortlist, not a plan — tasking happens on promotion, when the next release has a ROADMAP section
and its items can face the real admission test. Triage flow and categories are in
[README.md](README.md); the tag set is [Tags.md](Tags.md).

---

## Warnings-as-errors ratchet

`build` · `ci` · `blocked`

Enforce in **CI only**, never locally; ratchet on the count *not increasing* before demanding zero;
**MSVC first**, since `/WX` across three compilers means the union of three opinions must be zero.
Dormant-but-retained code warns by nature — suppress per-file in CMake pointing at the owning task,
never with `#pragma warning` in module code.

Blocked on isolating third-party warnings first: `/external:I` + `/external:W0` (`-isystem` for
Clang/GCC), targeting third-party header text pulled into Mila's own TUs rather than their sources
— `/W4` at `Mila/CMakeLists.txt:87` is `PRIVATE` and never reached them. Two frictions: those
headers enter through module global module fragments, and `/external:` does nothing for nvcc
diagnostics.

`GroupedQueryAttention.ixx:216`'s C4702 is the case that decides the shape. It is left deliberately —
it self-clears when the GQA training path is built, where a suppression would have to be remembered.
A blanket `/WX` forces it silent; escalating only the defect-class codes leaves it visible.

**Only `RelWithDebInfo` reports C4702, and no watched preset is `RelWithDebInfo`.** Eleven
`unreachable code` warnings appeared there and in no other build type on the identical tree, and only
with testing, samples, adaptors, tools and profiling enabled. Ten were the `copyFromBlob`
fall-through fixed at `+42`; the eleventh was `GroupedQueryAttention::backward`, resolved at `+43` by
refusing at the component boundary — unverified whether MSVC now deduces never-returns through
`attn_->backward` at `Llama.Block.ixx:373` instead. `x64-validate` is Release, so the ratchet has to
decide which build type it watches.

## v0.20 library-frozen tails

`api` · `mila-src`

The Generation API surface tail (`SamplerConfig` rename, Llama/Gpt seedable sampling, eager
sampler, config-accessor propagation, `contextLength()` hoist), the Sample-API device-sampler
migration for Llama/Gpt, and the Optimizer-dispatch migration onto `OperationTraits`.

All `Mila/Src` capability deferred out of v0.20 by the freeze rather than declined, so the release
that lifts the freeze is where they land.

## A component can declare one workspace and allocate another

`architecture` · `mila-src`

`BuildContext::withInstalledOutput` (`Component.BuildContext.ixx:208`) promises that what a
component says it needs is what it goes on to allocate, and nothing checks that it is. The
workspace factories describe the set of slots and allocate them in the same act, so the two can
only be kept in step by hand: the rule deciding which slots are pooled is written out three times
in `Qwen.ixx` (`:382`, `:602`, `:626`), and the ~6.5 GiB the DeltaNet layers were under-reported by
was one of those sites existing while another did not.

The proposed fix — bind the slots unallocated before `build()`, materialize them inside it — is
written up in `MemoryFootprint.md` §4.5. It changes a core component base class, which is why it
waits.

## Footprint predictions are only checked at whole-model level

`models` · `blocked`

A model's predicted footprint is compared against what it actually built; the blocks and components
underneath it are not, so an error inside one is only caught if it happens to be large enough to
show in the model total. Two cases are owed and neither can be written today.

**Gemma**, per block kind — the local-attention and global-attention blocks share one
max-geometry workspace, and `Gemma.Block.Cuda.cpp` never calls `getRequiredMemory`. Blocked on
Gemma building its workspace inside the private `GemmaTransformer::allocateBlockWorkspace`
(`Gemma.ixx:1110`), so a test has no way to construct one; it needs an exported factory.

**`Rope`**, at leaf level — and it cannot be a plain predicted-equals-built assertion.
`RopeCacheRegistry` keys on (theta, built sequence length, head_dim) and only the first component to ask for
a given key allocates, so the answer depends on the order the test builds things in. Deduplication
at transformer level is already in place.

## GPT-2 cannot report what it would allocate

`gpt` · `mila-src`

Every other family answers "will this fit on my card" before loading anything; GPT-2 does not
implement `getRequiredMemory`, so installing `gpt2-small` and asking Chat about it gets silence.
Eight of its components still throw the base class's by-design "not implemented" error — Gelu,
MultiHeadAttention, Lpe, MLP, SoftmaxCrossEntropy, LayerNorm, Softmax, GptBlock — and the
contract has been landing one family at a time (`Core/Component.ixx:615`), with GPT-2 the one left.

Its footprint is the simplest of the four: no quantization policy, no sliding-window ring, and
learned positional embeddings sized exactly to `context_length`.

## Footprint predictions have never been checked on an unquantized load

`models`

Both footprint suites predict what a model will allocate, load it, and compare against what the
card reports — and both use FP4. An unquantized load, which is what a store name carrying no
`-fp4`/`-fp8` suffix gets, has never been confirmed against real VRAM at all.

The case to add is `llama-3.2-3b-it` at BF16: predicted around 6.3 GiB, fits the 12 GB card, and
must not spill.

## Predicted and measured VRAM disagree, and the Windows counters cannot settle it

`models` · `measured`

The prediction and the measurement differ by about 1.0 GiB on Gemma and 0.45 GiB on Llama, with no
explanation. Scratch memory was measured and is not the cause. The cheap next step is per-allocation
rounding — read `MemoryAllocationStats::allocationCount` and divide, importing
`Compute.MemoryResourceTracker` directly since `Mila.ixx:95` comments the re-export out. Anything
under ~0.1 GiB is noise. (The larger Qwen gap was a different defect — un-pooled per-layer
transients — and its numbers must not be folded in.)

The instrument is part of the problem. On Windows `cudaMemGetInfo` cannot see memory the driver has
spilled to system RAM, so every measurement reads low; settling the gap means measuring with the
per-process counters Task Manager reads, `\GPU Process Memory(pid_N*)\Dedicated Usage` and
`\Shared Usage`. `MemoryFootprint.md` exists to answer "will this model fit" and does not yet say
which counter to trust.

**Gate B's residual bound fails in the dev container, and points at driver rounding.** Both families
on the 5060 Ti at context 8192: Gemma predicted 8.314 GiB against 8.604 consumed, residual
**0.290 GiB**; Llama predicted 9.811 against 10.078, residual **0.267 GiB**. The bound is 64 MiB.
What fails is only that bound — `predicted == reported` and `predicted <= consumed` both hold, so
the Models criterion (a reported footprint matching what a load allocates) is not what broke. Both
residuals sit at the rounding magnitude `MemoryFootprint.md` §11.8 records for Gemma 4 12B (0.27 GiB
at 1024 rows), and §11.10 flags Linux driver rounding as the one thing in that rule never measured.
First step is one reading: what `cuMemGetAllocationGranularity` returns in the container, against
the 2 MiB both Windows cards report. If it is coarser, the prediction is low by construction and the
bound is innocent.

Two things made this weaker evidence than it looked. The bound was **skipped whenever every visible
CUDA device drove a display** (`GemmaModel.Footprint.Cuda.cpp:290`), so a green Windows run may never
have executed it and the container may have been the first place it ever ran. And Docker Desktop is
WSL2-backed, which reaches the GPU through the Windows driver — §11.10 rules WSL2 out as a stand-in
for native Linux, so this is not the native-Linux number that section is waiting for.

**The first suspicion is confirmed: the bound had never run on native Windows, and it fails there
too.** Measured 2026-09-20, RTX 4070 pinned alone by UUID with `display_active: Disabled`, context
8192: Gemma 4 12B FP4 predicted 8.314 GiB against 8.572 consumed, residual **0.257 GiB**; Llama 3.1
8B **0.238 GiB**. Against the bound of 64 MiB.

Four measurements now exist and no single cause fits them:

| Card | Platform | Display | Gemma 4 12B | Llama 3.1 8B |
|---|---|---|---|---|
| RTX 4070 | native Windows | attached | 354-1180 MiB | 321, 369 MiB |
| RTX 4070 | native Windows | **disabled** | **257 MiB** | **238 MiB** |
| RTX 5060 Ti | native Windows | headless | 6-20 MiB | 21 MiB |
| RTX 5060 Ti | dev container | — | 290 MiB | 267 MiB |

Two axes are unexplained rather than one. **The display is not the whole account on the 4070** — a
quarter of a gigabyte survives turning it off, which is what 11.5's table attributes entirely to the
Windows budget cut. And **the platform matters on the 5060 Ti** — 6-20 MiB natively against 290 in
the container, on the same card. Driver rounding would explain the container and the 4070; it does
not explain why the 5060 Ti reads twenty times lower natively for the same model.

The bound was removed at `rc.1+27` (`MemoryFootprint.md` 11.5, superseding note), so this entry is
now the only thing tracking the number, and the 64 MiB figure never held anywhere except the one
card-and-platform combination the old test happened to select. The cheap reading named above --
`cuMemGetAllocationGranularity` on both cards and in the container -- is now the first step for both
axes, not just the container.

## Llama's Q, K and V views may not be contiguous

`llama`

The three splits of `qkv_out` at `Llama.Block.ixx:132` are taken as views, and the attention kernel
assumes a layout they may not have. Held to be benign rather than fixed: Llama's token-for-token
agreement with the HuggingFace reference is the evidence — a live aliasing defect could not produce
matching output. Worth confirming directly if that parity is ever re-established at a different
precision or on a different attention path.

## Nothing pins the training primitives independently of the samples

`training` · `ci`

v0.20 ships MNIST and Bard running and tested — that is the training claim, and it is met. What a
working sample cannot show is *which* piece is right: it proves the parts connect, not that the
optimizer steps in the right direction or that a loader hands over what it promises. When one of
them breaks, the sample says so and nothing says where.

The suite underneath is what closes that, and it is the whole of what was deferred:

- **Data-loader contracts.** `TokenSequenceLoader` is done. `MnistDataLoader` is not — normalization,
  one-hot targets, shuffle-on-reset, and the IDX magic number. Pin the TokenId signedness contract
  in the same pass (`TokenSequenceLoader.ixx:44`).
- **The AdamW CUDA path.** `AdamW.Cpu.cpp` is active with a convergence case; the `AdamW.Cuda.cpp`
  companion is not written. Strip or gate the debug `printf`s in `CudaAdamW.cu` and
  `CudaAdamWOptimizer.ixx:270` while there — a shipped optimizer should not print to stdout.
- **A step-convergence test.** Minimize a known convex objective in N steps, so the update direction
  and the bias correction are proven rather than just that `step()` returns.
- **A sample-independent training-loop integration test.** The MNIST spine is covered by
  `Network.Cpu.cpp`; a GPT-2-stack analogue for the Bard spine is not, nor is the
  `Core/Network.cpp` delta or the `Network.Cuda` companion.
- **Mode-transition coverage.** Assert that moving between training and runtime mode allocates and
  skips gradient buffers correctly. Three `REVIEW:` markers are the invariant, each guarding a state
  believed unreachable: `TokenEmbedding.ixx:221`, `Lpe.ixx:187`, `Lpe.ixx:495`.
- **CI gating.** None of the training-path tests run in CI, so coverage can rot silently the way it
  did the first time.

Scope stays FP32 GPT-2 / MLP. BF16 and GQA training are a later release entirely — see
[`Future.md`](Future.md).

- **The backward kernels the samples do not reach.** `CudaSoftmaxOp.ixx:73` and `:103` throw "needs
  review" with the real calls commented out, and `Gelu.Fp32.cu:65` records that the shipped backward
  is not the numerically stable `sech^2` form. Sweep the *unmarked* kernels precision twin by
  precision twin too — the RoPE FP32 backward was wrong while its BF16 sibling was correct, in a file
  carrying no marker at all.

## Chat configuration phase 7 — the two `ModelRecord` fields

`adaptors` · `distribution`

Phases 1 through 5 of the layered resolution have landed and Chat's configuration works. What is
left is the last phase, which reaches into Model Distribution for two fields on `ModelRecord`.
Design and phasing are in `Mila/Specifications/ChatConfiguration.md`.

## Saving never followed loading to safetensors

`architecture` · `mila-src` · `breaking`

Reading a model is safetensors end to end — `WeightsReader.ixx` and `SafeTensors.ixx`, neither of
which touches a serializer. Writing one is still the original archive stack:
`save_( ModelArchive&, SerializationMode )` is **public and pure virtual** on `Component`
(`Component.ixx:406`) and again on `Network` (`:344`), so every component must implement it — 24
`save_` and 5 `load_` today. Behind it sit
`Serialization/{ModelArchive,ArchiveSerializer,ZipSerializer,SerializationMetadata}.ixx` and
`Tensors/Tensor.Serialization.ixx`: roughly 1,800 lines whose only concrete backend is
`ZipSerializer`, whose only dependency is miniz, and whose only caller in `Mila/Src` is
`GptModel::fromCheckpoint` / `saveCheckpoint` (`GptModel.ixx:177`, `:231`).

**The writer already exists and is already used**, so this is not new machinery — it is pointing
`save_` at the machinery beside it. `LanguageModel::save` (`:173`) writes through
`Serialization::SafeTensorsWriter`. The hierarchical scopes `Network::saveComponentGraph` already
builds — `components/<name>/...` at `Network.ixx:516` and `:609` — flatten to safetensors keys the
way every other framework's checkpoints do, and `__metadata__` already carries the JSON that
`SerializationMetadata` carries now.

Doing it retires ModelArchive, ArchiveSerializer, ZipSerializer and miniz together and leaves save
and load speaking one format. **Do not substitute another container.** Tar was proposed and rejected
(Todd, 2026-09-09) — safetensors is the format, and no zip or tar will ever be needed.

Two things to settle first. Whether component-level checkpoints survive at all: training is the only
thing that wants them, `GptModel` is the only model exposing them, and `save` already
refuses a model reconstructed from a checkpoint (`LanguageModel.ixx:364`). And where the ~46 test
usages across 10 files that construct a `ZipSerializer` go — that round-trip coverage is real and
should move rather than evaporate. `ModelSerialization.md` is the design of record and needs amending
either way — its Phase 7 is stale for a different reason, recorded below.

**miniz goes with it** (moved from the v0.20 backlog at `rc.1+21`; the pin at 3.1.2 is current
upstream, which already meets the release's vendored-dependency criterion). Delete `ZipSerializer.ixx`
(the only `ArchiveSerializer` implementation and the only file naming miniz), the CPM block at
`CMakeLists.txt:228`, the `Mila.ixx:309` re-export, and the `PUBLIC` link plus the two `INTERFACE`
include directories at `Mila/CMakeLists.txt:962` and `:971-973` — which is what makes every
consumer's `import Mila;` recompile a module that includes `<miniz.h>`. Pinning before removal was
declined (Todd, 2026-09-09): a different miniz revision fails the build loudly or yields a
serializer no published path reaches, so it cannot silently change a shipped artifact.

## `ModelSerialization.md` Phase 7 describes shipped work as unwritten

`docs` · `distribution`

The distribution path exists end to end — `LanguageModel::save` (`LanguageModel.ixx:173`), the
`mila_quantization` metadata key, the reader, the policy check, `Linear`'s pre-packed load branch,
and `Tools/ExportArtifact` driving the whole thing. The phase text still calls it unwritten, and the
freeze-boundary table still lists it out of bounds.

A specification is the design of record, so a stale one misleads whoever reads it next. Deferred
because no user meets this file.

## The `mila` tool has no `pull` verb

`distribution`

Every other store verb is on the tool; the cold download is not, so it cannot be exercised from a
C++-only machine without a human sitting at Chat's `/install` prompt. Python is covered —
`ModelStore.pull` is bound at `Mila_py.cpp:309` and is what pulled 6.33 GB in the Linux clean room.

A missing verb rather than a broken path, which is why it waits. It lands on `mila` beside the other
store verbs, and is **not** `ExportArtifact --fetch`.

## `gpt2-small`'s installed record predates the licence role

`gpt` · `distribution`

The store copy declares weights and tokenizer only, so the hub repository carries LICENSE and the
local disk does not — the exact split the legal-files change exists to close. The obligation is met
where it is published; only the installed copy is short.

Reinstalling from `Data/Models/Packages/gpt2-small` fixes it, and both blobs are already adopted, so
it costs one small file.

## Editing `mila/__init__.py` alone leaves the build directory stale

`build` · `binding`

`Mila/Bindings/CMakeLists.txt:121` stages it with `copy_if_different` off
`add_custom_command(TARGET MilaPy POST_BUILD)`, which runs only when `MilaPy` relinks — so a change
to `__init__.py` and nothing else leaves `<build dir>/python/mila/` holding the old copy, and a
sample fails with a missing attribute. `add_custom_command(OUTPUT ...)` with `DEPENDS` on the source
is the fix.

Local development only: the wheel packages `__init__.py` from `Bindings/Package/src/mila/`, where
the file is tracked, so nothing stale can reach a published wheel.

## Two models cannot share an execution context

`api` · `mila-src`

`GemmaModel::load` takes a `DeviceId` (`GemmaModel.ixx:130`), not an `IExecutionContext`, so two
models loaded in one process cannot share a stream. `IExecutionContext.ixx:66-74` documents this as
deliberate: an overload would make the activation observer a cross-model leak.

Here to be confirmed or changed rather than because it is known wrong. The type is reachable — it is
the parameter type of the public `TensorOps` transfer functions (`TensorOps.Transfer.ixx:90`) and of
`Component::setExecutionContext` (`Component.ixx:896`), and `MnistClassifier.ixx:84` builds a network
on one.

## `mila-llm-server` is not on PyPI

`binding` · `ci` · `distribution`

MIS is restructured and its version derives from `Version.txt`, so what is left is the release step
itself: RELEASING covers the four CUDA wheels and says nothing about the server. One `py3-none-any`
file from `python -m build`, uploaded beside the wheels.

v0.20 ships MIS drivable from source and from the container, which is what the release bar asks for.

## CI installs the container's toolchain again instead of building `FROM` the image

`ci` · `build`

`build-pipeline.yml:47` starts from the bare `nvidia/cuda` devel image and apt-installs clang, gcc,
CMake and the rest on every run — the same set `Docker/Dockerfile` already bakes into the dev image.
Two definitions of one toolchain that can drift. Moved from the v0.20 backlog at `rc.1+21`.

## `mila serve <args>` loses every argument on Windows

`adaptors` · `build`

`runProgram` (`Cli.ixx:100`) hands a concatenated string to `std::system`, so `cmd.exe` strips the
outer quotes of the whole command line and nothing survives; the code returned is the shell's rather
than the server's. Launch with an argument vector — `CreateProcessW` or `posix_spawn` — behind a
CMake-selected module partition, since module code carries no `#ifdef`. Moved from the v0.20
backlog at `rc.1+21`: `Mila/Tools` does not ship, and the runtime image is Linux.

## Qwen refuses prompt-prefix reuse and never says so

`qwen` · `adaptors` · `mila-src`

`QwenDeltaNetBlock::rewindKvCache` always returns false — correctly, since a recurrent state is a
lossy summary and cannot be rewound — and `QwenTransformer::rewindKvCache` ANDs that into a refusal
for the whole stack. A server that reuses prefixes has to read this as a property of the model and
plan around it, not discover it as a failed retry. The per-block mechanism exists
(`snapshotState`/`restoreState`); a whole-model policy does not. Moved from the v0.20 backlog at
`rc.1+21`: prefix reuse lives inside each model's `generate` — Gemma's is transparent
(`GemmaModel.ixx:364`) and Qwen's always prefills from 0 (`QwenModel.ixx:355`) — and no adaptor or
the binding calls `rewindKvCache`, so nothing can meet the refusal as a failed retry. It becomes live
work when an adaptor manages reuse itself (`Direction.md:211` plans the agent core reading it from
the manifest).

## The samples are not built in CI

`ci` · `build`

Only the tests build today, so a sample can stop compiling without anything noticing — and the
QuickStart samples are a published surface the website's Get Started tabs link to.

The C++ quick start is the sharpest case: it is a standalone FetchContent project, so neither
`x64-validate` nor CI adds it, and `packaging_fetchcontent_consumer` compiles its own `main.cpp`
rather than this one. Its only builder is `Dockerfile.runtime`, which copies it into the devel image.
Configuring it with `-DFETCHCONTENT_SOURCE_DIR_MILA=<tree>` needs no network, which is what a gate
would do. `Samples/QuickStart/Cpp/main.cpp`

## Public Doxygen still describes a world that was refactored away

`docs`

Mechanical drift — `@file`, `@param`, `@tparam` names disagreeing with signatures — is what
Doxygen's own warnings catch and is v0.20 work. This is the half no tool can see: prose that reads
correctly and describes the pre-`OperationTraits` design, on the API a consumer calls.

There is no bounded worklist, which is why it is not release work. It is cleared opportunistically —
a file's prose gets fixed while the file is already open for another reason.

## There is no guided reading path through the source

`docs`

Mila's positioning is the stack you can read, and nothing shows a reader where to start. One token's
journey — embed, attend, sample, decode — through the real source, followable by a strong C++
developer unaided. Moved out of v0.20 at `+25`; the ROADMAP criterion and `MilaProductFamily.md`'s
release boundary were narrowed in the same commit.

Shape proposed then: Llama 3.2 3B, BF16, CUDA, the plainest architecture, with short detours to
where Gemma, Qwen and FP4 differ. About ten stops from QuickStart through `LlamaModel::load`,
tokenizer, `TokenEmbedding`, the block, `Linear`'s `OperationTraits` dispatch, lm_head, sampler,
decode loop and detokenize. Link file plus symbol, never `file:line`. Validate by handing it to a
reader with no context. No anchor: the finding is an absence.

## Counting a prompt's tokens needs Python

`api` · `tokenizer`

A C++ user asking whether a prompt fits the context has no command for it. `Mila/Tools/Tokenize`
trains and applies its own `--vocab` file and cannot take a store name; the `mila` CLI has
`install`, `models`, `serve` and `help`. The binding answers it in one line --
`BpeTokenizer.from_store( name ).encode( text )` -- so the library already carries the capability
and only the C++ surface lacks it. A `mila tokens <name> <file>` verb in `Mila/Tools/Cli`, printing
the count, is the shape. Found while measuring prefill rates, where the prompt length had to be
established before any rate meant anything.

## One template parameter, two spellings

`api` · `docs` · `mila-src`

The compute-precision axis is `TPrecision` in most of the tree and `TComputePrecision` in nine files
— 122 occurrences, split along no principle, with `Linear` and `GroupedQueryAttention` each
differing from their own siblings. It has already cost compile errors. The larger half is
`TWeightQuant` where CLAUDE.md mandates `TWeightQuantization`: 97 occurrences over 12 files, three
of them specifications.

`TWeightQuantization` is part of `Linear`'s public template signature, so a consumer meets both
spellings of one axis. Not a blind sweep — `GroupedQueryAttention.ixx` and `CudaRopeOp.ixx` use
both. Same files throughout, so it is one pass.

## Forty-six module files export more than one type

`architecture` · `adaptors` · `binding` · `mila-src`

CLAUDE.md requires one exported type per module file, and 46 of 367 `.ixx` files predate the rule
— about 105 new files to split fully (2026-09-19): `Mila/Src` 33 files (~60), Chat 8 (25), Bindings
1 (16, all in `Mila_py.Wrappers.ixx`), Tools 2 (4). `Metal`/`Rocm` `MemoryResource` define one
type twice across `#if`/`#else` and are not violations.

Src shapes: a class plus its request/result structs (`ModelStore` has nine types); an enum beside
the one class it configures (`Logger` + `LogLevel`, eight files); families of one idea
(`Weight/Policies`, `LearningRateScheduler`); four `*Registrar` pairs, which should be deleted with
the registrar pattern rather than split. Qwen's two blocks define their workspace inline where
Gemma uses a partition. Ruling needed first: whether an enum used by exactly one class counts as
that class's internal detail — it moves Src between ~25 and ~60 new files.

## The wider `Tensors/` tree has no coverage beyond `Tensor` itself

`architecture`

Distinct from "The authored test suite is green except for the files still commented out", which is
revival. This is the layer underneath, which
the old suite never covered at all — `TensorBuffer`, the `TensorDataType*` maps, `Partitioning`, and
`Serialization`, plus the `TensorOps.Transfer` device split. New coverage rather than revival.

Eight `REVIEW:` markers already name the exact contracts to pin; `Specifications/Testing.Tensors.md`
carries them.

## `import Mila;` degrades the standard library in a consumer translation unit

`api` · `build` · `mila-src` · `blocked`

Held to be an **MSVC modules defect** rather than a Mila one, which is why v0.20 ships it documented
and gated rather than fixed. Three failures, all compile-time, all absent the moment the import is
removed, re-verified 2026-08-30 on MSVC 14.51.36231:

1. **No C++ stream input.** `std::getline(cin, s)` and `cin.getline(char*, n)` both fail with
   `C2079: '_Ok' uses undefined class 'std::basic_istream<char>::sentry'`. Output is fine. It also
   bites *inside* `Mila/Tests`, and covers `<fstream>` — including that header in a test that
   imports Mila reproduces it.
2. **Instantiating a model needs `<sstream>` included first.** `Component::toString()` is pure
   virtual (`Component.ixx:632`), so `GemmaModel::toString()`'s body (`GemmaModel.ixx:261`) compiles
   into the consumer through the vtable and uses `std::ostringstream`.
3. **The import must come after the `#include`s.** Import-first is fatal — C1116, whose Microsoft
   documentation names mixing `import` and `#include` as a cause.

1 and 2 are independent: `<sstream>` does not repair input.

**Reachable is not visible, and more includes cannot fix it.** `GemmaModel.ixx:14` already includes
`<sstream>` in its own global module fragment, and 91 modules under `Mila/Src` do the same — the
consumer still needs its own copy first. GMF entities are reachable in an importing translation unit
but not visible, and MSVC will not instantiate a class template whose definition is only reachable.
One mechanism, all three symptoms, and no include added on Mila's side closes it.

**The intended fix is `import std;` throughout.** It is the only candidate that addresses the
mechanism rather than a symptom: with no textual std anywhere, there is no global module fragment to
own std entities and no reachable-but-invisible state for them to be in. It needs
`CMAKE_CXX_MODULE_STD` set before `project()`, so it means a fresh build directory and a full
rebuild, and it is a tree-wide change to every module's GMF — which is exactly why it waits for a
release that is not in flight.

The alternative is recorded only so it is not mistaken for the plan: moving the `toString()` bodies
out of the module interfaces would stop the consumer compiling those instantiations, but it
addresses failure 2 alone and leaves the mechanism intact.

Check a newer MSVC before starting either — the cheapest outcome is that the toolchain fixed it.
Filing upstream is worth considering: a 2026-08-13 search found no matching report, and MSVC emitted
its own report-a-modules-bug note. **Do not narrow `Mila.ixx` in response to this** — narrowing the
umbrella was measured and reverted, and this resolves by widening if it resolves in Mila at all.

**What v0.20 ships instead.** Both workarounds are commented at the point of use in
`Samples/QuickStart/Cpp/main.cpp` and in `Mila/Tests/Packaging/fetchcontent_consumer/main.cpp`, and
that fixture pins all three defects by compiling — ordering for 3, a never-called
`instantiateModelFromConsumerTranslationUnit()` that forces the vtable for 2, and the `fgets`
workaround for 1. It needs no GPU and no model, so it will report the day the workarounds become
unnecessary.

## A dispatch row that lies still fails as a constraint cascade

`architecture` · `api` · `mila-src`

The friendly primary-template assert — `OperationDispatch.md` §12 **A** — landed, so a *missing*
`(Op, Device, Precision, Policy)` specialization now reads as a sentence naming the unsupported
combination instead of an incomplete-type puzzle. That is the half v0.20 claims.

What remains is the worse failure mode: a specialization that **exists** but maps to a kernel whose
own constraints fail, so the dispatch table advertises a capability the backend does not have and
the diagnostic is a cascade deep inside the kernel that never names the cause. §12's motivating case
is `OperationTraits<GeluOp, Cuda, BF16>` mapping to a `cuda_gelu_impl` constrained to
`float || half`. The fix is §12 **B**, one authoritative `OperationSupported` predicate that the
kernel constraint, the traits specialization and a component-level assert all reference, so the two
places that can drift become one. §12 **C**, naming the kernel concepts, is marginal and free as
each op is touched.

**Its intended carrier is gone.** §12's Adoption note proposed pairing this with the FP16 removal,
when each op's supported-precision set would be made explicit anyway — but FP16 is not used in Mila,
so this needs its own pass or a new pairing.

## Llama parity has no automated regression test

`llama` · `ci`

Llama's agreement with the HuggingFace reference has been established by hand, but nothing holds it
— so the next change to the load path or the attention kernels can break it silently. Gemma has
`GemmaModel.Parity.Cuda.cpp` as the template, and it is the right one: Qwen's
`QwenModel.Parity.Cuda.cpp` compares hidden states one decoder block at a time, because a 50 GiB
BF16 reference cannot be resident, so it is not a decode test. Llama is the family where a
permanent token-for-token test is affordable at all.

## Llama's long-context scaling factor is stored, printed, and never used

`llama` · `mila-src`

The presets carry it — 32.0 for the 3.2 models, 8.0 for the 3.1 ones (`Llama.Presets.ixx:50`, `:70`,
`:96`) — `LlamaConfig` stores it, serializes it into the metadata and restores it, and three tests
assert it. It reaches nothing. `getRoPEScalingFactor()` has three non-test callers and all three are
serialization or `toString()`; `Llama.ixx:751` builds the Rope component with `.withRoPETheta()` and
nothing else, and `Rope.Config.ixx` has no scaling knob at all — `withBase`, `withRotaryDim`,
`withRotaryLayout`.

So the commented-out `.withRoPEScalingFactor( metadata.rope_scaling )` at `Llama.ixx:837` is a
symptom rather than the defect, and uncommenting it changes no output. The work is Llama 3's
NTK-by-parts frequency scaling implemented in the Rope component and its CPU and CUDA operations,
then wired through — which is why this is not the small item its `REVIEW:` marker makes it look.

It leaves one question open that is worth answering before the work rather than after: Llama's
token-for-token agreement with HuggingFace was established with the factor inert, and HuggingFace
applies Llama 3 scaling at every position rather than only past the original window. Either the
parity runs used checkpoints whose config declares no scaling, or the agreement is narrower than
recorded.

## Tool calling has never been driven end to end on Llama

`llama` · `adaptors`

Gemma 4 is validated through both adaptors across plain-chat, single-tool and tool-result-resume
flows. Llama 3.2 3B and 3.1 8B are the primary validated inference lineage in the product definition
and neither has been driven through a single tool call, from Chat or over the wire. The tool-calling
framework itself is complete and family-neutral; what is missing is the evidence that Llama's
grammar reaches it.

Moving this out narrowed two published claims in the same commit — the ROADMAP Models criterion and
`README.md`'s Llama section — so nothing now asserts it. Related: the parity entry above, which is
the other half of what "validated lineage" is asked to mean.

## GPT-2 and Llama 3 tokenize by approximation on every build and platform

`tokenizer` · `gpt` · `llama` · `mila-src`

`\p{L}` and `\p{N}` (`BpePreTokenizationMode.ixx:33`, `:57`) compile in no standard `std::regex`, so
`BpeTokenizer.ixx:344` throws and falls back to an ASCII scanner — in every build, the published
Linux container included. No parity test catches it, which raises a second question to settle in the
same pass: whether the tokenizer fixtures are English-only, because if they are, nothing here was
ever measurable.

The fix is PCRE2, RE2, or a hand-written Unicode scanner, which makes this a vendored-dependency
decision as much as a tokenizer one — the same ground the curl pin sits on.

## The authored test suite is green except for the files still commented out

`architecture`

v0.20 ships the revival: 153 test sources on disk, the concrete component, tensor and tokenizer
suites re-aligned to the post-refactor API, the redundant per-op mirror tests retired down to three
files covering the dispatch mechanism itself, and the CPU arm running in CI as a ratchet
(`build-pipeline.yml`, clang-21, `MILA_ENABLE_CUDA=OFF`). That is the claim, and it stops at what is
compiled.

What is left is the tail that was never re-enabled — ten CPU-side and four CUDA-side entries still
commented out in `Mila/Tests/CMakeLists.txt:154-166` and `:283-290`: `Network.cpp`,
`CrossEntropy.Config.cpp` and `.Cpu.cpp`, `Model.cpp`, `Model.Cpu.cpp`, `ModelArchive.cpp`,
`ZipSerializer.cpp`, `TensorBuffer.Tracking.cpp`, `Network.Cuda.cpp`, `AdamW.Cuda.cpp`,
`SoftmaxCrossEntropy.Cuda.cpp`, `CudaDevice.cpp`.

Several are not simply switch-on work and should not be estimated as if they were.
`SoftmaxCrossEntropy` waits on loss moving onto the device, and the `AdamW.Cuda` and `Network.Cuda`
entries belong with the training primitive suite rather than here — see "Nothing pins the training
primitives independently of the samples", which owns the backward-numeric cases skipped for the same
reason.

## The quantization and Llama inference paths have no coverage

`quantization` · `llama`

Both were built during the test drought and the authored suite never reached them. `OperationTraits`
dispatch is now covered; two gaps are not.

**The load-time quantization white-box** — `PerChannelFp8`, `PerGroupFp4` and the decode matvec
kernels. This is the one legitimate op-layer test in the tree, because the packed path cannot be
reached through the public component at all, which is why retiring the op-layer mirrors does not
retire it.

**The Llama path**, where the concrete case is the context limit. `LlamaModel.ixx:329` throws when a
prompt exceeds the deployment context and `:363` returns `ContextOverflow` when decode would write
past it; nothing exercises either. The comment at `:359-362` states the stakes precisely — RoPE is
computed rather than looked up, so there is no positional table to run off the end of and nothing
crashes. Where GPT-2 crashes, Llama overruns the cache quietly, so absence of reports is not
evidence. `Tests/Dnn/Models/GptModel.Cuda.cpp` is the template: a weightless checkpoint at a small
deployment context.

## Llama prefill has no flash path and is 3.8x slower than a larger Gemma

`llama` · `perf` · `mila-src` · `measured`

On the RTX 5060 Ti, one build, both FP4, 22496 tokens at context 49152: Gemma 4 12B prefills at
~1461 tok/s through `gqa_flash_prefill_mma_bf16_kernel`, Llama 3.1 8B at ~382 tok/s through
`Gqa::prefill_softmax_bf16_kernel`. Attention is 75.3% of Llama's prefill against Gemma's 59.5%.
Flash prefill is wired on Gemma's blocks and not Llama's. `Compute/Devices/Cuda/Operations/Gqa/`

## MIS reports every response as finished naturally, including truncated ones

`adaptors`

Five sites hardcode `finish_reason: "stop"` — `chat.py:66`, `completions.py:49`, `factory.py:137`,
`:155`, `:200` — and the Anthropic path returns `stop_reason: "end_turn"` the same way, so a client
is told a reply ended when it was cut off by `max_tokens` or the context. The binding's `generate` now
returns `GenerateStatus`; `ModelWorker.generate` and `generate_streaming` are where it would be
threaded to the routes. OpenAI spells the cap `length`, Anthropic `max_tokens`, and neither has a
spelling for a context overflow — that mapping is the decision owed.

## A download that fails part-way does not say that running it again resumes

`distribution`

`install` died at 35% of a 2.86 GiB transfer with `Transferred a partial file (TransportError)`
(`ModelStore.ixx:457`). The partial is named by digest precisely so the next run resumes (`:342`),
but the message says none of that, and on the evaluation path a raw transport error reads as broken.
Whether resume engages on the container's named volume is untested — `verify-image.sh` uses a
throwaway volume by design.

## A CUDA allocation failure is reported four ways, and in five places not at all

`architecture` · `api` · `mila-src`

`CudaDeviceMemoryResource` throws `CudaBadAlloc`; the pinned (`CudaPinnedMemoryResource.ixx:101`,
no message) and managed (`CudaManagedMemoryResource.ixx:89`, builds a message and discards it)
resources throw bare `std::bad_alloc`; `CudaExecutionContext`'s scratch, staging and cuBLASLt
workspace and the `CudaTensorOps.Random` buffers throw `std::runtime_error`; `RopeCacheRegistry`
throws `CudaError`. So "the model does not fit" cannot be caught as one type, and `import Mila;`
does not export `CudaBadAlloc` at all.

Underneath, two exception classes carry the same CUDA runtime error: `cudaCheck`
(`CudaUtils.h:31`, 157 kernel launch sites) throws `CudaException`, `cudaCheckStatus` /
`cudaCheckLastError` (`CudaError.ixx:133`, `:145`) throw `CudaError`, with different message formats,
and only `CudaException` exposes the code. And five allocations ignore `cudaMalloc`'s result entirely
— `CudaLinearOp.ixx:1509`, `:1510`, `TensorOps.Fill.cu:112`, `:133`, `CudaLinearGelu.cu:62` — so a
failure leaves a null pointer for the next kernel.

## `CudaExecutionContext` allocates its buffers without selecting its device

`architecture` · `mila-src`

The forward scratch, load staging buffer and scratch reservation call `cudaMalloc` without
`Cuda::setCurrentDevice`, where `CudaDeviceMemoryResource::do_allocate` selects first
(`CudaExecutionContext.ixx:274`). The constructor selects once, so in a two-GPU process a buffer lands
on the wrong card only if something changes the current device in between. Latent today; the layer
split (`LayerSplit.md`) makes it live.

## A 4 GiB single-tensor limit marked DEBUG ships in `TensorBuffer`

`architecture` · `mila-src`

Any tensor whose storage reaches 4 GiB throws `std::length_error` from the constructor
(`TensorBuffer.ixx:224`). The check sits between `// DEBUG:` and `// END DEBUG:` with no condition, so
every build carries it, and a tensor large enough to fail on the device never reaches `cudaMalloc`.

## The FP8 per-tensor weight quantizer has no caller and still stages the whole tensor

`quantization` · `mila-src`

`Detail::quantize_fp8_per_tensor` (`CudaLinearOp.Quantize.ixx:106`) and
`cuda_quantize_fp8_per_tensor` (`CudaFp8WeightQuantization.cu:251`) are exported and documented as
the Ada cuBLASLt path, and called from nowhere. When the per-channel path moved to row blocks it was
left alone, so it is the one quantizer that still needs the whole BF16 tensor on the device. Delete
it, or give it row blocks with a running maximum if a per-tensor path is wanted.

## A cuBLASLt plan with no algorithm defers the choice to every execution

`perf` · `mila-src`

When the heuristic succeeds with no algorithm, the builder logs "will use default at execution" and
returns `has_algorithm = false`, and execution passes a null algorithm to `cublasLtMatmul`
(`CublasLtPlan.ixx:333`, `:383`; also `CublasLtLinearPlan.ixx:441`, `:514`, and the FP8 prefill
builder at `:646`, `:681`). A plan built without a decision looks like one built with one, except for
a warning at build. `Deployment.md` §7 rules the same shape out for deployment plans.

## No test catches a scratch reservation larger than any request

`models`

Phase 6 step 2's negative — scratch summed across the tree instead of taking the maximum — reserved
12,386,304,000 bytes for Gemma 4 12B against a largest request of 121,901,056, and every permanent
test passed: predicted equals reported, nothing throws, memory does not grow. A per-model literal of
the reserved bytes in `ScratchReservation.Cuda.cpp` would fail on it, as the other footprint literals
do.

## The published binaries are built on CUDA 13.3 while 13.4 is current

`build` · `ci` · `distribution` · `blocked`

Move the declared toolkit to 13.4.2. Held for v0.20 (Todd, 2026-09-17): on that date Docker Hub's
`nvidia/cuda` had no 13.4 tag for any OS, and the Linux wheels, runtime image, dev container and CI
all build `FROM` it. Unblocks when `13.4.2-devel-ubuntu26.04` and `-runtime-ubuntu26.04` exist —
not by an apt-installed toolkit on a plain base.

Moves together (RELEASING.md, toolkit paragraph): `$cudaVersion` in
`scripts/pypi/build-wheel-windows.ps1:59`, `Docker/Dockerfile.wheel:23`, `Docker/Dockerfile.runtime:21`,
`Docker/Dockerfile:18`, `build-pipeline.yml:47`, and the docs naming 13.3 — `README.md:285`,
`:295`, `:336`, `:350`, `getting-started.md:26`, `:34`, `:119-134`, `:183-191`, `:241`,
`CONTRIBUTING.md:46`, `:58`, `:85`, `Docker/README.md:15`, `:20`, `Web/content/start.md:15`,
`RELEASING.md:382`.

Consequences to carry into the work. Wheel users see nothing — the `nvidia-*` dependencies and
minor version compatibility, and this is now checked rather than assumed: a 13.3-built wheel loads
and generates correctly against the `>=13.0` floor those dependencies declare, on Windows
empirically and on Linux by symbol (`RELEASING.md`, *What the declared toolkit is not*). Moving the
build to 13.4 does not disturb that, but re-check the floor if the cuBLASLt surface grows. Image
users' driver floor rises: the base image's `NVIDIA_REQUIRE_CUDA`
becomes `cuda>=13.4`, and the container toolkit refuses a GeForce driver below it. Every local build
directory is configured against v13.3 while `CUDA_PATH` names v13.4 (13.4.1 installed), so a fresh
configure already drifts — reconfigure all of them deliberately. Published tok/s figures and the
cuBLASLt findings in the specs are 13.3 measurements; re-measure or label them. CI's first run
starts with a cold ccache. The patch levels already differ today: Windows pins resolve to 13.3.1,
the Linux images to 13.3.0.

## A consumer's path budget is about thirty characters, spent by one seven-level include

`build` · `mila-src`

`ElementwiseActivation.cu:21` includes an 86-character `../../../../../../../` path, and MSVC applies
MAX_PATH to the unresolved string, so a FetchContent consumer with a source root deeper than about 97
characters gets C1083 on a header that exists. It tipped the CPM gate over at beta.3 (264 against
260). `Geglu.cu:17` (seven levels) and `Moe.cu:12` (six) include the same header. Fix: one `PRIVATE`
include directory, `Mila/Src/Dnn`, on the `Mila` target, and the three includes shortened to
`"Components/Activations/Activation/Kernels/ElementwiseActivation.h"`. Moved out of v0.20 at `rc.1+24`
(Todd): if the v0.20.0 CPM gate trips on it, it is fixed then, in the release.

## No Ampere or Turing card has ever run Mila

`build` · `binding`

The supported architecture list is `80;86;89;90;120` on the reasoning that SM 8.0 is the floor
Mila's kernels draw — the FP4 GEMM gates on `major >= 8` (`CudaLinearOp.ixx:661`) and both GQA
flash prefill paths throw below it (`Gqa.Flash.Fa2.cu:513`, `Gqa.Flash.Wmma.cu:632`). No one has
observed it: the dev box has only sm_89 and sm_120. At `rc.1+24` the published "what you need" lines
(website x5, `getting-started.md`, `scripts/dockerhub/overview.md`) were narrowed from RTX 30-series
to RTX 40-series (Todd), so a rented A10G or A100 hour is what widens them again.

The Turing half is closed: at `rc.1+27` `MILA_LIBRARY_CUDA_ARCHITECTURES` dropped 75, so nothing in
the tree compiles for it and the non-WMMA `cuda_fp4a16_gemm` fallback (`CudaLinearOp.ixx:882`) is
unreachable by construction rather than by argument. What remains is whether that fallback should be
deleted outright, which needs someone to confirm no non-GQA path can still dispatch to it. **Ampere
is the live question** — sm_80 and sm_86 ship in every artifact and have never run.

## Nothing now catches the footprint prediction drifting while it still fits

`models` · `perf`

Gate B used to bound the unmodelled residual at 64 MiB, which is how a prediction that quietly
worsened became visible. That bound was removed at `rc.1+27`: the residual varies by card and by
platform for reasons not yet understood (see the table in the entry above), so an absolute figure
cannot hold portably, and the old test reached green by selecting the one device where it did. The
tests now assert the decision a user depends on -- Mila read the VRAM actually free, said the model
fits, and it loaded -- which holds on any card but does not notice a prediction worsening by 300 MiB
on a card with room to absorb it.

The residual is still printed by both Gate B tests and by `QuantizeOnLoad.Footprint.Cuda.cpp`. What
is missing is anything that compares it with last time. It only means something against a stated
card, so it wants a measurement surface that records card and figure together, not an assertion in
a unit test. `MemoryFootprint.md` 11.5 carries the superseding note.

## A model listing that cannot price a row gives no reason for it

`adaptors`

`verdictFor` (`Chat.ModelCatalog.ixx:768`) distinguishes measured-and-too-big from
could-not-predict, but `RowVerdict` (`:596`) carries only the cell text and a tint, so the reason
is discarded at the point it is known. The load pre-flight now keeps its own reason —
`predictFootprint` returns one beside the optional — which leaves the listing as the one surface
that still shows an unexplained blank.

It belongs at `/verbose all`, matching `reportFootprintBeforeLoad`. The listing does not currently
receive the detail level, so that has to be threaded through first.
[[feedback_absent_output_is_evidence]]

## Sampling knobs are reachable in a session but not from the command line

`adaptors`

`temperature`, `top_k` and `top_p` are settable with `/set` and readable from `session.json`, so a
`-p` one-shot — the invocation most likely to want a fixed temperature — cannot vary them at all.

`main.cpp:1006` already reads all three from settings, so this is three flag producers rather than
a design.

## GPT-2's end-of-text token is a literal in the model rather than tokenizer metadata

`gpt` · `mila-src`

`Mila/Src/Dnn/Models/Gpt/GptModel.ixx:409` holds `static constexpr int32_t eos_token_ = 50256`. It
should come from the tokenizer.

## The Llama converter writes a metadata key the reader never parses

`llama` · `docs`

It emits `norm_eps` (`Mila/Tools/Converters/Llama/convert_weights.py:188`); `parseMetadataJSON`
extracts `norm_epsilon`, which is what Gemma and the packer both emit. Harmless only because
`LlamaModel::configFromMetadata` never reads the epsilon — so the guard against it becoming harmful
is an accident rather than a decision.

## Any response containing a bracket enters the tool-call parser

`adaptors`

`Chat.ToolCallParser.ixx:60` uses `response.find( '[' )` where the class's own doc comment at `:34`
says "Leading `[`", and the nested `parseTagged` path tests it correctly.

It degrades gracefully today, but any prose with a bracket enters the path, and a parse that ever
*succeeds* on prose would swallow the answer and emit a phantom tool call. Same failure shape as
the malformed-Gemma-call entry above, in the adaptor rather than the library.

## `ModelSize` is declared and read nowhere

`adaptors`

Four values in `Chat.Config.ixx:31`, plus a mention in the file-level Doxygen at `:5`, and no
reader anywhere in the tree — the model's identity is its store name, which is what replaced it.
Left in place it invites the next family to add a fifth value that nothing will ever read.

## A wrapped list item reads as a new paragraph

`adaptors`

A continuation line starts at the bullet's own indent rather than under the item text.
`wordWrap` (`Chat.RichText.ixx:192`) preserves a line's leading indent but has no notion of a
continuation indent, so the hanging indent a list needs cannot be expressed.

## Chat carries its own copy of the `nlohmann.json` module

`adaptors` · `build`

`Mila/Adaptors/Chat/Src/Json.ixx` duplicates `Mila/Src/Utils/json.ixx`, both including the same
header from their global module fragment. Chat then imports one in four translation units
(`Chat.ixx`, `Chat.MessageFormatter.ixx`, `Chat.SystemPrompt.ixx`, `Chat.ToolCallParser.ixx`) and
the other in two (`Chat.ModelCatalog.ixx`, `Chat.Settings.ixx`), so the same types arrive under two
module names in one binary.

Drop `Json.ixx` from the target and import `nlohmann.json` everywhere.

## The store's 24-hour lock reclamation is untested

`distribution` · `ci`

`ModelStore.ixx:1170`'s `isAbandoned` decides whether a `.lock` left by a dead process is
reclaimable, and the sweep at `:1035` is its only caller. Testing it needs a file with a backdated
write time.

Make the threshold a constructor parameter so a test can set it to zero — a better shape than
backdating with `last_write_time()`.

## `actions/setup-python@v5` still declares Node 20, which GitHub has deprecated

`ci`

It warns on every clean-room run (`.github/workflows/wheel-cleanroom.yml:60`). Every other action
in the tree is on `@v5` and clean; bump it and re-check the rest, since the deprecation applies by
action version rather than by repository.

## `Docker/README.md` credits Chat with a compiled-in models directory it does not have

`docs`

`Docker/README.md:58` says the Chat build compiles `MODELS_DIR` in. The only `MODELS_DIR` in the
tree is `Mila/Profiling/ProfileModel/CMakeLists.txt:24`. Chat resolves models through
`MILA_CACHE_DIR` and its config through the executable's own directory, which is why the published
image can drop the bind mount — so the claim reads as a hard dependency on `/mila` that is not
there.

## Thirteen `REVIEW:` markers have a recorded disposition and only need removing

`mila-src`

No analysis left, only removal: the 12 in `CudaGqa.Dispatch.ixx` answered by that file's own banner
at `:36`, plus `CudaOps.h:30`.

Three markers previously counted here do not belong to it. `Linear.cuh:97` is scoped by the
"Remove FP16" decision in [`Future.md`](Future.md) and goes when that does; `Component.ixx`'s
markers are at `:692` and `:841`, neither with a recorded disposition; and
`CudaDeviceMemoryResource.ixx` has none at all. 77 `REVIEW:` markers remain in `Mila/Src`, so this
is a first pass rather than the sweep.

## `Version`'s accessors are non-const

`api` · `mila-src`

`getMajor()`, `getMinor()` and `getPatch()` (`Mila/Src/Version.ixx:56,62,68`), so the version-skew
comparison needs a mutable copy of something it only reads.

## Nothing documents that CUDA's device 0 is not `nvidia-smi`'s

`docs` · `api`

`load`'s default `DeviceId{ Cuda, 0 }` picks whichever card CUDA enumerates first, which on a
mixed-capacity machine can be the smaller one. A load sized for the larger card then aborts in
about two seconds with no diagnostic, and reads as a model defect rather than a device choice.

The finding is an absence, not a location: a note wherever the default device is documented.
[[project_cuda_index_is_not_nvidia_smi_index]]
