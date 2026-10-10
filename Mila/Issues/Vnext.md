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

## Forty-five module files export more than one type

`architecture` · `adaptors` · `binding` · `mila-src`

CLAUDE.md requires one exported type per module file, and 46 of 367 `.ixx` files predated the rule
— about 105 new files to split fully (2026-09-19): `Mila/Src` 33 files (~60), Chat 8 (25), Bindings
1 (16, all in `Mila_py.Wrappers.ixx`), Tools 2 (4). `Weight/Policies` was split in the Q4_0 work
(`0.21.0-dev+12`), leaving 45. `Metal`/`Rocm` `MemoryResource` define one type twice across
`#if`/`#else` and are not violations.

Src shapes: a class plus its request/result structs (`ModelStore` has nine types); an enum beside
the one class it configures (`Logger` + `LogLevel`, eight files); families of one idea
(`LearningRateScheduler`); four `*Registrar` pairs, which should be deleted with
the registrar pattern rather than split. Qwen's two blocks define their workspace inline where
Gemma uses a partition. **Ruled 2026-09-26 (Todd, CLAUDE.md):** a type other modules must name gets
its own file; a type only its own file uses is not exported; a nested type is part of its owner only when
it is never used apart from it. An enum only its class's own methods take may be nested, so Src lands
between the ~25 and ~60 estimates, and each file is counted when it is touched. Any module a change
touches is brought under the rule in that change, so this entry drains without a sweep of its own.

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

[#30](https://github.com/ToddThomson/Mila/issues/30) is the record: the three failures, the
workarounds, the measured fix (`import std;` in the library) and its open unknowns. Standalone
repro for the Microsoft report:
[msvc-module-std-repro](https://github.com/ToddThomson/msvc-module-std-repro).

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

## Tool calling has never been driven end to end on Llama

`llama` · `adaptors`

Gemma 4 is validated through both adaptors across plain-chat, single-tool and tool-result-resume
flows. Llama 3.2 3B and 3.1 8B are the primary validated inference lineage in the product definition
and neither has been driven through a single tool call, from Chat or over the wire. The tool-calling
framework itself is complete and family-neutral; what is missing is the evidence that Llama's
grammar reaches it.

Moving this out narrowed two published claims in the same commit — the ROADMAP Models criterion and
`README.md`'s Llama section — so nothing now asserts it. Related: HuggingFace token parity, which is the other half of what
"validated lineage" is asked to mean, is held in the suite since `ModelFamilyParity.md` 8.4 L1.

## GPT-2 and Llama 3 tokenize by approximation on every build and platform

`tokenizer` · `gpt` · `llama` · `mila-src`

`\p{L}` and `\p{N}` (`BpePreTokenizationMode.ixx:33`, `:57`) compile in no standard `std::regex`, so
`BpeTokenizer.ixx:344` throws and falls back to an ASCII scanner — in every build, the published
Linux container included. No parity test catches it, which raises a second question to settle in the
same pass: whether the tokenizer fixtures are English-only, because if they are, nothing here was
ever measurable.

The fix is PCRE2, RE2, or a hand-written Unicode scanner, which makes this a vendored-dependency
decision as much as a tokenizer one — the same ground the curl pin sits on.

Qwen 3.8 takes the same path (2026-10-03). It surfaced in ContextProfile's loss arm: of the five PG-19
books the 2.82-bit build scored at 74752, the one that is UTF-8 -- 26183, curly quotes throughout,
13,532 non-ASCII bytes -- is the one whose loss rises with context and fails gate 1 in every band
(2.86 to 2.99 nats a token), where the four ASCII books fall with context and pass.

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

## Mila cannot import the format most quantized models on the Hub are published in

`quantization` · `distribution` · `mila-src`

Moved out of v0.21.0 on 2026-09-23 when the release widened. The Gemma 4 QAT build it was first meant
for was admitted to v0.21.0 on 2026-09-27 without it: `google/gemma-4-12b-it-qat-w4a16-ct` (read
2026-09-17) is `pack-quantized`, `int`, `num_bits` 4, `symmetric`, group 32, `lm_head` and the embedders
ignored, and the release may source that build from Google's BF16 QAT checkpoint instead (`BACKLOG.md`,
"Gemma 4 12B at 4 bits predicts worse the longer the context...").

compressed-tensors (the vLLM project's format) is safetensors plus a `quantization_config` in
`config.json`: a `format` (`pack-quantized`, `int-quantized`, `float-quantized`,
`nvfp4-pack-quantized`), per-group schemes (bits, `int`/`float`, symmetric, strategy tensor/channel/
group/block, `group_size`), and `targets`/`ignore`. A packed int4 Linear carries `weight_packed` (int32,
eight values each), `weight_scale` per group, `weight_shape`, and `weight_zero_point` only when
asymmetric. Packing order and scale dtype are from memory — settle them with a safetensors header read
before code.

Import it in `ExportArtifact` only, as a transcode into Mila's own safetensors with
`mila_quantization` metadata; the load contract, loaders, store and applications stay untouched, and a
layout mismatch surfaces at export rather than at load. Mapping, per format: `pack-quantized` int4
symmetric -> a new `PerGroupInt4<G>`; `float-quantized` FP8 per-channel -> the existing
`PerChannelFp8` (check scale shape and dtype agree); `nvfp4-pack-quantized` -> the native NVFP4
direction on SM120 (`Fp8ActivationPrefill.md`). Refuse any scheme with no matching policy, naming the
scheme.

## Prefilling in 2048-token chunks may be worth the context it costs

`models` · `perf` · `mila-src` · `measured`

Every family's largest prefill chunk is 1024. On Llama 3.1 8B at FP4 (RTX 5060 Ti, one build), 2048 prefilled
5,206 tokens/s at an 8K prompt against 4,763 at 1024 (+9.3%), and 2,766 against 2,559 at 32K (+8.1%), for 184 MiB
more activation memory, about 1,400 tokens of context. The planner holds context back to keep the largest chunk,
so adding the size costs that context on every card where memory binds. Measured with the chunk test in
`Llama.LogLikelihood.Cuda.cpp` (`DISABLED_PrefillRateByChunk_Fp4`, which already runs 2048); the full table is in
`ModelFamilyParity.md` 8.4, L4.

Not yet measured: Gemma 4 12B and Qwen 3.8 27B, whose activation widths and attention costs differ, so the gain
and the context price there are unknown; and the 12 GB card, where the context given up is a larger share. The
decision is one table for all three families or a per-family top chunk, and it is made on those numbers.


## Llama 3.2 runs on the CPU backend

`llama` · `architecture` · `mila-src`

Todd, 2026-09-29: wanted, and post-0.21. The CPU backend is FP32 only and GPT-2 shaped (`OperationTraits.Cpu.ixx`):
it has `LinearOp`, `RmsNormOp`, `ResidualOp`, `SoftmaxOp`, `SamplingOp`, `ElementwiseActivationOp`, `GeluOp`,
`LayerNormOp`, `MultiHeadAttentionOp`, `LpeOp`, `RouterOp` and `MoeOp`. Llama also needs `TokenEmbeddingOp` (CPU has
only GPT-2's token-plus-position encoder), `RopeOp` with Llama 3's frequency scaling, `GroupedQueryAttentionOp` with a
KV cache for chunked prefill and decode, and `SwigluOp`; then BF16 weights widened to FP32 at load, and the Llama
transformer and model building for `DeviceType::Cpu`. `Future.md` "The GPT-2 CPU path treats build-time extents as
runtime extents" is the defect the new ops must not repeat. Memory bandwidth bounds it: FP32 weights are about 5 GB
(1B) and 13 GB (3B), read once per token -- roughly 10 and 4 tokens a second on desktop memory. It is also the first
Llama that runs on an Apple M1 without a Metal backend (`Future.md` "Compute backends beyond CUDA"). The bar is the
CUDA families': token parity with HuggingFace.

## The 2.82-bit Qwen build prefills slowly because its 2- and 3-bit weights are widened to 16 bits before every matrix multiply

`qwen` · `quantization` · `perf` · `mila-src` · `measured`

Moved from `BACKLOG.md` on 2026-10-04 (Todd): Qwen's own tuning waits for a 24 or 32 GB card. At 8K the
codebook GEMMs are 68% of prefill and run cuBLASLt's BF16 kernel with its tensor pipe 99% busy, at the card's
BF16 ceiling, so no BF16 kernel closes it; the expansion to BF16 is another 6%. The 2.82-bit build prefills
705 tokens a second at 8K, three-quarters of llama.cpp on the format its users run. The fix is
`Quantization.md`'s Q4_0 decision 4 applied to codebooks: codes mapped through an INT8 copy of the codebook in
the INT8 prefill GEMM's tile load, no staging -- group 32 is one k32 MMA block, group 64 two. Projected 8K
prefill about 1,100 tokens a second. One new rounding (the fitted entries to INT8, at most 1/254 of the
largest); decode keeps its exact matvec. Gate: loss by band within noise of the BF16-staged path, then rates
before and after. `ModelFamilyParity.md` 8.3, Q9, lever 1.

## The 2.82-bit Qwen build generates below the card's memory bandwidth because its 2-bit kernel runs out of instructions first

`qwen` · `quantization` · `perf` · `mila-src` · `measured`

Moved from `BACKLOG.md` on 2026-10-04, as above. Decode reads at 65% of bandwidth against FP4's 79%. The 2-bit
matvec is SM-bound at 93% (a shuffle and a bit extraction per weight), the 3-bit one holds 24 warps of 48 at
79-80 registers, and DeltaNet's `in_proj_a` and `in_proj_b` are 96 launch-bound BF16 launches a token. Three
levers, about 3.1 ms a token together (33.8 to about 37.7 tokens a second at depth 0): a 16-entry FP16 pair
table so one shuffle returns two 2-bit weights; the 3-bit kernel's occupancy; `in_proj_a` and `in_proj_b` in
one launch. Gate: each codebook kernel DRAM-bound in Nsight Compute; greedy tokens and scores unchanged; rates
before and after. `ModelFamilyParity.md` 8.3, Q9, levers 3 to 5.

## Qwen's wikitext perplexity gate has only been run to 16K

`qwen` · `measured`

Moved from `BACKLOG.md` on 2026-10-04, as above. From 8K to 16K the FP4 oracle improves 7.2% while the 2.82-bit
plan improves only 3.4% (`Qwen3.8.md` §8 item 9; `DISABLED_QualityGateAcrossContextLengths`). Since then the
context profile measured the 2.82-bit build healthy to 74,752 on PG-19 -- loss falling with context, recall at
least 97% (`ContextProfile.md` section 8) -- which answers the release's question for the build it publishes;
the wikitext ratio past 16K, and the FP4 oracle's, are what is left.

## Qwen's head paths disagree in the third decimal, so a perplexity must fix the head width

`qwen` · `measured`

Moved from `BACKLOG.md` on 2026-10-04, as above. Same weights, same corpus: width 1 (the decode matvec) and width
64 (the W4A8-FP8 GEMM) do not produce identical numbers, so both arms of a quantization comparison use one
width. Probably already recorded at `Qwen3.8.md:509` and `:546` -- verify, and if so delete this entry rather
than work it.

## Qwen 3.8 27B FP4 predicts a book worse with the whole book before it than with 1024 tokens of it, from 8K on

`qwen` · `quantization` · `measured`

Moved from `Untriaged.md` on 2026-10-04 with Qwen's tuning (Todd); the FP4 build is not published again in v0.21.
Anchor: `QwenOraclePrecisionPlan` (`Qwen.PrecisionPlan.ixx:141`), ContextProfile's loss arm @ `0.21.0-dev+34`

Found 2026-10-03 by the first ContextProfile run of `qwen-3.8-27b-fp4` (RTX 5060 Ti, files `D:\Claude\context_profile\
provisional`, `qwen_loss`). Five PG-19 books at 16384, window 64: gate 1 fails in 9 of 10 book-bands, pooled 2.835 /
2.739 at 0-8K and 3.013 / 2.685 at 8K-16K (whole book / 1024 tokens of it); book 10356 reads 3.61 / 2.49 at 8K-16K.
The 2.82-bit build on the same books passes but for 0.015 on one band -- 3.341 / 3.459 on book 10321 at 8K-16K, where
FP4 reads 3.39-3.46 -- although FP4 has more bits on every role where the two plans differ (feed-forward, DeltaNet
query/key, value/gate/output); both share FP4 attention and head and BF16 gating. Not the prefill chunk: 64, 256 and
1024 move it by 0.05. Not the exported file: quantized on load from the BF16 blob it scores identically. Not mainly
FP8 activations: `QwenPackedArtifactTests.DISABLED_DecodeAgainstPrefillAlongTheBook` (added `+34`) scores 257
targets by decode (BF16 activations) and by prefill -- book 10356 after 12288 reads 5.10 decode / 5.25 prefill for
FP4 against 2.59 / 2.59 for the 2.82-bit build; after 2048, 2.87 / 2.87 against 3.03 / 3.02. In August FP4 improved
with context on wikitext to 16K (`Qwen3.8.md` section 8, 5.686 at 16K), segments scored cold, no chat turn. Recall
at 11K and 16K is 99-100% on the same build. One difference left between the plans besides bits: the codebooks
were fitted by GPTQ on calibration text, FP4 is round to nearest. The reference that would separate format from
defect is HuggingFace BF16 and HuggingFace with the same weights rounded to FP4, layer-streamed
(`hf_qwen_layer_stream.py` scores one prompt's last token today), at those positions.

## Gemma's small kernels between the large ones are 4 to 7% of its time

`gemma` · `perf` · `mila-src` · `measured`

Raised by Todd 2026-10-01: if RoPE is fused into attention, fuse more. Measured the same day (nsys kernel time, Q4_0,
RTX 5060 Ti, a 32,512-token prompt then 256 tokens at that depth; 26B-A4B with the FP8 global cache, 12B with BF16
caches, chunk 1024). Every kernel that is neither a weight GEMM nor attention's main kernel, summed -- the ceiling on
what fusing them could save:

| | 26B-A4B | 12B |
|---|---|---|
| Prefill | 422 ms of 8.28 s, 5.1% | 995 ms of 14.14 s, 7.0% |
| Decode | 0.64 ms of 9.0 ms a token, 7.1% | 0.75 ms of 20.5 ms a token, 3.7% |

The largest parts: in decode, RMSNorm, launched about 330 times a token at about 1.4 us each (5.0% of the 26B-A4B's
decode, 2.2% of the 12B's), mostly fixed cost per launch; in prefill, the 12B's GeGLU (2.7%), whose BF16 output the
down projection's INT8 quantize (1.1%) reads straight back. So GeGLU -> quantize and RMSNorm -> quantize are the
natural pairs. On the 26B-A4B's routed branch the router's norm and `pre_feedforward_layernorm_2` both normalize the
same unnormalized residual (`Gemma.Block.ixx`, `Gemma4MoE.md` Phase 1), so one reduction could feed both, 30 layers a
token, unmeasured. RoPE is 0.3 to 0.7% everywhere (`RopeInAttention.md` 4.5). Method: ProfileModel under
`nsys profile --cuda-graph-trace=node`, decode read as the kernels after the last prefill kernel.

## Qwen's DeltaNet prefill kernel has not been measured against FlashQLA's three ideas

`qwen` · `perf` · `mila-src`

Netra Runtime's write-up of FlashQLA (`github.com/QwenLM/FlashQLA`, MIT, TileLang, Hopper only), shared by Todd
2026-10-03: forward 2.2 to 2.9 times FLA Triton 0.5.0 for one 32,768-token sequence on an H200. Three ideas:
segments of one sequence run in parallel, each warmed up only over the preceding chunks whose accumulated gate
exceeds -10 (below what BF16 holds) -- an approximation, not bit-exact; the gate's cumulative sum computed once per
chunk and one triangular inverse shared by the output and state paths; warp-specialized fusion, a TMA producer beside
three consumer warpgroups. Mila's `gated_delta_rule_chunked_kernel` (`GatedDeltaRule.cu:399`) is 1.8 s of the
2.82-bit build's 8K prefill (37 ms a layer) on a 36-SM card for 48 value heads; its efficiency and how its blocks
fill the card are unmeasured. With Qwen's other tuning, it waits for the 24 or 32 GB card.
`ModelFamilyParity.md` 8.3, Q9 lever 6.

## The FP4 prefill computes by different arithmetic than decode, and its fallback path is probably wrong

`quantization` · `models` · `mila-src` · `measured`

Three findings on `CudaLinearOp`'s compile-time switches. **W4A8:** with `kUseFp8ActivationPrefill` on (the
default), the FP4 prefill rounds activations to FP8 per token and the FP4 weights to FP8 under one scale per
tensor, 2.0e-2 to 3.6e-2 relative L2 from W4A16 per projection on Gemma 4 12B layer 0 against exact FP64; decode
computes W4A16. It buys 1.285x prefill and shipped on short-prompt token parity; whether it costs quality on text
the model is built for is unknown (`Fp8ActivationPrefill.md`). **The fallback:** switched off as an experiment on
2026-09-26, the BF16-staging prefill landed far from decode (argmax against decode 20 of 32, mean KL 1.2), though
W4A16 is what decode computes -- a probable defect, in a path no shipped build takes. **Dead branches:**
`kUseW8A16Gemm` and `kUseFusedFp4Gemm` are false, so `cuda_w8a16_gemm` and the fused FP4 GEMMs are reached by no
build. One decision covers all three: which prefill arithmetic each 4-bit policy keeps, measured, and the rest
deleted. Gemma's 12B leaves the path for Q4_0 in v0.21; Llama's FP4 packages stay on it.

## What a plan was decided against is called a `DeviceReading`, and the model list prices against total memory

`api` · `adaptors` · `mila-src`

Todd, 2026-09-23: `DeviceReading` is a poor name, kept for Phase 3. It carries one device's identity, free and
total memory and allocation granularity, taken once after the graph is built; every plan and refusal holds one,
Chat's `-p` JSON reports its fields, and `Deployment.md` 3.3 and 9 use the word. Chat's `/model` GPU FIT column
(`Chat.ModelCatalog.ixx`, `largestFittingContext`) still walks its own context ladder against TOTAL memory,
deliberately, while a plan reads FREE memory; `Deployment.md` section 8 says the column renders plans. Both wait on
`Deployment.md` 12.5, planning for a device that is described rather than read: a described device is not a
reading, and the column needs one.

## A process's second automatic load chooses a shorter context than its first

`models` · `binding` · `mila-src`

Four `GemmaModel.from_store( "gemma-4-12b-it-fp4" )` loads in one process on the RTX 4070, each deleted before the
next, chose 121856, 120832, 120832, 120832; a fresh process always chooses 121856. Stable after the first, so not
a leak: memory the first load leaves held (lazily loaded kernel modules or a library workspace are candidates,
unmeasured). Reaches Chat's `/model` switching and any Python program that loads twice. `DeviceReading::take`.

## The scratch reservation test measures the driver loading kernels, not Mila

`models` · `build` · `measured`

`ScratchReservationCudaTests.Gemma4_12B_Fp4_Context8192` (`ScratchReservation.Cuda.cpp:150`) bounds device memory
growth during generation at 64 MiB on the current device. On the RTX 4070 the reading is bimodal, 262.6 or 10.0
MiB: under `CUDA_MODULE_LOADING=LAZY` (the default) five of six fresh processes read 262.6, under `EAGER` every run
reads 10.0. So the step is the driver loading kernels inside the measured window. Predicted and reported scratch
were identical in every run. The fix warms the kernels before the window opens. Not measured: why identical runs
land in different modes, and why the 5060 Ti never shows the step.

## An out-of-range token id is an illegal memory access, not an error

`models` · `api` · `mila-src`

Found 2026-09-26 through a test bug: uninitialized ints reached the FP8 embedding gather
(`TokenEmbedding.Fp8.cu:122`) as token ids, and the CUDA context died with `cudaErrorIllegalAddress`, reported first
as a cuBLAS error in the next GEMM. Nothing between a caller's ids and the gather checks them against the
vocabulary, and a dead context takes the whole process's CUDA state with it. Checked on the host where ids enter a
prefill, not per token on the device.

## A multi-line paste into Chat becomes one turn per line

`adaptors`

Raised by Todd 2026-10-01. The loop reads input with `std::getline` (`Chat.ixx:242`), so a pasted block arrives as
one turn per line: the model answers each line before the next is read, empty lines are skipped, and a pasted line
starting with `/` runs as a command. How a user writes a multi-line message by hand is the same question.

## A user cannot measure Mila's rates on their own card

`perf` · `distribution` · `docs`

Raised by Todd 2026-10-01: promote ProfileModel to `mila-bench`. Discussed, no decision: a separate shipped binary
beside `mila-chat`, not a rename -- `mila-bench <model>` naming a store model, planned through `planDeployment`,
prefill and generation rates at a depth, `--kv-cache`, machine-readable output -- with ProfileModel kept as the
profiler and the timing core (salted prefill, decode timed first token to last) shared. A binary rather than a
`mila` verb, by the August rule. It would let a reader reproduce the published comparison rows. A user surface:
end-user prose, the wheel and the image, tests.

## Only GPT-2 trains, and training has no row in the family parity matrix

`training` · `models` · `mila-src` · `blocked`

Todd, 2026-09-26: training is first-class, and asked for a second model with full training in v0.21; shelved the
same day. Blocked on GQA backward, which does not exist (`CudaGqaOp::backward` throws). Candidates: SmolLM2-135M,
with 360M as a step up (Llama architecture, about 2.2 and 5.8 GB of mixed-precision state; recommended, licence
unverified); Qwen3-0.6B (about 9.6 GB, a new chassis); Pythia-160M (MHA, so no GQA backward, published loss
curves); GPT-2 124M at full scale. Llama 3.2 1B full training is about 20 GB, a LoRA model on 16 GB. Proposed rows
for `ModelFamilyParity.md` section 3: backward, gradient parity against PyTorch, mixed precision, optimizer,
checkpoint save and resume, loss-curve parity. The choice also decides FP32's future (`Future.md`, "Remove FP16").

## Llama and Qwen rebuild the library's gated feed-forward inside their blocks

`llama` · `qwen` · `architecture` · `mila-src`

Raised by Todd 2026-09-30 ("we've broken our symmetry"). `Components/FFN/GatedMLP` is a fused `fc_gate_up`, a gate
activation and `fc_down`, with any gate and weight policy, backward, and shared-output pooling; Gemma uses it. Llama
(`Llama.Block.ixx:27`) and Qwen (`Qwen.AttentionBlock.ixx:244`) compose the same three children inline, each with its
own slot installation and footprint code. Moving them onto `GatedMLP<Silu>` renames their tensors (`fc_gate_up` to
`mlp.fc_gate_up`), so every published Llama and Qwen package needs a load alias under the compatibility rule
(`ModelDistribution.md` *Compatibility*), as Gemma's `ffn` names got. Smaller asymmetries found the same day:
`Router` and `MixtureOfExperts` sit beside `FFN/` rather than in it, and Gemma's sublayers report their child's
`ComponentType`. Discussed, not decided: `FFN/` holds feed-forward functions, a family directory holds the sublayer.

## A drafted reply runs one fixed K, though the best K changes with the kind of text

`gemma` · `perf` · `mila-src` · `measured`

Found 2026-10-06 measuring the speculative sampler: best K is 2 to 3 on prose, 3 to 4 on code, 4 to 5 on chat,
and a K too high costs prose far more than one too low costs chat. Todd: adaptive K will be needed eventually -- a
default from the measured costs, then a heuristic that updates it. `Gemma4Mtp.md` 4.8 holds the shape and the
offline greedy simulation that prices it before anything is built.

## A Debug CPU-only MSVC build cannot compile the weights metadata module

`build` · `mila-src`

`out/build/x64-claude-cpuonly` (MSVC 14.51.36231, Debug, `MILA_ENABLE_CUDA=OFF`) fails on
`WeightsMetadata.ixx` (`import nlohmann.json;`) with `json.hpp(20512): error C2678: binary '!=': no operator found
which takes a left-hand operand of type 'nullptr'`; the Release CUDA builds compile the same file, and a Release
build of the same directory passes. Not isolated: whether Debug or CUDA-off is the variable, and whether it is the
module/header interaction in "`import Mila;` degrades the standard library" above. Found 2026-09-29.

## Three places turn a model's name into the type that loads it, each its own way

`ai` · `architecture` · `api`

The architecture-to-concrete erasure exists three times in two languages — Chat's `ModelVariant`
(`Chat.ixx:73`, ten `std::visit` sites), the binding's per-family session classes, and the
inference server's `ModelFamily` enum (`model_worker.py:38`). Each consumer also writes its own
bridge from the manifest's architecture string to a family (`familyFromArchitecture` in
`Chat.ModelCatalog.ixx`, `architecture == "gemma"` in `Mila_py.Wrappers.cpp`).

One handle and one factory in the new `Mila/AI/` library (`Direction.md` §3.2, §8 decision 2). The
architecture's *identity* — the set of names and the concrete type each resolves to — is the
library's and lives in `Mila/Src` beside the manifest reader; the handle is its one consumer. The
factory takes a deployment request rather than a device, so it is built on the planner from the
start. A network composed from components meets the handle through a C++ concept, with no
registration (§8 decision 7).

Gate: adding an architecture is an edit in one place, and decode throughput through the handle is
unchanged against the direct type.

## A model loads with every feature its loader supports, and the caller cannot choose which

`models` · `api` · `mila-src` · `breaking`

A deployment request fixes or leaves to the planner the context, the weight format and the cache format,
and nothing else. A unified Gemma 4 package can carry image and audio paths, and the 26B-A4B a vision tower
of about 411 million parameters; Gemma 4 ships a draft model; and today the converter drops every
modality tensor (`convert_weights.py`, `SKIPPED_PREFIXES`) and the loader builds what the code supports.
So a use case cannot spend the card's memory on what it needs: on the 26B-A4B, a tower's 410 MB at FP8 is
about 39K tokens of its context.

Work: the manifest declares the features a package carries (`ModelHandle.md`); the request selects from
them, each fixed or left to the planner; the planner prices the selection exactly -- `PlanEqualsBuild`
holds a selection as it holds the rest -- and refuses it naming the feature and what fits without it; a
feature not selected is not built and allocates nothing. Each feature is a component built or not, never
a template axis of the core blocks, so a selection is not a new instantiation. Raised by Todd 2026-09-27
(modality as an axis of the request) and decided 2026-10-04 ("Mila::AI can deploy a model for a specific
use case; full control of the model features loaded").

Gemma 4's drafter is not part of this: v0.21.0 ships it in each package, on by default where it pays
(`BACKLOG.md`, Gemma 4 Complete), and choosing it in a request is this entry's.

`ROADMAP.md`, Future, v0.22.0 Deployment Planning · `Mila/Specifications/Deployment.md`

## Running a model from a program means naming its concrete C++ type

`ai` · `api`

Today the entry point is `GemmaModel<Cuda, BF16>::load` or its siblings, so a program that wants to
change models changes types, and the tool loop, streaming and conversation state are the program's
to write. The finding is an absence: the application-facing object does not exist.

`Mila::AI` in module `Mila.AI` (`Direction.md` §4, §8 decision 1), a `Mila/AI/` library target that
the wheel and a `FetchContent` consumer both receive. The contract is `Direction.md` §4.2's seven
rules: small; never decides a deployment; descent through `plan()` and `model()` is part of the
contract; one virtual call per `respond`; other languages project it; a composed model is
first-class; a model is named by its store name. Gate: the ten-line program runs Gemma, Llama and
Qwen by changing only the name, and a sample creates an `AI` over a network it composed itself.

## The tool loop exists only inside Chat

`ai` · `adaptors`

Parse a call, dispatch it, return the result, continue — written once, informally, in `Chat.ixx`,
so a developer's program that wants tools rewrites it. It moves into `Mila/AI/` as the agent core
(`MilaProductFamily.md`, Native Agent Core), and Chat keeps only its human approval gate. Tools are
compiled-in functions registered on the `AI`, a callable plus a schema (`Direction.md` §8 decision
6). The autonomy policy is not part of this.

## A tool result is re-rendered and re-tokenized with the whole conversation before generation continues

`ai` · `models` · `mila-src`

Decided in v0.20 and deferred (`MilaProductFamily.md` Decided 1): a tool result's tokens are appended
to the live KV cache, with no re-render of the conversation and no re-tokenize. Today Gemma recovers
the prefill through transparent prefix reuse (`GemmaModel.ixx:364`) but still renders and tokenizes
everything; Qwen prefills from 0 every turn (`QwenModel.ixx:355`). The grammar that frames a tool
result as tokens is model-intrinsic and belongs in `Mila/Src`; the loop that decides when to splice is
the agent core's.

Gate: across a multi-turn tool session, prefill tokens per turn equal the tokens the turn added,
measured, for every family that permits prefix reuse.

## A conversation that fills its context stops, and nothing carries it forward

`ai` · `adaptors`

Chat ends a reply at the context limit ("finish: context_limit") and nothing shortens the history, so a
long tool session cannot continue. Admitted 2026-10-03 with a `Mila::AI` success
criterion added the same day (Todd): scope grown by decision, not found in passing. Two levels, both
mechanism in `Mila/AI/`: reasoning from earlier turns dropped at turn boundaries -- kept within a turn,
which is the opposite of the failure in "Gemma loses its own reasoning between tool calls in a turn" --
and text compaction, the history summarized into a fresh context with instructions kept verbatim,
reusing the system prompt's cached prefix. It triggers at the configuration's reliable depth, which
ContextProfile measures, so it needs that tool's first profiles; Qwen resumes from the position it
saved at the end of each prompt (`savePosition`, `+33`). Compaction in the cache itself (deleting spans in place) is research, outside
this item (`.internal/Ideas/AgentStreams.md`).

`ROADMAP.md`, Future, v0.22.0 Mila::AI · `Mila/Specifications/ContextProfile.md`

## A program cannot ask for a model set up for its use case

`ai` · `api`

Choosing a model's features -- an image path, a draft model, a context and a cache format -- is a
decision every program would make again, and the right defaults are measurements, not guesses: what
context a model is reliable to (its context profile), what a draft model buys (its measured speedup).
`Mila::AI` names use cases -- a coding agent, a vision assistant, a long-document reader -- as
selections over the deployment request, each overridable before creation. A preset is a convenience,
never a second path: it produces an ordinary request, and the plan it produced reads like any other.

The finding is an absence; nothing selects features yet (the entry under Deployment Planning).

`ROADMAP.md`, Future, v0.22.0 Mila::AI

## A program that skips Mila's initialization is told its GPU reports no memory

`ai` · `api` · `mila-src`

`Mila::initialize` (`Mila.ixx:443`) sets the log sink and the random seed, which have defaults, and discovers the
devices by constructing `DeviceRegistrar`, which nothing else does. A program that skips it fails silently
everywhere but `DeviceRegistry::getDevice`: `getDeviceCount` is `noexcept` and returns 0, and `DeviceReading::take`
swallows why it could not read the device, so the planner refuses with `DeviceDoesNotReportMemory`. From Python,
`mila.GemmaModel.from_store( name, "auto", 1 )` without `mila.initialize()` says the device "does not report its
free memory" and advises passing a number, on both cards; with `initialize` first the same call opens at 262144.
Found by `Tools/ContextProfile` (2026-10-03) and through the binding (`Mila_py.Wrappers.cpp:447`).

Discovery is a ceremony because of an import cycle: `CudaDevice` imports `DeviceRegistry` to register into it, so
the registry cannot import the registrar. The shape discussed with Todd: the registry runs discovery once, on first
request, through a function the registrar module installs, leaving `initialize` as configuration only. Stale Doxygen
still names the retired operation registry as part of it (`Gelu.ixx:58`, `Residual.ixx:8`).

Gate: the ten-line program and the Python QuickStart run with no initialization call, and a device that cannot be
read is refused naming why.

`ROADMAP.md`, Future, v0.22.0 Mila::AI and A Developer Can Start

## Chat and the inference server each hold code that knows which model they are running

`adaptors` · `build` · `breaking`

Rebuilt as consumers of `Mila::AI`, reaching anything beneath it only through `model()`: Chat keeps
terminal rendering and the approval gate, the server keeps the wire shapes and per-request
statelessness. Both leave `Mila/Adaptors/` for `Mila/Applications/Chat` and
`Mila/Applications/Server` (`Direction.md` §8 decision 4), with the CMake option, the wheel and image
paths, and the `adaptors` tag in `Tags.md` following. Gate: neither contains model-specific code, and
the Codex CLI and Claude Code CLI tool flows still pass unchanged.

`routes/chat.py` and `routes/completions.py` are not moved:
nothing registers them, and `chat.py` carries its own request schema and prompt assembly.

## The inference server chooses each model's loader and grammar by a family enum of its own

`adaptors` · `binding`

`model_worker.py` maps `ModelFamily` to a session class (`:38-41`), branches on it for stop markers
(`:56`, `:59`) and for tool support (`:168`), and `/v1/models` does the same. Three latent `else means
llama` sites in this shape were fixed at `rc.1+31`, each correct only while there were exactly two
families. It serves the same three families Chat runs today, so the gap is the second copy of the
erasure rather than a model it refuses.

GPT-2 is not in it, and not by oversight here: it is a base model, Chat refuses base models by
decision, and the binding has no GPT-2 session (`model_worker.py:71`). Whether the server serves a
base model is not this release's question. Gate: the enum is gone and the server reads what it needs
from the handle.

## The Python binding carries one session class per family

`binding` · `api`

The per-family session types under `Mila/Bindings/` are the second of the three bridges, and the one
a Python consumer actually meets. They are replaced by `mila.AI`, projecting the same contract and the
same plan, without the binding gaining a component-level surface — it is consumer-blind by design and
stays that way.

The gate is that adding a family adds no binding type. The finding is a duplication rather than a
defect at one line, so it has no single anchor.

## MIS tool calling beyond the three flows the release names

`gemma` · `adaptors`

N sequential distinct tool calls within one turn, and channel-content parser polish. Moved from the
v0.20 backlog at `rc.1+21`: the release criterion names plain-chat, single-tool and
tool-result-resume only.

## Neither QuickStart shows a program putting a model to work

`docs` · `binding`

`Samples/QuickStart/Cpp` loads a typed model and `Samples/QuickStart/Python` opens a per-family
session; neither creates an object, calls a tool or shows what was decided about the deployment. Both
are published surfaces the website's Get Started tabs link to.

The C++ one becomes a `FetchContent` project that creates an `AI`, calls a tool and prints the plan;
the Python one moves to `mila.AI`. Each keeps a path that builds a network from components
(`Direction.md` §7). Gate: both run from a clean machine with only the documented prerequisites.

One risk to the C++ one, not yet reproduced: where CMake's feature table lacks `cxx_std_23` (Clang 21.x), the
exported `Mila` target falls back to advertising `cxx_std_20` (`Mila/CMakeLists.txt:71-75`), and a consumer
compiles Mila's module units under it, though they use C++23 library facilities (`std::ranges::fold_left`,
`Tensor.ixx:806`). A `FetchContent` consumer receives the same interface features, so the QuickStart is checked
on that compiler before it is declared clean.

## Every public surface describes Mila as a reference implementation first

`docs`

`README.md`, the website under `Web/` and `getting-started.md` lead with the v0.20 message, and
several use "adaptor". They move to the trait — a library for developers to harness intelligence — in
this release and not before, which means written against what the tag actually ships
(`Direction.md` §5.7; `.internal/Marketing/Positioning.md` is superseded by it).

The landing page's four claims -- Explicit, Validated, Fast, and the type is the configuration -- each gain a
link to a page that explains the claim fully. Each claim's one line still stands without the click, and each
link names its topic, not "Read more" alone. Explicit and the type is the configuration go to their sections of
the Design page (`Web/content/docs.md`), expanded: the decode recording explained as a cache of the explicit
calls, with its off switch (`DecodeGraph.md`), and the typed configuration of components told apart from a load
mapping a package's declared format to its type -- the section says today that the converter always writes BF16
and quantization happens on load, which published packages no longer do. Validated gets a page of its own:
what is checked, against what, at which precision and length, for every model including Qwen 3.8 -- replacing a
"token-for-token" claim that is wider than what the parity tests check. Fast's page belongs to the llama.cpp
entry below. `hugo.toml`'s comment that the claims carry no links is replaced with the reason these do. Written
late in the cycle, on the numbers the tag ships. The blog post `mis-with-claude-code-and-codex.md` (2026-05-14)
announces under "What's Coming: Tool Calling" a pybind11 `ToolCallParser` and a `MILA_TOOL_CALLING_ENABLED` flag that
were never built; the same pass gives it a dated note pointing at the tool calling that shipped. Gate: no public surface describes Mila as a reference
implementation first, or uses "adaptor", and every landing-page claim links to a page that states exactly what
backs it.

## The smaller dense Qwen members were never built

`qwen` · `mila-src`

The family shipped as the 3.8-27B hybrid alone, so "Qwen 3.8" names one model rather than a family.
The dense members reuse the Llama blocks rather than the DeltaNet chassis, which is what makes them
cheap next to everything already landed. Built after the handle, it is the first model added through
it, and so the first test of "a new architecture is added in one place".

The finding is an absence — there is no partial implementation to point at. Gate: a dense member
decodes token-for-token against HuggingFace at BF16 and FP8.
