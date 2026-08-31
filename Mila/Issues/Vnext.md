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

## One typed model handle + factory

`architecture` · `mila-src` · `gate`

The architecture-to-concrete erasure exists three times in two languages — Chat's `ModelVariant`,
the binding's `*Session` classes, MIS's `ModelFamily` — which is why GPT-2 is missing from MIS.

Lands in the runtime-adjacent native agent core; sequencing in `MilaProductFamily.md` Open
Decision 2. ROADMAP already calls it the first work after the v0.20 tag and a precondition for
every model entry below it.

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
`RopeCacheRegistry` keys on (theta, max_seq_len, head_dim) and only the first component to ask for
a given key allocates, so the answer depends on the order the test builds things in. Deduplication
at transformer level is already in place.

## GPT-2 cannot report what it would allocate

`gpt` · `mila-src`

Every other family answers "will this fit on my card" before loading anything; GPT-2 does not
implement `getRequiredMemory`, so installing `gpt2-small` and asking Chat about it gets silence.
Nine of its components still throw the base class's by-design "not implemented" error — Gelu,
MultiHeadAttention, Lpe, GatedMLP, MLP, SoftmaxCrossEntropy, LayerNorm, Softmax, GptBlock — and the
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

## The Gemma parity script compares two different precisions and calls it parity

`gemma`

`gemma_greedy_parity.py:70` loads Mila through the binding's FP4 default and diffs it against a
BF16 HuggingFace reference, so any divergence it reports mixes quantization error with a real
defect and a clean run proves less than it appears to. `from_pretrained` now takes `quantization=`,
so the honest comparison is one argument away — on a card that can hold a BF16 12B. Either way the
script should state which precision it ran.

Full path: `Mila/Tools/Converters/Gemma/gemma_4_BF16/gemma_greedy_parity.py`.

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

## Qwen answers in one block after a long silence

`qwen` · `adaptors`

`FamilyTraits::streaming_capable` is false for Qwen (`Chat.FamilyTraits.ixx`), because the harness
routes tokens by Gemma's four control-token ids and nothing else has them. Qwen has one marker pair,
`<think>`/`</think>`, which is enough to separate reasoning from answer; the per-token router has
simply not been written for it.

Not a gap against any other model — Llama and GPT-2 are buffered too, and Gemma is the only family
that streams. It matters most on Qwen because a 27B is the longest wait to sit through with nothing
on screen.

## `gemma_protocol.py` is dead and can be deleted

`gemma` · `binding`

Its 856 lines are superseded by `Gemma.Protocol.ixx` plus `gemma_bridge.py`, nothing imports it, and
it carries a header saying so. Kept on disk under the retire-don't-delete rule, which is the correct
state for now; removing it is a one-file deletion whenever the reconciled grammar has been driven
long enough to be sure.

## Chat configuration phase 7 — the two `ModelRecord` fields

`adaptors` · `distribution`

Phases 1 through 5 of the layered resolution have landed and Chat's configuration works. What is
left is the last phase, which reaches into Model Distribution for two fields on `ModelRecord`.
Design and phasing are in `Mila/Specifications/ChatConfiguration.md`.

## `ModelSerialization.md` Phase 7 describes shipped work as unwritten

`docs` · `distribution`

The distribution path exists end to end — `savePretrained` (`LanguageModel.ixx:116`), the
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

`Mila/Bindings/CMakeLists.txt:95` stages it with `copy_if_different` off
`add_custom_command(TARGET MilaPy POST_BUILD)`, which runs only when `MilaPy` relinks — so a change
to `__init__.py` and nothing else leaves `<build dir>/python/mila/` holding the old copy, and a
sample fails with a missing attribute. `add_custom_command(OUTPUT ...)` with `DEPENDS` on the source
is the fix.

Local development only: the wheel packages `__init__.py` from `Bindings/Package/src/mila/`, where
the file is tracked, so nothing stale can reach a published wheel.

## Qwen's perplexity gate has only been run to 16K

`qwen` · `measured`

From 8K to 16K the FP4 oracle improves 7.2% while the 2.82-bit plan improves only 3.4%, so the
quantized arm captures about half the benefit of the extra context — the compounding signature the
recurrent layers make plausible. Not release work, because nothing claims a context above 16K: the
model card stops there and records the ratio as flat from 1K.

It becomes release work the moment a longer context is advertised. The table and caveats are in
`Qwen3.8.md` §8 item 9; `DISABLED_QualityGateAcrossContextLengths` is the harness.

## The head's two paths disagree in the third decimal, so perplexity must fix the width

`qwen` · `measured`

Same weights, same corpus: width 1 (the decode matvec) and width 64 (the W4A8-FP8 GEMM) do not
produce identical numbers. Small, but head width is part of the measurement protocol rather than a
free performance knob, so both arms of a quantization comparison have to use the same one.

Probably already recorded at `Qwen3.8.md:509` and `:546` — verify, and if so this entry is a
duplicate and should be deleted rather than worked.

## Two models cannot share an execution context

`api` · `mila-src`

`fromPretrained` takes a `DeviceId` (`GemmaModel.ixx:130`), not an `IExecutionContext`, so two
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

## The samples are not built in CI

`ci` · `build`

Only the tests build today, so a sample can stop compiling without anything noticing — and the
QuickStart samples are a published surface the website's Get Started tabs link to.

## Public Doxygen still describes a world that was refactored away

`docs`

Mechanical drift — `@file`, `@param`, `@tparam` names disagreeing with signatures — is what
Doxygen's own warnings catch and is v0.20 work. This is the half no tool can see: prose that reads
correctly and describes the pre-`OperationTraits` design, on the API a consumer calls.

There is no bounded worklist, which is why it is not release work. It is cleared opportunistically —
a file's prose gets fixed while the file is already open for another reason.

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

## The wider `Tensors/` tree has no coverage beyond `Tensor` itself

`architecture`

Re-greening the authored tensor suite is v0.20 work and is committed. This is the layer underneath
it, which the old suite never covered at all: `TensorBuffer`, the `TensorDataType*` maps,
`Partitioning`, and `Serialization` — plus the `TensorOps.Transfer` device split. New coverage
rather than revival, which is why it is not in the release.

Eight `REVIEW:` markers already name the exact contracts to pin; `Specifications/Testing.Tensors.md`
carries them.

## Llama parity has no automated regression test

`llama` · `ci`

Llama's agreement with the HuggingFace reference has been established by hand, but nothing holds it
— so the next change to the load path or the attention kernels can break it silently. Gemma has
`GemmaModel.Parity.Cuda.cpp` as the template; Qwen has no equivalent because a BF16 27B reference
fits no card here, which makes Llama the family where a permanent token-for-token test is actually
affordable.
