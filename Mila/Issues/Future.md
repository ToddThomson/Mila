# Future

Next-cycle work: real, and for a later release. Flat and coarse **by design** — detailed tasking
happens only when an item promotes into a release, and elaborating it here is work spent on a plan
that will be rewritten before it is used.

Moved out of `BACKLOG.md` so that file means exactly one thing: work committed to the release in
flight. Triage flow and categories are in [README.md](README.md); the tag set is
[Tags.md](Tags.md).

---

## One typed model handle + factory

`architecture` · `mila-src` · `gate` · `next`

The architecture-to-concrete erasure exists three times in two languages — Chat's `ModelVariant`,
the binding's `*Session` classes, MIS's `ModelFamily` — which is why GPT-2 is missing from MIS.

Lands in the runtime-adjacent native agent core; sequencing in `MilaProductFamily.md` Open
Decision 2. **After the v0.20 tag, before the chassis expansion below.**

## The library should own architectural identity

`architecture` · `distribution` · `mila-src`

The set of architectures is the set of model classes `Mila/Src` implements, held today as a
compile-time type and an unvalidated manifest string with nothing connecting them — so each
consumer writes its own bridge (`familyFromArchitecture` in `Chat.ModelCatalog.ixx:159`,
`architecture == "gemma"` at `Mila_py.Wrappers.cpp:413`).

Home is `Distribution`, beside the manifest reader, not `Dnn`. The library owns the identity only;
traits merely keyed on it stay with the consumer they describe.
[[project_architecture_identity_ownership]]

## Qwen 3 — the dense members

`models` · `quantization` · `mila-src`

The dense decoder, thinking-mode suppression, model-agnostic tool calling, and FP8 KV cache. The
`OperationTraits<GqaOp, Cuda, BF16, PerChannelKvFp8<>>` specialization lands here.

## Qwen 3.8 FP4 — FP8 the embedding table

`models` · `quantization` · `mila-src`

`QwenOraclePrecisionPlan::EmbeddingTable` is `NoWeightQuant` and host-resident, so every decode step
gathers a row over PCIe. At `PerChannelFp8<>` the 1.271 B table halves to ~1.19 GiB and sits beside
the FP4 body inside a 15.93 GiB card.

Gemma 4's D4 Design B is the precedent, and `TokenEmbedding` already accepts a per-channel table
policy. `Qwen.PrecisionPlan.ixx:153`

## Widen the quality harness beyond the shipped gate

`models` · `quantization` · `mila-src`

The v0.20 bar is the perplexity ratio alone; everything around it is thin. Two halves.

The Phase 5 prompt set is six prompts with one code prompt and no tool-call, multi-turn or
long-context class — too few to put a threshold on the mean the way 1.25 sits on the perplexity
ratio. Widening is two loads and ~18 s per six (`DISABLED_DivergenceAgainstTheOracle`).

And only Qwen can be scored at all: Gemma, Llama and GPT-2 build their heads at T=1 with no width
parameter (`Gemma.ixx:279`, `Llama.ixx:239`, `GptTransformer.ixx:365`), so `scoreTokens` throws the
base's `logic_error`. Each needs a head width in its config plus the window loop.

## `PerGroupFp4` should carry FP16 scales, not FP32

`quantization` · `mila-src` · `breaking`

`Policies.ixx:112` sets `kScaleDtype = FP32` — 4.25 bits/weight. `Qwen3.8.md` §5 budgets 4.125 and
`formats.py`'s `fake_fp4_e2m1` simulates the FP16 rounding, so the packer's simulated damage is not
the damage the runtime inflicts.

Worth 0.125 bits/weight to Gemma and Llama independently. Changes the wire format of every
published FP4 model, so it re-quantizes and re-publishes them — which is why it waits.

## A normative wire format for FP4 and FP8

`quantization` · `ci` · `gate`

`cuda_quantize_fp4_per_group` is the sole place the nibble order and `/6.0f` scale convention are
written down, so every published model's byte layout is unreadable from CPU-only CI and unstatable
in the spec.

Copy `CodebookPacking.ixx`: normative layout, host codec, and `--emit-fixture` holding kernel and
Python packer to it in both directions. Add `Fp4Packing.ixx` / `Fp8Packing.ixx`. That fixture is
the only reason `packing.py`'s `quantize_fp4_e2m1` / `dequantize_fp4_e2m1` are retained, and it is
the dependency **Retire quantize-on-load** names below.

## Re-measure the FP4/FP8 rates before quoting any of them again

`perf` · `docs`

Every such number in the tree is Ada-at-x16, and the 4070 now sits on a chipset Gen4 x4 link with
no baseline captured before the move. The caveat is recorded with the numbers in `Qwen3.8.md` §8;
this is the re-run.

## Architecture / MoE

`architecture` · `models` · `mila-src`

The presumptive post-v0.20 tentpole: one router chassis unlocks Gemma 26B-A4B, Qwen3-30B-A3B and
gpt-oss-20b. [[project_moe_tentpole_direction]]

## Gemma 4 MTP

`models` · `mila-src`

The self-speculative drafter, sequenced ahead of MoE.

**The MTP head cannot be gated against HuggingFace at all.** transformers 5.12.1 declares
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` and implements no MTP class, so the parity
harness has nothing to compare against and the wiring is read from tensor shapes and family
convention. The converter skips the tensors today.

## Ministral

`models` · `mila-src`

SWA transformer; reuses the Llama foundation, Qwen 3 tool calling, and the Gemma 4 SWA mask +
bounded-KV ring.

## v0.20 library-frozen tails

`api` · `mila-src`

The Generation API surface tail (`SamplerConfig` rename, Llama/Gpt seedable sampling, eager
sampler, config-accessor propagation, `contextLength()` hoist), the Sample-API device-sampler
migration for Llama/Gpt, and the Optimizer-dispatch migration onto `OperationTraits`.

All `Mila/Src`, which is why they wait. Adaptor work does not.

## Model serialization

`api` · `mila-src`

The remaining checkpoint round-trip and distribution phases. Design and phase plan in
`Specifications/ModelSerialization.md`.

## Retire quantize-on-load — one load shape for every policy

`quantization` · `api` · `mila-src` · `breaking` · `blocked`

`Linear::loadParameter` refuses a compute-precision blob, uploads packed bytes, binds, derives; the
dtype sniff at `Linear.ixx:601` and `CudaLinearOp::quantize()` go, and FP8/FP4 fitting joins the
sub-4-bit fitter in `Tools/Quantization` — one producer for every format, and model production
stops needing a GPU. The codebook path is already this shape (`:574`).

Blocked on the FP4/FP8 codecs above. Takes Chat's load-time quantization keyword with it. An API
change to `Mila/Src`, which is why it waits. [[project_quantization_offline]]

## Python binding — numeric access, not component access

`binding`

Add a session-level `forward()` returning logits, plus final hidden states, to
`LlamaSession`/`GemmaSession`; from Python a parity run can compare token ids and nothing else
today.

Component, tensor and training bindings are ruled out: `TDeviceType x TPrecision x
TWeightQuantization` is erased only at the session PIMPL. `Mila_py.Wrappers.ixx:362`

## API Coherence

`api` · `mila-src` · `breaking`

The pre-1.0 consistency pass, and the precursor to any API-stability promise. Three named items.

**`loadModel`/`saveModel` and `loadCheckpoint`/`saveCheckpoint`** — verb plus what you get, both
directions. "Pretrained" is relative to a fine-tuning stage Mila does not have and is doubly wrong
on the write side; "artifact" is build vocabulary for a file that is simply a model; `from` names
the *source* form, so `fromCheckpoint` earns it and `fromModel` cannot. Document the distinction: a
checkpoint carries epoch and loss as one of a series, a model is terminal. One wrinkle:
`Network::load( archive, mode )` restores into an existing graph. **The methods are the small
half** — `kArtifactMinimumMilaVersion`, `ModelDistribution.md`, both model cards, `from_pretrained`,
MIS and the samples all speak the old vocabulary. Sequence with the `ExportArtifact` rename and the
binding's `quantize_fp8` fix.

**`getStorageSize` exists three times** — `Mila::Dnn::detail::getStorageSize` (`Tensor.ixx:81`,
carrying a `REVIEW:` that already asks why), `Detail::getStorageSize` (`TensorBuffer.ixx:221`) and
`Mila::Dnn::storageBytes` (`Component.MemoryStats.ixx`), the two namespaces differing only in case.
Blocked on `Tensor.ixx` not being able to import `Dnn.Component` without a cycle.

**`save_` is public on `Component` and protected on `CompositeComponent`** (`Component.ixx:407`), so
one virtual's accessibility depends on the static type held, and a caller with a concrete composite
cannot invoke it — C2248, worked around by `exposeSave()` in `Tests/Dnn/Core/CompositeComponent.cpp`.

**`Component` carries training's bookkeeping without training's act.** `zeroGradients()` (`:362`)
and `getGradients()` (`:718`) are on the base and `backward` is not, so `Linear::getGradients()`
returning empty (`Linear.ixx:524`) leaves "inference-only" and "nothing accumulated yet"
indistinguishable. `TransformerApiReadiness.md` item 8 argues this at network level only.

**`AttentionOutputGate` has two callers and one of them is not attention.** `QwenDeltaNetBlock` uses
it for the mixer's output gate. The component is mechanically generic (`out = TGate(gate) * value`);
the name is not. Rename, or accept the mismatch deliberately. `Components/Attention/OutputGate/`

## Warnings-as-errors ratchet

`build` · `ci` · `blocked`

Enforce in **CI only**, never locally; ratchet on the count *not increasing* before demanding zero;
**MSVC first**, since `/WX` across three compilers means the union of three opinions must be zero.
Dormant-but-retained code warns by nature — suppress per-file in CMake pointing at the owning task,
never with `#pragma warning` in module code. Land it **after** v0.20 ships.

Blocked on isolating third-party warnings first: `/external:I` + `/external:W0` (`-isystem` for
Clang/GCC), targeting third-party header text pulled into Mila's own TUs rather than their sources
— `/W4` at `Mila/CMakeLists.txt:87` is `PRIVATE` and never reached them. Two frictions: those
headers enter through module global module fragments, and `/external:` does nothing for nvcc
diagnostics.

`GroupedQueryAttention.ixx:216`'s C4702 is the case that decides the shape. It is left deliberately —
it self-clears when the GQA training path is built, where a suppression would have to be remembered.
A blanket `/WX` forces it silent; escalating only the defect-class codes leaves it visible.

## `#ifdef` inside module purviews

`architecture` · `build` · `mila-src`

A guard in a module purview rather than the global module fragment, in two shapes.
`ExecutionContextFactory.ixx:30-33` puts `#ifdef MILA_HAS_CUDA` in the exported
`createExecutionContext` body, where the CUDA arm belongs in a partition or a CMake-selected unit.

`Chat.Footprint.ixx:24-28` is the same class and is one of the tree's two live warnings — C5202, a
GMF admitting only preprocessor directives while two `inline constexpr bool` definitions sit in it.
The `#ifdef` cannot simply move to the body either, so a CMake-supplied definition is the shape that
fits.

Five more files hand-roll the staging memory resource that `DeviceTypeTraits` now carries, each
writing `#ifdef MILA_HAS_CUDA` plus a `conditional_t` that is exactly `host_staging_memory_resource`
— `Gemma.Block.ixx:820`, `Gemma.ixx:527`, `Llama.ixx:484`, `GptTransformer.ixx:615`,
`GemmaModel.ixx:110` and `LlamaModel.ixx`. Converting them removes six preprocessor blocks from
module purviews. [[feedback_no_ifdef_in_modules]]

## Materializing softmax kernels store and reload the score row

`perf` · `mila-src`

`Gqa.Prefill.Fp32.cu:67`/`:73` and the FP32 common softmax park unnormalized exponentials in
`att_row` and reload them to normalize — a wasted round trip over the widest prefill transient. The
BF16 kernels recompute `expf` on the store pass; in FP32 that buys no accuracy, so measure first.

Nothing stops a new BF16 kernel reacquiring it either: the four materializing sites were found by
grep rather than by a failing test, and one was written three months after the decode path was
fixed. Masking differs per site — causal, causal+padded, causal+window, ring-slot — so a shared
helper buys little; a test pinning single-narrowing against an FP32 oracle would.

## No build or CI step runs `compute-sanitizer`

`ci` · `mila-src`

`Mila/Src` carries zero error checks across its 110 kernel launch sites — defensible, since an
in-kernel fault is async, but it makes an out-of-bounds access that changes no output invisible. The
W4A8 staging defect survived the full passing suite, and one sanitizer run named it.

A pass over a targeted CUDA subset (`compute-sanitizer --tool memcheck`, roughly 10x slowdown).

## The Llama chassis never received Gemma's memory gates

`models` · `quantization` · `mila-src`

8B FP4 therefore costs more than 12B FP4: the embedding and `lm_head` ignore the weight-quantization
policy and are untied. Three fixes, each mirroring Gemma — pass the policy to `TokenEmbedding`
(`Llama.ixx:117`) and `lm_head` (`:119`), and implement tying when `tie_word_embeddings` is set.

Llama's `preatt`/`att` also span the full context where Gemma's ring does not. Separate, and
dominant at long context. [[project_llama_chassis_memory_gates]]

## `FamilyTraits::default_context` is a compiled-in guess

`adaptors`

`Chat.FamilyTraits.ixx:61` hard-codes 512 for Gemma, 4096 for Llama and 1024 for GPT-2, while
`resolveAutomaticContext()` derives the answer in milliseconds and no VRAM.

Keep the constant only as the no-CUDA fallback, which is already the role `main.cpp:863` passes it
in.

## Chat's model-and-session commands have no tests

`adaptors`

`/context`, `/set`, `/thinking` and the `/model` subcommands are the first Chat commands that
rebuild a model, refuse an input on derived arithmetic, or resolve a name case-insensitively.

Cover the context floor, the ladder's fit search, and `resolveStoredName` — that last matters most,
store lookup being a path lookup and so case-sensitive on Linux and not on Windows.

A full manual QA pass is owed alongside them: an uninstalled name through `/model <name>` (the only
hub-fetch path), `resolveStoredName`'s ambiguity refusal, `/context` below the derived floor, `/set`
bounds rejection, and the `unknown` GPU FIT verdict. **Watch for:** `/model <name>` no longer loads,
and the break is silent.

## `/models` measures a per-model context and throws it away

`adaptors`

`LadderFit::context_length` holds the largest fitting rung and the column does not print it, because
the ladder tests memory alone — its top rung claimed `128K` for Gemma where the session runs 56320.

A `CONTEXT` column needs the chunk test on the ladder
(`FootprintPrediction::prefill.isBudgetConstrained()`) and finer rungs; at 1-2 ms per probe a
six-rung ladder is ~12 ms per row. Only worth it if users pick models by context.

## A per-row disk figure should be reclaimable bytes

`distribution`

If one ever returns: the blobs that model alone references. That is what deciding-what-to-delete
wants, and prune's mark-and-sweep already computes the refcount — it is simply not exposed as a
per-model query.

## GQA's standalone `forward()` paths are unverified

`mila-src`

Component-level Gemma/Llama attention has no independent correctness oracle, and
`GroupedQueryAttention.ixx:177` returns an un-computed `output_view_` on an unreached branch. This
is the precondition for retiring the legacy GQA path, and it clears the deliberate C4702 with it.
See `Specifications/GqaMemory.md`.

The non-KV-cache fallback at `:299` is stale in the same way — recorded as needing a correctness
review, with the shape derivation commented out beneath it.

## `CudaMhaOp.ixx:433` initializes `active_max_seq_len_ = T_` with the reason unrecorded

`mila-src`

Confirm against the two-phase KV-cache contract: prefill takes the full sequence, decode runs at
`outer_size == 1`.

## `cuda_fp4a16_gemm` falls through to a silent no-op on an unsupported group size

`quantization` · `mila-src`

Both `cuda_fp4a16_gemm` and `cuda_fp4_dequantize_to_bf16` switch on `group_size` with
`default: break` (`CudaW4A16Gemm.cu:398`, `:428`), so a size outside {64, 128} launches nothing and
leaves the staging buffer holding the previous strip — wrong logits, no error.

`CodebookDequantize.cu` now throws; make these match. Only reachable by adding a `PerGroupFp4<N>`
policy, which is why it has never fired.

## CI has three cost and reliability gaps

`ci`

**No `timeout-minutes`, so a hang costs six hours.** It has recurred in two different jobs on one
day — once stalling in `Run CPU test suite` at 75+ min against a 14m29s baseline, once in `Build` on
the pybind11 wrapper TU that compiled in 3m43s on the identical tree in a parallel run. A re-run is
the only remedy available, and against a normal ~45-minute round trip a bound near 60 turns a repeat
into a legible failure. `.github/workflows/build-pipeline.yml`

**The packaging gate should be its own job that configures but does not build.** It does not consume
the parent build at all — it passes `MILA_SOURCE_DIR=${CMAKE_SOURCE_DIR}` and compiles Mila from
scratch under `_deps/mila-build/`, needing only that CMake has configured. Today the release PR
builds Mila for ~45 min and the gate builds it again, in series.
`Mila/Tests/Packaging/CMakeLists.txt:43`

**A `dev` push and an open PR for the same SHA run the whole pipeline twice**, and the redundant one
blocks the merge. The PR run is a strict superset — same tree, plus the packaging gates
`build-pipeline.yml:114` skips on a `dev` push — but both report the same check names on the same
SHA. Suppress the push run, and comment exactly when each job runs: an `if:` in this same file once
hid a broken packaging gate for 32 commits, and a first pass proposed suppressing the wrong run.

## Add Python 3.14 once 3.12 is proven

`ci` · `build`

It is the interpreter Ubuntu 26.04 ships and therefore the dev container's `python3`, which is why
`Docker/build-mis.sh` still restates MIS's dependency list and installs MIS with
`--ignore-requires-python`. Only a 3.14 wheel retires that duplication.

Needs `uv python install 3.14` on Windows and one deadsnakes line in `Dockerfile.wheel`.

## Broaden CI compiler coverage toward the supported matrix

`ci` · `build`

Adds MSVC and GCC 16 to clang-21 — the second module-compiler oracle. Moved here by the
Linux/clang item's own conclusion.

## Nothing cites `scripts/dockerhub/`

`ci` · `docs`

Four files remain, in two channel groups. `RELEASING.md` and `wheel-cleanroom.yml` both reach into
`pypi/`, but neither `README.md`, `getting-started.md` nor `Docker/README.md` names the image half.
`build-runtime-image.sh` carries the published arch list, so it is undiscoverable knowledge until
the publish script absorbs it.

## The binding's staged extensions accumulate in the source tree

`build` · `binding`

MilaPy's POST_BUILD writes `_mila*.so`/`_mila*.pyd` into `Mila/Bindings/Package/src/mila/`, so a
checkout collects one per interpreter and platform ever built — all untracked, and all swept into a
Docker build context until `.dockerignore` excluded them.

Clean stale ones on build, or stage outside the source tree.

## Stage model weights off the Windows bind mount for the container

`perf` · `build`

For native disk speed.

## Module import hygiene

`build` · `mila-src`

Phase 0 exact-duplicate dedup, Phase 1 candidate report, Phase 2 compiler-verified removal
(Clang/GCC, not MSVC), plus domain-qualifying the generic single-segment module names
(`Core`/`Utils`/`Components`/`Profiling` → `Dnn.*`).

## The `fopen` → `<fstream>` conversion is still available in three modules

`build` · `mila-src`

`SafeTensors.ixx` and `TokenSequenceLoader.ixx` are straight swaps and the library's only source of
C4996. **`PretrainedReader.ixx` is not**: it deliberately uses positioned `ReadFile`/`pread`
alongside the mapping, because faulting a large model through the mapped view throttles below disk
bandwidth — that one needs the exemption.

Clearing the first two unblocks the warnings ratchet above.

## ProgressReporter

`api` · `mila-src`

An injected per-operation progress facility for long-lived ops — BPE vocab training,
`PretrainedReader` load, load-time quantization. `BpeVocabulary.ixx:624` is the concrete call site:
an every-100-merges elapsed-time print asking to become an async callback.

## The BPE ASCII-fallback warning fires for every Llama and GPT-2 session

`tokenizer` · `adaptors` · `mila-src`

`BpeTokenizer.ixx:378` warns at construction under `std::call_once`, so an evaluator typing English
is told about a path they never take, as the first thing after the welcome box.

The claim is true and should not be softened — it should be timely, warning on first non-ASCII
input. Touches `Mila/Src` and needs agreement; a cheaper interim is one console line with the detail
in docs.

## If C1128 recurs, move `/bigobj` project-wide

`build`

It is per-target on `ChatApp` today. If `MilaTests`, `ProfileModel` or `ExportArtifact` hit it,
switch to one `add_compile_options`. **Todd's call** — it touches every target's flags.

## `CLAUDE.md` documents the retired Chat alias set

`docs`

`gpt2`, `llama-1b`, `llama-3b`, `llama-8b`, plus the `llama31`/`llama32` filename-prefix rule. There
is no catalogue and no filename construction — `/model` takes a store name.

Agent-facing rather than user-facing, which is why it rotted unnoticed, but it actively misdirects
work.

## `ExportArtifact` needs a name, subcommands, and its store verbs handed back to `mila`

`distribution` · `build`

Its install/rename/validate verbs duplicate the store tool, and Chat (`Chat.ModelCatalog.ixx:387`)
and MIS (`model_worker.py:90`) point users at the wrong one. The nine modes should be subcommands,
since `--package` is both a mode and an option of one (`ExportArtifact.cpp:212`). Seven touch no GPU,
yet `Tools/CMakeLists.txt:10` gates the whole binary behind `MILA_ENABLE_CUDA`. **Name is Todd's
call**, and `mila-compress` is not it.

Naming drift inside the tool travels with the rename: `--emit-manifest` is a synonym for
`--package <dir>` differing only in its default directory (`:394`);
`ExportOptions`/`InstallRequest`/`PackageArtifactRequest` are three suffixes for one role; and
`weightQuantizationVariantName` (`:103`) sits one character from `Src`'s `weightQuantizationName`
while returning `cb2-3` where that returns `codebook` — a Qwen constant behind a generic name.

## `ModelCards/TEMPLATE.md` does not exist

`docs` · `distribution`

`Publishing/README.md:40` says "the two Llama cards are the template", and template-by-example is
what propagated one meaningless sentence into all six cards verbatim. The end-user prose rules are
written; the template comes next, then the card rewrites. [[feedback_end_user_prose_boundary]]

## `/models --online` answers SUPPORT but cannot answer FIT

`distribution`

How much context fits is unanswerable because `ModelManifest` (`ModelManifest.ixx:53`) carries no
geometry. Two ways in, different owners: a `Range` read of the safetensors header so the online row
runs the same `largestFittingContext` as the installed one — blocked on the footprint path taking a
path rather than a byte range — or geometry fields in the manifest, which is phase 7. Never an
estimate.

It also costs one GET per listed model: invisible at one, N+1 requests at N. Only worth revisiting
if the published set grows; noted so the cause is known when it does.

## Publish `Llama-3.2-1B-Instruct-fp4` as the evaluation model

`models` · `distribution`

Sequenced after the 3B path is proven. Roughly 0.7-0.9 GB against the 3B's 2.87, dropping the
evaluation path's VRAM floor to about a gigabyte so an 8 GB card stops being excluded from "does it
work".

Convert, export, validate **generation** rather than per-layer parity, publish with a card. Test
against the tools-free system prompt first — a 1B is more prompt-sensitive than the 3B.

## The HuggingFace org card defines a Mila model as "already quantized"

`docs` · `distribution`

`gpt2-small` makes that false: the catalogue is now pre-quantized deployment models plus a reference
model for reading and training. Say there that MIS does not serve GPT-2, so nobody files it as a bug.
Card source is `.internal/Marketing/HuggingFaceOrgCard.md`.

## The attribution rule is written twice in two languages

`distribution`

`requiredAttributionFor` in `Chat.ModelCatalog.ixx` and `license_id.startswith("llama")` in
`publish_model.py:209`. They agree today — a third family with a display duty is what separates them.

## Packaging then installing hashes every file twice

`distribution` · `perf`

`buildPackage` hashes to derive the manifest digests and `install` hashes again to verify adoption —
~50 s of the ~60 s Llama 3B migration, ~2 minutes on the 8B. Neither check is wrong alone, so the fix
is a combined verb. `publish_model.py` has the same defect for its own reason.

## The store has no garbage collector

`distribution`

A 15.09 GiB blob is orphaned locally — no record references the pre-export cb2-3 weights since the
11.05 GiB build replaced them — and nothing reclaims it. The general gap, not the one file. A `mila`
verb that lists unreferenced blobs and removes them on request is the shape.

## Buffer Gemma Anthropic streaming only when tools are present

`adaptors`

Today it buffers unconditionally.

## Qwen tool results are not merged into one user turn

`adaptors` · `mila-src`

The checkpoint's template folds consecutive `tool` messages into a single `<|im_start|>user` turn
holding several `<tool_response>` spans; `Qwen.Protocol.ixx` emits one turn each.

Unreachable today — the harness dispatches one call per round — and it becomes wrong the moment
parallel calls land.

## Model capabilities belong in the manifest, not in a family switch

`adaptors` · `distribution`

`thinking_capable` and `streaming_capable` are both `family == Gemma` (`Chat.FamilyTraits.ixx`), and
`default_context`/`max_context` are per-family constants beside them — so two models of one family
cannot differ, and a non-Gemma reasoning model reads as having no channel.

`instruct` is already record-declared and proves the pattern, and the manifest tolerates unknown
fields, so this is additive. Do it before the next chassis threads a second switch.

## A session cannot move cards without restarting

`adaptors`

`--device N` and the `device` key choose the card at startup and every device question follows it,
but there is no `/device` command — `/set` is sampling knobs only, by its own contract. `/context`
shows the shape a reload-on-change command takes (`Chat.ixx`).

## Chat's console output has two rough edges

`adaptors`

**Library log output collides with the spinner**, so the first thing an evaluator reads is a
corrupted line: the spinner is mid-line when the BPE warning fires and the warning is spliced into
the loading message. `main.cpp:927` installs a stock `ConsoleSink`; since it derives from `Logger`
and `Mila::initialize()` takes a sink, Chat can supply a spinner-aware one — adaptor-side, no
library change. That sink should also drop `file:line:function` from user-facing output.

**`printThinking` still takes the plain-text path.** The answer block paints style spans; the
reasoning block does not, so a heading or bold label inside a thought renders unstyled
(`Chat.Renderer.ixx:176`). Harmless today, but it is the one renderer entry point that ignores
attributes — which is how a second convention starts.

## Chat and the quick starts have no test path

`adaptors` · `ci`

**No test model.** `gpt2-small` loaded in seconds and surfaced both the `context_length` crash and
the thinking-row defect; every remaining model is multi-gigabyte, and Chat now refuses base models.
A single-shot sample — prompt in, tokens out, exit code — is CI-shaped given a model in the store.
Both Chat and the quick starts need one fixture that requires no download.

**`Chat.StreamingDisplay` has no tests.** `RichText` now has 18 (`Mila/Tests/Adaptors/Chat/`), but
`holdPoint` and the chunk-boundary behaviour that produced the nested-bullet defect are unpinned.
Harder than RichText: the module imports `Chat.Renderer` and `Chat.Config`, so it needs either a
seam or those modules in the test target.

## Where Chat's configuration lives, and what `context_length: auto` means

`adaptors`

**One session config serves every model a session loads**, so `context_length` is either too small
for a 12B or fatal for GPT-2's 1024-row learned positions; today it is clamped by `maxContextFor`
(`Chat.ModelCatalog.ixx`), a per-family constant honest only for GPT-2. The answer is the largest
context that fits the card, which `getRequiredMemory(BuildContext)` computes. Open: the headroom
fraction, and the no-fit case.

**A container user has nowhere to put settings.** `session.json` ships inside the image layer, so
changing `temperature` means mounting a file over it, and `--config` assumes a file you can already
write. Related: `chat-state.json` sits in the store root, which `resolveStoreRoot()` puts in a
*cache* directory on Linux. Two shapes weighed — beside the store, or `MILA_CONFIG_DIR` — and they
settle together, because `context_length: auto` wants the same home.

## A non-interactive `chat` must name its model

`adaptors` · `docs`

Inferring one from a single-model store changes the command's meaning once a second is installed,
and persisting the choice would put it in `chat-state.json`, which lives in a cache directory — so
the quick start would work, then fail after an eviction.

Site copy is fixed; what remains is the sweep of every surface showing a scripted `install` then
`chat`. `CMD ["chat"]` is fine — interactive.

## The download bar restarts per file and never says which file

`distribution` · `mila-src` · `breaking`

A model is a manifest, a tokenizer and the weights, so the user watches 0-100% twice with nothing
distinguishing the runs. `ProgressCallback` is `(received, total)` only (`HttpClient.ixx:63`), so the
CLI cannot label what it is drawing.

Adding the file name, or an (index, count) pair, is a library signature change. The sub-megabyte
manifest is already suppressed in `Mila/Tools/Cli/Cli.ixx`.

## `main.cpp` re-checks what the store already guarantees

`adaptors`

After `resolveModel` succeeds it tests `exists()` on both paths, but `locate()` refuses an incomplete
record. Harmless duplication — except `/model` has no equivalent check, so if the guarantee is
doubted the check belongs in the store rather than in one caller.

## Does a Python completion sample need a `GptSession`?

`binding`

`Samples/QuickStart/Python/generate.py` already shows completion via `--raw`, so the only gap is
GPT-2 itself — `LlamaModel`, `GemmaModel` and `QwenModel` are the sessions the binding exposes, which
is also why MIS refuses the architecture. A binding decision, not a sample one.

## The GPT-2 CPU path treats build-time extents as runtime extents

`mila-src`

CPU inference has never run a prompt shorter than its built context. Remaining after the encoder and
LayerNorm fixes: `CpuLinearOp:259,264`, `CpuSoftmaxOp`, `CpuSoftmaxCrossEntropyOp` and
`CpuAttentionOp` — the last not mechanical, since `B_`/`T_` size its `{B,NH,T,T}` buffer at `:269`.
Pattern to follow: `CudaLpeOp:192-196`.

Here because the release criteria say "GPU-first: full CPU op parity is not a gate". **Contested** —
it reproduces as a crash, and if the QuickStart samples reach this path on a CPU-only machine it is
an onboarding failure and belongs back in the release.

## Revive the loss + backward path

`training` · `mila-src`

CrossEntropy and SoftmaxCrossEntropy. Both samples compute loss host-side, so this is off the
critical path to a converging sample.

It has never been compiled: three test files are commented out (`Mila/Tests/CMakeLists.txt:155`,
`:156`, `:284`) and written against a 3-parameter component that now takes 2, and the vocab >= 1024
block kernel has never run — its `__shfl_sync` gives warps 1+ `-INFINITY`. `OperationTraits.Cuda.ixx`
also maps `<CrossEntropyOp, Cuda, BF16>` to a type with no `__nv_bfloat16` specialization, the `half`
one a silent stub (`CudaSoftmaxCrossEntropyOp.ixx:99`) — deleting that row is the separate,
freeze-compatible half that stays in the release.

## Revive the `Dropout` component

`training` · `mila-src`

Training-only, and authored from scratch rather than revived.

## Training (advanced)

`training` · `mila-src`

Llama fine-tuning, loss-function GPU migration, gradient checkpointing, and BF16/GQA training.

## Performance

`perf` · `mila-src`

Gemma 4's levers — the fused W4A16 prefill GEMM, flash-attention on the global layers — and the
codebook path's own, each a measured gap with its numbers in `Qwen3.8.md` §8: the decode GEMVs'
bandwidth shortfall against FP4 (amortize the unpack across output rows, or bucket activations by
code); staging the sub-4-bit prefill to FP8 so it reaches the sm89 tensor GEMM instead of a BF16
one, gated on e4m3's 3 mantissa bits over 2.82-bit codes; the per-chunk staging dequantize,
unmeasured across the rung ladder (`Qwen.ixx:110`); tensor cores for the DeltaNet chunked kernel,
worth ~13% of prefill; and whether Gemma's ring softmax is reachable from Qwen's 16 full-attention
layers. [[project_w4a16_prefill_gemm]]

## Whole-model prefix caching for Qwen

`perf` · `adaptors` · `mila-src`

The 48 DeltaNet layers need the snapshot/restore copy and the 16 attention layers the positional
rewind, and nothing combines them. Deferred as a policy question: how many prefixes to hold in host
RAM, and eviction. `Qwen3.8.md` §8, `PromptCaching.md`

## Native low-precision compute (Blackwell+)

`quantization` · `mila-src`

The microscaling data path and finer per-arch gating.

## Compute backends beyond CUDA

`architecture` · `mila-src`

ROCm and Metal; `DeviceType::Rocm` / `::Metal` are reserved and unimplemented.

## Platform portability — aarch64 + coherent memory

`build`

Mila has never been built on ARM.

## Model loading

`perf` · `mila-src`

A load-time FP4 sidecar cache, and concurrent/async read I/O for real queue depth.

## Ungated GPT-2 zero-auth quick-start

`distribution` · `docs`

A first-run HTTPS weights fetch.

## `ComponentType` vitality

`api` · `mila-src`

Does `getType()` earn its keep, or does the unused converter surface retire?

## Discoverability

`docs`

Internal, not a README theme — the site is live at `mila.toddt.me`.

## A value-reading observation sink has to name the model's compute precision

`observability` · `mila-src`

The sink gets `const ITensor&`, whose `rawData()` is type-erased, so anything wanting numbers does a
`dynamic_cast` to `Tensor<TPrecision, MR>` then its own `toHost`; all three consumers do this.

Whether observation should offer a typed convenience is decided-deferred to v0.21 —
`Observability.md` §11.2.

## Remove FP16, superseded by BF16 — measure first

`api` · `mila-src` · `blocked`

Woven through live code; trace live-vs-dead first, and 8 `REVIEW:` markers already scope it.

Note the odd row it collides with: CUDA `LayerNormOp` is registered at FP32 and FP16 and *not*
BF16, so deleting the FP16 row leaves CUDA LayerNorm FP32-only. Pinned by a `static_assert`, so this
work must confront it.
