# Mila — Roadmap

Where Mila is going — the durable narrative of each release and what it means.

- **Open tasks** -> [BACKLOG.md](BACKLOG.md) · **Completed work** -> [CHANGELOG.md](CHANGELOG.md)
- **How versions, milestones, labels, and releases work** -> [RELEASING.md](RELEASING.md)
- **Design rationale** -> `Mila/Specifications/`

The roadmap shows two releases at a time — the one in flight and the one after (**vNext**) — plus a
**Future Directions** tail. Each release is reached through **milestones** tracked by task completion
(see [RELEASING.md](RELEASING.md)). Current version: **`0.20.0-alpha.6+105`**.

The **Gemma 4 12B dense chassis** has been delivered into v0.20 (HF token-for-token parity, 2026-06-23
— see CHANGELOG); its memory-fit gates (weight-tying + bounded-KV ring cache) both landed in +78, and
the 26B-A4B MoE follow-on stays a Future Direction.

---

## v0.20.0 — First Production Release

**Release Date:** _Target — H2 2026 (range; expanded to a craft-complete scope — see below)_

Mila is a craft-mastery project — understanding LLMs at the metal — not a llama.cpp/vLLM
competitor. For a project like this, "first release" means **complete and beautiful**, not a minimal
slice. v0.20 delivers everything Mila has implemented and validated, as one coherent, tested,
documented package: GPT-2 and Llama; **inference and training**; FP32 / BF16 / FP8 / FP4; tool
calling.

This scope is a deliberate reunion of two bodies of work. The last year built the inference path
(Llama, quantization, the `OperationTraits` dispatch, the chat harness). The year before built a
fully **test-driven, Doxygen-documented** GPT-2 + training foundation — the MNIST and Bard samples,
the optimizer, the loss and backward kernels — which worked well, then fell behind the inference-era
API churn and was parked (the samples and ~70 test files are commented out, awaiting re-alignment).
v0.20 recovers that foundation to current-API quality rather than shipping inference alone. The work
is **resurrection, not invention**: the designs, tests, and docs already exist in the tree.

v0.20 also ships under a **locked product definition**
([MilaProductFamily.md](Mila/Specifications/MilaProductFamily.md)): Mila is an inference runtime
library plus adaptors distinguished by who closes the generation loop — the Chat harness (human
gate) and MIS (wire) ship with this release; the **Agentic adaptor is explicitly post-release**,
named there as the scope-creep vector to refuse. The release bar for the definition is that its
claims are demonstrable: clone, build, run Gemma 4 12B FP4 on a 12 GB card, drive it from a
foreign harness through MIS, and read the whole path from prompt to kernel with no hidden engine.
One consolidation item follows from the definition into v0.20 scope: the Gemma token grammar folds
down into the runtime (it has already drifted between Chat and MIS — see BACKLOG, *Product Family —
Grammar-in-Runtime Consolidation*).

Pre-1.0: "production" means validated and polished, not API-frozen. Breaking changes remain
acceptable. The release is reached through the milestones below, in dependency order — consolidation
lands the shared architectural foundation; the test suite is revived to become the correctness
oracle; training is resurrected against green tests; documentation describes the stabilized surface;
beta validates and packages it for the public.

### Milestone: Consolidation

*Feature freeze + debt burndown — earn the right to call it beta, and lay the foundation the
revival milestones build on. No new features.*

Beyond closing the alpha line, this milestone fixes the **load-bearing architectural debt that both
the test and training revivals depend on**: the `CompositeComponent`/`setTraining` lifecycle bug
(which is why the component tests and training guards are currently disabled), the CPU-op
`OperationTraits` migration, and coupling parameter initialization to runtime mode. Doing this debt
work properly — not re-stubbing it — makes the tree training-ready as an honest byproduct of
consolidation.

- [x] OperationTraits close-out — CPU Linear traits specialization live; the `CpuLinearOpTypeMap` holdout retired (out of build + `RETIRED` banner)
- [x] OperationTraits close-out — retire the legacy dispatch files in place (out of build + `RETIRED` banner, not deleted): done for the Linear/Gqa typemaps, registry/registrar helpers, `OperationsRegistrar`, `FusedComponent`. **Consolidation-scope complete;** `OperationRegistry` + the arity bases stay live only for the *disabled* CrossEntropy/fused ops and are **relocated to Training Revival** (retired there when the loss path is re-authored — training-only work, outside the freeze). `Dropout` parked at `Dev/Components/Regularization/`; its `DropoutOp` re-authoring is a Training Revival item
- [x] Marker burndown — all `FIXME:`/`TODO:` converted to `REVIEW:` (2026-06-19) so public source reads clean, and the surviving `REVIEW:` markers triaged into buckets A-H with dispositions and homes in the BACKLOG "Marker debt triage" item. The one stray literal `FIXME` left in the primary target (the GQA `forward()` dead-branch comment) was reworded to a clean `REVIEW` pointing at its tracked bug (+113); the only remaining literal `FIXME`s were in the orphaned `Dnn/Decoders/` skeleton, moved out of the tree to the user's graveyard (superseded by `Dnn.Samplers`). **Consolidation-scope complete;** the bucket resolutions themselves are **relocated to their owners** — bucket-C `dim_t` + bucket-D correctness (incl. `Llama.Block.ixx:132`) to Production Hardening, bucket-H org/docs to the Doxygen pass. The old lifecycle linchpin (the `CompositeComponent` setTraining/build concern) was already resolved by the RuntimeMode/TrainingMode two-axis redesign — `isTraining()` is gone
- [x] Debug-instrumentation strip — kernel anomaly guards removed (Residual/Gelu/LayerNorm/RmsNorm; `Swiglu.Fp32.v1.cu` deleted) and BPE tokenizer console routed to `Logging::Logger` or removed; the two inert commented-out CPU-op debug dumps (`CpuAttentionOp` K/dQ, `CpuLinearOp` input) deleted (0.20.0-alpha.6+81). The remaining console sites are deliberate/legitimate, kept: the `TokenSequenceLoader` init summary (gated on `config_.verbose_logging`) and `TensorBuffer` alloc trace (`if constexpr (TrackMemory)`) are opt-in diagnostics; `CudaExecutionContext` stderr-on-stream-error and the `TokenEmbeddingConfig` `isfinite` validate-guard are error/validation paths. Training-path instrumentation (AdamW anomaly `printf`, `BpeVocabulary`/`OptimizerBase`/`TrainingLogger`) deferred to Training Revival
- [x] Debug-instrumentation strip — per-step `synchronize()` removed from `GptBlock`/`LlamaBlock` `forward()`/`backward()` (single-stream ordering + the caller's host-read boundary sync suffice). Llama `prefill`/`decode` were already sync-free; `GptBlock::decode()` still carries its 8 per-op debug syncs (GPT-2 inference path) — the same correctness-neutral strip, left for when the Bard/GPT-2 decode path is next exercised
- [x] FFN consolidation — de-polymorphize `MLP`: the **blocking** runtime activation dispatch is gone (`MLP` holds a concrete `Gelu` child; the `mlp_activation_impl` / `std::function` bridge / SwiGLU branch retired in `MLP.Dispatch.ixx`, out of build), `Swiglu` -> `FFN/Swiglu/` and `MLP` -> `FFN/MLP/` relocated, and `MLP<Cpu>` / `GptBlock<Cpu>` / `GptTransformer<Cpu>` now build (their `.Cpu.cpp` tests are active). **Consolidation-scope complete;** folding the fixed `Gelu` child onto the generalized `Activation<…, ActivationType TFn = Gelu>` (the compile-time-`TActivation` endgame) is **net-new work — FROZEN, deferred to vNext**. Design: `Specifications/FfnAndMoE.md`
- [x] FFN consolidation — `Activation` elementwise primitive: compile-time `TFn`, the `MILA_HD` functor library, and the Cpu/Cuda `ElementwiseActivationOp` (functor-templated, not a traits axis). **Landed alongside `Gelu`:** the functor library (all 8 functions), `Cpu`/`Cuda` `ElementwiseActivationOp` (FP32 / FP32+BF16), the member-template `op_for<Functor>` traits, the `Activation<…, TFn>` component + config, and its CPU/CUDA tests. The reusable `GatedMLP` gated FFN (single-expert reference, `TGate` fixed to SiLU) landed with it. **Consolidation-scope complete;** the remaining folds (`Gelu`/`MLP`-child onto `Activation`; CPU `SwigluOp` + `Swiglu<…, TGate>` generalization) are **net-new / contributor work — FROZEN, deferred to vNext** (tracked in BACKLOG)
- [x] Hardening — couple the `initialize_parameters` default to `RuntimeMode` — the `BuildContext` third argument is now `std::optional<bool>` (nullopt => derived: Training initializes, Inference skips), so an inference-mode build can no longer silently run then discard parameter initialization by omitting the flag; all existing explicit call sites (`false` on the three `fromPretrained` paths, propagated `shouldInitializeParameters()` in Llama/Gemma) compile unchanged
- [x] Resolve the poisoned BF16 dispatch rows — **DONE (0.20.0-alpha.6+112, VS2026 build green):** the four rows `OperationTraits<{Gelu,MultiHeadAttention,Softmax,Lpe}Op, Cuda, BF16>` are dropped (FP32-only is the honest advertisement — those ops' kernels are `float || half` only, the BF16 FFN/attention paths use Geglu/GQA/RoPE, and GPT-2 MHA/Lpe run FP32). Paired with the **Dispatch error UX** deliverable below (`OperationSupported<...>` predicate + declaration-only primary => a missing tuple reads as a one-line "undefined type" naming the tuple, not a cascade). The desync audit is discharged: those four were the only poisoned rows (CrossEntropy's BF16 row is honest — its kernel is `float || nv_bfloat16`). The active CUDA tests were already FP32-only by explicit design, so no test needed changing. See BACKLOG, *Consolidation* (poisoned-row item) and *Dispatch error UX*

**Exit:** every box checked, no literal `FIXME` in public source, debug instrumentation gone, and the
Component lifecycle is sound enough to re-enable the component/training tests.

**CLOSED 2026-07-18 (0.20.0-alpha.6+113).** All boxes checked: the poisoned rows dropped (+112), the
four scope-complete items ticked with their net-new/training remainders relocated to vNext / Training
Revival under the v0.20 **feature freeze**, the last stray `FIXME` reworded, and the orphaned
`Dnn/Decoders/` skeleton moved out of the tree. Debug instrumentation is gone and the RuntimeMode/
TrainingMode redesign made the Component lifecycle sound (component tests re-enabled). This milestone
earns the beta label; the remaining beta.1 gates are Test Suite Revival's CI ratchet and
`find_package(Mila)`.

### Milestone: Test Suite Revival

*Re-green the authored test suite to the current API and gate it in CI — the anti-rot ratchet and
the correctness oracle for everything after it.*

The first year of Mila was test-driven; the authored suite was largely commented out during the
inference-era refactors (the CMake note is explicit: "too many tests to refactor for Component
lifecycle changes"), leaving only ~24 of ~70 files active. The revival has since re-enabled and
area-split them into ~100 active test translation units, with only ~11 parked by explicit
disposition. This is recovery, not greenfield — the
test *logic* is authored; the work is re-aligning it to the post-refactor API (`OperationTraits`
dispatch, the `Operation` base-class collapse, the precision axes, the lifecycle fix from Consolidation).
Two distinct, genuinely-new slices remain. The inference features built *during* the test drought
(quantization, the Llama path) need coverage the old suite never had. And — the larger gap — the
authored suite was **forward-only**: inference validated forward passes against HuggingFace, so every
`backward()` the training samples drive (per-component gradients, the optimizer step, train-from-scratch
init at the sample precision) has *zero* coverage. A finite-difference gradient-check archetype is the
precondition for Training Revival — reviving the samples without it validates convergence by eye, not
by test. That oracle is built MNIST-spine first (Linear / Gelu / Network / AdamW — the simplest graph)
before the Bard transformer stack, mirroring the sample-revival order in the next milestone.

The deliverable is not just green tests — it is the **CI ratchet** that keeps them green. The suite
atrophied because nothing gated it; revival without a gate merely reschedules the next rot.

**Success criteria:** the authored component / tensor / tokenizer suites re-enabled and
green against the current API; the redundant op-layer mirror tests retired (backend ops are
implementation detail, tested through the public component — the sole op-level exception is the
unreachable weight-quantization white-box); new coverage for the quantization and Llama inference
paths; a
per-component gradient-check archetype covering the training backward path (MNIST spine first, Bard
transformer stack second); the suite gated in CI (building on the `MILA_ENABLE_CUDA=OFF` CPU-only gate)
so a future API churn fails loudly instead of silently rotting coverage.

- [~] Re-green the authored component / tensor / tokenizer suites to the current API — the concrete component-class set (Linear, Gelu, LayerNorm, RmsNorm, Swiglu, Residual, TokenEmbedding, Lpe, Rope, Softmax, MHA, GQA, MLP, GptBlock, GptTransformer, LlamaTransformer) is re-enabled and build-green; only `SoftmaxCrossEntropy` (loss) is parked for the loss-on-device milestone. 3 backward-numeric cases stay `GTEST_SKIP`'d pending filed bug fixes (CUDA Softmax backward stub, BF16 Swiglu backward dtype, GptBlock/MHA-CPU composed gradient)
- [x] Retire the redundant op-layer mirror tests (delete-not-revive — backend ops are implementation detail, tested through the public component) — out of the CMake build, dispositions recorded in `Tests/CMakeLists.txt` Section 3; files kept on disk pending an explicit delete
- [~] Core `Tensor.ixx` coverage to the value-type / god-module archetype (8 area files + `.Cuda` companions) — done; remaining: `TensorOps.Transfer` device-split, `Structural`(`split`) backfill, and the wider `Tensors/` tree (`TensorBuffer`, `TensorDataType*` maps, `Partitioning`, `Serialization`)
- [x] Gradient-check archetype (finite-difference numeric backward) — shared fixture `Common/GradientCheck.h` + `Gelu`/`LayerNorm` reference applications landed; `Backward_MatchesNumericGradient` fanned out to Linear (dX/dW/dB), Residual (binary da/db), and MLP (dX + all four child parameter gradients), with GptBlock carrying the equivalent own-forward finite-difference check. The standalone MHA probe added to isolate the suspected CpuAttentionOp backward is **green (validated on VS2026 2026-07-02)** — `CpuAttentionOp::backward` is numerically correct, exonerating the prime suspect and confirming the GptBlock composed sentinel is genuinely green (not masked by the residual skip paths). The MNIST/Bard training spine now has per-component numeric backward coverage. Precondition for Training Revival's convergence oracle — met
- [ ] Backfill the inference-drought coverage the old suite never had — load-time quantization (`PerChannelFp8` / `PerGroupFp4`, the decode matvec kernels), `OperationTraits` dispatch, and the Llama path (RmsNorm / SwiGLU / GQA / RoPE components + `LlamaModel::fromPretrained`); the `CudaLinearOp` quantization white-box is the sole legitimate op-layer test. Genuinely new, not recovery
- [~] Re-green in sample-revival order — MNIST spine first, Bard spine second (mirrors Training Revival sequencing). MNIST spine mostly landed (`CompositeComponent`, `AdamW.Cpu`, `DataLoader`(+`.Cuda`), `Network.Cpu`, `TensorOps/Random.Cpu`); remaining: the thin `Core/Network.cpp` delta + the GPU companions (`Network.Cuda` / `AdamW.Cuda`), then the Bard GPT-2 stack tail
- [x] Verify the full suite green in one pass (CPU-only `MILA_ENABLE_CUDA=OFF` and the CUDA build) — **CPU-only leg verified 2026-07-18:** a from-scratch `linux-clang-cpu-debug` build (new preset) compiled clean under clang-21 (the CPU-only library path had *not* bit-rotted — no portability fixes needed) and `ctest` ran green in one pass (~980 cases, zero failures; GPU/data-dependent cases self-skip). The CUDA-build leg is green in the user's VS2026 / WSL `Build All`. Baseline established for the CI gate
- [~] **[gate]** Wire the suite into CI as the anti-rot ratchet, building on the `MILA_ENABLE_CUDA=OFF` CPU-only gate so a future API churn fails the build instead of silently re-commenting coverage — the deliverable that keeps the revival alive. **Wired (0.20.0-alpha.6+114, awaiting first CI run):** a `cpu-only-tests` job in `build-pipeline.yml` runs the CPU suite on every push/PR to dev+master — plain `ubuntu:26.04` (clang-21, no CUDA image/toolkit, no CUTLASS fetch), mirroring the `linux-clang-cpu-release` preset, `ctest --output-on-failure`. This is the first CI job that *runs* tests (compile-and-gate only compiles; GPU tests stay local). Confirm green on the first GitHub Actions run, then close

### Milestone: Training Revival

*Resurrect the validated GPT-2 / MLP training path — MNIST and Bard — to current-API quality, proven
by its own revived tests. Scope is GPT-2 / MLP only; Llama 3.1/3.2 training stays a Future Direction.*

MNIST (MLP) and Bard (GPT-2 generation) were complete, working training samples, parked behind an
explicit `FIXME: Re-enable after alpha.5 completed` trigger that has now fired. Reviving them
reactivates the half of the library inference never exercises: the AdamW optimizer, the loss and
backward kernels, gradient flow, and train-from-scratch parameter initialization (the per-device
init subsystem was restored in Alpha.5). The revived **primitive/component tests** from the previous
milestone are the correctness oracle — the samples are usage demos and the bug-discovery mechanism,
not the test target. Convergence is an emergent consequence of correct primitives (forward, backward,
optimizer step, loader, init), not a separate thing to assert against sample code.

The work is sequenced **MNIST first, then Bard**, for both the test revival and the source edits:
MNIST is a pure MLP (Linear / Gelu / Network / AdamW) with no transformer, tokenizer, or BPE surface,
so it exercises the full training spine — forward, gradient-check, optimizer step, train-from-scratch
init, loader contract, end-to-end convergence — on the smallest possible graph. Bard then stacks the
`GptTransformer` (Lpe / GptBlock / MLP / MHA / LayerNorm / Residual), the BPE/char tokenizers, and the
`TokenSequenceLoader` on top of an already-proven spine. Both samples currently compute loss and the
output gradient host-side, so the library `CrossEntropy` path is *not* on the critical path to a
converging sample — it is decoupled, later work.

Known correctness work beyond mechanical API re-alignment: the CUDA `fill_normal`/`fill_uniform`
FP32-only gap (corrupts BF16 train-from-scratch init), the AdamW test re-enablement, and the
Component-lifecycle fix landed in Consolidation.

**Success criteria:** the training-path **primitive suite** is the green/red oracle — the
gradient-check archetype, the AdamW step-convergence test, the concrete data-loader contract tests,
and init-at-precision — with a small **sample-independent** training-loop integration test (a tiny
graph built from library primitives in `Tests/`, loss strictly decreasing over a fixed step budget)
as composition/wiring insurance; train-from-scratch validated at the precisions the samples use; all
training-path tests CI-gated. The MNIST and Bard samples are re-enabled and **run** against the
current API (MNIST trains to target accuracy, Bard generates coherent text) — validated by running
them as usage demos, not gated in the unit suite.

Status: the hard part is done — **both samples are revived and validated** (by eye, on CUDA). The
remaining work is the **primitive test suite** that proves the training path correct at the component
level (the samples are the discovery tool, not the test target), plus the deferred training-only
pieces (loss path, Dropout, progress reporting). The net-new test items below overlap the Test Suite
Revival gradient-check fan-out and CI ratchet — the two revival milestones are one work-front.

- [x] Revive the MNIST (MLP) sample to the current API + validate — builds green (VS2026/CUDA), trains FP32 from scratch to ~97.9% test accuracy over 20 epochs; the full spine (forward chain, backward gradient flow, AdamW step, train-from-scratch init) exercised end-to-end. MNIST spine tests (Linear / Gelu / Network / `AdamW.Cpu` / DataLoader) green
- [x] Revive the Bard (GPT-2) sample to the current API + validate — builds green (VS2026/CUDA), trains FP32 to perplexity <3 / loss ~1.09 by epoch 17 with coherent Shakespeare-structured text; surfaced + fixed 3 latent CUDA-training-backward bugs (cuBLASLt bias epilogue, inverted attention eval-guard, `TensorOps` math no-op)
- [x] Flip the `FIXME: Re-enable after alpha.5 completed` triggers — both samples re-enabled in `Samples/CMakeLists.txt` (CUDA-gated until the CPU-only build-coherence work lands)
- [~] Concrete data-loader contract tests — `TokenSequenceLoader` done (construction / iteration / reset / target-is-input-shifted / threading stress); remaining: the `MnistDataLoader` contract test (pixel normalization, one-hot targets, shuffle-on-reset, IDX magic-number validation)
- [~] Re-enable + re-align the AdamW path — `AdamW.Cpu.cpp` re-greened (Section 1, includes a closed-loop convergence case); remaining: the `AdamW.Cuda.cpp` companion + resolve the deferred AdamW debug-instrumentation strip-vs-gate (`CudaAdamW.cu` `printf` guards + `CudaAdamWOptimizer.ixx:270`)
- [~] **[net-new]** Training-loop integration test (sample-independent) — a small test in `Tests/` that builds a tiny graph from library primitives *in the test* (Network + AdamW + a few batches) and asserts the loss strictly decreases over a fixed step budget. Validates composition/wiring between validated primitives (the class of bug the Bard revival surfaced), not whether the `Samples/` code converges — it never imports sample code. Largely delivered for the MNIST spine by `Network.Cpu.cpp` (`Linear->Gelu->Linear` + `createOptimizer` + `zeroGradients` + loss-decrease); remaining is a GPT-2-stack analogue for the Bard spine. Small budget so it runs in the `MILA_ENABLE_CUDA=OFF` CI gate
- [ ] **[net-new]** Optimizer step-convergence test — "minimizes a known convex objective in N steps," proving the update direction + bias-correction are correct, not just that `step()` runs
- [ ] **[net-new]** TrainingMode / RuntimeMode behavior coverage — assert build-mode and runtime-mode transitions allocate/skip gradient buffers correctly, so the two-axis lifecycle fix has a regression guard
- [ ] Fix the CUDA `fill_normal` / `fill_uniform` FP32-only gap (corrupts BF16 train-from-scratch init) — the CUDA counterpart to the `CpuTensorOps.Random` backend; pair with a BF16 init-at-precision `TYPED_TEST` that turns the silent corruption into a red test
- [ ] **[decoupled]** Revive the loss + backward path (CrossEntropy / SoftmaxCrossEntropy) — both samples compute loss host-side, so this is off the critical path to a converging sample; the dispatch struct was started (alpha.6+68) but is not wired in
- [ ] **[net-new, training-only]** Revive the `Dropout` component from `Dev/Components/Regularization/` — re-author `CpuDropoutOp` / `CudaDropoutOp` + `OperationTraits` rows + the two-axis `Component<TDeviceType, TPrecision>` rewrite; the mask/backward path is exercised only by training
- [ ] ProgressReporter mechanism — an injected per-operation progress facility for long-lived ops (BPE vocab training, `PretrainedReader` load, load-time quantization); the Consolidation debug strip left the BPE training progress in place to migrate here
- [ ] Validation — the training path proven correct **by the primitive test suite** (gradient-checks, optimizer step-convergence, loader contracts, init-at-precision, plus the sample-independent training-loop integration test), CI-gated; train-from-scratch validated at the precisions the samples use; the samples run as demos (MNIST trains to target accuracy, Bard generates coherent text), validated by running them rather than gated in the unit suite

### Milestone: API Documentation

*Reconcile the Doxygen surface to the post-refactor reality and publish it — documentation held to
the same standard as the code.*

Doxygen-equal-to-features was a first-year discipline; the inference churn left the prose describing
a retired world (components "registering with `OperationRegistry`", "deriving from
`UnaryOperation`"), `@file` tags drifted from filenames, and `@param`/`@tparam` names no longer match
signatures. This milestone restores documentation accuracy, narrows the published surface to the
public `import Mila;` API (not every private member of 287 modules), and publishes via a GitHub
Action — never committing generated docs to the tree. Detailed staleness tiers and the docs-CI
mechanics live in [BACKLOG.md](BACKLOG.md).

**Success criteria:** `@file`/`@param`/`@tparam` drift cleared; file-level and symbol Doxygen
reflects the `OperationTraits` world and the spelled-out naming style; the published docs scope
matches the public API surface; the docs job renders C++23 module units faithfully and publishes
from `master`; Doxygen's own warnings (`WARN_IF_DOC_ERROR`/`WARN_NO_PARAMDOC`) gated as errors in
the docs job so doc drift fails the build instead of silently re-accumulating — the documentation
analogue of the Test Suite Revival test-CI ratchet.

Not a heroic read-everything sweep. The Doxygen already exists pervasively (~1,950 `@brief`,
~1,100 `@param`, ~257 `@tparam`, ~218 `@file` across 258 files) — this is reconciling *drift*, not
authoring. Two levers make it bounded: **narrowing the published surface** to the public `import
Mila;` allowlist deletes the internal ops/kernels/registries from the denominator (today the Doxyfile
extracts every private member of 287 modules), and **the Oracle** (Doxygen's own `WARN_*` output)
turns an open-ended audit into a shrinking, tool-generated worklist. Tasks are ordered so each step
shrinks or bounds the next; the judgment-heavy semantic tier is amortized into the Test Suite Revival
(which already opens each file). Engineering detail (tiers, export allowlist, docs-CI mechanics) lives
in [BACKLOG.md](BACKLOG.md) under *Module Hygiene*, *Public API Surface*, and *Release Assets & CI*.

- [x] **Narrow the published surface — DONE, verified (2026-07-02).** Two parts, both in `Mila/Docs/CMakeLists.txt` (NOT the empty `Doxyfile.in`): (1) **EXTRACT flip** — `EXTRACT_ALL`/`EXTRACT_PRIVATE`/`EXTRACT_PACKAGE`/`EXTRACT_STATIC`/`EXTRACT_LOCAL_METHODS` `YES` -> `NO` (scopes to documented symbols AND is the prerequisite that lets the Oracle fire — `EXTRACT_ALL=YES` had suppressed the param warnings); (2) **INPUT scoping** via `EXCLUDE_PATTERNS` (`*/Compute/Devices/*/Operations/*`, `*/Kernels/*`, `Cuda/Helpers`) + `EXCLUDE_SYMBOLS` (`*::Detail`), chosen over a fragile hand-maintained 90-file allowlist. **Verified by direct Doxygen 1.17 runs:** warnings 235 -> 72, public API fully retained (Linear/Gelu/Tensor/Llama/Gemma/MLP/AdamW documented; internal `CudaGqaOp`/`CpuGeluOp`/`CudaLinearOp` and `Detail::` builders dropped; 348 -> 280 class pages). The parallel *Public API Surface* task narrows the `Mila.ixx` umbrella exports themselves (code-level), independent of this docs scoping
- [x] **Oracle** — wired **and run** (2026-07-02). **Correction:** the docs config is CMake `DOXYGEN_*` vars in `Mila/Docs/CMakeLists.txt` via `doxygen_add_docs` (the `Doxyfile.in` is an empty, unreferenced placeholder); `WARN_NO_PARAMDOC` was already set. Added `WARN_IF_DOC_ERROR` + `WARN_LOGFILE`. Then ran Doxygen 1.17 directly (warnings-only config over `Src/**/*.ixx`, `EXTRACT_ALL=NO` so the param warnings fire) — modules parse fine. **Generated worklist: 265 warnings** = 54 stale `@param` names (drift) + ~184 undocumented params (completeness) + 6 parse-noise; 129 warning-sites, 71 of them internal ops/kernels (empirically confirming the narrowing-first rationale: the `import Mila;` `INPUT` allowlist roughly halves the count). The shrinking count is now the definition of "done" — it both drives and locks the tiers below
- [x] Tier 0 — non-ASCII / mojibake in comments **DONE (2026-07-02)**: the `Src` tree is ASCII-clean (verified 0 non-ASCII bytes outside 4 legitimate UTF-8 BOMs, left in place as MSVC-benign file-encoding markers). Rename fold-in (`Comonent` -> `Component.TrainingMode.ixx`) done; then a two-phase byte-exact pass converted 547 valid-UTF-8 glyphs (em-dash -> `--`, arrow -> `->`, box-rule -> `-`, box-corner/tee -> `+`, `×` -> `x`, `≡` -> `==`) + 28 stray CP1252 bytes (0x97 em-dash -> `--`; one 0xB7 middle-dot -> `.` in `Q.K^T`) + the 4 literal U+FFFD (all em-dash context). Box-drawing tree diagrams (e.g. `Llama.Block.ixx`) flattened 1:1 to aligned `+ - |` ASCII art. Scope extended beyond comments to ~12 non-ASCII em-dashes in exception/log **string literals** (display-only text, a source-portability fix)
- [x] Tier 1 — `@file` rename drift: 32 files whose `@file` tag != filename corrected to `basename` (scripted, verified drift == 0). The `Comonent.TrainingMode.ixx` case resolved via the Tier 0 rename
- [x] Tier 2 — `@param`/`@tparam` name mismatches vs. the signature — batch-fixed from the Oracle worklist (review before applying; signatures span lines). **Started (2026-07-02):** the `@param is_training` -> `training_mode` cascade fixed (10 warnings from 3 sites — `CompositeComponent::onTrainingModeChanging` inherited into 7 subclasses, plus `Gelu`/`Softmax` overrides — a leftover from the pre-`TrainingMode` `bool isTraining` API; also corrected the stale "true = training, false = eval" prose to the `Normal`/`Eval` enum). Re-run confirmed 10 -> 0 (total 265 -> 235). **"Review before applying" already paid off:** `TokenSequenceLoader`'s `@param is_training` was verified to be a *real* `bool is_training` ctor parameter and correctly left alone. **DONE (2026-07-02):** with the surface narrowing in place, all stale `@param`/`@tparam` name mismatches were fixed to **zero** (from 54) across ~15 public-API sites — the `onTrainingModeChanging`/`onBuilding` context params, GQA `prefill`/`decode` (concatenated-QKV -> q/k/v/position_offset), Component `build`, ModelConfig orphaned-doc detach, RmsNorm/Residual/TensorBuffer ctor params, and the memory-resource `do_deallocate` unnamed/named-param docs. Each reviewed against its real signature (the `TokenSequenceLoader` false-positive and the `CudaManaged`/`CudaPinned` named-`alignment` nuance were caught by review). Total worklist **265 -> 20**. The residual 20 are NOT name-drift — see the Ratchet item below
- [ ] Tier 3 — semantic staleness (retired-world prose: `OperationRegistry`/`UnaryOperation`, `TWeightQuant`-style naming drift, over-long file `@brief`s). Per-subsystem judgment; **folded into Test Suite Revival** — fix a file's prose while it is open for re-greening, not as a separate megasweep
- [x] **Ratchet** — `WARN_AS_ERROR = FAIL_ON_WARNINGS_PRINT` set (the docs analogue of the test-CI ratchet); doc drift now fails the docs build. **The residual 20 are CLEARED (2026-07-02):** the `@example`-misuse (-> `@code`/`@endcode`, which also fixes the broken example rendering), the unescaped `<pad>`/`<name>` HTML-tag token strings (-> backticks), the undocumented `Tensor`/`TrainerFactory`/`Softmax`-ctor params, the stray `@param` on zero-arg methods (orphaned docs above `//`-commented methods, `/**` -> `/*`), and the `@return`-on-void. **The `.ixx` public-API code-doc worklist is now ZERO (265 -> 0).** Also fixed latent docs-config bugs found while validating a faithful mirror of the real CMake config: `EXTENSION_MAPPING = ixx=C++` (Doxygen was NOT parsing the module units as C++ at all), `FILE_PATTERNS = *.ixx`, `WARN_IF_UNDOCUMENTED = NO` (the Oracle gates on drift + param docs, not document-every-symbol), `GENERATE_LATEX = NO` (HTML-only; kills the epstopdf/TeX error). **README cross-links FIXED + ratchet FLIPPED (2026-07-02):** the two relative README-mainpage links (`[getting-started.md]`/`[License.md]`) were converted to absolute GitHub URLs -- matching the README's OWN existing convention (its ROADMAP.md links already use `https://github.com/ToddThomson/Mila/blob/dev/...`), so it was an inconsistency fix, not a structure change; Doxygen treats `https://` as external so no `\ref`. A faithful mirror of the real `doxygen_add_docs` config (README mainpage + HTML) then verified **TRUE ZERO**, and `WARN_AS_ERROR = FAIL_ON_WARNINGS_PRINT` is set in `Mila/Docs/CMakeLists.txt`. Ratchet mechanism verified: doxygen exits 0 on the clean tree and fails on any reappearing warning. The earlier call-graph truncation risk is **eliminated** -- the docs-CI decouple (below) set `HAVE_DOT=NO` in the canonical Doxyfile, so there are no dot graphs to truncate and the ratchet is now fully reproducible locally (no graphviz dependency)
- [x] Docs-site CI — **DECOUPLED (2026-07-02).** A canonical standalone `Mila/Docs/Doxyfile` is now the single source of truth (all Doxygen settings); `docs.yml` runs `doxygen Mila/Docs/Doxyfile` directly on a plain `ubuntu-24.04` runner — **no CUDA container, no CMake configure, no CPM, no Graphviz** — with Doxygen pinned to 1.17 (matches local validation + the `.ixx` module parsing). `Docs/CMakeLists.txt` (`add_custom_target(docs …)`) and `doxygen-build.sh` invoke the *same* Doxyfile, so no config drift; both `mkdir -p build/docs` first (Doxygen does not create nested `OUTPUT_DIRECTORY` parents). Doxygen 1.17 confirmed; `.ixx` units render (`EXTENSION_MAPPING = ixx=C++` — they were not parsed as C++ before). Verified end-to-end locally: `bash Mila/Docs/doxygen-build.sh` exits 0 with 276 class pages and the ratchet passing. Publish still uses the first-party Pages actions from `master`. **Pending: a real docs-CI run** (GitHub Actions can't be run locally) to confirm the pinned-Doxygen download URL and the Pages publish

### Milestone: Production Hardening

*Validate, package, and distribute for external contributors. No new features beyond the frozen set.*

- [~] Llama HF-oracle parity — 1B FP32 (*Alpha.2*) and 3.2 3B BF16 (*Alpha.3*) were validated
  token-for-token against HuggingFace at delivery (see CHANGELOG), but that validation is **not captured
  as a permanent regression test** — Gemma has `GemmaModel.Parity.Cuda.cpp`, Llama has none — so it is
  unguarded against the inference-era API churn (OperationTraits, quantization, the `Operation` base
  collapse) that landed after it. Remaining: add the `LlamaModel` HF-parity regression test (the Gemma
  equivalent) covering 1B FP32 / 3B BF16, and formally validate + record 3.1 8B FP8 (not in the
  CHANGELOG). Folds into Test Suite Revival's Llama-path backfill
- [ ] Triage the `Llama.Block.ixx:132` view-aliasing concern in the primary validated target (the Q/K/V splits of `qkv_out` may not be contiguous) — confirm live-vs-benign and fix if live before claiming Llama HF validation. See BACKLOG, *Project Hygiene* marker bucket D
- [ ] Tool calling validated on Llama 3.2 3B and 3.1 8B Instruct
- [x] Gemma 4 12B FP4 fits a 12 GB card — both memory gates DONE (0.20.0-alpha.6+78): weight-tying (~2 GB reclaimed) + bounded-KV sliding-window ring (persistent-KV growth now 16 KB/token, the 8 global layers only). Coherent 8192-context chat with the ring engaged. Extended 2026-07-03 by activation pooling (shared block workspace, Gemma4InferenceReview.md section 7): the 48 per-layer activation buffer sets collapsed to one, retiring the chunk-32 operating point — prefill runs at chunk 512 via the activation-aware heuristic v2 (2048-token prefill 20.77 s -> 1.57 s same-day)
- [ ] `CONTRIBUTING.md` coding standards + `getting-started.md` onboarding guide
- [ ] Guided reading path — the comprehensibility deliverable: a document tracing one token's
  journey (embed -> attend -> sample -> decode) through the actual source, readable by a strong
  C++ developer unaided; distinct from `getting-started.md` (build/run onboarding)
- [~] Grammar-in-runtime consolidation — canonical C++ Gemma token grammar in the runtime, Chat
  consuming it, the Chat/MIS `<|"|>` drift closed (correctness consolidation, not a new feature —
  see BACKLOG, *Product Family — Grammar-in-Runtime Consolidation*). The C++ side is done (runtime
  `Dnn.Components.GemmaProtocol` module + Chat consuming it); the remaining MIS execution-time scope
  call moves to the *Product Family — Adaptor Validation* milestone below
- [x] An external consumer can build against Mila — **via FetchContent, the supported path (gate met).**
  Decided 2026-07-19: a C++23 module library is a *source distribution* (BMIs are not portable, so a
  consumer recompiles the module graph either way), which voids `find_package`'s prebuilt-binary benefit
  while carrying an install-layout apparatus and toolchain/ABI coupling. FetchContent compiles Mila once,
  in the consumer's own toolchain — `FetchContent_Declare(Mila GIT_REPOSITORY/URL ...)` +
  `FetchContent_MakeAvailable(Mila)` + link `Mila::Mila` — the way Mila already consumes its own deps
  (googletest, CUTLASS, nlohmann). This is validated green by the `packaging_fetchcontent_consumer` gate.
  **`find_package(Mila)` is PARKED** (retired in place, opt-in `MILA_ENABLE_FIND_PACKAGE_GATE` / `MILA_INSTALL`
  OFF by default) — a non-gate. See BACKLOG, *Packaging*
- [ ] Freeze the narrowest defensible public export surface — define the `Mila.ixx` allowlist, demote
  internal modules, stop re-exporting vendored `nlohmann`. At freeze the cost is asymmetric (too-broad
  can only be undone by a breaking removal). See BACKLOG, *Public API Surface*
- [~] Dispatch error UX — a missing/broken `(Op, Device, Precision)` specialization must read as a
  sentence, not a 200-line MSVC constraint cascade. **Core landed (0.20.0-alpha.6+112, VS2026 build
  green); optional (C) named kernel concepts + §12 spec reconcile remain:** the
  `OperationTraits` primary stays declaration-only, so an unsupported tuple names an incomplete type
  and the compiler emits a one-line "use of undefined type `OperationTraits<Op,Device,Precision,Policy>`"
  naming the exact tuple; the shared **`OperationSupported<...>` concept** (SFINAE-safe completeness
  probe, covers both `type`- and `op_for`-bearing specializations) lets a multi-precision typed test
  skip unsupported precisions via `if constexpr`. Note vs. the original sketch: a literal
  `static_assert(always_false)` *on the primary body* is mutually exclusive with a SFINAE-safe
  predicate (the probe would instantiate the primary and fire the assert), so the diagnostic rides the
  declaration-only primary + the predicate instead. Pairs with the poisoned-row drop under
  Consolidation. See BACKLOG, *Project Hygiene — Dispatch error UX*
- [ ] Add the Samples build to CI (only the tests build today) so a contributor's first sample build is not the thing that breaks
- [~] Linux build validated as a first-class platform — for v0.20 the supported Linux compiler is
  **Clang (19+/21), CUDA 13.3**. The WSL Clang oracle exists, CI compiles the tree under clang-21, and
  the dev container now builds Mila + runs Gemma 4 12B FP4 Chat on the GPU (2026-07-17). The full WSL
  `Build All` (library + tests + samples + Python binding) is now green under clang-21 (2026-07-18),
  after a batch of MSVC-invisible fixes — PIC on the static library so the binding `.so` links, the
  C++23 module direct-imports the umbrella does not re-export, and the two-phase-lookup `->template`
  disambiguator in the samples (see CHANGELOG). So the samples and binding — not just lib + tests — now
  compile clean under clang, de-risking their addition to CI. **GCC 16 as a second module oracle is
  deferred to vNext** — the full compiler matrix is a post-v0.20 hardening pass, not a beta gate, so the
  v0.20 Linux claim is Clang-only. See BACKLOG, *Module Hygiene* (cross-compiler oracle) and *Release
  Assets & CI* (broaden compiler coverage)
- [~] Reproducible container build — a pinned build container (CUDA `-devel` on Ubuntu 26.04) that
  builds Mila from a clean clone, so a contributor or CI reproduces the Linux build without host
  toolchain drift. This is the *build* environment; distinct from the runtime image below, which only
  packages the already-built artifacts. **Validated 2026-07-17:** `Docker/` reworked onto the CI
  toolchain (clang-21 modules + gcc-15 nvcc host, CUDA 13.3, CMake 4.2.3, Ninja, ccache); the image
  builds Mila + the Chat adaptor and runs Gemma 4 12B FP4 on the GPU at native decode speed (~49 tok/s),
  surfacing + fixing two clang-only transitive-import breaks (`Network.ixx`, `Gemma.ixx`). Extended
  2026-07-18: the container now also builds the full product set (`mila-build-all`) and the **MIS wire
  server** (`mila-build-mis` / `mila-mis`) — validated end-to-end, serving Gemma 4 12B FP4 on `:6452` with
  the FastAPI docs reachable from a host browser. **Remaining:** validated against the bind-mounted working
  tree rather than a from-scratch in-container `git clone`, and CI still apt-installs the toolchain rather
  than building `FROM` this image. See BACKLOG, *Module Hygiene* (dev-container build)
- [ ] Published Docker runtime image (slim multi-stage GPU runtime, release-tagged)
- [ ] Ungated GPT-2 quick-start path for zero-auth first run
- [ ] `good first issue` labels on GitHub

GPU-first: the CUDA backend is the validated inference path (HuggingFace is the correctness oracle);
full CPU op parity is not a gate. Engineering detail (packaging, module hygiene, public-API
narrowing, dispatch diagnostics, CI) lives in [BACKLOG.md](BACKLOG.md).

### Milestone: Product Family — Adaptor Validation

*Prove the locked product definition's central claim: the whole path is demonstrable end-to-end. The
[MilaProductFamily.md](Mila/Specifications/MilaProductFamily.md) definition ships two adaptors with
v0.20 — Chat (human gate) and MIS (Python wire) — distinguished by who closes the generation loop.
The release bar is that a foreign agentic harness can drive Gemma 4 12B FP4 through MIS with tool
use. The existing Chat surface is validated (net-new Chat feature work is its own milestone below);
this milestone closes MIS.*

The Agentic adaptor stays explicitly post-release (the named scope-creep vector). This milestone is
validation and a single execution-time scope call, not new adaptor surface.

**Success criteria:** a foreign harness (Codex CLI and Claude Code CLI over the OpenAI/Anthropic wire
shapes) drives Gemma 4 12B FP4 through MIS across plain-chat, single-tool, and tool-result-resume
flows with no leaked control tokens; the C++/Python grammar duplication is resolved by an explicit
decision (single-sourced via pybind, or pinned by a cross-language parity test).

- [~] MIS Gemma 4 tool-calling validated end-to-end through the foreign-harness test suite — the
  Codex + Claude Code CLI round-trips are live, and the Claude Code `/v1/messages` tool path is
  driving real tool calls (the "tool-blind today" note here was stale). **2026-07-16: the native
  grammar was reconciled against Google's canonical chat template — nine divergences found and
  fixed, one of which was silently corrupting every tool call carrying a list or object argument.**
  MIS now emits a prompt byte-identical to Google's own template renderer, pinned by a permanent
  oracle (see BACKLOG, *MIS -> Gemma 4 migration*). Remaining tail: N sequential distinct tool calls
  in one turn, channel-content parser polish, and direct Codex-CLI re-validation on the reconciled
  grammar
- [~] Grammar-in-runtime execution-time scope call — **decided in practice 2026-07-16: MIS stays on
  Python, pinned.** The C++ and Python grammars are now held together by a cross-language parity test
  (both suites assert the same golden literals — it caught them emitting different argument order for
  identical input), and MIS's prompt construction is additionally pinned to Google's vendored template
  as an independent oracle. **Left open for explicit user sign-off:** whether this closes the item or
  the pybind single-sourcing is still wanted for v0.20 — see BACKLOG, *Product Family — Grammar-in-Runtime Consolidation*

### Milestone: Chat

*Net-new Chat harness features and changes for v0.20. Chat is a first-class adaptor — the human-gate
peer of MIS under `Mila/Adaptors/Chat/` ([MilaProductFamily.md](Mila/Specifications/MilaProductFamily.md)),
a maintained surface that gains rigor over time, not a throwaway sample. This milestone owns feature
work beyond the current validated Chat surface (which the Adaptor Validation milestone above records
as done).*

> **Scope pending — full spec to be authored 2026-07-16.** The success criteria and task list below
> are placeholders; they are replaced from that spec once it lands (spec home: `Mila/Specifications/`).
> Per the locked product definition, Chat features stay human-gate concerns built on the runtime's
> compute primitives — the Agentic adaptor remains explicitly post-release.

**Success criteria:** _TBD from the Chat feature spec._

- [ ] _Scope TBD — itemized when the spec lands (2026-07-16)._

### Milestone: LanguageNetwork — Sample API

Move token sampling off the host and behind a clean model-level API. Today each `LanguageModel`'s
`onGenerating` copies the full logits tensor to the host every decode step and runs three
near-identical CPU `sampleToken` implementations. The replacement is a single `TokenSampler`
**orchestrator tool** owned by the `LanguageModel` base — the structural sibling of `Optimizer`
(model-owned, shares the model's `ExecutionContext`, **not** a graph `Component`) — dispatching a
device `CudaSamplingOp` / `CpuSamplingOp` through the unified `OperationTraits` table. Sampling runs
on the device: logits never leave it, and only the 4-byte int32 token is read back. Full design in
[TokenSampling.md](Mila/Specifications/TokenSampling.md).

One concrete `TokenSampler` carries temperature / top-k / top-p / min-p as composable per-call
*filters* (not a class per strategy); the `Sampler` base is the seam for a future stateful strategy
(e.g. Mirostat). A prerequisite refactor migrates the `Optimizer` dispatch off its legacy
`conditional_t` / `#ifdef` facade onto `OperationTraits`, so both model-level orchestrator tools
dispatch identically (see BACKLOG).

**Success criteria:** `onGenerating` samples on-device with the per-step D2H reduced from the full
logits tensor to 4 bytes; greedy decode reproduces the current host argmax token-for-token (Gemma
parity preserved); stochastic top-k / top-p validated against a host reference with a fixed seed and
injected uniform; the three copied host `sampleToken`s replaced by the one `LanguageModel`-owned
`TokenSampler`; `OperationTraits<SamplingOp, Cuda, {FP32, BF16}>` specializations live.

- [ ] Migrate Optimizer dispatch onto `OperationTraits` (follows the sampler — proves the pattern on working code)
- [x] `Sampler` base + `TokenSampler` facade (`Dnn.Samplers`) + `SamplingConfig`, retiring the `Dnn/Decoders` skeleton — green; skeleton files pending user deletion in VS2026
- [x] `CudaSamplingOp` (+ `CpuSamplingOp`) with `OperationTraits<SamplingOp, …>` specializations — greedy + full-multinomial + top-k/top-p, green. Perf rewrite SHIPPED 2026-07-03: the single-block correctness-first kernel (measured 11.05 ms/token at the Gemma vocab) replaced by a multi-block histogram-refinement + chunked inverse-CDF pipeline (55.5 us/token; sampled decode 25.6 -> 35.7 tok/s); the original kernel retained as the `forwardReference()` parity oracle
- [x] Injected-`r` unit oracle (`Tests/Dnn/Samplers/Sampling.Cuda.cpp`) — greedy exactness, inverse-CDF boundaries, determinism, top-k/top-p support restriction, softcap monotonicity (caught + fixed a top-k off-by-one)
- [x] **GemmaModel** on-device sampling — `TokenSampler` hoisted to the `LanguageModel` base (lazy, shared context) via `sampleNext()`; path A (host `sampleToken`) retired; `logits_staging_` + `decode_token_staging_` removed; per-step H2D restage gone; greedy validated token-for-token vs HostA, stochastic coherent in chat
- [ ] Migrate `LlamaModel` (mechanical mirror) + `GptModel` (GPT-2 variant) onto the base `sampleNext()` + delete their host `sampleToken`s — deferred to when those paths are next built/run
- [x] Phase D tail — decode-stream sharing SHIPPED + VALIDATED 2026-07-04 as **D1 decode-ahead**: `enqueueForward()`/`awaitToken()` run the sampler on the context stream with an async pinned readback + event, and `GemmaModel::onGenerating` pipelines forward N+1 ahead of token N's host readback — per-token `synchronize()` deleted, host removed from the per-token path (event-sync 79 us avg). Measured recovery ~0.2-0.3 ms/token: the calibration's "host gap" was mostly launch micro-gap tax (~1165 kernels/token x ~1.3 us), which re-ranks D2 fusion / CUDA Graphs as the next decode lever (review section 4.1). Single-block kernel perf optimization shipped 2026-07-03 (see the `CudaSamplingOp` item above)

### Milestone: Generation API

The Sample API milestone moved sampling onto the device; this one makes `LanguageModel::generate` a lean,
fast **token generator**: prompt tokens in, tokens out through a push callback, and the one fact only the
model knows — *why generation stopped* — returned. Everything the caller can observe for itself (timing,
throughput, token counts) or that belongs to the model/sampler lifetime (the stop set, the RNG seed) is
kept off the per-call path. Sessions, prompt caching, and multi-conversation routing stay *harness* concerns
the apps (Chat, the Mila Inference server) build on the compute primitives (`prefill`/`decode`, later
`prefillFrom`/`rewindKvCache`) — the library exposes the primitives, not the policy.

An initial config-in / result-out reshape landed first, but a **design review (2026-07-01) superseded it**
with the leaner surface below, which **shipped in 0.20.0-alpha.6+79** (green build + green Gemma chat).
Rationale + the decided micro-choices in [BACKLOG.md](BACKLOG.md). The family term stayed `Generate*` (the
code already used it). Shipped signature:

```cpp
[[nodiscard]] GenerateStatus generate(
    std::span<const int32_t> prompt_tokens,
    const std::function<void( int32_t )>& on_token,
    const GenerateParams& params = {},
    std::stop_token stop = {} );
```

- [x] collapsed `generate` + `generateStreaming` + the vector-returning `generate` into the single `generate` above (one blocking, callback-streaming primitive); `onGenerating` kept as the protected hook, retyped to `GenerateStatus` / `GenerateParams`
- [x] `generate` returns `GenerateStatus` — the finish reason, the only result the model uniquely knows; `GenerateResult` deleted (the status + stats + count bundle dissolved)
- [x] `GenerationStatistics` deleted **entirely** — the model owns no stopwatch; Chat + ProfileModel self-time from the callback cadence (TTFT = call -> first token, decode = first -> last)
- [x] the mutable `last_generation_statistics_` member + `getLastGenerationStatistics()` are gone — stays gone, never reintroduced
- [x] `GenerateParams { std::optional<int> max_new_tokens; SamplingParams sampling; std::vector<TokenId> stop_tokens; }` — `max_new_tokens` nullopt => run to EOS / context bound (no magic default, no silent truncation)
- [x] `SamplingParams { float temperature; int top_k; float top_p; }` in its own module (`Dnn.SamplingParams`); `generate` forwards only `params.sampling` to the sampler; the `using SamplingParams = GenerateConfig` alias retired
- [x] stop set off the per-call path — model defaults established at construction, with an optional per-call `stop_tokens` override for advanced structured generation
- [x] `seed` off the per-call path — public `LanguageModel::seedSampler(uint64_t)` seeds the sampler once; per-call `seed` removed
- [x] `top_p` reachable end-to-end through the public `generate` API
- [x] `Generate*` naming family kept (code already used it); one type per module (`GenerateResult` / `GenerationStatistics` deleted, `SamplingParams` split out)

Remaining to close the milestone:

- [ ] `SamplingConfig` -> `SamplerConfig` rename (deferred this pass — highest-risk cross-module rename)
- [ ] Llama/Gpt reproducibility — their host sampler is time-seeded only until the deferred device-sampler migration wires `seedSampler`
- [ ] eager sampler construction before the first generation (the lazy first-use allocation still adds first-token latency the harness measures)
- [ ] hoist `contextLength()` to the `LanguageModel` base + make it mode-aware (inference -> deployment ctx, training -> arch max)
- [ ] propagate the `getNetworkConfig()` / `getModelConfig()` accessor pair to `LlamaModel` / `GptModel` (mechanical mirror; Gemma done — stores `GemmaModelConfig`, derives `contextLength()`, bare `context_length_` gone)
- [ ] `GemmaTransformer::getConfig()` network self-description hygiene; settle the `int64_t`-vs-`dim_t` static_cast smell in the same pass

**Deferred — harness layer, not a milestone gate:** prompt-caching / KV reuse (`prefillFrom` / `rewindKvCache`)
built into an app-level `GenerateSession`; its bounded-KV-ring gate closed 2026-06-30, so it is unblockable
but out of scope here.

### Milestone: Gemma 4 Inference Competitiveness (Prefill + Decode)

*A craft goal, **not a release gate.** Mila is a mastery project, not a llama.cpp/vLLM competitor — the
release ships regardless of where this lands. This milestone records the flash-attention + FP8-GEMM
prefill work and the decode campaign delivered in the alpha.6 line, plus the deferred levers, so the
shipped result reads as done rather than aspirational.*

The target was framed as "close, not miles behind" on Gemma 4 12B FP4 prefill at 48K context (4070,
22496 tokens), measured against llama.cpp (LM Studio, Q4_K_M) as the external reference. That criterion
is **MET**: the campaign drove the gap from **1.95x → 1.136x behind** (39208 → 12382 ms; llama.cpp 10903
ms / 2063 tok/s → Mila 1817 tok/s), through a stacked sequence of tensor-core flash-attention kernels
and an FP8-activation GEMM path. Crossing *under* llama.cpp is a further stacked campaign whose biggest
remaining bucket (global flash, ~36% of prefill) sits at the Ada architectural floor — so the remaining
levers are **deferred by choice** ("scrape the bones later"), not blocked.

**Success criterion (MET):** Gemma 4 12B FP4 prefill "close, not miles behind" llama.cpp at 48K context
— achieved 1.136x (was 1.95x). Full gap map, ncu evidence, and lever arithmetic live in
[BACKLOG.md](BACKLOG.md) *Gemma 4 — Dense Chassis* and the perf memory files.

- [x] Bounded-ring tensor-core flash prefill on the 40 local sliding layers (+101, then a row-split FA-2 kernel at +104) — 1.71x → 1.136x
- [x] W4A8-FP8 activation prefill GEMM shipped ON (+103) — per-token activation scales, stale-`sB` root cause fixed; 1.50x → 1.17x
- [ ] **[deferred — needs core-API sign-off]** Lever A: FP4→FP8 upcast hoist (~−1.0s) — layer-outer/chunk-inner prefill restructure + an `IDecoderLayer::beginPrefillPass/endPrefillPass` hook; projects to ~1.08x, does not alone cross the line
- [ ] **[deferred]** Remaining levers — `warps=7` local-FA-2 probe, global FP8-K/V-in-smem two-block (borderline), small fusions (rope/rmsnorm+residual/epilogue), global-flash ILP, local-ring L2 rework. Any of Lever A + fusions + one attention nibble lands at/under the line

**Decode campaign (2026-07-16, +105):** Gemma 4 12B FP4 decode **38.65 → 49.09 tok/s @32K (+27%)**,
40.15 → 48.87 @4K. Decode no longer scales with *allocated* context (the 4K/32K spread is gone). The
GQA attention bucket fell 4.63 → 0.37 ms/token (12.5x). Remaining wall is FP4 weight bytes at ~92% of
DRAM peak (the format floor); measured next levers are RMSNorm fusion (2.36 ms, 337 launches/token) and
a CUDA-Graphs decode step (~1.4 ms launch-gap tax), projecting ~56 tok/s against a ~62–65 format ceiling.

- [x] FP4 decode matvec inner-loop diet — PRMT byte-permute nibble decode (no constant-LUT replay), per-group scale fold, dual accumulators, one-ahead load pipeline; 63–78% → 79–90% of DRAM peak
- [x] Fused decode-attention kernel (`Gqa.Decode.Bf16.cu`) — streaming online-softmax over the live band only (`slot = position % capacity` serves unbounded and ring caches), split-K across blocks + fixup merge for the MQA global layers, cp.async double-buffered K/V tiles, tile-granular rescale; replaces the cuBLASLt QK→softmax→AV pipeline and its T=1 identity permutes; `CudaGqaDecodeParity` oracles + sanitizer trio green
- [ ] **[deferred]** RMSNorm fusion + CUDA-Graphs decode step (the two measured levers above); FP8 KV cache re-scoped as a memory/long-context enabler (~1 GB at 64K), no longer a decode-perf lever

---

## vNext — Qwen 3

**Release Date:** _Target — 2027 (range; version and tag assigned at promotion)_

Mila's third architecture family: Qwen 3 dense decoder with thinking mode, model-agnostic tool
calling, and FP8 KV cache compression — validated on Qwen 3 8B Instruct at BF16 and FP8. Reuses the
Llama blocks (RMSNorm, SwiGLU, GQA, RoPE); the new work is the Chat layer (ChatML template,
`ToolCallParser`, thinking-mode suppression) and FP8 KV cache (`PerChannelKvFp8<>`).

**Success criterion:** greedy decode of Qwen 3 8B Instruct at BF16 and FP8 each match HuggingFace
token-for-token; tool calling validated end-to-end; thinking-mode suppression confirmed; FP8 KV
cache quality acceptable vs. the BF16 baseline.

Tasks are itemized when the milestone opens.

Carried-over hardening (deferred from v0.20): add GCC 16 as a second, independent C++23-module oracle
(distinct strictness from Clang — two-phase lookup, export/linkage — so it catches a portability class
both MSVC and Clang miss) and wire it into CI, broadening the Linux compiler matrix beyond the
Clang-only v0.20 claim.

---

## Future Directions

Uncommitted vision — no milestone, no date. An item **promotes** into a real milestone (its own
version, date, GitHub Milestone) when it is scheduled.

- **Ministral** — Ministral transformer with Sliding Window Attention; 3B Instruct (BF16) and 8B
  Instruct (FP8). Builds on the Llama foundation and the Qwen 3 tool-calling pipeline, and reuses the
  SWA mask + bounded-KV ring cache landed by the Gemma 4 milestone.
- **Training (advanced)** — beyond the revived GPT-2 / MLP training foundation now in v0.20: a full
  LLaMA fine-tuning pipeline, loss-function GPU migration, gradient checkpointing, and checkpoint
  save/restore.
- **Architecture** — Mixture-of-Experts components (the `GatedMLP` reusable gated FFN, the grouped
  `MoeOp` over stacked expert weights, `Router` + `MixtureOfExperts`; design and the MoE-readiness
  seams already specified in `Specifications/FfnAndMoE.md`, with the FFN-layer foundation landing in
  v0.20 Consolidation). The **Gemma 4** dense chassis delivered in v0.20 is the precursor that de-risks
  this: the 26B-A4B MoE model reuses the Gemma chassis, swapping only the FFN block. Also: speculative
  decoding, additional attention variants.
- **Performance** — tensor parallelism and deterministic gradient accumulation for training
  reproducibility. (Tensor-core flash-attention prefill already shipped in the v0.20 alpha.6 line —
  see the *Gemma 4 Prefill Competitiveness* milestone above; what remains there is deferred by choice.)
