# Mila — Roadmap

Where Mila is going — the durable narrative of each release and what it means.

- **Open tasks** -> [BACKLOG.md](BACKLOG.md) · **Completed work** -> [CHANGELOG.md](CHANGELOG.md)
- **How versions, milestones, labels, and releases work** -> [RELEASING.md](RELEASING.md)
- **Design rationale** -> `Mila/Specifications/`

The roadmap shows two releases at a time — the one in flight and the one after (**vNext**) — plus a
**Future Directions** tail. Each release is reached through **milestones** tracked by task completion
(see [RELEASING.md](RELEASING.md)). Current version: **`0.20.0-alpha.6+79`**.

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
- [~] OperationTraits close-out — retire the legacy dispatch files in place (out of build + `RETIRED` banner, not deleted): done for the Linear/Gqa typemaps, registry/registrar helpers, `OperationsRegistrar`, `FusedComponent`; `OperationRegistry` + the arity bases remain until the Training Revival loss-path re-authoring (still referenced by the disabled CrossEntropy/fused ops kept in `Src`). `Dropout` parked at `Dev/Components/Regularization/`; its `DropoutOp` re-authoring moves to Training Revival — training-only, net-new op work outside this freeze
- [~] Marker burndown — all `FIXME:`/`TODO:` converted to `REVIEW:` (2026-06-19) so public source reads clean; the ~94 surviving `REVIEW:` markers are triaged into buckets A-H with dispositions and homes in the BACKLOG "Marker debt triage" item. Remaining burndown: bucket-C `dim_t` canonicalization, bucket-D correctness items (incl. the `Llama.Block.ixx:132` aliasing check in the primary target), bucket-H org/docs, and demoting the E/F/G design/lifecycle/nit notes. The old lifecycle linchpin (the `CompositeComponent` setTraining/build concern) was already resolved by the RuntimeMode/TrainingMode two-axis redesign — `isTraining()` is gone
- [x] Debug-instrumentation strip — kernel anomaly guards removed (Residual/Gelu/LayerNorm/RmsNorm; `Swiglu.Fp32.v1.cu` deleted) and BPE tokenizer console routed to `Logging::Logger` or removed; the two inert commented-out CPU-op debug dumps (`CpuAttentionOp` K/dQ, `CpuLinearOp` input) deleted (0.20.0-alpha.6+81). The remaining console sites are deliberate/legitimate, kept: the `TokenSequenceLoader` init summary (gated on `config_.verbose_logging`) and `TensorBuffer` alloc trace (`if constexpr (TrackMemory)`) are opt-in diagnostics; `CudaExecutionContext` stderr-on-stream-error and the `TokenEmbeddingConfig` `isfinite` validate-guard are error/validation paths. Training-path instrumentation (AdamW anomaly `printf`, `BpeVocabulary`/`OptimizerBase`/`TrainingLogger`) deferred to Training Revival
- [x] Debug-instrumentation strip — per-step `synchronize()` removed from `GptBlock`/`LlamaBlock` `forward()`/`backward()` (single-stream ordering + the caller's host-read boundary sync suffice). Llama `prefill`/`decode` were already sync-free; `GptBlock::decode()` still carries its 8 per-op debug syncs (GPT-2 inference path) — the same correctness-neutral strip, left for when the Bard/GPT-2 decode path is next exercised
- [~] FFN consolidation — de-polymorphize `MLP`: the **blocking** runtime activation dispatch is gone (`MLP` holds a concrete `Gelu` child; the `mlp_activation_impl` / `std::function` bridge / SwiGLU branch retired in `MLP.Dispatch.ixx`, out of build), `Swiglu` -> `FFN/Swiglu/` and `MLP` -> `FFN/MLP/` relocated, and `MLP<Cpu>` / `GptBlock<Cpu>` / `GptTransformer<Cpu>` now build (their `.Cpu.cpp` tests are active). **Remaining (deferred):** fold the fixed `Gelu` child onto the generalized `Activation<…, ActivationType TFn = Gelu>` — the compile-time-`TActivation` endgame — tracked under the `Activation` elementwise-primitive item below. Design: `Specifications/FfnAndMoE.md`
- [~] FFN consolidation — `Activation` elementwise primitive: compile-time `TFn`, the `MILA_HD` functor library, and the Cpu/Cuda `ElementwiseActivationOp` (functor-templated, not a traits axis); `Gelu` folds in. **Landed alongside `Gelu`:** the functor library (all 8 functions), `Cpu`/`Cuda` `ElementwiseActivationOp` (FP32 / FP32+BF16), the member-template `op_for<Functor>` traits, the `Activation<…, TFn>` component + config, and its CPU/CUDA tests. The reusable `GatedMLP` gated FFN (single-expert reference, `TGate` fixed to SiLU) landed with it. **Remaining:** fold `Gelu` (+ `MLP`'s child) onto `Activation`; CPU `SwigluOp` + `Swiglu<…, TGate>` generalization — tracked in BACKLOG
- [ ] Hardening — couple the `initialize_parameters` default to `RuntimeMode`

**Exit:** every box checked, no literal `FIXME` in public source, debug instrumentation gone, and the
Component lifecycle is sound enough to re-enable the component/training tests.

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
- [~] Gradient-check archetype (finite-difference numeric backward) — shared fixture `Common/GradientCheck.h` + `Gelu`/`LayerNorm` reference applications landed; remaining: fan `Backward_MatchesNumericGradient` out to Linear / MLP / Residual / GptBlock and use it to isolate the suspected MHA CPU backward. Precondition for Training Revival's convergence oracle
- [ ] Backfill the inference-drought coverage the old suite never had — load-time quantization (`PerChannelFp8` / `PerGroupFp4`, the decode matvec kernels), `OperationTraits` dispatch, and the Llama path (RmsNorm / SwiGLU / GQA / RoPE components + `LlamaModel::fromPretrained`); the `CudaLinearOp` quantization white-box is the sole legitimate op-layer test. Genuinely new, not recovery
- [~] Re-green in sample-revival order — MNIST spine first, Bard spine second (mirrors Training Revival sequencing). MNIST spine mostly landed (`CompositeComponent`, `AdamW.Cpu`, `DataLoader`(+`.Cuda`), `Network.Cpu`, `TensorOps/Random.Cpu`); remaining: the thin `Core/Network.cpp` delta + the GPU companions (`Network.Cuda` / `AdamW.Cuda`), then the Bard GPT-2 stack tail
- [ ] Verify the full suite green in one pass (CPU-only `MILA_ENABLE_CUDA=OFF` and the CUDA build) — much of the above landed "awaiting VS2026 build"; establish the verified-green baseline the CI gate wires against
- [ ] **[gate]** Wire the suite into CI as the anti-rot ratchet, building on the `MILA_ENABLE_CUDA=OFF` CPU-only gate so a future API churn fails the build instead of silently re-commenting coverage — the deliverable that keeps the revival alive

### Milestone: Training Revival

*Resurrect the validated GPT-2 / MLP training path — MNIST and Bard — to current-API quality, proven
by its own revived tests. Scope is GPT-2 / MLP only; Llama 3.1/3.2 training stays a Future Direction.*

MNIST (MLP) and Bard (GPT-2 generation) were complete, working training samples, parked behind an
explicit `FIXME: Re-enable after alpha.5 completed` trigger that has now fired. Reviving them
reactivates the half of the library inference never exercises: the AdamW optimizer, the loss and
backward kernels, gradient flow, and train-from-scratch parameter initialization (the per-device
init subsystem was restored in Alpha.5). The revived tests from the previous milestone are the
oracle — a sample "converges" only when its test says so.

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

**Success criteria:** the MNIST and Bard samples re-enabled in the build and running against the
current API; MNIST trains to its target accuracy and Bard generates coherent text; a per-sample
end-to-end convergence test (loss strictly decreases over a fixed step budget) is the green/red
oracle, backed by the gradient-check archetype, the AdamW step-convergence test, and the concrete
data-loader contract tests; train-from-scratch validated at the precisions the samples use; all
training-path tests CI-gated.

Status: the hard part is done — **both samples are revived and validated** (by eye, on CUDA). The
remaining work is the **test oracle** that makes their convergence provable, plus the deferred
training-only pieces (loss path, Dropout, progress reporting). The net-new test items below overlap
the Test Suite Revival gradient-check fan-out and CI ratchet — the two revival milestones are one
work-front.

- [x] Revive the MNIST (MLP) sample to the current API + validate — builds green (VS2026/CUDA), trains FP32 from scratch to ~97.9% test accuracy over 20 epochs; the full spine (forward chain, backward gradient flow, AdamW step, train-from-scratch init) exercised end-to-end. MNIST spine tests (Linear / Gelu / Network / `AdamW.Cpu` / DataLoader) green
- [x] Revive the Bard (GPT-2) sample to the current API + validate — builds green (VS2026/CUDA), trains FP32 to perplexity <3 / loss ~1.09 by epoch 17 with coherent Shakespeare-structured text; surfaced + fixed 3 latent CUDA-training-backward bugs (cuBLASLt bias epilogue, inverted attention eval-guard, `TensorOps` math no-op)
- [x] Flip the `FIXME: Re-enable after alpha.5 completed` triggers — both samples re-enabled in `Samples/CMakeLists.txt` (CUDA-gated until the CPU-only build-coherence work lands)
- [~] Concrete data-loader contract tests — `TokenSequenceLoader` done (construction / iteration / reset / target-is-input-shifted / threading stress); remaining: the `MnistDataLoader` contract test (pixel normalization, one-hot targets, shuffle-on-reset, IDX magic-number validation)
- [~] Re-enable + re-align the AdamW path — `AdamW.Cpu.cpp` re-greened (Section 1, includes a closed-loop convergence case); remaining: the `AdamW.Cuda.cpp` companion + resolve the deferred AdamW debug-instrumentation strip-vs-gate (`CudaAdamW.cu` `printf` guards + `CudaAdamWOptimizer.ixx:270`)
- [ ] **[net-new]** End-to-end convergence oracle — a per-sample integration test (build the tiny model, run a fixed step budget on a few batches, assert the loss strictly decreases; for MNIST also that train accuracy rises). The literal milestone exit ("a sample converges only when its test says so"); MNIST first, Bard second; keep the budget small so it runs in the `MILA_ENABLE_CUDA=OFF` CI gate
- [ ] **[net-new]** Optimizer step-convergence test — "minimizes a known convex objective in N steps," proving the update direction + bias-correction are correct, not just that `step()` runs
- [ ] **[net-new]** TrainingMode / RuntimeMode behavior coverage — assert build-mode and runtime-mode transitions allocate/skip gradient buffers correctly, so the two-axis lifecycle fix has a regression guard
- [ ] Fix the CUDA `fill_normal` / `fill_uniform` FP32-only gap (corrupts BF16 train-from-scratch init) — the CUDA counterpart to the `CpuTensorOps.Random` backend; pair with a BF16 init-at-precision `TYPED_TEST` that turns the silent corruption into a red test
- [ ] **[decoupled]** Revive the loss + backward path (CrossEntropy / SoftmaxCrossEntropy) — both samples compute loss host-side, so this is off the critical path to a converging sample; the dispatch struct was started (alpha.6+68) but is not wired in
- [ ] **[net-new, training-only]** Revive the `Dropout` component from `Dev/Components/Regularization/` — re-author `CpuDropoutOp` / `CudaDropoutOp` + `OperationTraits` rows + the two-axis `Component<TDeviceType, TPrecision>` rewrite; the mask/backward path is exercised only by training
- [ ] ProgressReporter mechanism — an injected per-operation progress facility for long-lived ops (BPE vocab training, `PretrainedReader` load, load-time quantization); the Consolidation debug strip left the BPE training progress in place to migrate here
- [ ] Validation — MNIST/Bard convergence proven **by test** (not by eye), train-from-scratch validated at the precisions the samples use, and all AdamW / training-path tests green and CI-gated

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

- [ ] **Narrow the published surface first** — scope the Doxyfile from `EXTRACT_ALL`/`EXTRACT_PRIVATE`/`EXTRACT_STATIC` over all of `Mila/Src` to the public `import Mila;` allowlist, so internal ops/kernels/registries drop off the documented surface. Shrinks the denominator before anything is counted; pairs with the *Public API Surface* `Mila.ixx` allowlist work
- [ ] **Oracle** — enable `WARN_IF_DOC_ERROR` + `WARN_NO_PARAMDOC` in `Mila/Docs/Doxyfile.in` (no `WARN_*` knobs set today) to mechanically generate the drift worklist and give a shrinking warning count as the definition of "done". Highest-leverage item — it both drives and locks the tiers below
- [ ] Tier 0 — non-ASCII / mojibake in comments (scriptable, no judgment); fold in the `Comonent.TrainingMode.ixx` -> `Component.TrainingMode.ixx` file rename
- [ ] Tier 1 — `@file` rename drift: the 34 files whose `@file` tag != filename (correct value is `basename`; fully scriptable)
- [ ] Tier 2 — `@param`/`@tparam` name mismatches vs. the signature — batch-fix from the Oracle's `WARN_IF_DOC_ERROR` candidate list (review before applying; signatures span lines)
- [ ] Tier 3 — semantic staleness (retired-world prose: `OperationRegistry`/`UnaryOperation`, `TWeightQuant`-style naming drift, over-long file `@brief`s). Per-subsystem judgment; **folded into Test Suite Revival** — fix a file's prose while it is open for re-greening, not as a separate megasweep
- [ ] **Ratchet** — once the warning count is zero, flip `WARN_AS_ERROR = FAIL_ON_WARNINGS_PRINT` in the docs job so doc drift fails the build (the docs analogue of the test-CI ratchet)
- [ ] Docs-site CI — decouple the docs job from the CUDA-dependent CMake configure (standalone Doxyfile or CUDA-free docs configure); bump Doxygen 1.15 -> 1.17; verify C++23 module units render faithfully; publish via GitHub Action from `master`, never committing generated docs to the tree

### Milestone: Production Hardening

*Validate, package, and distribute for external contributors. No new features beyond the frozen set.*

- [ ] Llama 3.2 1B FP32, 3.2 3B BF16, 3.1 8B FP8 validated against the HuggingFace oracle
- [ ] Tool calling validated on Llama 3.2 3B and 3.1 8B Instruct
- [x] Gemma 4 12B FP4 fits a 12 GB card — both memory gates DONE (0.20.0-alpha.6+78): weight-tying (~2 GB reclaimed) + bounded-KV sliding-window ring (persistent-KV growth now 16 KB/token, the 8 global layers only). Coherent 8192-context chat with the ring engaged
- [ ] `CONTRIBUTING.md` coding standards + `getting-started.md` onboarding guide
- [ ] `find_package(Mila)` validated by an external consumer build
- [ ] Published Docker runtime image (slim multi-stage GPU runtime, release-tagged)
- [ ] Ungated GPT-2 quick-start path for zero-auth first run
- [ ] `good first issue` labels on GitHub

GPU-first: the CUDA backend is the validated inference path (HuggingFace is the correctness oracle);
full CPU op parity is not a gate. Engineering detail (packaging, module hygiene, public-API
narrowing, dispatch diagnostics, CI) lives in [BACKLOG.md](BACKLOG.md).

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
- [x] `CudaSamplingOp` (+ `CpuSamplingOp`) with `OperationTraits<SamplingOp, …>` specializations — greedy + full-multinomial + top-k/top-p, green (single-block correctness-first kernel, threshold binary search; perf optimization deferred)
- [x] Injected-`r` unit oracle (`Tests/Dnn/Samplers/Sampling.Cuda.cpp`) — greedy exactness, inverse-CDF boundaries, determinism, top-k/top-p support restriction, softcap monotonicity (caught + fixed a top-k off-by-one)
- [x] **GemmaModel** on-device sampling — `TokenSampler` hoisted to the `LanguageModel` base (lazy, shared context) via `sampleNext()`; path A (host `sampleToken`) retired; `logits_staging_` + `decode_token_staging_` removed; per-step H2D restage gone; greedy validated token-for-token vs HostA, stochastic coherent in chat
- [ ] Migrate `LlamaModel` (mechanical mirror) + `GptModel` (GPT-2 variant) onto the base `sampleNext()` + delete their host `sampleToken`s — deferred to when those paths are next built/run
- [ ] Phase D tail — share the decode stream (real `getStream()`; currently the op runs on the default stream) + single-block kernel perf optimization

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
- **Performance** — Flash Attention integration, tensor parallelism, deterministic gradient
  accumulation for training reproducibility.
