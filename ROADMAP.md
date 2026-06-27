# Mila — Roadmap

Where Mila is going — the durable narrative of each release and what it means.

- **Open tasks** -> [BACKLOG.md](BACKLOG.md) · **Completed work** -> [CHANGELOG.md](CHANGELOG.md)
- **How versions, milestones, labels, and releases work** -> [RELEASING.md](RELEASING.md)
- **Design rationale** -> `Mila/Specifications/`

The roadmap shows two releases at a time — the one in flight and the one after (**vNext**) — plus a
**Future Directions** tail. Each release is reached through **milestones** tracked by task completion
(see [RELEASING.md](RELEASING.md)). Current version: **`0.20.0-alpha.6+73`**.

The **Gemma 4 12B dense chassis** has been delivered into v0.20 (HF token-for-token parity, 2026-06-23
— see CHANGELOG); its remaining memory-fit work (bounded-KV ring cache + weight-tying) folds into
Production Hardening as a release gate, and the 26B-A4B MoE follow-on stays a Future Direction.

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
- [ ] Debug-instrumentation strip — kernel anomaly guards removed (Residual/Gelu/LayerNorm/RmsNorm; `Swiglu.Fp32.v1.cu` deleted) and BPE tokenizer console routed to `Logging::Logger` or removed; remaining: the `TokenSequenceLoader` verbose dump and the already-gated diagnostics; training-path instrumentation (AdamW, `BpeVocabulary` progress) deferred to Training Revival
- [ ] Debug-instrumentation strip — per-step `synchronize()` removed from `GptBlock`/`LlamaBlock` `forward()`/`backward()` (inference `prefill`/`decode` already run sync-free; single-stream ordering + the caller's host-read boundary sync suffice)
- [ ] FFN consolidation — de-polymorphize `MLP` to a compile-time `Activation<…, TActivation = Gelu>` dense FFN (drop the runtime activation switch / `mlp_activation_impl` / `std::function` bridge / SwiGLU branch / `fc1` doubling / dead LayerNorm), unblocking `MLP<Cpu>` / `GptBlock<Cpu>` / `GptTransformer<Cpu>`; relocate `Swiglu` `Activations/` -> `FFN/Swiglu/`; relocate `MLP` into `FFN/MLP/`. Design: `Specifications/FfnAndMoE.md`
- [~] FFN consolidation — `Activation` elementwise primitive: compile-time `TFn`, the `MILA_HD` functor library, and the Cpu/Cuda `ElementwiseActivationOp` (functor-templated, not a traits axis); `Gelu` folds in. **Landed alongside `Gelu`:** the functor library (all 8 functions), `Cpu`/`Cuda` `ElementwiseActivationOp` (FP32 / FP32+BF16), the member-template `op_for<Functor>` traits, the `Activation<…, TFn>` component + config, and its CPU/CUDA tests. The reusable `GatedMLP` gated FFN (single-expert reference, `TGate` fixed to SiLU) landed with it. **Remaining:** fold `Gelu` (+ `MLP`'s child) onto `Activation`; CPU `SwigluOp` + `Swiglu<…, TGate>` generalization — tracked in BACKLOG
- [ ] Hardening — couple the `initialize_parameters` default to `RuntimeMode`

**Exit:** every box checked, no literal `FIXME` in public source, debug instrumentation gone, and the
Component lifecycle is sound enough to re-enable the component/training tests.

### Milestone: Test Suite Revival

*Re-green the authored test suite to the current API and gate it in CI — the anti-rot ratchet and
the correctness oracle for everything after it.*

The first year of Mila was test-driven; ~70 test files exist in the tree, but only ~24 are active.
The rest were commented out during the inference-era refactors (the CMake note is explicit: "too
many tests to refactor for Component lifecycle changes"). This is recovery, not greenfield — the
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

### Milestone: Production Hardening

*Validate, package, and distribute for external contributors. No new features beyond the frozen set.*

- [ ] Llama 3.2 1B FP32, 3.2 3B BF16, 3.1 8B FP8 validated against the HuggingFace oracle
- [ ] Tool calling validated on Llama 3.2 3B and 3.1 8B Instruct
- [ ] Gemma 4 12B FP4 fits a 12 GB card — bounded-KV ring cache + weight-tying (the two memory gates in BACKLOG)
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

- [ ] Migrate Optimizer dispatch onto `OperationTraits` (follows Phase A — proves the pattern on working code)
- [~] `Sampler` base + `TokenSampler` facade (`Dnn.Samplers`) + `SamplingConfig`, retiring the `Dnn/Decoders` skeleton — **landed (Phase A); skeleton files pending user deletion in VS2026**
- [~] `CudaSamplingOp` (+ `CpuSamplingOp`) with `OperationTraits<SamplingOp, …>` specializations — **greedy/argmax landed (Phase A); stochastic top-k/top-p are Phase B/C**
- [~] Wire `TokenSampler` into `onGenerating` behind the `SamplingPath` A/B toggle — **`GemmaModel` wired (Phase A, default `HostA`); `LanguageModel`-base hoist + `logits_staging_` removal are Phase D**

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
