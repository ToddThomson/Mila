# Mila — Backlog

Open engineering tasks not yet completed. This is the working task list.

- **Milestone vision and success criteria** live in [ROADMAP.md](ROADMAP.md).
- **Completed, validated work** lives in [CHANGELOG.md](CHANGELOG.md).
- **Design rationale** lives under `Mila/Specifications/`.

Each `##` section is a **milestone**, named by **theme** — the section title matches its ROADMAP
milestone title exactly, and that shared name is the only join between the two files. Milestones are
namespaced by their planned release in ROADMAP and on GitHub (see [RELEASING.md](RELEASING.md)); this
file carries **no stage or release numbers** and no per-task index. Cross-references point at a
milestone by theme name or at a task by a short descriptive phrase, never by a number. (`Alpha.5` and
`alpha.5` left in the text are historical references to a shipped release, not milestone labels.)

As contributors arrive (the `0.20` production cycle), the contributor-facing items here are
promoted to `good first issue` discovery Issues (a GitHub mechanism, distinct from inbound
user issues); the rest stay in this file. Until then it is the single flat backlog for a solo
maintainer.

Legend: **[gate]** blocks the named milestone · **[deferred]** parked, revisit on the
stated trigger · **[contributor]** good-first-contribution / demand-driven, not a release gate.

---

## Beta.1 Exit Checklist (release-gate rollup)

Not a milestone — a **synthesis view** of what honestly gates the `alpha.6 -> beta.1` stage flip.
Per [RELEASING.md](RELEASING.md), `beta.X` means **"feature-frozen; hardening only"**, and a beta is
a **trust signal** ("the project not contradicting itself"). So this is *not* the full craft-complete
v0.20 scope — most ROADMAP milestones are either done, self-declared non-gates, or feature work you
**freeze** rather than finish. Each line points at where the real task lives; do not double-book status
here — tick the owning milestone item, and re-derive this rollup from it.

**Decision (not code) — the biggest unblock:**
- [x] **Freeze the feature set — DECIDED 2026-07-18 (user, emphatic: "no more additions, period").**
  A hard freeze — nothing is "finished," every open *feature* milestone is **deferred to vNext**: the
  *Generation API* tail (SamplerConfig rename, Llama/Gpt seedable sampling, eager sampler, accessor
  propagation), the *LanguageNetwork — Sample API* Llama/Gpt migration, and the **unspecced *Chat*
  milestone** (deferred whole — no spec authored, no surface added). This is the beta.1 posture
  (`beta.X` = feature-frozen, hardening only). What stays in v0.20 is **not** feature work: Test Suite
  Revival, Training Revival (recovery of already-validated samples), API Documentation, Production
  Hardening, and MIS Adaptor Validation — the path *to* beta. New "nice additions" go to the vNext /
  Future section, never into v0.20.

**Engineering gates (the three that are real work):**
- [x] **Close Consolidation — DONE 2026-07-18 (0.20.0-alpha.6+112..+113).** The milestone that literally
  "earns the right to call it beta." The hard item — the **poisoned BF16 dispatch rows** — dropped at +112
  (VS2026 green), landed with the **Dispatch error UX** `OperationSupported<...>` predicate. The four
  scope-complete `[~]` remainders (legacy-dispatch retire, marker burndown, both FFN items) were ticked
  at +113 with their net-new/training remainders relocated to vNext / Training Revival under the feature
  freeze; the last stray literal `FIXME` (GQA `forward()` dead-branch) reworded, and the orphaned
  `Dnn/Decoders/` skeleton moved out of the tree. See ROADMAP *Consolidation* (CLOSED note).
- [~] **Test Suite Revival CI ratchet — wired (0.20.0-alpha.6+114, awaiting first CI run).** The
  correctness keystone: full suite green in one pass + the `MILA_ENABLE_CUDA=OFF` CPU-only gate (see
  *Test Suite Revival*, the `[gate]` item). CPU-only leg verified 2026-07-18 (from-scratch clang-21
  build clean, ~980 ctest cases green in one pass); the `cpu-only-tests` CI job runs the CPU suite on
  every push/PR. Closes when the first GitHub Actions run is green. (Unblocked as predicted by the
  +112 poisoned-row drop — no BF16 typed test hard-errors.)
- [ ] **`find_package(Mila)` builds for an external consumer** — today it **fails** (see
  *Packaging*, a `[gate]`). Shipping a beta a contributor cannot `find_package` + `import Mila;`
  contradicts the trust signal.

**Trust-signal hygiene (cheap, mostly not code — but the export freeze is an asymmetric decision):**
- [ ] **Freeze the narrowest defensible public export surface** (see *Public API Surface*) — must
  happen *before* the freeze; too-broad can only be undone by a breaking removal.
- [ ] `CONTRIBUTING.md` + `getting-started.md`, `good first issue` labels, ungated GPT-2 quick-start,
  default-branch flip `dev -> master` (see *Release Assets & CI* / *Production Hardening*).

**Explicitly NOT beta.1 gates** (do not let these compete for attention):
- **Gemma 4 Inference Competitiveness (prefill + decode)** — self-declared *"NOT a release gate"*,
  deferred by choice. Zero beta weight.
- **API Documentation** — effectively done (ratchet green; Tier 3 folded into Test Suite Revival;
  only a live docs-CI run pending).
- **Published Docker runtime image** — *optional* beta deliverable.
- **Training Revival** beyond "samples run" — the primitive test suite overlaps the Test Suite
  Revival gate above; Llama training is a Future Direction.
- **MoE / Qwen 3 / Ministral / advanced training** — future.

---

## Consolidation

Alpha.5's success criteria are met (greedy decode at FP8 with no catastrophic divergence on
Llama 3.2 3B and 3.1 8B, 8B within the 12 GB budget). Consolidation closes the alpha line:
feature-freeze (no new features) and burn down the debt so the public release earns the beta
label. The FIXME/TODO burndown + debug-strip work is itemized under
Project Hygiene below; the migration/cleanup tasks specific to closing alpha are here.

- [x] **[gate]** CPU Linear traits — `OperationTraits<LinearOp, Cpu, FP32, NoWeightQuant>` specialization is live ([OperationTraits.Cpu.ixx](Mila/Src/Dnn/Compute/Devices/Cpu/Operations/OperationTraits.Cpu.ixx)); the `CpuLinearOpTypeMap` holdout is retired (out of build + `RETIRED` banner)
- [~] Retire the legacy dispatch files **in place** (out of the build + a `RETIRED` banner — not deleted; the user keeps superseded source on disk for reference). **Done:** the Linear/Gqa typemap clusters (`LinearOpTypeMap`(+`.Template`/`:Cpu`/`:Cuda`), `GqaOpTypeMap`(+`.Template`/`CudaGqaOpTypeMap`)), `OperationRegistryHelpers`, `OperationRegistrarHelpers`, `OperationsRegistrar`, and `FusedComponent`. **Still live, gated on the Training Revival loss-path re-authoring:** `OperationRegistry` and the `UnaryOperation`/`BinaryOperation`/`PairedOperation` arity bases — still imported by the disabled `CpuCrossEntropyOp` / `CpuSoftmaxCrossEntropyOp` / `CudaMatMulBiasGeluOp` (kept in `Src` by request). Retire those once the CrossEntropy ops are re-authored off the registry. (`IPositionalPairedOp` is NOT in this set — it is a live RoPE interface.)
- [~] **Retire the `CudaGqaOp` legacy A/B path (the `kUseOptimizedPath` losing side)** — **Pass 1 DONE (awaiting VS2026 build):** collapsed the gate in [CudaGqaOp.ixx](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx) (`prefill`/`decode`/`build` call the `_optimized` variants unconditionally; `kUseOptimizedPath`/`use_optimized_path_` gone), deleted the legacy `prefillImpl`/`decodeImpl`/`initializeState`/`buildCublasLtPlans`/partial-plan helpers + all legacy plan/state-tensor/raw-pointer members (kept only the `k_tensor_`/`v_tensor_` KV cache + the `_optimized` survivors, suffix strip is Pass 2), and refreshed the class doc. **Decision point surfaced:** `Operation` has no `forward`/`backward`, so `GroupedQueryAttention.ixx:181/:209` call the concrete `CudaGqaOp::forward`/`backward` directly — deleting them outright breaks the **public** component compile. Kept them as honest throwing stubs ("GQA is inference only") to preserve the contract; **fully retiring `GroupedQueryAttention`'s standalone `forward()`/`backward()` is a public-API decision** (pairs with the GQA forward-stub bug below) and is out of scope for this inference cleanup. **Pass 2 (after Pass 1 builds green) — REVISED:** do NOT delete the now-uncalled expanded-layout dispatch kernels (`CudaGqa.Dispatch.ixx`: `expand_kv`/`permute_qkv`/`reduce_kv_grad`/`permute_backward`/`softmax_forward`/`softmax_backward`) or the legacy plan builders (`CudaGqa.Plans.ixx`: `build_qk_score_plan`/`build_att_value_plan`/`build_backward_*_plan`). The legacy GQA forward/backward derived from a working MHA and likely functioned before being deprecated for the optimized inference path, so the expanded layout is the cheap, known-good substrate for a FUTURE GQA training path (clean gradient: `expand_kv` broadcast <-> `reduce_kv_grad` grouped sum). **Retire these in place as dormant training substrate** (banner, keep compilable) rather than prune. **Banners DONE** (`CudaGqa.Plans.ixx` + `CudaGqa.Dispatch.ixx` carry "DORMANT — expanded-layout … retained as future GQA-training substrate" notes that also resolve the per-method `REVIEW:` markers). **Remaining (optional polish):** the `_optimized` identifiers in `CudaGqaOp.ixx` — note this is NO LONGER a "strip the suffix" job: since the expanded builders survive (dormant), the two layouts coexist permanently, so the honest rename is a `compact` (live) vs `expanded` (dormant) scheme, not suffix removal. Touches validated inference code — best done as a VS2026 IDE rename. See [[project_gqa_backward_never_validated]]. **Original analysis:** the 11 GQA `REVIEW:` markers in [CudaGqa.Dispatch.ixx](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqa.Dispatch.ixx) + [CudaGqaOp.ixx](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx) are not 11 decisions — they are one unfinished migration. `kUseOptimizedPath = true` ([CudaGqaOp.ixx:63](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx)) is a constant, so the entire legacy `else` world is unreachable yet still compiles and **allocates** (the expanded `[B,NH,T,HS]` `k_exp_tensor_`/`v_exp_tensor_`/`q_tensor_`, legacy cuBLASLt plans `:471-483`, legacy state tensors `:503-513`, `initializeState()`/`buildCublasLtPlans()` vs the `_optimized` twins, the `use_optimized_path_` bool). The optimized compact-NKV path is the **documented, validated** inference winner (`Specifications/GqaMemory.md`: legacy expanded layout was a misdesign that negated GQA's memory advantage, 3.2 GB -> 736 MB state). **Gate (the only one):** validation was manual token-for-token inference traces — no automated parity test exists, so the legacy forward is currently the *only* inference oracle. Build an independent oracle first (golden vectors, or a small-shape CPU GQA forward reference fitting the Test Suite Revival finite-diff/CPU archetype) **before** deleting legacy forward. **Then** retire the whole legacy path in one pass — forward `:281-334`, `backward()` `:336-416`, the throwing BF16 dispatch stubs (`reduce_kv_grad`/`permute_backward`/`unpermute_*`), legacy plans, legacy state tensors, the dual init/plan builders, and the `kUseOptimizedPath`/`use_optimized_path_` gate (collapse to one path, mechanical `_optimized` suffix strip per GqaMemory.md §5). The legacy **`backward()` is dead aspirational scaffolding** — Llama 3.1/3.2 are inference-only and GQA training is scoped in **no** milestone (Training Revival is MHA/GPT-2 only), so it carries zero retirement risk and gets no BACKLOG demotion; leave a one-line "GQA training is a future aspiration, not scoped" note so the absence reads as intentional. Coordinates with the FP16-removal item below (the `*_fp16` GQA stubs are its "bucket B"). Two riders: `permute_qkv_decode` in the **BF16** specialization ([CudaGqa.Dispatch.ixx:288](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqa.Dispatch.ixx)) still takes `half*` and calls the fp16 kernel — type-inconsistent (and unreachable, corroborating dead); and `GqaMemory.md` §7 ("Training path gradient buffer ownership migration… tracked separately") oversells the backward as something that exists to migrate — correct it when the spec is next touched

- [x] **Couple parameter initialization to runtime mode** — `BuildContext::initialize_parameters` defaulted `true` independently of `RuntimeMode`, so an inference-mode build could silently run (then immediately discard) full parameter initialization. `GptModel::fromPretrained` hit exactly this; the interim fix passed `false` explicitly (matching `LlamaModel`), which relied on every load path remembering the third argument. **Done (0.20.0-alpha.6+82, awaiting VS2026 build):** the `BuildContext` explicit-ctor third argument is now `std::optional<bool> initialize_parameters = std::nullopt` and resolves via `value_or( runtime_mode == RuntimeMode::Training )` ([Component.BuildContext.ixx](Mila/Src/Dnn/Core/Component.BuildContext.ixx)) — Training initializes (train from scratch), Inference skips (weights loaded), and an explicit `true`/`false` still wins. No load path can regress by omitting the flag. All existing explicit call sites compile unchanged (`std::optional<bool>` accepts the `false` on the three `fromPretrained` paths and the propagated `shouldInitializeParameters()` bool in Llama/Gemma sub-context construction). The derivation also **corrected a second latent instance** of the original bug: `GptTransformer::fromPretrained` ([GptTransformer.ixx:147](Mila/Src/Dnn/Components/Transformers/Gpt/GptTransformer.ixx)) built an Inference context with the flag omitted, so it initialized parameters then immediately overwrote them via `loadParameters()`; it now skips init. The Bard trainer's two-arg `RuntimeMode::Training` context derives to `true` (correct for train-from-scratch), and the Inference `EXPECT_THROW` device-mismatch tests throw before init so are unaffected. The per-device init wiring (TensorOps `zero`/`fill`/`xavier`/`fill_normal`) gated per component via `shouldInitializeParameters()` is unchanged

- [~] **[gate — Bard CPU stack] FFN consolidation: de-polymorphize `MLP`** — **the blocking de-polymorphization is DONE** (landed alongside the `Activation` primitive; this item was stale). The runtime `switch` that force-instantiated the `Swiglu` case is gone: `MLP` now holds a concrete `Gelu` child ([MLP.ixx](Mila/Src/Dnn/Components/FFN/MLP/MLP.ixx) `createGraph()` -> `addGelu()`), the `mlp_activation_impl` / `std::function` bridge / SwiGLU branch are RETIRED in [MLP.Dispatch.ixx](Mila/Src/Dnn/Components/FFN/MLP/MLP.Dispatch.ixx) (out of the CMake build, banner in place), and `Swiglu` -> `FFN/Swiglu/` + `MLP` -> `FFN/MLP/` are relocated (tests mirrored). With `CpuGeluOp` + `OperationTraits<GeluOp, Cpu, FP32>` live, `MLP<Cpu>` / `GptBlock<Cpu>` / `GptTransformer<Cpu>` compile — their `MLP.Cpu.cpp` / `GptBlock.Cpu.cpp` / `GptTransformer.Cpu.cpp` tests are **active** in `Tests/CMakeLists.txt`, so the Test/Training Revival CPU stack is unblocked. Per-step debug `synchronize()` stripped from `GptBlock`/`LlamaBlock` `forward()`/`backward()` (0.20.0-alpha.6+81); `GptBlock::decode()` still carries 8 per-op syncs (GPT-2 inference path — correction: the "inference decode already sync-free" note held only for Llama), left for when Bard/GPT-2 decode is next run. **Remaining (deferred, not a gate):** fold the fixed `Gelu` child onto `Activation<…, ActivationType TFn = Gelu>` — the literal compile-time-`TActivation` endgame — tracked under the `Activation` elementwise-primitive item below (which already parked it: "keep `Gelu` for now"); and decide the `MLPConfig` residual activation surface (`withActivation()` / `activation_type_` / `getActivationType()`, kept per FfnAndMoE.md §5 as serialization metadata but Gelu-or-throw as an ergonomic setter). Full design: [FfnAndMoE.md](Mila/Specifications/FfnAndMoE.md). The reusable `GatedMLP` + grouped `MoeOp` foundation it specifies is a Future Direction (Architecture / MoE), not a 0.20 gate
- [~] **`Activation` elementwise primitive** — collapse the 8 elementwise `ActivationType` entries into one compile-time `Activation<Device, Precision, ActivationType TFn>` over a shared `MILA_HD` functor library + Cpu/Cuda `ElementwiseActivationOp` (functor-templated, NOT a 5th `OperationTraits` axis; design in FfnAndMoE.md §5/§5.1). `Gelu` folds in (its tests become a function-sweep). One CPU op = all eight elementwise functions on CPU. The Gelu-only foundation is what `MLP` de-polymorphization holds; the remaining 7 functors land incrementally after it. The CPU `SwigluOp` gate op stays the demand-driven contributor item below — now near-free once the shared functor library exists, since the gate reuses it. **DONE (component + op layer, added alongside `Gelu`):** `ElementwiseActivation.h` functor library (`MILA_HD`; all 8 functions Identity/GeluTanh/SiLU/ReLU/Tanh/Sigmoid/LeakyReLU/Mish), `CpuElementwiseActivationOp<Functor>` (FP32, all eight), `CudaElementwiseActivationOp<Precision, Functor>` + functor-templated kernels (FP32+BF16, grid-stride, explicit-instantiation bridge), `OperationType::ElementwiseActivationOp` + the member-template `op_for<Functor>` traits specializations (Cpu FP32 / Cuda FP32+BF16 — the "resolve the op template, no 5th axis" shape), and the `Activation<…, TFn>` component (`functor_of` enum->functor map, undefined-primary => hard error on non-elementwise) + `ActivationConfig`. Tests: `Activation.Cpu` typed function-sweep (7 functions, fwd+backward numeric), `ActivationConfig`, `Activation.Cuda` (FP32+BF16 SiLU sweep + FP32 multi-functor). **REMAINING:** fold `Gelu` (and `MLP`'s child) onto `Activation` (deferred — keep `Gelu` for now); CPU `SwigluOp` (contributor item below); CPU-only **install** shipping of `ElementwiseActivation.h` (registered in the CUDA `cuda_headers` set only — builds fine everywhere via file-relative include; a core HEADERS file set for the CPU-only install tree is the gap)
- [ ] **`GatedMLP` (gated FFN + MoE-expert reference)** — DONE for the single-expert reference: `GatedMLP<Device, Precision, ActivationType TGate = Silu>` composite (`fc_gate_up` Linear(in->2H, fused) -> `Swiglu` gate -> `fc_down` Linear(H->in)), `GatedMLPConfig`, `ComponentType::GatedMlp`, with the FfnAndMoE.md §9 MoE-readiness seams (injected-context norm, trailing-dim-agnostic forward). Tests: `GatedMLPConfig` (CPU) + `GatedMLP.Cuda` (FP32+BF16 wiring: shape contract, parameter count, bias-free zero-input identity, backward shape). **`TGate` is currently constrained to `Silu` by `static_assert`** — the existing `SwigluOp` is SiLU-fixed; generalizing the gate (GeGLU/ReGLU) needs `Swiglu<…, TGate>` over the shared functor library + the CPU `SwigluOp`, pairing with the activation-unification follow-ups. **REMAINING (Future Direction — Architecture/MoE, not a 0.20 gate):** `LlamaBlock` delegating to `GatedMLP` (delete its inline FFN); grouped `MoeOp` + `Router`/`MixtureOfExperts`

- [x] **`ComponentType` / `ModelType` axis split (resolves the `ComponentType.ixx:58` REVIEW marker).**
  The REVIEW questioned whether named architectures (`Gpt2`/`Llama`/`Mistral`/`Bert`) belong in
  `ComponentType`. They do not: `ComponentType` is the structural *kind* (Linear / Transformer block /
  Network), and architecture identity is an orthogonal axis. Confirmed dead/miswired before removal —
  the four values were absent from all four converters (so `toString(ComponentType::Llama) == "Unknown"`),
  `getType()` has NO production consumer (only self-asserting unit tests read it), and model
  serialization already self-identifies via a string literal (`save_` writes `"LlamaTransformer"`).
  **Done 2026-06-20:** removed the four values; the three top-level networks (`LlamaTransformer`,
  `GptTransformer`, `GemmaTransformer`) drop their `getType()` override (inheriting
  `Network::getType() == ComponentType::Network`) and gain a `getModelType()` accessor returning the new
  `ModelType` enum ([ModelType.ixx](Mila/Src/Dnn/Core/ModelType.ixx): `Unknown`/`Gpt2`/`Llama`/`Gemma`/
  `Mistral`/`Bert` + `toString`/`modelTypeFromString`). The three `GetType_Is*` tests now assert
  `getType()==Network` (kind) AND `getModelType()==ModelType::X` (architecture). Wired into
  Mila/CMakeLists.txt + Mila.ixx. `Network`'s enum value is unchanged (it precedes the removed values;
  `CustomComponentStart`/`MockComponent` are pinned at 1000), so no serialization value drift.
- [ ] **[deferred] `ComponentType` vitality — does `getType()` earn its keep?** The REVIEW above is the
  visible tip of a larger question: `getType()` and all four `ComponentType` converters
  (`toString`/`fromString`/`toTypeId`/`fromTypeId`) have NO production consumer — the accessor is read
  only by tautological self-asserting tests, and the converters are not called anywhere. Either wire them
  to a real consumer (e.g. a typed serialization/registry key replacing the `save_` string literals) or
  retire the unused surface. A `ComponentType`-wide decision, larger than the architecture-axis split;
  deferred so it does not bloat that focused cleanup. Same latent question applies to the new
  `ModelType` accessor — keep it anchored to a real consumer (the model-agnostic Chat layer on the
  Qwen 3 roadmap is the natural one) rather than letting it drift into the same test-only limbo.

Deferred / not alpha-close gates:

- [~] **[milestone: LanguageNetwork — Sample API]** Token sampling (temperature / top-k / top-p) — `OperationTraits<SamplingOp, Cuda, FP32>` and `<…, BF16>` specializations; device `CudaSamplingOp`/`CpuSamplingOp` driven by a `TokenSampler` per `Specifications/TokenSampling.md`. **Phases A-D (Gemma) COMPLETE + green (2026-06-27):** `Sampler` base + `TokenSampler` facade + `SamplingConfig` (`Dnn.Samplers`), `Cuda`/`CpuSamplingOp` with all branches — greedy argmax, full multinomial, and top-k/top-p (device: threshold binary search, single-block correctness-first; host: exact `nth_element`/`sort`). `TokenSampler` is now hoisted to the `LanguageModel` base (lazy, shared context) via `sampleNext()`; **GemmaModel migrated, path A (host `sampleToken`) retired**, `logits_staging_` + `decode_token_staging_` removed, per-step H2D restage gone. Greedy DeviceB validated token-for-token vs HostA; stochastic + top-k/top-p coherent in chat; injected-`r` unit oracle green (`Tests/Dnn/Samplers/Sampling.Cuda.cpp` — it caught + fixed a top-k off-by-one). The sync-contract `forward()` runs on the **default CUDA stream** (the model syncs the network before sampling and the facade reads back synchronously, so default-stream ordering is correct); the enqueued path added 2026-07-04 (below) calls `context_->getStream()` — the MSVC per-BMI module-reachability issue that originally forced the default stream did not have a reproducer left behind, and `RmsNormOp`/`CudaLinearOp` call `getStream()` from the same import shape, so it is expected clear (verify on first VS build). `Network` gained a public `getExecutionContext()`. **REMAINING:** migrate `LlamaModel` (mechanical mirror) + `GptModel` (GPT-2 variant) onto the base `sampleNext()` and delete their host `sampleToken`s (deferred to when those paths are built/run); ~~share the decode stream via `getStream()` (Phase D tail)~~ — **DONE + VALIDATED 2026-07-04 (D1 decode-ahead; tests green, chat coherent, HF parity green, gemma_decode_d1 captures analyzed):** `enqueueForward()`/`awaitToken()` on `Cuda`/`CpuSamplingOp` (CUDA: sampler kernels on the context stream + async 4-byte D2H into a pinned slot + event; CPU: compute-and-stash mirror), surfaced as `TokenSampler::enqueueSample`/`awaitToken` and `LanguageModel::enqueueSampleNext`/`awaitSampledToken`; `GemmaModel::onGenerating` is now a decode-ahead pipeline — the next forward is enqueued BEFORE the pending token id is host-visible, deleting the per-token `synchronize()` + sync D2H. **Measured: pipeline works as designed (513 stream-syncs/run -> 1 + 256 event-syncs at 79 us avg — host arrives after the event has fired; host-caused >= 20 us gaps 0.27 -> 0.10 ms/token; single kernel stream), but the calibration's "1.5-2 ms host gap" was ~85% MISATTRIBUTED — it is launch-granularity micro-gap tax (~1165 kernels/token x ~1.3 us dead time between tiny kernels), unchanged by D1. True D1 throughput recovery ~0.2-0.3 ms/token (wall 24.17, busy 22.57 greedy; sampled mirrors it and matches 41.6 tok/s chat); the real wins are latency hygiene + host removed from the per-token path (launch API now back-pressure paced, elastic). Re-rank consequence: D2/D3 fusion pays double (kernel time + gap per launch removed), and a CUDA Graphs decode step (static graph, device-side position, one launch/token) attacks the whole ~1.5 ms tax — evaluate with D2 (review section 4.1).** Semantic delta: a sampled stop token is now decoded into the KV cache before the host sees it — `kv_token_history_` records it (exact bookkeeping; the cached `<end_of_turn>` K/V is itself reusable next turn); RNG draw count/order unchanged. Sync `forward()` (default stream) retained as the Llama/Gpt contract until their migration. Enqueued-path oracle added to `Sampling.Cuda.cpp` (sync-vs-enqueued token parity, single-slot reuse, stream-ordering-without-sync, await-without-enqueue throw); ~~single-block kernel perf optimization~~ — **DONE + VALIDATED 2026-07-03: the measured 11.05 ms/token `stochastic_kernel` (29.9% of sampled-decode wall — Gemma4InferenceReview.md section 10.3) is replaced by a multi-block pipeline in `Sampling.cu` (histogram threshold refinement + chunked index-order inverse-CDF, no host round-trip); re-profiled at 55.5 us/token (200x kernel reduction), sampled decode 25.6 -> 35.7 tok/s (+39%), chat coherent. Single-block kernel retained as `forwardReference()` parity oracle; `Sampling.Cuda.cpp` locks truncated-case token parity + a host-double CDF bracket for the full multinomial (serial-vs-chunked float summation makes full-vocab token equality unattainable in flat CDF regions — observed 732-index/~1e-5-mass shift at r=0.999; TokenSampling.md section 5 records the design)** (`argmax_kernel` measured 26 us — greedy untouched). User must delete the orphaned `Dnn/Decoders/` skeleton + `Core/Decoder.ixx` in VS2026. **Design resolved 2026-06-27 (spec §3-7):** `TokenSampler` is an **orchestrator tool owned by the `LanguageModel` base** (the structural sibling of `Optimizer` — model-owned, shares the model's `ExecutionContext`), **not** a graph `Component`; ONE concrete `TokenSampler` carries top-k/top-p/min-p as composable per-call *filters* (not a class per strategy), with the `Sampler<Device,Precision>` base kept as the seam for a future stateful strategy (Mirostat); dispatch is the unified `OperationTraits` table (NOT the legacy `conditional_t` facade — see the Optimizer-migration item below, which lands first). Retires the `Dnn/Decoders/` skeleton (`Decoder`->`Sampler`, `TopKDecoder`->`TokenSampler`, `TopKConfig`->`SamplingConfig`). Not a 0.20 gate — greedy decode is already validated, so this is additive
- [ ] **Migrate Optimizer dispatch onto `OperationTraits`** (prerequisite/sibling of Token sampling — do FIRST to prove the pattern on working code). `AdamWOptimizer` ([AdamW.ixx](Mila/Src/Dnn/Optimizers/AdamW.ixx)) selects its device impl with `std::conditional_t` + `#ifdef MILA_HAS_CUDA` — the simplest dispatch that worked, predating `OperationTraits`. It does not scale (every facade re-implements the conditional and carries its own CUDA `#ifdef`) and diverges from how graph ops and the new `TokenSampler` dispatch. Bring it onto the unified table so both `LanguageModel` orchestrator tools (Optimizer, Sampler) dispatch identically: (1) add an **algorithm-keyed** entry to `OperationType` (`AdamWOp`; optimizers key by algorithm because SGD/AdamW are distinct classes — unlike the role-keyed single `SamplingOp`); (2) add an `OptimizerOpConcept` to [OperationTraits.Template.ixx](Mila/Src/Dnn/Compute/Operations/OperationTraits.Template.ixx) (`addParameter`/`step`/`get`+`setLearningRate` — not `forward`); (3) add `OperationTraits<AdamWOp, {Cuda,Cpu}, TPrecision>::type` specializations in the `:Cuda`/`:Cpu` partitions resolving to `Cuda`/`CpuAdamWOptimizer`; (4) replace the facade's `conditional_t` block with the traits alias and drop its `#ifdef` + direct backend imports. Net: N per-facade `#ifdef`s collapse to the one guarded `OperationTraits.ixx` aggregator (aligns with the no-`#ifdef`-in-modules rule); a new backend = a new partition specialization, facades untouched. Validate against the existing (currently disabled) AdamW optimizer tests. Also update [OperationType.ixx](Mila/Src/Dnn/Compute/Operations/OperationType.ixx)'s header note ("Operations are an implementation detail of Components") — it becomes false once model-level orchestrator tools are keyed; broaden to "the compile-time dispatch key for any device-backed compute unit, whether owned by a Component or a model-level orchestrator". Optional larger follow-on (NOT now): rename `OperationTraits`/`OperationType` -> `ComputeTraits`/`ComputeUnitType` to match the broadened scope. See `Specifications/TokenSampling.md` §3.2
- [ ] **[deferred, training-only]** AdamW debug instrumentation — the per-value `isfinite`/limit `printf` anomaly guards in `CudaAdamW.cu` (6 sites) plus the leftover `printf` in `CudaAdamWOptimizer.ixx:270` are training bring-up scaffolding. Left untouched by the Consolidation debug strip because the AdamW path is training-only: off the validated inference path, exercised solely by the parked MNIST/Bard samples, and untested (`AdamW.Cuda.cpp`/`AdamW.Cpu.cpp` disabled in `Tests/CMakeLists.txt`). When Training is picked up, decide strip-vs-gate (the `KERNEL_ASSERT` invariant checks are already `NDEBUG`-gated and zero-cost in release; the `printf`s are not) and re-enable the optimizer tests in the same pass
- [ ] **[deferred, training-only]** CUDA `fill_normal`/`fill_uniform` are FP32-only — they cast the raw buffer to `float*` and `curandGenerate` into it, so BF16/FP16 reduced-precision **train-from-scratch on CUDA** corrupts weight/embedding init. Reachable now that `xavier`/`normal` init is wired (`TokenEmbedding` wte is BF16 on the Llama path). Harmless for inference (init gated off) and for CPU (the `CpuTensorOps.Random` added this cycle converts element-wise). Fix: generate into a temp float buffer + a convert pass — the CUDA dtype counterpart to the CPU Random backend
- [ ] **[deferred, needs recall + live-vs-dead analysis]** Remove FP16 — superseded by BF16. FP16 was implemented first; once BF16 landed there is no reason to carry both for LLM inference (BF16's wider exponent range is strictly preferable, no loss scaling). Scaffolding is woven through *live* code: `CudaDataTypeMap<half>`, the `half`/`CUDA_R_16F`/`CUBLAS_COMPUTE_32F_FAST_16F` branches in `CudaLinearOp`, `half` throw-stubs in `CudaLinearOp.Plans`, and the commented `*_fp16` backward/permute stubs across the GQA/MHA/LPE/Softmax dispatch (these are the marker-triage "bucket B"). Trace live-vs-dead `half` paths before removal — not a mechanical delete
- [x] **Poisoned BF16 dispatch rows — DROPPED (0.20.0-alpha.6+112, VS2026 build green).** `OperationTraits<{GeluOp,MultiHeadAttentionOp,SoftmaxOp,LpeOp}, Cuda, BF16>` each advertised a `CudaXxxOp<BF16>` whose kernel is constrained `float || half` ([CudaGeluOp.Dispatch.ixx:57](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Gelu/CudaGeluOp.Dispatch.ixx), `CudaMhaOp.Dispatch.ixx:24`, `CudaSoftmaxOp.ixx:48`, `CudaLpeOp.Dispatch.ixx:31`), so instantiating the component at BF16 hard-errored the moment a typed test constructed the op. **All four rows removed** from [OperationTraits.Cuda.ixx](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/OperationTraits.Cuda.ixx) (each replaced by a comment explaining why FP32-only is honest and how to re-add a real BF16 kernel later; the MHA/Lpe dispatch `REVIEW:` markers are resolved by reference). FP32-only is the honest advertisement: all four are GPT-2 lineage / off the BF16 inference path (MHA: GPT-2 attention, FP32 weights, Llama uses GQA; LPE: GPT-2 positional, Llama uses RoPE; GELU: BF16 FFN uses Geglu; Softmax: attention softmax is fused inside the GQA/decode kernels). **Safe:** the green-build invariant + the four active CUDA tests being already FP32-only by explicit design (each a one-entry `Types<Fp32Precision>` with a "add a Bf16Precision tag once a BF16 kernel exists" note) prove nothing forms a BF16 `OpType` alias for these. **Desync audit discharged:** these four were the only poisoned rows — the remaining BF16 rows (Residual/RmsNorm/Swiglu/Geglu/Rope/Sampling/ElementwiseActivation exercised in production; CrossEntropy's kernel is `float || nv_bfloat16`) all back their advertisement. Root-cause readability fix landed alongside as the `OperationSupported<...>` predicate (Dispatch error UX below). Loosely pairs with the FP16-removal item above but did not require it
- [ ] **[deferred → Qwen 3 cycle (vNext)]** `OperationTraits<GroupedQueryAttentionOp, Cuda, BF16, PerChannelKvFp8<>>` specialization — pending `CudaGqaOp` FP8 KV cache support
- [ ] **[contributor]** Llama-lineage CPU ops (`RmsNormOp`, `SwigluOp`, `RopeOp`, `TokenEmbeddingOp`, `CrossEntropyOp`) in `OperationTraits.Cpu.ixx` — demand-driven; compile-time dispatch makes their absence zero-cost on the GPU path and a localized compile error if a `<Cpu, …>` Llama is instantiated
- [ ] **[deferred, measure first]** Phase 6b H2D pipelining — a dedicated load stream + CUDA events threaded through `loadParameter`/`quantize` would overlap H2D-with-H2D, but ~16 GB over PCIe 4.0 (~2.3s) is the floor, so pursue only if a profile shows the load is sync-bound rather than disk-bound

---

## Test Suite Revival

Recover the authored test suite, not write one. The first Mila year was test-driven; ~70 test
files exist under `Mila/Tests/**` but only ~24 are active — the rest were `#`-commented during the
inference-era refactors (`Tests/CMakeLists.txt:107` names the cause: "too many tests to refactor for
Component lifecycle changes"). The work is three buckets: (1) re-green what exists, (2) translate it
to the post-refactor API, (3) backfill what the old suite never covered. The non-negotiable
deliverable is the **CI ratchet** — the suite rotted because nothing gated it, so revival without a
gate just reschedules the next rot. Gated behind the Consolidation `CompositeComponent`/`setTraining`
lifecycle fix, which is what currently forces the component tests off.

- [~] **[gate]** Re-green the authored component / operation / tensor / tokenizer suites against the current API — re-enable the `#`-commented files in `Tests/CMakeLists.txt`. Bucket 1 (uncomment + fix trivially broken) + bucket 2 (translate to the post-refactor surface: `OperationTraits` dispatch, the `Operation` base-class collapse, the precision axes, the Consolidation lifecycle fix). _Done (per-component completion tracked here; the consolidated CHANGELOG entry is written when this milestone closes, not per component):_ `Component`/`ComponentConfig` (base contract); `Gelu` (stateless leaf reference); `Linear` (`LinearConfig` + `Linear.Cpu` + `Linear.Cuda` — first leaf-with-parameters reference and the first `TYPED_TEST` precision sweep, FP32+BF16); `RmsNorm` (config + `.Cuda` FP32+BF16, CUDA-only); `LayerNorm` (config + `.Cpu` FP32 backward-numeric + `.Cuda` FP32); `Swiglu` (config + `.Cuda` FP32+BF16, CUDA-only); `Residual` (config + `.Cpu` + `.Cuda` FP32+BF16 — binary stateless leaf); `TokenEmbedding` (config + `.Cuda` FP32+BF16, CUDA-only — gather + prefill/decode shape regime); `Lpe` (config + `.Cpu` FP32 backward-scatter + `.Cuda` FP32-only, gather + positional add + decode — GPT-2 lineage, no BF16 kernel; see poisoned-row note); `Rope` (config + `.Cuda` FP32+BF16, CUDA-only — forward/prefill/decode + inverse-rotation backward round-trip); `Softmax` (config + `.Cpu` FP32 forward+backward-numeric + `.Cuda` FP32-only forward — no BF16 kernel; see poisoned-row note); `MultiHeadAttention` (config + `.Cpu` FP32 causal-attention forward-numeric + `.Cuda` FP32-only forward via prefill + decode-after-prefill numeric — GPT-2 lineage, no BF16 kernel; see poisoned-row note below); `GroupedQueryAttention` (config + **surface-only** `.Cuda` FP32+BF16 — construction/accessors/build-validation/stateless/type; numerics deferred, see GQA bug below); `MLP` (`MLPConfig` + `.Cpu` composed-forward numeric + finite-difference backward + `.Cuda` surface/shape/finiteness/gradient-flow — composite fc1->Gelu->fc2; the CUDA composed numeric reference is left to `.Cpu` and the leaf numerics to `Linear.Cuda`/`Gelu.Cuda`); `GptBlock` (`GptBlock.Config` + `.Cpu` + `.Cuda` — GPT-2 pre-LN block composite, gradient-flow asserted; the composed finite-difference probe is deferred to the gradient-check archetype); `GptTransformer` (`Gpt.Presets` + `.Cpu` + `.Cuda` — GPT-2 network surface + finite logits, with `.Cuda` also asserting training-mode backward, exercised by Bard); `LlamaTransformer` (`Llama.Config` + `Llama.Presets` + **surface-only** `.Cuda` — construction/build/mode/components/type + forward-shape; numerics deferred per the GQA standalone-forward stub, and the retired `LlamaTransformer::fromPretrained` cases dropped). With these the concrete **component-class** set is fully re-greened to the methodology (build green); only `SoftmaxCrossEntropy` (the loss component) remains, parked for the loss-on-device milestone. **Bugs surfaced (filed below, fixed outside the methodology session):** `loadParameter` recursion (Linear fixed; RmsNorm/LayerNorm/TokenEmbedding/Lpe filed), Swiglu BF16 backward dtype, non-quantized cuBLASLt GEMM, `Linear::forward` dead fast-path. **Greening pass 2026-06-17 (0.20.0-alpha.6+64):** ran the re-greened suites failure-by-failure to actual green. Landed: CharTokenizer empty-corpus test fixes; CPU TensorOps math revived (C1116 root-caused to `<execution>`, FIXED below) -> all `Math.Cpu` ops green; `getGradients()` inference contract unified across the tree to return-empty (was throw on CompositeComponent/LayerNorm/TokenEmbedding) + the throw-asserting tests (LayerNorm Cpu/Cuda, CompositeComponent, MlpNetwork) flipped to assert-empty + `MockChild` made mode-aware; `Softmax.Cpu` backward test corrected (pass Y, zero accumulation target); `TokenEmbedding.Cuda` BF16 test dim made `% 8`-clean; un-skipped the `Linear.Cuda` forward/backward numeric cases (cuBLASLt bias-epilogue fixed) and the `GptBlock.Cpu` composed-backward sentinel (CPU math live). **Remaining red: exactly 2 tests, both unimplemented-backward (filed below):** CUDA Softmax backward stub and BF16 Swiglu backward dtype. (The former MHA-CPU-backward-suspect GptBlock composed gradient check is now GREEN — see the RESOLVED bug entry below; validated 2026-07-02 by the standalone MHA numeric probe)
- [~] **Tensor suite (non-component slice)** — re-green + complete `Tensor.ixx` core coverage to the methodology, tracked by the per-member matrix in [Testing.Tensors.md](Mila/Specifications/Testing.Tensors.md). Establishes the **value-type / god-module archetype** (area-split instead of one-file-per-module; dtype `TYPED_TEST` sweep only where behavior varies; memory-resource = file split). Core `Tensor.ixx` first; the wider `Tensors/` tree (`TensorBuffer`, `TensorDataType*` maps, `TensorOps/*`, `Partitioning`, `Serialization`) is a follow-on slice. _Core `Tensor.ixx` complete (awaiting VS2026 build):_ all eight area files (`Constructors`, `DataAccess`, `DataPointers`, `Identity`, `Io`, `MemoryProperties`, `Properties`, `ShapeTransform`) + `.Cuda.cpp` companions, on the `Mila::Tests::Dnn::Tensors` namespace with the device axis as a file split (every inline `#ifdef MILA_HAS_CUDA` removed). New coverage: `elementSize`/`getStorageSize`, the shape-transform area (no file before), `item()` + scalar negatives, and the device-tensor host-only SFINAE contract. The **value-type / god-module archetype** is now documented in [Testing.md](Mila/Specifications/Testing.md) §1. _TensorOps slice:_ `zero` done (`Zero.Cpu.cpp`/`Zero.Cuda.cpp`, methodology); `Fill`/`Math` re-greened (namespace + header, promoted to Section 1); `Transfer` namespace+header re-greened but **device-split still pending** (cross-device shared fixture -> `TensorOps.Transfer.Cuda.cpp` follow-up); `Structural`(`split`) missing -> backfill; `Random`(`fill_normal`/`fill_uniform`/`xavier`) deferred to Training Revival (training-init, CUDA FP32-only bug parked there). _Next (follow-on slice):_ the rest of the wider `Tensors/` tree — `TensorBuffer`, `TensorDataType*` maps, `Partitioning`, `Serialization`
- [ ] Backfill coverage for the inference-drought features the old suite never had — load-time quantization (`PerChannelFp8`/`PerGroupFp4`, the decode matvec kernels), `OperationTraits` dispatch, the Llama path (RmsNorm/SwiGLU/GQA/RoPE components, `LlamaModel::fromPretrained`). Genuinely new, not recovery. **This is also the *only* legitimate op-layer test** — the `CudaLinearOp` quantization white-box (scales == host absmax, FP4 nibble packing, exactly-representable round-trip): the surface unreachable through the public component, per the reachability rule in [Testing.md](Mila/Specifications/Testing.md) §1. Scope it to those unreachable assertions only; the forward/decode numerics are component-test territory (a black-box wiring proof on the quantized component, not an op-numeric mirror)
- [x] **[net-new]** Build the **gradient-check archetype** — the authored suite was *forward-only* (inference validated forward passes against HuggingFace), so every component `backward()` the training samples drive has zero coverage. Add a finite-difference gradient verifier (perturb input/parameter by ±eps, compare numeric vs analytic gradient within tolerance) as a reusable test fixture, then a `Backward_MatchesNumericGradient` case per training component. A few backward-numeric cases already exist (`LayerNorm.Cpu`, `Softmax.Cpu`, `Lpe.Cpu` scatter) — generalize them into the shared archetype rather than re-deriving per file. This is the largest net-new category and the precondition for Training Revival's convergence oracle. Document the archetype in [Testing.md](Mila/Specifications/Testing.md) alongside the value-type / component archetypes. **DONE (shared fixture + reference applications, awaiting VS2026 build):** [Common/GradientCheck.h](Mila/Tests/Common/GradientCheck.h) — signature-agnostic black-box verifier (`centralDifferenceGradient` over the scalar loss `L = sum(output*g)`, `expectGradientsClose` with combined abs/rel tolerance; the forward seam is a caller lambda so the helper absorbs the non-uniform `forward(input)->output&` vs `forward(input,output)` signatures), wired via `target_include_directories(MilaTests PRIVATE Tests-root)` so it includes as `"Common/GradientCheck.h"`. Reference applications: `Gelu.Cpu.cpp` `Backward_MatchesNumericGradient` (stateless input grad) and `LayerNorm.Cpu.cpp` `Backward_MatchesNumericGradient` (input + weight + bias grads via one `evaluate` lambda). Archetype documented in Testing.md §1; the pre-existing analytic cases kept as an independent second oracle. **Fan-out DONE (awaiting VS2026 build):** `Backward_MatchesNumericGradient` added to `Linear.Cpu.cpp` (dX/dW/dB), `Residual.Cpu.cpp` (binary da/db — perturb one input while the other is held fixed), and `MLP.Cpu.cpp` (dX + all four child parameter gradients fc1/fc2 W/B against MLP's own forward — the composite parameter backward had no numeric coverage before); `GptBlock.Cpu.cpp` already carries the equivalent own-forward finite-difference check (`Backward_InputGradientMatchesFiniteDifference`). Each uses the shared `Common/GradientCheck.h` helper with eps 1e-2, abs 1e-2, rel 1e-2 (Residual abs 1e-3 — exact addition). **MHA isolation probe added:** `MultiHeadAttention.Cpu.cpp` `Backward_MatchesNumericGradient` perturbs the full concatenated-QKV input (covers Q/K/V grad contributions in one check) — the sensitive standalone test the residual skip paths in GptBlock can mask. **VALIDATED GREEN (VS2026, 2026-07-02):** all four new cases pass, including the MHA probe — so `CpuAttentionOp::backward` is numerically correct, the "prime suspect" is exonerated, and the eps/tolerances (eps 1e-2, abs/rel 1e-2) held on the first real run with no tuning needed. The MNIST/Bard training spine now has per-component numeric backward coverage; the archetype item is complete
- [~] **Re-green in sample-revival order — MNIST spine first, Bard spine second** (mirrors the Training Revival source sequencing). MNIST spine (the minimum to make MNIST train + test): `Network.cpp`/`Network.Cpu.cpp`/`Network.Cuda.cpp`, `CompositeComponent.cpp`, `AdamW.Cpu.cpp`/`AdamW.Cuda.cpp`, `DataLoader.cpp`, `Tensor.Initializers.cpp`, plus Linear/Gelu **backward** via the gradient-check archetype above (forward already green). Bard spine (adds the GPT-2 stack): `GptBlock.Config/Cpu/Cuda.cpp`, `MLP.Cpu/Cuda.cpp` + `MLPConfig.cpp`, `GptTransformer.Cpu/Cuda.cpp` + `Gpt.Presets.cpp`, `Encoder.Cpu/Cuda.cpp` + `EncoderConfig.cpp` (Lpe), `TokenSequenceLoader.cpp`, and the BPE/char tokenizer block (`BpeTokenizer*`/`BpeTrainer`/`BpeVocabulary`/`CharTokenizer`/`CharTrainer`/`CharVocabulary`). **(The CUDA op-level tests `CudaLayerNormOpTests`/`CudaMatMulBiasOpTests`/`CudaMultHeadAttentionOpTests`/`CudaResidualOpTests`/`CudaSoftmaxOpTests`/`CudaEncoderOpTests` were on this list but are now `delete, not revive` — they mirror the component suite; see the reachability rule below and in Testing.md §1.)** **Parked (not on the sample critical path):** `CrossEntropy.*` / `SoftmaxCrossEntropy.Cuda.cpp` (both samples compute loss host-side — these belong to the later loss-on-device work), and `Model.Cpu.cpp` / `ModelArchive.cpp` / `ZipSerializer.cpp` (the MNIST `Model`-abstraction + serialization path is a commented WIP; `loadComponentWeights` is a TODO stub, so serialization does not gate "MNIST trains"). **Landed (MNIST spine, this session — build green):** `CompositeComponent.cpp` (container contract, mock harness); `AdamW.Cpu.cpp` (re-greened against the public `AdamWOptimizer` wrapper + NET-NEW closed-loop convergence test; dropped the retired `zeroGrad`/`withName` surface); `DataLoader.cpp` + new `DataLoader.Cuda.cpp` (pinned-memory split, since `CudaPinnedMemoryResource`/`PinnedDataLoader` are `#ifdef MILA_HAS_CUDA`); `Modeling/Network.Cpu.cpp` (NET-NEW composite forward/backward gradient-flow oracle — Linear->Gelu->Linear with analytic dX/dW/dB references + `createOptimizer`/`zeroGradients`/training-loop loss-decrease); `TensorOps/Random.Cpu.cpp` (NET-NEW live init — `fill_uniform`/`fill_normal`/`xavier`). Linear/Gelu backward already green at the leaf. **Blocker found:** `Tensor.Initializers.cpp` is NOT revivable — its target module `Dnn.TensorInitializers` (`random`/`normal`/`ones`) is fully commented out in Src (depends on the parked `Core.RandomGenerator`); the live init moved to `Dnn.TensorOps:Random`, so the new `Random.Cpu.cpp` covers the path and the old file stays retired. Decide later: re-author `TensorInitializers` onto `TensorOps:Random`, or delete the obsolete test. **Still pending MNIST spine:** `Core/Network.cpp` (now-thin Network delta — `getType`/`save_`; the container surface is covered by `CompositeComponent.cpp`) and the GPU-local `.Cuda.cpp` companions (`Network.Cuda.cpp`/`AdamW.Cuda.cpp`). The MNIST API surface is otherwise test-covered; the sample can re-enter the build
- [~] **[gate]** Wire the suite into CI as the anti-rot ratchet — build on the `MILA_ENABLE_CUDA=OFF` CPU-only gate so a future API churn fails the build instead of silently re-commenting coverage. This is the deliverable that keeps the revival alive. **Wired (0.20.0-alpha.6+114, awaiting first CI run):** `cpu-only-tests` job in `.github/workflows/build-pipeline.yml` runs the CPU suite (`ctest`) on every push/PR to dev+master, on plain `ubuntu:26.04` (clang-21, no CUDA image/toolkit/CUTLASS) mirroring the new `linux-clang-cpu-{debug,release}` presets. First CI job that *runs* tests. CPU-only baseline verified 2026-07-18 (from-scratch clang build clean, ~980 ctest cases green one-pass). Close when the first Actions run is green
- [ ] Do not revive tests for code being deleted, and **do not mirror the component suite at the op layer** — backend ops are implementation detail (not in `import Mila;`) and their numerics are reachable through the component, so the authored `Cpu*OpTests` / `Cuda*OpTests` are redundant mirrors -> **delete, not revive** (dispositions recorded in [Tests/CMakeLists.txt](Mila/Tests/CMakeLists.txt) Section 3; reachability rule in [Testing.md](Mila/Specifications/Testing.md) §1). The sole legitimate op-layer coverage is the weight-quantization white-box, which is net-new (the quantization backfill item above), not a revival. The GQA case — numerics that look op-only — is really the `GroupedQueryAttention::forward` stub bug: fix the component and test there. Likewise retire the disabled `UnaryOperation`/`BinaryOperation` tests alongside the base-class removal (Consolidation), the registry/typemap tests, and `Tensor.Initializers.cpp` (its module is now `Dnn.TensorInitializers_RETIRED`; live init is `TensorOps:Random`, covered by `Random.Cpu.cpp`)
- [ ] Calibration is the MNIST-plus-tests spike under Training Revival — it measures the per-file bucket-2 translation cost on a representative slice before the suite-wide estimate is trusted

### Bugs surfaced by the test revival (deferred — fix outside the methodology session)

Found while writing tests; recorded here rather than fixed inline so the test-revival diff stays
scoped to test code.

- [ ] **[bug]** `loadParameter` infinite recursion in `RmsNorm`, `LayerNorm`, `TokenEmbedding`, and `Lpe` (and likely other parameterized components) — the unknown-name fall-through calls `this->loadParameter( name, blob )` (virtual self-dispatch) instead of the base, same bug fixed in `Linear` ([RmsNorm.ixx](Mila/Src/Dnn/Components/Normalization/RmsNorm/RmsNorm.ixx) line 244, [LayerNorm.ixx](Mila/Src/Dnn/Components/Normalization/LayerNorm/LayerNorm.ixx) line 296, [TokenEmbedding.ixx](Mila/Src/Dnn/Components/Embeddings/TokenEmbedding.ixx) line 240, [Lpe.ixx](Mila/Src/Dnn/Components/Encodings/Lpe/Lpe.ixx) line 315). Apply the same fix (throw `std::invalid_argument` naming the bad parameter) and add the unknown-name negative test to each component's suite. Tests currently avoid the unknown-name path so they don't hang
- [ ] **[bug]** BF16 `Swiglu` backward has a gradient dtype mismatch — `CudaSwigluOp::backward` ([CudaSwigluOp.ixx](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Swiglu/CudaSwigluOp.ixx) lines 109-114) keeps gradients in FP32 by kernel design (the BF16 backward kernel takes `float* dX` / `const float* dY`), casting `output_gradient`/`input_gradient` `rawData()` to `float*`. But `Swiglu<Cuda, BF16>` (the component) allocates its `input_grad_` and receives `output_grad` as BF16 tensors, so the cast reinterprets BF16 bytes as float and writes 4-byte floats into a 2-byte-per-element buffer — garbage + out-of-bounds. Either the component must allocate FP32 gradient buffers for BF16 activations (matching the documented FP32-gradient-boundary design) or the op must convert. Unexercised because training is parked. FP32 backward is consistent. Surfaced by `Swiglu.Cuda.cpp` `Backward_MatchesReferenceGradients` (BF16 case `GTEST_SKIP` pending this fix)
- [x] **[bug, FIXED 2026-06-16]** Non-quantized CUDA Linear cuBLASLt GEMM was `CUBLAS_STATUS_NOT_SUPPORTED` — **root cause corrected**: it is NOT the NT row-major layout (MNIST, bias-free FP32, runs multi-row NT GEMMs fine and trains), it is the **bias epilogue**. `build_linear_plan` set `CUBLASLT_EPILOGUE_BIAS` for the non-quantized path, and `cublasLtMatmulAlgoGetHeuristic` returns `CUBLAS_STATUS_NOT_SUPPORTED` (15) for `CUBLAS_COMPUTE_32F` + bias epilogue (the same epilogue carries the Ada multi-row `INVALID_VALUE` constraint the FP8 path already documents). With no biased non-quant model shipping (Llama is bias-free), GPT-2/Bard was the first to hit it: `buildCublasLtPlans()` threw, `use_cublaslt_` fell to false, and batch forward + all backward threw "no valid forward execution path available". **Fix**: the non-quantized path now mirrors the FP8 path — plan built `has_bias=false`, bias added post-GEMM by `cuda_add_bias` (added an FP32 overload beside the existing BF16 one in [CudaFp8Prefill.cu](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Fp8Prefill/CudaFp8Prefill.cu)/`.cuh`). This also unblocks **all CUDA non-quantized training backward** (the backward plans carry no bias epilogue, so once forward builds, `use_cublaslt_` stays true). Decode (`outer_size == 1`, matvec) was always unaffected. The `Linear.Cuda.cpp` `Forward_MatchesReference` / `Backward_MatchesReferenceGradients` cases can be un-`GTEST_SKIP`'d once validated on-GPU. **Un-skipped 2026-06-17 (stale skip comments also blamed the NT row-major layout, corrected to the bias epilogue) — awaiting on-GPU (Ada) validation**
- [x] **[bug, FIXED 2026-06-16]** Inverted eval-mode guard in CUDA attention backward — `CudaMhaOp::backward` ([CudaMhaOp.ixx](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/MHA/CudaMhaOp.ixx) ~line 552) and `CudaGqaOp::backward` ([CudaGqaOp.ixx](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx) ~line 340) both guarded with `if ( !this->isEvalMode() ) throw "...backward called in inference mode"`. Since `isEvalMode()` is true only in `TrainingMode::Eval`, the `!` inverts it: backward threw in **training/Normal** mode (the legitimate case) and would have silently proceeded in Eval. Latent because Llama/GQA is inference-only and GPT-2 MHA training (Bard) is the first to call attention backward. Fix: dropped the `!` and corrected the message to "called in eval mode" in both ops. `CudaLinearOp::backward` already had the correct (non-inverted) `if ( this->isEvalMode() )` form; LayerNorm/Gelu/Residual ops have no eval guard. Surfaced by the Bard training loop's first backward
- [x] **[bug, FIXED 2026-06-16]** `TensorOps` element-wise math (`add`/`subtract`/`multiply`/`divide`) were **silent no-ops** — the generic wrappers in [TensorOps.Math.ixx](Mila/Src/Dnn/Tensors/Operations/TensorOps.Math.ixx) had their device-dispatch bodies commented out (`// FIXME: TensorOps<device>::add(...)`), stubbed during the alpha.5 FP8/modularization refactor (0.13.24) to dodge the MSVC C1116 ICE on the `CpuTensorOps:Math` import (the open blocker below). Dormant the whole inference line (element-wise math is training-only); Bard's backward was the first to hit it. `GptBlock::backward` accumulates the residual-stream gradient via `add(...)` into `d_res1_accum_`/`d_input_`, so the no-op killed gradient to everything behind the residual add — only the last block's MLP + `ln_final` + `lm_head` trained, parking the model at the **bigram floor** (loss ~2.4, incoherent text), never training attention or earlier layers. Diagnosed with a per-parameter grad-norm probe in the Bard trainer (dead boundary = the residual `add`). **Fix**: re-wired the four dispatch calls, each guarded `if constexpr (device == Cuda)` so CUDA computes and CPU stays compilable without the ICE-blocked `TensorOps<Cpu>` math members. Validated: Bard trains from the bigram floor to perplexity <3 (loss ~1.09 by epoch 17) with coherent text. The CPU-side no-op is the separate open blocker below
- [x] **[bug, FIXED 2026-06-17]** CPU `TensorOps` math unavailable — **MSVC C1116 ICE** — **root cause found and resolved**: the ICE was NOT `Compute.MemoryResource` (that module pulls only `<memory_resource>`/`<cstddef>`, and the CUDA Math partition imports it transitively without ICEing). The real trigger was **`#include <execution>`** in the `CpuTensorOps:Math` global module fragment — MSVC's parallel-algorithms backend transitively includes `<stop_token>` (the header named in the original backtrace), and importing that partition ICEs. The CUDA partition never hit it because it computes via device kernels, not parallel STL. **Fix**: dropped `<execution>` and replaced the two `std::execution::par_unseq` fast paths (`performElementwiseOperation`, `sum`) with serial loops ([CpuTensorOps.Math.ixx](Mila/Src/Dnn/Compute/Devices/Cpu/Tensors/Operations/CpuTensorOps.Math.ixx)) — element-wise math is the CPU reference/correctness backend, so the >10000-element parallel path is not worth re-introducing the dependency. Then un-gated the import + `MathOps` base ([CpuTensorOps.ixx](Mila/Src/Dnn/Compute/Devices/Cpu/Tensors/Operations/CpuTensorOps.ixx)) and removed the `if constexpr (device == Cuda)` guards from `add`/`subtract`/`multiply`/`divide` ([TensorOps.Math.ixx](Mila/Src/Dnn/Tensors/Operations/TensorOps.Math.ixx)). **Validated 2026-06-17: builds green (VS2026) and the full `Math.Cpu.cpp` suite passes** (was failing with uninitialized-result garbage). Unblocks the `GptBlock<Cpu>` finite-difference sentinel below and the GptTransformer CPU convergence oracle (Training Revival)
- [ ] **[bug]** `Linear::forward` dead fast-path — `leading_shape_` ([Linear.ixx](Mila/Src/Dnn/Components/Linear/Linear.ixx) line 603) is declared and read (line 173) but never assigned in `onBuilding`, so `input_shape == leading_shape_` is always false. Every `forward()` takes the `output_view_` branch and heap-allocates a fresh view wrapper, even when the runtime shape equals the build shape — the build-time `output_` fast path is unreachable. Functionally correct (right shape and values), but a per-call allocation on the decode hot path (8B FP4, one token per step). Fix: set `leading_shape_ = input_shape` in `onBuilding`; optionally cache `output_view_` so repeated decode steps reuse it. Surfaced by the Test Suite Revival prefill->decode shape-regime test design
- [ ] **[bug]** `GptTransformer` ctor creates the `ExecutionContext` before validating the device type — the initializer list runs `owned_context_( createExecutionContext( device_id ) )` before the body's `device_id.type != TDeviceType` check ([GptTransformer.ixx](Mila/Src/Dnn/Components/Transformers/Gpt/GptTransformer.ixx) ~line 99). So a mismatched (e.g. CUDA) `DeviceId` on a `<Cpu, ...>` network constructs a CUDA context first; on the CUDA dev build it still throws `std::invalid_argument` from the body, but under `MILA_ENABLE_CUDA=OFF` the context creation may fail first with a different exception. Fix: validate the device type before creating the context (match MLP/GptBlock, which check first). Surfaced by `GptTransformer.Cpu.cpp` `Construct_DeviceTypeMismatchThrows` (kept as a sentinel)
- [x] **[bug, RESOLVED 2026-07-02]** GptBlock composed-backward finite-difference gradient check — `GptBlock.Cpu.cpp` `Backward_InputGradientMatchesFiniteDifference` is **green**, and the newly added standalone `MultiHeadAttention.Cpu.cpp` `Backward_MatchesNumericGradient` probe (which the residual skip paths cannot mask) is **also green**, so `CpuAttentionOp::backward` is numerically correct — the prime suspect is exonerated and the composed sentinel passes on its own merit. The chain that produced the original failure (CPU `TensorOps::add` no-op killing residual-gradient accumulation) was fixed by the C1116 resolution; with CPU math live the composed gradient matches. No remaining defect. Historical context retained below. Prime suspect: the **MHA CPU backward**, which is only shape-tested (`MultiHeadAttention.Cpu.cpp` `Backward_ReturnsInputShapedGradient`), never numerically. Other candidates: the `Residual` pair-backward accumulation in `GptBlock::backward`, `LayerNorm` backward composition, or FP32 finite-diff noise through softmax+LayerNorm at the chosen tolerance. Triage MHA-first: add a finite-difference numeric gradient check to `MultiHeadAttention.Cpu.cpp` to isolate, then re-check GptBlock. Case is `GTEST_SKIP`'d (kept in place) so the suite stays green; re-enable on fix. Surfaced bringing the Bard GPT-2 stack online. **Update (2026-06-16):** prime suspect revised — CPU `TensorOps::add` is a no-op (CPU math gated off pending the MSVC C1116 blocker above), so `GptBlock::backward`'s residual-gradient accumulation (`add` into `d_res1_accum_`/`d_input_`) does nothing on CPU, which alone breaks the composed gradient. The identical bug on CUDA was the Bard bigram-floor regression (now fixed). This sentinel cannot pass until CPU math is restored; revisit after the C1116 fix, then re-check whether MHA CPU backward also needs numeric coverage. **Update (2026-06-17): the C1116 blocker is RESOLVED (CPU math live, see FIXED entry above) — the residual-gradient `add` now computes on CPU. Re-run this sentinel: if it still fails, the remaining suspect is MHA CPU backward (only shape-tested), which then needs a finite-difference numeric check to isolate**
- [ ] **[bug]** `GroupedQueryAttention::forward` standalone path is a no-op stub — the concatenated-QKV `forward()` ([GroupedQueryAttention.ixx](Mila/Src/Dnn/Components/Attention/GQA/GroupedQueryAttention.ixx) ~line 176) has its `positional_op_->prefill( input, *output_view_, 0 )` call commented out behind a `FIXME`, so for the KV-cache (CUDA) backend `forward()` returns an **uninitialized** `output_view_` without running attention. The only working compute paths are `prefill(q, k, v, offset)` / `decode(q, k, v, offset)` — which take **separate** q/k/v tensors (not the concatenated layout `forward()`/`validateConcatenatedQKVShape` advertise), require `setState(GqaState)` scratch wiring, and the KV-cache lifecycle normally driven by the owning transformer. Fix: either wire the standalone `forward()` to prefill the concatenated input, or remove it and make prefill/decode the only entry points (documenting that GQA is transformer-driven). Until then GQA attention numerics are untestable at the component level — they belong to an operation-level `CudaGqaOp` test that owns the `GqaState` scratch + cache orchestration (the GQA backfill item). Surfaced while scoping `GroupedQueryAttention.Cuda.cpp` (component test is surface-only as a result)
- [ ] **[bug/API]** `Softmax::backward` has an inconsistent gradient contract vs the other leaves — its signature is `backward(input, output_grad, input_grad)` where (a) the first argument must be the forward **output Y**, not the input X (both `CpuSoftmaxOp` and `CudaSoftmaxOp` alias arg1 as `probs`/`Y` — [CpuSoftmaxOp.ixx:222](Mila/Src/Dnn/Compute/Devices/Cpu/Operations/CpuSoftmaxOp.ixx), [CudaSoftmaxOp.ixx:263](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/Softmax/CudaSoftmaxOp.ixx)), and (b) `input_grad` is a caller-provided out-param the op **accumulates** into (`slice_dx[...] += grad`, [CpuSoftmaxOp.ixx:249](Mila/Src/Dnn/Compute/Devices/Cpu/Operations/CpuSoftmaxOp.ixx)) but neither the component nor the op zeroes it first — so a caller that doesn't pre-zero reads uninitialized memory. By contrast `Linear`/`LayerNorm` own, zero, and *return* `input_grad` and take X. Decide the harmonized contract: either Softmax owns+zeroes+returns input_grad like the other leaves (preferred for API consistency — public-API change), or the out-param accumulate semantics are documented and every caller must zero. Surfaced by `Softmax.Cpu.cpp` `Backward_MatchesReferenceGradient` (greened by fixing the test to pass Y and zero input_grad; the component contract is the residual debt)
- [ ] **[bug]** CUDA Softmax backward is an unimplemented throwing stub — `Detail::cuda_softmax_impl<float>::backward` and `<half>::backward` are commented `// FIXME` with `throw std::runtime_error("...needs review")` ([CudaSoftmaxOp.ixx:73,100](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/Softmax/CudaSoftmaxOp.ixx)), so `Softmax<Cuda>::backward` throws rather than computing. Harmless today (standalone Softmax backward is unused on the inference path; attention softmax is fused inside MHA/GQA), but any `Softmax.Cuda` backward-numeric coverage or a training stack that calls the standalone op will hit it. Implement + validate the `cuda_softmax_backward` kernel (dX = Y*(dY - dot(Y,dY))) when CUDA softmax backward is needed. Surfaced reviewing the Softmax backward contract above
- [ ] **[bug/robustness]** BF16 `TokenEmbedding` CUDA kernels require embedding dim `C % 8 == 0` and enforce it with a kernel-launcher `assert` ([TokenEmbedding.Bf16.cu:107,125,143](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Embeddings/Kernels/TokenEmbedding.Bf16.cu)) — they vectorize 8 bf16 per `int4` load. A non-conforming `C` **aborts the process** (crashes the GTest harness in debug) rather than failing gracefully. Harmless for shipping models (Llama embedding dims are always multiples of 8), but two gaps: (1) the constraint is undocumented at the `TokenEmbedding` component level (a caller building `<Cuda, BF16>` with `C % 8 != 0` gets a crash, not a validation error) — add an `onBuilding` precondition check that throws; (2) consider a scalar-tail path if an arbitrary `C` ever needs support. FP32 kernels need only `C % 4` (4 floats per `int4`). Surfaced by `TokenEmbedding.Cuda.cpp` BF16 `Forward_GathersEmbeddingRows` crashing at `kEmbed = 4` (test fixed to use `kEmbed = 8`)
- [ ] **[bug/API]** `getGradients()` before-build behavior is inconsistent across the component tree — `CompositeComponent::getGradients()` throws `std::runtime_error` if called before `build()` ([CompositeComponent.ixx](Mila/Src/Dnn/Core/CompositeComponent.ixx), kept `isBuilt()` guard), but the leaves (`Linear`, `LayerNorm`, `TokenEmbedding`, etc.) silently return an empty vector before build (their gradient pointers are simply null). The base-class doc contract ([Component.ixx](Mila/Src/Dnn/Core/Component.ixx) `getGradients`) says `@throws ... if called before the component has been built`, which the leaves do not honor. This is a leftover seam from the 2026-06-17 inference-mode contract change (throw-in-inference -> return-empty; `CompositeComponentTests.GetGradients_ThrowsBeforeBuild` still asserts the composite throw). Decide the unified before-build rule: either add an `isBuilt()` guard to the leaves so all components throw before build, or relax the composite + base doc to "empty before build / inference, populated in training". No test currently fails (no leaf asserts throw-before-build), so this is a contract-coherence cleanup, not a live break

---

## Training Revival

Recover the validated GPT-2 / MLP training path. MNIST (MLP) and Bard (GPT-2 generation) were
complete, working samples parked behind an explicit `FIXME: Re-enable after alpha.5 completed`
trigger (`Mila/Samples/CMakeLists.txt:3-4`) that has now fired. Reviving them reactivates the half
of the library inference never exercises — the AdamW optimizer, the loss and backward kernels,
gradient flow, train-from-scratch init. The **primitive/component tests are the correctness oracle**
(forward numerics, finite-difference gradient-checks, optimizer step-convergence, loader contracts,
init-at-precision); convergence is an emergent consequence of those, not an independent assertion.
The samples are **usage demos** ("this is how to use Mila") and a **bug-discovery mechanism** — the
Bard revival surfaced three latent CUDA-training-backward bugs the unit suite lacked — and each such
discovery becomes a primitive regression test rather than a "does the sample still converge" gate.
Several deferred Consolidation items (AdamW debug instrumentation, the CUDA
`fill_normal` FP32-only gap) fold into this milestone. **Scope is GPT-2 / MLP training only** —
Llama 3.1/3.2 fine-tuning is explicitly out of this release, remaining a Future Direction.

- [~] **(lead — timeboxed spike)** Revive **MNIST + its tests** against the current API — MNIST is the MLP (simpler than Bard's GPT-2/BPE/transformer surface), so it is the cheapest representative slice. Re-enable the sample (`Mila/Samples/CMakeLists.txt:3`) and its tests; pass/fail = builds, runs, trains to target accuracy, tests green. **Sample + spine tests GREEN (2026-06-15): sample builds and trains to ~97.9% test accuracy; spine tests (Linear/Gelu/Network/AdamW/DataLoader) landed in alpha.6+58.** Remaining for full spike closure: the end-to-end convergence oracle + `MnistDataLoader` contract test (net-new items below). The MNIST surface is already partly re-aligned (`MnistClassifier`/`MnistDataLoader` use the component-owned forward/backward API, `BuildContext`/`RuntimeMode::Training`, `setTrainingMode`, `Network::createOptimizer`), so the spike measures the MNIST spine specifically: `Network` + `AdamW` re-green (Test Suite Revival), Linear/Gelu gradient-check (the net-new archetype), the `MnistDataLoader` contract test, and the end-to-end convergence oracle below. Measures all three revival buckets at once and sets the milestone dates on evidence rather than the day-or-3 estimate. **Do this first**
- [~] Re-enable MNIST + Bard in the build — flip the `FIXME: Re-enable after alpha.5 completed` triggers (`Mila/Samples/CMakeLists.txt:3-4`) **staged: MNIST first** (its spine + tests green from the lead spike), **Bard second** (once the GptTransformer/tokenizer/`TokenSequenceLoader` spine is green) — and add the Samples build to CI (pairs with the Project Hygiene "Samples build to CI" item). **MNIST sample re-enabled at source** (`Samples/CMakeLists.txt` now `add_subdirectory(MNIST)` gated under `MILA_ENABLE_CUDA`, mirroring Chat — the sample links `CUDA::cudart` and instantiates the `CudaPinnedMemoryResource` path unconditionally, so it is CUDA-only until the CPU-only build-coherence refactor); `MnistClassifier::onBuilding` corrected to build each child with `build_config.withShape(...)` (a single shared `build_config` threw in `Linear::onBuilding` for fc2/fc_output — the spine-test gotcha). **MNIST validated 2026-06-15: builds green (VS2026/CUDA) and trains from scratch FP32 to ~97.9% peak test accuracy (99.85% train) over 20 epochs — full spine (forward chain, backward gradient flow, AdamW step, train-from-scratch init) exercised end-to-end; mild late-epoch overfit (test loss U-turns at epoch ~9), expected with no dropout.** **Bard sample re-enabled at source 2026-06-16** (`Samples/CMakeLists.txt` now `add_subdirectory(Bard)` under the same `MILA_ENABLE_CUDA` gate; `Bard/CMakeLists.txt` cleaned of stale commented module refs and switched to the `MILA_DATASETS_DIR` absolute-path macro like MNIST). Sample drift fixed to current API, mirroring MNIST's bring-up: dropped the retired `ComputePrecision::Policy precisionPolicy` field + `--precision-policy` CLI from `BardConfig`/`Bard.cpp`, and pointed the data dir at the real `Data/Datasets/Shakespeare` layout. `BardTrainer.ixx` already used the current surface (`GptConfig` builders, `GptTransformer`, `TokenSequenceLoader`, `createOptimizer<AdamW>`, `BuildContext`/`RuntimeMode::Training`, `setTrainingMode`). Bard's sample-local `CharLMTransformer`/`CharLMDataLoader` stay retired-in-place; the live Bard sample compiles only `BardTrainer.ixx` + `BardConfig.ixx` and leans on the library `GptTransformer` + `TokenSequenceLoader`. **Bard VALIDATED 2026-06-16: builds green (VS2026/CUDA) and trains FP32 from scratch to perplexity <3 / loss ~1.09 by epoch 17 with coherent Shakespeare-structured text, after fixing three latent CUDA-training-backward bugs surfaced one at a time (cuBLASLt bias epilogue -> inverted attention eval-guard -> TensorOps math no-op, all FIXED above).** Remaining: the Bard end-to-end convergence oracle (the CPU oracle is gated behind the GptBlock CPU finite-difference failure / CPU-math C1116 blocker, but Bard trains on CUDA so this is not a hard blocker for the sample)
- [~] Re-enable + re-align the AdamW path — `AdamW.Cpu.cpp` is **already re-greened and active** (`Tests/CMakeLists.txt` Section 1, with a NET-NEW closed-loop convergence case); only `AdamW.Cuda.cpp` remains disabled (Section 3, "REVIVE later — MNIST/training spine GPU companions"). Remaining: re-enable the `AdamW.Cuda.cpp` companion and resolve the deferred AdamW debug instrumentation (strip-vs-gate the `CudaAdamW.cu` printf guards + `CudaAdamWOptimizer.ixx:270`) in the same pass
- [ ] Fix the CUDA `fill_normal`/`fill_uniform` FP32-only gap (the deferred Consolidation training-only item) — it corrupts BF16 train-from-scratch init; the CUDA dtype counterpart to the `CpuTensorOps.Random` backend. Pair the fix with the **init-at-precision test**: revive `Tensor.Initializers.cpp` (+ the deferred `TensorOps/Random` slice) as a `TYPED_TEST` precision sweep (FP32 **and** BF16), turning this latent corruption into a red test rather than a silent one
- [~] **[net-new]** **Training-loop integration test (sample-independent)** — a small integration test living in `Tests/` that builds a tiny graph **from library primitives in the test itself** (Network + AdamW + a couple of batches), runs a fixed step budget, and asserts the loss strictly decreases. It validates *composition/wiring* between individually-validated primitives — the class of bug the Bard revival surfaced (cuBLASLt bias epilogue, inverted eval-guard, `TensorOps::add` no-op) — **not** whether the `Samples/` code converges; it must never import sample code. **Largely already delivered for the MNIST spine:** `Modeling/Network.Cpu.cpp` landed exactly this (`Linear->Gelu->Linear` + `createOptimizer` + `zeroGradients` + a training-loop loss-decrease assertion). Remaining: a GPT-2-stack analogue for the Bard spine (a tiny `GptBlock`/`GptTransformer` loop). Keep the budget small so both ride the `MILA_ENABLE_CUDA=OFF` CI gate. The samples themselves are validated by **running** them (by eye, done on CUDA) as demos; "does the sample still build/run" is a Production Hardening smoke concern, not a unit-suite gate
- [ ] **[net-new]** **Optimizer step-convergence test** — beyond `AdamW.*` config/mechanics re-green, add a "minimizes a known convex objective in N steps" case proving the update direction + bias-correction are correct, not just that `step()` runs. Foundational to trusting both samples' training loops
- [~] **[net-new]** **Concrete data-loader contract tests** — the base `DataLoader` test re-greens under Test Suite Revival, but the two concrete loaders are untested (the `MnistDataLoader` lives in the sample with no test at all): `MnistDataLoader` (pixel normalization to [0,1], one-hot target encoding, shuffle on reset, batch shapes, IDX magic-number validation) and `TokenSequenceLoader` (INT32 input/target offset-by-one, pad handling, `numBatches`). MNIST loader with the MNIST spike, token loader with the Bard slice. **`TokenSequenceLoader.cpp` contract test re-enabled 2026-06-16 with the Bard slice** (Section 2 of `Tests/CMakeLists.txt`; CPU cases ride the `MILA_ENABLE_CUDA=OFF` gate, CUDA-pinned cases stay `#ifdef`-guarded): construction validation, batch iteration, reset, target-is-input-shifted, tensor shapes, multi-epoch, threading stress. Remaining: the `MnistDataLoader` contract test
- [ ] **[net-new]** **TrainingMode / RuntimeMode behavior coverage** — the two-axis lifecycle (`BuildContext(RuntimeMode::Training)`, `setTrainingMode(Eval/Normal)`, `isInferenceMode`) gates gradient-buffer allocation and is *why* the component tests were disabled (Consolidation bucket E linchpin). Add explicit cases asserting build-mode and runtime-mode transitions allocate/skip gradients correctly, so the lifecycle fix has a regression guard
- [ ] Revive the loss + backward path — CrossEntropy / SoftmaxCrossEntropy components and tests (`Mila/Tests/Dnn/Components/Losses/*` exist, commented) and the backward-pass stubs (Consolidation bucket D). **Started (alpha.6+68):** a pattern-conforming [CudaSoftmaxCrossEntropyOp.Dispatch.ixx](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Loss/CudaSoftmaxCrossEntropyOp.Dispatch.ixx) was added (the `cuda_softmax_crossentropy_impl<TComputeType>` FP32/BF16 dispatch struct) and put in the build, but is **not wired in yet** (no primary `Compute.CudaSoftmaxCrossEntropyOp` unit imports `:Dispatch`; the op does not call it). Loose ends to resolve when wiring: (1) **`always_false_v` is undefined/unimported** — used in the primary-template `static_assert` but defined nowhere in the tree (only `always_false` appears, as prose in `OperationDispatch.md:372`); needs a `template<typename> inline constexpr bool always_false_v = false;` helper imported here, or switch to the sibling idiom; (2) **pattern divergence** — the existing CUDA ops (`CudaSoftmaxOp.ixx:48`, etc.) use a *forward-declared* primary (`struct cuda_X_impl;`, no body) so an unsupported type fails as an incomplete type, whereas the new file gives the primary a defined body + `static_assert`; the spec (`OperationDispatch.md:372`) actually prescribes the `static_assert` form, so spec and existing code disagree — pick one as canonical; (3) the **BF16 specialization is a no-op stub** (kernel calls commented, `half`-typed) — same FP16/BF16 desync as the poisoned-row / FP16-removal bucket, resolve consistently (drop BF16 row vs implement); (4) the `probs` out-params are commented out throughout — decide whether the fused op materializes probabilities
- [ ] Revive the `Dropout` component — parked at `Dev/Components/Regularization/` (`Dropout.ixx` + `DropoutConfig.ixx`), moved out of `Src` during the Consolidation dispatch close-out so the legacy `OperationRegistry`/`UnaryOperation` cluster could be deleted. It is pre-refactor on every axis: three-axis `Component<TDeviceType, TInput, TOutput>` + `ModuleBase`, registry-string op creation (`"Cpu::DropoutOp"`/`"Cuda::DropoutOp"`), and no concrete `DropoutOp` exists. Re-authoring is net-new training-only work (inference never uses dropout): write `CpuDropoutOp`/`CudaDropoutOp`, add a `DropoutOp` `OperationTraits` specialization per backend, rewrite the component to the two-axis `Component<TDeviceType, TPrecision>` surface, restore it to `Mila/CMakeLists.txt` + `Mila.ixx`, and add its tests. Belongs here because the mask/backward path is exercised only by the training revival
- [ ] **ProgressReporter mechanism** — design the cross-cutting progress facility for long-lived ops (the `BpeVocabulary` training `\r` progress at `:600`/`:613`, plus `PretrainedReader` load and load-time quantization are candidates). Injected per-operation (on the op's config, **not** a global facade — progress is scoped to one call, unlike the process-wide logger), null default, library owns throttling, cancellation first-class (`bool` return or `std::stop_token`), documented threading contract. Mirrors the Logging subsystem's *shape* but is a separate concern (progress = transient/overwrite-in-place; logging = append-only events). The Consolidation debug strip leaves the `BpeVocabulary` training progress in place as living training-path code — it migrates here, it is not deleted
- [ ] Validation — the training-path **primitive suite** (gradient-checks, optimizer step-convergence, loader contracts, init-at-precision, and the sample-independent training-loop integration test) green and CI-gated; train-from-scratch validated at the precisions the samples use; the samples run as demos (MNIST trains to target accuracy, Bard generates coherent text) — validated by running them, not gated in the unit suite

---

## Packaging

A downstream app consuming Mila via `find_package(Mila)` currently fails to build. C++23
module interface units cannot ship as portable BMIs, so the consumer's toolchain
recompiles the installed `.ixx` units, and each pulls its kernel header via a file-relative
quoted include that resolves against the wrong tree on install. The real defect is how the
`Mila` target is composed: kernel `.cuh`/`.h` headers are raw `add_library` sources (no base
dir, no install rule, no usage requirement); CUDA sources are added unconditionally though
`enable_language(CUDA)` is gated on `MILA_HAS_CUDA`; and three categories (`.cu` link-only
instantiations, `.cuh`/`.h` shippable declarations, `.ixx` interface units) are flattened
into one list.

- [ ] **[gate]** Single coherent restructuring (not a destination patch): model headers as `FILE_SET HEADERS TYPE HEADERS BASE_DIRS Src`; migrate file-relative quoted includes to angled includes anchored at one `Src` root (vendored `Deps` gets its own root so nlohmann becomes `<nlohmann/json.hpp>`); set `BASE_DIRS` on the `CXX_MODULES` file sets to the same `Src` root; move all CUDA `.cu`/`.cuh` sources under `if(MILA_HAS_CUDA)` via `target_sources`; replace the `install(DIRECTORY …)` glob with `install(TARGETS Mila … FILE_SET HEADERS)`. The include root must be on Mila's own build path (current root at `Mila/CMakeLists.txt:128` is INTERFACE-only — make it PUBLIC or add a PRIVATE entry or the in-tree build breaks once includes are anchored); install the generated `Version.h` and `Deps/` alongside the modules. Validate with a throwaway `find_package(Mila)` + `import Mila;` consumer wired into CI (Mila's own CI stays green and will not catch packaging regressions on its own)
- [ ] Suggested sequencing: convert one CUDA op to angled includes and get the in-tree build green first (proves the root/`-I` model), then bulk-convert backend-by-backend (the compiler flags every missed header), then do the install-side CMake and the consumer test last
- [ ] **[deferred, later in Beta]** Decide whether the kernel `.cuh` *declarations* belong in the public install surface at all — kernels are explicitly instantiated per precision in `.cu` files compiled into the archive, so consumers link the symbols and only need declarations to call the launch wrappers; the shippable surface may be reducible. Separate architectural decision, out of scope for the packaging fix above

---

## Module Hygiene (includes/imports + Doxygen)

Over alpha the module surface accumulated unneeded `#include`s/`import`s and drifted
Doxygen. Both are large, mechanical, low-risk-per-edit, high-volume diffs, deferred until
a cross-compiler build existed (a hard prerequisite for the include work). The WSL Clang
oracle now exists (Clang 21 + CUDA 13.3 + gcc-15 host), and the dev-container build is validated
(2026-07-17: `Docker/` reworked onto the CI toolchain, builds Mila + runs Gemma 4 12B FP4 Chat on the
GPU — surfaced + fixed two clang-only transitive-import breaks, `Network.ixx` + `Gemma.ixx`). GCC 16
(a second, independent module oracle) is **deferred to vNext** — the v0.20 supported Linux compiler is
Clang-only (see [ROADMAP.md](ROADMAP.md), *Production Hardening* / vNext hardening carry-over). Surface: 287 `.ixx` units, ~1,810 `import` lines, ~1,419 `#include`
lines, ~1,950 `@brief` / ~1,100 `@param` / ~257 `@tparam` / ~218 `@file` tags across 258 files.

There is no reliable off-the-shelf tool for C++23 module `import` cleanup (IWYU and clangd
do not understand the module graph), so the compiler is the only ground truth. MSVC
transitive resolution is the trap: a line can be removed and MSVC still compiles because the
symbol arrives transitively — so "still builds on MSVC" does not prove the line was unused.
The honest oracle is a **Clang or GCC** build. The cruft is real and visible — even
`Linear.ixx`, the dispatch reference, imports `Dnn.TensorOps` twice.

Includes/imports:

- [ ] Phase 0 — exact-duplicate `import`/`#include` dedup within each file; pure text analysis, scriptable across all 287 units, zero compile cost and zero risk
- [ ] Phase 1 — candidate report (no edits): heuristic scan flagging imports/includes whose symbols never appear in the file body; over-reports by design, so it is a worklist to size the job, not a verdict
- [ ] Phase 2 — compiler-verified removal, leaf modules first: scripted remove → rebuild → revert-on-failure, batched per file with binary-search on failures, verified against Clang/GCC rather than MSVC so visible cruft is not traded for invisible transitive coupling

Doxygen staleness (these tiers, plus the docs-site CI items under "Release Assets & CI" below, are the engineering detail of the **API Documentation** milestone in [ROADMAP.md](ROADMAP.md)). Comments have no compiler, so the prerequisite for the whole pass is an oracle that makes drift machine-visible — do not approach this as one heroic read-everything sweep:

- [x] Oracle + ratchet — turn Doxygen's own warnings into both the worklist and the anti-rot gate. **Mechanism correction (2026-07-02):** the docs are configured by CMake `DOXYGEN_*` variables in `Mila/Docs/CMakeLists.txt` via `doxygen_add_docs`, NOT by `Mila/Docs/Doxyfile.in` (that file is empty and referenced nowhere — a vestigial placeholder, retire candidate). `WARN_NO_PARAMDOC` was already `YES` there. **Added `DOXYGEN_WARN_IF_DOC_ERROR YES` + `DOXYGEN_WARN_LOGFILE = <build>/docs/doxygen-warnings.log`** (docs-build-only; `MILA_ENABLE_DOCS` is opt-in/OFF by default, so no impact on normal or CI C++ builds) so a docs run writes the Tier 2 candidate list (`@param`/`@tparam` mismatches + undocumented params) to a captured log. **Ran it (2026-07-02):** Doxygen **1.17** (already installed locally) over `Src/**/*.ixx` with `EXTRACT_ALL=NO` (so the `@param` warnings fire) produced a **265-warning worklist** — 54 stale `@param` names + ~184 undocumented params + 6 parse-noise; 71 of 129 warning-sites are internal ops/kernels, so the `INPUT` allowlist narrowing roughly halves it. C++23 module `.ixx` units parse fine (`EXTENSION_MAPPING = ixx=C++`). **Tier 2 + full residual cleanup DONE (2026-07-02): the `.ixx` public-API code-doc worklist is ZERO (265 -> 0).** After the narrowing (235 -> 72) and the stale-`@param` fixes (-> 0), the remaining 20 were cleared: `@example`-command misuse -> `@code`/`@endcode` (also fixes broken example rendering); unescaped `<pad>`/`<name>` HTML-tag token strings -> backticks; undocumented `Tensor`/`TrainerFactory`/`Softmax`-ctor params documented; stray `@param`/`@return` from orphaned `/** */` blocks above `//`-commented methods (`BuildContext::withQuantization`, `MemoryResourceTracker::memcpy`/`memset`, `Softmax::getConfig`) neutralized by `/**` -> `/*`. **Latent docs-config bugs fixed** (found validating a faithful mirror of the real `doxygen_add_docs` config with README mainpage + HTML): `EXTENSION_MAPPING = ixx=C++` (Doxygen was not parsing the modules as C++ at all — real bug), `FILE_PATTERNS = *.ixx`, `WARN_IF_UNDOCUMENTED = NO` (Oracle gates on drift + param docs, not document-every-symbol), `GENERATE_LATEX = NO` (HTML-only; kills the epstopdf/TeX error). **Ratchet FLIPPED (2026-07-02):** the two relative README-mainpage links (`[getting-started.md]`/`[License.md]`) were converted to absolute GitHub URLs — matching the README's own existing convention (its ROADMAP.md links already use `.../blob/dev/...`), so Doxygen treats them as external (no `\ref`). A faithful mirror of the real `doxygen_add_docs` config (README mainpage + HTML, LaTeX off) then verified **TRUE ZERO**, and `DOXYGEN_WARN_AS_ERROR = FAIL_ON_WARNINGS_PRINT` is set. Mechanism verified: doxygen exits 0 on the clean tree and would fail on any reappearing warning. The earlier call-graph truncation risk is **eliminated**: the docs-CI decouple set `HAVE_DOT=NO` in the canonical Doxyfile, so there are no dot graphs to truncate and the ratchet is now fully reproducible locally (no graphviz). Highest-leverage item — it both drove and now locks the tiers, and gave the shrinking-count-to-zero definition of "done"
- [x] Tier 0 — non-ASCII / mojibake in comments **DONE (2026-07-02)** — the `Src` tree is ASCII-clean, verified 0 non-ASCII bytes outside 4 legitimate UTF-8 BOMs (left in place: file-encoding markers, not comment mojibake, MSVC-benign). **Rename fold-in:** `Comonent.TrainingMode.ixx` -> `Component.TrainingMode.ixx` (file + `@file` tag + the single `Mila/CMakeLists.txt:409` reference; module partition `Dnn.Component:TrainingMode` unchanged so no importer touched). **Comment/string scan (byte-exact, two-phase, CRLF-preserving):** Phase A rewrote 77 files converting 547 valid-UTF-8 glyphs — em-dash `—` x353 -> `--`, box-rule `─` x102 -> `-`, arrow `→` x49 -> `->`, `×` x21 -> `x`, `≡` x4 -> `==`, box corner/tee `┐└┤` -> `+`, box vertical `│` -> `|` — plus the 4 literal U+FFFD (em-dash context) -> `--`; Phase B rewrote 9 files handling 28 stray CP1252 bytes (27 x 0x97 em-dash -> `--`, one 0xB7 middle-dot -> `.` in `Q.K^T`). Box-drawing tree diagrams (e.g. `Llama.Block.ixx`) flattened 1:1 to aligned `+ - |` ASCII art (each glyph one monospace column, so alignment preserved). Decision taken: **flatten to ASCII** (the ASCII-only-comments rule makes "keep Unicode" inconsistent; `+ - |` is the standard `tree --charset=ascii` convention). Scope deliberately extended beyond comments to ~12 non-ASCII em-dashes in exception/log **string literals** (display-only messages, so behavior-neutral; removes a real source-encoding-portability hazard) to reach a fully ASCII-clean tree
- [x] Tier 1 — `@file` rename drift: **DONE (2026-07-02)** — 32 files whose `@file` tag != filename corrected to `basename` via a scripted first-`@file`-token rewrite (e.g. `RocmDevice.ixx` was tagged `VulkanDevice.ixx`, `CudaMhaOp.ixx` tagged `CudaAttentionOp.ixx`, `Lpe.ixx` tagged `Gpt2Encoder.ixx`); verified drift == 0 and no doubled `@file` lines. The `Comonent.TrainingMode.ixx` case was resolved by the Tier 0 rename above. Comments-only, build-neutral
- [x] Tier 2 — `@param`/`@tparam` name mismatches: documented names no longer in the signature. Candidate list is the Oracle's `WARN_IF_DOC_ERROR` output (not a hand-built grep). **Started (2026-07-02):** fixed the `@param is_training` -> `training_mode` cascade — 10 warnings from 3 sites (`CompositeComponent::onTrainingModeChanging` inherited into 7 subclasses + `Gelu`/`Softmax` overrides), a leftover from the pre-`TrainingMode` `bool isTraining` API; also fixed the stale "true = training, false = eval" prose to `Normal`/`Eval`. Oracle re-run confirmed 10 -> 0 (total 265 -> 235). **The "review before applying" discipline paid off immediately:** `TokenSequenceLoader`'s `@param is_training` looked identical but is a *real* `bool is_training` ctor parameter — verified against the log (not flagged) and correctly left untouched. The surface narrowing landed (EXCLUDE of internal op/kernel/optimizer subtrees + `Detail::`), dropping the worklist 235 -> 72. **DONE (2026-07-02):** all stale `@param`/`@tparam` name mismatches fixed to **zero** (from 54) across ~15 public sites — `onTrainingModeChanging`/`onBuilding` context params, GQA `prefill`/`decode` concatenated-QKV -> q/k/v/position_offset, `Component::build`, the `ModelConfig::withQuantization` orphaned-doc detach (`/**` -> `/*` on a commented-out method), RmsNorm/Residual/TensorBuffer ctor params, and the four memory-resource `do_deallocate` docs (Cpu/CudaDevice both size params unnamed -> drop both; CudaManaged/CudaPinned have a *named* `alignment` -> keep `@param alignment`, drop unnamed `bytes`). Reviewed per-site: caught the `TokenSequenceLoader` false positive (real `bool is_training` param) and the named-vs-unnamed `alignment` nuance. **Total worklist 265 -> 20.** The residual 20 are not name-drift — they are the "path to zero" for the Ratchet: `@example`-command misuse (~10), unescaped `<...>` token strings read as HTML (7), 2 undocumented public params, 2 stray `@param` on zero-arg methods, 1 `@return` on void
- [ ] Tier 3 — semantic staleness (per-subsystem judgment): `@brief`/descriptions describing the retired world (components "registering with `OperationRegistry`", "deriving from `UnaryOperation`/`BinaryOperation`", string-keyed dispatch), naming drift (`TWeightQuant` in prose vs. the spelled-out style), file-level `@brief`s exceeding the 1-3 sentence rule. One settled subsystem at a time; leave subsystems mid-refactor alone. Cheapest folded into the Test Suite Revival, which already opens each of the ~70 files — review a file's prose while it is open for re-greening rather than as a separate megasweep

---

## Public API Surface (narrowing the `Mila` umbrella)

The supported public entry point is a single `import Mila;`. Internal module names
(`Dnn.*`, `Compute.*`) are an implementation detail (also why they are intentionally not
`Mila.`-prefixed — the `Mila::` namespace already scopes symbols). Tests/samples import
submodules directly (14 today) and are not bound by the public contract. The mechanism is
correct; the open work is *scope*. At freeze the failure modes are asymmetric: too narrow is
widened later by adding exports (non-breaking); too broad can only be fixed by removing
exports (breaking). Beta should freeze the **narrowest defensible** surface. Today `Mila.ixx`
re-exports essentially the whole tree, locking in (1) every consumer recompiling the full
transitive closure into BMIs, and (2) every re-exported symbol as a frozen promise.

- [ ] Define an explicit public allowlist for `Mila.ixx` — the inference surface (models, components, tensors, execution context, `initialize`/`shutdown`, tokenizers); treat the export list as the literal API spec
- [ ] Demote non-public modules to unexported internal (still directly importable by tests/samples): `OperationRegistry`/`OperationRegistryHelpers`/`OperationsRegistrar`, `UnaryOperation`/`BinaryOperation` (both slated for removal), `Dnn.TensorBuffer` ("remove after testing"), the per-device operation modules
- [ ] Stop re-exporting the vendored `nlohmann` module/namespace through the public surface — it hands a breaking change to a third party's release schedule; the Chat sample's direct `import nlohmann.json` is a sample-layer concern
- [ ] Domain-qualify generic single-segment module names that are global-collision magnets on co-link — `Core`, `Utils`, `Components`, `Profiling` (e.g. `Dnn.Core`, `Dnn.Utils`); targeted handful of renames, independent of the no-`Mila.`-prefix rule
- [ ] **[deferred, non-breaking]** If training becomes a first-class public concern, add a separate `Mila.Training` umbrella rather than widening `Mila` — the additive direction keeps the inference surface tight

---

## Release Assets & CI

Mila is source-distributed (clone to contribute; `find_package(Mila)` from a source install
to consume), so most release-asset machinery is unnecessary — GitHub auto-generates source
archives per tag, so **tagging `master` is the release**. Release flow is a `dev` → `master`
PR; CI validates on the PR; docs publish only from `master`. During alpha the **default
branch is `dev`**; at beta **switch the default to `master`** (README/roadmap links are
branch-agnostic, so no content change needed).

Documentation site (the genuinely GitHub-bound deliverable):

- [ ] Docs generated by a GitHub Action, never committed to the source tree (Doxygen output for 287 modules with call graphs is thousands of files + binary graphs; committing per release poisons the repo history)
- [x] Fully decouple the docs job from the build — **DONE (2026-07-02).** Added a canonical standalone `Mila/Docs/Doxyfile` (single source of truth for all Doxygen settings) and rewrote `docs.yml` to run `doxygen Mila/Docs/Doxyfile` directly on a plain `ubuntu-24.04` runner: **no CUDA container, no CMake configure, no CPM cache, no Graphviz** — just a pinned Doxygen 1.17 binary download. `Docs/CMakeLists.txt` now `add_custom_target(docs …)` invoking the same Doxyfile (retired `doxygen_add_docs`), and `doxygen-build.sh` runs it too — one config, no drift; both `mkdir -p build/docs` first (Doxygen won't create nested `OUTPUT_DIRECTORY` parents). Graphs disabled (`HAVE_DOT=NO`) so no Graphviz dependency and no call-graph truncation risk against the `WARN_AS_ERROR` ratchet. Verified end-to-end locally: `doxygen-build.sh` exits 0, 276 class pages, ratchet clean. The empty vestigial `Doxyfile.in` is now fully superseded (delete candidate). **Pending a real GitHub-Actions docs run** to confirm the pinned-Doxygen release URL and the Pages publish (not locally runnable)
- [ ] Narrow what docs expose to match the public API surface — current config sets `EXTRACT_ALL`/`EXTRACT_PRIVATE`/`EXTRACT_STATIC` recursively over all of `Mila/Src`; published docs should show the `import Mila;` public surface, not every private member of 287 modules (pairs with Public API Surface)
- [ ] Verify Doxygen renders C++23 module units faithfully — module support is young; `export module`/partitions/`import` may misrepresent structure. Depends on the Doxygen staleness pass so generated docs are not loud with `WARN_NO_PARAMDOC`
- [ ] Bump docs Doxygen version — `docs.yml` installs via apt (Ubuntu 26.04 pins **1.15**; latest **1.17**); module rendering fidelity improves across releases, so install 1.17 from the upstream tarball. Pairs with module-rendering verification

CI correctness:

- [ ] CI/CD pipeline efficiency pass (**[deferred, optimization not a trust gate]**) — measured master run: Build ~18.5 min + packaging gates ~25 min = ~44 min; dev pushes ~20 min. Dominant cost is **C++23 module compilation, not `.cu` kernels** — so the CUDA-only ccache launcher misses the bottleneck. The module tree compiles up to **three times per master run** (Build, `find_package` gate, FetchContent gate). Levers ranked: (1) module-aware caching — spike whether clang-21 + ccache 4.x can reliably cache module compiles (BMIs are compiler/path-sensitive) — highest value, hardest, only one hitting the bottleneck; (2) `-O0`/Debug in CI (partial win — BMI generation is front-end work `-O0` does not reduce); (3) move FetchContent full-rebuild to tag-only if cadence rises
- [ ] Broaden compiler coverage toward the supported matrix — CI builds only **clang-21**; the primary dev compiler (MSVC 2026) and the working GCC 16 path are untested, so the compiler that previously broke the build (VS 2026 pre-18.6.2 module regression) is the one CI cannot catch. A multi-compiler CI is also the cross-compiler oracle the include/import hygiene pass needs
- [~] Reproducible **build container** — pin a CUDA `-devel` Ubuntu 26.04 image that builds Mila (+ tests) from a clean clone, the same image CI uses, so the Linux build reproduces without host toolchain drift. This is the `dev-container build` noted in the Module Hygiene intro (currently prose, not a task). Distinct from the **runtime** image under *Project Hygiene* (Distribution), which packages the already-built artifacts, not the build toolchain. Surfaces the Linux-platform ROADMAP gate (Production Hardening) alongside the compiler-matrix item above. **Build path DONE (2026-07-17):** `Docker/` reworked onto the CI toolchain (clang-21 + gcc-15 host, CUDA 13.3, CMake 4.2.3, Ninja, ccache); `mila-build-chat` builds Mila + `ChatApp`, `mila-chat` runs Gemma 4 12B FP4 on the GPU (~49 tok/s), GPU passthrough via `docker compose run` deploy-reservation confirmed. New: `Docker/Dockerfile` (rewrite, was the stale clang-19/CUDA-13.0/24.04 dev container), `build-chat.sh`/`run-chat.sh` baked helpers, `docker-compose.yml` (source bind + `/build` + `/ccache` named volumes off the bind mount), `README.md`, `scripts/chat-{build,run}.{sh,ps1}`. **Not yet [~]:** from-scratch in-container `git clone` build (validated against the bind-mounted tree), the `+tests` build (the Chat path builds `ChatApp` only), and CI building `FROM` this image (still apt-installs)
- [ ] Container build parallelism is a **memory** limit, not a core-count one — `build-chat.sh` caps `-j ${MILA_BUILD_JOBS:=4}` because the heavy module TUs (OperationTraits + the CUDA-op `.ixx` units instantiate the full device x precision x quantization dispatch table, multi-GB each) OOM a RAM-limited Docker VM at `-j $(nproc)`. On the dev box (`.wslconfig processors=10`, no `memory=` line -> ~15.5 GiB VM on a 32 GB host) `-j 10` starved and **wedged the whole Docker engine** (500s on `_ping`); `-j 3` builds safely. Follow-up: raise the WSL2 VM to `[wsl2] memory=24GB` (+`wsl --shutdown`) so a higher `-j` is safe, and/or a CMake `JOB_POOLS` that throttles only the heavy TUs
- [ ] Stage model weights off the Windows bind mount — Chat's ~24 GB weight load is slow because `Data/Models` is read across the `/mnt/d` -> Docker Desktop 9p/virtiofs share; copy the `.bin`/tokenizer into a Docker named volume (or an ext4 clone) mounted at the compiled-in `MODELS_DIR` (`/mila/Data/Models`) so the load runs at native disk speed. One-time-per-run cost today; does **not** affect inference throughput (decode measured ~49 tok/s regardless)
- [ ] Optional: trim the container library build to a single arch — the Mila library pins `CUDA_ARCHITECTURES "75;80;86;89;90"` (`Mila/CMakeLists.txt:26`, a portable fat binary for `find_package` consumers), and a target property makes that override the global `-DCMAKE_CUDA_ARCHITECTURES`, so the container builds all five arches. A default-off opt-in (e.g. `MILA_CUDA_ARCH_OVERRIDE`) would let the container build just `sm_89` for a faster first build without changing the shipped fat-binary default. **Low priority** — the 5-arch build turned out fast (single-digit minutes), so this is a nicety, not a need

Docker image publish is optional and only if the runtime image stays a beta deliverable —
a release-tagged GHCR push is a natural CI-on-tag job but equally a local `docker build &&
docker push`; automation-of-convenience, not a gate.

---

## Project Hygiene & Contributor Readiness

A beta is a trust signal; these items are about the project not contradicting itself or
wasting a newcomer's first hour.

- [x] **Third-party licensing policy — DECIDED + IMPLEMENTED 2026-07-17. THE RULE: everything in the repo that
  Mila did not write is recorded in the single root [NOTICE.md](NOTICE.md), and nowhere else.**
  Raised because Google's Gemma 4 `chat_template.jinja` was vendored as a test oracle and then documented ad hoc
  (a per-file `PROVENANCE.md` sidecar + an `ATTRIBUTIONS.md` section) — two mechanisms, neither decided. Both
  reverted; `NOTICE.md` replaces them.
  - **`NOTICE.md`** — vendored files (path, origin, license, modified?) + the build-fetched dependencies and
    their licenses. One place answers "what isn't ours, and under what terms?".
  - **`License.md`** — untouched, purely Mila's MIT grant. Deliberately NOT extended: license scanners parse it,
    and mixing third-party notices in makes Mila's own licensing harder to read.
  - **`ATTRIBUTIONS.md`** — intellectual debt ONLY (Karpathy, FlashAttention, online softmax). It defines itself
    as *the ideas, algorithms, and software projects that influenced Mila's implementation*; a vendored test
    fixture is a licensing matter, not an influence. Carries no licensing meaning.
  - **Vendored files stay pristine.** No prepended headers. If one is ever modified, say so in NOTICE.md AND in
    the file (also what Gemma ToU 3.1(3) would require). Engineering notes (how to refresh, upstream quirks) live
    with the consuming code, not in the licensing doc.
  - **Gap this closed:** `Cmake/CPM.cmake` has always been vendored third-party (MIT, Lars Melchior) and nothing
    outside the file itself said so. Now listed.
  - **Legal question RESOLVED by reading the terms (I had asserted it unverified):** the
    [Gemma Terms](https://ai.google.dev/gemma/terms) define *Gemma* as the models/weights/parameters, *Model
    Derivatives* as pattern-transfer from the weights, and *Output* as what the model emits. **A chat template is
    none of those**, so the §3.1 notice/passthrough obligations are very likely NOT triggered by vendoring it.
    Recorded in NOTICE.md anyway — provenance and hygiene, not compliance theater. NOTE this makes the earlier
    "governed by the Gemma ToU, not MIT" phrasing an over-claim.
  - **Sizing rationale (revisit if it changes):** a single root list fits a repo with TWO vendored files. The
    Chromium pattern (`third_party/<dep>/` + a per-directory `README.chromium` recording name/URL/version/
    license) is the better answer at scale and is what the improvised sidecar was unknowingly imitating — adopt
    it if Mila ever grows a real `third_party/` tree. `ThirdPartyNotices.txt` (the Microsoft style) targets
    binary redistribution; Mila ships source.
  - **STILL OPEN (separate decision, deliberately not bolted on):** a binary release that links cutlass /
    nlohmann / pybind11 / gtest / miniz / benchmark would need to carry their notices. Source distribution
    carries no such obligation, so this only bites when release artifacts contain binaries. NOTICE.md flags it.

- [ ] Marker debt triage (IN PROGRESS) — the earlier `FIXME`/`TODO` burndown (the bypassed weight initializers + CUDA `setCurrentDevice`, both DONE + validated, see CHANGELOG) is complete; **2026-06-19 all surviving `FIXME:`/`TODO:` were converted to `REVIEW:`** so nothing reads as "known broken" in public source. A fresh bucket analysis of the ~94 remaining `REVIEW:` markers (56 files) sorted them; dispositions and homes:
  - **A — dead/deprecated dispatch (~15).** GQA portion is the `CudaGqaOp` legacy A/B retirement (see Consolidation item above; Pass 1 done, Pass 2 = rename). Stragglers folded there: `CudaResidualOp.ixx:116-117` unused `input_A`/`input_B` params, `CudaOps.h:30` "declarations no longer needed", `Linear.cuh:83` "are these functions required".
  - **B — FP16 → BF16 / poisoned dispatch rows.** Already tracked: "Remove FP16" + the poisoned-row item under Consolidation.
  - **C — `dim_t` canonicalization.** New dedicated item below.
  - **D — correctness REVIEW items.** New dedicated item below (incl. the `Llama.Block.ixx:132` aliasing concern in the primary target).
  - **E — API/surface decisions.** `Mila.ixx:35` (Operations internal-only) is covered by **Public API Surface** above; the rest (MHA/GQA `initializeKVCache`/`resetKVCache` public surface, `ComponentFactory` "half-baked", factory-based tokenizers, config naming `withTrainedMaxSequenceLength`/getter-names/ambiguous LayerNorm ctors) are design calls — demote-to-note, decide per subsystem; not a release gate.
  - **F — build/training-lifecycle assertions** (`Lpe.ixx:143/187/495`, `TokenEmbedding.ixx:182`). Mostly settled by the RuntimeMode/TrainingMode two-axis redesign (see [[project_alpha6_triage]]); disposition = convert to hard assert or drop the comment. Low priority.
  - **G — cleanup nits** (defensive `cudaSetDevice`, misleading `Component.MemoryStats` field names, shared op-boilerplate helper, etc.). Demote to neutral notes; opportunistic, not gated.
  - **H — organizational / docs.** New dedicated item below.
  - Rule unchanged: fix the real ones; demote the rest to neutral notes + tracked tasks **here in BACKLOG** (not GitHub issues), ship no literal `FIXME`/`TODO` in public source. Distinct from the debug-instrumentation strip gate below.
- [ ] **[marker bucket C] `dim_t` canonicalization — kill the shape-dimension `static_cast` band-aids** — config structs use `int` for shape/vocab/embedding dimensions, forcing casts at every boundary. Marker sites: [Linear.ixx:718](Mila/Src/Dnn/Components/Linear/Linear.ixx) ("get rid of these ugly static_casts, use dim_t everywhere"), [TokenEmbedding.ixx:427](Mila/Src/Dnn/Components/Embeddings/TokenEmbedding.ixx) (cast-to-`dim_t` band-aid; `config_` uses `int` for vocab/embedding sizes), [GroupedQueryAttention.ixx:470](Mila/Src/Dnn/Components/Attention/GQA/GroupedQueryAttention.ixx) ("int? Should be dim_t?"), [Rope.Config.ixx:269](Mila/Src/Dnn/Components/Encodings/Rope/Rope.Config.ixx) (establish `dim_t` as canonical model-dimension type), [Tensor.ixx:263](Mila/Src/Dnn/Tensors/Tensor.ixx) (sub-byte types: return bits vs bytes), [TokenSequenceLoader.ixx:44](Mila/Src/Data/Loaders/TokenSequenceLoader.ixx) (`int*` token ids for the CUDA encoder kernels — semantically unsigned). One deliberate refactor touching config surfaces; do it as a unit, not opportunistically
- [ ] **[marker bucket D] correctness REVIEW items** — surfaced by the marker triage, not by tests; each needs eyes-on:
  - **[bug, priority]** [Llama.Block.ixx:132](Mila/Src/Dnn/Components/Transformers/LlaMa/Llama.Block.ixx) — "the use of tensor views here is incorrect. The Q,K,V splits of qkv_out are not actually contiguous in memory" — a potential aliasing/stride bug in the **primary validated target** (Llama). Confirm whether it is live or benign before anything else in this bucket
  - [Llama.ixx:751](Mila/Src/Dnn/Components/Transformers/LlaMa/Llama.ixx) — a scaling factor commented out for reasons "unclear"; determine whether it is needed
  - [Gelu.Fp32.cu:65](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Gelu/Kernels/Gelu.Fp32.cu) — "should be the numerically stable implementation"
  - [CudaAdamWOptimizer.ixx:169,178](Mila/Src/Dnn/Compute/Devices/Cuda/Optimizers/CudaAdamWOptimizer.ixx) — "precision check and master-parameter logic is outdated"; master-param init needs type-converting copy (training-only; pairs with the AdamW debug-instrumentation item under Consolidation)
  - [TensorOps.Transfer.ixx:93](Mila/Src/Dnn/Tensors/Operations/TensorOps.Transfer.ixx) — `Copy` should be a no-op on empty tensors
  - (Already tracked elsewhere: CUDA `Softmax` backward stub + `Softmax.ixx:370` "Why???" = the CUDA Softmax-backward bug above; `GroupedQueryAttention.ixx:176/290` = the GQA forward-stub bug above)
- [ ] **[marker bucket H] organizational / docs REVIEW items** — [TensorOps.ixx:9](Mila/Src/Dnn/Tensors/Operations/TensorOps.ixx) regroup `Zero`/`Random`/`Fill` under a `TensorOps.Init` partition (initialization-related, currently scattered); [TensorBuffer.ixx:78](Mila/Src/Dnn/Tensors/TensorBuffer.ixx) update storage-size logic for packed sub-byte (FP4) types; [GptModel.ixx:301](Mila/Src/Dnn/Models/GptModel.ixx) special-token ids should come from tokenizer metadata once tokenizer support lands; [Tensor.Partitioning.ixx:12](Mila/Src/Dnn/Tensors/Tensor.Partitioning.ixx) empty `REVIEW:` marker — drop. (The `BpeVocabulary.ixx:591` async-progress marker is the ProgressReporter item under Training Revival; the spec-doc `REVIEW:`s in `Compute.md`/`GqaMemory.md`/`Testing.Tensors.md` fold into the Doxygen/spec passes)
- [x] Debug instrumentation fully gated or removed — done by the Consolidation debug-instrumentation strip (kernel `printf`/anomaly guards removed; the BPE tokenizer warning + vocab-load info routed to `Logging::Logger`, the encode timer and progress prints deleted; the two inert commented-out CPU-op debug dumps in `CpuAttentionOp` (K/dQ) and `CpuLinearOp` (input) deleted 0.20.0-alpha.6+81). A full source audit confirmed the surviving console sites are deliberate/legitimate and were kept: the `TokenSequenceLoader` init summary (gated on `config_.verbose_logging`) + `TensorBuffer` alloc/dealloc trace (`if constexpr (TrackMemory)`) are opt-in diagnostics; `CudaExecutionContext` stderr-on-stream-error + `TokenEmbeddingConfig` `isfinite` validate-guard are error/validation paths; `Tensor.Helpers` `clog` is the Tensor print API. Training-path instrumentation is intentionally NOT stripped — it is deferred to its owning milestone (the AdamW debug item above; the `BpeVocabulary` training progress -> ProgressReporter under Training Revival)
- [ ] Test coverage of core components — now owned by the **Test Suite Revival** milestone above (re-green the ~70 authored test files, the CI ratchet, and the inference-drought backfill). No longer a loose Beta line
- [~] **Dispatch error UX — core landed (0.20.0-alpha.6+112, VS2026 build green).** Compile-time dispatch makes the compiler error the *user interface* for an unsupported `(Op, Device, Precision, Policy)` combination, and it was hostile: MSVC emitted constraint cascades with no semantic cause, and a present-but-broken specialization (the poisoned BF16 rows under Consolidation) was worse than a missing one. **Shipped in [OperationTraits.Template.ixx](Mila/Src/Dnn/Compute/Operations/OperationTraits.Template.ixx):** the `OperationTraits` primary is left **declaration-only** so an unsupported tuple names an incomplete type — MSVC/Clang report a single-line "use of undefined type `OperationTraits<Op,Device,Precision,Policy>`" naming the exact tuple, not a cascade — plus the shared **`OperationSupported<...>` concept** (SFINAE-safe completeness probe; covers both `type`- and `op_for`-bearing specializations) for `if constexpr`/`static_assert` use at instantiation sites and in multi-precision typed tests. **Design correction vs the original §12 sketch:** option (A)'s literal `static_assert(always_false)` *on the primary body* is mutually exclusive with a SFINAE-safe (B) predicate — any probe of an always-asserting primary instantiates it and fires the assert, turning the detectable `false` back into a hard error. The declaration-only primary delivers the readable diagnostic *and* keeps the predicate probeable, so (A)+(B) collapse into one mechanism. **Remaining (incremental, not a gate):** reconcile [OperationDispatch.md](Mila/Specifications/OperationDispatch.md) §12 to this decision; optional (C) named kernel concepts + wiring `OperationSupported` into the kernel `requires`-clauses so the table cannot advertise what a kernel rejects (the poisoned-row class becomes structurally impossible); optional domain-specific `static_assert(OperationSupported<...>, "...")` at component `OpType` sites if the terse incomplete-type error proves insufficient in practice. Contributor-readiness gate — a newcomer's first unsupported instantiation no longer produces 200 lines of MSVC noise
- [ ] Add the Samples build to CI (currently only tests build) so a contributor's first sample build is not the thing that breaks
- [ ] `good first issue` labels on GitHub (Beta requirement) — the exact label is `good first issue` (spaces, lowercase; hyphens break GitHub's `/contribute` + aggregator discovery). These are maintainer-authored discovery Issues promoted from this backlog (a GitHub *mechanism*, distinct from inbound user issues), each well-scoped with acceptance criteria + file paths. Mint when courting contributors (~default-branch switch); pairs with the community-health files already landed and the `CONTRIBUTING.md` gate
- [ ] `CONTRIBUTING.md` coding-standards section + `getting-started.md` onboarding guide (user-first, contributor superset) (Beta requirements)
- [ ] Ungated GPT-2 quick-start path for zero-auth first run (Beta requirement) — pre-converted permissively-licensed weights hosted on Hugging Face, fetched on first run via `resolve/` URLs over HTTPS (no Python/venv/auth); gated weights (Llama) stay a user-supplied offline conversion step
- [ ] Published Docker runtime image — slim multi-stage GPU runtime (built in CUDA `-devel`, artifacts copied into `-runtime`), release-tagged; gated weights never baked in (Beta requirement, see Distribution in ROADMAP)

---

## Native low-precision compute (Blackwell+) — Future Direction (no milestone)

**Make "BF16 is the primary reduced-precision target" a parameter, not an assumption.**
Pre-Blackwell this assumption was correct — BF16 won industry training because its FP32-equal
exponent range removed FP16's loss-scaling fragility, so Llama et al. ship BF16 and Mila treats
BF16 as the canonical activation/compute precision. Blackwell (sm_120) native FP4/FP8 tensor cores
change the calculus: the frontier moves the matmul-dominated FLOPs to FP4/FP8 + a recovered scale
(the inference analogue of Mila's existing `PerChannelFp8`/`PerGroupFp4` *storage* policies). Capture
the scoping here so the future work starts from the analysis, not a cold read. Pairs with the
Blackwell A/B rig (4070 sm_89 + 5060 Ti sm_120) and the vendored CUTLASS Blackwell kernels.

The compile-time architecture makes the *dispatch* side genuinely "add a precision": `TPrecision` is
already a first-class axis, `OperationTraits<Op, Device, Precision, Policy>` resolves the op, and a
missing specialization is a hard compile error. What makes it **more than adding a kernel**:

- [ ] **[deferred]** **Microscaling is a data-path change, not a dtype swap.** Today FP4/FP8
  is *storage-only*: weights quantized at load, scales on the **weight** side (`TWeightQuantization`,
  `weight_scales_`), GEMM dequantizes to BF16 activations (W4A16). Native Blackwell FP4 matmul wants
  **both operands** microscaled (MXFP4 = 32-element blocks sharing an E8M0 scale; or NVFP4 = 16-element
  blocks with an FP8 scale + a tensor-level FP32 scale), so **activations** need per-block scales
  computed *in the forward pass, every call* — a new hot-path activation-quantize step + scale tensors
  riding alongside activations. Format choice (MXFP4 vs NVFP4) is itself a decision.
- [ ] **[deferred]** **The one real design decision: is "compute precision" a concept distinct
  from `TPrecision`? -- LEANING (2026-07-15 FP8-linear design discussion): a new activation-quant
  policy axis, NOT redefining `TPrecision`.** `TPrecision` today conflates two roles: the activation
  *dataflow* dtype (the residual stream that RMSNorm/RoPE/softmax/residual all consume) AND the
  *accumulation* dtype -- and tensor-core accumulation is FP32 regardless of input encoding
  (FP8xFP8->FP32, FP4xFP4->FP32), so FP8/FP4 is only ever a per-GEMM **input encoding**, never the
  network compute type. That rules out "redefine `TPrecision` = FP8/FP4" as a category error (it would
  assert an FP8 residual stream + FP8 accumulate, both false; BF16-primacy is baked in around it -- the
  FP32 gradient boundary, KV-cache dtype, and residual/norm/softmax up-convert points). The clean shape
  is a `TActivationQuantization` policy axis **mirroring `TWeightQuantization`** (`NoActQuant` /
  `PerTokenFp8` / microscaled `MXFP4`/`NVFP4`), leaving `TComputePrecision` = BF16 as the
  dataflow/accumulate type it already is; configs then read as the (W,A) pairs the literature uses --
  W4A16 / W8A8 / W4A8 / W4A4. **NOT purely Blackwell future work -- the present-day first consumer
  already exists:** the Ada W4A8-FP8 prefill path (`kUseFp8ActivationPrefill`, shipped ON +103,
  [CudaLinearOp.ixx:162](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/CudaLinearOp.ixx)) is
  exactly activation-FP8 done as an internal bool with prefill-only per-token scales; the axis would
  re-express it as `TActivationQuant = PerTokenFp8` and retire the magic toggle, unifying today's Ada
  W4A8 and tomorrow's Blackwell W4A4 under one seam (aligns with the "generalize the weight-side scale
  machinery for activations first" sequencing item below). Wrinkles to resolve: the axis is
  **phase-dependent** (prefill quantizes activations; decode's memory-bound matvec does not -- unlike
  weight quant, which is phase-uniform), and a 5th `Linear` template axis grows the `OperationTraits`
  specialization set, so gate instantiation of unvalidated (W,A) pairs per
  [[feedback_validate_generation_not_just_oracle]].
- [ ] **[deferred]** **Per-arch gating gets finer.** Blackwell kernels (sm_120, CUTLASS 4.x)
  must be gated so the Ada (sm_89) build still compiles and runs — same discipline as the
  `MILA_ENABLE_CUDA` split but arch-conditional and partly *runtime* (one binary, two GPUs in the A/B
  rig). A traits row advertising native-FP4 on an op whose kernel only exists for sm_120 is the
  poisoned-row bug (see the FP16-removal item) again, arch-conditional. Decide compile-time build
  variant vs runtime capability check.
- [ ] **[deferred]** **The correctness oracle shifts.** HuggingFace token-for-token /
  BF16-reference-within-tolerance needs recalibrated tolerances (and possibly a different reference)
  for FP4-*compute* vs FP4-*storage*. Extends the Test Suite Revival precision-tag test pattern, but the
  tolerances are real work.
- [ ] **[deferred]** **Sequencing + pre-work.** Generalize the weight-side scale machinery
  into something activations can also use **first**, then add the Blackwell kernels against it —
  adding kernels first against the BF16-activation assumption forces a hot-path retrofit anyway.
  Before scoping, audit where BF16 is *assumed* vs *parameterized* (default template args,
  KV-cache dtype, conversion/loader code, the FP32 gradient boundary) — the `= TensorDataType::BF16`
  defaults are where "BF16 is the target" hides in places a kernel addition will not reach.

---

## Model Loading — load-time performance

Model load for Gemma 4 12B FP4 is ~12 s native (SN850X Gen4 NVMe). Measured decomposition 2026-07-17
(ProfileModel `[load]` + nsys `model_load` NVTX range + a temporary read/consume split): the load is
**read-bound** — H2D PCIe (2.65 s for 23.8 GB) and on-device FP4 quantize (2.21 s) both overlap and hide
under the disk read; the wall is the mmap→pinned staging read. The shipped `readAt` positioned-read fix
(15.5 → 12.1 s, 1.43 → 1.91 GiB/s, +22%) removed the mmap 4 KB page-fault storm, but the read still runs
at ~2.34 GiB/s on a drive rated ~7.3 GB/s.

- [ ] **Concurrent / async read I/O for real queue depth** — the read is throttled at effective queue
  depth 1; a synchronous single-thread `pread`/`ReadFile` loop cannot saturate a Gen4 NVMe (it needs
  several in-flight requests). **PROVEN USELESS 2026-07-17 (do not repeat):** N reader threads sharing
  ONE file handle — Windows `ReadFile` serializes concurrent ops on a *synchronous* handle, so 4 threads
  just split the same ~2.4 GB/s (measured summed-read ≈ 4× wall, zero speedup); a shared-buffer pool also
  **deadlocks** (in-order consumer + the oldest-blob reader starved of a buffer by readers racing ahead).
  **START FROM:** per-reader file handles (each reader opens its OWN `HANDLE`/`fd` → independent I/O
  streams) OR overlapped/async I/O (`FILE_FLAG_OVERLAPPED` on Windows, io_uring/aio on POSIX) issuing
  several reads at once. The reverted **striped private-double-buffer** reader (in `PretrainedReader`
  git history, 2026-07-17) is the correct deadlock-free *thread* skeleton if per-handle proves out — only
  the shared handle was wrong. **Measure before investing:** if per-handle reads don't raise throughput,
  the access pattern genuinely caps ~2.4 GB/s and this lever is dead. Est. best case ~7 GB/s → read ~3 s
  → load ~6 s.
- [ ] **[post-0.20] FP4 weight sidecar cache** — the checkpoint is 22 GB BF16 but the resident model is
  ~6 GB FP4; every load reads 22 GB and re-quantizes. A first-load-writes / later-loads-read sidecar of
  the quantized FP4 weights + scales next to the `.bin` cuts the read 22 → 6 GB *and* skips the 2.2 s
  on-device quantize (≈ 2.6 s load even at the current QD1 rate; stacks with the read-QD lever above).
  Canonical distributable checkpoint stays BF16 (design intact — CLAUDE.md "no quantized checkpoint
  format"); the cache is a local perf artifact like a `.pyc`, needing a checkpoint mtime/hash
  invalidation. New on-disk artifact → post-0.20. The bigger single win on this model.

---

## Gemma 4 — Dense Chassis (residual / follow-ups)

The Gemma 4 12B dense chassis is **complete and HF-validated** — greedy decode matches HuggingFace
token-for-token (2026-06-23) and the Chat sample runs Gemma 4 12B FP4 as its default model. The full
build-out (Gemma.md §9 foundation sequence) and the multi-week parity-resolution history are
collapsed into [CHANGELOG.md](CHANGELOG.md); the design lives in [Gemma.md](Mila/Specifications/Gemma.md)
([[project_gemma_chassis_design]], [[project_gemma_rmsnorm_raw_weights]]). What remains are the
follow-ups below — two of them **v0.20 release gates**: together they are what lets Gemma 4 12B FP4 fit
a 12 GB card.

- [x] **[gate — v0.20 release] Bounded KV ring cache** (Step 2b — a `TKvPolicy` sibling). **DONE + VALIDATED
  0.20.0-alpha.6+78** — see CHANGELOG "Gemma 4 memory-management gates". Gemma's local layers now use the
  bounded ring; coherent 8192-token chat with eviction active + op-level parity harness (decode/prefill vs
  full-cache oracle) + closed-form KV-footprint assertion + compile-time policy-routing test. Design in
  [SlidingWindowKvCache.md](Mila/Specifications/SlidingWindowKvCache.md). Step 5 HF parity is validated,
  so the deferral trigger has fired and this is unblocked. A memory optimization, not a correctness gate
  — the Step 2a sliding-window mask already gives correct numerics against the full `[B,NKV,T,HS]` cache;
  the payoff is long-context (sliding KV 80 GB -> ~0.34 GB at 256K, BF16; persistent-KV growth slope
  336 -> 16 KB/token). Mechanism (Option A, chosen): size each sliding layer's cache to a ring of
  `min(T, window + prefill_chunk - 1)` with modular write/wrap + a slot->absolute-position softmax mask,
  folded onto the existing KV-cache policy axis
  ([Quantization/KvCache/Policy.ixx](Mila/Src/Dnn/Quantization/KvCache/Policy.ixx)) — NOT a new axis and
  NOT conflated with the window number. A standalone `SlidingWindowKvCache` (bounded, uncompressed)
  suffices for Gemma; bounded+FP8 is a later combinatorial concern. **The prefill ring is the hardest
  kernel work in the chassis** — a chunk's queries need >W keys at once; Option A reuses the existing
  cuBLASLt plans (capacity-for-T substitution) rather than a full flash rewrite (Option B, rejected as
  out of scope). prefill+decode share one K/V buffer so it is all-or-nothing. Build the ring against the
  validated full-cache path as the oracle. Reused by the **Ministral** Future Direction.
  - [x] **Phase 0 — policy + dispatch axis.** `SlidingWindowKvCache` struct; `CudaGqaOp<TPrecision, bool kBounded>`
    axis with build-time capacity computation + `window > 0` guard; two `OperationTraits` rows; bounded
    component compiles/constructs (compile-time dispatch test in `GroupedQueryAttention.Cuda.cpp`).
    `GemmaBlock` already forwards `TKvPolicy`. With `kBounded = false` the unbounded path is byte-identical.
  - [x] **Phase 1 — bounded decode.** Ring KV write (`(start_pos+t) % capacity`, identity when unbounded);
    slot->abs decode softmask (`softmax_decode_ring_*`); `cache_capacity_` threaded into alloc/plans/write/softmax
    stride (no-op unbounded). Op-level parity harness `CudaGqaOp.Cuda.cpp` (drives GqaState + cache lifecycle):
    capacity/`window=0`-throw checks + decode-vs-oracle (no-eviction near-exact, past-window ring wrap). Green
    (build + chat + tests) 2026-06-30. Plan simplification vs spec D5: bounded reuses the shared scratch as a
    contiguous `capacity`-prefix (`N=capacity`), no `ldc=T_ctx` decoupling needed.
  - [x] **Phase 2 — bounded prefill.** Slot->abs prefill softmask (`prefill_softmax_ring_*`) using
    `end = position_offset + chunk_len - 1` (cache newest) for the slot mapping and `window_start(abs_t) <= p_j <= abs_t`
    (causal excludes same-chunk-future keys). `prefill_optimized` branches on `kBounded`. Parity tests: single-chunk,
    multi-chunk-across-window (cross-chunk eviction), partial-final-chunk, prefill-then-decode over one shared ring.
    Hand-verified `capacity = window+chunk-1` makes each chunk's needed span exactly the resident range. Green
    (build + tests) 2026-06-30.
  - [x] **Phase 3 — wire Gemma.** `GemmaTransformer` routes `SlidingWindowKvCache` to local layers, hardwires
    `NoKvCompression` on global (full-attention) layers; `GemmaModel` selects it via the `GemmaSlidingKvPolicy`
    flip-point. Coherent 8192-token chat (ring engaged). Footprint asserted closed-form (`StateMemory_MatchesClosedFormAndShrinks`)
    + policy routing pinned compile-time (`KvPolicy_RoutesBoundedRingToLocalLayersOnly`). Green 2026-06-30.
  - Follow-ups (deferred, tracked elsewhere in this file): **FP8 KV** (bounded + FP8 on the 8 global MQA
    layers — the new context wall at 16 KB/token, ~4 GB at 256K — extends the deferred `PerChannelKvFp8<>`
    GQA specialization); **activation-aware prefill-chunk heuristic** (`computeGemmaPrefillChunkSize` is
    attention-scratch-sized, blind to the dominant GeGLU FFN floor — see the VRAM-footprint item).
- [x] **[gate — v0.20 release] Weight-tying optimization (Gemma).** Design in
  [WeightTying.md](Mila/Specifications/WeightTying.md). DONE 2026-07-01 — Gemma 4 12B chat coherent on
  the re-converted (raw-embedding, tied) checkpoint. `lm_head` now shares the token embedding storage,
  reclaiming one `vocab x model_dim` tensor (262144 x 3840 x 2B ~= 2 GB at BF16) in steady-state VRAM.
  The scale-fold conflict was resolved by storing the embedding RAW and moving Gemma's sqrt(hidden_size)
  scale to runtime (`TokenEmbeddingConfig::embedding_scale`, applied in `TokenEmbedding::forward`); the
  shared-ownership path is `wte_` shared_ptr + `Linear::installSharedWeight` + post-load aliasing in
  `GemmaTransformer::loadParameters`. Required a Gemma re-convert (old checkpoints double-scale — no
  graceful-degradation path, by design). This is option (C) of the VRAM footprint item below.
- [ ] **[VRAM, load-time] Tied lm_head double-allocates 2 GB during load — WDDM spill on the
  12 GB card.** UPDATE 2026-07-04: the D4 Design B FP8 table (below) halves this transient on
  quantized builds — both the embedding table and the lm_head self-allocation become FP8
  (~1.0 GB each instead of 2.0 GB BF16), and the BF16 table never lands on device (quantize
  staging reuses the shared scratch). The double-allocation PATTERN remains; the pre-build
  install fix described here is still the structural close.
  Observed 2026-07-03 (user, Task Manager): dedicated VRAM pegs at 12 GB plus a
  WDDM shared-memory blip until the model finishes loading, then falls to ~10.1 GB. Cause:
  `lm_head` self-allocates its vocab x model_dim BF16 weight (262144 x 3840 x 2B ~= 2.01 GB) at
  build, and the tie (`installSharedWeight` with the embedding table) only replaces + frees it at
  the END of `GemmaTransformer::loadParameters` — so the load-time peak carries both copies while
  the FP4 weights and the ~470 MB quantize staging accumulate. Fix: `tie_word_embeddings` is in
  the checkpoint metadata that `fromPretrained` already reads BEFORE build (`configFromMetadata`)
  — thread it into the config and either install the shared table pre-build or skip the lm_head
  weight self-allocation (the pooling Phase 2 install-before-build idiom;
  `Linear::installSharedWeight` needs to accept pre-build install or Linear gains a deferred-weight
  build path). Payoff: flat ~10.1 GB load peak, no WDDM spill, likely faster load. Same fix
  applies to the deferred Llama tying item below when that lands. Deferred follow-up — WeightTying.md §6.
  The architecture-agnostic plumbing already shipped with the Gemma gate (`embedding_scale` defaults to
  identity for Llama). Remaining surface is small: add `tie_word_embeddings_` member + post-load aliasing
  + `getMemoryStats` correction to `LlamaTransformer` (identical to Gemma), and write the flag + skip the
  `lm_head.weight` blob in `Llama/convert_weights.py`. Saves ~789 MB (3B) / ~524 MB (1B) — on models that
  already fit, hence deferred. Validation caveat: parity needs the checkpoint + HF reference + greedy
  oracle; decide whether a contributor owns parity or only the code. Llama 3.1 8B is untied — no change.
- [x] **`parameterCount()` double-counts tied weights (display only).** FIXED 2026-07-01 — overrode
  `GemmaTransformer::parameterCount()` to subtract the tied `lm_head` contribution, mirroring the D7
  `getMemoryStats` correction. (`getMemoryStats` was already corrected by the gate.)
- [ ] **C++ test-checkpoint writer + transformer load-tie test (WeightTying.md §7.3).** The aliasing
  primitive is unit-tested (`Linear::installSharedWeight` identity/forward, `TokenEmbedding` scale), but
  the full `GemmaTransformer::loadParameters` tie round-trip is NOT — `PretrainedModelReader` is
  mmap/file-only with no C++ writer (writer is Python-only, `Tools/Converters/.../common.py`) and the
  transformer's `token_embedding_`/`lm_head_` are private. Add a small reusable C++ helper that writes a
  synthetic checkpoint (header + metadata JSON + tensor index + blobs) to a temp file so a test can load
  a 2-layer tied Gemma and assert shared-pointer identity + no `getMemoryStats`/`parameterCount` double-
  count. Also unblocks the deferred Llama 3.2 tying parity test. Until then the load-tie path is covered
  only by the validated Gemma chat run.
- [x] **VRAM footprint reduction (Gemma 12B FP4 on 12 GB) — beyond the two gates above.
  CLOSED 2026-07-03:** lever (A) shipped as heuristic v2 (`resolvePrefillChunkSize`, complete
  row-cost model — see the pooling item above); lever (C) was the weight-tying gate (DONE); lever
  (D) shipped as the pooling item above (chunk 512 on the 4070, prefill 1.57 s / 2048 tokens).
  Lever (B) (a `GemmaModelConfig` / `--prefill-chunk` override surface) is DROPPED: the
  `kGemmaPrefillChunkOverride` constexpr remains the sweep knob, matching the preferred
  edit-constant-and-rebuild tuning workflow. Historical analysis below. FP4 12B is
  ~9.14 GB resident params + ~5.9 GB State (~15 GB) on the 12 GB dev card -> WDDM paging thrash
  (correct values, slow). The State floor is the **48 per-layer prefill ACTIVATION buffers sized at the
  prefill chunk** (GeGLU-FFN-dominated), NOT context — ctx 4096->512 barely moved it. `computeGemmaPrefillChunkSize`
  ([Gemma.ixx:96](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.ixx)) is activation-blind (it caps on
  GQA attn scratch only) so it over-picks chunk 512. Levers, all internal-impl: (A) make chunk sizing
  activation-aware (budget num_layers x per-token activation bytes) so it auto-drops to 128/64 under a
  real VRAM budget — **full heuristic-v2 design now specified in
  [Gemma4InferenceReview.md section 6](Mila/Specifications/Gemma4InferenceReview.md)** (cost model
  ~11.3 MB/chunk-row, `cudaMemGetInfo` budget, ladder floor 64, formula-vs-`getMemoryStats` pinning
  test; picks ~128 on the 4070 today, 512 after pooling); (B) expose a prefill-chunk / VRAM-budget override on `GemmaModelConfig` +
  `--prefill-chunk`; (C) reclaim the ~2 GB untied lm_head = the **weight-tying gate** above (DONE); (D) **pool
  the 48 per-layer activation buffers** — PROMOTED 2026-07-02 to its own elevated defect item at the
  top of this section (design in Gemma4InferenceReview.md section 7). Both gates have since shipped
  and 12B FP4 now fits the 12 GB card at the chunk-32 operating point; what remains of this item is
  the (A)/(B) chunk-sizing work, which folds into heuristic v2 + the pooling item. See
  [[project_gemma_chat_vram]].
- [ ] **[perf, prefill] Fuse FP4 weight dequant into the prefill GEMM (W4A16) -- ~17% of prefill, and it
  is REDUNDANT.** Nsight capture 2026-07-07 (Gemma 12B FP4, seq-len 35719, ctx 40960, on the 4070 --
  `ProfileModel --phase prefill`, capture in scratch `gemma_prefill_35k.nsys-rep`): the prefill spends
  **9.39 s / 56.9 s (16.6%) in a SEPARATE `dequantize_fp4_to_bf16_kernel`**, then feeds a BF16 cutlass GEMM.
  26,880 instances = each of the ~336 FP4 weight matrices dequantized ~80x (once per prefill chunk) -- the
  weights are re-materialized to BF16 for every chunk instead of dequantized once, or (better) dequantized
  inline in the GEMM tile load. The DECODE path already fuses this (`matvec_decode_bf16_qfp4`); prefill does
  not, and `Quantization.V2` / CLAUDE.md describe "dequantized inline during the W4A16 GEMM tile load via LUT"
  as the intent -- so this is a design/impl gap, not a missing design. Fix: a W4A16 prefill GEMM (dequant in
  prologue/tile load) deletes the separate pass. Pairs with the "Native Blackwell FP4 matmul" item above
  (line ~306). Biggest easy prefill win; directly cuts Claude Code first-token latency. See [[project_prefill_perf_beta]].
  CONFIRMED 2026-07-10 at 8K + 32K (flash A/B session, cuBLASLt path): dequantize_fp4 = 1.08s/6.4s (16.7%)
  at 8K, 4.32s/35.4s (12.2%) at 32K -- clean O(S) scaling (4x for 4x ctx), pure waste.
  CORRECTION 2026-07-11: the existing fused kernel (`CudaW4A16Gemm.Wmma.cu`, behind `kUseFusedFp4Gemm=false`)
  is NAIVE (1 warp/block, one 16x16 tile, ~2.5 TFLOPS ~2% peak) -- that is WHY it is toggled off and the
  2-phase dequant->cuBLASLt path (~3x faster) is default. So this is NOT "wire the existing kernel" (wiring it
  makes prefill slower); it is OPTIMIZE the fused kernel to tensor-core rates. STAGE 1 IMPLEMENTED 2026-07-11
  (rewrote `CudaW4A16Gemm.Wmma.cu`): multi-warp register-accumulator tiling (64x64 block, BK=32, 4 warps 2x2,
  each warp 2x2 register-resident wmma [16x16] accumulators; safe wmma API -- a GEMM has no online-softmax
  rescale so no mma.sync PTX; naive orientations/decode reused). Synchronous loads still (will Long-Scoreboard
  stall like flash 2b); Stage 2 = cp.async double-buffer + swizzle + ldmatrix + bigger tiles (fa-5090 ladder).
  W4A16 (BF16 acts, no numerics risk); W4A8 = later ceiling. VALIDATE: flip `kUseFusedFp4Gemm=true` + rebuild
  -> `Linear.Cuda` `Forward_MatchesReference` (Linear<Cuda,BF16,PerGroupFp4<128>>, 5e-2) + Gemma parity. See
  [[project_w4a16_prefill_gemm]]. RANK: at 32K the O(S^2)
  global-softmax (flash, ~29%) has overtaken this as the #1 prefill cost, but this stays the highest-CERTAINTY
  win (O(S), independent of attention); do BOTH. See [[project_gqa_flash_attention]] for the full 32K attribution.
  RE-BASELINE FIRST 2026-07-12 (decision: Gemma 4 prefill SPEED is now the active goal; measure before more
  W4A16 Stage 2). The "4-5x llama.cpp gap" is STALE -- it predates flash 2c + 5.6 reclaim + chunk-1024, so
  Mila's 40960 prefill is now faster; re-measure before scoping the optimization. TEST = head-to-head Gemma 4
  prefill, Mila vs llama.cpp (LM Studio), on the 4070. RUNBOOK:
  - COLD CACHE both sides -- MIS KV prefix reuse falsely reports ~instant on a 2nd identical request; use
    first-request-after-load only (or disable). ONE MODEL AT A TIME (12 GB can't hold two 12B; load, measure,
    unload, swap).
  - FIXED workload = the ~35719-token Claude-CLI harness prompt; CAPTURE the blob once and REPLAY the same
    bytes to each server (don't let live Claude CLI deliver it twice -- harness/network drift). Mila:
    MILA_CONTEXT_LENGTH=40960 (its real chunk-1024 config).
  - PREFILL METRIC = TTFT / prompt-eval tok/s. LM Studio's server log prints "prompt eval time / tok/s" = the
    clean llama.cpp number.
  - TIER 1 (kernel gap = the optimization target, no wire): `ProfileModel --phase prefill --seq-len 35719
    --context-length 40960 --quantization fp4` vs LM Studio prompt-eval.
  - TIER 2 (felt gap): Claude CLI -> MIS vs Claude CLI -> LM Studio -- includes Python/wire serving overhead,
    which W4A16 does NOT touch (a separate lever if Tier 2 diverges from Tier 1).
  - CEILING CAVEAT (sets the target): llama.cpp Gemma = Q4_K_M + MMQ = INT8 activations (W4A8); Mila = FP4 +
    BF16 activations (W4A16). Int8 tensor cores ~2x BF16 throughput, so a PERFECT Mila W4A16 still sits ~2x
    behind llama.cpp; closing that last 2x needs the W4A8 activation-quant fork (changes Mila's BF16-activation
    numerics). W4A16 prize = "close most of the gap", NOT "match llama.cpp" -- read the result that way.
  - MIS setup ([[project_mis_test_environment]]): WSL client -> MIS on Windows 0.0.0.0:8000, MIS's own cp313
    venv (not the uv-3.11 shadow), MILA_CONTEXT_LENGTH=40960 to fit the harness prompt.
  Then resume W4A16 Stage 2 (cp.async + swizzle + ldmatrix, the fa-5090 ladder) scoped against the measured gap.
  RE-BASELINE RESULTS 2026-07-12 (4070, Gemma 4 12B; llama.cpp via LM Studio Q4_K_M; Mila FP4 W4A16 2-phase,
  kUseFusedFp4Gemm=false; ProfileModel x64-profile). The live `claude -p` harness tokenized to 22496 tokens
  (llama.cpp Gemma tokenizer), NOT the stale 35719 -- that older figure was a larger MIS-config harness, so
  runs were matched at 22496. Tier 1 (kernel; ProfileModel --seq-len 22496 dummy tokens, --quantization fp4):
    - llama.cpp @ 48K ctx: prompt eval 10902.98 ms / 22496 tok = 2063 tok/s (cold; declining 2410->2063 ramp
      confirms a real full prefill, not a cache hit). Q4_K_M/MMQ = W4A8 (int8 activations).
    - Mila @ 48K ctx (49152): min 39208 ms = 574 tok/s  -> 3.6x slower than llama.cpp.
    - Mila @ 24K ctx (24576): min 21797 ms = 1032 tok/s -> 2.0x slower than llama.cpp.
    - Mila @ 40960 ctx, 35719 tok (unmatched ref): min 47188 ms = 757 tok/s.
  KEY FINDING: the stale "4-5x" gap is now 2.0-3.6x, decomposing into TWO independent factors:
    (1) ~1.8x is Mila's VRAM-DRIVEN PREFILL CHUNK, not the kernel. Same 22496 tokens, only the context ceiling
        changed (24K->48K) and throughput HALVED (1032->574). At 48K on the 12 GB card the chunk heuristic
        shrinks the chunk to fit VRAM, multiplying (a) dequant redundancy (2-phase re-dequantizes every FP4
        weight per chunk) and (b) small-M GEMM inefficiency (chunk == the prefill GEMM's M; cuBLASLt loses
        tensor-core utilization at small M). This is a MEMORY/CONFIG lever (bigger card / flash+pooling reclaim
        -> bigger chunk), independent of the kernel.
    (2) residual ~2.0x (Mila best-chunk 1032 vs llama.cpp 2063) is the W4A16-vs-W4A8 tensor-core ceiling, as
        the runbook's CEILING CAVEAT predicted (int8 acts ~2x BF16). A perfect W4A16 still sits ~2x behind;
        closing it needs the W4A8 activation-quant fork, NOT W4A16 Stage 2.
  STAGE 2 SCOPE (revised by this baseline): the fused W4A16 GEMM's real prize is killing per-chunk dequant
  redundancy so chunk size stops costing dequant work -- it attacks factor (1) at high context where the
  2-phase path is worst; it will NOT close factor (2). Worth resuming, scoped as "recover the high-context
  chunk penalty", with W4A8 tracked separately as the final-2x lever. PENDING: Tier 2 (felt) = same `claude -p`
  -> MIS vs -> LM Studio, to add Python/wire overhead AND confirm Mila's tokenizer encodes the same text to
  ~22496 (a large divergence from llama.cpp's count would itself be a tokenizer finding). Logs in scratchpad
  tier1_mila_prefill*.log. See [[project_w4a16_prefill_gemm]].
  STAGE 2 IMPLEMENTED 2026-07-12 (rewrote CudaW4A16Gemm.Wmma.cu, same launcher signature -> no cuh/Dispatch/Op
  change). cp.async double-buffered software pipeline over Stage 1's exact 64x64/4-warp/BK=32 geometry: while
  the current K-tile's MMAs run, the next tile's activations (BF16, wmma-ready) + packed FP4 weights stream
  global->smem via __pipeline_memcpy_async; the FP4 decode stays a smem->smem dequant pass into a single BF16
  scratch (cp.async can't decode inline). Two-stage buffers: sA[2] + rawW[2] double-buffered, bf16W single.
  Prologue-load + __pipeline_commit/wait_prior(1) hides the Long-Scoreboard global-load latency that pinned
  Stage 1 (occupancy 33%, Compute 37%). Key simplifier: FP4 guarantees K%group_size==0 (group 64/128) so
  K%64==0 -> BK divides K with no tail and all cp.async addrs are 16-byte aligned; bounds reduce to M/N row
  edges (OOB rows zero-filled in smem). smem 18 KB (no opt-in).
  VALIDATED + PROFILED 2026-07-12 (build green, Forward_MatchesReference + Gemma parity green, chat coherent =
  Stage 2 CORRECT end-to-end). ProfileModel prefill A/B, 22496 tok:
    - @48K: Stage 2 44178 ms (509 tok/s) vs 2-phase 39208 ms (574 tok/s) = 1.13x SLOWER.
    - @24K: Stage 2 42660 ms (527 tok/s) vs 2-phase 21797 ms (1032 tok/s) = 1.96x SLOWER.
  cp.async WORKED (Stage 1 was 4-7x regression -> Stage 2 is 1.1-2x): the Long-Scoreboard stall is largely
  gone. KEY FINDING = the fused kernel is now nearly CHUNK-INDEPENDENT (~510-527 tok/s flat vs 2-phase's
  574@48K -> 1032@24K), which VALIDATES the Stage 2 thesis (no per-chunk dequant redundancy, no small-M
  penalty) -- BUT the flat level sits BELOW cuBLASLt at every measured chunk. It is now COMPUTE-bound below
  cuBLASLt's BF16 GEMM, not latency-bound. So Stage 2 does NOT beat 2-phase; kUseFusedFp4Gemm reverted to
  FALSE (don't ship regression; Stage 2 kept in-tree as the cp.async foundation for the ladder). STRATEGIC
  PIVOT this surfaces: 2-phase at a BIG chunk (1032 tok/s @24K) already beats the fused kernel, so the
  highest-CERTAINTY attack on the re-baseline's 1.8x chunk penalty is to UN-SHRINK the KV-aware chunk budget at
  48K. NOTE: pooling + flash-5.6 reclaim are ALREADY SPENT (both baked into resolvePrefillChunkSize's row_cost:
  pooling -> workspace_bytes, flash -> attention_bytes uses window-bounded prefillScoreWidth). What throttles
  the chunk at 48K is the KV-AWARE BUDGET (Gemma.ixx:820): budget = 1536 MiB fixed - global_KV(T_ctx); global
  KV is BF16 (8 layers x 1 KV head x hd512), = 384 MiB @24K (budget 1152 -> big chunk -> 1032 tok/s) vs 768 MiB
  @48K (budget 768 -> small chunk -> 574). The one un-spent lever = WIRE FP8/INT8 GLOBAL-KV: the config's
  FP8-KV label is INERT (Gemma.ixx:772, OperationTraits has no PerChannelKvFp8 spec, op stores TPrecision=BF16).
  Halving 768->384 MiB restores 48K's budget to 1152 (== 24K's) -> 48K picks the same big chunk -> ~1032 tok/s,
  closing ~all of the 1.8x. It is a KV numerics change (parity gate) + a real OperationTraits KV-policy spec
  ("bounded+FP8 on globals" follow-up), and mainly helps the 12 GB 4070 (16 GB card likely already big-chunks
  @48K). This is NOT more pooling. The fused kernel is NOT the lever for factor (1). The fused-kernel ladder (XOR swizzle -> ldmatrix -> bigger tiles 128x64/128x128; fa-5090 proved 94%
  SOL is reachable) only pays off if it crosses cuBLASLt AND stays chunk-independent (flat >1032 would beat
  2-phase at ALL contexts). DECISION PENDING user steer: (A) continue fused ladder, (B) pivot to memory-reclaim
  big-chunk, (C) W4A8 for the residual 2x. ncu on fp4a16_wmma_gemm_kernel still useful to confirm the new
  compute-bound bottleneck (bank conflicts / smem-instruction) before picking (A).
  FACTOR 1 RESOLVED 2026-07-12 (chunk override experiment -- SUPERSEDES the FP8-KV framing above). Forced
  kGemmaPrefillChunkOverride=1024 (2-phase, kUseFusedFp4Gemm=false), rebuilt, reprofiled 22496 tok:
    - @48K forced-1024: 21350 ms = 1054 tok/s (vs heuristic 39208 ms / 574) = 1.84x FASTER, and it FIT
      (used 11531 MiB, 750 MiB free -- no OOM). Matches 24K exactly.
    - @24K forced-1024: 21289 ms (control, ~= heuristic 21797 -> 24K already used ~1024).
  So the entire factor-(1) 1.8x is a FREE HEURISTIC FIX, NOT a fundamental chunk/KV limit and NOT needing
  FP8-KV. The chunk heuristic throttles 48K to ~256 even though 1024 fits with 750 MiB to spare: the fixed
  1536 MiB conservative cap (Gemma.ixx:115, minus full 768 MiB KV = 768 MiB modeled budget) is FAR more
  pessimistic than real headroom. FIX = make resolvePrefillChunkSize budget against LIVE cudaMemGetInfo free
  VRAM (the follow-up already noted at Gemma.ixx:111-113) minus a safety margin, so it picks the largest chunk
  that ACTUALLY fits (1024 @48K, self-limiting toward 512 @64K where a fixed 1024 would OOM). No numerics
  change, no new kernel. REVISED SCOREBOARD (4070, 22496 tok, 48K): llama.cpp 2063 / Mila-heuristic 574 (3.6x)
  / Mila-chunk-fixed 1054 (2.0x, FREE) / +W4A8 ~2000 (~parity, structural). PLAN: (1) land the live-VRAM chunk
  budget = free 1.8x -> 2.0x behind; (2) W4A8 for the structural 2x. FP8-KV demoted (helps 64K memory, not
  this speed gap). Fused ladder + Stage 2 stay parked (kUseFusedFp4Gemm=false; big-chunk 2-phase now clearly
  the better path). kGemmaPrefillChunkOverride reverted to 0.
  CORRECTION 2026-07-12 (instrumented + re-measured CURRENT binary -- OVERTURNS "FACTOR 1 RESOLVED" above;
  there is NO factor 1). A [chunk-diag] print in resolvePrefillChunkSize showed BOTH 24K and 48K pick
  chunk=1024 (row_cost=662464=0.63 MiB, scoreWidth=1023 -> flash reclaim IS in the cost model). Re-ran the
  CURRENT heuristic binary (override=0) at 22496 tok: @24K 21312 ms / @48K 21306 ms = 1055/1056 tok/s, IDENTICAL.
  So the shipping code already runs 48K at chunk 1024 = 1056 tok/s. The slow 39208 ms / 574 tok/s @48K "baseline"
  was a STALE BINARY (the first ProfileModel @08:57 predated flash-5.6's cost-model reclaim -> it alone picked a
  small chunk). The whole "1.8x chunk penalty / free heuristic fix / live-VRAM budget / FP8-KV" thread is VOID --
  flash-5.6 already fixed the chunk at build. TRUE SCOREBOARD (4070, 22496 tok, 48K): llama.cpp 2063 / Mila
  SHIPPING 1056 (1.95x behind) / +W4A8 ~2000 (~parity). The ONLY remaining prefill lever vs llama.cpp is W4A8
  (int8 activations = MMQ's 2x tensor-core rate); factor 1 does not exist. Chunk-heuristic diag + <iostream>
  removed from Gemma.ixx (tree clean). LESSON: re-measure the CURRENT binary before trusting a baseline across
  rebuilds -- a stale ProfileModel drove ~an entire session of wrong analysis.
- [ ] **[perf, prefill] Gemma prefill competitiveness -- at 1.136x behind llama.cpp (1817 tok/s @48K) as of
  +104; remaining levers DEFERRED ("scrape the bones later", user call).** Success criterion ("close, not
  miles behind") MET: this arc drove 1.95x -> 1.136x (39208 -> 12382 ms, 22496 tok @48K; llama.cpp 10903 ms
  / 2063 tok/s). Shipped: +103 (W4A8-FP8 + ring flash, 1.17x) then +104 (local FA-2 kernel, 1.17x ->
  1.136x). Crossing UNDER the line is a stacked campaign, not one fix, and the biggest bucket is at its Ada
  floor -- deferred by design. Gap map (nsys, +103 base, GPU-bound): global flash 4.51s/36% (Ada floor) |
  FP8 GEMMs 4.11s/33% (DONE) | local flash ~0.9s (was 1.12s ring, now FA-2 kernel) | FP4->FP8 upcast
  1.10s/8.8% | epilogue+quant 0.65s | rope/geglu/rmsnorm/misc ~0.94s.
  DEFERRED LEVERS (ranked; full scope + arithmetic in [[project_w4a16_prefill_gemm]] +
  [[project_gqa_flash_attention]]):
  - **Lever A -- FP4->FP8 upcast HOIST (~-1.0s, biggest clean win).** The 1.10s upcast bucket is REDUNDANCY:
    192 linears x 22 chunks re-upcast ALL weights every chunk (356 GB/run vs 16.2 GB once). Kernel is
    already ~325 GB/s (~65% peak) -- not slowness, repetition. FIX = layer-outer/chunk-inner prefill
    restructure in Gemma.ixx (embed full-seq into ONE in-place residual buffer; per-layer: upcast once,
    run all chunks, stitch outputs back) + a per-layer-pass upcast-reuse hook. REQUIRES core API change
    (new IDecoderLayer beginPrefillPass/endPrefillPass -> GemmaBlock -> CudaLinearOp; ~224 MB per-layer FP8
    staging pool). MEMORY: ~+330 MB resident (full-seq residual +165, FP8 pool +165) vs ~623 MiB free =
    ~290 MiB margin, FITS (chunk-inner keeps the 691 MB full-seq FFN intermediate OFF the table -- the trap
    to avoid). GATE: GemmaModelParity token-for-token (same math reordered) -> chat -> scoreboard vs 12761.
    Projected ~1.08x (does NOT alone cross the line). BLOCKED on user sign-off for the core-library hook.
  - **[x] Local sliding-layer row-split FA-2 kernel -- DONE +104 (2026-07-14): 1.17x -> 1.136x, local
    bucket ~tapped.** New `Gqa.Flash.Fa2.cu` owns `cuda_gqa_flash_prefill_ring_bf16` (the HS=256 local
    layers) as a true FA-2 kernel: each warp owns a full 16-row query tile with register-resident O and P,
    one block-barrier per key tile (vs the HS-split's three). 1.47 ms/launch (HS-split ring) -> 1.00
    ms/launch (-32%). ncu-confirmed levers: register-P + warp independence beat the HS-split's latency
    bound; occupancy raised via kFa2WarpsPerBlock=6 (1.0 -> 1.5 warps/sched, the barrier structure lets
    more warps/block lift warps/SM directly). FALSIFIED en route (do NOT retry): a two-pass HS split to
    kill the 4 MB O spill REGRESSED (+276 ms) -- at 1 warp/sched the recompute cost more than the
    L1-cached spill; occupancy, not the spill, was the lever. Full arc in [[project_gqa_flash_attention]].
  - **Global FP8-K/V-in-smem two-block attempt (BORDERLINE, big-upside/high-risk).** Global KV already FP8;
    keep K/V FP8 in smem (halve the bytes) + register upcast. Real carve: 93 KB -> ~60 KB, STILL ~10 KB
    over the ~50 KB two-block threshold -> needs ALSO single-buffering (kills cp.async overlap) or FP8-Q
    (numerics). 2-3 stacked risky changes for an uncertain 2-block payoff. Prove smem crosses 50 KB before
    writing code.
  - **Small fusions ~-0.3..0.5s** (rope into split/kv-write, rmsnorm+residual, epilogue folds); **global
    flash ILP ~-0.3..0.7s** (fragment/register double-buffer + mma pipelining vs the 36% exec-dependency
    stall -- refinement, not rewrite; global kernel is at its Ada occupancy floor); **local ring L2 rework
    ~-0.3s**. Any A + fusions + one attention nibble lands at/under the line.
- [x] **[perf, prefill] FP8-activation prefill GEMM (W4A8-FP8) -- SHIPPED ON at +103 (2026-07-13):
  1763 tok/s @48K, 1.17x behind llama.cpp.** (History below; resolution at the "VALIDATED -- ALL GATES
  GREEN" line. The per-token-scale + stale-sB fixes closed the +98/+99 incoherence.) Original framing:
  GEMM is correct + 1.24x faster @48K, BUT per-tensor FP8-activation scales produced
  INCOHERENT Gemma generation (per-layer Forward_MatchesReference 5e-2 passed; shipped ON in +98, reverted
  OFF in +99 after a clean build generated garbage -- the per-layer oracle does NOT gate generation). OPEN
  WORK to re-enable: (1) per-token activation absmax scale (cuBLASLt vector B_SCALE) and/or per-channel
  weight scale (vector A_SCALE), spec 5.1/5.2; (2) RE-GATE on Gemma token-for-token parity vs BF16 + a
  coherent chat, NOT the per-layer oracle. Then flip kUseFp8ActivationPrefill back on only if both green.
  SPEC: `Mila/Specifications/Fp8ActivationPrefill.md`.
  Run the batched prefill linear GEMMs on FP8 tensor cores (~2x BF16) instead of BF16, entirely inside
  CudaLinearOp (internal op optimization; BF16 in/out contract preserved; same category as the existing FP4
  weight quant, gated by the same Forward_MatchesReference 5e-2 + Gemma parity oracle). HARDWARE-VERIFIED:
  cuBLASLt FP8xFP8 = ~2.0x BF16 on the 4070 (microbench scratchpad/fp8_gemm_bench.cu, 1.90-2.11x across prefill
  shapes; regular FP32 accum already 2x -> no fast-accum). PIVOT CONSTRAINT: Gemma 4 12B weights STAY FP4 in
  VRAM (12B/12GB fit); only a TRANSIENT FP4->FP8 E4M3 upcast feeds the GEMM (half the bytes of today's
  FP4->BF16 staging; FP4->FP8 ~lossless). Activations: BF16->FP8 E4M3 dynamic scale (the one lossy step ->
  the numerics gate). Decode (FP4 matvec) untouched. WHY FP8 not int8: same 2x, better numerics (float),
  cuBLASLt-native (no MMQ hand-roll). EXPECTED: prefill 1.95x-behind -> ~1.1-1.3x (competitive). KEY RISK =
  weight FP8 scale granularity (per-tensor may lose FP4 per-group precision -> escalate to per-channel/vector
  scaling). Implement per spec S8 checklist behind kUseFp8ActivationPrefill (OFF until it beats BF16 + passes
  parity). Current --quantization fp8 is W8A16 (dequant-to-BF16, no 2x -- measured fp8=fp4 on llama-3B), so
  this is net-new wiring. See [[project_w4a16_prefill_gemm]].
  IMPLEMENTED 2026-07-12 behind `kUseFp8ActivationPrefill` (CudaLinearOp.ixx, default FALSE). New:
  `cuda_quantize_bf16_to_fp8` (act BF16->FP8 + dynamic per-tensor absmax scale; CudaFp8Prefill),
  `cuda_fp4_dequantize_to_fp8` + `cuda_compute_fp8_weight_scale` (sB=(6/448)*max(FP4 group scales), derived
  from stored scales -- no weight reread; CudaW4A16Gemm), `build_fp8_prefill_plan`/`execute_fp8_prefill_plan`
  (TN col-major, BOTH operands E4M3, A=weight/A_SCALE=sB, B=act/B_SCALE=sA, FP32 accum, NO fast-accum, BF16
  out; CublasLtLinearPlan). Op owns 2 device scalars (dynamic sA rewritten per forward, static sB @build) +
  a conditional FP8 plan cache (std::monostate when off -> zero extra instantiation for other policies).
  Shared scratch: weight-FP8 (16B-aligned) | activation-FP8. Decode (outer_size==1) untouched.
  VALIDATED (toggle ON): Forward_MatchesReference 5e-2 + Gemma parity + chat coherent all GREEN, and the
  PER-TENSOR weight scale SUFFICED -- the flagged per-channel-escalation risk did NOT bite.
  PROFILED (4070, Gemma 4 12B, 22496 tok @48K, ProfileModel prefill fp4): FP8 17210 ms min/5 (mean 17235) =
  **1307 tok/s** vs 2-phase OFF 21350 ms / 1056 = **1.24x prefill speedup**; 1.95x -> **1.58x behind
  llama.cpp** (2063). Fits VRAM (used 11549 MiB, free 733, chunk 1024 held).
  IMPORTANT CONTEXT: flash prefill was ALREADY ON in BOTH runs. It is build-time context-gated
  (useFlashPrefillForContext = BF16 && T_ctx >= kGemmaFlashPrefillMinContext(16384), Gemma.ixx:716;
  wired on global blocks via setUseFlashPrefill, Gemma.ixx:498); both runs used --context-length 49152.
  So baseline 1056 = flash-on + FP8-off, and 1307 = flash-on + FP8-on: 1307 @48K IS the combined flash+FP8
  target config, and the 1.24x is the PURE FP8-GEMM delta with flash held on in both.
  (An earlier draft of this entry wrongly claimed the ~60% non-GEMM remainder was un-accelerated O(S^2)
  BF16 attention and that flash was the next lever -- WRONG: flash was already applied to the global layers,
  and the local layers are window-bounded (1024) = already cheap. The 40/60 split was a hand-wave from one
  GEMM-delta subtraction, not a measured breakdown.)
  NSYS KERNEL BREAKDOWN 2026-07-12 (combined flash+FP8 @48K, 22496 tok; capture = ~all 17.2s wall, GPU ~fully
  busy, no launch-gap tax; report scratchpad/gemma_flash_fp8_48k.nsys-rep):
    - Global flash attn (gqa_flash_prefill_wmma_bf16): 41.9% (176 inst = 8 global x 22 chunks) -- #1 cost.
    - Local sliding attn (prefill_softmax_ring_bf16 15.4% + cutlass BF16 QK/AV GEMMs 4.8%): ~20.2% (softmax
      880 inst = 40 local x 22 chunks; the 40 sliding layers are NOT flashed -- still cuBLASLt + ring softmax).
    - FP8 linear GEMMs (sm89_xmma_e4m3 tn): 23.9% -- already fast (the landed FP8 win).
    - FP4->FP8 weight upcast: 6.4%; RoPE 2.2%; activation->FP8 quant 1.8% (negligible); GeGLU/RmsNorm/misc ~3.6%.
  CONCLUSION: attention is ~62% of prefill (42 global-flash + 20 local); linear GEMMs only ~24% (already FP8-
  fast). llama.cpp's remaining 1.58x edge @48K is ATTENTION, NOT the matmul; FP8 quant overhead is negligible.
  TWO LEVERS (ranked by measured cost): (1) optimize the flash WMMA kernel (41.9%; Stage 2c compute-bound below
  cuBLASLt -> fa-5090 ladder swizzle/ldmatrix/bigger tiles) = biggest single win; (2) extend fused flash to the
  40 LOCAL sliding-window layers (~20%; kill the ring-softmax + separate QK/AV BF16 GEMMs). Both are
  [[project_gqa_flash_attention]] work; the matmul side is done (low ceiling for more matmul effort).
  W4A8-FP8 clears its bar (beats BF16, parity green) and stacks on flash for the combined 1307 @48K; it
  stays in-tree behind the toggle (returns to FALSE for the shipped default until a ship-on decision, same
  discipline as kUseFusedFp4Gemm). Reusable microbench: scratchpad/fp8_gemm_bench.cu.
  UPDATE 2026-07-13 -- PER-TOKEN ACTIVATION SCALES IMPLEMENTED (the numerics fix; pending VS2026 build +
  gates). Design: Ada cuBLASLt accepts only per-tensor scale pointers (outer-vector scale modes are
  Blackwell-era), so the per-token scales are NOT bound to the GEMM descriptor; instead the exact
  factorization Y[m,n] = sA[m] * (sB * sum_k X8[m,k]*W8[n,k]) is applied post-GEMM. Changes:
  (1) CudaFp8Prefill: `cuda_quantize_bf16_to_fp8` (per-tensor, 3 launches + memset + global atomics)
  REPLACED by `cuda_quantize_bf16_to_fp8_per_token` (ONE kernel, block per row: shared-mem row absmax ->
  sA[row]=absmax/448 -> quantize in the same launch); new `cuda_fp8_apply_per_token_scales` epilogue
  (block per row, output[t,:] = output[t,:]*sA[t] + bias, folds the old cuda_add_bias pass -- for
  bias-less Gemma this is the one new kernel per linear, ~M*N BF16 read+write, small vs the GEMM win).
  (2) CudaLinearOp: scratch carve now weight-FP8 | activation-FP8 | per-token scales (each 16B-aligned);
  B_SCALE binds a persistent constant-1.0f scalar (`activation_fp8_unit_scale_`, set once at build) so
  the plan descriptor stays byte-identical to the proven-fast +98 config for the heuristic; toggle
  flipped ON for the validation build (ship-default decision = after gates). (3) NEW ORACLE
  `LinearCudaQuantizedTests.Forward_Fp4PrefillMatchesDecodeAcrossTokenMagnitudes` (Linear.Cuda.cpp):
  prefill FP8-GEMM leg vs decode FP4-matvec leg (independent proven path, same loaded weights), 16 rows
  spanning 1e-8..1e+7 magnitudes, per-row tolerance 5e-2 * row absmax -- this fixture FAILS under the
  +98 per-tensor scheme (one outlier row crushes all other rows' FP8 resolution) and passes per-token;
  it is the missing gate the +98 incident showed (no FP4 forward-numerics oracle existed at Linear
  level). Double BF16 rounding (GEMM out, then epilogue) is ~2^-9 relative twice = noise vs FP8 quant
  error. GATES (in order, none skippable): new Linear oracle -> GemmaModelParity token-for-token ->
  coherent chat at default ctx -> ProfileModel scoreboard @48K vs 16395 ms (est ~1650-1700 tok/s on the
  +101 base; old 1307 was on the 2c kernel base).
  UPDATE 2026-07-13 (second pass) -- REAL ROOT CAUSE FOUND: STALE sB, not activation-scale granularity.
  First per-token build: Linear oracle GREEN but GemmaModelParity RED + chat incoherent. Forensics: the
  static FP8 weight scale sB was computed in buildCublasLtPlans() (build() time) by reading the FP4
  group-scales DEVICE buffer -- which quantize() only fills LATER, during loadParameter() (Linear.ixx
  allocates at build, fills at load). sB is therefore derived from UNINITIALIZED memory. Because sB
  cancels algebraically in the GEMM (A_SCALE = sB, weights staged as W/sB), the bug is LUCK-DEPENDENT:
  junk in a benign band (~1e-4..2 for these weights) generates CORRECTLY; zeroed pages give
  sB = 1e-12*6/448 ~ 1.3e-14 -> every FP4 weight saturates to +-448 in E4M3 -> garbage forward. This
  explains the ENTIRE +98/+99 flip-flop (validated coherent 07-12, incoherent clean build 07-13 -- the
  per-tensor activation scale was likely INNOCENT) and the oracle-green/Gemma-red split (test process =
  recycled allocator junk = benign sB; fresh model process = zeroed pages = saturation). FIX: sB now
  computed at the tail of quantize() (same stream as the scale upload, so the reduction reads the fresh
  values); buildCublasLtPlans() only allocates/binds the scalar pointers via new idempotent
  ensureFp8ScaleScalarsAllocated() (either order of build/load works). LESSON for the gate ladder:
  an uninitialized-memory read is deterministically catchable with compute-sanitizer
  --tool initcheck on the Linear tests -- add that to the FP8-path gates alongside memcheck (parity
  oracles CANNOT catch luck-dependent bugs reliably).
  VALIDATED 2026-07-13 -- ALL GATES GREEN, SHIP-ON BAR CLEARED. Build green; Linear oracle green;
  GemmaModelParity token-for-token GREEN; chat coherent (user). SCOREBOARD (ProfileModel --model gemma
  --phase prefill --seq-len 22496 --context-length 49152 fp4; clean 1162 MiB baseline verified, orphan
  cleaned): **12761 ms min/5 (mean 12791) = 1763 tok/s @48K** on the +101 ring-flash base -- vs 16395 ms
  / 1372 tok/s FP8-off = **1.285x GEMM-path speedup, -22.2% wall**, beating the ~1650-1700 estimate.
  llama.cpp gap: 10903 ms / 2063 tok/s -> **1.17x behind** (was 1.95x at the start of this thread).
  VRAM @48K: load peak 11658 MiB (min free 623), after-run 11431 -- fits with margin, chunk 1024 held.
  Toggle kUseFp8ActivationPrefill stays TRUE in tree: the kUseFusedFp4Gemm-style bar (beats BF16 +
  parity + coherent chat) is cleared, so ON is now the shipped default (user commit decision).
  Remaining prefill gap map (~1.9 s): flash ladder headroom (~1.5 s, global kernel 42% compute vs
  fa-5090's 94% SOL), local ring L2-bound rework, FP4->FP8 upcast transient.
- [ ] **[perf, prefill] Flash-attention-style global prefill kernel -- 36% of prefill (the single largest
  cost).** STATUS 2026-07-10: Iteration 1 (naive scalar, no shared-memory K/V tiling) landed behind the
  runtime `use_flash_prefill_` toggle (default OFF). It is CORRECT -- token-for-token model parity green,
  op-level oracle `CudaGqaFlashPrefillParity.FlashMatchesCublasLt_GemmaGlobalConfig` green at the Gemma
  global config (HS=512, NKV=1, window=0) -- but profiled a REGRESSION: `gqa_flash_prefill_bf16_kernel` =
  74% of prefill GPU time, ~100x memory-bound (every query-row warp re-streams all of K/V from global).
  So Iteration 1 is the correctness foundation only; the actual win is the TILED kernel
  (`GqaFlashAttention.md` 5.2: block owns Br query rows, streams Bc key tiles into shared memory, K/V read
  once per tile not per row), then WMMA (5.3) and scratch reclaim (5.6).
  UPDATE 2026-07-10: Iteration 2 (5.2 tiled kernel) IMPLEMENTED in `Gqa.Flash.Bf16.cu` (Br=16 x Bc=32
  smem tiling, K-then-V two-pass, FA-2 one-rescale-per-tile, causal tile-skip), correctness gates GREEN,
  then PROFILED (flash-on vs cuBLASLt, 8192 prefill, 4070). RESULT: 5.2 is a NON-FIX. Flash 8192 prefill =
  19068 ms wall vs cuBLASLt 6465 ms (still ~3x regression); gqa_flash kernel = 13.1 s (69.7%), 103 ms/inst
  vs Iter1 127 ms -- tiling cut global traffic ~16x but bought only ~1.2x. Nsight Compute root cause: NOT
  global-DRAM-bound (0 spilling, 65% occupancy, L2 hit 99.7%, only 10 GB/s) -- the scalar warp-per-row
  kernel is LSU/shared-memory-INSTRUCTION bound (Mem Busy 87%, Compute 44%): one shared load per FMA. The
  whole scalar family (Iter1 register-stream + Iter2 smem-tile) hits the same 1-FLOP/load ceiling; tiling
  moved the loads (L2 -> shared) without cutting their count. `use_flash_prefill_` STAYS FALSE. The actual
  fix is TENSOR CORES (5.3 WMMA): MMA does hundreds of FLOPs/load, the only escape from the ceiling. Iter2
  kept in-tree (parity-green) as the smem-tiling SCAFFOLD 5.3 builds on (same block/tile/two-pass/online-
  softmax, scalar loops -> MMAs). Spec §5.2.1 records the full attribution; §5.3 is now mandatory-next.
  UPDATE 2026-07-10: Iteration 3 / WMMA Stage 1 (single-warp, O-in-smem, safe `wmma` ops only) landed in
  `Gqa.Flash.Wmma.cu` and went GREEN on `CudaGqaFlashPrefillParity` -- tensor cores produce correct
  attention; the hard correctness questions (WMMA GEMM orientations, cross-tile online softmax, causal mask)
  are answered. Slow by design (single warp), not the design point.
  UPDATE 2026-07-11: WMMA Stage 2a IMPLEMENTED (rewrote `Gqa.Flash.Wmma.cu`, same launcher signature so no
  cuh/Dispatch/Op change) -- `W`-warp HS-split (warp `w` owns HSt=HS/W output columns; W=min(8,HS/16) pow2
  keeping HSt%16==0 -> HS=512 gives W=8/HSt=64, 256 threads), split-K QK + smem `S` reduction, two-pass K/V,
  per-warp `O_w` slice kept in smem + rescaled by per-row smem indexing (no `mma.sync` PTX -- that is Stage
  2b). Disjoint HS slices -> PV/rescale need no cross-warp sync; ~6 `__syncthreads`/tile vs Stage 1's ~64.
  smem 81 KB at HS=512 (under ~99 KB Ada opt-in). PENDING VS2026 build + `CudaGqaFlashPrefillParity`; then
  profile vs cuBLASLt (2a may already win, deferring 2b). See spec §5.3.1.
  PROFILED 2026-07-11 (8192 prefill, Gemma 12B FP4, 4070): oracle GREEN; a real step but NOT yet a win.
  Wall = 11702 ms (vs cuBLASLt 6465, Iter2 scalar 19068). `gqa_flash_prefill_wmma_bf16_kernel` = 6.08 s
  (52%), 47.5 ms/inst avg = 2.2x faster than the scalar Iter2 (103 ms/inst) with tensor cores confirmed
  engaged, but still ~3.5x heavier than the cuBLASLt global attention it replaces (~1.7 s). ncu (heavy
  81 ms instance): occupancy 16.67% SMEM-LIMITED to 1 block/SM (83.65 KB dyn smem), Compute 22% / Mem 16%
  / DRAM 0.9% / 0 spills = LATENCY/STALL bound (NOT the scalar family's LSU ceiling). ROOT CAUSE = the
  32 KB O[Br x HS] FP32 accumulator IN SMEM: caps occupancy AND forces the PV store->sync->smem-add->sync
  serial chain. FIX = Stage 2b (move O_w into `mma.sync.m16n8k16` accumulator REGISTERS + per-row alpha on
  those regs) -- frees ~32 KB smem + kills the chain; ncu vindicates 2b as the lever. `use_flash_prefill_`
  stays FALSE. NEXT = implement Stage 2b (the raw-PTX squeeze).
  STAGE 2b IMPLEMENTED 2026-07-11 (rewrote `Gqa.Flash.Wmma.cu`, same launcher signature). Risk-scoped:
  ONLY PV is raw `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` -- QK keeps 2a's proven `wmma`
  split-K (its S accumulator is transient -> smem for softmax anyway). Per-warp `O_w[Br x HSt]` is register-
  resident (`float o_acc[HSt/8][4]`/thread, 32 regs at HS=512), A(P)/B(V) loaded manually from smem per the
  documented m16n8k16 fragment layout, per-row alpha applied as two scalars/thread (c0,c1->row g; c2,c3->row
  g+8). Drops O + PV scratch from smem: 81 KB -> 41 KB, should lift occupancy 1->2 blocks/SM. PENDING VS2026
  build + `CudaGqaFlashPrefillParity`, then re-profile (watch occupancy + Compute-throughput %, spilling==0).
  See spec §5.3.1.
  PROFILED 2026-07-11 (8192, 4070): oracle green; worked as designed. Per-instance 47.5 -> 26.9 ms; wall
  11702 -> 8820 ms (cuBLASLt 6465, now 1.36x off); flash kernel 3.44 s (37.8%, was 6.08/52%). ncu: Local
  Memory Spilling = 0 (o_acc register-resident, confirmed), occupancy 16.7 -> 33% (regs 97/thd + smem 42.7
  KB co-limit to 2 blocks/SM), Compute (SM) still ~30% = STILL latency-bound because the pipeline is FULLY
  SYNCHRONOUS (K/V global->smem loads block the MMAs). Kernel now ~2x cuBLASLt attention (was 3.5x). NEXT =
  Stage 2c: cp.async double-buffered K/V loads (overlap load with MMA -- the core FA2 software pipeline,
  biggest remaining lever); cheaper first knobs = W=16 (occupancy up / O-regs down) + convert QK to mma.sync
  (drop the split-K smem round-trip + 2 barriers). This is the llama.cpp fattn-mma playbook (no CUTLASS).
  UPDATE 2026-07-11: FA re-justified as a MEMORY / context-scaling play (user goal = 64K on 12 GB), not
  speed (~18% ceiling). 5.6 MEMORY RECLAIM IMPLEMENTED (Gemma, partial): with global attention on flash the
  shared `preatt`/`att` buffer shrinks from `T_ctx` to the sliding layers' window-bounded width
  (`min(T_ctx, window+chunk-1)`), reclaiming ~1 GB at 64K. Single source of truth
  `GemmaTransformer::useFlashPrefillForContext(T_ctx)` (`kGemmaFlashPrefillMinContext` default 16384) couples
  the op toggle (new `setUseFlashPrefill` passthrough GemmaBlock->GroupedQueryAttention->op) to the buffer
  width; `CudaGqaOp::use_flash_prefill_` default restored to false (safe). Flash now runs only at
  ctx >= threshold (below = cuBLASLt, faster + fits). PENDING VS2026 build + validate 64K fits. Stage 2c
  (flash speed) de-prioritized. FOLLOW-UP: LlamaTransformer (all-global) can reclaim `preatt`/`att` entirely
  with the same wiring. Next active work = fast fused W4A16 prefill GEMM. See spec 5.6.
  UPDATE 2026-07-12 (CATCH-UP -- the following all landed in commit +97 but were never written back
  here; the omission cost a re-derivation the next session): STAGE 2c IMPLEMENTED + GREEN + PROFILED =
  THE WIN. cp.async double-buffered K/V software pipeline (`__pipeline_memcpy_async`/commit/wait_prior
  prefetches the next tile's K+V while the current tile's QK+softmax+PV run; math byte-identical to 2b
  so the oracle stays exact) removed the #1 stall. Stall-reason ncu (2b, 8192) had pinned it: Long
  Scoreboard 3.97 cyc/inst (~35%, synchronous global K/V loads) = #1; Short Scoreboard / bank conflicts
  (536M) ~8.6% = #2. cp.async targets #1. Arc @32K: flash 2b ~98000 ms (2x regression) -> 2c wins.
  CLEAN 32K A/B 2026-07-12 (threshold-flip rebuild, same binary both arms, min of 5 measured runs):
  flash 2c = 35624 ms vs cuBLASLt (`kGemmaFlashPrefillMinContext=0`) = 39265 ms -> flash WINS ~9.3%
  end-to-end at 32K (closes the "exact matched cuBLASLt @32K owed" item). 16K = ~parity (pre-cliff).
  CAVEAT on the 9.3%: prefill chunk is coupled to the flash decision via `prefillScoreWidth`, so arm A
  runs chunk 1024 while arm B (full-width score buffer) falls to ~512 -- the margin is the honest
  end-to-end CONFIG delta (and the chunk-1024 win is itself flash-ENABLED: cuBLASLt's full-width buffer
  can't fit chunk 1024), not a pure-kernel delta.
  5.6 RECLAIM VALIDATED -> 64K FITS on a 12 GB 4070 (the memory/context-scaling payoff = the real
  justification, not the speed). Fit-probes (short seq, full context) confirmed the KV-aware chunk
  boundary: 48K -> chunk 1024, used 11652 MiB, free 630 MiB, load transient min-free 438 MiB; 64K ->
  chunk 512, used 11524 MiB, free 757 MiB, load transient 470 MiB. Both load + fit positive, no OOM.
  KV-AWARE PREFILL BUDGET landed (subtracts KV(T_ctx) from the 1536 MB activation budget; removed the
  magic 32768 cap): chunk 1024 holds to ~57-58K then drops to 512, so the user's 40960 Claude-CLI
  workload gets chunk 1024 (~8% faster across 16K-48K) and 64K auto-falls to 512 (safe). Tightest case
  = the load transient in the ~48-57K band (chunk 1024); if a 50-57K run ever load-OOMs, nudge
  `kGemmaPrefillActivationBudgetBytes` down (~1410 MB) to move the 1024->512 transition earlier without
  touching 40K or 64K. (48K/64K full-timing runs skipped; throughput follows the measured 16K/32K curve.)
  REMAINING flash levers (fa-5090 ladder, data-backed order): XOR-swizzle smem (kills the 536M-conflict
  Short Scoreboard, their V2) -> ldmatrix.x4 (V4) -> convert QK to mma.sync (drops split-K smem round-
  trip + a barrier). `use_flash_prefill_` op default STILL false; `GemmaTransformer` wires it on at
  ctx >= 16384. Llama sec 5.6 (all-global -> reclaim `preatt`/`att` entirely) still pending the same wiring.
  UPDATE 2026-07-13: smem PADDING LANDED + MEASURED -- bank conflicts were NOT the binding constraint.
  `kSmemSkew=8` in `Gqa.Flash.Wmma.cu` pads the Q/K/V smem row stride HS->HS+8 bf16 (breaks the
  1024-B-row / 128-B-bank-cycle collision; storage stride only, math unchanged; 0 = exact unpadded
  layout). Oracle green (strengthened same commit: `kGContext` 80->83 so the parity test now covers a
  ragged 19-token query tile + the cp.async OOB row clamp -- 80 covered neither; nsys-verified the test
  really runs both legs: 3 flash + 3 cuBLASLt-pipeline instances). ncu A/B on the heavy instance (seq
  8192, ctx 49152, 4070): shared-load bank conflicts 944M -> 27.5M (34x reduction, padding works as
  designed), L1/TEX pipe 52.95% -> 35.12%, Compute (SM) 25.6% -> 33.95% -- BUT Duration 22.78 -> 22.82 ms
  = ZERO wall win. The prior "conflicts = smoking gun" diagnosis is FALSIFIED by the A/B: at 16.67%
  occupancy (1 block/SM, ~91 KB smem double-buffer) the kernel is LATENCY-bound and the conflict
  serialization sat off the critical path. Padding KEPT (free, correct, pre-clears the L1/TEX pipe for
  when occupancy rises and conflicts would bind; costs +3 KB smem -> W=16 + skew fits the 101376-B Ada
  cap with only 64 B spare). REAL remaining lever = OCCUPANCY: (a) W=16 warps/block (16 warps/SM at the
  same 1 block/SM = 33% occupancy, HSt=32 halves O-regs; s_S_partial doubles to 16 KB), (b) single-V
  buffer + prefetch-K-only (fa-5090 V5, frees ~17 KB), (c) QK -> mma.sync (drops the split-K smem
  round-trip). Shared-STORE conflicts now dominate the residual (72M vs 27.5M loads: wmma
  `store_matrix_sync` of S_w ldm=16 + s_P writes) -- only worth chasing after occupancy.
  UPDATE 2026-07-13 (Stage 2d, later same day): FIRST REAL KERNEL WIN -- heavy instance 22.82 -> 20.64 ms
  (-9.6%), wall @seq8192/ctx49152 ~5900 -> 5768 ms, Compute (SM) 34 -> 42% (highest yet). Two-step arc:
  2d-v1 (QK -> raw mma.sync manual loads + barriers 5->3 + reduction fused into the 16-lane softmax) was
  GREEN but a 57% REGRESSION (35.8 ms): fusing the 8-warp reduction into 16 lanes serialized ~1.5k
  cycles/tile on half of warp 0 while 7.5 warps idled. THE STRUCTURAL LESSON: the single-warp softmax
  section was the kernel's critical path all along -- it is why padding (34x fewer conflicts), W=16 (2x
  occupancy), and barrier cuts alone never moved Duration. 2d-v2 DISTRIBUTED the softmax (each warp owns
  Br/W=2 rows, 16 lanes/row, per-lane partial sums, __shfl_xor row max/sum; shuffles outside branches;
  float2 partial stores at even stride 18) -> the win. Kernel renamed gqa_flash_prefill_mma_bf16_kernel
  (no nvcuda::wmma left in the file). Per-instance project arc: 127 -> 103 -> 47.5 -> 26.9 -> 22.8 -> 20.6 ms.
  SCOREBOARD IMPACT (22496-tok prefill @48K, 4070, min/5): 21350 -> 18941 ms = 1056 -> 1188 tok/s
  (+11.3% end-to-end, FP8 OFF, zero numerics risk); llama.cpp gap 1.95x -> 1.74x. __expf adopted
  (instance-neutral post-distribution, kept as free).
  LDMATRIX RUNG LANDED 2026-07-13 (fa-5090 V4): all four fragment-load sites (Q, K-as-B, P, V.trans)
  are warp-collective ldmatrix.x4 -- instance 20.4 -> 19.1 ms, scoreboard 18941 -> 18674 ms = 1205
  tok/s, gap 1.71x. Session total: 21350 -> 18674 = 12.5%. HARD-WON CORRECTNESS LESSON riding this
  rung: the first two builds did ILLEGAL shared reads that the parity oracle PASSED over (bogus
  generic-space addresses alias correct data at small smem offsets, then crash at real scale --
  cudaErrorIllegalAddress in ProfileModel/chat at ctx>=16384 only). Root cause was twofold and both
  halves are load-bearing: (1) convert pointers with raw-PTX cvta.to.shared.u64 (CUTLASS
  cast_smem_ptr_to_uint form), AND (2) the ldmatrix instruction MUST carry the `.shared` qualifier --
  without it PTX uses generic addressing and dereferences the converted offset as a generic address.
  (learn-cuda's reference wrapper omits `.shared` and survives only via truncated generic addresses;
  do not copy it.) NEW GATE for any PTX-level kernel change: compute-sanitizer memcheck over
  CudaGqaFlashPrefillParity (30 s; caught 11 invalid reads deterministically at the tiny oracle
  geometry where parity alone stayed green). Also: chat/oracle "green" never covers flash below
  ctx 16384 -- scale validation requires ProfileModel --model gemma --phase prefill at ctx >= 16384. Component breakdown (nsys, GPU-bound): BF16
  linear GEMMs 43.4% | global flash attn 25.2% (avg 27.2 ms/inst) | local sliding attn 18.5% (ring
  softmax 14.1 + QK/AV 4.4) | FP4 dequant 7.8% | rope/geglu/rmsnorm/rest ~5%. Remaining 8.0s gap to
  llama.cpp fully mapped: ~4s linear (FP8 activations, blocked on per-token scale numerics), ~2-2.5s
  local-layer flash (5.4 ring variant), ~1.5s flash ladder (42% compute vs fa-5090 94% SOL), ~0.8s
  dequant.
  fa-5090 / learn-cuda 07_attention (user-supplied, https://gau-nernst.github.io/fa-5090/) CROSS-CHECK:
  their best kernel also runs exactly 3 barriers/iteration (we match); their register-only P flow needs
  DIM<=128 (warp owns whole rows) so our smem-P round trip is the correct HS=512 adaptation; their
  __expf (MUFU fast exp) ADOPTED in-tree (2 softmax call sites, pending build+oracle). REMAINING LADDER
  from their V2/V4/V5, in likely value order for us: (a) single-V-buffer + interleaved prefetch
  placement (prefetch V during the QK mma, K during PV -> also frees ~17 KB smem), (b) XOR-swizzle +
  ldmatrix.x4 on the K/V loads (47M residual load conflicts; not binding at 42% compute, revisit after a).
  UPDATE 2026-07-13 (same day): W=16 warps/block TRIED + MEASURED WORSE -> REVERTED to 8. Occupancy
  doubled exactly as designed (16.67 -> 33.33% achieved) but the heavy instance got ~9% SLOWER
  (22.82 -> 24.8-25.5 ms; load bank conflicts 27.5M -> 43.3M). WHY: the block is smem-limited to 1/SM,
  so all warps share every per-tile `__syncthreads` -- extra warps in the SAME block cannot hide latency
  across a block-wide barrier; they just halve per-warp work per barrier interval and double the split-K
  reduction traffic. Occupancy-via-bigger-block is the WRONG axis; independent 2nd block/SM is
  unreachable (any K+V double-buffer at HS=512 >> the ~48 KB that would allow 2 blocks). VERDICT: the
  incremental knob ladder on Stage 2c is EXHAUSTED (padding = no wall change, W=16 = regression). The
  remaining flash-kernel win requires the STRUCTURAL Stage 2d rewrite: QK -> `mma.sync` with manual
  smem loads (drops the split-K S_w smem round-trip AND its 72M store conflicts AND ~2 of the ~5
  barriers/tile), deterministic swizzle now that loads are manual, and ideally fewer/narrower barriers
  so warps-in-flight starts paying. Reference wall at this config (seq 8192, ctx 49152, FP4, 4070,
  min of 5): 5906 ms with W=8+skew8 kernel class (W=16 run measured 5906 before its instance regression
  was known; W=8 wall to be re-confirmed next profiling pass). Report: flash2c_w16_ncu.txt (session
  scratchpad).
  UPDATE 2026-07-13 (LOCAL SLIDING LAYERS, spec 5.4): BOUNDED-RING FLASH VARIANT IMPLEMENTED (pending
  VS2026 build + gates). Targets the 18.5% local-attention bucket @48K (ring softmax 2.69 s + QK 0.46 +
  AV 0.38 over 22496 tok); expected ~1.5-2 s off the 18674 ms wall (-> ~1330-1350 tok/s). Design = fork
  of the OPTIMIZED 2d skeleton in `Gqa.Flash.Wmma.cu` as a `kBoundedRing` compile-time template axis
  (if constexpr deltas, NOT a copy). Three deltas only: (1) RING INDEXING -- cp.async source row =
  abs_pos % cache_capacity (replaces the OOB clamp); safe because every aliased ring slot (newer or
  older absolute position than requested) lands on a column the per-row causal+window mask already
  forces to -inf (band-minimum argument documented in the kernel header). (2) BAND-START key loop --
  first tile = min window_start over the block's rows (the FIRST row's, since window_start grows with
  row index); makes the sliding key loop CONSTANT (~window/16 + 1 tiles) instead of causal-triangular.
  Computed at runtime from `window`, so the global path (window 0) is instruction-identical. (3) W=4
  at HS=256 (`kMaxWarpsBounded`): block smem 47552 B + 1 KB reserve < half of Ada's 102400 B/SM ->
  TWO independent blocks/SM, the occupancy regime HS=512 cannot reach (W=8 at HS=256 = 52160 B = 1
  block/SM = the barrier-shared-warps axis W=16 proved a regression); per-warp tile shape (hs_tile 64,
  8 n-tiles) identical to the proven HS=512/W=8 shape. WIRING: new `cuda_gqa_flash_prefill_ring_bf16`
  launcher + `flash_prefill_ring` dispatcher; `CudaGqaOp::prefill_optimized` flash gate extended from
  `!kBounded && BF16` to all BF16 (kBounded routes to the ring launcher); Gemma build loop now sets
  `setUseFlashPrefill` on LOCAL blocks too (same `useFlashPrefillForContext` threshold, ctx >= 16384).
  `prefillScoreWidth` deliberately UNCHANGED (locals stop reading preatt/att when flashed, but the
  width keeps the cuBLASLt fallback valid + keeps the A/B chunk-config clean); shrinking it further
  (~100+ MB at chunk 1024) is a follow-up on this item. NEW ORACLE: `CudaGqaFlashRingPrefillParity`
  (bounded op flash-on vs flash-off cuBLASLt ring path, real local geometry HS=256/NKV=8/ctx 83 ragged
  chunks 32+32+19) x2 windows: 24 (capacity 55 -> ring WRAPS mid-prefill) and 64 (capacity 83 ->
  identity ring, isolates band-start + masking). GATES (per the ldmatrix lesson, none skippable):
  (1) both parity fixtures green, (2) compute-sanitizer memcheck over BOTH fixtures (ring math changes
  the cp.async/ldmatrix address paths; parity alone once passed over generic-address UB), (3)
  `ProfileModel --model gemma --phase prefill` at ctx >= 16384 (chat-default ctx never runs flash),
  (4) chat coherence at ctx >= 16384. Then ncu the ring kernel (occupancy: expect 2 blocks/SM) + the
  22496-tok scoreboard vs 18674 ms.
  VALIDATED 2026-07-13 (same day, all four gates green): parities green in VS2026 AND under
  compute-sanitizer memcheck (0 errors, 3 tests); chat coherent (user); ProfileModel green.
  SCOREBOARD (4070, 22496 tok, ctx 49152, FP4, min/5): 18674 -> 16395 ms (mean 16408) = 1205 -> 1372
  tok/s, -12.2% end-to-end -- BEAT the 1330-1350 estimate; llama.cpp gap 1.71x -> 1.50x. seq-8192
  wall 5768 -> 4827 ms (-16%). ncu ring instance (chunk 1024, demangled filter
  `gqa_flash_prefill_mma_bf16_kernel<(bool)1>`, skip 200): Duration 1.46 ms CONSTANT per instance;
  2 BLOCKS/SM CONFIRMED (Waves/SM 11.13 = 1024 blocks / 46 SMs / 2; smem block limit 2 at 47.55 KB
  + 1 KB driver). CAVEAT understood: occupancy % READS 16.67% -- 2 blocks x 4 warps = the same 8
  warps/SM as the global variant's 1x8; the win is INDEPENDENT barriers, not more warps -- do not
  chase the percentage. Counters: Compute (SM) 31.4%, L2 49% (now the top unit -- the band loop
  re-reads K/V through L2), DRAM 5%, regs 129/thread (limit 3 blocks, not binding). ACCOUNTING
  CLOSED: 880 instances x 1.46 ms = 1.29 s local-attention bucket vs 3.52 s prior = the full 2.28 s
  wall delta, no mystery. Updated gap map @48K (16.4 s wall): BF16 linear GEMMs ~8.2 s (FP8 acts =
  the big remaining lever) | global flash ~4.8 s (ladder headroom ~1.5 s: 42% compute vs fa-5090 94%
  SOL; ring kernel's own headroom = L2-bound at 49%) | local ring flash ~1.29 s (DONE) | FP4 dequant
  ~1.5 s | rest ~0.7 s. Remaining on this item: preatt/att further shrink (locals no longer read
  them when flashed) + Llama all-global reclaim (5.6 sibling).
  See [[project_gqa_flash_attention]]. Original analysis below still holds:
  Same capture: **`prefill_softmax_bf16_kernel` = 20.4 s / 56.9 s (36.1%)**, 1120 instances, 18.2 ms
  avg (max 27.8 ms). This is the 8 global layers' O(n^2) full-context attention at 35K tokens; the kernel
  materializes the full ~35K-wide score rows (memory-bound). The local bounded layers' `prefill_softmax_ring`
  add another 3.65 s (6.4%), so attention is ~42% of prefill total. A flash-attention formulation (tiled,
  online softmax, no materialized scores) does the same FLOPs with far less memory traffic -- the biggest
  single lever for long-context prefill (i.e. the Claude Code regime, ~35.7K-token harness prompts). Harder
  than the W4A16 fusion; sequence after it. KEY CONTEXT: the whole prefill is **~100% GPU-bound** (NVTX-projected
  GPU time 56.909 s / 56.910 s wall = 99.9996%, 2 stream syncs, 1 cudaMalloc) -- the Task Manager "spikes and
  gaps" during prefill are a coarse WDDM packet-cadence artifact (giant softmax kernels vs swarms of sub-10us
  kernels), NOT real idle; decode looks flat because its per-token kernel stream is uniform. So there is no gap
  to close; the levers are pure compute reduction (W4A16 fusion, flash attention). Measured prefill throughput
  curve (4070, nsys NVTX): pp512 = 1,405 tok/s (0.364 s), pp2048 = 1,341 tok/s (1.527 s), pp35719 = 628 tok/s
  (56.91 s). Throughput is ~flat (GEMM-bound) to 2K then falls off a cliff by 35K -- that cliff IS the O(n^2)
  attention, isolating the flash-attention win to long context. See [[project_gemma_inference_review]].
- [ ] **[perf, benchmarking] Direct on-hardware prefill comparison vs a mature local engine (llama.cpp).**
  Replace the estimated competitor ranges with real same-silicon numbers: run `llama-bench -p 512,2048,...`
  with a ~12-13B Q4 model on THIS 4070 and put it head-to-head with `ProfileModel --phase prefill` at the
  same context points. Our measured baseline (Gemma 12B FP4, 4070): pp512 = 1,405, pp2048 = 1,341,
  pp35719 = 628 tok/s. Goal: quantify the gap at short context (dequant tax) and long context (no flash
  attention) with hard data, and re-measure after the two prefill items above land. Needs a comparable GGUF
  build + a fair context-matched harness (same seq lens, warmup handling). See [[project_prefill_perf_beta]].
- [ ] **[tooling] ProfileModel prefill sweep mode (load-once, self-timed).** PARTIAL 2026-07-10: part (b) done
  -- the prefill phase now runs a few measured iterations under a `std::chrono` bracket and prints
  `seq_len, context_length, min_ms, mean_ms` directly (no nsys needed for the throughput curve). Still open:
  (a) load-once multi-seq-len sweep (each point still reloads the model ~30 s) and (c) per-point try/catch for
  OOM ceilings. Current `ProfileModel` does ONE
  seq-len per process and the prefill phase prints no timing (throughput must be read from an nsys NVTX range),
  so a multi-point context sweep (e.g. 512 -> 64K) means N model reloads (~30 s each) + N nsys captures --
  hours, most of it wasted reload + warmup double-counting. Add: (a) a sweep spec (`--seq-len-start/-end/-step`
  or a list) that loads the model ONCE and loops seq-lens (scratch grows monotonically so later points are
  warm); (b) a `std::chrono` timer around `profilePrefill` printing `seq_len, prefill_ms, tok/s` directly (no
  nsys needed for the throughput curve -- reserve nsys for dissecting a single point); (c) per-point try/catch
  so an OOM at high context reports "ceiling reached at N" instead of aborting the sweep. Enables the 512->64K
  sweep + the llama.cpp comparison above in one run. See [[project_gemma_inference_review]].
- [ ] **[tooling, minor] ProfileModel prints `FATAL ERROR: std::bad_function_call` at process teardown.**
  Observed 2026-07-07 on the `--phase prefill` path (Gemma 12B FP4), AFTER the measured NVTX range closes, so
  measurements are unaffected. Did not fire at ctx=512 but did at 2048/40960 (not obviously context-linked).
  The prefill phase never invokes the generate callback, so an empty `std::function` is being called during
  Mila/model shutdown -- likely a destructor or deferred cleanup calling an unset callback. Chat is green, so
  it is ProfileModel-harness-local, not the runtime hot path. Repro: run `ProfileModel --model gemma --phase
  prefill --seq-len 512 --context-length 2048 --quantization fp4` and watch stderr at exit.
- [~] **[perf/memory, CONFIRMED] Tied `lm_head` allocates a ~1 GB FP8 weight at graph build that is freed
  immediately at tie time -- a wasted load-time VRAM transient that lowers the loadable-context ceiling.**
  FIX IMPLEMENTED 2026-07-07 (pending VS2026 build + on-GPU validation): thread the checkpoint tie flag into
  the config (`GemmaConfig::withTieWordEmbeddings`, set in `configFromMetadata`) and, in the transformer
  onBuilding, `installSharedWeight` the embedding table into `lm_head` BEFORE `lm_head->build()` when tied. New
  `Linear::weight_installed_` flag (mirrors `output_installed_`): `installSharedWeight` now works pre-build
  (sets the flag, defers operation wiring to onBuilding), and `initializeParameters` early-returns without
  allocating the weight when installed. Files: Linear.ixx, Gemma.ixx, Gemma.Config.ixx, GemmaModel.ixx.
  VALIDATION: re-run the ProfileModel sampler -- `largest transient freed` should drop ~1 GB->~0 at ctx 512 and
  the load peak should no longer hit free=0 at 40960; `after model load` used should be UNCHANGED (settled state
  identical); chat token-parity green; Linear/TokenEmbedding tying tests green. Original finding below.
  Measured 2026-07-07 (ProfileModel VramHighWaterSampler, cudaMemGetInfo @3ms; nvidia-smi is blind under WDDM).
  Gemma 12B FP4, 4070: during load VRAM PEAKS ~1 GB above the settled resident set, then returns -- ctx 512
  peak used 11275 MiB -> settled 10278 (997 MiB freed); ctx 40960 peak hits 12282 = whole card (free 0) then
  settles to 11568 (713 free). ROOT CAUSE: `createGraph` builds `lm_head` (Linear<...,PerChannelFp8<>>) with its
  OWN `[vocab_size, model_dim]` FP8 weight (262144*3840*1B ~= 1 GB) because `tie_word_embeddings_` is not known
  until `loadParameters` reads checkpoint metadata; tying then calls
  `lm_head_->installSharedWeight(token_embedding_->getWeightTensorShared(), ...)`
  ([Gemma.ixx:413-425](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.ixx)) which swaps in the shared embedding
  table and DROPS the original 1 GB. So the head's own weight is pure waste on every tied load. IMPACT: the load
  PEAK (not the steady state) is the binding constraint on max loadable context -- at 40960 the load momentarily
  exhausts the card. FIX: peek `tie_word_embeddings` from the checkpoint header BEFORE building `lm_head`
  (fromPretrainedImpl already opens the reader before build) and, when tied, build the head WITHOUT its own weight
  allocation (install the shared table directly). Removes the ~1 GB load high-water -> raises the loadable-context
  ceiling on the 12 GB 4070. Core change (thread the tie flag into the build path). See [[project_gemma_chat_vram]].
  Also note nvidia-smi under WDDM cannot see this (reports idle baseline) -- ProfileModel's cudaMemGetInfo
  sampler is the measurement tool.
- [x] **[perf, CONFIRMED root cause] Global prefill/decode attention runs at width `context_length`, not used
  KV length -- ~2x tax on short prompts in a large context.** PREFILL FIX IMPLEMENTED + VALIDATED 2026-07-07
  (all three spec gates green: token-for-token parity via chat + 46/46 `CudaGqaOpTests` oracles Fp32/Bf16
  incl `Prefill_PartialFinalChunk_MatchesOracle`/`PrefillThenDecode_MatchesOracle`; tax-gone sweep; existing
  oracles green): unbounded prefill QK/AV GEMM N/K, softmax width, and Step-4
  zeroing now run at `attended_len = position_offset + chunk_len`, with `T_` kept only as the physical row
  stride (cuBLASLt ld/strides). Plans keyed on `makePlanKey(chunk_len, attended_len)`; bounded/local layers
  unchanged. DECODE still descoped (only -8%, weight-bandwidth-bound; softmax-bound freebie left as follow-up).
  TAX-GONE VALIDATED 2026-07-07 (nsys :prefill range, isolated per ctx -- back-to-back nsys runs contaminate,
  giving a bogus fixed ~4594 ms; each ctx MUST run in its own process): fixed 512-token prompt prefill now
  296.9 (ctx 512) -> 332.1 (2048) -> 332.3 (8192) -> 332.7 (16384) -> 405.5 ms (40960), vs old
  298.6/343.5/380.6/430.4/639.5. FLAT ~332 ms through 16K (attention tax removed); short-prompt tax at 40960
  cut 2.15x -> 1.37x. RESIDUAL DECOMPOSED (kernel breakdown): (1) +35 ms 512->2048-then-flat is the bounded
  local-layer `prefill_softmax_ring_bf16_kernel` (17->40 ms) as ring capacity `min(T_, window+chunk-1)` grows
  512->1535 and SATURATES -- inherent sliding-window cost, correct, not this fix's scope; (2) +73 ms at 40960
  only is the chunk heuristic dropping 512->256 (512-tok prompt splits into 2 chunks: every kernel instance
  count doubles, 40 ring + 8 global softmax = 48 Gemma layers/chunk) because the GQA preatt/att scratch is
  sized to full `T_ctx` ([Gemma.ixx:791-793](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.ixx),
  budgeted by `computeChunkRowCostBytes(B, T_ctx)` at :676) -- this is exactly the score-materialization the
  post-0.20 flash-attention rewrite (GqaAttentionExtent.md 7) subsumes, so NOT worth a separate fix before it.
  Extent fix cut prefill COMPUTE to attended_len but preatt ALLOCATION stays `T_ctx`-wide.
  Isolation experiment 2026-07-07 (fix seq_len=512,
  vary context_length, Gemma 12B FP4, 4070): prefill of the SAME 512-token prompt goes 298.6 ms (ctx 512) ->
  343.5 (2048) -> 380.6 (8192) -> 430.4 (16384) -> **639.5 ms (40960)** -- 2.1x, ~linear in context. ROOT CAUSE
  (code): the unbounded/global GQA path uses `T_stride = cache_capacity_ = T_ = context_length`
  ([CudaGqaOp.ixx:270](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx), passed as
  `T_stride` at :638-640) as BOTH the physical row stride AND the key count, so the QK GEMM, the softmax Step-4
  zeroing ([Gqa.Prefill.Bf16.cu:97](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Prefill.Bf16.cu)),
  and the AV GEMM all run at width `context_length` regardless of prompt length. Bounded/local layers use
  `cache_capacity_ = min(T_, window+chunk-1)` so they are already capped -- tax is global-layers-only (8 of 48).
  DECODE is taxed too but only MINORLY (measured 2026-07-07, same isolation, 128-token greedy decode): 44.4 tok/s
  (ctx 512) -> 40.9 (ctx 40960), just -8%. The code DOES read `cache_capacity_ = T_` columns per token
  ([CudaGqaOp.ixx:710-712], comment "Both read exactly cache_capacity_ columns"), but decode is
  WEIGHT-BANDWIDTH-bound (streaming ~6 GB of 12B FP4 weights/token dominates), so the O(context) attention is a
  small slice -- widening it 512->40960 moves only 8%. So the fix's payoff is ALMOST ENTIRELY PREFILL, not decode.
  Also note the attention is RECTANGULAR not triangular:
  even at ctx=seq_len each chunk's GEMMs span the full width, not just to its causal position. FIX: decouple the
  logical attended length (`position_offset + chunk_len` prefill / `position+1` decode) from the physical stride
  `T_` -- use attended length for GEMM K/N dims + softmax bounds + Step-4 zeroing, keep `T_` only for row
  addressing. Removes the over-alloc tax AND makes prefill attention causal-triangular; overlaps the
  flash-attention prefill item (which subsumes it). Core Mila (`Src/.../GQA` op + prefill/decode kernels +
  cuBLASLt plan geometry) -- needs agreement + a VS2026 build. MIS impact: `MILA_CONTEXT_LENGTH=40960` for Claude
  Code taxes every SHORT conversation. Full sweep curve (12 points, no OOM to 57344 -- the 12 GB 4070 holds 56K
  context when allocated tightly): pp512=1713, 2048=1440, 4096=1362, 8192=1235, 12288=1130, 16384=1050,
  20480=972, 24576=901, 32768=808, 40960=635, 49152=587, 57344=565 tok/s. Fit T(n)=a*n+b*n^2 gives a~=0.65 ms/tok,
  b~=1.95e-8 s/tok^2; attention crossover (a*n=b*n^2) ~=33K tokens -- Claude Code (35.7K) sits just past it.
  DESIGN: [Mila/Specifications/GqaAttentionExtent.md](Mila/Specifications/GqaAttentionExtent.md) (attended-length
  vs physical-stride fix; interim ahead of the post-0.20 flash-attention rewrite that subsumes it). See [[project_gemma_inference_review]].
  UPDATE 2026-07-16: the decode side was re-measured and decomposed — see the "GQA decode attention" item below;
  the fused decode-attention kernel proposed there subsumes the descoped decode freebie.
- [ ] **[perf, MEASURED 2026-07-16] GQA decode attention costs 4.63 ms/token = 18.6% of decode GPU busy
  (Gemma 12B FP4, 4070, 32K allocated context) — a fused decode-attention kernel is the lever.**
  MEASUREMENTS (ProfileModel `--model gemma --phase decode --tokens 128`, greedy, decode positions ~16-144,
  i.e. near-empty cache): 40.15 tok/s (ctx 4096) -> 39.82 (16384) -> 38.65 (32768) — decode slows with
  ALLOCATED context at a fixed position, ~0.96 ms/token tax at 32K, matching the full-capacity global K/V
  read model exactly (8 global layers x 2 tensors x 32768 x 512 x 2B = 537 MB of mostly-uninitialized reads
  @ ~430 GB/s). nsys per-token decomposition @32K (64-token run, 63 decode steps, GPU busy 24.86 ms vs wall
  ~25.9 — the ~1 ms residual is the known launch-gap tax): FP4 weight matvecs 15.17 ms (190 launches) |
  RMSNorm 2.47 ms (337 launches — the D2 fusion target) | lm_head FP8 matvec 2.16 ms (262K vocab, at
  bandwidth floor) | **GQA total 4.63 ms** = local AV GEMM 1.74 (193 GB/s effective — worst kernel) + local
  QK 0.93 (361 GB/s) + global AV 0.67 (398 GB/s) + global QK+splitK 0.61 (456 GB/s) + ring softmax 0.42 +
  global softmax 0.10 + permute/unpermute/kvwrite 0.16 | rope/residual/geglu/split3/scale 0.41 | argmax 0.03.
  ROOT CAUSES (all in `CudaGqaOp::decode_optimized` + the decode plans, [CudaGqaOp.ixx:733-801](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx)):
  (1) decode QK/AV plans are built once with `N = cache_capacity_` — globals read the full allocated
  `T_ctx` rows of K and V every token regardless of `actual_len`; (2) local (bounded) layers read the full
  ring capacity `window + chunk - 1 = 2047` rows when only `window+1 = 1025` are ever in-window — and
  post-wrap the live slots are rotated, so a GEMM extent shrink CANNOT fix locals, only a slot-mapped
  kernel can; (3) the matvec-shaped cuBLASLt GEMMs are inefficient (local AV is M=GS=2, N=HS=256, K=2047,
  batch=8 -> 193 GB/s = ~40% of achievable); (4) `permute_q_compact` and `unpermute_output` at T=1 are
  byte-identity copies ([B,1,NH*HS] and [B*NKV,GS,HS] have the same linear layout) — 2 dead launches/layer
  = 96 launches + ~0.11 ms/token; (5) decode softmax zero-fills [actual_len, capacity) every token solely
  to keep the full-width AV GEMM valid. RANKED LEVERS:
  1. **Fused decode-attention kernel** (flash-decode style, one launch/layer, the llama.cpp approach):
     Q resident in registers/smem (global: GS=16 rows x HS 512, NKV=1 MQA; local: GS=2 x HS 256 x NKV=8),
     walk ONLY the live band (globals [0, pos]; locals the slot-mapped window), online softmax, K+V each
     read once, no preatt/att global round-trip, input taken straight from Xq and output written straight
     to Y (identity permutes fall out). 6 launches/layer -> 2 (kvcache_write + fused). Expected GQA bucket
     4.63 -> ~1.0 ms (mid context) / ~1.9 ms (full 32K, bandwidth floor) => ~+14-16% decode tok/s at 32K
     alloc; the in-tree `matvec_decode_bf16_qfp4` kernels prove ~460-470 GB/s is attainable on this GPU.
     Gates: op-level decode oracle vs the cuBLASLt path (unbounded + bounded ring-wrap fixtures),
     compute-sanitizer memcheck (PTX lesson), GemmaModelParity, coherent chat, ProfileModel decode
     scoreboard at ctx 4096 + 32768 (the 4K/32K spread should collapse).
  2. Interim if (1) is deferred: **bucketed decode plan extents on the globals** — reuse the existing
     `makePlanKey` plan-cache machinery with `N = actual_len` rounded up to 2048; reclaims the
     allocation-scaled tax (~1 ms @32K, ~1.6 @49K near-empty; decays as the cache fills). Helps locals
     only pre-wrap (see root cause 2).
  SCOPED 2026-07-16 (fused kernel, lever 1 — user green-lit attempt; payoff re-checked on post-matvec-diet
  baselines 43.24/41.26 tok/s: ~+8 tok/s common case, ~+5 worst case full-cache => clears the +5 bar):
  - KERNEL (new, lives in the empty placeholder `Gqa.Decode.Bf16.cu`): one flash-decode kernel templated
    on kHeadSize {128, 256, 512}, runtime GS/window/capacity/actual_len/scale — serves Llama (GS=4/HS128),
    Gemma local ring (GS=2/HS256), Gemma global MQA (GS=16/HS512). Walks ABSOLUTE positions p in
    [window_start, actual_len), slot = p % capacity (identity when unbounded; provably reads the same rows
    as the ring softmax's slot->abs reconstruction since the KV write wraps by capacity). Block = GS warps,
    one warp per Q row: per-lane O accumulator = HS/32 fp32 regs (16/8/4 — fits every geometry), online
    softmax state per warp, K/V staged to smem in double-buffered position tiles (8 positions @HS512 =
    32 KB + Q 16 KB; 16 @HS256), cooperative load + 1 barrier/tile. Q read directly from Xq
    ([B,1,NH*HS] — the identity-permute elimination falls out), output written directly to Y.
  - SPLIT-K ACROSS BLOCKS (mandatory — Gemma global has NKV=1: without splits, 1 block = 45 idle SMs):
    grid (NKV, splits, B); each block covers a contiguous position chunk and writes partial (m, l, O[GS][HS])
    fp32 to scratch; a small fixup kernel merges via the online-softmax combine. splits =
    clamp(ceil(band_len/256), 1, ~128); splits==1 writes Y directly (no fixup launch). Scratch = splits_max
    x NH x (HS_max+2) x 4B (~4 MB) carved from `ExecutionContext::getDeviceScratchBuffer()` AT decode time
    (never cached — the FP8-staging lesson), so NO GqaState/transformer plumbing.
  - WIRING (mirrors the flash-prefill pattern): CudaGqa.cuh decl + Dispatch `decode_attention` + BF16
    branch in `decode_optimized` (both kBounded instantiations, one kernel) behind a runtime
    `setUseFlashDecode` A/B hook, default OFF at the op; Gemma.ixx (and Llama.ixx as a stretch gate)
    enable it unconditionally for BF16. cuBLASLt decode plans + preatt/att_decode scratch KEPT as the
    fallback/oracle leg (reclaim recorded as follow-up). kvcache_write/RoPE/cache layout/FP32 path/public
    API untouched. Launches per layer 6 -> 2-3.
  - NEW ORACLE `CudaGqaDecodeParity` in CudaGqaOp.Cuda.cpp (mirrors the ring-prefill parity harness):
    fused-vs-cuBLASLt on all three real geometries, positions pinning band edges {pos < window, pos ==
    window boundary, ring-wrapped, splits forced > 1}, tol ~3e-2. compute-sanitizer memcheck on it is
    MANDATORY (new kernel + smem/cp.async addressing — the +100 ldmatrix lesson).
  - PHASES/GATES: P1 kernel+wiring+oracle (gates: oracle green + sanitizer 0 errors); P2 enable in Gemma
    (gates: GemmaModelParity token-for-token, coherent chat, ProfileModel decode 4K/32K vs 43.24/41.26,
    long-context decode probe at position >> window for the full-cache case); P2b Llama enable (own parity
    gate). FOLLOW-UPS (recorded, not blocking): single-launch atomic fixup, decode-plan/scratch reclaim
    when fused is on, FP8-KV inline dequant in the tile load (multiplies with this kernel).
  - RISKS: split-K merge numerics (classic fixup — oracle forces splits>1); ring/window edges (fixtures
    pin); smem budget at HS=512 (32+16 KB, checked); barrier cadence amortized by position tiles.
    Estimated 2-3 sessions (kernel is family-adjacent to the validated flash-prefill work).
  **P1+P2 VALIDATED ALL GATES GREEN 2026-07-16** (v1 single-buffered kernel, user build): all 3
  `CudaGqaDecodeParity` oracles green (Gemma global MQA / Gemma local ring-wrap / Llama), compute-sanitizer
  memcheck on the in-tree oracle **0 errors**, GemmaModelParity + coherent chat (user), scoreboard
  **43.24 -> 48.87 tok/s @4K (+13%)**, **41.26 -> 49.09 @32K (+19%)** — the 4K/32K allocation spread is
  GONE (32K now ~= 4K, the predicted signature). Decode campaign cumulative 2026-07-16: 40.15 -> 48.87 @4K
  (+21.7%), 38.65 -> 49.09 @32K (+27%). Implementation: `Gqa.Decode.Bf16.cu` (filled the empty placeholder,
  already in CMake) + cuh/Dispatch entries + `setUseFlashDecode` on op/component/block + Gemma.ixx enables
  unconditionally in inference mode; split-K scratch from `getDeviceScratchBuffer` fetched per call;
  cuBLASLt decode plans + scratch kept as fallback/oracle leg. Pre-build validation method (worked twice
  now — record): standalone nvcc harness including the production .cu, FP64 host reference, NaN-poisoned
  out-of-band cache slots (proves band-limited reads), 19 fixtures pinning splits==1/split-K/pre-window/
  window-boundary/ring-wrap/batch-2.
  **RUNG 2 IN TREE (pending VS2026 rebuild + regates): cp.async double-buffer + tile-granular online
  softmax.** v1 measured (standalone bench, rotating DRAM-honest buffers): local full-window 22.7 us/layer
  (3.4x over old 77) but global full-band only 227 GB/s (295 us/layer at 32K fill — WORSE than the old
  GEMMs' ~158 us there; single-buffer load/compute serialization + per-position warp serialization).
  Rung 2 = double-buffered cp.async stages (2x16 KB) + phase-split tile compute (all tile scores with
  independent interleaved reduce chains, then ONE m/l rescale per tile; ragged tail rows zero-filled +
  -inf score mask). Standalone-validated: 19/19 oracle fixtures + memcheck/initcheck/racecheck all 0;
  bench: **local 14.5 us/layer (5.3x over old)**, global full-band 247 GB/s (271 us). Expected in-model:
  a bit more off the locals; full-cache global case still net-positive overall (~-1.6 ms/token vs old
  path at full 32K: locals -2.5, globals +0.9) but not maximal.
  **RESIDUAL (follow-up, analyzed): global MQA kernel at full band is DRAM-LATENCY-LIMITED, not
  bandwidth-limited** — Little's law needs ~300 KB in flight/SM at ~600 ns latency; the design carries
  ~45 KB (one 16 KB prefetch x 2.8 blocks/SM). Deeper smem pipelines fight the smem budget (3 stages ->
  48 KB -> fewer blocks). Candidate fixes if revisited: (a) direct per-warp LDG streaming with
  position-unrolled ILP (no smem share; L1/MSHR merges the 16-row reuse); (b) smaller stages x more of
  them; (c) **FP8 KV cache halves the band bytes and therefore halves this residual** — it is the
  planned next multiplier anyway, so the residual may never need its own fix.
  **RUNG 2 VALIDATED ALL GATES GREEN 2026-07-16 — CAMPAIGN ITEM COMPLETE.** User rebuild: parity oracles
  green, chat coherent, ~49 tok/s (no wall change vs v1 — expected: rung 2's ~0.2-0.3 ms is noise at a
  20.4 ms wall). Fresh nsys decomposition on the fused build @32K (63 steps, busy 18.9 / wall 20.4 ms):
  **GQA decode total 0.369 ms/token (local 0.181 + global 0.109 + kvwrite 0.058 + fixup 0.022) — 12.5x
  down from the 4.63 ms that opened this item.** New wall map: FP4 matvecs 13.60 (67%, post-diet ~92%
  DRAM = format floor) | RMSNorm 2.36 (337 launches) | lm_head 2.18 (floor) | rope/misc 0.40 |
  launch-gap ~1.4. CONSEQUENCE: **FP8 KV is demoted from decode-perf lever to memory/long-context lever**
  (GQA decode reads are now worth ~0.15 ms/token at most). Remaining decode levers, re-ranked: (1)
  RMSNorm fusion (2.36 ms, 337 launches/token — the D2 item, est ~+3 tok/s); (2) CUDA Graphs decode step
  (the ~1.4 ms launch-gap tax, D1 finding, est ~+3.5 tok/s); together ~56 tok/s. Absolute ceiling for
  this model/GPU/format ~62-65 tok/s (FP4 weight bytes at DRAM floor).
  EXTERNAL REFERENCE MEASURED 2026-07-16: llama.cpp decode = **50.3-50.7 tok/s** (LM Studio Gemma 4 12B
  Q4_K_M, same 4070, 32K context, 128-token generations, temp 0, 3 runs via `lms` + native REST
  /api/v0/chat/completions stats). Mila 49.09 => **decode gap 1.03x** (was 1.30x at session start);
  prefill gap 1.136x. The two queued levers above would cross under llama.cpp decode.
  3. **Identity-copy elimination**: pass Xq directly as the QK A operand and write the AV output straight
     to Y — deletes `permute_q_compact` + `unpermute_output` from decode (~0.11 ms + 96 launches/token).
     Trivial, zero-risk, subsumed by (1).
  4. **FP8 KV cache** (existing follow-up, both layer kinds): halves decode K/V traffic AND KV VRAM;
     multiplies with (1) via inline dequant in the fused kernel.
  MEMORY NOTES @32K alloc (B=1): KV VRAM is locals 671 MB (40 layers x 8 KV heads x 2047-slot ring x 256
  x 2B — locals DOMINATE despite the ring) + globals 537 MB (MQA NKV=1 keeps them cheap); the ring
  capacity is `window + prefill_chunk - 1`, so the chunk-1024 rung costs +33% local KV VRAM vs chunk 512
  (2047 vs 1535 slots) — a decode-time reason to keep the chunk heuristic honest; decode scratch is
  trivial (preatt/att_decode [1,16,1,T_ctx] ~4 MB total); no per-token VRAM growth over a 128-token run
  (one-time cuBLASLt workspace +63 MB). Core Mila (`Src/.../GQA` op + decode kernels + plans) — needs
  agreement + VS2026 build. See [[project_gemma_inference_review]].
- [ ] **[perf, MEASURED 2026-07-16] FP4 decode matvec is NOT at the bandwidth floor: 63-78% of DRAM peak
  vs 97% for the FP8 lm_head matvec in the same decode — dequant ALU co-limits.** The 15.17 ms/token FP4
  matvec bucket (61% of decode busy) was assumed bandwidth-floor'd; ncu per-shape measurement (Gemma 12B
  FP4 decode, 4070, 5 instances/shape, `matvec_decode_bf16_qfp4_wide_kernel` in
  [CudaMatVecBias.Bf16.cu](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/MatVec/CudaMatVecBias.Bf16.cu))
  says otherwise: gate_up C=3840 77.8% DRAM / **82.4% SM** | local qkv C=3840 67.2 / 77.0 | down C=15360
  68.2 / 69.7 (occupancy only 31% — investigate) | o_proj C=4096 63.1 / 67.9. **SM > DRAM on every shape**:
  the inner loop spends, per weight, an `fp4_e2m1_decode` + a per-nibble `* scale` + an FP32 FMA into ONE
  serial accumulator — the kernel is arithmetic-co-limited, not purely latency-bound (long-scoreboard ~4
  warps/issue is secondary). The FP8 lm_head (1 FMA/byte, no dequant) hits 490 GB/s / 97.2% in the same
  token loop, so the machine can stream at the floor — the FP4 inner loop cannot. LEVERS (kernel-local,
  no format change): (1) **fold the group scale out of the inner loop** — accumulate a per-group partial
  and multiply once per group (kills 1 FP32 mul per weight, the cheapest big ALU cut); (2) **paired
  bf16x2/half2 FMA** (`__hfma2`-style) on dequantized pairs to halve FMA issue count (numerics
  tolerance-gated, weights are 4-bit anyway); (3) two+ independent accumulators to break the serial FMA
  dependency chain; (4) fc_down occupancy (31%) — check register count / blocks-per-SM. PAYOFF: lifting
  63-78% -> ~92% cuts the bucket 15.17 -> ~12.2 ms/token = **~+12-15% decode tok/s** — the single biggest
  decode lever found in the 2026-07-16 review (larger than the GQA fused-decode item above; the two stack
  to ~25.9 -> ~19 ms/token ~= 52 tok/s). NOT a lever: FP8 weight storage for decode (doubles DRAM bytes;
  FP4 at 70% still streams more params/s than FP8 at 97%), and "FP8 scalar" compute (does not exist on
  Ada CUDA cores — FP8 is storage + tensor-core-only; scalar math would convert to half/float regardless).
  Gates: existing Linear decode oracles + `Forward_Fp4PrefillMatchesDecodeAcrossTokenMagnitudes` +
  GemmaModelParity + chat + ProfileModel decode scoreboard. Core Mila (Linear kernels) — needs agreement.
  IMPLEMENTED 2026-07-16 (rungs 1+2 together, in tree, PENDING VS2026 build + gates): rewrote
  `matvec_decode_bf16_qfp4_wide_kernel` (both 16/32-nibble instantiations; 8-nibble fallback and FP8/BF16
  matvecs untouched) — (a) new `fp4x8_decode_bf16x2` helper decodes 8 nibbles to raw BF16 pairs via
  `__byte_perm` table selects (all E2M1 magnitudes exact in BF16; PRMT selectors masked 0x7777 because
  selector bit 3 = sign-replicate mode; FP4 sign injected into BF16 bit 15); (b) group scale folded out of
  the per-weight math into one `fmaf(scale, sub_even + sub_odd, acc)` per iteration with two independent
  raw sub-accumulators; (c) weight/activation/scale loads software-pipelined one iteration ahead.
  PRE-BUILD VALIDATION (standalone nvcc harness incl. the production file, scratchpad matvec_diet_check/
  matvec_ab): decode helper EXHAUSTIVE pass (65536 words x 8 nibbles vs `fp4_e2m1_decode`, 0 mismatches);
  full-kernel oracle vs FP64 host reference PASS on all 7 real shapes (worst rel 3.9e-3 = bf16 output
  rounding); old-vs-new A/B with rotating weight buffers (defeats L2, old side reproduces the in-model
  356-404 GB/s baseline): gate_up +16.7% (370->432 GB/s) | local qkv +17.7% (365->430) | local o_proj
  +20.2% (356->427) | global qkv +19.1% (371->442) | fc_down +14.7% (404->463) | global o_proj +25.6%
  (364->457). Bucket-weighted estimate ~-2.3 ms/token => decode ~40.15->~44 tok/s @4K alloc. Rung 3
  (bf16x2 paired FMA) held back pending post-build ncu (only if SM% still > DRAM%); rung 4 (fc_down tail)
  deferred.
  **VALIDATED ALL GATES GREEN 2026-07-16** (user: build + Linear oracles + GemmaModelParity + coherent
  chat with better stats t/s; me: scoreboard + ncu): ProfileModel decode **40.15 -> 43.24 tok/s @4K
  (+7.7%)**, **38.65 -> 41.26 @32K (+6.8%)** (-1.8 / -1.6 ms/token). ncu per-shape: the SM-vs-DRAM
  co-limit is GONE — **SM% now BELOW DRAM% on every shape**: gate_up DRAM 77.8 -> **90.1%** (SM 82.4 ->
  73.8) | fc_down 68.2 -> **87.0** (SM 69.7 -> 60.5) | local qkv 67.2 -> **81.6** (SM 77.0 -> 70.0) |
  o_proj 63.1 -> **78.7** (SM 67.9 -> 63.7). Long-scoreboard stall rose 4.3 -> 6.4 warps/issue as
  expected (kernels now genuinely memory-bound). RUNG 3 SKIPPED per the pre-agreed rule (SM% < DRAM%
  everywhere — paired-FMA issue relief would not pay). RESIDUAL headroom (accepted for now): the two
  smallest shapes (qkv 81.6%, o_proj 78.7%) trail the big ones — tail/ramp at 25-50 us kernel size and
  59/57% occupancy, i.e. rung-4 territory (tail shaping), revisit only if the decode campaign needs it;
  fc_down occupancy still 31% yet hits 87% DRAM (occupancy was never the binding constraint). COMMIT-READY.
- [ ] **[minor, stats accounting] Shared RoPE cos/sin cache is attributed to whichever op built
  first.** `CudaRopeOp` keeps the frequency tables in a process-wide shared cache
  ([CudaRopeOp.Cache.ixx](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Rope/CudaRopeOp.Cache.ixx))
  and only the creating op reports the bytes (`owns_cache_` in `getStateMemorySize`), so
  `getMemoryStats` totals shift by the table size depending on component build order across
  models/tests in one process (surfaced by the pooling State-slope test, whose floor now
  documents and subtracts the term). Deliberate sharing, wrong attribution seam — consider
  reporting it at a process/context level (or amortized) rather than per-first-owner, next time
  the stats surface is touched (same family as the D7 tied-weight/workspace no-double-count rules). When a checkpoint
  lacks an expected norm weight and `shouldInitializeParameters()` is false, `RmsNorm::weight_` keeps its
  zero-allocated value, so the norm multiplies its input by 0 and silently annihilates the activation
  (exactly what hid the Gemma `v_norm` missing-tensor bug as "zero attention" for a whole debugging
  session). Options: (a) default-init norm weight to 1.0 (identity) even when not initializing, so a
  missing weight degrades to "no scale" instead of "zero everything"; or (b) have the
  PretrainedModelReader / loadParameters track which expected parameters were never visited and ERROR
  (or at least WARN) on a missing one. (a) risks masking load bugs; (b) is the safer choice. File:
  [RmsNorm.ixx:326](Mila/Src/Dnn/Components/Normalization/RmsNorm/RmsNorm.ixx) + the Gemma/Llama
  `loadParameters` consume loops.
- [ ] **[minor, pre-existing] Llama metadata `norm_eps` vs reader `norm_epsilon` key mismatch.** The
  Llama converter writes `'norm_eps'` ([convert_weights.py](Mila/Tools/Converters/Llama/convert_weights.py))
  but `PretrainedModelReader::parseMetadataJSON` reads `'norm_epsilon'`, so `metadata.norm_epsilon` is
  always 0 for Llama checkpoints. Harmless today only because `LlamaModel::configFromMetadata` never
  calls `withRMSNormEpsilon` (RmsNorm uses the `LlamaConfig` default). Surfaced while wiring Gemma's eps
  through metadata (Gemma DOES read it; its converter writes `'norm_epsilon'` to match the reader). Fix:
  align the Llama converter key to `'norm_epsilon'` (and have `LlamaModel` read it) when the Llama load
  path is next touched.
- [ ] **[defect, inference path — ELEVATED 2026-07-02] Per-layer activation-buffer ownership wastes
  47/48 of prefill State; pool into one shared workspace.** Every component retains its own
  chunk-sized output for its lifetime (`output_ = make_unique<TensorType>` in RmsNorm / Linear /
  Residual / Swiglu / GroupedQueryAttention, plus block-owned `res0_/q_/k_/v_` in
  [Gemma.Block.ixx](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.Block.ixx)) — a training-first
  design (backward needs retained activations) that is pure waste on the inference-only Gemma path,
  where exactly one layer is live at a time. It is the root cause of the chunk-32 operating point
  and therefore of the 16x prefill weight re-read. Fix = transformer-owned `BlockActivationWorkspace`
  (slot per graph position, max local/global widths) + `installSharedOutput` on the five component
  types (mirrors `installSharedWeight`) + a BuildContext defer-allocation flag; single stream slot is
  alias-safe (input last read mid-block at `res_1`, written only at block end). ~10.4 MB -> ~0.23 MB
  per chunk-row; chunk 512 fits the 4070; prefill floor for an 8K prompt ~3.4 s -> ~0.22 s. Gemma
  wiring only (components keep self-allocation by default; training/Llama untouched). Phased plan +
  aliasing analysis + `getMemoryStats` no-double-count rule (D7-shaped) in
  **[Gemma4InferenceReview.md section 7](Mila/Specifications/Gemma4InferenceReview.md)**; validation =
  HF-greedy parity + closed-form State assertion. Supersedes lever (D) of the VRAM-footprint item
  below; finish with heuristic v2 (section 6) + revert `kGemmaPrefillChunkOverride` to 0.
  **Measured caveat (2026-07-03 Rec-0 profiling, review section 10.2):** the W4A16 prefill GEMM is
  compute-bound (~40 us per chunk-row), so chunk 512 alone buys only ~12% prefill wall-clock until
  that kernel is fixed (the W4A16 GEMM item below). Pooling stays correct as the VRAM fix and the
  chunk-lever enabler — sequence it with the GEMM work; end-state for a 2048-token prefill is
  ~1-2 s (FLOP-bound), not the 54 ms the traffic-only model suggested. *(The GEMM fix has since
  shipped — see the W4A16 item — so the chunk lever is live and this item is the multiplier.)*
  **Phase 3 MEASURED 2026-07-03: 2048-token prefill 10.21 s -> 1.57 s at chunk 512 (13.2x over
  the same-day 20.77 s baseline; review section 6.4 note has the kernel breakdown — linear GEMMs
  now ~61 TFLOPS at M = 512, softmaxes collapsed 2.01 -> 0.36 s from the added parallelism alone,
  so P2's port is worth ~0.3 s not ~2 s). Remaining validation before commit: Gemma test suite
  (State-slope floor corrected) + HF FP4 parity (covers pooling Phases 2+3) + chat at the new
  chunk-512 operating point.** Implementation: heuristic v2 as `GemmaTransformer::resolvePrefillChunkSize` — complete row-cost
  model (shared-`WorkspaceWidths` slot sum + chunk-scaled GQA attention scratch + bounded-ring
  growth under `TKvCachePolicy::kIsActive`), ladder {512,256,128,64} with a warned floor, tiny
  contexts (< 64) single-chunk; `kGemmaPrefillChunkOverride` reverted to **0** (escape hatch
  kept); v1's attention-only free function + `kGemmaPrefillScratchByteCap` replaced. Deviation
  from review section 6.4, flagged there: fixed 1536 MB activation budget instead of live
  `cudaMemGetInfo` (no device free-memory query surface exists yet; on the 4070 both pick 512).
  **Follow-ups:** (a) live `cudaMemGetInfo` budget once a device memory-info query lands in
  Compute (CMake-selected implementation unit per the no-#ifdef rule); (b) the section 6.5
  formula-vs-measured drift-pin test — the State-slope test covers the leak class, the exact
  row-cost slope pin is deferred; (c) **heuristic v3 — agentic profile (design note 2026-07-04):**
  for agent-loop serving (Codex/Claude Code via MIS, long transcripts + P4 prefix reuse) the
  scarce resource is context KV, not prefill speed — v3 should reserve the target-context KV
  budget FIRST and size the chunk workspace from the remainder (context-first, inverting v2's
  priority), fold in the live free-VRAM query from (a), and release/re-grow the FP4
  dequant-staging scratch (~470 MB, idle during decode) after prefill. Pairs with the D4
  tied-table FP8 conversion (below): ~9.5 GB steady state + reclaimed workspace makes a
  16K-32K-context agentic build realistic on the 12 GB 4070.
  **Phase 2 VALIDATED 2026-07-03 (build + chat green; State-slope test floor corrected for the
  shared-RoPE-cache attribution quirk — see the stats-accounting item below):** `installSharedOutput` on
  RmsNorm/Linear/Swiglu/Residual/GroupedQueryAttention (always-view rule guards the
  Linear/Residual leading-shape fast path; GQA pools the prefill output only, decode output
  stays owned; D7 no-double-count in every component); the BuildContext defer flag was dropped
  per approval — install-before-build (the Phase 1 idiom) covers it with zero BuildContext
  change. `GemmaBlockWorkspace` (q/k/v + 15 graph-position slots, max local/global widths,
  ~230 KB/chunk-row total) allocated once by the transformer, routed to children in
  `GemmaBlock::onBuilding`; `installSharedScratch` -> `installSharedWorkspace`. New test:
  `StateMemory_PerLayerSlopeIsKvCacheNotActivations` (Gemma.Cuda.cpp) pins the pooling
  contract closed-form. Expected on the 12B at chunk 32: per-layer activations ~333 MB -> ~7 MB.
  Remaining after validation: Phase 3 — heuristic v2 (review section 6.4) + revert
  `kGemmaPrefillChunkOverride` to 0 (-> chunk 512 on the 4070).
  **Phases 0 + 1 DONE + VALIDATED 2026-07-03 (HF-greedy parity + chat green):** Phase 0 — the `res0`
  copy deleted (see the res0 item below; baseline pinned green pre-change). Phase 1 — the
  block-owned `q_/k_/v_` split scratch pooled into one transformer-owned shared set
  (`allocateBlockScratch`, max local/global widths, install-before-build via
  `GemmaBlock::installSharedScratch`; blocks validate coverage + view prefixes; self-allocation
  stays the standalone default; D7 no-double-count in both getMemoryStats). Wiring pattern proven
  for Phase 2 (`installSharedOutput` on the five component types — the public-API step, user
  sign-off pending) and Phase 3 (heuristic v2 + chunk 512).
- [x] **[perf, measured 2026-07-03] W4A16 prefill GEMM is compute-bound at ~2.5 TFLOPS — 87.9% of
  prefill wall ("P0").** Nsight (Gemma4InferenceReview.md section 10.2): `fp4a16_wmma_gemm_kernel`
  ([CudaW4A16Gemm.Wmma.cu](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/W4A16Gemm/CudaW4A16Gemm.Wmma.cu))
  costs ~188 us fixed (weight read, ~160 GB/s) + ~40 us per chunk-row per launch, so a 2048-token
  prefill spends 17.97 s in this kernel (20.77 s total wall, 24x the review's traffic-only floor)
  and ~15.6 s of that is chunk-size-independent — the pooling/chunk-512 item above cannot pay out
  on wall-clock until this kernel is fixed. Reference points from the same capture: cuBLASLt BF16
  GEMMs hit ~26 TFLOPS at M = 32; the decode matvec reads the same weights at 379 GB/s.
  **FIX SHIPPED + VALIDATED 2026-07-03: 2048-token prefill 20.77 s -> 10.21 s (2.03x); linear
  term 17.97 s -> 7.39 s (dequant 3.98 s at ~275 GB/s + cuBLASLt GEMMs 3.41 s at ~13 TFLOPS
  aggregate, M = 32); chat coherent (HF FP4 parity test not yet re-run — opt-in). The chunk
  lever is restored: dequant traffic scales 1/chunk, GEMM efficiency rises with M — pooling ->
  chunk 512 now projects ~3-3.5 s, then P2 (softmaxes, 2.01 s = 20% of the new wall) -> ~1.5 s.
  Residual micro-item: vectorize the dequant kernel's byte loads / bf162 stores (275 -> ~400
  GB/s, worth ~1.2 s at chunk 32, less after pooling).** Implementation: FP4 -> BF16
  dequant-staging + cuBLASLt as the new default batch path, mirroring the FP8 2-phase baseline —
  new `cuda_fp4_dequantize_to_bf16` kernel (in
  [CudaW4A16Gemm.cu](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/W4A16Gemm/CudaW4A16Gemm.cu);
  BF16 rounding bit-matches the fused kernels' weight treatment), a `kUseFusedFp4Gemm = false`
  A/B toggle in [CudaLinearOp.ixx](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/CudaLinearOp.ixx)
  mirroring `kUseW8A16Gemm` (WMMA/tiled fused kernels stay live behind it), and the per-group
  branch of `buildCublasLtPlans` now builds the BF16 forward-plan cache. Staging peak = largest
  linear (fc_gate_up 61440 x 3840 ~= 472 MB BF16) via the grow-on-demand context scratch,
  fetched per forward. Expected ~2.8-3x prefill wall at chunk 32; restores the chunk lever for
  pooling. Decode (outer_size == 1 matvec) untouched. Oracle: opt-in HF-greedy FP4 parity test
  (`GemmaModel.Parity.Cuda.cpp`) + chat; prefill capture is the benchmark. **Follow-up (test
  gap, Test Suite Revival):** no component-level FP4 forward test exists — `Linear.Cuda.cpp`
  only pins the tying-throw on the quantized instantiation; backfill a decode-vs-prefill
  consistency test (identical rows through matvec and the batch path, small tolerance for
  accumulation-order differences) so the quantized batch path has a non-checkpoint oracle.
  **Raised priority 2026-07-04:** the D6 wide FP4 decode matvec now ships without a component
  guard, and the dispatch is a three-rung ladder. The test must cover all rungs: 32-nibble
  (`C % 32 == 0 && C >= 8192`), 16-nibble (`C % 16 == 0`, short), and the 8-nibble fallback
  (`C % 16 != 0`, `% 8 == 0`) — each with `C >= 2*group_size` so threads cross a group boundary
  (verifies per-group scale selection). Needs `loadParameter("weight", bf16_blob)` to drive the
  FP4 quantize+pack path (raw `copy()` into params[0] bypasses quantization).
- [x] **[perf + memory, approved 2026-07-04] D4 "Design B": convert the tied embedding/lm_head
  table to FP8 — one shared FP8 table + per-vocab-row scales, both consumers read it.
  CLOSED 2026-07-04 — all gates green same day (see VALIDATED note below).**
  **IMPLEMENTED 2026-07-04 (all four seams + tests; awaiting build + on-GPU validation):**
  (1) `TokenEmbedding` `TTableQuantization` axis + quantize-on-load via new
  `CudaTokenEmbeddingOp:Quantize` partition (reuses the Linear per-channel FP8 kernel — per-vocab-row
  IS per-channel with out=vocab, in=d); quantized path is inference-only (training build and
  backward throw); (2) FP8 gather-dequant kernels (`TokenEmbedding.Fp8.cu`, forward + decode,
  int2 FP8 loads / int4 BF16 stores) behind `cuda_token_embedding_fp8_impl`; TokenEmbeddingOp
  traits re-keyed void -> `NoWeightQuant` + new `PerChannelFp8<>` BF16 specialization;
  (3) `Linear::installSharedWeight(weight, scales)` overload installs per-channel FP8
  (weight_scales_ now shared_ptr); single-arg throws on any quantized instantiation; per-group
  throws on both; tests flipped/added incl. tied-vs-direct-quantized exact-equality oracle
  (`InstallSharedWeight_PerChannelFp8_MatchesDirectQuantizedLoad`); (4) Gemma
  `TableQuantizationPolicy = kIsQuantized ? PerChannelFp8<> : NoWeightQuant` drives both
  `TokenEmbeddingType` and `LmHeadLinearType`; NoWeightQuant body keeps the BF16 tied head
  (HF parity oracle preserved). WeightTying.md D4/§7.2 updated.
  **VALIDATED 2026-07-04 (build + suites + chat green): 40.9-41.6 tok/s sampled chat (from
  38.47 greedy baseline) — the ~-2 ms/token claim landed.** The VRAM claim initially INVERTED:
  Task Manager showed 11.2 GB (was 10.5) because quantize-on-load staged the full 2.01 GB BF16
  table through the grow-only shared scratch (previous high-water mark ~470 MB from prefill
  dequant staging) — the scratch kept the extra ~1.5 GB for the process lifetime, exceeding the
  ~1.0 GB the FP8 table saves. FIXED same day: `quantize_table_fp8_per_row` now loops row chunks
  through a `kQuantizeStagingLimitBytes` (256 MB) staging window (row scales are row-local, so
  chunking is exact; stream ordering serializes upload vs prior chunk's kernel). Post-fix
  rebuild: VRAM re-measured good (user-confirmed), all suites green, and
  `GreedyDecode_MatchesHuggingFaceTokenForToken` GREEN on the FP4 build — **decision (5)
  RESOLVED: the FP8 head flipped no greedy argmax, exact token-for-token parity remains the
  acceptance criterion** (the top-1-agreement fallback documented in the parity test header
  stays as the contingency if a longer horizon ever flips one).
  Supersedes WeightTying.md section D4 ("lm_head is never quantized") — that invariant was a
  deliberate deferral and this item is the decision to open it. The axis coincidence that makes
  it clean: per-channel FP8 scales sit on the lm_head's output-channel axis, which IS the vocab
  row the embedding gathers, so one FP32 scale tensor [vocab] serves both. Expected on the 12B
  FP4 4070 build: **~-2 ms/token decode** (lm_head weight read 2.01 GB BF16 -> ~1.0 GB FP8;
  `matvec_decode_bf16_qfp8_kernel` already exists) **and ~-1.0 GB steady-state VRAM**
  (measured 10.5 / 12 GB -> ~9.5 = 16K-context headroom for agentic sessions). Seams:
  (1) `TokenEmbedding` gains a `TTableQuantization` axis (`NoWeightQuant` | `PerChannelFp8<>`)
  mirroring `Linear` — quantized path stores `wte_` as FP8_E4M3 [vocab, d] + FP32 scales [vocab],
  quantized host-side at `loadParameter` (side effect: fixes the load-peak tied-lm_head item —
  the BF16 device table never exists);
  (2) FP8 gather-dequant kernel variant in the TokenEmbedding op (`bf16 = fp8 * scale[row]`;
  `embedding_scale` stays a forward-time multiply);
  (3) `Linear::installSharedWeight` quantized branch becomes a real install path for
  `PerChannelFp8` (weight + scales; per-GROUP FP4 stays excluded — input-axis scales do not
  transfer to a row gather); flip `InstallSharedWeight_QuantizedPath_Throws` to pin the new
  contract (throws for per-group, accepts per-channel);
  (4) Gemma wiring: `LmHeadLinearType` + the embedding table policy become conditional on the
  model's `TWeightQuantization` — a `NoWeightQuant` body keeps the BF16 tied head, so the exact
  HF token-parity oracle is PRESERVED in the reference config; FP4/FP8 bodies get the FP8 table;
  (5) DECIDE BEFORE MERGE: acceptance criterion for the FP4-mode parity test — FP8 logits can
  flip near-tie argmax, so keep exact-token over the current test length if it holds, else move
  that test to top-1 agreement rate + bounded max-logit delta vs the BF16 head.
- [x] **`TokenEmbedding::loadParameter` unknown-name fallback recursed on itself.** FIXED
  2026-07-04 (found during the D4 axis work): the else branch called `this->loadParameter(name,
  blob)` — the same override — so an unknown parameter name was a stack overflow instead of the
  base `Component::loadParameter` "does not support parameter loading" throw. Now calls
  `ComponentBase::loadParameter`.
- [~] **[perf, measured 2026-07-03] Decode calibration follow-ups: FP4 matvec bandwidth ("D6") +
  RmsNorm launch shape.** From the greedy decode capture (review section 10.1, 37.7 tok/s = 26.5
  ms/token): (a) **SHIPPED 2026-07-04** — `matvec_decode_bf16_qfp4_kernel`
  ([CudaMatVecBias.Bf16.cu](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/MatVec/CudaMatVecBias.Bf16.cu))
  sustained ~379 GB/s (15.3 ms/token, 60% of decode) with 32-bit (`uint32`) weight loads while the
  BF16 lm_head matvec in the same file proves ~484 GB/s (96% of the 4070's 504 peak) with 64-bit
  (`int2`) loads on the same access pattern. Root cause = load width. New
  `matvec_decode_bf16_qfp4_wide_kernel` loads 32 nibbles/thread via a single 128-bit `int4` (weights)
  + matching `int4` activation loads; identical arithmetic + per-group scale semantics; 32-bit kernel
  retained as fallback. **Measured 2026-07-04 (gemma_decode_d6 capture, 256-token greedy): net
  regression +1.1 ms/token.** Per-shape: fc_down (C=15360) improved 379 -> 396 GB/s, but gate_up/qkv
  (C=3840) fell to ~343-349 GB/s and o_proj (C=4096) to ~260 GB/s — at C=3840 each thread gets only
  3-4 iterations of the 1024-element stride, too few outstanding loads to hide latency plus a
  divergent tail. Same day: dispatch restricted to `C % 32 == 0 && C >= 8192` (keeps the fc_down
  win; short shapes back on the 8-nibble kernel). **Post-fix validated 2026-07-04
  (gemma_decode_d6_fixed): 37.79 tok/s greedy, regression closed** — FP4 matvec total 14.89
  ms/token (below the 15.3 calibration; gate_up ~401 GB/s on the 8-nibble kernel, fc_down +
  global-layer o_proj (C=8704) wide at ~388-400). **Follow-up IMPLEMENTED 2026-07-04 (awaiting
  on-GPU measure):** the wide kernel is now templated on `kNibblesPerThread` (16 = one `int2`
  load — the lm_head proof point's width — giving 7-8 iterations at C=3840; 32 = one `int4`);
  dispatch ladder = 32 nibbles when `C % 32 == 0 && C >= 8192`, 16 nibbles when `C % 16 == 0`
  (all short projection shapes), 8-nibble kernel otherwise. **Measured 2026-07-04
  (gemma_decode_d6_int2): 38.47 tok/s, FP4 total 14.89 -> 14.57 ms/token** — gate_up 401 -> 411
  GB/s, o_proj 339 -> 372, fc_down rung unchanged as designed. **D6 (a) CLOSED at this number:**
  the short-C plateau at ~410 GB/s is NOT load-width-limited — the FP4 matvec issues 4 bytes of
  activation loads per 0.5 bytes of weights (the BF16 lm_head proof point is 1:1) plus per-nibble
  decode ALU, so the residual ~2.8 ms/token to the 11.8 ms traffic floor would need structural
  work (stage activations in shared memory once per block, or a different dequant fusion) —
  recorded here as the re-entry point, deprioritized behind D4/D2/D3.
  (b) `rmsnorm_forward_bf16_kernel`
  D2 fusion STILL OPEN, but the same capture uncovered a separate defect, **FIXED 2026-07-04**:
  `CudaRmsNormOp::forward` launched with build-time geometry (`outer_size_` frozen at `build()`), so
  decode norms processed the full prefill-chunk row count — 32 rows at the calibration chunk 32
  (silently inflating the 2.63 ms/token baseline), 512 rows after heuristic v2 (chunk 512), doubling
  per-launch cost to ~15.6 us (5.26 ms/token) and reading up to 511 rows past the end of single-row
  decode input tensors (silent OOB reads; row 0 output stayed correct, which is why parity and chat
  never caught it). Forward now derives outer/inner from the runtime input shape with a
  max-slice-count guard, mirroring `CudaLayerNormOp::forward`. **Post-fix validated 2026-07-04:
  all 337 launches back to grid 1, 2.27 ms/token** (below the 2.63 calibration — a single warp
  reducing a 3840-wide row costs ~10 us regardless of slice count, so only the narrow QKV norms
  got cheaper; the D2 fusion target is now 2.27 ms/token). Original calibration context:
  337 norm launches/token (Gemma's sandwich + QKV norms) — more than split+scale+rope combined, so
  norm fusion (or a multi-block norm) leads the D2 cheap-fusion batch, ahead of the split->views and
  layer_scalar folds.
  `GemmaTransformer::prefill` ([Gemma.ixx:236](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.ixx))
  and `LlamaTransformer::prefill` ([Llama.ixx:211](Mila/Src/Dnn/Components/Transformers/LlaMa/Llama.ixx))
  extract the final position as `view( {B, 1, model_dim}, (T_last - 1) * model_dim )` — a contiguous
  window that is the last position of batch row 0 only; for B > 1 it spans row 0's tail plus row 1's
  head. Latent today (inference models are always built with B = 1), but the code carries B through
  every shape as if batched. Fix with a strided last-position gather, or assert B == 1 at prefill
  entry so the assumption is explicit. Found in the 2026-07-02 generate() pipeline review
  ([Gemma4InferenceReview.md](Mila/Specifications/Gemma4InferenceReview.md) — full findings +
  ranked perf recommendations: decode sync/launch structure, fused decode attention, lm_head FP8,
  prefill chunk/softmax/GEMM-extent, incremental prefill).
- [ ] **[minor, tooling] ProfileModel measured `prefill_ms` is contaminated by P4 KV prefix reuse.**
  Since the transparent prefix reuse landed (+85), the warmup run seeds `GemmaModel::kv_token_history_`
  and the measured run — same prompt, same live model
  ([ProfileModel.ixx](Mila/Profiling/ProfileModel/ProfileModel.ixx), warmup + measured both call
  `runGeneration`) — takes the rewind + `prefillFrom` path, skipping all but the last prompt token
  (observed 2026-07-04: warmup prefill 1837.9 ms vs measured 38.6 ms for a 16-token prompt).
  Decode tok/s is unaffected (same KV state either way). Fix seam: reset the model's KV history
  between warmup and measured runs (or log the reuse hit in the measured line) so `prefill_ms`
  stays comparable across builds; note the warmup number also carries first-call cuBLASLt plan
  building and allocations, so neither line currently measures steady-state prefill.
- [ ] **[minor, API edge] `max_new_tokens = 0` still emits one token.** `GemmaModel::onGenerating`
  ([GemmaModel.ixx:288](Mila/Src/Dnn/Models/GemmaModel.ixx)) emits the prefill-sampled token before
  the `max_new` bound is consulted (the decode loop starts at step 1), so a caller passing
  `max_new_tokens = 0` gets one token instead of none. Same loop structure in Llama/Gpt. Decide the
  contract (0 => no tokens, or reject 0) and guard before the first `on_token`.
- [x] **Prefill per-layer `res0` copy is suspected redundant — CONFIRMED redundant, deleted,
  HF-greedy parity + chat green 2026-07-03.** `GemmaBlock::prefill`
  ([Gemma.Block.ixx](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.Block.ixx)) copied the
  block input into `res0_` ("component buffers get overwritten downstream"), but `decode()` feeds
  the same `input` reference through the identical Residual structure with no copy, and no component
  inside the block writes the previous block's output buffer. **DELETED 2026-07-03 (pooling
  Phase 0, awaiting parity re-run):** the parity baseline was pinned green immediately before the
  change; prefill's `res_1` now reads the caller's input directly (the decode pattern), `res0_`
  and its per-chunk copy launch + full stream read/write per layer are gone. Aliasing argument
  recorded at the `res_1` call site.
- [ ] **Correctness-oracle dependency (GQA standalone-forward stub).** Component-level Gemma attention
  numerics are blocked until the `GroupedQueryAttention::forward` standalone-stub bug is resolved (see
  the GQA no-op-stub item under Test Suite Revival's bug list) — windowed-vs-global + local/global
  geometry reference cases belong to an operation-level `CudaGqaOp` test owning the `GqaState` scratch +
  cache, not the public component (whose standalone forward is a throwing stub because GQA is
  inference-only). End-to-end parity is already covered by the HF token-for-token test; this is the
  missing per-op unit oracle.
- [ ] **[Chat harness] Advertise tools to Gemma in its native `<|tool>declaration:...<tool|>` format.**
  The Gemma tool-calling round trip works (stop-at-`<tool_call|>` / dispatch / splice `<|tool_response>` /
  resume — shipped 0.20.0-alpha.6+80, see [GemmaChatProtocol.md](Mila/Specifications/GemmaChatProtocol.md)),
  but tools are still *advertised* the wrong way: `Chat::clearHistory` ([Chat.ixx](Mila/Adaptors/Chat/Src/Chat.ixx))
  primes the Gemma system turn with `"You have access to the following tools:"` + `serializeTools`'s
  **Llama-shaped plain JSON**. Gemma 4's function-calling doc
  ([ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4))
  specifies the native declaration form `<|tool>declaration:function_name{description:...,parameters:{properties:{...},required:[...]}}<tool|>`
  — the `<|tool>`/`<tool|>` tokens are already registered (`loadGemma`) and already in `stripSpecialTokens`.
  **Hypothesis worth testing empirically (per the "see for ourselves" methodology, since the doc has been
  wrong before — e.g. the `<|"|>` string-delimiter it claims, never observed in live output):** advertising
  tools in the format the model was trained to read may fix the quirks we currently normalize downstream
  (the inconsistent `default_api:` name namespacing that `GemmaToolCallParser::stripNamespace` papers over;
  possibly the empty leading thought channels) at their source. Plan: add a Gemma-native tool-declaration
  serializer (in the Chat adaptor), route the Gemma branch of `clearHistory` to it, keep the Llama
  JSON path for Llama, then re-run a tool query with `/verbose all` and capture raw to confirm the call
  format stabilizes. If `<|"|>` string delimiters actually appear once we advertise natively, add them to
  `GemmaToolCallParser` then (not speculatively now). Related still-open protocol questions in
  GemmaChatProtocol.md: parallel/multi tool calls, the model's own (vs harness-invented) `<|tool_response>`
  body grammar. Ties into the MIS tool-calling work ([ToolCalling.md](Mila/Specifications/ToolCalling.md),
  discussion #10).

---

## Generation API

The library's `LanguageModel::generate` is a **fast token generator**: prompt tokens in, tokens out through
a push callback, and the finish reason returned — nothing else. Sessions, prompt caching, and
multi-conversation routing are **harness/app concerns** (Chat, the Mila Inference server) built on the
compute primitives (`prefill`/`decode`, later `prefillFrom`/`rewindKvCache`); the library exposes those
primitives but does not implement the policy. Design + full rationale: [[project_ongenerating_overhaul]].

**Design review 2026-07-01 — the surface below SUPERSEDES the 2026-06-29 config-in / result-out reshape.**
A first-principles review found the first reshape still carried telemetry and lifetime-state that don't
belong at the model layer. The milestone now closes against this target. Decided micro-choices:
`const std::function&` (not `function_ref`); the `Generation*` naming family; `max` blessed as an
established term (like `Kv`/`Gqa`) — no `maximum` expansion; `onGenerating` kept as the protected hook name.

Target signature:

```cpp
[[nodiscard]] GenerationStatus generate(
    std::span<const TokenId> prompt_tokens,
    const std::function<void( TokenId )>& on_token,
    const GenerationParams& params = {},
    std::stop_token stop = {} );
```

**Why — the reasoning that drove each cut:**
- **ONE primitive.** `generate` + `generateStreaming` + the vector-returning `generate` are the same
  blocking, serial loop with different output sinks (the vector form is `generateStreaming` + a `push_back`
  lambda). Collapse to one callback-streaming `generate`; `onGenerating` stays the protected hook, retyped.
- **Return `GenerationStatus` only.** The finish reason is the sole result the caller cannot reconstruct
  (natural EOS vs context overflow is model-only knowledge). `MaxNewTokensReached` / `ClientCancelled` stay
  distinct precisely because the model — not a caller-side `stop_token` — owns the length cap.
- **Delete `GenerateResult` and `GenerationStatistics`.** Statistics are telemetry the harness reconstructs
  from the callback stream (prompt count = input size; TTFT = time to first callback; throughput = callback
  timestamps) — the model owns no stopwatch. `GenerateResult` was only a status+stats bundle; it dissolves.
  Permanently retires `last_generation_statistics_` / `getLastGenerationStatistics()`.
- **`max_new_tokens` / `seed` / `eos_token_id`** each ran the test "property of the request, or of the
  model/sampler?" Only `max_new_tokens` is per-call (and it earns its place because model-enforced capping is
  the only way to keep `MaxNewTokensReached` distinct from `ClientCancelled`). `seed` is a stream property
  (per-call reseed correlates outputs) -> `seedSampler(uint64_t)` once. `eos_token_id` is a model/tokenizer
  property -> established at construction (harness owns the tokenizer).

**Value types (one per module; `Generation*` / `Sampl*` families):**
- `GenerationParams { std::optional<int> max_new_tokens; SamplingParams sampling; }` — per-call request.
  `max_new_tokens` nullopt => run to EOS / context bound (no magic 128, no silent truncation). Single
  defaulted arg so the four-arg signature never churns as knobs grow.
- `SamplingParams { float temperature; int top_k; float top_p; }` — the sampling shape, forwarded to the
  sampler as `params.sampling` (loop control never reaches the sampler). Retires `using SamplingParams =
  GenerateConfig`.
- `GenerationStatus` — the return enum (rename from `GenerateStatus`; `to_string` follows).
- `SamplerConfig` — construction-time, model-fixed (vocab, softcap); rename from `SamplingConfig`.

**Renames (reverses the earlier Params->Config rename — per-call = Params):** `GenerateStatus` ->
`GenerationStatus` (module `Dnn.GenerateStatus` -> `Dnn.GenerationStatus`); `GenerateConfig`(struct) in module
`Dnn.GenerateParams` -> `GenerationParams` in `Dnn.GenerationParams`; `SamplingConfig` -> `SamplerConfig`
(`Dnn.Samplers.SamplingConfig` -> `...SamplerConfig`); new `SamplingParams` module. The pending
`Dnn.GenerateParams -> Dnn.GenerateConfig` polish item is dropped — it was backwards.

**Superseded (in the tree, to be unwound):** the 2026-06-29 reshape (green build + Gemma chat) authored
`GenerateResult`/`GenerationStatistics` in `LanguageModel.ixx`, made `generate`/`generateStreaming`/
`onGenerating` take `const GenerateConfig&` return `GenerateResult`, wired per-call `seed`
(`TokenSampler::reseed`) and per-call `eos_token_id` union (`kEosToken=1`/`kEndOfTurnToken=106`), and exported
the vocabulary through the `Mila` umbrella. The tasks below unwind the parts that don't survive.

**Shipped 0.20.0-alpha.6+79 (green build + green Gemma chat).** The family term stayed `Generate*` (the code
already used it; the `Generation*` naming in the design notes above was not adopted), and the `SamplingConfig`
-> `SamplerConfig` rename was deferred as the highest-risk cross-module rename. Delivered:
- [x] Collapsed to one `generate(prompt_tokens, on_token, params, stop) -> GenerateStatus`; deleted the
  vector-returning `generate` + `generateStreaming`; `onGenerating` retyped (`GenerateStatus` / `const GenerateParams&`).
- [x] Deleted `GenerateResult` + `GenerationStatistics` from [LanguageModel.ixx](Mila/Src/Dnn/Core/LanguageModel.ixx);
  the model records no timing. Chat + ProfileModel self-time from the callback cadence (TTFT = call -> first
  token, decode = first -> last).
- [x] `GenerateParams { std::optional<int> max_new_tokens; SamplingParams sampling; std::vector<TokenId> stop_tokens; }`
  + `SamplingParams { temperature, top_k, top_p }` in its own module `Dnn.SamplingParams`; `generate` forwards
  only `params.sampling`; the `using SamplingParams = GenerateConfig` alias retired.
- [x] `max_new_tokens` = `std::optional<int>`, nullopt => EOS / context bound (silent-truncation default fixed).
- [x] Stop set moved to construction — model defaults (`stopTokens()`), with an optional per-call
  `GenerateParams::stop_tokens` override; the per-call `eos_token_id` union retired.
- [x] `seed` -> public `LanguageModel::seedSampler(uint64_t)` (seed once); per-call `seed` removed.
- [x] `top_p` reaches the device sampler.
- [x] `last_generation_statistics_` / `getLastGenerationStatistics()` removed, stays removed.
- [x] Propagated across GemmaModel/LlamaModel/GptModel `onGenerating`, Chat, ProfileModel, and the Gemma parity test.
- [x] **MIS pybind bindings propagated (2026-07-06).** The +79 propagation list above MISSED the MilaPy
  extension: [Mila_py.Wrappers.cpp](Mila/Bindings/Mila_py.Wrappers.cpp) still built `GenerateConfig`
  and called the deleted vector-returning `generate` + `generateStreaming`, so the `mila.pyd` target had been
  red since +79. Rewrote all four `LlamaSession`/`GemmaSession` bodies onto the streaming-only
  `generate(prompt, on_token, params, stop) -> GenerateStatus` primitive (blocking flavor = seed the prompt
  vector + a `push_back` callback; both discard the `[[nodiscard]]` status via `(void)`). The `Mila.Bindings`
  interface, the pybind surface (`Mila_py.cpp`), and the Python server are all unchanged -- Python-facing
  signatures + return contracts preserved.

## MIS -> Gemma 4 migration

The whole point of adding Gemma 4 (weight-tying, FP4, bounded-KV sliding ring) was tool use with 3rd-party
agentic harnesses; Llama 3.x through MIS strips tool schemas and replies plain text. Test suite: Codex CLI +
Claude Code CLI (WSL coding harnesses) + one agentic harness (Hermes provisional). See
[[project_mis_test_environment]].

- [x] **Steps 1-2 done 2026-07-06 (bindings + server; NO core `Mila/Src/` changes -- everything Gemma needs
  already exists there).** Bindings: added `BpeTokenizer.load_gemma` (wrapper `.ixx`/`.cpp` + pybind) mirroring
  `load_llama32`; `GemmaSession::fromPretrained` now hardcodes `.withFP4Quantization()` (the default `None`/BF16
  would need ~24 GB and OOM the 12 GB 4070). Server: `MILA_MODEL_FAMILY` config (default `gemma`), `model_worker`
  loads `GemmaModel` + `load_gemma`, `prompt.py` dispatches to a Gemma template (`<|turn>`/`<turn|>` control
  tokens per the Mila checkpoint -- NOT `<start_of_turn>` -- roles system/user/model, `<bos>` prefix,
  `<|turn>model\n` primer, thinking-off `<|channel>thought\n<channel|>` prime), decode strips Gemma control
  tokens (atomic specials, so per-token stripping cleans streaming + buffered across all 5 routes untouched).
  `.env` repointed to `Data/Models/Gemma/gemma4_12b_it_bf16.bin` + `gemma_tokenizer.bin`, context 4096.
  **Requires a VS2026 `mila.pyd` rebuild.**
- [x] **Step 3 DONE -- empirical VRAM pass ("bang on the door"), 2026-07-06.** Gemma 4 12B FP4 serves a full
  **16384 context on the 12 GB 4070 with ~1.5 GB free**, generating clean text end-to-end (OpenAI
  `/v1/chat/completions`, `finish_reason: stop`, no leaked control tokens). Footprint walk (used / free MiB):
  4096 -> 10037 / 1974; 8192 -> 10352 / 1659 (+315 for +4096 tokens = **~79 KiB/token**, linear); 16384 idle ->
  10791. **8045-token prompt peaked at 10797 MiB (+6 over idle -- prefill scratch already baked into idle).** The
  sliding-KV ring is what bought it: 40 of 48 layers pinned at the 1024 window, only 8 globals grow with context.
  `.env` settled at `MILA_CONTEXT_LENGTH=16384`. Reported params 7.46B (tied embedding counted once). Optional
  final: a near-16384-token prompt to close the absolute worst case (scratch was insensitive 38->8045, so low risk).
- [x] **Codex harness connected -- Gemma 4 native tool calling end-to-end, 2026-07-06.** Codex CLI (WSL,
  `/v1/responses`, `wire_api=responses`, model id MUST match the `/v1/models` card `gemma-4-12b-it`) drives
  Gemma 4 12B FP4 through single-tool, plain-chat, and tool-result-resume flows. New Python-only module
  `gemma_protocol.py` mirrors the chat harness (GemmaToolCallParser/ChannelParser/SystemPrompt): native grammar
  `<|tool_call>call:name{key: "val", n: 42}<tool_call|>` (namespace-stripped, `<|"|>` alternate string delimiter
  handled), plain tool advertisement (NO invented syntax -- the Llama `<|python_tag|>` bridge confused Gemma into
  an empty-thought loop), `<|tool_response>response:name{...}<tool_response|>` replay leaving the model turn OPEN
  to resume. Worker stops generation at `<tool_call|>` (else the model fabricates the result) + a raw-passthrough
  mode + a degeneration backstop (bounded reasoning-channel/token-repeat caps, deliberately high). `responses.py`
  is family-branched (Gemma native vs Llama legacy). Fixes en route: `extract_content` dropped `output_text`
  (blank assistant history turns); tool-result JSON metadata (chunk ids) leaked as content. The empty-thought
  prime is LOAD-BEARING on the agentic path (removing it degenerates -- do not). REMAINING: N sequential distinct
  tool calls in one turn (lightly tested -- Codex batched); channel-content parser polish; top_p; then Claude
  Code `/v1/messages` (tool-blind today) + Hermes.
- [ ] **First-pass Gemma limitations to revisit after step 3:** (a) **DONE 2026-07-08 -- see the
  channel/pipe-token root-cause entry below;** (b) `top_p` still dropped (item below);
  (c) server `README.md` still describes Llama 3.2 3B -- rewrite once the context/VRAM envelope is known.

- [x] **Claude Code `/v1/messages` garbled output root-caused + fixed 2026-07-08 (channel leak + stray
  pipe-token leak).** A WSL Claude Code HelloWorld session surfaced two leaks, both instances of "the Gemma
  grammar recognizer is an incomplete allowlist":
  - **Channel leak (Bug 1, was item (a) above -- worse than assumed).** `extract_answer`
    ([gemma_protocol.py](Mila/Adaptors/Inference/Server/gemma_protocol.py)) unwrapped only the FIRST
    `<|channel>...<channel|>` run; interior/trailing channels survived, and `strip_control_tokens` erased only
    the `<|channel>` markers, leaving the label + reasoning body as literal text (a `thought\n<code>` block in the
    answer). The transcript proved the "fine while thinking is primed off" caveat FALSE on the agentic path: the
    12B emits mid-answer thought channels DESPITE the empty-thought prime, and the leaked content was the file
    body the model meant to Write -- so it never became a tool call and nothing was created. **Fix:** `extract_answer`
    now removes ALL channel spans via `_remove_spans(text, CHANNEL_OPEN, CHANNEL_CLOSE)` (drops an unclosed
    trailing reasoning channel too), matching the tool-span handling.
  - **Stray pipe-token leak (Bug 2).** A bare `<|>` (single-pipe registered token, NOT the enumerated `<|"|>`
    STRING_DELIM) is in neither `CONTROL_TOKENS` nor the string-delimiter check, so it rode verbatim into both the
    text answer and parsed tool arguments (`file_path: "../HelloWorld.cpp<|>"`, stray leading `,` on content).
    **Fix:** new `strip_pipe_tokens` catch-all regex `<\|[^|>]*\|>|<\|>` (two-pipe delimiter family + bare `<|>`;
    deliberately does NOT match the angle-form `<|channel>`/`<|tool_call>` markers). Wired into `strip_control_tokens`
    (after the enumerated pass) and applied to the tool name + string arg values in `parse_tool_call`.
  - **Diagnostic gap closed.** `_stream_buffered_tool` (the Anthropic/Claude Code path) had NO raw-output logging,
    unlike `_stream_responses` (the Codex path). Added a `logger.info(... %r, full_text)` so the exact channel
    structure + any un-enumerated tokens are visible ([factory.py](Mila/Adaptors/Inference/Server/routes/factory.py)).
    **NEXT (needs the raw log from a live re-run):** confirm the exact byte form of `<|>` and whether the leading-comma
    artifact has a residual cause in `_parse_arguments`' bare-literal fallback beyond the pipe-token (the catch-all
    handles the observed case; the log will confirm there is nothing else). Verified via standalone tests:
    multi-channel strip, unclosed channel, angle-marker safety, `<|>`/`<|"|>` scrub in text + args, and happy-path
    round-trips (plain answer, single leading channel, normal + rendered tool calls) all pass.
  - **Orthogonal (still open):** the 12B narrating file bodies as prose instead of emitting `<|tool_call>` at all on
    some turns is a tool-reliability/prompting concern (pairs with the `<|tool>` declaration A/B item below), not a
    parsing bug -- the fixes above ensure that WHEN it does call, the args are clean, and when it reasons, the
    reasoning does not leak as the answer.

Still open to close the milestone:

**Align `gemma_protocol.py` with Google's official Gemma 4 tool-calling spec** (verified against the docs
2026-07-06: [Function calling with Gemma 4](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4),
[Gemma 4 Prompt Formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)). The empirically
reverse-engineered token grammar is CONFIRMED correct (`<|turn>`, `<|tool_call>call:name{...}<tool_call|>`,
`<|tool_response>`, `<|"|>`, `<|channel>thought`) -- these four are the remaining divergences from the trained
format, surfaced while bringing up Hermes (which round-tripped tool calls fine; its own read-loop was a
client-side `verify_on_stop`/`file_mutation_verifier` nag, NOT a wire bug -- see [[project_mis_test_environment]]).
- [~] **Tool DECLARATIONS use plain text, not the trained `<|tool>...<tool|>` tokens -- A/B WIRED 2026-07-08,
  awaiting measurement.** `build_tool_injection`
  ([gemma_protocol.py](Mila/Adaptors/Inference/Server/gemma_protocol.py)) now takes `use_trained_declarations`
  and renders `<|tool>declaration:name{description: <|"|>...<|"|>, parameters: {...json...}}<tool|>` via the new
  `_build_trained_tool_declarations` when set; the plain-text JSON list stays the default. Gated by
  `MILA_USE_TRAINED_TOOL_DECLARATIONS` ([config.py](Mila/Adaptors/Inference/Server/config.py)), threaded through
  both Gemma call sites (Anthropic `messages.py`, OpenAI `responses.py`). **Motivating evidence (Claude Code
  HelloWorld, raw log 2026-07-08):** with plain-text declarations the 12B DOES reach for `<|tool_call>` but
  improvises an off-spec grammar -- `call:bash:command=<raw value>` (no `{...}` brace args), NO `<tool_call|>`
  close (empty `<|channel>thought\n<channel|>` used as a separator between calls), and lowercased name `bash` vs
  the offered `Bash`. `parse_tool_call` (needs `call:name{`) returns None on all of it. So the hypothesis is
  concrete: the trained declaration frame should prime the trained call frame. **NEXT: set the env true, re-run the
  exact "build the cmake project" turn, read the `buffered_tool full_text` log -- does the model flip to
  `call:name{command: "..."}<tool_call|>`?** If it still improvises, only THEN consider tolerant parsing (the
  `bash`/`Bash` case mismatch would still need handling) or a minimal trained call-syntax hint. Docstring's
  "deliberately NO call-syntax instructions" stays -- declaration wrapper is orthogonal to call-format teaching.
- [x] **Malformed/unclosed `<|tool_call>` no longer blanks the answer -- safety net DONE 2026-07-08.** When
  `parse_tool_call` cannot classify a turn (e.g. the improvised `call:bash:command=` form above), the text path
  ran `extract_answer` -> `_remove_spans(TOOL_CALL_OPEN, TOOL_CALL_CLOSE)`, which on a dangling open returns
  `text[:start]`; a response STARTING with an unclosed `<|tool_call>` collapsed to an empty string (the client saw
  a blank message -- worse than garbage). New `_strip_tool_spans` drops complete spans whole but removes only the
  MARKER of a dangling open, keeping the body, so a malformed turn degrades to readable text (the trailing human
  summary survives). Channels keep the truncating `_remove_spans` (a dangling reasoning channel SHOULD drop its
  tail). Also note: because the model omits `<tool_call|>`, the worker's stop-at-`<tool_call|>` never fires (long
  generations) and `parse_tool_call`'s `rfind` would only ever see the LAST of several calls -- both are subsumed
  by the trained-declaration fix landing the proper close token.
- [x] **`<|tool_response>` added as a stop sequence -- DONE 2026-07-08 (Gemma 4 spec compliance).** The Gemma 4
  docs state `<|tool_response>` "acts as an additional stop sequence for the inference engine" -- it is the
  ENGINE's turn to supply the result, never the model's. `_on_token`
  ([model_worker.py](Mila/Adaptors/Inference/Server/model_worker.py)) stopped only at `TOOL_CALL_CLOSE`; now stops
  at `TOOL_CALL_CLOSE OR TOOL_RESPONSE_OPEN`. Backstops a call the model fails to close (`<tool_call|>` stop never
  fires): the moment it starts fabricating a `<|tool_response>` result we cut it off. Pure Python (the stop is
  enforced in this decode callback, not C++). No risk to the good path -- a well-formed turn stops at `<tool_call|>`
  before any `<|tool_response>` could be generated. NOTE: does not catch results fabricated as PLAIN PROSE (the
  observed `</div>` + "build succeeded" case emitted no `<|tool_response>` token) -- that is the model-reliability
  problem below, not a stop-sequence gap.
- **DELIMITER DECISION SETTLED 2026-07-08 (Gemma 4 spec) -- keep `<|"|>`, do NOT flip replay to plain quotes.** The
  spec is explicit: `<|"|>` is THE trained delimiter and "all string literals in declarations, calls, and responses
  MUST be enclosed" in it, precisely so embedded `{ } , "` are literal. So MIS re-rendering replayed calls as
  `command: <|"|>...<|"|>` (the 2026-07-06 item below) is CORRECT and REQUIRED; the model's fresh plain-`"` output
  is the off-spec side. A mid-investigation hypothesis to replay in plain quotes is therefore REJECTED (it would
  reintroduce the embedded-quote parse break `<|"|>` was added to fix). Ground truth from the instrumented run
  (Claude Code HelloWorld, 2026-07-08): the pipeline is CORRECT end-to-end -- inbound `messages` clean, the
  assembled prompt tail well-formed (`<|tool_call>call:Bash{command: <|"|>ls -F<|"|>}<tool_call|>` +
  `<|tool_response>...<tool_response|>` + open turn), and the model mirrors the replayed `<|"|>` (cleanly one run,
  as the malformed near-miss `<|>` -- middle quote dropped -- another). REMAINING = model RELIABILITY reproducing
  `<|"|>` under sampling, NOT a format choice. **Sampling is NOT the lever (corrected 2026-07-08 vs the Gemma 4
  model card):** the card standardizes temp=1.0 / top_p=0.95 / top_k=64 for ALL use cases, and `.env` ALREADY sets
  exactly those (they override the stale config.py Field defaults 0.6/40/0.9). So the fumbling happens AT the
  recommended config -- lowering temperature would be wrong (Gemma is calibrated for 1.0; low temp feeds the
  degeneration backstop). Real remaining levers: (1) **wire top_p -- DONE 2026-07-08 (pending VS2026 mila.pyd
  rebuild).** Root cause was NOT just the worker: the binding `generate`/`generate_streaming`
  ([Mila_py.cpp](Mila/Bindings/Mila_py.cpp), [Mila_py.Wrappers.cpp](Mila/Bindings/Mila_py.Wrappers.cpp)) had no
  top_p parameter at all, so `params.sampling.top_p` stayed at its struct default 1.0 (nucleus OFF) no matter what
  `.env`/the client sent -- the Gemma sampler was running with top-p disabled the whole time. The kernel/op already
  implement top_p ([CudaSamplingOp.ixx:224](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Sampling/CudaSamplingOp.ixx),
  [SamplingParams.ixx:26](Mila/Src/Dnn/Components/Transformers/SamplingParams.ixx)), so this was pure plumbing:
  added a `top_p` arg (default 1.0f -- backward-compatible) to both sessions' `generate`/`generate_streaming` across
  the `.ixx` decls, `.cpp` defs (set `params.sampling.top_p`), and pybind `.def` args, and forwarded it from
  `model_worker.py`. Requires a mila.pyd rebuild to take effect. (2) if still flaky after top_p lands, bounded
  parser tolerance treating `<|>` as a degraded `<|"|>` (targets the one recurring malformation, not arbitrary junk
  like `</div>`). Effective per-request sampling is logged in `_dispatch` to reveal any client override of `.env`.
- [x] **Replay spans now emit the `<|"|>` string delimiter -- DONE 2026-07-06.** New `_render_string_value` wraps
  string values as `key:<|"|>value<|"|>` (trained form); `_render_gemma_args` (tool-call args) and
  `format_tool_response` both use it. Numbers/bools stay bare. No backslash escaping (the trained format has none,
  and the value between the delimiter tokens is literal); the only guard neutralizes an embedded `<|"|>` that would
  close the span early. Removed the now-wrong `_escape_value` (it doubled backslashes / `\"`-escaped quotes, which
  the parser never unescapes). Worker `<tool_call|>` stop is unaffected (args-rendering is prompt-side only) and
  `extract_answer` still strips `STRING_DELIM` via CONTROL_TOKENS. Our `parse_tool_call` round-trips the delimiter
  form (verified). NOTE: this diverges from the C++ chat harness `GemmaToolCallParser`, which still renders + parses
  plain quotes only -- MIS is the spec-aligned one now.
- [ ] **In-turn thoughts are dropped between tool calls.** Google's multi-turn rule: strip the model's thoughts
  from *prior* turns before replay, but KEEP thoughts *between the function calls within a single model turn*. MIS
  drops model thoughts entirely -- `_build_gemma_prompt` ([responses.py:98-111](Mila/Adaptors/Inference/Server/protocols/openai/responses.py))
  replays only the `<|tool_call>`/`<|tool_response>` spans (the preceding `<|channel>thought` is discarded by
  `parse_tool_call`/`extract_answer`), and each turn re-primes an empty `THOUGHT_PRIME`
  ([gemma_protocol.py:43](Mila/Adaptors/Inference/Server/gemma_protocol.py)). Harmless while thinking is primed off, but
  wrong once a single turn chains multiple tool calls with reasoning between them (the untested N-round case). Fold
  into the channel-parser work already listed under first-pass limitations.
- [x] **Failed tool errors now reach the model -- DONE 2026-07-06.** `format_tool_response` skips empty-string
  `_OUTPUT_KEYS` candidates (an empty `content` no longer shadows a real `output`/`stdout` nor wins over an error)
  AND surfaces `error` explicitly (it is not in `_OUTPUT_KEYS`). A failed `{"content":"","error":"file not found"}`
  now renders `<|tool_response>response:read{error: <|"|>file not found<|"|>}<tool_response|>`; both output + error
  render as two fields; a truly empty envelope falls back to the whole JSON. Verified across the empty/real/both/
  plain-string cases.

- [x] **Bring Gemma 4 tool calling to the Anthropic Messages path (`/v1/messages`) -- DONE 2026-07-06 (non-streaming).**
  `AnthropicMessagesAdapter` ([messages.py](Mila/Adaptors/Inference/Server/protocols/anthropic/messages.py)) is now
  family-branched (Gemma-native vs Llama-legacy). Inbound: `body["tools"]` (`input_schema` shape) -> a small
  `_normalize_tools` adapter -> `gemma_protocol.build_tool_injection`; assistant `tool_use` blocks ->
  `format_tool_call`, user `tool_result` blocks (`tool_use_id`+`content`) -> `format_tool_response`, both replayed as
  model tool spans spliced into an OPEN model turn (`continue_open`). Preamble text + `tool_use` in one Anthropic
  assistant message now merge into ONE model turn (back-to-back `<|turn>model` is off-distribution). Outbound:
  `parse_tool_call` on raw text -> Anthropic `tool_use` block (`stop_reason:"tool_use"`), else `extract_answer` ->
  text block (`end_turn`); `tool_use.id` = the deterministic `call_id` so it round-trips as the next
  `tool_result.tool_use_id`. Extracted `gemma_protocol.assemble_prompt` as the shared single source of truth (both
  `responses.py` and `messages.py` call it). **KEY FINDING: the worker's `<tool_call|>` stop + raw-passthrough were
  NOT shared -- they lived only in `generate_streaming` (gemma + `stop_ctrl`). The blocking `generate()`/`decode()`
  path has neither and `decode()` strips the `<|tool_call>` markers, so the non-streaming Responses tool path was
  never actually functional (Codex works because it streams).** Added `ModelWorker.generate_collect` -- drives the
  streaming primitive to completion, accumulates the RAW decode, honors the `<tool_call|>` stop + degeneration
  backstop -- and `factory._dispatch` routes tool-capable gemma adapters (gated on
  `hasattr(adapter,"parse_tool_call_from_text")` + gemma family) through it. Streaming `tool_use`
  (`content_block_start{type:tool_use}` + `input_json_delta`) still DEFERRED -- if Claude Code always streams
  `/v1/messages`, that becomes the next required step. The grammar-alignment items below were NOT bundled in
  (kept the port minimal); they still apply to both paths. See [[project_mis_test_environment]].
- [x] **Follow-up: streaming `tool_use` on `/v1/messages` -- DONE + VALIDATED 2026-07-07.** Claude Code DOES always
  stream `/v1/messages` (confirmed live: its plain turn streamed the raw `<|tool_call>` grammar back as garbled
  `text_delta`s with `stop_reason:"end_turn"`, so no tool ever fired). New `factory._stream_buffered_tool`
  ([factory.py](Mila/Adaptors/Inference/Server/routes/factory.py)) handles the tool-capable gemma streaming path:
  runs `generate_streaming` with `strip_control_tokens=False` (raw grammar kept), buffers the turn, then at close
  emits a single `tool_use` block (`content_block_start{type:tool_use}` + `input_json_delta` full args +
  `content_block_stop` + `message_delta{stop_reason:"tool_use"}`) or a clean text block -- mirrors
  `_stream_responses`. `_dispatch` computes `tool_capable` before the stream branch and routes to it when the
  adapter provides `format_stream_tool_use_block` (four new formatters on `AnthropicMessagesAdapter`); tool-blind
  adapters and Llama keep the live `_stream` token path. **Validated end-to-end**: streaming weather curl now
  returns a real `tool_use`; full Claude Code round-trip (tool_use -> local exec -> tool_result -> continue) gives a
  correct final answer, KV prefix reuse skips ~35664/35723 tokens on the continuation turn. TRADE-OFF: all gemma
  Anthropic streaming is now BUFFERED (loses live token-by-token) because block type is only known at close;
  follow-up below narrows it to tools-present only. See [[project_mis_test_environment]].
- [ ] **Refine: only buffer gemma Anthropic streaming when tools are present.** `_stream_buffered_tool` currently
  handles ALL gemma streaming (tool_capable is family-gated), so plain no-tools chat lost live token streaming.
  Thread a `has_tools` flag through `InferenceRequest` (set in `parse_chat_request`) and gate the buffered path on
  it; no-tools requests keep the live `_stream` path. Correctness is fine either way -- this is a UX/latency refinement.
- [ ] **MIS `top_p` is dropped before the sampler.** `SamplingParams` now carries `top_p`, but the pybind
  `generate`/`generate_streaming` never grew a `top_p` arg, and `ModelWorker.generate`/`generate_streaming`
  ([model_worker.py](Mila/Adaptors/Inference/Server/model_worker.py)) accept `top_p` then omit it from the `self._model`
  call. The server plumbs `top_p` all the way from the OpenAI/Anthropic request to the worker boundary
  ([routes/factory.py](Mila/Adaptors/Inference/Server/routes/factory.py)) where it is silently discarded. Wire it
  through: add `top_p` to the two `*Session` signatures (`.ixx` + `.cpp`, set `params.sampling.top_p`), a
  `py::arg("top_p") = 1.0f` in `Mila_py.cpp`, and forward it in `model_worker.py`. Small + mechanical, all
  inside the editable Inference subsystem.
- [ ] `SamplingConfig` -> `SamplerConfig` rename (deferred this pass -- highest-risk cross-module rename).
- [ ] Llama/Gpt reproducibility -- their host samplers are time-seeded only until the deferred device-sampler
  migration wires `seedSampler`.
- [ ] Note in the style guide that `max` is a blessed term (like `Kv`/`Gqa`) so the no-abbreviation rule leaves it.

- [~] **Config ownership + accessors** — Gemma DONE (stores `GemmaModelConfig`, `getNetworkConfig()` +
  `getModelConfig()`, derived `contextLength()`, bare `context_length_` gone, pybind uses `getNetworkConfig()`).
  STILL OPEN: hoist `contextLength()` to the `LanguageModel` base + make it mode-aware (training -> arch max);
  propagate the accessor pair to Llama/Gpt; `GemmaTransformer::getConfig()` hygiene; settle `int64_t`-vs-`dim_t`.
  Original design detail below:
- [ ] **(design ref) Config ownership + accessors (REVIEWs [GemmaModel.ixx:360](Mila/Src/Dnn/Models/GemmaModel.ixx:360),
  [:418](Mila/Src/Dnn/Models/GemmaModel.ixx:418), [:430](Mila/Src/Dnn/Models/GemmaModel.ixx:430), [:349](Mila/Src/Dnn/Models/GemmaModel.ixx:349)).**
  Decided 2026-06-28:
  - **Store the deployment config object** (`GemmaModelConfig model_config_`) instead of the bare
    `int64_t context_length_`. The model is a long-lived, shareable artifact (Chat hot-swaps + holds it
    after the caller's config local is gone), so deployment provenance — context length AND the
    weight-quant / kv-compression currently DISCARDED after the dispatch switch — belongs with the
    artifact for diagnostics/`toString()`. Not a pure echo to the caller.
  - **Two whole-struct const-ref accessors:** `getNetworkConfig() const -> const GemmaConfig&`
    (architecture, from metadata) and `getModelConfig() const -> const GemmaModelConfig&` (deployment).
    Return the struct, NOT per-field forwarding getters: both configs already carry their own getters, so
    a model-level `getContextLength()` would be a getter-wrapping-a-getter that drifts as fields grow; the
    structs are already public types. **Rule: never write a getter whose body is just `return
    config_.getX()`.**
  - **Promote individual scalars ONLY when earned** — hot/cross-cutting call site, mode-dependent
    resolution, or base-class contract. Qualifying set: `maxSequenceLength()` + `vocabSize()` (exist,
    pure-virtual on `LanguageModel`) + new `contextLength()` (decode-loop bound; mode-aware: inference ->
    deployment ctx, training -> arch max — the clean home for the RuntimeMode note at [:360]). Hoist the
    deployment scalar + `contextLength()` to the `LanguageModel` base so Llama gets the same treatment.
  - **Architectural-config duplication is ACCEPTED, not deduped:** `GemmaTransformer::config_` and
    `GemmaModel::config_` both build from one `configFromMetadata(metadata)` call (single source). The
    model is policy-erased (`LanguageNetwork<TDev,TPrec>` handle), the config-holder is policy-templated
    (`GemmaTransformer<...,TWeightQuant,TKvPolicy>`), and there is no policy-independent config accessor
    at the `LanguageNetwork` layer — so reading through the network would require either re-introducing
    the policy type (defeats erasure) or a heavy facts-interface virtual. Two immutable copies of a small
    value type is the pragmatic call; document the why.
  - **Add `GemmaTransformer::getConfig() -> const GemmaConfig&`** for network self-description
    (serialization already calls `config_.toMetadata()`; diagnostics want it). Concrete-transformer
    method, NOT a `Network`/`LanguageNetwork` virtual (no common arch-config type across MNIST/Gpt/Llama/
    Gemma). Note: this is hygiene; it does NOT let the model drop its copy (erasure boundary above).
  - **Type names kept as-is** (`GemmaConfig`/`LlamaConfig`/`GptConfig`): consistent two-tier convention
    (`<Arch>Config` = architecture, `<Arch>ModelConfig` = deployment); the `getNetworkConfig()`/
    `getModelConfig()` accessors carry the disambiguation at the call site. Gemma's standalone config
    MODULE (`Dnn.Components.GemmaConfig`, vs Llama/Gpt's `:Config` partition) is purposeful — lets the
    policy-erased model import the config without the templated transformer — and stays.
  - Make the stored deployment context `const dim_t`; settle `int64_t`-vs-`dim_t` static_cast smell in
    the same pass.
- [~] **Stop tokens / EOS — being moved to construction (supersedes the 2026-06-29 per-call union).**
  The 2026-06-29 solution (named `kEosToken=1` / `kEndOfTurnToken=106` defaults unioned with a per-call
  `eos_token_id`) landed green, but EOS is a model/tokenizer property, not a request parameter — passing the
  same id per call is pure repetition, and it belongs with the model. Target: the harness supplies the stop
  set when the model is built (it owns the tokenizer); an optional per-call `stop_tokens` override survives
  for advanced structured generation only. The checkpoint-metadata route stays rejected (ids aren't in the
  checkpoint; sourcing them would need a converter + format change, against the harness-owns-tokenizer split).
  Folded into the "move the stop set to construction" task above.
- [ ] **Eager sampler construction.** Build the sampler once before the first generation rather than lazily
  in `sampleNext` on the post-prefill sample. With `GenerationStatistics` deleted the model no longer
  mis-times its own prefill (that bug leaves with the stopwatch), but the lazy allocation still adds real
  first-token latency the harness will measure — construct up front.

- [x] **Chat streaming display (Gemma-first) — SHIPPED + VALIDATED 2026-07-05 (all four gates green).**
  All five phases landed in one pass, zero library changes. New modules: `Chat.RichText` (formatRich +
  wordWrap extracted from the renderer so streamed and buffered paths share one pipeline) and
  `Chat.StreamingDisplay` (UTF-8 tail-hold `StreamingDetokenizer`; `IncrementalRichFormatter` with
  line-bounded construct holds; `StreamingResponseDisplay` token-keyed router over
  `<|channel>`/`<channel|>`/`<|tool_call>`/`<tool_call|>` ids probed at tokenizer load — all four must
  resolve or the model stays buffered). `ConsoleRenderer` gained a streaming block writer that mirrors
  `wordWrap` incrementally (pending-word + deferred-whitespace buffers, ragged-right paint) plus an
  interruptible spinner sleep (<=10 ms stop, keeps the gap census clean). Decisions taken: thought
  channel dim-streams live at `/verbose thoughts`; streamed-vs-buffered validator is always on and warns
  loudly (one line via the gate-1 oracle in `emitAssistantResponse`). `streaming_capable` is a
  `ModelEntry` column (Gemma true, Llama/Gpt false). The four gates below validated in chat 2026-07-05.
  Original scope follows.
  Stream the response to the console as tokens arrive instead
  of buffering the whole turn. Zero library changes — `generate()` is already push-streaming; full-buffering
  was a display decision forced by Llama 3.x's TEXT-convention tool calls (JSON in content, detectable only
  by parsing accumulated text). Gemma 4's tool calls are PROTOCOL TOKENS (the harness already matches the
  close token by id inside `on_token`), so exact stream-suppression needs no lookahead. Timing precondition
  already met: D1 decode-ahead gives `on_token` a ~22 ms/token host budget (pre-D1, every display ms added
  directly to token time). All work in `Mila/Adaptors/Chat/Src/`; scoped phases:
  - **Phase 0 — capability flag + invariants.** Per-model streaming-capable flag (Gemma true; Llama/Gpt stay
    buffered until their deferred tool/sampler migration — no speculative JSON lookahead for a format slated
    for rework). The full `response` string keeps accumulating regardless (history + post-hoc parser retained
    as the display's validator).
  - **Phase 1 — incremental detokenizer with UTF-8 tail-hold.** Per-token `decode()` can split multi-byte
    sequences; hold incomplete tails, emit only completed characters, final flush at round end. Buffered
    concatenation hid this; streaming loses that safety net.
  - **Phase 2 — token-keyed channel router.** Small state machine on Gemma special ids: respond channel ->
    stream; thought channel -> dim-stream at `/verbose thoughts`, hidden at `off`; `<|tool_call>` open ->
    suppress + spinner note until close. Properly fixes the known cosmetic leak of a raw `<|channel>thought`
    marker into display.
  - **Phase 3 — streaming word-wrap renderer (the bulk of the work).** `ConsoleRenderer` wraps complete
    buffered blocks today; streaming needs a pending-word buffer + column tracking (flush at whitespace,
    wrap before overflow), per-line solid-color paint, leading-indent preservation, long-word/URL hard-break.
  - **Phase 4 — spinner handoff.** Spinner (+ live token counter) owns the line through prefill and
    suppressed tool rounds; first displayable token stops it and streaming takes over; `/stats` unchanged.
  - **Gates:** (1) streamed transcript characters == the buffered render of the same response (the buffered
    path is the oracle); (2) `/stats` gap census unchanged (median/p99/max within noise — display stays
    inside the D1 host budget); (3) a tool-call turn round-trips stream -> suppressed -> stream; (4) thinking
    traces route correctly at all three `/verbose` levels. Side benefits: TTFT perception (first words at
    token one instead of ~14 s for a full story), and live anomaly visibility (a capped ramble or repeated-
    word event is watchable in situ instead of a silent spinner — see the 2026-07-04 runaway incident).

- [~] **Prompt-caching / KV prefix reuse ("P4") — SHIPPED for Gemma 2026-07-03 (awaiting build +
  validation).** The latency need showed up measured (full 8K re-prefill ~5-6 s per chat turn even
  post-chunk-512; up to 4 re-prefills per tool round). Shipped per the updated
  [PromptCaching.md](Mila/Specifications/PromptCaching.md): `rewindKvCache(int) -> bool` on
  `IKvCacheLifecycle` (bool = the bounded-ring correction — the ring refuses when the stale tail
  exceeds `capacity - window`, pinned by `RewindKvCache_BoundedRingEnforcesWindowValidity`);
  delegation `CudaGqaOp`/`CudaMhaOp` -> GQA component (session stays live, unlike resetKVCache) ->
  `IDecoderLayer`/`GemmaBlock` -> `GemmaTransformer` (all-or-nothing AND); `prefillFrom(input,
  start_offset)` with `prefill = prefillFrom(input, 0)` (one chunk loop); `LanguageNetwork` base
  gains both as safe-default virtuals (throw / false — Llama/Gpt untouched). Policy = TRANSPARENT
  model-side longest-common-prefix reuse (approved deviation from the draft's caller hint):
  `GemmaModel::kv_token_history_` tracks cache contents in lockstep with decode; zero generate()/
  GenerateParams/harness/pybind changes — Chat turns, tool rounds, and MIS all win for free;
  retokenization drift degrades savings, never correctness. Tests:
  op-level rewind validity (unbounded + ring bound) + transformer full-vs-incremental logits
  parity on real (explicitly initialized) weights + offset-contract throws.
  **Remaining:** [ ] Llama chain mirror (LlamaBlock/LlamaTransformer/LlamaModel — mechanical, op
  layer already done); [ ] multi-turn TTFT measurement in chat (the "KV prefix reuse" log line +
  per-round stats). The per-conversation `GenerateSession` convenience and multi-session paged
  caching (vLLM-style radix/eviction) stay deferred as before.

- [x] **Google published a canonical Gemma 4 chat template 2026-07-09 — audited 2026-07-16, 9 divergences found,
  ALL RESOLVED. MIS now matches the reference prompt byte-for-byte on both the fresh-turn and resumed-after-tool
  cases.** (1)(2)(3)(4)(5)(6)(7)(9) fixed + gated; (8) investigated and correctly closed as LATENT (no reasoning
  exists to preserve while thinking is off). Remaining follow-ups, none blocking: promote the reference diff to a
  permanent test (needs a vendoring decision); optional N=20/arm declaration A/B to make that result quotable;
  re-check whether the `extract_answer` all-channels sweep and unclosed-tool-call net are still load-bearing now
  that (7) and (6) are fixed. Original audit below.** Upstream
  [chat_template.jinja](https://huggingface.co/google/gemma-4-12B-it/raw/main/chat_template.jinja) (header:
  "Fixed tool-calling loops, turn closures, and thinking content-ordering"; HF discussions
  [12B #35](https://huggingface.co/google/gemma-4-12B-it/discussions/35),
  [26B #20](https://huggingface.co/google/gemma-4-26B-A4B-it/discussions/20),
  [#47](https://huggingface.co/google/gemma-4-26B-A4B-it/discussions/47)) is now a REFERENCE SPEC we can diff
  against — it is not a drop-in, we build prompts natively in
  [Gemma.Protocol.ixx](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.Protocol.ixx) +
  [gemma_protocol.py](Mila/Adaptors/Inference/Server/gemma_protocol.py). **Confirmed correct:** the `<|turn>`/
  `<turn|>` + `<|channel>` + `<|"|>` vocabulary; tool spans living INSIDE an open model turn (upstream never
  closes the turn between call and response — our `continue_open` matches); and the empty-thought prime, which
  upstream emits verbatim as `<|channel>thought\n<channel|>` when `add_generation_prompt and not enable_thinking`
  — our empirical "load-bearing" finding is canonical. Divergences, highest-suspicion first:
  - **(1) Argument separator whitespace — every call and response we replay is off-spec.** Upstream emits
    `key:value` and `,` (template L172-175, L251-255). We emit `key: value` and `", "`
    (`detail::renderArguments` [Gemma.Protocol.ixx:244](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.Protocol.ixx:244),
    `_render_gemma_args` [gemma_protocol.py:354](Mila/Adaptors/Inference/Server/gemma_protocol.py:354)). Different
    tokens in a TRAINED format. We parse tolerantly but emit wrong — asymmetry hid it.
  - **(2) Nested/array argument values fall out of the DSL into raw JSON.** Upstream recurses via
    `format_argument(..., escape_keys=False)`: an array renders `[<|"|>a<|"|>,<|"|>b<|"|>]`, a nested object
    `{k:<|"|>v<|"|>}`. We `json.dumps(value)` / `it.value().dump()` → `["a", "b"]` with plain double quotes. Any
    tool with a non-scalar parameter (Claude Code's Edit/MultiEdit, Codex apply_patch) hits this on every call.
  - **(3) `result:` vs upstream's `value:` for a non-mapping tool response.** Template L178 emits
    `response:name{value:...}`; we emit `result:` ([gemma_protocol.py:422](Mila/Adaptors/Inference/Server/gemma_protocol.py:422),
    [Gemma.Protocol.ixx:390](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.Protocol.ixx:390)).
  - **(4) DONE 2026-07-16 (Python green; C++ leg pending user build). Argument ORDER — C++ and Python disagreed
    with each other.** Upstream `| dictsort` (sorted keys) everywhere. C++ `nlohmann::json` is `std::map` → sorted
    (accidentally correct); Python dicts kept insertion order → unsorted. `format_tool_response` emitted
    `result,error` (Python) vs `error,result` (C++) for identical input — **reproduced empirically before fixing**,
    not just read off the source. A cross-implementation parity bug independent of upstream. **Fix:** Python-only —
    `_render_gemma_args` now sorts by key ([gemma_protocol.py:349](Mila/Adaptors/Inference/Server/gemma_protocol.py:349));
    the C++ renderer was already correct and is now PINNED (its sort is implicit via `std::map`, so
    `nlohmann::ordered_json` would silently reintroduce the split). **Parity mechanism:** both suites assert the
    same golden literals, cross-referenced by comment — no shared-fixture plumbing (tests are one `MilaTests`
    target with no data-file precedent; revisit if (1)/(2)/(5) make the goldens unwieldy). New
    `GemmaProtocolParity.*` x4 in [Gemma.Protocol.cpp](Mila/Tests/Dnn/Components/Transformers/Gemma/Gemma.Protocol.cpp)
    + **the first Python tests in MIS** (`Server/tests/{conftest.py,test_gemma_protocol.py}`, 6 passing; `conftest`
    puts the run-in-place server dir on `sys.path`; `pytest` installed into the MIS venv — it was a declared `dev`
    extra that had never been installed). **Gate remaining:** the 4 `GemmaProtocolParity` C++ tests on a VS2026
    build — their goldens are PREDICTED from reading the renderer, not yet executed. NOTE: the goldens encode the
    CURRENT `": "` / `", "` spacing, which divergence (1) will rewrite on both sides in one commit.
  - **(5) Trained tool DECLARATIONS are a hand-rolled approximation.** `_build_trained_tool_declarations`
    ([gemma_protocol.py:185](Mila/Adaptors/Inference/Server/gemma_protocol.py:185)) emits
    `declaration:name{description:<|"|>...<|"|>, parameters:{...compact JSON...}}` joined by `\n`. Upstream
    `format_function_declaration` (L92-123) emits a full DSL: `parameters:{properties:{...},required:[<|"|>x<|"|>],
    type:<|"|>OBJECT<|"|>}`, types UPPERCASED, `nullable:true`, enum/items recursion, and NO separator between
    declarations. Per [[project_mis_test_environment]] the trained-declaration flag was *the* lever for tool
    reliability — so closing the gap between "approximately trained" and "actually trained" is the highest-value
    item here even though it is listed 5th.
  - **(6) We re-prime an empty thought channel after every tool response; upstream primes NOTHING there.**
    `assemble_prompt` appends `THOUGHT_PRIME` unconditionally including the `continue_open` path
    ([gemma_protocol.py:101-111](Mila/Adaptors/Inference/Server/gemma_protocol.py:101)). Upstream: when
    `prev_message_type == 'tool_response'` and thinking is off it emits nothing at all (L381-389). HYPOTHESIS
    (untested, plausible not proven): this stray mid-turn `<|channel>thought\n<channel|>` is what provokes the
    "12B emits mid-answer thought channels DESPITE the empty-thought prime" behavior the channel-leak entry above
    documents — i.e. our safety net may be treating a prompt bug as a model quirk. Same question applies to the
    unclosed-tool-call net (upstream opens a bare `<|tool_response>` for an unanswered call, L369-370).
  - **(7) FIXED 2026-07-16 (Python green, 25 tests; C++ leg pending user build). The PARSER was not the inverse
    of the renderer — the model->us direction (worse).** `_parse_arguments` / `detail::parseArguments` handled
    delimiter-quoted
    strings and bare scalars only; they have no recursion for `[...]` or `{...}`, so a non-scalar value is fed to
    the bare-literal branch, which terminates at the FIRST comma inside the container. **Measured repro:**
    `<|tool_call>call:edit{lines:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>` parses to `{"lines": "[a"}`;
    `{opts:{n:1}}` parses to `{"opts": "{n:1}"}`. Silent truncation to a wrong-typed string — no error, no
    nullopt. **NOT a regression** (the old raw-JSON render mangled identically: `["a", "b"]` -> `["a"`), but (2)
    makes it testable and raises the stakes: a model trained on the canonical grammar WILL emit
    `[<|"|>a<|"|>,<|"|>b<|"|>]`, and we hand the tool a truncated string. Any array/object-valued tool argument
    from the model is corrupt today. **Fix = a real recursive-descent parse of the value grammar** (mirror of the
    now-recursive renderer) in both implementations, with the render->parse round trip as the oracle — currently
    only flat-scalar calls round-trip. Deliberately NOT fixed in the (1)+(2)+(5) change: different direction,
    different design, and it deserves its own gate.
    **CONFIRMED IN THE WILD 2026-07-16 during the harness A/B, and WORSE than the repro above.** Live Claude Code
    trial (arm A, todo task): the model emitted `metadata:{alpha:Done,beta:Done,gamma:Done}` and we handed the
    harness `{"metadata": "{alpha:Done", "beta": "Done", "gamma": "Done", "},subject": "Create todo list"}`.
    The container does not merely truncate — its remaining contents **SHRED INTO BOGUS SIBLING ARGUMENTS**, and
    the corruption reaches legitimate keys: the model's correct `subject` argument was destroyed into a key
    literally named `},subject`. So the blast radius is the ENTIRE argument object, not just the container-valued
    key. A second trial did the same with quoted keys (`"metadata": "{\"alpha\": \"Done\"", "\"beta\"": "Done"`).
    Frequency: 1 of 5 todo trials (the model only sometimes reaches for a container-valued argument), 0 of 5 on
    the flat-string write task — consistent with the mechanism.
    **FIX LANDED 2026-07-16 — recursive-descent parse, both implementations.** A shared `Cursor` (text +
    position) replaces the find-next-comma scan; `parseValue` dispatches on the first character
    (`<|"|>` / `"` / `{` / `[` / bare) and `parseObject`/`parseArray` recurse through it, so container boundaries
    are actually seen. `null` added to `coerceBare`. Python's pipe-token scrub in `parse_tool_call` made
    recursive too (`_scrub_pipe_tokens_deep`) — with containers now parsing, a nested string would otherwise
    escape the scrub that top-level strings get. Renderer stays STRICT (trained form only); parser is LENIENT
    (accepts plain quotes + stray whitespace) because it reads what the model emits. Malformed input degrades:
    a truncated call keeps the arguments parsed so far rather than throwing or blanking.
    **Oracle = render->parse identity** (12 cases incl. arrays, nested objects, arrays-of-objects, deep nesting,
    empty containers, null/bool, and grammar punctuation inside strings — `"func() { return [1,2]; }"` must be
    inert). Plus a shredding regression pinning the wild case and, critically, that SIBLING arguments survive.
    **Verified the tests fail against the old parser before trusting them** (old: `{"lines": "[a"}`,
    `{"opts": "{n:1}"}`, and the metadata shred; new: correct containers).
    CAVEAT: the regression fixture RECONSTRUCTS the wild call's shape — only the parsed wreckage was captured
    from the transcript, not the model's verbatim bytes; it reproduces the truncation + sibling leak but not the
    exact `},subject` key. **Gate owed: VS2026 build + the new C++ `GemmaProtocolRoundTrip`/`GemmaProtocolShredding`
    tests.**
  - **(8) LATENT, NOT ACTIVE — corrected 2026-07-16 within the hour of filing it. Read the correction at the
    bottom of this item before acting on it.** Originally filed as "we delete the model's reasoning on every tool
    round"; that claim did not survive checking. Keep for when thinking mode goes ON.
    **(original framing, premise now known false)** The Gemma 4 doc forbids stripping thoughts mid-tool-turn and
    names loops as the consequence.
    [prompt-formatting-gemma4](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4): thoughts are
    stripped BETWEEN turns, but *"If a single model turn involves function or tool calls, thoughts must NOT be
    removed between the function call and the tool response"* — the model needs its own prior reasoning to know
    why it called the tool and how to read the result; stripping it mid-turn is documented to trap the model in
    cyclical reasoning loops. The upstream template honors this (its thinking gate renders reasoning for any
    message after the last user message, i.e. the in-flight tool turn, independent of `preserve_thinking`, which
    only covers OLDER turns).
    **MEASURED, not theorized.** A rebuilt round-2 prompt from the 2026-07-16 A/B (mis_armA.log) reads:
    ```
    <|turn>model
    <|tool_call>call:Write{content:<|"|>hello world<|"|>,...}<tool_call|>
    <|tool_response>response:Write{value:<|"|>File created successfully...<|"|>}<tool_response|>
    <|channel>thought
    <channel|>
    ```
    Google's shape is `<|turn>model <|channel>thought ...reasoning... <channel|><|tool_call>..<|tool_response>..`
    then generate. **Ours is the inverse: the real reasoning is GONE from before the call, and an EMPTY channel is
    bolted on after the response.** (6) and (8) are the same four lines.
    **Root cause — the reasoning cannot survive the round trip.** `extract_answer` strips all channels before the
    response leaves (verified: a `<|channel>thought\nI should write the file first.<channel|><|tool_call>...`
    emission reduces to `''`), and MIS has NO reasoning field on the wire in either direction — `grep` for
    thinking/reasoning across `protocols/anthropic/messages.py`, `routes/factory.py`, `protocols/openai/responses.py`
    returns nothing. The prompt is rebuilt purely from the client's message history, so what we deleted can never
    come back.
    **Fix direction (needs design agreement, NOT started):** carry reasoning on the wire the way the template
    expects it — the template reads `message.get('reasoning') or message.get('reasoning_content')` off the
    assistant message and re-renders it into the channel. Anthropic's analog is `thinking` content blocks, which
    Claude Code already round-trips across tool calls. So: emit reasoning as `thinking` blocks, and on rebuild
    render them back into `<|channel>thought\n..<channel|>` BEFORE the tool call. Server-side caching keyed on
    tool_use id is the stateful alternative (MIS is stateless per request today — worse fit).
    **CORRECTION 2026-07-16 — (8) IS LATENT, NOT ACTIVE. I over-claimed; the premise failed on inspection.**
    I verified that the rebuilt prompt carries no reasoning before the tool call, then *inferred* that we had
    deleted some. Never checked whether the model PRODUCES any. It does not: with thinking off, across the A/B's
    **39 (arm A) / 37 (arm B) raw model outputs, only 2 / 4 contained a `<|channel>` at all** — and every one of
    those is an EMPTY channel, not reasoning. The model goes straight to `<|tool_call>`. The doc's retention rule
    presupposes thinking mode is ON (there must be thoughts to retain); we run it OFF. **So there is nothing to
    preserve and no wire carrier is needed today.** Revisit only if thinking mode is enabled for agentic use — at
    which point the fix direction above (reasoning as Anthropic `thinking` blocks / the template's
    `message['reasoning']`) is the right starting point.
    **What the same evidence DOES show — the model PARROTS the empty prime (this is (6)'s real mechanism):**
    ```
    arm A (682 chars): <|channel>thought\n<channel|><|channel>thought\n<channel|>...   <- RUNAWAY, capped by the backstop
    arm B (165 chars): <|channel>thought\n<channel|><|tool_call>call:TaskCreate{...}<tool_call|>
    arm B (205 chars): <|channel>thought\n<channel|><|channel>thought\n<channel|><|tool_call>call:TaskCreate{...}
    ```
    We show it an empty thought channel; it emits empty thought channels back, sometimes several, once runaway.
    **That 682-char output is the documented "empty-thought loop" caught live** — and it is the degeneration
    backstop earning its keep. NOTE the arm-B cases follow a TURN-START prime (round 1, no tool response yet), so
    the echo is NOT exclusively (6)'s stray injection — the turn-start prime is parroted too. Tension to resolve:
    the doc offers the prime as the CURE for ghost channels, our notes say removing it degenerates, and this data
    shows the model echoing it. All three can be true only if the prime trades reasoning-degeneration for a
    low-rate empty-echo. **Do not "fix" this by intuition — it needs the measurement.**
    **METHOD LESSON (second over-claim of the session, same shape):** I called this "MEASURED, not theorized" while
    having measured only the *absence* in the output and inferred the cause. Absence of X in the result does not
    establish that X was produced and removed. Check the producer before blaming the remover.
  - **(6)+(9) FIXED 2026-07-16 — MIS now matches the reference template BYTE-FOR-BYTE on both cases.**
    `assemble_prompt` ([gemma_protocol.py:116](Mila/Adaptors/Inference/Server/gemma_protocol.py:116)): the
    thought prime moved INSIDE the `else` (fresh-turn) branch and the `continue_open` branch lost its trailing
    `"\n"` — a resumed turn now stops dead at `<tool_response|>` so the next token continues it.
    `append_model_tool_span` in BOTH adapters ([messages.py](Mila/Adaptors/Inference/Server/protocols/anthropic/messages.py),
    [responses.py](Mila/Adaptors/Inference/Server/protocols/openai/responses.py) — Codex had the identical bug)
    joins spans with `""` not `"\n"`.
    **GATE GREEN: the reference diff is EMPTY on both cases**, driving the real adapter (`_build_gemma_prompt`),
    not a hand-built span. Pinned by `TestPromptShape` (4 tests) + 29 Python tests green. Verified the new tests
    fail against the old assembler first (old resumed prompt ended `<tool_call|>\n<|channel>thought\n<channel|>`).
    NOTE: Python-only change — no C++ involvement, `assemble_prompt` has no runtime twin.
    **LIVE MIS VERIFICATION 2026-07-16 (user asked for it before commit — correctly; (6)/(7)/(9) had only ever
    seen unit tests and string diffs, never a model). Same 10 trials as the A/B, directly comparable to arm A:**
    | metric | arm A (before 6/7/9) | V (current tree) |
    |---|---|---|
    | write task artifacts | 5/5 correct | **5/5 correct** |
    | todo task first tool | `TaskCreate` x5 | **`TaskCreate` x5** |
    | shredded sibling keys | 1 (the `},subject` wreck) | **0** |
    | raw outputs with a `<|channel>` | 2/39 (5%) | **12/32 (37.5%)** |
    | longest raw output | **682 chars (runaway loop)** | 324 chars |
    - **NO REGRESSION.** Every artifact correct, no errors, no shredding, tool selection unchanged.
    - **MY (6) HYPOTHESIS IS FALSIFIED — record it.** I predicted removing the stray prime would REDUCE ghost
      channels. It did the opposite: 5% -> 37.5%. The prime was suppressing the model's own emission on the
      resumed path; without it the model generates the empty channel itself, which is precisely what the
      reference produces (reference primes nothing there -> model supplies its own). Benign — the channels are
      EMPTY and `extract_answer` strips them; cost is ~6 tokens per resumed turn. **(6) is still right, but for
      spec-conformance, NOT for the reason I gave.**
    - **Do NOT claim the runaway is fixed.** The 682-char loop did not recur, but baseline rate was 1/39 (2.6%);
      P(0 in 32) ~= 0.43 by chance. Not evidence.
    - **(7) was NOT exercised live this run** — the model emitted no container-valued argument in 32 outputs (it
      reached for one 1/5 times in the baseline; chance). So (7) is "no regression" here; its fix rests on the
      unit tests + reconstructed repro, not a live container.
    - **Parser tolerance vindicated on real output:** the model emitted `status:<|"|>completed,<|"|>` (comma
      INSIDE the string -- faithfully preserved, model sloppiness not our bug) and `taskId:<|"|>1}` with an
      UNTERMINATED delimiter -- the degradation path recovered `"1"` instead of dropping the argument.
    - **OPEN TRADEOFF (do not paper over):** keeping the off-spec prime measurably suppresses the resumed-turn
      echo (37.5% -> 5%). We chose reference-conformance over echo suppression. Defensible (the reference is what
      the model trained on; the doc scopes the remediation to turn-start; the one runaway happened WITH the
      prime) but it IS a real trade, not a free win.
  - **(9) — stray `\n` between tool spans (FIXED, see above).** `append_model_tool_span`
    ([messages.py:199](Mila/Adaptors/Inference/Server/protocols/anthropic/messages.py:199)) joins spans with
    `"\n"`; the reference concatenates directly (`<tool_call|><|tool_response>`). Also `assemble_prompt`'s
    `continue_open` branch appends a trailing `"\n"` after the turn content. Found only because the reference
    diff below is byte-exact. Fix rides with (6) — same two lines.

- [x] **REFERENCE-PROMPT DIFF RIG 2026-07-16 (user's idea, redirected from LM Studio) — the strongest oracle we
  have, and it VALIDATES the whole (1)+(2)+(4)+(5) change byte-for-byte.** Rather than trace LM Studio (llama.cpp
  + Q4_K_M + a *stale* bundled template — see
  [lmstudio-bug-tracker#2012](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2012), its Gemma 4
  template crashes on tool calls from a missing `format_type_argument` macro), render Google's
  `chat_template.jinja` **directly with jinja2** and diff against `assemble_prompt`. No GPU, no quant confound,
  exact answer, seconds per run. Scripts: session scratchpad `render_ref.py` + `diff_prompt.py`.
  - **Environment must mirror `transformers.apply_chat_template`:** `ImmutableSandboxedEnvironment`,
    `trim_blocks=True`, `lstrip_blocks=True`, and **DEFAULT (non-strict) undefined**. `StrictUndefined` blows up
    on `value['enum']` — the template relies on missing schema keys being falsy, so PR #47's "consistent .get()
    access to prevent StrictUndefined errors" fixed the *message* keys but NOT the parameter-schema ones.
  - **RESULT — fresh turn (thinking off, 1 tool): BYTE-FOR-BYTE MATCH.** Our system turn, our trained declaration
    DSL (hand-written from reading the Jinja), our `key:value` spacing, our sorting, our turn-start prime — all
    identical to the reference. Independent confirmation that (1), (2), (4), (5) are right.
  - **RESULT — resumed after tool result: differs in EXACTLY two places**, both ours to fix: the stray `\n`
    between tool spans (9), and the trailing empty thought channel (6). Nothing else.
  - **PROMOTED TO A PERMANENT ORACLE 2026-07-16 (user: "the Google template obtained today should be the
    oracle").** Template vendored UNMODIFIED at
    [tests/reference/](Mila/Adaptors/Inference/Server/tests/reference/gemma4_12b_chat_template.jinja) with
    [PROVENANCE.md](Mila/Adaptors/Inference/Server/tests/reference/PROVENANCE.md) (source URL, retrieval date,
    sha256 `ae53464b...`, Gemma Terms of Use — NOT Mila's MIT); `jinja2` added to the `dev` extra (test-only —
    MIS never runs Jinja at serving time); attribution added to ATTRIBUTIONS.md under a new *Vendored
    Third-Party Files* section. New `test_reference_parity.py` drives the REAL adapter and asserts
    byte-identity on both cases. `conftest.py` now defaults the model-path env vars so the suite runs from any
    directory (verified: green from the server dir AND the repo root). 31 tests green.
    **Why this test is different from every other one in the suite:** the others encode *our reading* of the
    format, so a misreading yields a confidently green suite over a broken prompt — which is precisely what
    happened for nine divergences. This one takes its answer from Google's implementation. **If it goes red
    after a template re-download, that is a FINDING (upstream moved), not a test bug — do not edit the vendored
    file to make it pass.**
  - **Sequencing:** these are prompt-construction fixes in the adaptor + the runtime grammar module (a
    `Mila/Src/` change → needs agreement per CLAUDE.md). (4) landed first (self-evident bug with an oracle);
    (1)+(2)+(3)+(5) landed together 2026-07-16 as the "emit the trained format" change; (6) is behavioral and
    needs the ghost-channel symptom as its gate; (7) is its own session.

- [~] **(1)+(2)+(3)+(5) IMPLEMENTED 2026-07-16 — Python green (16 tests), C++ leg + harness A/B PENDING.**
  One change, because all four are "emit the trained format" and share a gate. (3) was folded in beyond the
  originally-agreed (1)+(2)+(5) scope: it is one word, and leaving it would put a known deviation INSIDE the
  spec-conformant arm of the A/B.
  - **Runtime** ([Gemma.Protocol.ixx](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.Protocol.ixx)): new
    recursive `detail::renderValue` (null/bool bare, strings delimiter-wrapped, objects/arrays recurse with bare
    keys per the template's `escape_keys=False`); `renderArguments` now emits `key:value` / `,` with no
    whitespace; non-mapping tool response keys on `value:` not `result:`.
  - **MIS** ([gemma_protocol.py](Mila/Adaptors/Inference/Server/gemma_protocol.py)): mirror `_render_value`
    (bool checked BEFORE the numeric fallback — bool is an int subclass in Python); `_render_gemma_args` matches;
    `_build_trained_tool_declarations` rewritten from a compact-JSON approximation to the full
    `format_function_declaration` DSL — `properties:{name:{description:..,type:<|"|>STRING<|"|>}}`,
    `required:[<|"|>x<|"|>]`, UPPERCASED types, enum/items/nullable branches, positional (NOT alphabetical) field
    order, declarations concatenated with no separator and no leading blank line.
  - **Deliberate deviation from upstream (documented in-code):** the template closes the `parameters` block from
    inside its `type:` branch, so a schema with no `type` leaves it unclosed with a dangling comma. We always
    close it.
  - **Tests:** Python 16 passing (`TestTrainedValueGrammar`, `TestTrainedToolDeclarations` + the (4) parity
    class); C++ goldens updated for the new spacing + 5 new `GemmaProtocolParity` recursion/`value:` tests.
    Declaration grammar is MIS-only (the runtime module has no declaration renderer), so it has no parity twin.
  - **Gates: BOTH CLOSED 2026-07-16.** (1) VS2026 build + `GemmaProtocolParity` C++ tests GREEN (user); the
    predicted goldens were correct. (2) Harness A/B RUN — see below.

- [x] **HARNESS A/B RUN 2026-07-16 — trained declarations WIN on tool-name reliability, and cost 15% LESS
  context. FLAG AND LOSING BRANCH DELETED 2026-07-17 (user: "why do we need the flag at all?" — correct).**
  `use_trained_tool_declarations` (config Field + `.env` line + the `build_tool_injection` parameter + 3 call
  sites + 5 test kwargs) is gone; `build_tool_injection(tools)` always emits the trained grammar.
  **The deciding argument was conformance, NOT this A/B** (which was N=5 and not significant): the false branch
  was never a Gemma format — it was prose + `json.dumps`, invented before Google published the template, and the
  reference-parity test can never match it. Gemma has exactly ONE declaration form, so there was nothing to
  toggle between. Also removes a trap: the code default was `False`, so anyone running MIS without the shipped
  `.env` silently got the off-spec path. The evidence below stays here — where a settled question belongs —
  rather than as a live switch.
  Setup: Claude Code CLI 2.1.207 (WSL Ubuntu-Dev, `networkingMode=mirrored` so `localhost:8000` reaches the
  Windows host) -> MIS `/v1/messages` -> Gemma 4 12B FP4, ctx 49152, temp 1.0 / top_k 64 / top_p 0.95 (.env
  defaults, NOT pinned — every trial is a fresh sample). 2 tasks x 5 trials x 2 arms. Arm B via env override
  `MILA_USE_TRAINED_TOOL_DECLARATIONS=false` (server restart between arms; the setting is read once at startup).
  Scripts + raw JSONL transcripts: WSL `~/mis_ab/` (run_trial.sh, score.py, out/*.jsonl).
  - **RESULT — first tool call per trial:**
    | arm | write task | todo task |
    |---|---|---|
    | A (trained) | `Write` x5 — **5/5 correct** | `TaskCreate` x5 |
    | B (plain JSON) | `default_write`, `Write`, `Write`, `default_write_file`, `Write` — **2/5 HALLUCINATED** | `TaskCreate` x5 |
  - **The hallucinated names are the tell:** `default_write` / `default_write_file`. That `default_` prefix is the
    `default_api` tool-module habit already documented in this file's namespace-stripping entry — the model
    reaches for its trained convention, and a plain JSON list gives it nothing to anchor to. The trained
    `<|tool>declaration:` frame does. This is exactly the hypothesis `_build_trained_tool_declarations` was
    written against ("the trained declaration frame primes the trained call frame") and it now has direct
    evidence. Cost: B_write_1 never created the file at all (2 failed calls, gave up); B_write_4 burned 2 turns
    before recovering to `Write`.
  - **Context cost runs the SAME direction (bonus, was expected to be the tradeoff):** the trained DSL is
    **79,456 chars vs 93,955** for the plain JSON list at 27 Claude Code tools = **~15% cheaper**. Better AND
    smaller. (Pre-run worry that the fuller DSL would eat the 49152 ceiling was BACKWARDS.)
  - **STATISTICS — do not oversell:** 2/5 vs 0/5 at N=5/arm is **not statistically significant** (Fisher exact
    two-tailed p ~= 0.44). What carries the result is the *mechanism* (the `default_` namespace habit is a known,
    documented failure mode), its *asymmetry* (0 hallucinations in 10 arm-A trials, 2 in 5 arm-B write trials),
    and the *context win* (which is deterministic, not sampled). Treat as strong-directional, not proven. Worth
    N=20/arm before the result is quoted as fact.
  - **NULL RESULT on the todo task:** both arms picked `TaskCreate` 5/5 — the declaration form made no difference
    there. The signal is confined to the write task. Also note the model chose `TaskCreate` over the requested
    `TodoWrite` in all 10 trials, both arms — a separate tool-SELECTION question, unrelated to declarations.
  - **METHOD LESSON (cost me a wrong claim mid-run):** a Claude Code `--output-format stream-json` transcript is
    appended live, so scoring a file before its `{"type":"result"}` event lands reports "no tool call" for a
    trial that is merely still running. I reported one such phantom before catching it. **Gate every score on the
    result event** (`grep -q '"type":"result"'`), never on file existence.

## Product Family — Grammar-in-Runtime Consolidation

*Home: [MilaProductFamily.md](Mila/Specifications/MilaProductFamily.md) (Key correction + Release
Boundary — in v0.20 scope). Grammar spec home: [GemmaChatProtocol.md](Mila/Specifications/GemmaChatProtocol.md).*

The Gemma native token grammar is implemented twice and has drifted: Python `gemma_protocol.py`
(MIS) is now the spec-aligned side — it renders and parses the `<|"|>` string delimiter
(2026-07-06) — while the C++ `GemmaToolCallParser` in Chat still renders and parses plain quotes
only. Per the locked product definition, the grammar is a property of the model, not of either
adaptor: fold it DOWN into the runtime, not across. String-level parse/format helpers are the
v0.20 deliverable; token-level splice is the decided direction but explicitly post-release.

- [x] Canonical C++ grammar module in the runtime (`Dnn.Components.GemmaProtocol`,
  `Src/Dnn/Components/Transformers/Gemma/Gemma.Protocol.ixx`, 2026-07-07) — control-token
  constants + `parseToolCall` / `formatToolCall` / `formatToolResponse`, seeded from the union
  of the two prior implementations. Folds in the Python side's spec-verified behaviors: the
  `<|"|>` string delimiter (parse + render), integer-preserving argument coercion, tool-response
  output-field distillation with failed-tool error surfacing. Own test:
  `Tests/Dnn/Components/Transformers/Gemma/Gemma.Protocol.cpp`. Turn/channel parse + control-token
  stripping remain in Chat (`Chat.ChannelParser`, `stripSpecialTokens`) — not drifted, deferred to
  a follow-up so they can single-source the constants without disturbing the streaming-display oracle.
- [x] Chat consumes the runtime grammar (2026-07-07) — `GemmaToolCallParser` retired in place
  (banner + out of the ChatApp module set); `Chat.ixx` now calls `Mila::Dnn::Gemma::parseToolCall`
  / `formatToolResponse`; the `<|"|>` render/parse drift closed.
- [ ] Scope call at execution time: expose the grammar via pybind so `gemma_protocol.py` consumes
  the same source — OR, if not bounded for v0.20, keep MIS on Python and pin the two
  implementations together with a cross-language parity test (same fixture corpus, both parsers)
- [ ] Token-level splice (tool-result tokens appended straight into the live KV cache) —
  POST-release; recorded here so its absence from v0.20 is a decision, not an oversight

## Product Family — Python Binding Surface

*Home: [MilaProductFamily.md](Mila/Specifications/MilaProductFamily.md) (Layer Responsibilities —
Python binding surface).*

- [x] Promote the Python binding to a first-class, runtime-adjacent surface (2026-07-07) — moved
  `Adaptors/Inference/Bindings` -> `Mila/Bindings` (peer of `Src`/`Adaptors`); `add_subdirectory`
  hoisted from `Adaptors/CMakeLists.txt` to `Mila/CMakeLists.txt` under `MILA_ENABLE_PYTHON_BINDINGS`.
  It is consumer-blind (module `Mila.Bindings`; no wire/chat) and has two consumers — MIS and the
  HuggingFace parity/converter tooling — so it belongs beside the runtime, not under MIS.
- [ ] Neutral binding output location. `Bindings/CMakeLists.txt:49` still copies `mila.pyd` into
  `Mila/Adaptors/Inference/Server` via a POST_BUILD step (now an absolute-from-source path since the
  binding no longer sits beside the server). That is the one consumer-specific reach a consumer-blind
  surface should not make. Build to a neutral location (build output or an install/dist dir) that MIS
  and the parity tooling both pull from (PYTHONPATH or install), and drop the reach into Server.
