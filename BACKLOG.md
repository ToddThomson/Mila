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
- [ ] **[deferred, pairs with FP16 removal]** `OperationTraits<GeluOp, Cuda, BF16>` is a **poisoned dispatch entry** — it advertises `CudaGeluOp<BF16>`, but the kernel `cuda_gelu_impl` is constrained to `float || half` ([CudaGeluOp.Dispatch.ixx:57](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Gelu/CudaGeluOp.Dispatch.ixx)), so instantiating `Gelu<Cuda, BF16>` is a hard compile error (surfaced by the Test Suite Revival Gelu typed test). CUDA GELU is FP32-only in practice — an FP16-era kernel never migrated to BF16, and no BF16 GELU path exists (Llama uses SwiGLU; GPT-2 GELU runs FP32). Fix: remove the BF16 traits row (FP32-only is honest) **or** add a BF16 kernel specialization. Audit other ops' traits rows for the same traits-vs-kernel desync during the FP16 sweep. **Confirmed further instances (same desync, all surfaced by Test Suite Revival CUDA tests):** `OperationTraits<MultiHeadAttentionOp, Cuda, BF16>` ([OperationTraits.Cuda.ixx:226](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/OperationTraits.Cuda.ixx)) vs `CudaMhaOp.Dispatch.ixx:24`; `OperationTraits<SoftmaxOp, Cuda, BF16>` ([OperationTraits.Cuda.ixx:194](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/OperationTraits.Cuda.ixx)) vs `CudaSoftmaxOp.ixx:48` + `Softmax.cuh`; `OperationTraits<LpeOp, Cuda, BF16>` ([OperationTraits.Cuda.ixx:258](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/OperationTraits.Cuda.ixx)) vs `CudaLpeOp.Dispatch.ixx:27` — each advertises a BF16 op whose kernel is `float || half` only. All three are GPT-2 lineage / not on a BF16 inference path (MHA: GPT-2 attention, FP32 weights, Llama uses GQA; LPE: GPT-2 positional, Llama uses RoPE; Softmax: Llama fuses attention softmax inside GQA). Resolution is the same: drop the BF16 rows (FP32-only is honest). Each test's precision sweep lists FP32 only (one-entry `Types<Fp32Precision>`) to compile green without touching the poisoned rows; adding a `Bf16Precision` tag re-runs the suite once a BF16 kernel exists. Root-cause fix is the capability predicate under "Dispatch error UX" below
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
- [ ] **[gate]** Wire the suite into CI as the anti-rot ratchet — build on the `MILA_ENABLE_CUDA=OFF` CPU-only gate so a future API churn fails the build instead of silently re-commenting coverage. This is the deliverable that keeps the revival alive
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
oracle now exists (Clang 21 + CUDA 13.3 + gcc-15 host); GCC 16 and the dev-container build
remain to be validated. Surface: 287 `.ixx` units, ~1,810 `import` lines, ~1,419 `#include`
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

Docker image publish is optional and only if the runtime image stays a beta deliverable —
a release-tagged GHCR push is a natural CI-on-tag job but equally a local `docker build &&
docker push`; automation-of-convenience, not a gate.

---

## Project Hygiene & Contributor Readiness

A beta is a trust signal; these items are about the project not contradicting itself or
wasting a newcomer's first hour.

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
- [ ] **Dispatch error UX** — compile-time dispatch makes the compiler error the *user interface* for an unsupported `(Op, Device, Precision, Policy)` combination, and today it is hostile: MSVC emits `C7602` constraint cascades with no semantic cause, and a present-but-broken specialization is worse than a missing one (see the poisoned BF16 GELU row under Consolidation). Pattern recorded in [OperationDispatch.md](Mila/Specifications/OperationDispatch.md) §12: **(A)** a friendly `static_assert(always_false, ...)` on the `OperationTraits` primary so a missing specialization reads as a sentence, not an incomplete-type puzzle; **(B)** a single authoritative `OperationSupported<...>` bool predicate shared by the kernel `requires`-clause, the traits row, and a component-level `static_assert` (safe in the class body — it probes no op members, unlike the §5 member-probing concept), so the table cannot advertise what the kernel rejects and the error surfaces at the user's instantiation line; **(C)** named kernel concepts. `Src` work spanning the dispatch core; adopt incrementally, paired with the FP16 removal. Contributor-readiness gate — a newcomer's first unsupported instantiation should not produce 200 lines of MSVC noise
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
  from `TPrecision`?** `TPrecision` today effectively means "activations = BF16," and BF16-primacy is
  baked in around it (the FP32 gradient boundary, the KV-cache dtype, the up-convert points in
  component orchestration — residual/norm/softmax run higher precision). Native FP4 compute adds
  FP4<->BF16<->FP32 transitions where there is only BF16<->FP32 now. Decide: redefine `TPrecision`, add
  a new axis, or add an activation-quant policy mirroring `TWeightQuantization`.
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
  but tools are still *advertised* the wrong way: `Chat::clearHistory` ([Chat.ixx](Mila/Samples/Chat/Src/Chat.ixx))
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
  serializer (Samples/Chat is free to edit), route the Gemma branch of `clearHistory` to it, keep the Llama
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

Still open to close the milestone:
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
