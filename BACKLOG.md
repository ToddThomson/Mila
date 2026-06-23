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

- [ ] **Couple parameter initialization to runtime mode** — `BuildContext::initialize_parameters` defaults `true` independently of `RuntimeMode`, so an inference-mode build can silently run (then immediately discard) full parameter initialization. `GptModel::fromPretrained` hit exactly this; fixed for now by passing `false` explicitly (matching `LlamaModel`), but that relies on every load path remembering the third argument. Structural fix: derive the `initialize_parameters` default from `RuntimeMode` (`Inference` => no init unless explicitly requested) so no load path can regress by omitting the flag. The per-device init wiring (TensorOps `zero`/`fill`/`xavier`/`fill_normal`) is now live and gated per component via `shouldInitializeParameters()`; this default-coupling remains as hardening so a future load path cannot silently re-run (and discard) init

- [ ] **[gate — Bard CPU stack] FFN consolidation: de-polymorphize `MLP`** — `MLP::addActivation`'s runtime `switch` force-instantiates the `Swiglu` case, so `MLP<Cpu>` (hence `GptBlock<Cpu>` / `GptTransformer<Cpu>`) cannot compile — the exact CPU transformer stack the Test/Training Revival needs; the old commented `GptBlock.Cpu`/`GptTransformer.Cpu` could never have compiled either. Strip the activation polymorphism: `MLP<…, ActivationType TActivation = Gelu>` holds a compile-time `Activation<…, TActivation>` (no `mlp_activation_impl` / `std::function` bridge / SwiGLU branch / `fc1` doubling / dead LayerNorm). Relocate `Swiglu` `Activations/` -> `FFN/Swiglu/` and `MLP` -> `FFN/MLP/` (+ mirror the tests); strip the per-step debug `synchronize()` from `GptBlock`/`LlamaBlock` `forward()`/`backward()` (inference `prefill`/`decode` already run sync-free — single-stream ordering + the caller's host-read boundary sync suffice). Full design + decisions: [FfnAndMoE.md](Mila/Specifications/FfnAndMoE.md). The reusable `GatedMLP` + grouped `MoeOp` foundation it specifies is a Future Direction (Architecture / MoE), not a 0.20 gate
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

- [ ] **[deferred, milestone TBD]** Token sampling (temperature / top-k / top-p) — `OperationTraits<SamplingOp, Cuda, FP32>` and `<…, BF16>` specializations; `TokenSampler` component + `CudaSamplingOp` per `Specifications/TokenSampling.md`. **Pushed out of Consolidation** (feature freeze — no new features); milestone undecided, to be assigned later. Not a 0.20 gate — greedy decode is already validated, so this is additive
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

- [~] **[gate]** Re-green the authored component / operation / tensor / tokenizer suites against the current API — re-enable the `#`-commented files in `Tests/CMakeLists.txt`. Bucket 1 (uncomment + fix trivially broken) + bucket 2 (translate to the post-refactor surface: `OperationTraits` dispatch, the `Operation` base-class collapse, the precision axes, the Consolidation lifecycle fix). _Done (per-component completion tracked here; the consolidated CHANGELOG entry is written when this milestone closes, not per component):_ `Component`/`ComponentConfig` (base contract); `Gelu` (stateless leaf reference); `Linear` (`LinearConfig` + `Linear.Cpu` + `Linear.Cuda` — first leaf-with-parameters reference and the first `TYPED_TEST` precision sweep, FP32+BF16); `RmsNorm` (config + `.Cuda` FP32+BF16, CUDA-only); `LayerNorm` (config + `.Cpu` FP32 backward-numeric + `.Cuda` FP32); `Swiglu` (config + `.Cuda` FP32+BF16, CUDA-only); `Residual` (config + `.Cpu` + `.Cuda` FP32+BF16 — binary stateless leaf); `TokenEmbedding` (config + `.Cuda` FP32+BF16, CUDA-only — gather + prefill/decode shape regime); `Lpe` (config + `.Cpu` FP32 backward-scatter + `.Cuda` FP32-only, gather + positional add + decode — GPT-2 lineage, no BF16 kernel; see poisoned-row note); `Rope` (config + `.Cuda` FP32+BF16, CUDA-only — forward/prefill/decode + inverse-rotation backward round-trip); `Softmax` (config + `.Cpu` FP32 forward+backward-numeric + `.Cuda` FP32-only forward — no BF16 kernel; see poisoned-row note); `MultiHeadAttention` (config + `.Cpu` FP32 causal-attention forward-numeric + `.Cuda` FP32-only forward via prefill + decode-after-prefill numeric — GPT-2 lineage, no BF16 kernel; see poisoned-row note below); `GroupedQueryAttention` (config + **surface-only** `.Cuda` FP32+BF16 — construction/accessors/build-validation/stateless/type; numerics deferred, see GQA bug below); `MLP` (`MLPConfig` + `.Cpu` composed-forward numeric + finite-difference backward + `.Cuda` surface/shape/finiteness/gradient-flow — composite fc1->Gelu->fc2; the CUDA composed numeric reference is left to `.Cpu` and the leaf numerics to `Linear.Cuda`/`Gelu.Cuda`); `GptBlock` (`GptBlock.Config` + `.Cpu` + `.Cuda` — GPT-2 pre-LN block composite, gradient-flow asserted; the composed finite-difference probe is deferred to the gradient-check archetype); `GptTransformer` (`Gpt.Presets` + `.Cpu` + `.Cuda` — GPT-2 network surface + finite logits, with `.Cuda` also asserting training-mode backward, exercised by Bard); `LlamaTransformer` (`Llama.Config` + `Llama.Presets` + **surface-only** `.Cuda` — construction/build/mode/components/type + forward-shape; numerics deferred per the GQA standalone-forward stub, and the retired `LlamaTransformer::fromPretrained` cases dropped). With these the concrete **component-class** set is fully re-greened to the methodology (build green); only `SoftmaxCrossEntropy` (the loss component) remains, parked for the loss-on-device milestone. **Bugs surfaced (filed below, fixed outside the methodology session):** `loadParameter` recursion (Linear fixed; RmsNorm/LayerNorm/TokenEmbedding/Lpe filed), Swiglu BF16 backward dtype, non-quantized cuBLASLt GEMM, `Linear::forward` dead fast-path. **Greening pass 2026-06-17 (0.20.0-alpha.6+64):** ran the re-greened suites failure-by-failure to actual green. Landed: CharTokenizer empty-corpus test fixes; CPU TensorOps math revived (C1116 root-caused to `<execution>`, FIXED below) -> all `Math.Cpu` ops green; `getGradients()` inference contract unified across the tree to return-empty (was throw on CompositeComponent/LayerNorm/TokenEmbedding) + the throw-asserting tests (LayerNorm Cpu/Cuda, CompositeComponent, MlpNetwork) flipped to assert-empty + `MockChild` made mode-aware; `Softmax.Cpu` backward test corrected (pass Y, zero accumulation target); `TokenEmbedding.Cuda` BF16 test dim made `% 8`-clean; un-skipped the `Linear.Cuda` forward/backward numeric cases (cuBLASLt bias-epilogue fixed) and the `GptBlock.Cpu` composed-backward sentinel (CPU math live). **Remaining red: exactly 3 tests, all unimplemented-backward (filed below):** CUDA Softmax backward stub, BF16 Swiglu backward dtype, and the MHA-CPU-backward-suspect GptBlock composed gradient check
- [~] **Tensor suite (non-component slice)** — re-green + complete `Tensor.ixx` core coverage to the methodology, tracked by the per-member matrix in [Testing.Tensors.md](Mila/Specifications/Testing.Tensors.md). Establishes the **value-type / god-module archetype** (area-split instead of one-file-per-module; dtype `TYPED_TEST` sweep only where behavior varies; memory-resource = file split). Core `Tensor.ixx` first; the wider `Tensors/` tree (`TensorBuffer`, `TensorDataType*` maps, `TensorOps/*`, `Partitioning`, `Serialization`) is a follow-on slice. _Core `Tensor.ixx` complete (awaiting VS2026 build):_ all eight area files (`Constructors`, `DataAccess`, `DataPointers`, `Identity`, `Io`, `MemoryProperties`, `Properties`, `ShapeTransform`) + `.Cuda.cpp` companions, on the `Mila::Tests::Dnn::Tensors` namespace with the device axis as a file split (every inline `#ifdef MILA_HAS_CUDA` removed). New coverage: `elementSize`/`getStorageSize`, the shape-transform area (no file before), `item()` + scalar negatives, and the device-tensor host-only SFINAE contract. The **value-type / god-module archetype** is now documented in [Testing.md](Mila/Specifications/Testing.md) §1. _TensorOps slice:_ `zero` done (`Zero.Cpu.cpp`/`Zero.Cuda.cpp`, methodology); `Fill`/`Math` re-greened (namespace + header, promoted to Section 1); `Transfer` namespace+header re-greened but **device-split still pending** (cross-device shared fixture -> `TensorOps.Transfer.Cuda.cpp` follow-up); `Structural`(`split`) missing -> backfill; `Random`(`fill_normal`/`fill_uniform`/`xavier`) deferred to Training Revival (training-init, CUDA FP32-only bug parked there). _Next (follow-on slice):_ the rest of the wider `Tensors/` tree — `TensorBuffer`, `TensorDataType*` maps, `Partitioning`, `Serialization`
- [ ] Backfill coverage for the inference-drought features the old suite never had — load-time quantization (`PerChannelFp8`/`PerGroupFp4`, the decode matvec kernels), `OperationTraits` dispatch, the Llama path (RmsNorm/SwiGLU/GQA/RoPE components, `LlamaModel::fromPretrained`). Genuinely new, not recovery. **This is also the *only* legitimate op-layer test** — the `CudaLinearOp` quantization white-box (scales == host absmax, FP4 nibble packing, exactly-representable round-trip): the surface unreachable through the public component, per the reachability rule in [Testing.md](Mila/Specifications/Testing.md) §1. Scope it to those unreachable assertions only; the forward/decode numerics are component-test territory (a black-box wiring proof on the quantized component, not an op-numeric mirror)
- [~] **[net-new]** Build the **gradient-check archetype** — the authored suite was *forward-only* (inference validated forward passes against HuggingFace), so every component `backward()` the training samples drive has zero coverage. Add a finite-difference gradient verifier (perturb input/parameter by ±eps, compare numeric vs analytic gradient within tolerance) as a reusable test fixture, then a `Backward_MatchesNumericGradient` case per training component. A few backward-numeric cases already exist (`LayerNorm.Cpu`, `Softmax.Cpu`, `Lpe.Cpu` scatter) — generalize them into the shared archetype rather than re-deriving per file. This is the largest net-new category and the precondition for Training Revival's convergence oracle. Document the archetype in [Testing.md](Mila/Specifications/Testing.md) alongside the value-type / component archetypes. **DONE (shared fixture + reference applications, awaiting VS2026 build):** [Common/GradientCheck.h](Mila/Tests/Common/GradientCheck.h) — signature-agnostic black-box verifier (`centralDifferenceGradient` over the scalar loss `L = sum(output*g)`, `expectGradientsClose` with combined abs/rel tolerance; the forward seam is a caller lambda so the helper absorbs the non-uniform `forward(input)->output&` vs `forward(input,output)` signatures), wired via `target_include_directories(MilaTests PRIVATE Tests-root)` so it includes as `"Common/GradientCheck.h"`. Reference applications: `Gelu.Cpu.cpp` `Backward_MatchesNumericGradient` (stateless input grad) and `LayerNorm.Cpu.cpp` `Backward_MatchesNumericGradient` (input + weight + bias grads via one `evaluate` lambda). Archetype documented in Testing.md §1; the pre-existing analytic cases kept as an independent second oracle. **REMAINING:** fan out `Backward_MatchesNumericGradient` across the rest of the training components (Linear, MLP, Residual, GptBlock); use it to **isolate the MHA CPU backward** (add the numeric check to `MultiHeadAttention.Cpu.cpp` — the prime suspect behind the red `GptBlock.Cpu` composed-gradient sentinel); tune the FP32 eps/tolerances against the first real run
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
- [ ] **[bug]** GptBlock composed-backward finite-difference gradient check fails — `GptBlock.Cpu.cpp` `Backward_InputGradientMatchesFiniteDifference` (central finite diff of the block's own forward vs `GptBlock::backward`) does not match within tolerance; all structural/forward/backward-flow cases pass and the gradient is finite + non-zero, so backward *runs* but is numerically off somewhere in the stack. Prime suspect: the **MHA CPU backward**, which is only shape-tested (`MultiHeadAttention.Cpu.cpp` `Backward_ReturnsInputShapedGradient`), never numerically. Other candidates: the `Residual` pair-backward accumulation in `GptBlock::backward`, `LayerNorm` backward composition, or FP32 finite-diff noise through softmax+LayerNorm at the chosen tolerance. Triage MHA-first: add a finite-difference numeric gradient check to `MultiHeadAttention.Cpu.cpp` to isolate, then re-check GptBlock. Case is `GTEST_SKIP`'d (kept in place) so the suite stays green; re-enable on fix. Surfaced bringing the Bard GPT-2 stack online. **Update (2026-06-16):** prime suspect revised — CPU `TensorOps::add` is a no-op (CPU math gated off pending the MSVC C1116 blocker above), so `GptBlock::backward`'s residual-gradient accumulation (`add` into `d_res1_accum_`/`d_input_`) does nothing on CPU, which alone breaks the composed gradient. The identical bug on CUDA was the Bard bigram-floor regression (now fixed). This sentinel cannot pass until CPU math is restored; revisit after the C1116 fix, then re-check whether MHA CPU backward also needs numeric coverage. **Update (2026-06-17): the C1116 blocker is RESOLVED (CPU math live, see FIXED entry above) — the residual-gradient `add` now computes on CPU. Re-run this sentinel: if it still fails, the remaining suspect is MHA CPU backward (only shape-tested), which then needs a finite-difference numeric check to isolate**
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
gradient flow, train-from-scratch init. The revived tests are the oracle: a sample "converges" only
when its test says so. Several deferred Consolidation items (AdamW debug instrumentation, the CUDA
`fill_normal` FP32-only gap) fold into this milestone. **Scope is GPT-2 / MLP training only** —
Llama 3.1/3.2 fine-tuning is explicitly out of this release, remaining a Future Direction.

- [~] **(lead — timeboxed spike)** Revive **MNIST + its tests** against the current API — MNIST is the MLP (simpler than Bard's GPT-2/BPE/transformer surface), so it is the cheapest representative slice. Re-enable the sample (`Mila/Samples/CMakeLists.txt:3`) and its tests; pass/fail = builds, runs, trains to target accuracy, tests green. **Sample + spine tests GREEN (2026-06-15): sample builds and trains to ~97.9% test accuracy; spine tests (Linear/Gelu/Network/AdamW/DataLoader) landed in alpha.6+58.** Remaining for full spike closure: the end-to-end convergence oracle + `MnistDataLoader` contract test (net-new items below). The MNIST surface is already partly re-aligned (`MnistClassifier`/`MnistDataLoader` use the component-owned forward/backward API, `BuildContext`/`RuntimeMode::Training`, `setTrainingMode`, `Network::createOptimizer`), so the spike measures the MNIST spine specifically: `Network` + `AdamW` re-green (Test Suite Revival), Linear/Gelu gradient-check (the net-new archetype), the `MnistDataLoader` contract test, and the end-to-end convergence oracle below. Measures all three revival buckets at once and sets the milestone dates on evidence rather than the day-or-3 estimate. **Do this first**
- [~] Re-enable MNIST + Bard in the build — flip the `FIXME: Re-enable after alpha.5 completed` triggers (`Mila/Samples/CMakeLists.txt:3-4`) **staged: MNIST first** (its spine + tests green from the lead spike), **Bard second** (once the GptTransformer/tokenizer/`TokenSequenceLoader` spine is green) — and add the Samples build to CI (pairs with the Project Hygiene "Samples build to CI" item). **MNIST sample re-enabled at source** (`Samples/CMakeLists.txt` now `add_subdirectory(MNIST)` gated under `MILA_ENABLE_CUDA`, mirroring Chat — the sample links `CUDA::cudart` and instantiates the `CudaPinnedMemoryResource` path unconditionally, so it is CUDA-only until the CPU-only build-coherence refactor); `MnistClassifier::onBuilding` corrected to build each child with `build_config.withShape(...)` (a single shared `build_config` threw in `Linear::onBuilding` for fc2/fc_output — the spine-test gotcha). **MNIST validated 2026-06-15: builds green (VS2026/CUDA) and trains from scratch FP32 to ~97.9% peak test accuracy (99.85% train) over 20 epochs — full spine (forward chain, backward gradient flow, AdamW step, train-from-scratch init) exercised end-to-end; mild late-epoch overfit (test loss U-turns at epoch ~9), expected with no dropout.** **Bard sample re-enabled at source 2026-06-16** (`Samples/CMakeLists.txt` now `add_subdirectory(Bard)` under the same `MILA_ENABLE_CUDA` gate; `Bard/CMakeLists.txt` cleaned of stale commented module refs and switched to the `MILA_DATASETS_DIR` absolute-path macro like MNIST). Sample drift fixed to current API, mirroring MNIST's bring-up: dropped the retired `ComputePrecision::Policy precisionPolicy` field + `--precision-policy` CLI from `BardConfig`/`Bard.cpp`, and pointed the data dir at the real `Data/Datasets/Shakespeare` layout. `BardTrainer.ixx` already used the current surface (`GptConfig` builders, `GptTransformer`, `TokenSequenceLoader`, `createOptimizer<AdamW>`, `BuildContext`/`RuntimeMode::Training`, `setTrainingMode`). Bard's sample-local `CharLMTransformer`/`CharLMDataLoader` stay retired-in-place; the live Bard sample compiles only `BardTrainer.ixx` + `BardConfig.ixx` and leans on the library `GptTransformer` + `TokenSequenceLoader`. **Bard VALIDATED 2026-06-16: builds green (VS2026/CUDA) and trains FP32 from scratch to perplexity <3 / loss ~1.09 by epoch 17 with coherent Shakespeare-structured text, after fixing three latent CUDA-training-backward bugs surfaced one at a time (cuBLASLt bias epilogue -> inverted attention eval-guard -> TensorOps math no-op, all FIXED above).** Remaining: the Bard end-to-end convergence oracle (the CPU oracle is gated behind the GptBlock CPU finite-difference failure / CPU-math C1116 blocker, but Bard trains on CUDA so this is not a hard blocker for the sample)
- [ ] Re-enable + re-align the AdamW path — `AdamW.Cuda.cpp` / `AdamW.Cpu.cpp` (disabled in `Tests/CMakeLists.txt:190-191`); resolve the deferred AdamW debug instrumentation (strip-vs-gate the `CudaAdamW.cu` printf guards + `CudaAdamWOptimizer.ixx:270`) in the same pass
- [ ] Fix the CUDA `fill_normal`/`fill_uniform` FP32-only gap (the deferred Consolidation training-only item) — it corrupts BF16 train-from-scratch init; the CUDA dtype counterpart to the `CpuTensorOps.Random` backend. Pair the fix with the **init-at-precision test**: revive `Tensor.Initializers.cpp` (+ the deferred `TensorOps/Random` slice) as a `TYPED_TEST` precision sweep (FP32 **and** BF16), turning this latent corruption into a red test rather than a silent one
- [ ] **[net-new]** **End-to-end convergence oracle** — a per-sample integration test that builds the tiny model, runs a fixed step budget on a handful of batches, and asserts the loss strictly decreases (and, for MNIST, that train accuracy rises). This is the literal pass/fail the milestone exit names — "a sample converges only when its test says so." MNIST first (MLP), Bard second (GPT-2). Keep the step budget small so it is CPU-runnable in the `MILA_ENABLE_CUDA=OFF` CI gate
- [ ] **[net-new]** **Optimizer step-convergence test** — beyond `AdamW.*` config/mechanics re-green, add a "minimizes a known convex objective in N steps" case proving the update direction + bias-correction are correct, not just that `step()` runs. Foundational to trusting both samples' training loops
- [~] **[net-new]** **Concrete data-loader contract tests** — the base `DataLoader` test re-greens under Test Suite Revival, but the two concrete loaders are untested (the `MnistDataLoader` lives in the sample with no test at all): `MnistDataLoader` (pixel normalization to [0,1], one-hot target encoding, shuffle on reset, batch shapes, IDX magic-number validation) and `TokenSequenceLoader` (INT32 input/target offset-by-one, pad handling, `numBatches`). MNIST loader with the MNIST spike, token loader with the Bard slice. **`TokenSequenceLoader.cpp` contract test re-enabled 2026-06-16 with the Bard slice** (Section 2 of `Tests/CMakeLists.txt`; CPU cases ride the `MILA_ENABLE_CUDA=OFF` gate, CUDA-pinned cases stay `#ifdef`-guarded): construction validation, batch iteration, reset, target-is-input-shifted, tensor shapes, multi-epoch, threading stress. Remaining: the `MnistDataLoader` contract test
- [ ] **[net-new]** **TrainingMode / RuntimeMode behavior coverage** — the two-axis lifecycle (`BuildContext(RuntimeMode::Training)`, `setTrainingMode(Eval/Normal)`, `isInferenceMode`) gates gradient-buffer allocation and is *why* the component tests were disabled (Consolidation bucket E linchpin). Add explicit cases asserting build-mode and runtime-mode transitions allocate/skip gradients correctly, so the lifecycle fix has a regression guard
- [ ] Revive the loss + backward path — CrossEntropy / SoftmaxCrossEntropy components and tests (`Mila/Tests/Dnn/Components/Losses/*` exist, commented) and the backward-pass stubs (Consolidation bucket D). **Started (alpha.6+68):** a pattern-conforming [CudaSoftmaxCrossEntropyOp.Dispatch.ixx](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Loss/CudaSoftmaxCrossEntropyOp.Dispatch.ixx) was added (the `cuda_softmax_crossentropy_impl<TComputeType>` FP32/BF16 dispatch struct) and put in the build, but is **not wired in yet** (no primary `Compute.CudaSoftmaxCrossEntropyOp` unit imports `:Dispatch`; the op does not call it). Loose ends to resolve when wiring: (1) **`always_false_v` is undefined/unimported** — used in the primary-template `static_assert` but defined nowhere in the tree (only `always_false` appears, as prose in `OperationDispatch.md:372`); needs a `template<typename> inline constexpr bool always_false_v = false;` helper imported here, or switch to the sibling idiom; (2) **pattern divergence** — the existing CUDA ops (`CudaSoftmaxOp.ixx:48`, etc.) use a *forward-declared* primary (`struct cuda_X_impl;`, no body) so an unsupported type fails as an incomplete type, whereas the new file gives the primary a defined body + `static_assert`; the spec (`OperationDispatch.md:372`) actually prescribes the `static_assert` form, so spec and existing code disagree — pick one as canonical; (3) the **BF16 specialization is a no-op stub** (kernel calls commented, `half`-typed) — same FP16/BF16 desync as the poisoned-row / FP16-removal bucket, resolve consistently (drop BF16 row vs implement); (4) the `probs` out-params are commented out throughout — decide whether the fused op materializes probabilities
- [ ] Revive the `Dropout` component — parked at `Dev/Components/Regularization/` (`Dropout.ixx` + `DropoutConfig.ixx`), moved out of `Src` during the Consolidation dispatch close-out so the legacy `OperationRegistry`/`UnaryOperation` cluster could be deleted. It is pre-refactor on every axis: three-axis `Component<TDeviceType, TInput, TOutput>` + `ModuleBase`, registry-string op creation (`"Cpu::DropoutOp"`/`"Cuda::DropoutOp"`), and no concrete `DropoutOp` exists. Re-authoring is net-new training-only work (inference never uses dropout): write `CpuDropoutOp`/`CudaDropoutOp`, add a `DropoutOp` `OperationTraits` specialization per backend, rewrite the component to the two-axis `Component<TDeviceType, TPrecision>` surface, restore it to `Mila/CMakeLists.txt` + `Mila.ixx`, and add its tests. Belongs here because the mask/backward path is exercised only by the training revival
- [ ] **ProgressReporter mechanism** — design the cross-cutting progress facility for long-lived ops (the `BpeVocabulary` training `\r` progress at `:600`/`:613`, plus `PretrainedReader` load and load-time quantization are candidates). Injected per-operation (on the op's config, **not** a global facade — progress is scoped to one call, unlike the process-wide logger), null default, library owns throttling, cancellation first-class (`bool` return or `std::stop_token`), documented threading contract. Mirrors the Logging subsystem's *shape* but is a separate concern (progress = transient/overwrite-in-place; logging = append-only events). The Consolidation debug strip leaves the `BpeVocabulary` training progress in place as living training-path code — it migrates here, it is not deleted
- [ ] Validation — MNIST trains to its target accuracy; Bard generates coherent text; train-from-scratch validated at the precisions the samples use; the AdamW / loss / training-path tests green and CI-gated

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

- [ ] Oracle + ratchet — turn Doxygen's own warnings into both the worklist and the anti-rot gate. `Mila/Docs/Doxyfile.in` sets no `WARN_*` knobs today (all default): enable `WARN_IF_DOC_ERROR` + `WARN_NO_PARAMDOC` to mechanically generate the Tier 2 candidate list, then once the count reaches zero flip `WARN_AS_ERROR = FAIL_ON_WARNINGS_PRINT` in the docs job so doc drift fails the build instead of silently re-accumulating (the docs analogue of the Test Suite Revival test-CI ratchet). Highest-leverage item — it both drives and locks the tiers below, and gives a shrinking warning count as the definition of "done"
- [ ] Tier 0 — non-ASCII / mojibake in comments: UTF-8 corruption (em-dashes rendered `�`, e.g. throughout `Comonent.TrainingMode.ixx`) violates the ASCII-only-in-comments rule; a non-ASCII scan over comment lines is a complete, scriptable worklist with no judgment. Fold in the stray misspelled filename `Comonent.TrainingMode.ixx` -> `Component.TrainingMode.ixx` (file rename + CMake reference, beyond the comment edit)
- [ ] Tier 1 — `@file` rename drift: 34 files whose `@file` tag does not match the filename (e.g. `RocmDevice.ixx` tagged `VulkanDevice.ixx`, `CudaMhaOp.ixx` tagged `CudaAttentionOp.ixx`, `Lpe.ixx` tagged `Gpt2Encoder.ixx`). The correct value is `basename` — fully scriptable, no judgment
- [ ] Tier 2 — `@param`/`@tparam` name mismatches: documented names no longer in the signature. Mechanical and high-confidence, but signatures span lines, so emit a candidate list for review before batch-fixing — the candidate list is the `WARN_IF_DOC_ERROR` output from the Oracle bullet, not a hand-built grep
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
- [ ] Fully decouple the docs job from the build — PARTIALLY DONE (docs is its own workflow running `cmake --build --target docs`, compiling nothing). Remaining: the job still runs CMake configure inside the CUDA container because the `docs` target is CMake-registered and the root configure requires the CUDA toolkit. Full decoupling = driving Doxygen without a CUDA-dependent configure (standalone Doxyfile, or a CUDA-free docs-only configure path)
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
- [ ] Debug instrumentation fully gated or removed — substantially done by the Consolidation debug-instrumentation strip (kernel `printf`/anomaly guards removed; the BPE tokenizer warning + vocab-load info routed to `Logging::Logger`, the encode timer and progress prints deleted). Training-path instrumentation is intentionally NOT stripped — it is deferred to its owning milestone (the AdamW debug item above; the `BpeVocabulary` training progress -> ProgressReporter under Training Revival)
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

## Gemma 4 — Dense Chassis (SWA + Dual-RoPE Foundation)

Committed milestone (promoted 2026-06-19 from the former "Sliding Window Attention — Future Direction"
section, which this absorbs). Target: **Gemma 4 12B Unified** (dense, text) validated token-for-token
vs HuggingFace. New `Components/Transformers/Gemma` family modeled on the Llama work, **not** a bent
`LlamaBlock`. Full design + confirmed config + the template-vs-runtime decision table:
[Gemma.md](Mila/Specifications/Gemma.md); design decisions recorded in [[project_gemma_chassis_design]],
model target in [[project_gemma4_moe_target]]. Governing principle: **template axes for types/layouts,
runtime config for arithmetic.** Built tests-first (MNIST/Bard methodology) on the now-clean
compact-NKV GQA op (alpha.6+69). Tasks below are the Gemma.md §9 foundation sequence, in dependency
order.

- [~] **[gate] Step 0 — explicit `head_dim` in the new `GemmaConfig` (NOT the leaf configs).** Blast
  radius confirmed by reading the configs + op (2026-06-19): **the leaf configs already decouple.**
  `GqaConfig` ([GroupedQueryAttention.Config.ixx](Mila/Src/Dnn/Components/Attention/GQA/GroupedQueryAttention.Config.ixx))
  and `RopeConfig` ([Rope.Config.ixx](Mila/Src/Dnn/Components/Encodings/Rope/Rope.Config.ixx)) both take
  the **Q-projection width** (`num_heads*head_dim`) as their first ctor arg (documented as such) and
  derive `head_dim = width/num_heads`; fed `num_heads*head_dim` they are correct for Gemma with **no
  change** (`GqaConfig(8192,16,1).getHeadDim()==512` today). `CudaGqaOp` reads `HS_=config_.getHeadDim()`
  and packs via `(getNumHeads()+2*getNumKvHeads())*getHeadDim()` ([CudaGqaOp.ixx:209,378](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx))
  — it does NOT re-derive head_dim from a residual dim, so it picks up Q-width correctly with zero op
  change. The coincidence is baked in ONLY at the model/block level: `LlamaConfig` stores `embedding_dim_`
  (residual) and `withNumHeads` validates `embedding_dim % num_heads == 0` ([Llama.Config.ixx:72](Mila/Src/Dnn/Components/Transformers/LlaMa/Llama.Config.ixx)),
  hard-deriving `head_dim=embedding_dim/num_heads` — for Gemma that passes silently (3840%16==0) but
  yields the wrong 240. **So Step 0 = `GemmaConfig` carries `head_dim` explicit + separate from
  `embedding_dim` (residual); leaf configs untouched (Option A — minimal, no risk to validated Llama
  code).** The `GemmaBlock` (Step 5) feeds `num_heads*head_dim` into `GqaConfig`/`RopeConfig` and wires
  **non-square o_proj** `Linear(num_heads*head_dim, embedding_dim)` (4096->3840 sliding / 8192->3840
  global; Llama's square o_proj is the `num_heads*head_dim==embedding_dim` special case). Tests-first:
  `GemmaConfig` sliding (`embedding_dim=3840,num_heads=16,num_kv_heads=8,head_dim=256`) + global
  (`head_dim=512,num_kv_heads=1`) asserting derived Q-width, QKV packing dim, o_proj shape — no kernel.
  **LANDED 2026-06-19 (awaiting VS2026 build):** standalone module `Dnn.Components.GemmaConfig`
  ([Gemma.Config.ixx](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.Config.ixx)) with explicit
  `withHeadDim` (default 0 = derive `embedding_dim/num_heads`, Llama-compatible fallback), decoupled
  derived geometry (`getQProjectionWidth`/`getKVProjectionWidth`/`getPackedQKVWidth`), validate() that
  accepts `num_heads*head_dim != embedding_dim` (drops the Llama `embedding_dim % num_heads` check),
  metadata round-trip. Test [Gemma.Config.cpp](Mila/Tests/Dnn/Components/Transformers/Gemma/Gemma.Config.cpp)
  (13 cases incl. the 240-vs-256 decoupling, sliding+global derived widths, Q-width!=residual validate,
  metadata head_dim preservation). Wired: Mila/CMakeLists.txt module source, Mila.ixx umbrella export,
  Tests/CMakeLists.txt Section 1. Leaf `GqaConfig`/`RopeConfig` untouched (Option A). NEXT: Step 1
  (global K=V geometry -> `TAttentionKind`).
- [~] **Step 1 — global-layer geometry in `GemmaConfig` (config only; NO op/traits/`TAttentionKind`).**
  Walked back the `TAttentionKind`-as-op-axis plan 2026-06-19 after reading `CudaGqaOp`: the global
  geometry (`head_dim` 512, single KV head, K=V) **already rides the existing GQA op**. `CudaGqaOp`
  derives every dim from config (`HS_=getHeadDim()`, `NKV_=getNumKvHeads()`, `GS_=NH/NKV`) and its live
  `prefill`/`decode` take *separate* q/k/v pointers — so `head_dim 512` is just config, single KV head
  is MQA (`NKV=1`, already supported), and K=V is the caller aliasing the V pointer to K
  (`prefill(q,k,/*v=*/k,...)` -> `kvcache_write_kv` writes K into both caches). The
  `(num_heads+2*num_kv_heads)*head_dim` packing lives only in the stubbed standalone `forward()` +
  `validateConcatenatedQKVShape`, NOT the live path, so K=V packing is a *block* concern. **Result: NO
  `TAttentionKind` policy, NO new `OperationType`, NO new traits row, NO new op class, NO new template
  param on `GroupedQueryAttention`, ZERO change to the Llama path.** The local/global distinction is a
  `GemmaBlock` wiring selector (the two instantiations differ in `qkv_proj` width / V split / `GqaConfig`),
  used only for `if constexpr` block wiring — never reaching the op/traits. So Step 1 = extend
  `GemmaConfig` with `global_head_dim` (512), `num_global_kv_heads` (1), `key_equals_value` (true) +
  a K=V packed-width helper `(num_heads + num_kv_heads)*head_dim` (8704), tests-first. Block wiring +
  V-aliasing deferred to Step 5. **Runtime check deferred to Step 5:** confirm the hand-written GQA
  kernels (`permute_q_compact`, prefill/decode softmax, unpermute) carry no static `head_dim`
  assumption that breaks at 512 (Llama only runs 128/256; cuBLASLt GEMMs handle 512). See Gemma.md §5/§8.
  **LANDED 2026-06-19 (awaiting VS2026 build):** `GemmaConfig` extended with `withGlobalHeadDim`/
  `withNumGlobalKVHeads`/`withKeyEqualsValue` (Gemma defaults 512/1/true, fallback-0 to the sliding
  fields) + derived `getGlobalQProjectionWidth`/`getGlobalKVProjectionWidth`/`getGlobalPackedQKVWidth`
  (K=V-aware: 8704 vs 9216), validate() (even global_head_dim, divisibility), metadata round-trip.
  Tests added (global widths incl. K=V vs not, fallback, divisor throw, odd-dim throw, defaults). ZERO
  op/traits/Llama touch confirmed. Also a Step 0 consistency fix folded in: `head_dim_` default 0->256
  (a default-constructed `GemmaConfig` is now Gemma-12B-correct on head_dim, was deriving 240); the
  derive-fallback is now tested via explicit `withHeadDim(0)`. NEXT: Step 2 (sliding-window mask runtime
  + bounded-KV TKvPolicy sibling).
- [~] **Step 2a — sliding-window mask (runtime `window`).** The causal mask is a hardcoded upper bound
  `max_t2 = min( abs_t, T_stride - 1 )`, inner loops `t2 = 0 .. max_t2`
  ([Gqa.Prefill.Bf16.cu:65-66](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Prefill.Bf16.cu),
  mirrored in `Gqa.Prefill.Fp32.cu`). Add a lower bound `window_start = max( 0, abs_t - window + 1 )`
  and zero both tails; the decode path (`softmax_decode_forward` over the cache) needs the same bound
  or long-context generation silently attends outside the window. Pass `window` (0 = global) through
  the dispatch beside `position_offset`. Identical shapes -> **runtime field, not a template** (Gemma.md §6).
  **LANDED 2026-06-19 (awaiting VS2026 build, GPU). Option (a) chosen for the shared decode softmax.**
  `GqaConfig.window` (default 0; `withWindow`/`getWindow`/validate/metadata + tests). Op reads `window_`
  at build, threads to both softmax calls. Mask = `window_start = (window>0)?max(0,abs_t-window+1):0`
  (prefill) / `max(0,actual_len-window)` (decode); attends `[window_start, ...]`, zeros below-window +
  future. **`window=0` reproduces the unbounded path byte-for-byte -> Llama unchanged (prefill AND decode).**
  Files: GQA prefill softmax kernels (`Gqa.Prefill.{Fp32,Bf16}.cu`) + `.cuh` + dispatch (GQA-specific);
  shared common decode softmax (`CudaAttention.Softmax.{Fp32,Bf16}.cu`) with a **trailing defaulted
  `int window = 0`** in `CudaAttention.cuh` so existing callers are unchanged. NOTE: MHA does NOT call the
  common decode softmax (uses its own `cuda_softmax_decode_forward_*`), so the touch is effectively
  GQA-contained anyway. Windowed numerics need GPU + the Step-5 oracle; the window=0 regression is
  checkable now by running chat. NEXT: Step 2b.
- [ ] **[deferred -> after Step 5 HF validation] Step 2b — bounded KV ring cache as a `TKvPolicy` sibling (the payoff).**
  Resequenced 2026-06-19: 2b is a **memory optimization, not a correctness gate** — the Step 2a mask gives
  correct sliding numerics against the full `[B,NKV,T,HS]` cache, so Gemma runs correctly (and fine at
  modest context) without it; the payoff is long-context only (~20GB->~80MB across 40 sliding layers at
  256K). The **prefill ring is the hardest kernel work in the chassis** (a chunk's queries need >W keys
  at once -> needs a block-sparse/flash-style windowed rewrite; prefill+decode share one K/V buffer so
  it's all-or-nothing). Doing it before HF parity = optimizing an unproven path with no oracle. So defer
  until the full-cache Gemma is HF-validated (Step 5), then build the ring and diff against the validated
  full-cache path. Mechanism foreshadowed in [Policy.ixx](Mila/Src/Dnn/Quantization/KvCache/Policy.ixx)
  ("future SlidingWindow" KvCachePolicy, no dtype fields, consumed via `if constexpr`); note TKvPolicy
  currently conflates compression with bounding (orthogonal) — a standalone `SlidingWindowKvCache`
  (bounded, uncompressed) suffices for Gemma; bounded+FP8 is a later combinatorial concern. Original:
   The cache grows to
  full `T` and decode sweeps the whole valid length; a sliding layer only needs the last `window` keys.
  Size the per-layer cache to `min(T, window)` with ring-buffer write/wrap + modular decode indexing.
  This is a layout+kernel difference -> fold onto the existing KV-cache policy axis
  ([Quantization/KvCache/Policy.ixx](Mila/Src/Dnn/Quantization/KvCache/Policy.ixx)), NOT a new axis and
  NOT conflated with the window number. Mask-only gives Gemma's numerics with none of its memory win.
- [~] **Step 3 — proportional partial-rotary RoPE (cache-build change; extend Rope, NO component).** Two variants threaded
  through `Rope`'s `OperationTraits`: `RopeDefault` (full rotation, theta 10000, sliding) and
  `RopeProportional<Num,Den>` (rotate first `partial_rotary_factor`=0.25 of head_dim -> 128 of 512,
  pass the rest through, theta 1e6, global). Partial-rotary is a structural skip -> specialized
  branch-free kernel, mirroring the `GeluTanh` activation-functor pattern. **`RopeConfig` already
  carries `rotary_dim`** (`withRotaryDim`, default 0 = full head_dim, [Rope.Config.ixx:78](Mila/Src/Dnn/Components/Encodings/Rope/Rope.Config.ixx))
  and base theta (`withBase`) as runtime fields — so `partial_rotary_factor 0.25` of head_dim 512 =
  `withRotaryDim(128)` is plumbed at the config level. Open: confirm the Rope **kernel** honors
  `rotary_dim` (rotates the first `rotary_dim`, passes the rest through) and what "proportional"
  rope_type adds beyond partial rotation; only then decide whether `TRopePolicy` needs a distinct op
  specialization or whether the existing runtime `rotary_dim`/`base` suffices. The per-layer RoPE seam
  is the same plumbing as the per-layer window and lands alongside it. See Gemma.md §4.
  **INVESTIGATED + REORDERED 2026-06-19 (now AFTER Step 4):** `rotary_dim` is plumbed in `RopeConfig`
  but **the op/kernel IGNORE it** — `CudaRopeOp` forward/backward/prefill/decode + `build_cache` all pass
  `config_.getHeadDim()`, never `getRotaryDim()` ([CudaRopeOp.ixx](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Rope/CudaRopeOp.ixx)),
  so RoPE always rotates the FULL head_dim. The cache key honors `base` (per-layer theta works) but not
  `rotary_dim`. So Step 3 = (1) thread `rotary_dim` op->kernel (4 call sites + cache build), (2) kernel
  change to rotate first `rotary_dim` dims + pass through the rest, (3) **HF-reference check on the
  "proportional" frequency denominator** (`base^(-2i/rotary_dim)` vs `/head_dim` — do NOT guess, silent
  wrong numerics). Llama-shared but `rotary_dim=0` default = full rotation = Llama unchanged. NOTE the
  **sliding-layer RoPE already works** (full rotation + per-layer base via `withBase`); only the 8 GLOBAL
  layers need partial/proportional. Parked for an HF-reference pass; do Step 4 (GeGLU) first (genuinely
  easy: existing GatedMLP + GeluTanh functor).
  **HF-REFERENCE RESOLVED 2026-06-19 + ARCHITECTURE DECIDED (extend Rope, NO new PRoPE component):**
  `_compute_proportional_rope_parameters` (transformers `modeling_rope_utils.py`): `rope_angles =
  int(partial_rotary_factor * head_dim // 2)` = `int(0.25*512//2)` = 64 pairs (first 128 of 512 dims);
  `inv_freq_rotated = 1/base**(arange(0,2*rope_angles,2)/head_dim)` -- **denominator is head_dim (512),
  NOT rotary_dim**; then PAD the remaining `head_dim/2 - rope_angles` = 192 pairs with **ZERO**. So
  "proportional" = full head_dim/2 freq table with only the first rotary_dim/2 pairs real, rest zeroed.
  KEY: a **zero frequency -> cos=1, sin=0 -> rotation is identity (pass-through)**, so feeding the
  EXISTING rotation kernel a cache with zeroed upper freqs yields partial-rotary with **ZERO kernel
  change**. So Step 3 = (1) `build_cache` zeroes freq pairs at index >= rotary_dim/2; (2) add `rotary_dim`
  to the `RopeCacheRegistry::CacheKey` + `makeCacheKey` + hash. NO rotation-kernel change, NO op-path
  change, NO new component. Llama byte-identical (`rotary_dim=0` -> all freqs real -> identical cache;
  no intrinsic shift, unlike GeGLU). Completes the intent of the already-present `RopeConfig::rotary_dim`
  field. `TRopePolicy` compile-time selector NOT needed -- runtime cache-build difference. Confirm the
  exact `base^(-2i/head_dim)` formula in Mila's `build_cache` (Rope.Bf16/Fp32.cu) when implementing.
  **LANDED 2026-06-19 (awaiting VS2026 GPU build).** Confirmed Mila's `rope_build_cache_kernel`
  ([Rope.Fp32.cu](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Rope/Kernels/Rope.Fp32.cu))
  uses exactly `theta = base^(-2i/head_dim)` -- matches HF. Change: kernel gains `rope_pairs`; for
  `i >= rope_pairs` writes `cos=1,sin=0` (identity). Launcher `cuda_rope_build_cache_fp32` gains
  `rotary_dim` -> `rope_pairs = (rotary_dim>0 && rotary_dim<head_dim) ? rotary_dim/2 : half_dim`.
  Threaded through: `.cuh` decl, both Dispatch `build_cache` wrappers (FP32+BF16, cache is always FP32),
  `RopeCacheRegistry::CacheKey` (+`rotary_dim` field +hash mix), op `makeCacheKey` + `build_cache` call.
  **Rotation kernel, forward/prefill/decode, op dispatch: ALL untouched.** Llama byte-identical
  (`rotary_dim=0` -> `rope_pairs=half_dim` -> every pair real -> identical cache). Test: `Rope.Cuda.cpp`
  `Forward_PartialRotary_PassesThroughUpperDims` (rotary_dim=4 of head_dim=8 -> first 2 pairs rotated,
  upper 2 pass through; `ropeRotate` reference gained a `rope_pairs` arg). NOTE the global layer's theta
  1e6 already works via `withBase`; the block (Step 5) sets `withRotaryDim(128)` + `withBase(1e6)` for
  global layers. NO new files (extended existing Rope op cluster + test).
- [~] **Step 4 — GeGLU via `TGate` (option C: separate kernel/op, SiLU path untouched).** Gemma FFN =
  `GatedMLP<..., TGate=Gelu>` (`gelu_pytorch_tanh`, `intermediate 15360`). **LANDED 2026-06-19 (awaiting
  VS2026 GPU build).** Chose **option C** over functor-templatizing the optimized SiLU kernel, because
  the shared `Silu` functor uses portable `expf` while the SiLU kernel uses fast `__expf`/`__frcp_rn` --
  templatizing would shift Llama's SiLU bits. C leaves the SiLU kernel + `CudaSwigluOp` byte-for-byte
  untouched. **Forward-only** (Gemma inference-only; GeGLU backward = throwing stub, deferred to Training
  Revival). New: `Geglu.cuh`/`Geglu.cu` (scalar `GeluTanh(gate)*up`, shared functor via file-relative
  include like the Elementwise kernel), `CudaGegluOp` (FP32+BF16 forward, throwing backward),
  `OperationType::GegluOp` (+name+case), 2 `OperationTraits<GegluOp,Cuda,{FP32,BF16},void>` rows. Component
  `Swiglu<Device, Precision, ActivationType TGate = Silu>` (static_assert Silu|Gelu) selects the op by gate
  (`kGateOp = Silu?SwigluOp:GegluOp`) -- **Llama byte-identical** (`Swiglu<…>` defaults to Silu ->
  existing SwigluOp). `GatedMLP` static_assert lifted to Silu|Gelu + `SwigluType = Swiglu<…,TGate>`.
  CMake: Geglu.cu (kernel sources), Geglu.cuh (cuda_headers), CudaGegluOp.ixx (cuda_modules). Test:
  `Swiglu.Cuda.cpp` `Forward_GeGLU_MatchesReference` (FP32+BF16, `Swiglu<…,Gelu>` vs `GeluTanh(gate)*up`).
  DEFERRED (not Gemma-blocking): CPU `SwigluOp`/`GegluOp` (Gemma is CUDA), GeGLU backward (training),
  uint4 vectorization of the GeGLU kernel, kernel unification of SiLU+GeGLU (would need a SiLU
  numeric-parity test). See Gemma.md §7.
- [~] **Step 5 — `GemmaBlock` + `IDecoderLayer` + `GemmaTransformer` + converter.** **HF block topology
  confirmed 2026-06-19** (`Gemma4TextDecoderLayer`): FOUR Gemma-specific structural deltas beyond Steps
  0-4 — (a) **sandwich norm, 4 RMSNorms/layer** (input / post_attention / pre_feedforward /
  post_feedforward) vs Llama's 2; (b) **QK-norm** = RMSNorm over head_dim applied per-head to Q and K
  BEFORE RoPE (`q_norm`/`k_norm`, normalized_shape=head_dim); (c) **embedding x sqrt(hidden_size)**;
  (d) **final logit softcap** `tanh(logits/30)*30` post-lm_head. ONE detail to verify in 5c: HF reports
  attention `scaling=1.0` (not 1/sqrt(head_dim)) but Mila's GQA op hardcodes 1/sqrt(HS) in softmax --
  confirm Gemma's effective query scaling (QK-norm changes the convention). **Decomposed into sub-steps
  (build between each, the established rhythm):**
  **5a GemmaConfig completion** -- **LANDED 2026-06-19 (awaiting build):** added window(1024),
  sliding_window_pattern(6), global_rotary_dim(128), rope_theta_local(10000)/global(1e6),
  final_logit_softcapping(30) + setters/getters/validate/metadata/toString + `getEmbeddingScale`
  (sqrt(embedding_dim)) + per-layer helpers (`isGlobalLayer`, `getHeadDimForLayer`/`getNumKVHeadsForLayer`/
  `keyEqualsValueForLayer`/`getWindowForLayer`/`getRoPEThetaForLayer`/`getRotaryDimForLayer`/
  `getQProjectionWidthForLayer`/`getPackedQKVWidthForLayer`) -- the interface 5c consumes. Tests:
  isGlobalLayer 5:1 (global at 5,11,..,47), per-layer sliding+global geometry, defaults, metadata
  round-trip. No new files. **5b QK-norm wiring -- INVESTIGATED 2026-06-19 (2 findings, both keep
  RmsNorm/Llama untouched):** (1) QK-norm needs NO new component -- the RmsNorm kernel works on an
  `[outer, norm_dim, inner]` layout normalizing each `norm_dim` group, so per-head QK-norm = view Q
  `[B,T,n_heads*head_dim]` as `[B, T*n_heads, head_dim]` (valid contiguous view) + `RmsNorm(shape{head_dim})`;
  weight is `[head_dim]` shared across heads = Gemma's q_norm/k_norm. Pure 5c wiring. (2) **Gemma RMSNorm
  uses `x_norm*(1+weight)`** but Mila's kernel does standard `x_norm*weight` -> resolve in the CONVERTER
  (5e) by adding 1.0 to every RMSNorm weight at load (`weight'=1+weight_hf`); applies to all 6 norms/block
  + final norm; keeps the kernel Llama-safe. **Query-scaling resolved 2026-06-19 + FIXED:** HF Gemma4
  `self.scaling=1.0` (NO 1/sqrt(head_dim) -- QK-norm controls magnitude); QK-norm is applied BEFORE RoPE.
  Mila GQA op hardcoded 1/sqrt(HS) -> parameterized: `GqaConfig.withAttentionScale`/`getAttentionScale`
  (0 = derive 1/sqrt(head_dim) = Llama-identical; Gemma sets 1.0); op reads `attention_scale_` at build,
  both prefill+decode use it (was `1.0f/sqrtf(HS_)`). Tests added. LANDED (awaiting build). **5c GemmaBlock
  + IDecoderLayer** (two instantiations, sandwich norm,
  QK-norm, GeGLU, geometry; HAS design decisions -- align before building). **5d GemmaTransformer**
  (heterogeneous layers -> final norm -> lm_head -> KV-cache orchestration). **Design decisions
  2026-06-20:** (1) **Embedding scale (sqrt(hidden_size)) folds into the CONVERTER (5e), not the
  transformer** -- Mila keeps token-embedding and lm_head UNTIED (separate blobs; see the weight-tying
  optimization item below), so the converter scales the embedding table by sqrt(3840) and writes lm_head
  its own UNSCALED copy from the same (tied-in-HF) tensor. 5d's forward stays structurally identical to
  LlamaTransformer (no runtime `scale` primitive). Nuance: pre-store fold rounds fp32->bf16 vs HF's
  bf16 runtime multiply -- negligible for greedy; revisit at 5f if a token flips. (2) **Logit softcap
  (30*tanh(logits/30)) deferred to host-side at the sampler** -- strictly monotonic, so it does NOT
  change greedy argmax (5f parity unaffected); only reshapes the distribution for temperature/top-p
  sampling. `GemmaConfig::getFinalLogitSoftcapping()` carries the scalar; no device kernel in 5d.
  (3) **Heterogeneous layers**: `vector<IDecoderLayer*>` over two `GemmaBlock` instantiations (kGlobal
  false/true), `config_.isGlobalLayer(i)` selects per layer. (4) **One shared `GqaState` workspace** sized
  at the MAX head_dim (global 512) for q_permute/v_out and [B,NH,chunk,T] for preatt/att (NH=16 shared);
  `CudaGqaOp::setState` takes only the raw pointer and indexes with its own HS_, so the local (HS=256)
  layers use a prefix of the 512-sized buffer. **LANDED 2026-06-20 (awaiting VS2026 build):** new module
  `Dnn.Components.GemmaTransformer` ([Gemma.ixx](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.ixx)) —
  `LanguageNetwork` subclass, inference-only (forward/backward throw; prefill/decode drive generation),
  heterogeneous `vector<IDecoderLayer*>` built from the two GemmaBlock instantiations per `isGlobalLayer`,
  one shared `GqaState` workspace at max head_dim, chunked prefill + single-token decode mirroring
  LlamaTransformer, `loadParameters` via `PretrainedModelReader`. Wired: Mila/CMakeLists.txt source +
  Mila.ixx umbrella export. NO runtime `scale` primitive and NO softcap kernel (folded to converter /
  sampler per the decisions above). Transformer-level validation is 5f HF parity + the GQA operation-level
  oracle (the "Correctness-oracle dependency" item below). **Structural tests LANDED 2026-06-20 (TDD
  catch-up; 5c/5d had drifted to config-tests-only):** [Gemma.Block.Cuda.cpp](Mila/Tests/Dnn/Components/Transformers/Gemma/Gemma.Block.Cuda.cpp)
  — both GemmaBlock instantiations: construction, build contract, getType, 15-child graph, and the Gemma
  GEOMETRY deltas (decoupled head_dim; local 256 vs global K=V 320 packed-QKV widths); inference-only, no
  forward/backward (modeled on Llama, not GptBlock's training path). [Gemma.Cuda.cpp](Mila/Tests/Dnn/Components/Transformers/Gemma/Gemma.Cuda.cpp)
  — GemmaTransformer: construction/build/type/getModelType, inference-only throw contract, heterogeneous
  (local+global) BUILD, and prefill/decode logits-shape + finiteness on an ALL-LOCAL config (validated
  compact-NKV GQA path). Registered in Tests/CMakeLists.txt CUDA section. **The test instantiation
  immediately earned its keep — concretely instantiating GemmaBlock/GemmaTransformer for the first time
  surfaced (and fixed) two latent gaps the library-only build never compiled: (1) `GroupedQueryAttention`
  had no public `resetKVCache()` (its header comment claimed one; `GemmaBlock`'s virtual IDecoderLayer
  override forced it) -- added, delegating to the internal `kv_cache_op_->resetKvCache()`; the latent
  `LlamaBlock::resetKVCache()` typo is now valid too. (2) the Gemma global K=V split `[Q|K]` needs a
  2-output `split`, but the Cuda backend only implemented the 3-output one (Llama only ever splits Q|K|V)
  -- added `Cuda::StructuralOps::split(in, out0, out1, ctx)` reusing the 3-way kernel with a zero-width
  third output (no new .cu). Both were invisible until the types were instantiated.** **Still blocked (deferred, not
  skipped):** numeric correctness + the global K=V / head_dim-512 execution path (run only at 5f against
  the HF oracle, or via the GQA operation-level oracle below) — these are genuinely gated, not an
  oversight. A standalone `Llama.Block` test does not exist either; Gemma's block coverage now exceeds it. **5e GemmaModel::fromPretrained + HF->Mila
  Python converter** (folds the embedding sqrt(d) scale + the (1+weight) RMSNorm convention; writes untied
  lm_head). **LANDED 2026-06-20 (awaiting VS2026 build + a real Gemma 4 checkpoint to validate):**
  (a) `PretrainedMetadata` extended with `head_dim` + the Gemma chassis fields (`global_head_dim`,
  `num_global_kv_heads`, `key_equals_value`, `window`, `sliding_window_pattern`, `global_rotary_dim`,
  `rope_theta_local`/`global`, `final_logit_softcapping`) + JSON parsing
  ([PretrainedReader.ixx](Mila/Src/Dnn/Serialization/PretrainedReader.ixx)); (b)
  `Dnn.Models.GemmaModelConfig` (mirror of LlamaModelConfig, deployment-only); (c) `Dnn.Models.GemmaModel`
  ([GemmaModel.ixx](Mila/Src/Dnn/Models/GemmaModel.ixx)) — `fromPretrained` None/FP8/FP4 dispatch +
  `configFromMetadata` building the full `GemmaConfig`, inference-only (onTraining throws); (d) Python
  `Tools/Converters/Gemma/convert_weights.py` folding the sqrt(d) embedding scale (untied unscaled
  lm_head), the (1+weight) RMSNorm on all 6 norms + QK-norms + final norm, per-layer K=V QKV fusing
  (`[Q|K]` global / `[Q|K|V]` sliding), GeGLU gate+up fuse, and the full metadata. Wired into
  Mila/CMakeLists.txt + Mila.ixx. **TWO things to confirm at 5f (flagged in-code as `REVIEW:`):** the
  Gemma EOS / `<end_of_turn>` stop-token ids (`GemmaModel::eosToken`/`stopTokens` use 1 / 106), and the
  HF Gemma 4 config attribute names + state_dict keys (read defensively with Gemma.md defaults; the
  converter prints every resolved value on first run). **Gemma tokenizer converter LANDED 2026-06-20:**
  [Tools/Converters/Gemma/convert_tokenizer.py](Mila/Tools/Converters/Gemma/convert_tokenizer.py) modeled
  on the validated [Llama/convert_tokenizer.py](Mila/Tools/Converters/Llama/convert_tokenizer.py) — HF
  `AutoTokenizer` but **fact-driven** (NO SentencePiece runtime dependency — convert-time HF gives vocab +
  scores + merges, the Mila runtime decodes over the extracted data): parses the fast tokenizer's
  serialized `model` block to detect **BPE vs Unigram**, extracts **merges** (BPE) or per-piece **scores**
  (Unigram), and writes a Gemma-extended binary (adds a `model_type` byte + `num_merges` section to the
  shared format) so `loadGemma` picks the matching decode path. Special tokens read off the tokenizer
  (also prints the real `<start_of_turn>`/`<end_of_turn>` ids, resolving the `GemmaModel` stop-token
  REVIEW) + HF round-trip sanity print. Wired into the Converters README.
  **REMAINING in 5e — runtime `BpeVocabulary::loadGemma` / `BpeTokenizer::loadGemma`, GATED on the model
  type the converter reports (run it once to confirm):** common to both = `byte_level=false` (pieces are
  raw UTF-8), a NEW SentencePiece pre-tokenization (Metaspace: space->U+2581 + add_prefix_space, no
  tiktoken regex), `SpecialTokens::gemmaStyle()`, byte-fallback (`<0xNN>` pieces, gated so Llama/GPT-2 are
  byte-for-byte unaffected). **If BPE** (expected): reuse Mila's existing merge-by-rank path
  (`encodeSegmentBpe`) with a Gemma initial-unit step = UTF-8 *character* split + byte-fallback for unknown
  chars (not the per-byte / GPT-2-byte-encode splits Llama/GPT-2 use). **If Unigram:** a NEW Viterbi
  decode over the scores (Mila has none). Per [[feedback_minimal_reversible_change]] build the matching
  path on the CONFIRMED type, not the assumption. Confirmed necessary by reading the runtime: `loadLlama32`
  sets `byte_level=true` + Llama3 regex + GPT-2 byte-encoder ([BpeVocabulary.ixx:1150](Mila/Src/Data/Tokenizers/Bpe/BpeVocabulary.ixx),
  [BpeTokenizer.ixx](Mila/Src/Data/Tokenizers/Bpe/BpeTokenizer.ixx)), none of which match Gemma's pieces.
  Validate with an encode round-trip vs the HF tokens the converter prints. **5f HF parity** then runs
  end-to-end through Mila's own tokenizer.
  **UPDATE 2026-06-20 — type CONFIRMED BPE** (real run: 262144 vocab, 514906 merges) **and the runtime
  LANDED (awaiting VS2026 build + the round-trip test):** new `PreTokenizationMode::SentencePiece`
  Metaspace pre-tokenization (space->U+2581, split at marks, NO leading prefix — HF showed `The` not
  `_The`), `SpecialTokens::gemmaStyle()` + `<start_of_turn>`/`<end_of_turn>` registered from the loaded
  vocab, `BpeVocabulary::containsToken` (raw lookup; the byte-fallback test can't use `tokenToId` which
  substitutes UNK), `BpeVocabulary::loadGemma` reading the extended binary, and in `BpeTokenizer` a
  SentencePiece encode (UTF-8 *character* initial units + `<0xNN>` byte-fallback, then the shared merge
  loop extracted as `applyMerges`) + decode (U+2581->space, `<0xNN>`->byte). All gated behind
  `is_sentencepiece_` so the Llama/GPT-2 byte-level paths are byte-for-byte untouched. Gated test
  [BpeTokenizer.Gemma.cpp](Mila/Tests/Data/Tokenizers/Bpe/BpeTokenizer.Gemma.cpp) asserts HF encode parity
  (`"The capital of France is Paris."` -> `{818,5279,529,7001,563,9079,236761}`) + round-trip + atomic
  special tokens; skips unless the binary is at `<TEST_DATA_DIR>/models/gemma/gemma_tokenizer.bin`.
  **5f greedy-parity harness LANDED 2026-06-20:** building on the existing Llama validation pattern
  ([Dev/Scripts/llama_32_BF16/hf_llama_greedy_validation.py](Dev/Scripts/llama_32_BF16/hf_llama_greedy_validation.py)),
  [Dev/Scripts/gemma_4_BF16/hf_gemma_greedy_validation.py](Dev/Scripts/gemma_4_BF16/hf_gemma_greedy_validation.py)
  is the HF reference (Gemma 4 12B-it, bf16/CUDA): it greedy-decodes once and prints the prompt +
  generated token ids (copy-paste `kPromptIds`/`kExpectedGen` lines). **5f parity is an in-suite C++
  TEST, not a Python comparison:** those ids are hardcoded as ground truth in
  [Tests/Dnn/Models/GemmaModel.Parity.Cuda.cpp](Mila/Tests/Dnn/Models/GemmaModel.Parity.Cuda.cpp), which
  loads `GemmaModel::fromPretrained` and asserts greedy decode reproduces them token-for-token (Mila omits
  the trailing EOS HF emits, so the assertion is a token-exact PREFIX). Same pattern as the tokenizer
  suite's HF ground truth; uses `GemmaModel` directly (no Python / no binding at test time). Gated: skips
  without a CUDA device, populated ids, AND the checkpoint at
  `<TEST_DATA_DIR>/models/gemma/gemma4_12b_it_bf16.bin` (opt-in integration, never in CI). Feeding the HF
  prompt ids isolates MODEL parity (incl. the global K=V / head_dim-512 path); the tokenizer is validated
  separately (BpeTokenizerGemma). **To run 5f:** convert the weights (`convert_weights.py`, ~24 GB bf16),
  run the HF script, paste the two id lines into the test, rebuild + run. Locks in the `<end_of_turn>`
  stop-token id the converter now prints.
  **2026-06-20 — ground truth applied + FP4 dev-card fit; test is RED (structural bug, NOT FP4 noise).**
  `kPromptIds`/`kExpectedGen` are populated; the test loads with `.withFP4Quantization()` because the 12B
  BF16 weights (~24 GB) do not fit the 12 GB dev card — FP4 (`PerGroupFp4<128>`) brings the resident
  footprint to ~1/4 BF16 (BF16 checkpoint still read from disk + quantized host-side at load). First run
  diverges at generated token 0 with garbage high-vocab argmax (e.g. 255999) and stays wrong for all 9
  tokens. That is NOT FP4 rounding (quant noise accumulates gradually and at worst flips a LATE argmax;
  it does not produce garbage from token 0). **Root-cause framing:** 5f is the FIRST numeric validation
  of (a) the full Gemma stack end-to-end and (b) the global K=V / head_dim-512 attention path — by design
  ([Gemma.Cuda.cpp](Mila/Tests/Dnn/Components/Transformers/Gemma/Gemma.Cuda.cpp) header: the global path
  is BUILD-only/finiteness-only, numerics deferred to 5f; the all-local compact-NKV path is the only
  executed one). The 12B checkpoint uses a 5:1 local:global pattern, so the parity run is the first time
  the global path AND the full-stack integration (embedding sqrt(d) scale, dual RoPE, QK-norm, sandwich
  norm) execute against a real reference. FP4 is a SECOND, unseparated confound: a BF16 12B baseline does
  not fit the 12 GB card, so "run BF16" cannot isolate model-correctness from FP4 here. **Next step:**
  layer-wise activation diff vs HF — instrument `hf_gemma_greedy_validation.py` to dump per-layer hidden
  states for `kPromptIds`, add a matching Mila activation dump (FP4, fits), and find the FIRST divergent
  layer. Divergence at the first GLOBAL layer => global-attention kernel bug; divergence from layer 0
  across the board => embedding scale / RoPE / norm (full-stack), FP4-independent. Logit softcap is ruled
  OUT as a greedy cause (tanh softcap is monotonic; argmax unaffected). The `REVIEW:` markers in
  [GemmaModel.ixx](Mila/Src/Dnn/Models/GemmaModel.ixx) `eosToken()`/`stopTokens()` are confirmed by the
  ground truth (`<end_of_turn>`=106, `<eos>`=1) and can be retired independently of parity going green.
  **Chat sample now selects Gemma (2026-06-20):** [Samples/Chat/Src](Mila/Samples/Chat/Src) gained
  `ModelType::Gemma`/`ModelSize::B12`, a `gemma-12b` alias (instruct, BF16 compute, **defaults to FP4** so
  it fits the dev card), the Gemma instruct chat template (`<start_of_turn>{user,model}\n...<end_of_turn>`
  with the system turn folded into the first user turn), `loadGemma` tokenizer wiring, and Gemma special
  tokens added to the response stripper. Paths: `<models>/gemma/gemma4_12b_it_bf16.bin` +
  `<models>/gemma/gemma_tokenizer.bin`. Chat-only edits (no core API change).
  **VRAM reality on the 12 GB dev card (2026-06-20) — FP4 12B does NOT fit 12 GB, by a wide margin.**
  Resident = ~9.14 GB params (FP4 transformer ~5.4 GB; embedding BF16 ~2 GB; `lm_head` **deliberately
  untied** — [Gemma.ixx:27,496](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.ixx) "writes lm_head its
  own unscaled copy" so the embedding can carry the sqrt(d) scale — a second BF16 262144x3840 matrix,
  ~2 GB) + ~5.9 GB State = ~15 GB. WDDM spills to shared system RAM -> per-forward PCIe paging ->
  95-100% util thrash (correct values, just slow; the parity test runs this way — 42 s for 9 tokens).
  **State does NOT scale with `context_length` (earlier assumption was wrong):** ctx 4096->1024->512 gave
  State 7.2->6.10->5.92 GB. The ~5.8 GB floor is **per-layer prefill ACTIVATION buffers across all 48
  layers, sized at the prefill CHUNK (512), dominated by the GeGLU FFN (gate_up=30720, hidden=15360)**:
  ~4.4 GB activations + ~0.6 GB block scratch + ~0.4 GB shared RoPE cache. Only the small KV part
  (~0.35 MB/token) tracks context. **Root cause of the over-budget chunk:**
  `computeGemmaPrefillChunkSize` ([Gemma.ixx:96](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.ixx))
  picks the largest chunk in {512,256,128} whose GQA attention scratch fits a 1536 MB cap — it is BLIND
  to the 48xFFN activation cost (the dominant term), so it returns 512 (attn scratch there is only
  ~25 MB). The chunk is not exposed as a config/CLI knob. **Lever = smaller prefill chunk** (chunk 128 ->
  activations ~1.1 GB; chunk 64 -> ~0.6 GB), but even chunk 64 is marginal because params alone (9.14 GB)
  + driver overhead (~1 GB) leave <2 GB. **Fix shipped (partial, chat-only):** Gemma chat default context
  lowered (4096 -> 512) via `defaultContextLength()` in
  [Chat.Config.ixx](Mila/Samples/Chat/Src/Chat.Config.ixx) — necessary but NOT sufficient (context is not
  the lever). **Open core options (need a decision, all internal-impl):** (A) make
  `computeGemmaPrefillChunkSize` activation-aware (budget num_layers x per-token activation bytes, not
  just attn scratch) so it auto-drops to 128/64 under a real VRAM budget; (B) expose a prefill-chunk /
  VRAM-budget override on `GemmaModelConfig` + a `--prefill-chunk` CLI flag; (C) **reclaim the ~2 GB
  untied lm_head** by sharing storage with the embedding via an unscaled read-path view (params
  9.14 -> ~7.1 GB; the comfortable win) — deferred, needs care because the scale decoupling is why it is
  untied; (D) pool the 48 per-layer activation buffers (only one layer is live at a time in the
  sequential forward — the real architectural waste) — biggest win, biggest refactor. **Sequencing note:**
  footprint work is independent of and lower-priority than the 5f structural correctness bug above —
  coherent chat is blocked on correctness regardless, the parity test already exercises the model
  (thrash-but-correct) at ctx 512, and a 16 GB Blackwell card (where 12B-FP4 fits comfortably) is
  inbound. Recommend: fix correctness first, treat footprint as a separate track.
  **Decode-past-context crash FIXED (2026-06-20).** With a small prefill chunk (the run finally went fast
  enough to reach it), Gemma chat crashed in `CudaGqaOp::decode_optimized`
  ([CudaGqaOp.ixx:572](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx)) with
  `position >= active_max_seq_len_` (position_offset 512, capacity 512). Root cause: `GemmaModel` bounded
  the prompt AND ran the decode loop against the ARCHITECTURAL max (`config_.getMaxSequenceLength()` =
  262144), not the deployment context the KV cache was actually built with (512). With max_new_tokens=2048
  > context=512, decode wrote past the cache. Fix: `GemmaModel` now stores the build `context_length_` and
  bounds both `truncateIfNeeded` (prompt) and the decode loop (`if (position >= context_length_) break;`)
  against it — generation stops cleanly at the context boundary. **Latent in Llama too:**
  `LlamaModel::onGenerating` ([LlamaModel.ixx:320](Mila/Src/Dnn/Models/LlamaModel.ixx)) has the identical
  unguarded loop + architectural-max truncation; never triggers only because the Llama default context
  (4096) exceeds max_new_tokens (2048). Apply the same `context_length_` bound to `LlamaModel` (TODO).
  **Activation-diff instrumentation in flight (2026-06-20, TEMPORARY — remove once localized).** Simple
  print-to-screen (no files): to find WHERE Gemma diverges from HF, both sides PRINT a one-line summary of
  each prefill layer's LAST-TOKEN hidden state (l2 / mean / min / max + first 3 values) and you eyeball the
  two consoles top-down — first stage whose magnitudes diverge by orders of magnitude is the culprit.
  (1) `kGemmaDumpActivations` constexpr toggle in
  [Gemma.ixx](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.ixx) -> `printActivationSummary` prints
  `[GEMMA-DUMP] {embed,layer_00..47,final_norm} ...`; only fires when the whole prompt is ONE chunk, so keep
  `kGemmaPrefillChunkOverride` >= prompt length. (2)
  [hf_gemma_activation_dump.py](Mila/Tools/Converters/Gemma/gemma_4_BF16/hf_gemma_activation_dump.py)
  prints `[HF-DUMP] ...` in the same format, capturing each layer's output via forward hooks (robust to
  Gemma-4's `.language_model` wrapper). Run the PARITY TEST (single prefill of kPromptIds) with the toggle
  on for the Mila side; run the script where the HF ground truth was captured. Interpretation: embed
  off=front-end/sqrt(d) scale; first GLOBAL layer (5:1 -> layers 5,11,17,..)=global K=V/head_dim-512 path;
  a local layer=QK-norm/RoPE/GeGLU/sandwich-norm. Remove the toggle + `printActivationSummary` + the temp
  `<iostream>`/`<cmath>` includes once the divergent stage is found.
  **Localized to the FIRST decoder block (2026-06-20).** Per-layer dump result: `embed` MATCHES HF
  (Mila l2=64.08 vs HF 64.12, mean/min/max identical) -> embedding load + sqrt(d) scale are CORRECT.
  `layer_00` EXPLODES (Mila l2=1619 vs HF 88, ~25x, non-uniform per-element ratios 11-39x -> structural,
  not a scalar). layer 0 is LOCAL (globals at 5,11,..). All downstream layers (incl. the layer-11 jump to
  ~5100) are accumulation on top. So the bug is in the first local `GemmaBlock` forward (never value-
  validated; Gemma.Cuda.cpp only checked finiteness). Suspects, in order: the FP4 Linears (qkv/o/gate_up/
  down -- first FP4 ops in the net, embed proved the unquantized path) vs the block wiring (QK-norm / local
  GQA scale / GeGLU / sandwich-norm). NEXT PROBE LANDED: `kGemmaBlockDumpActivations` in
  [Gemma.Block.ixx](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.Block.ixx) prints per-sub-step
  LAST-TOKEN magnitudes (`[BLOCK-DUMP]`) for the first prefill call of each block kind -- self-contained
  (no HF; the explosion is 25-80x, far beyond FP4 rounding), so the first sub-step whose l2 jumps to the
  thousands names the culprit op. Also TEMPORARY, remove with the rest.
  **Localized to the ATTENTION sub-block; FP4 + norms EXONERATED (2026-06-20).** Sub-step + HF compare:
  Mila `input_norm.W`(1103, min-142/max+194) == HF `input_norm.W(1+w)` EXACTLY -> norm weights load
  correctly and ARE legitimately large (Gemma quirk; the earlier "corrupted weight" guess was WRONG).
  Mila `input_norm` output (1157) == HF (1132). First DIVERGENCE is the attention output: Mila `o_proj`
  l2=**1567** vs HF `self_attn` l2=**50.5** (~31x). Since everything upstream matches and the FP4 Linears
  feed off a correct `input_norm`, FP4 is NOT the cause. Prime suspect: the **attention scale**. The block
  hardcodes `GqaConfig.withAttentionScale(1.0f)` per the 2026-06-19 decision above ("HF Gemma self.scaling
  =1.0, QK-norm controls magnitude") -- a judgment call that the 31x attention divergence contradicts. NEXT
  (HF probe extended, no Mila rebuild needed): print HF `self_attn.scaling` / `query_pre_attn_scalar` /
  `head_dim`, and dump HF `q_norm`/`k_norm`/pre-`o_proj` attention to compare against Mila `qk_norm(q)`=130
  / `attn`=1893. If HF scaling != 1.0 (e.g. 256^-0.5=0.0625) the hardcoded 1.0 is the bug; if HF
  pre-o_proj attention << 1893 it is the attention math (scale), if ~1893 it is `o_proj`. Re-validate the
  2026-06-19 "scaling=1.0" conclusion against the loaded model.
  **Root cause near-certain: QK-norm `(1+w)` over-application (2026-06-20).** HF `self_attn.scaling=1.0`
  (matches Mila -> scale exonerated, the 2026-06-19 call was right). HF attention internals: `q_norm` out
  RMS 1.02, `k_norm` out RMS **0.12** (tiny K -> small scores -> SOFT softmax -> `attn_pre_oproj` RMS 0.78,
  heavy averaging). Mila `qk_norm(q)` RMS **2.03 (exactly 2x HF)** and `attn_pre_oproj`=1893 (RMS 29.6 ~ V
  -> over-SHARP softmax, ~argmax). So Mila's Q (and K) are too large -> ~4x scores -> softmax flips
  soft->sharp -> 38x attention blowup; `o_proj` is just a passthrough (exonerated). The exact-2x on q_norm
  fits the converter applying the **`(1+w)`** convention to q_norm/k_norm when Gemma's per-head QK-norms
  use the weight DIRECTLY: q raw w~1.02 -> `1+1.02`=2.02 (2x, matches); k raw w~0.12 -> `1.12` (~9x, the +1
  dominates the small weight). `input_norm` (a sandwich norm, genuinely `(1+w)`) is unaffected -> matches
  HF exactly. CONFIRM: Mila now prints `qk_norm(k)`/`q_norm.W`/`k_norm.W`; HF prints `q_norm.W_raw` vs
  `(1+w)`. Whichever HF weight the q_norm OUTPUT RMS (1.02) matches is the convention HF applies -- if RAW,
  the [convert_weights.py](Mila/Tools/Converters/Gemma/convert_weights.py) `_rmsnorm_to_numpy(+1)` must NOT
  be applied to `q_norm`/`k_norm` (they are NOT sandwich norms). FIX is converter-side (re-convert) or a
  load-time compensation; the (1+w) `_RMSNORM_KEYS` set should exclude the two QK-norms.
  **CONFIRMED + FIXED in converter (2026-06-20).** Decisive numbers: HF `q_norm.W_raw`=1.0234,
  `(1+w)`=2.0312; HF q_norm OUTPUT RMS=1.02 == RAW. HF `k_norm.W_raw`=0.1221, `(1+w)`=1.1250; HF k_norm
  OUTPUT RMS=0.122 == RAW. Mila loaded `q_norm.W`=2.0312 / `k_norm.W`=1.1250 == HF `(1+w)`. So Gemma's
  per-head QK-norms apply the RAW weight; the layer/sandwich norms use `(1+w)` (input_norm matched HF on
  `(1+w)`). The converter wrongly ran `_rmsnorm_to_numpy(+1)` on q_norm/k_norm -> Q 2x, K ~9x (the +1
  swamps the ~0.12 weight) -> ~18x scores -> over-sharp softmax -> 38x attention blowup -> garbage. FIX:
  `q_norm`/`k_norm` now written via `_tensor_to_numpy` (raw), not `_rmsnorm_to_numpy`. FP4, the attention
  scale (1.0, correct), o_proj, RoPE, and the norm LOAD path are all exonerated.
  **STILL RED after the converter fix (2026-06-20 EOD) -- a SECOND bug remains (or the re-convert did not
  land).** The QK-norm fix is correct (data-proven) but insufficient. RESUME PLAN (instrumentation is still
  in place, toggles ON): (1) Re-run the parity test with `kGemmaBlockDumpActivations` ON and FIRST verify
  the fix is actually in the loaded checkpoint -- `[BLOCK-DUMP] tf_layer_0 q_norm.W` should now read RMS
  ~1.02 (l2~65, was 2.03/l2 32.5 over n=256... note q_norm.W is [head_dim], so check the printed value
  ~1.02 not ~2.03) and `k_norm.W` ~0.12 (was 1.12), and `qk_norm(q)` RMS ~1.02 / `qk_norm(k)` ~0.12, and
  `attn`/`attn_pre_oproj` should have dropped from ~1893 toward HF's ~50. If those are UNCHANGED, the
  re-convert did not regenerate `gemma4_12b_it_bf16.bin` (check the converter output path / that the test
  loaded the new file) -- cheap to rule out first. (2) If the QK-norm values ARE fixed but the model still
  diverges, walk the per-layer `[GEMMA-DUMP]` vs `[HF-DUMP]` table for the NEW first divergent layer and
  repeat the sub-step localization there. HF ground truth (`hf_gemma_activation_dump.py`) is captured and
  reusable. Do NOT strip the instrumentation yet.
  **2026-06-21: QK-norm fix VERIFIED in checkpoint; SECOND bug is in the LOCAL attention op.** Re-convert
  landed: Mila `q_norm.W`=1.0234 / `k_norm.W`=0.1221 == HF raw; `qk_norm(q)` RMS 1.02 / `qk_norm(k)` 0.12
  == HF exactly. BUT local `attn` still RMS 24.5 (HF 0.78), barely moved from the pre-fix value -- that
  INSENSITIVITY to the now-correct Q/K means the local softmax is effectively one-hot regardless of
  scores. Verified correct in code: scale=1.0 in the QK GEMM, the prefill_softmax causal+window mask
  (local window=1024 -> window_start=0 for 18 tokens, == global; max_t2 excludes the uninitialized
  [18:512] cache), and the converter QKV packing `[Q|K|V]` == block split. (Global's small `attn` is NOT
  proof of health -- global V=K is QK-normed tiny ~0.06, so its output is small just from small V; the
  clean signal is LOCAL Mila 24.5 vs HF 0.78.) NEXT PROBE LANDED:
  [CudaGqaOp.ixx](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx) `dumpScoreRow`
  (static-once, first Gemma GQA prefill = layer 0 local) prints `[GQA-DUMP]` preatt (raw scores) + att
  (softmax weights) for head 0 / last query. Read: att sum~1 & max~1 => one-hot (scores wrong -> QK GEMM);
  att soft (max~1/n) but block `attn` still ~V => att*V GEMM/grouping wrong; preatt min/max huge =>
  scale/GEMM. TEMPORARY, strip with the rest.
  **2026-06-21 GQA dump result: ONE-HOT on key 0 (BOS) via a huge preatt[0].** `[GQA-DUMP]` layer 0 head 0
  last query: `att` sum=1.0004 (softmax healthy), max=0.957, argmax=0; `preatt` max=+23.875 at key 0 vs
  +-4.5 elsewhere. So the last query attends ~96% to the FIRST token (BOS) -> output collapses to V0 (~24).
  Classic ATTENTION-SINK signature. Two opposite-fix hypotheses, decided by HF's layer0/head0/lastq
  attention distribution (HF script now loads `attn_implementation="eager"` + `output_attentions=True`,
  prints `[HF-ATTN]`): (A) HF ALSO sinks on key 0 (argmax 0, max ~0.95) -> the PATTERN is correct and the
  bug is Mila's **V0**: sinks work because the sink token's VALUE is ~0 (absorbs attention, injects no
  content); Mila V0 ~24 (==its attn output 24.5) vs HF V0 ~0.78 (==HF attn output) means Mila's BOS value
  is wrong. (B) HF SPREADS -> Mila's preatt[0] score is spurious -> QK/RoPE bug. NOTE: because both sides'
  attn output ~= V0 when sinking, comparing the existing attn outputs (Mila 24.5 vs HF 0.78) already
  doubles as the V0 comparison IF (A). Strong prior: attention sink is real, so (A)/V0 is likely.
  **2026-06-21 CONFIRMED (A): both sink on BOS, the bug is V0 DIRECTION.** `[HF-ATTN]` layer0/head0/lastq:
  argmax=0 max=0.941 == Mila att 0.957. So the attention pattern is correct; the BOS value vector is wrong.
  Note the V0 MAGNITUDE (~24) is what the pipeline produces for any token; HF's V0 is small by DIRECTION
  (BOS aligned with v_proj null space = the learned sink), which magnitude dumps can't see (they matched
  on the LAST token). NEXT PROBE LANDED: token-0 (BOS) DIRECTION trace -- Mila `printBlockToken0`
  (`[BLOCK-T0]` input(t0)/input_norm(t0)/V(t0) with head values) + HF `t0()` (`[HF-T0]` embed/input_norm/
  v_proj first token). Walk embed -> input_norm -> V at token 0: first stage whose head values diverge is
  the bug (embed differs => BOS embedding row/scale; input_norm differs => norm; V differs => qkv/V proj).
  **2026-06-21 ROOT CAUSE (airtight) + FIXED: the converter `(1+w)` is wrong for ALL norms; Gemma 4 uses
  RAW weights.** Token-0 trace: `input(t0)` (BOS embedding) MATCHES HF exactly; `input_norm(t0)` diverges.
  Element-wise effective weight (output/(x/rms)): Mila == HF + **exactly 1.0** on every channel
  ([0] 18.9 vs 17.9, [1] 31.5 vs 30.4, [2] +1.0 vs -0.0001, [3] 17.1 vs 16.1). So HF applies the RAW
  stored weight (Mila kernel is raw x*weight -- proven by Llama + the QK-norm fix); the converter's `+1`
  over-applies. The killer is element [2], a MASSIVE-ACTIVATION channel (embedding 12.375): HF's raw
  weight ~0 SUPPRESSES it; the spurious +1 (->1.0) turns it back on -> v_proj makes a large V0 -> the BOS
  attention sink (both Mila & HF put ~95% on BOS) injects garbage instead of ~nothing -> residual
  explodes. The QK-norm fix was the first instance of this same bug. FIX:
  [convert_weights.py](Mila/Tools/Converters/Gemma/convert_weights.py) `_rmsnorm_to_numpy` no longer adds
  1.0 (all norms -- sandwich/QK/final -- written RAW); header + GemmaBlock header comments corrected.
  **ACTION: re-run `convert_weights.py` to regenerate the checkpoint, then the 5f parity test with ALL
  dump toggles OFF.** If green, STRIP the instrumentation: `printActivationSummary`/`printBlockActivation`/
  `printBlockToken0` + toggles + temp includes in Gemma.ixx & Gemma.Block.ixx; `dumpScoreRow` + temp
  includes in CudaGqaOp.ixx; the `[HF-BLOCK]`/`[HF-ATTN]`/`[HF-T0]` hooks in hf_gemma_activation_dump.py;
  and retire the GemmaModel eos/stop `REVIEW:` markers.
  **2026-06-21 norm fix VERIFIED but NOT the last bug: local attention still wrong (multi-head).** After
  re-convert: `input_norm(t0)` head[2]=-0.0012 == HF (dead channel suppressed), `V(t0)` l2 266 == HF 267,
  input_norm/QK all match. `layer_00` head ratios dropped ~25x->~16x (real progress) but residual still
  explodes (L0 1634 vs HF 88). Sharpest remaining divergence: attn output still l2=1503 (RMS 23.5) vs HF
  attn_pre_oproj 49.8 (RMS 0.78), ~30x -- DESPITE head-0 BOS sink now correct (att max 0.94 argmax 0 ==
  HF) and V0 correct. So head 0 contributes ~0 now; the large output is from the OTHER heads not
  averaging/sinking. Prime suspect: the GQA head->KV-head grouping (local NKV=8/GS=2) -- head 0 lands on
  the right KV head, others may pair with the wrong K/V. (post_attn_norm normalizes the bad attn to the
  right MAGNITUDE (314 == HF 326) but the DIRECTION is wrong, so the residual still diverges; magnitude
  dumps can't see it. The HF [HF-BLOCK] post-norm hook values (326/1594) are inconsistent with HF's small
  residual growth (64->88) by triangle inequality -- they over-capture; trust the LAYER outputs.) NEXT
  PROBE LANDED: GQA dump extended to heads {0,1,NH/2,NH-1} with the paired KV head in the label -- if
  head 0 sinks but the others don't, the grouping/pairing is the bug. FP4 not yet ruled in/out for the
  ~28% FFN delta (Mila fc_down 875 vs HF mlp 680) but that is secondary to the 30x attn divergence.
  **2026-06-21 multi-head GQA dump: pairing OK, scores too SHARP.** Heads attend to distinct keys
  (h0->BOS k0 max0.94, h1->k15 max0.77, h8->k12 max0.20 soft, h15->self k17 max0.98), sums~1 -- normal
  multi-head, grouping not scrambled. But most heads are SHARP (latch one key); non-BOS V is ~23 RMS
  (only BOS V suppressed), so a sharp head outputs ~V=23 -> the 1503. HF full attn RMS 0.78 => HF heads
  must AVERAGE, so Mila's softmax over-sharpens despite correct Q/K/scale. DECISIVE next: HF per-head att
  (HF script extended to heads {0,1,NH/2,NH-1}). The h15 SELF-score (query==key==17) is RoPE-INDEPENDENT
  (rotation cancels in Q.K), so it must match HF. If HF h15 is soft but Mila 0.98 -> Q/K DIRECTION wrong
  (qk_norm direction or qkv_proj FP4), NOT RoPE; if HF h15 matches (0.98) but h1 (relative-pos k15)
  differs -> RoPE rotation wrong (changes off-diagonal scores, preserves magnitude -> invisible to all
  dumps so far); if HF matches all -> pattern right, chase V/output. No Mila rebuild needed.
  **split() checked (2026-06-21): CORRECT for Gemma dims, but a LATENT validation gap found.** The BF16
  3-way split kernel `split3_bf16_vectorized_kernel`
  ([Structural.cu:106](Mila/Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/Structural.cu)) routes
  one `uint4` = 8 BF16 elements at a time by the vector's START column, so it is only correct when every
  output boundary is 8-aligned. `SplitOps::split` validates only `% 4`
  ([CudaTensorOps.Structural.ixx:217](Mila/Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Structural.ixx)).
  Gemma local (D0=4096,D1=2048,D2=2048) and global (8192,512) are all multiples of 8 -> not triggered, so
  split is NOT the parity bug. BUG TO FIX: tighten the split validation to `% 8` (or handle a
  boundary-straddling vector), else a future `%4`-not-`%8` split silently corrupts at the boundary.
  **2026-06-21 BIG narrowing: the entire attention SCORE path is correct; bug is V or att*V.** HF per-head
  att == Mila: h0 0/0.94, h1 15/0.77, h8 soft~0.20, h15 17/0.98 (self-sink matches => Q/K direction right;
  h1 off-diagonal matches => RoPE right). So split, QK-norm, RoPE, GQA grouping, scale, softmax are ALL
  exonerated. Yet attn output still 30x (1503 vs 49.8). att==HF and output!=HF => the difference is V (the
  values averaged) or the att*V GEMM. `V(t0)` matched HF, so suspect V at NON-BOS positions (the heads
  attend to k15/k17/...) or the AV GEMM. NEXT PROBE LANDED: `printBlockTokenN` dumps input_norm + V at
  token 0 AND the LAST token (the self-attended query) with head VALUES; HF script dumps the same
  (`input_norm(tL)`, `V(tL)`). If Mila V(tL) != HF V(tL) -> V projection wrong for non-BOS (trace to
  input_norm(tL) direction or FP4 on qkv_proj's V section); if V(tL) == HF but attn output still 30x ->
  the att*V GEMM is broken (output exceeds the att-weighted V). Mila rebuild + HF rerun.
  **2026-06-21 V(tL) MATCHES HF too (991.6 vs 990.9) -> att AND V both correct, att*V output still 30x.**
  So the bug is purely in how att*V COMBINES them. The tell: overall V(tL) matches but att*V pulls V
  PER-KV-HEAD -- an l2-preserving PERMUTATION of the V KV-heads (vs K) would match the overall V dump, keep
  att correct (att uses K, right order), yet make every query head pull the WRONG kv head's V. NEXT PROBE
  LANDED: dump V per-kv-head l2 for the last token (`[V-KVHEAD]` Mila / `[HF-V-KVHEAD]` HF). If the per-head
  l2 set matches HF but in a DIFFERENT ORDER -> V kv-head ordering is permuted (bug in the V cache write /
  per-head V arrangement vs K); if same order -> V per-head fine and the att*V (AV) plan/stride combines
  wrong. This is the last localization step before the fix. Mila rebuild + HF rerun.
  **2026-06-21 V per-kv-head MATCHES HF in value AND order (no permutation) -> attention FULLY correct;
  HF sub-step hooks are UNRELIABLE.** kv0..7: Mila 275/398/212/319/503/338/212/436 == HF
  273/393/211/321/506/337/211/437. So att (per head) + V (per kv head, right order) both match HF -> the
  attention is correct, and Mila's attn output ~1503 is RIGHT (sharp att on V~27 RMS -> output ~V). The HF
  o_proj-input hook (49.8) is mathematically IMPOSSIBLE (att 0.98 on key17 x V17[kv7] l2 437 must give
  ~428 for that one head alone > the claimed 49.8 for the whole vector) -- same failure as the HF
  post-norm hooks (326/1594 vs residual growth 64->88). So the forward-hook reference for the post-V
  attention/FFN/post-norm steps cannot be trusted; only HF LAYER outputs (output_hidden_states, validated)
  and the element-wise dumps that cross-check are reliable. PROVEN CORRECT element-wise vs HF: embed,
  input_norm (incl. dead-channel suppression), V (per-kv, right order), att (per head), GeGLU gate/up
  order. STILL: Mila L0 1634 vs HF 88. Remaining unverified (needs a RELIABLE HF reference, not the
  forward hooks): the attention-output HEAD ORDERING (prefill_unpermute_output_padded could scramble heads
  -> right magnitude, wrong direction -> res1 wrong -> FFN can't cancel -> residual grows to ~post_ffn_norm
  magnitude), o_proj (FP4) direction, and whether the post-norm contributions cancel (HF) vs add (Mila).
  NEXT (two clean options): (A) get RELIABLE HF intermediate residuals via a forward_PRE_hook on
  pre_feedforward_layernorm (= res1, an actual residual tensor) and compare res1 head values -> splits
  attention-output-direction (unpermute) from FFN; (B) read/审 prefill_unpermute_output_padded for a
  head-ordering bug. The att*V GEMM folds GS into M (batch=B*NKV); verify the unpermute inverts that fold
  in the SAME head order the rest of the stack expects.
  **2026-06-22 OPTION (A) RAN -> ATTENTION FULLY EXONERATED; BUG IS IN THE FFN SUB-BLOCK; the "unreliable
  HF hooks" verdict was WRONG.** Added the reliable `res1` reference -- a `forward_pre_hook` on
  `pre_feedforward_layernorm` ([hf_gemma_activation_dump.py](Mila/Tools/Converters/Gemma/gemma_4_BF16/hf_gemma_activation_dump.py),
  printed as `res1(+attn)` to match Mila's existing [Gemma.Block.ixx:298](Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.Block.ixx)
  dump). Result: **Mila `res1`=320.6 (min-84/max+96.5) == HF `res1`=333.3 (min-79/max+87), ~4%.** Since
  `res1 = res0 + post_attn_norm(attn)`, the WHOLE attention sub-block (incl. `prefill_unpermute_output_padded`
  head ordering -- option B) lands where HF lands AFTER the post-norm renormalizes it. So the raw-attn
  magnitude gap the backlog chased (Mila 1503 vs the HF o_proj-input hook 49.8) was a RED HERRING:
  `post_attn_norm` normalizes it away, and the direction was right all along (consistent with att/V/RoPE
  already matching). **Correction to the 2026-06-21 "hooks unreliable" conclusion:** the HF
  `post_attn_norm` hook (326) was being compared to the LAYER output (88); the correct anchor is `res1`,
  and `embed(64) (+) post_attn_norm(326) -> res1(333)` reconciles exactly (l2 of two ~orthogonal vectors
  in [262,390]). The hooks are RELIABLE; only the anchor was wrong -> the HF `mlp`/`post_ffn_norm` hooks
  are trustworthy when compared against `res1`/the layer output, not each other. **The explosion is
  entirely `res1 -> res2`:** Mila `res2`/L0 ~1634 vs HF L0 88 (18x), through
  `pre_ffn_norm -> fc_gate_up -> geglu -> fc_down -> post_ffn_norm -> res2`. `pre_ffn_norm` is fine
  (Mila 235.8 vs HF 219.9; the 7% is the inherited ~4% `res1` direction error amplified by normalization)
  -- a 4% input error cannot make 18x through a normalizing FFN, so the FFN itself is broken. NEXT: read
  the Mila FFN sub-step dump (`gate_up`/`geglu`/`fc_down`/`post_ffn_norm`/`res2`) against HF `mlp`/
  `post_ffn_norm` + the reliable bookends (HF `res1`=333, HF L0=88). Decisive pair: Mila `fc_down` vs HF
  `mlp` (were ~28% apart, 875 vs 680, pre-norm-fix) -- if close now, the bug is `post_ffn_norm` (weight
  load) or the residual add; if `fc_down` is wild, it is `geglu`/the FP4 FFN Linears. FP4 on
  `fc_gate_up`/`fc_down` is the first FFN-path FP4 not yet isolated.
  **2026-06-22 FFN dump (intermediate, SUPERSEDED -> see CORRECTION below).** Full FFN chain, Mila vs HF
  forward-hooks: `res1` 320.6 vs 333.3; `pre_ffn_norm` 235.8 vs 219.9; `fc_down`/`mlp` 875.8 vs 680.6;
  `post_ffn_norm` 1578.4 vs 1595.4; Mila `res2`/layer_00 1634.8. A first reading concluded "layer 0 is
  correct, the 88 is stale" by trusting the HF `post_ffn_norm` forward-hook (1595). That was WRONG -- the
  norm forward-hooks over-capture (see correction); the layer-output 88 is the truth.
  **2026-06-22 CORRECTION + FULL PER-LAYER TRAJECTORY: bug is the LAYER-0 FFN; HF norm FORWARD-hooks
  over-capture (the backlog's original "hooks unreliable" instinct was right).** HF's own hooks are
  mutually inconsistent: `post_ffn_norm`(1595) (+) `res1`(333) forces layer_00 >= 1262, but `[HF-DUMP]
  layer_00`=88. Three checks prove the 88 (layer-output) is truth and the norm forward-hooks lie:
  (1) the layer trajectory is smooth/Gemma-plausible (88,79,75,76,68,84,77,49,111,72,35,33,72,132,161,
  193,220(peak L16),...,6.6 @L47, final_norm 304); (2) **`final_norm` scale-invariance** -- RMSNorm output
  is input-magnitude-independent, HF final_norm 304 from a 6.6 residual vs Mila 14357 from a 4665 residual:
  if the residual were really ~1600 HF's final_norm would be huge too, it isn't -> Mila's residual is
  garbage (per-channel outliers +-668..844 vs HF +-200); (3) embed(64)/`res1`(333)/layer_00(88) are
  genuine residual TENSORS (input-capture/pre-hook/layer-output, cannot be norm-inflated) and are
  self-consistent. **So DISCARD the `[HF-BLOCK]` post_attn_norm/mlp/post_ffn_norm forward-hook numbers.**
  Reliable layer-0 picture: embed 64==64; **`res1` 320==333 (attention CORRECT)**; **`res2` 1634 vs 88
  (18x, FFN BROKEN)**. HF's FFN REDUCES the residual (333->88); Mila's INFLATES it (320->1634). Mila's
  effective FFN contribution (post_ffn_norm 1578) is ~4-6x HF's true contribution (must be in [245,421]
  to take 333->88). It then compounds (Mila L0 1634 -> L47 4665; HF stays 30-220) with a SECOND jump at
  the first global layer 11 (Mila 2117->4673) -- but the PRIMARY bug is the layer-0 FFN; fix first.
  **FP4 likely EXONERATED:** the attention Linears (qkv_proj/o_proj) are also FP4 and attention is correct
  (res1 matches) -- same mechanism -> the bug is FFN-SPECIFIC logic, not quantization. Prime suspects:
  the **GeGLU** (gate/up fusion order / GeluTanh applied to the wrong half of the fused fc_gate_up) or the
  fc_gate_up/fc_down packing, or post_ffn_norm.W. NEXT PROBES (split them): (1) dump Mila
  post_ffn_norm.W / pre_ffn_norm.W / post_attn_norm.W vs HF RAW post_feedforward/pre_feedforward/
  post_attention_layernorm.weight -- only input_norm.W + the QK-norms were element-wise verified after the
  RAW-weight re-convert; the other 3 sandwich norms were NOT (pre_ffn_norm 235 vs 219 ~matches, so likely
  fine, but confirm). (2) Dump the gate vs up halves of fc_gate_up separately and confirm GeluTanh hits
  the GATE half AND that the converter's fused [gate|up] packing order matches the geglu split (a swap
  gives GeluTanh(up)*gate -> wrong direction -> FFN stops cancelling). Mila per-layer l2: L00-10
  1634.8/1753.5/1778.8/1784.7/1788.1/1808.8/1829.8/1843.6/1906.4/1971.2/2117.9, L11 4673.6 (global jump),
  L12-47 ~4640-4665, final_norm 14357.6. HF per-layer in the trajectory above.
  **2026-06-22 PROBES RAN: sandwich-norm weights load PERFECTLY -> bug is the FFN BODY (fc_down direction),
  FP4 on the large FFN Linears the prime suspect (REVERSES the "FP4 exonerated" line above).** Element-wise:
  Mila loaded post_attn_norm.W/pre_ffn_norm.W/post_ffn_norm.W == HF RAW to 4 decimals (172.35/786.59/
  1173.34, same min/max). **Weight-load RULED OUT.** The `88` is now triple-confirmed (layer-0 forward-hook
  AND layer-1 input pre-hook both 88) and the post_ffn_norm forward-hook (1595) is provably bogus by law of
  cosines (`||333 + 1595 v|| = 88` needs cos=-2.49). HF `ffn_reliable` pre-hook (680.6) == HF `mlp` hook
  (680.6) -> the mlp ACTIVATION hook is reliable; only the NORM forward-hooks over-capture. Reliable
  picture: HF FFN CANCELS (res1 333 -> res2 88, true contribution ~-330); Mila INFLATES (res1 320 +
  post_ffn_norm 1578 -> res2 1634). Since W is exact, GeGLU is VERIFIED ([Geglu.cu:33-39] reads gate=first
  half/up=second half; converter cats [gate|up] [convert_weights.py:306-310]), and RMSNorm normalizes away
  magnitude, the only explanation is **Mila `fc_down` output DIRECTION is wrong** -- it lands on
  post_ffn_norm.W's outlier channels (effective w 1578/62=25.5 > W rms 18.9) while HF lands on small-w
  channels (330/62=5.3); the 28% fc_down magnitude gap (875 vs 680) is the visible tip. **Prime suspect:
  FP4 on fc_gate_up/fc_down** -- far larger than the attention Linears, and post_ffn_norm.W's max-+37
  outliers AMPLIFY a small FP4 direction error into the lost cancellation (o_proj FP4 being only mostly-fine,
  4% res1 error, fits same-noise/amplified). **DECISIVE ISOLATION (no code change): re-run FP8 not FP4**
  (`/model gemma-12b fp8` or `.withFP8Quantization()`); ~half the error -> res2 toward 88 = FP4 precision is
  the cause (FFN path needs FP8 / higher-precision fc_down / tighter FP4 group); res2 stays ~1634 =
  STRUCTURAL FFN bug, drill fc_gate_up/fc_down packing. PROBES LANDED for next run: Mila [BLOCK-T]
  head-value dumps of res1/post_ffn_norm/res2 + HF [HF-BLOCK] head values -- compare post_ffn_norm[i]
  direction (HF ~ -res1[i] cancel vs Mila additive). NEXT: run FP8; if structural, compare fc_down head
  direction Mila vs HF mlp.
  **2026-06-22 PINPOINTED to the GeGLU step (FP4 dropped per user: Llama's FFN uses the same FP4 Linears).**
  Element-wise head values at the last token (token 17) finally localize it. FFN contribution to the
  residual (`res2 - res1`, head): HF `[1.71, 1.65, +47.55, -2.10]` -- small everywhere EXCEPT channel 2,
  where +47.55 CANCELS `res1[2]=-50.75` -> `res2[2]=-3.20`. Mila `[31.4, 24.9, -43.5, -0.86]`: channel 2
  is **-43.5 -- same magnitude (~45), OPPOSITE sign** -> doubles `res1[2]=-48.5` -> `res2[2]=-92`. That
  sign flip on the dominant channel IS the blow-up. (Also confirms element-wise that the HF post_ffn_norm
  FORWARD hook `[36.75,39.25,-9.75]` is NOT the residual contribution -- bogus, as the magnitudes said.)
  Magnitude trace through the FFN body (HF genuine-activation hooks now captured): pre_ffn_norm 235.8/219.9
  (1.07), gate_up combined 817.8 / HF sqrt(gate692.4^2+up490.2^2)=848.4 (0.96 -- MATCHES), **geglu 1448 /
  1161 (1.25 -- JUMP)**, fc_down 875.8/680.6 (1.29). **The jump is at GeGLU**: gate_up matches HF but geglu
  leaps 1.25x. HF `gate_proj` head is all-NEGATIVE `[-7.16,-0.31,-2.97,-7.81]` so `GeluTanh(gate)~=0` and
  HF `geglu` head is `[-0.0,+0.23,-0.009,-0.0]` (near zero); if Mila's gate half is less-negative (wrong
  gate input direction, or a gate/up identity issue), GeluTanh passes far more through -> geglu too big ->
  fc_down wrong -> post_ffn_norm sign-flips channel 2. NOTE res1 also has channel-level errors (head[1]
  0.457 HF vs 1.508 Mila, ~3.3x) from attention, but res1[2] (the killer channel) MATCHES (-50.75 vs
  -48.5) -- so the FFN itself flips channel 2, not just inherited res1 error. PROBE LANDED: Mila
  [BLOCK-T] head dumps of gate_up/geglu/fc_down. NEXT (decisive): compare Mila gate_up head[0:4] vs HF
  gate_proj `[-7.16,-0.31,-2.97,-7.81]` (if NOT all-negative -> fc_gate_up output wrong: gate input
  direction or weights), Mila geglu head vs HF `[-0.0,+0.23,-0.009,-0.0]` (GeluTanh), Mila fc_down head vs
  HF mlp `[+10.875,+19.25,-2.89,-17.75]`.
  **2026-06-22 REDIRECT -- the FFN is INNOCENT (faithful amplifier); the SEED is res1 DIRECTION = the
  ATTENTION OUTPUT. Back to the never-checked option B (unpermute head ordering).** Head values settle it:
  Mila gate_up[0:4]=`[-5.31,+0.95,-2.27,-7.81]` vs HF gate_proj `[-7.16,-0.31,-2.97,-7.81]` -- channel 3
  is EXACT, channel 1 SIGN-FLIPPED. fc_gate_up can only get channel 3 exact if its weights+GEMM are
  correct, so it is computing right and the INPUT (pre_ffn_norm<-res1) is what's off. The flip then
  cascades through GeluTanh's knee: gate_up[1] +0.95 (HF -0.31) -> GeluTanh +0.78 (HF -0.12) -> geglu[1]
  -1.59 (HF +0.23, flipped+6.7x) -> fc_down -> post_ffn_norm outlier channels amplify -> res2 explodes.
  **So GeGLU/FFN is a catastrophic AMPLIFIER of res1 direction error, not the source.** The seed: res1 L2
  MATCHES (320 vs 333) but DIRECTION is wrong -- res1 head Mila `[0.25,1.51,-48.5,2.28]` vs HF
  `[0.25,0.46,-50.75,2.17]`; backing out post_attn_norm = res1 - embed gives channel 1 Mila +0.19 vs HF
  -0.86, SIGN-FLIPPED. **A direction error preserves L2 -- exactly what a head-ordering/permutation bug
  does, and exactly why option A (res1 L2) gave false confidence.** This resurrects backlog option B
  (`prefill_unpermute_output_padded`, never audited): att-weights/V/scores all match HF (verified) but the
  attention OUTPUT direction was never checked element-wise -- a permuted unpermute scrambles direction
  while preserving magnitude (att 1503, res1 320 both "right" size). PROBE LANDED: Mila [BLOCK-T] head
  dumps of attn/o_proj/post_attn_norm. NEXT: compare Mila post_attn_norm head vs HF reliable hook
  `[-0.0001,-0.8633,-50.75,+0.1445]` (confirm attention-output direction divergence), then AUDIT
  `prefill_unpermute_output_padded` head ordering (the att*V GEMM folds GS into M=B*NKV; verify the
  unpermute inverts that fold in the head order the rest of the stack expects). The fix is in the GQA
  attention output, NOT the FFN.
  **2026-06-22 CONFIRMED attention-output direction is the seed + isolated to the PARTIAL-CHUNK path (test
  pending).** Reliable `post_attn_norm` head: Mila `[-0.003, +0.193, -48.5, +0.258]` vs HF
  `[-0.0001, -0.863, -50.75, +0.144]` -- channel 1 SIGN-FLIPPED, channel 3 ~1.8x, **dominant channel 2
  matches**. All in head 0 (channels 0-255), and head 0 does NOT uniformly match/mismatch -> NOT a clean
  head permutation; it is a per-channel/within-head direction error. Audited
  [CudaGqaOp.ixx:583-633](Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx): the
  `POSSIBLE BUG` padded_T comment is a since-fixed red herring -- the partial path builds partial plans at
  chunk_len and padded_T=chunk_len, internally consistent. BUT the dump runs the PARTIAL path
  (`kGemmaPrefillChunkOverride` ~64 > prompt 18 -> is_full_chunk=false), and the whole attention pipeline
  mixes prefill_chunk_size_ and chunk_len (softmax line 600 takes BOTH; partial QK/AV plans; unpermute) --
  a Gemma-specific config (forced small override + short prompt) the Llama path rarely hits the same way.
  **DECISIVE TEST (one constant, fits 12GB): set kGemmaPrefillChunkOverride = exact prompt length (18) ->
  is_full_chunk=true -> all prefill_chunk_size_/chunk_len mixing collapses to one value.** If res2 -> ~88,
  the bug is the partial-chunk attention path (stride handling across softmax/AV-plan/unpermute); if res2
  stays ~1634, chunking is innocent and the per-channel error is in core attention (att*V combine /
  per-channel V). Confirmed-correct so far: GeGLU op+kernel+fusion, all 6 norm weights ==HF raw, FP4 path
  (=Llama), converter FFN writes, fc_gate_up (gate_up[3] exact). The FFN is a faithful amplifier; the bug
  is upstream in the GQA attention output, likely the partial-chunk path.
  **2026-06-22 MAJOR REVERSAL via an fp32 ORACLE: the ATTENTION IS CORRECT; the HF sub-step forward-hooks
  (post_attn_norm / res1-pre-hook) were BOGUS IN DIRECTION too -> the entire attention-output investigation
  (the chunk test, option B, "res1 direction wrong") was chasing bad reference data. The bug is the FFN.**
  Built a numpy fp32 oracle in [hf_gemma_activation_dump.py](Mila/Tools/Converters/Gemma/gemma_4_BF16/hf_gemma_activation_dump.py)
  that computes layer-0 att*V -> o_proj -> post_attn_norm from RELIABLE inputs only (softmax weights from
  output_attentions + the v_proj activation), bypassing the magnitude-bogus attn/o_proj hooks. Result:
  oracle `attn`=1480.5 head`[-1.65,-8.54,-6.02,-1.23]` == Mila 1503 `[-1.48,-8.50,-6.09,-1.11]`; oracle
  `post_attn_norm`=314.3 `[-0.001,+0.286,-50.44,+0.326]` == Mila 313.7 `[-0.003,+0.193,-48.5,+0.258]`. So
  **Mila's att*V combine, V-per-channel, o_proj, and res1 are all CORRECT** (oracle used HF's V and
  reproduced Mila's attn -> V matches per-channel; combine matches). DECISIVE: oracle post_attn_norm[1]
  =+0.286 and Mila=+0.193 BOTH disagree with the HF forward-hook's -0.863 -> the HF post_attn_norm/res1
  hooks are bogus IN DIRECTION (not just magnitude). The "res1 head[1] 0.457 vs Mila 1.508" divergence was
  the bogus HF res1 pre-hook; Mila's res1 (head[1]~1.5) matches the oracle = CORRECT. So ALL L0 heads' att
  match HF (user-verified) AND the attention output is oracle-correct -> attention fully exonerated.
  **Therefore: a CORRECT res1 (~320) enters the FFN and Mila yields res2=1634 vs the reliable HF layer-0
  output 88 -> the FFN is the bug** (prior FFN "clearing" relied on the bogus HF post_ffn hooks, so it is
  void). Only the LAYER OUTPUTS (embed 64, res2 88/79/.../6.6) and the oracle are trustworthy; every HF
  sub-step forward-hook is bogus. NEXT (landed): oracle extended through the FFN (res1=embed+post_attn ->
  pre_ffn_norm -> GeGLU(GeluTanh) -> fc_down -> post_ffn_norm -> res2) all fp32 from HF weights. Re-run:
  if oracle res2 ~= 88, the FFN bug is confirmed and the per-step oracle vs Mila [BLOCK-T17]
  (pre_ffn_norm/geglu/fc_down/post_ffn_norm) pinpoints the failing op; if oracle res2 ~= 1634 then 88 is
  wrong (unlikely -- triple-confirmed). FFN suspects to re-examine WITHOUT the bogus hooks: GeluTanh exact
  form, fc_down FP4, post_ffn_norm application.
  **2026-06-22 ROOT CAUSE FOUND (HF SOURCE): Gemma RMSNorm is `x_norm * (1 + weight)`, Mila uses RAW
  `x_norm * weight` -> the +1 removal was WRONG; restore it.** `output_hidden_states` (canonical, reliable
  -- NOT a forward hook) confirms HF residual is SMALL: embed 64 -> L0 88 -> ... -> L47 304, matching the
  earlier layer-output hooks (those WERE reliable; only the norm/res1 sub-step forward hooks are bogus).
  Mila residual 64 -> 1634 -> ... -> 14357 (18-47x). The fp32 oracle (standard sandwich-norm, HF weights,
  RAW) gave res2=1643 == Mila, NOT 88 -> the standard RAW math is wrong and Mila faithfully implements the
  same wrong math. Fetched the HF transformers Gemma3 source: `Gemma3RMSNorm.forward` = `_norm(x.float())
  * (1.0 + self.weight.float())` -- **(1 + weight)**, and `Gemma3DecoderLayer` is standard sandwich-norm
  with NO scaling. So HF applies (1+w); Mila applies raw w (the backlog removed the +1 in convert_weights.py
  `_rmsnorm_to_numpy`). The "raw" proof (input_norm dead-channel suppression) was a BOGUS forward-hook
  artifact: with (1+w) that channel is 1+(-0.0001)~=1 (active), not suppressed -- the hook lied, like every
  other sub-step hook. **FIX: restore `+1` in convert_weights.py `_rmsnorm_to_numpy` for ALL norms (input/
  post_attn/pre_ffn/post_ffn/QK/final) so Mila's raw kernel computes normalize*(1+w_hf)=HF; OR apply (1+w)
  in a Gemma-specific RmsNorm path (kernel is shared with Llama which is genuinely raw, so the converter
  route is cleaner).** VERIFY: oracle switched to (1+w) (pan + FFN rmsnorm); re-run -> if [ORACLE] res2 ~=
  88, confirmed and `[[project_gemma_rmsnorm_raw_weights]]` memory is WRONG (the "+1 caused garbage" was the
  bogus-hook-era misattribution; the 15-turn attention detour was all on lying forward hooks). After the
  re-convert, re-run 5f parity; the attention/GeGLU/FP4/att*V were all correct (oracle-matched) and need no
  change.
  **2026-06-22 FIX LANDED (awaiting VS2026 build + 5f re-run): `(1+weight)` applied at the KERNEL via a new
  `RmsNormConfig::withUnitOffset` -- NO re-convert needed.** Added `unit_offset` (default 0.0 = raw =
  Llama/GPT-2) to [RmsNorm.Config.ixx](Mila/Src/Dnn/Components/Normalization/RmsNorm/RmsNorm.Config.ixx)
  (setter/getter/metadata/toString); threaded through the FORWARD path only -- RmsNorm.cuh decls,
  RmsNorm.{Fp32,Bf16}.cu kernels+launchers (`w = weight[i] + weight_offset`), RmsNormOp.Dispatch.ixx,
  RmsNormOp.ixx forward (`config_.getUnitOffset()`). Offset 0 is byte-identical -> Llama/GPT-2 untouched;
  backward left raw (Gemma inference-only, all training models use offset 0). Gemma sets
  `withUnitOffset(1.0)` on ALL norms: the `rms()` helper in Gemma.Block.ixx (input/post_attn/pre_ffn/
  post_ffn + QK q_norm/k_norm) + the final norm in Gemma.ixx. Converter UNCHANGED (writes RAW = HF
  weights); only stale "Gemma uses raw" comments corrected (convert_weights.py + Gemma.Block.ixx header).
  Memory [[project_gemma_rmsnorm_raw_weights]] rewritten (was wrong). VERIFY: build in VS2026, re-run 5f
  parity on the EXISTING checkpoint -> expect coherent generation / token-match. CPU RmsNorm op left raw
  (Gemma is CUDA-only). TODO after green: strip the temporary [GQA-DUMP]/[BLOCK-DUMP]/[GEMMA-DUMP]/oracle
  instrumentation (Gemma.ixx, Gemma.Block.ixx, CudaGqaOp.ixx, hf_gemma_activation_dump.py).
  **2026-06-22 (1+w) WAS WRONG -- REVERTED to raw; residual bug is NOT the norm convention; Gemma 4 !=
  Gemma 3.** Built + ran 5f with (1+w) on all Gemma norms: (a) attention went ONE-HOT (att max 1.0 on most
  heads; qk_norm(q) 65->129 x2, qk_norm(k) 5.5->50.8 x9 -> Q.K ~18x -> argmax softmax) whereas HF
  `output_attentions` is SOFT (max ~0.94) -> **Gemma 4's QK-norm is effectively RAW; (1+w) is wrong for
  it**; (b) res2 still 1620 (was 1634 raw) vs HF 88 -> **(1+w) does NOT fix the residual**. With raw the
  att is already soft/correct (matches HF) AND res2 is 1634 -> the ~18x residual blow-up is INDEPENDENT of
  the norm convention. So the Gemma3 `(1+weight)` source I used is wrong for Gemma 4 (it one-hots the att):
  REVERTED the Gemma block + final-norm `withUnitOffset(1.0)` back to raw (offset 0). The generic
  `RmsNormConfig::withUnitOffset` infra (config + FP32/BF16 forward kernels + dispatch + op) is harmless
  (default 0 = Llama-byte-identical) and KEPT for potential later use. Converter unchanged (raw). The
  residual bug remains OPEN and is architectural: an fp32 oracle (HF weights + reliable att/V, standard
  sandwich GeGLU) reproduces Mila's ~1640 not 88, so Mila implements the *standard* layer correctly but
  Gemma 4's real layer/FFN/norm math differs (post_ffn_norm.W RMS 18.9 makes a standard post-norm ~1234+,
  incompatible with res2=88 under a plain add; HF's fc_down must land on small-weight channels). NEXT
  (landed in hf_gemma_activation_dump.py): `inspect.getsource` dump of the LOADED Gemma 4
  RMSNorm.forward/_norm + DecoderLayer.forward + MLP.forward + Attention.forward + config.architectures --
  read the real Gemma-4 layer math and diff vs Mila's GemmaBlock; the residual/FFN delta is there. Reliable
  HF refs ONLY: output_attentions + output_hidden_states + the fp32 oracle -- the per-submodule FORWARD
  hooks all lie (caused the ~15-turn attention detour). Memory [[project_gemma_rmsnorm_raw_weights]]
  rewritten to reflect UNSOLVED state.
  **2026-06-22 ROOT CAUSE FOUND via `inspect.getsource` of the LOADED model (arch
  `Gemma4UnifiedForConditionalGeneration`, `Gemma4UnifiedTextDecoderLayer`/`Gemma4UnifiedRMSNorm` -- NOT
  Gemma3). THREE Gemma-4 deltas Mila is missing:** (1) **`hidden_states *= self.layer_scalar`** at the END
  of every decoder layer (after the FFN residual add) -- THIS is the 18x residual bug; Mila has no such
  multiply so the residual grows unbounded (1634->4665) while HF scales each layer output back to ~88-220.
  From L0: layer_scalar ~= 88/1634 ~= 0.054 (exact value/per-layer TBD). (2) **`v_norm`** on the value
  states (`value_states = self.v_norm(value_states)`) -- Gemma4 has q_norm/k_norm AND v_norm; Mila's block
  only does QK-norm, missing V-norm (the oracle missed it too -- used pre-v_norm v_proj). (3) **RMSNorm is
  RAW** `normed * weight` gated by a `with_scale` flag (NO (1+weight)) -- confirms Gemma4 != Gemma3, the
  (1+w) detour was from trusting the Gemma3 source; raw was right. Some norms may have `with_scale=False`
  (pure normalize, no weight). ALSO (later layers, not L0): `is_kv_shared_layer`/`shared_kv_states` -- Gemma4
  shares KV across layers from a sharing point onward. MLP is plain GeGLU (`down(act(gate)*up)`),
  attention scaling=1.0 (both already match Mila). NEXT: read layer_scalar value(s) + with_scale flags +
  v_norm (prints added to hf_gemma_activation_dump.py), then wire into GemmaBlock: (a) multiply res2 by
  layer_scalar at end of prefill+decode, (b) add v_norm_ (per-head RmsNorm on V like q/k_norm), (c) honor
  with_scale (skip weight where False). KV-sharing is a follow-up after L0 parity.
  **2026-06-22 STEP 1a LANDED: `layer_scalar` (the dominant 18x fix) wired end-to-end (awaiting VS2026
  build + RE-CONVERT + 5f run).** (1) New reusable `TensorOps::scale(in, float, out, ctx)` in-place scalar
  multiply: interface in TensorOps.Math.ixx, Cuda `MathOps::scale`+`scaleImpl` (reuses existing
  `launch_scalar_multiply_kernel`, +added the missing `__nv_bfloat16` instantiation in Math.Elementwise.cu),
  Cpu `MathOps::scale`. Default-safe; Llama/GPT-2 unaffected. (2) Converter writes `tf_layer_{i}.layer_scalar`
  (HF `[1]` scalar, FP32). (3) `GemmaBlock`: `float layer_scalar_` (default 1.0), `loadParameter("layer_scalar")`
  override (loads [1] FP32 blob -> temp device tensor -> host copy -> float), and `scale(res2, layer_scalar_,
  res2)` at end of prefill+decode (dumped res2 reflects the scaled output). **RE-CONVERT REQUIRED** (an old
  checkpoint loads fine with layer_scalar_=1.0 = no-op). Expected: L0 res2 1634 -> ~86.6 (HF 88), residual
  trajectory tracks HF (no 18x). Remaining ~1.6% L0 gap is the still-missing `v_norm` (STEP 1b: per-head
  no-scale RMSNorm on V, converter writes ones[head_dim]). Global layers (5+) also need V=v_norm(k_proj) +
  KV-sharing (STEP 2). Norms RAW (Gemma4 != Gemma3) + attention scale 1.0 already correct.
  **2026-06-22 STEP 1a VALIDATED + STEP 1b (`v_norm`) LANDED.** After build+reconvert, `layer_scalar`
  fixed the WHOLE residual trajectory: Mila now tracks HF across all 48 layers (L0 86.6 vs 88.1, L8 111 vs
  111, L16 219 vs 220, L47 6.63 vs 6.63 EXACT; was 86->14357). **The first generated token now MATCHES HF**
  (greedy divergence moved from token 0 to token 1; was garbage 255999 from the start). Remaining: DIRECTION
  drift accumulates -> tokens 1+ diverge and degenerate (Mila loops 236786/495), final_norm 284 vs 304
  (~6.5%). STEP 1b `v_norm` (per-head V normalize, with_scale=False) wired: converter writes
  `tf_layer_{i}.v_norm.weight = ones[head_dim]` (256 local / 512 global; HF has no v_norm weight, ones makes
  Mila's RmsNorm a pure normalize); GemmaBlock adds `v_norm_` (per-head RmsNorm, built like k_norm) applied
  to V in the LOCAL prefill+decode attention branch (global V still uses the K=V alias -> STEP 2). Component
  present on global blocks too (loads the weight; unused until step 2). **RE-CONVERT REQUIRED** (v_norm.weight
  new). Expect local layers (0-4,6-10,...) to tighten in direction; parity should improve. If still
  diverging, STEP 2 = global `V = v_norm(k_proj)` (not the current k_norm+RoPE'd K alias) + KV-sharing.
  **Inference-server groundwork (kept, not the 5e/5f mechanism):** a `GemmaSession` (CUDA BF16) mirroring
  `LlamaSession` was added to the pybind layer ([Mila_py.Wrappers.ixx](Mila/Inference/Bindings/Mila_py.Wrappers.ixx)/
  `.cpp` + [Mila_py.cpp](Mila/Inference/Bindings/Mila_py.cpp)) — retained for the future Gemma-in-the-
  Inference-Server integration, independent of the converter/parity work above.
- [ ] **[minor, pre-existing] Llama metadata `norm_eps` vs reader `norm_epsilon` key mismatch.** The Llama
  converter writes `'norm_eps'` ([convert_weights.py](Mila/Tools/Converters/Llama/convert_weights.py)) but
  `PretrainedModelReader::parseMetadataJSON` reads `'norm_epsilon'`, so `metadata.norm_epsilon` is always 0
  for Llama checkpoints. Harmless today only because `LlamaModel::configFromMetadata` never calls
  `withRMSNormEpsilon` (RmsNorm uses the `LlamaConfig` default). Surfaced while wiring Gemma's eps through
  metadata (Gemma DOES read it, and its converter writes `'norm_epsilon'` to match the reader). Fix: align
  the Llama converter key to `'norm_epsilon'` (and have `LlamaModel` read it) when the Llama load path is
  next touched.
  Original: Assemble Steps 0-4
  into the two block instantiations (local/global), add QK-norm wiring + `final_logit_softcapping 30.0`
  (runtime scalar), and introduce the virtual **`IDecoderLayer`** interface (`prefill`/`decode`/`forward`)
  so the transformer holds the heterogeneous 5:1 layer list (final layer global) — every existing model
  is homogeneous and lacks this (`std::variant` alternative rejected; Gemma.md §8). Then `GemmaTransformer`,
  the HF->Mila converter (`Tools/Converters/`), and the HF token-for-token parity oracle.
- [ ] **Correctness-oracle dependency.** Component-level attention numerics are blocked until the
  `GroupedQueryAttention::forward` standalone-stub bug is resolved (see the GQA no-op-stub item under
  Test Suite Revival's bug list) — windowed-vs-global + local/global-geometry reference cases belong to
  an operation-level `CudaGqaOp` test owning the `GqaState` scratch + cache. Build that oracle as Steps
  1-2 land, not after.

- [ ] **Weight-tying optimization (Llama + Gemma) — Future Direction.** Mila currently stores the token
  embedding table and the `lm_head` projection as two SEPARATE parameter blobs even when the source model
  ties them (Llama 3.2 1B/3B, Gemma 4). Untied is the simpler load path and is what lets Gemma fold the
  embedding sqrt(d) scale into the table while keeping lm_head unscaled (Step 5d/5e decision above).
  Tying them back (lm_head reuses the embedding storage) would save one `vocab x model_dim` tensor in
  VRAM (Gemma 4: 262144 x 3840 x 2B ~= 2 GB at BF16) at the cost of a shared-ownership / load-aliasing
  path and re-introducing the scale-conflict the untied design sidesteps. Memory optimization, not a
  correctness gate; no milestone owns it yet.

**Reuse note:** the Step 2 SWA mask + bounded-KV ring cache are the foundation the **Ministral** Future
Direction reuses (SWA named explicitly there).
