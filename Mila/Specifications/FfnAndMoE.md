# Mila Feed-Forward Network and Mixture-of-Experts Specification

## Overview

This document specifies the feed-forward-network (FFN) component family and the
forward-looking layering that lets a Mixture-of-Experts (MoE) layer drop in
without reworking the FFN layer. It supersedes the original single `MLP`
component, which conflated two structurally different FFNs (dense and gated)
behind one runtime-configurable type.

The design rests on one taxonomic distinction — **elementwise activations versus
gated FFN structure** — and one execution decision for MoE — **grouped GEMM over
stacked expert weights (decision B)**. `Linear` remains the reference for the
component/operation dispatch pattern (see `OperationDispatch.md`); this spec
applies that same component-orchestrates / operation-holds-the-kernel division to
the FFN family.

---

## 1. Motivation

The original `MLP<TDeviceType, TPrecision>` tried to be a general configurable FFN
spanning both GPT-2 (dense) and Llama (gated) architectures, selecting the
activation from an `MLPConfig` enum at runtime. This proved structurally wrong:

- **Activation type was dispatched by a runtime `switch`.** Both the `Gelu` and
  `Swiglu` case bodies instantiate whenever `MLP<Device, Precision>` instantiates.
  The `Swiglu` case drags in `Swiglu<Cpu>`, which has no CPU operation, so
  **`MLP<Cpu>` is uncompilable regardless of the activation actually configured** —
  which cascades to `GptBlock<Cpu>` and `GptTransformer<Cpu>`, the exact CPU
  coverage the test suite needs.
- **The gated path was dead.** Nothing instantiates `MLP` + SwiGLU. Llama's block
  hand-wires its FFN (`fc_gate_up -> swiglu -> fc_down`) inline because the dense
  `MLP` shape contract (in -> hidden -> in) cannot express the gated 2H -> H split.
  The result is an asymmetry — one block delegates to a composite, the other
  inlines — that reflects a limitation of `MLP`, not a design choice.
- **Type erasure for no gain.** The activation is held behind a `Component` base
  pointer plus `std::function` forward/backward lambdas. That indirection buys
  runtime flexibility no caller uses (each architecture's activation is fixed), and
  forces compile-time instantiation of every activation variant anyway — worst of
  both models.
- **The activation enum is overwhelmingly elementwise.** Of nine `ActivationType`
  values, eight (`None, Relu, Gelu, Silu, Tanh, Sigmoid, LeakyRelu, Mish`) are
  elementwise unary functions differing only in a scalar function and its
  derivative; only `Swiglu` is structural. Modeling eight near-identical
  component/config/operation/test quadruples is duplication the enum is begging to
  collapse.

---

## 2. The Taxonomy: Elementwise vs. Gated

The dividing line that the original design erased, and that this spec restores as a
**type boundary**:

| Category | Shape contract | Examples | Home |
|---|---|---|---|
| **Elementwise unary** | `[..., H] -> [..., H]`, per-element `f` | GELU, SiLU, ReLU, Tanh, Sigmoid, LeakyReLU, Mish | `Components/Activations/` |
| **Gated structure** | `[..., 2H] -> [..., H]`, split + gate + multiply | SwiGLU, GeGLU, ReGLU, GLU | `Components/FFN/` |

A gated FFN *composes* an elementwise function on its gate half:
`down( gate_fn(W_gate . x) (*) (W_up . x) )`. The GLU family (Llama, Mistral, Qwen,
Gemma, DeepSeek, Phi-3) differs in exactly one thing — the gate function. So the
gate is parameterized by an elementwise function, and **the elementwise function is
the shared primitive both the dense and gated FFNs consume**.

`Swiglu` therefore moves out of `Activations/` (where it never belonged — it is not
a general-purpose activation) and into `Components/FFN/`, as the gate sub-structure
of the gated FFN.

---

## 3. Layering

```
Elementwise function primitives  (GELU/SiLU/ReLU/... functor + derivative; one library)
   |
   +-- Activation component (dense, [...,H] -> [...,H])  --->  MLP        (dense FFN: GPT-2)
   |
   +-- Swiglu gate op (split . gate_fn . multiply, 2H -> H) -->  GatedMLP  (gated FFN: Llama, ...)
                                                                    |
                                                                    +--->  MixtureOfExperts
                                                                           (Router + stacked experts + combine)
```

The expert in an MoE layer *is* a `GatedMLP`. Designing `GatedMLP` for
expert-replication from the start is the "do it properly" requirement, even though
the MoE layer itself is deferred.

---

## 4. Directory Layout and Naming

`FFN/` is the family directory; each FFN component gets a subdirectory holding its
`*.ixx` / `*.Config.ixx` / `*.Dispatch.ixx` trio, matching the existing
`Activations/Gelu/`, `Attention/MHA/`, `Normalization/LayerNorm/` convention.

```
Src/Dnn/Components/
  Activations/
    Activation/   Activation.ixx, Activation.Config.ixx, Activation.Dispatch.ixx   (unified elementwise)
    Gelu/         (retained as reference component; folds into Activation over time)
  FFN/
    MLP/          MLP.ixx, MLP.Config.ixx, MLP.Dispatch.ixx
    GatedMLP/     GatedMLP.ixx, GatedMLP.Config.ixx, GatedMLP.Dispatch.ixx
    Swiglu/       Swiglu.ixx, Swiglu.Config.ixx, Swiglu.Dispatch.ixx               (moved from Activations/)
```

The test tree mirrors this exactly (`Tests/Dnn/Components/FFN/MLP/`,
`FFN/GatedMLP/`, `FFN/Swiglu/`).

**Module names are flat** — `Dnn.Components.MLP`, `Dnn.Components.GatedMLP`,
`Dnn.Components.Swiglu`, `Dnn.Components.Activation` — not path-qualified with
`FFN`. The directory provides the grouping; the module name does not repeat it.
This is the established component convention.

| Concept | Component | Config | ComponentType |
|---|---|---|---|
| Dense FFN (GPT-2) | `MLP` | `MLPConfig` | `Mlp` |
| Gated FFN (Llama, MoE experts) | `GatedMLP` | `GatedMLPConfig` | `GatedMlp` |
| Gate sub-structure | `Swiglu` | `SwigluConfig` | `Swiglu` |
| Elementwise activation | `Activation` | `ActivationConfig` | `Activation` |

`MLP` and `GatedMLP` are **two distinct types sharing a name-family**, never one
polymorphic type branching on a gated flag. The structural split stays a type
boundary; collapsing it would resurrect the `fc1_out = 2*hidden` special-casing
this spec removes.

---

## 5. The Elementwise Activation Primitive

The eight elementwise activations collapse into a single component
`Activation<TDeviceType, TPrecision, ActivationType TFn>` — the function is a
**compile-time template parameter**, the same way `Linear` carries
`TWeightQuantization`. The `ActivationType` enum remains in `ActivationConfig` /
`MLPConfig` as serializable architecture *metadata*; the model factory performs the
one `switch(activationType)` that bridges the runtime enum to the compile-time
instantiation (the `QuantizationMode -> PerChannelFp8<>` pattern). Config-as-data
and component-as-type coexist: the config describes, the type realizes, the factory
bridges.

- **Function selection is hoisted to the factory boundary**, executed once at model
  build — not per `forward()`. Each `forward()` launches its one specialized kernel
  directly; the per-element hot path is branch-free with no warp divergence.
- An invalid combination — a non-elementwise function such as `Swiglu` used as an
  `Activation` — is a `static_assert`, matching the "missing specialization = hard
  compile error" contract.
- A single CPU op implementation yields GELU / SiLU / ReLU / Tanh / Sigmoid /
  LeakyReLU / Mish on CPU, closing the "CPU only has GELU" gap for the whole family.
- The same functor library is consumed by the `Swiglu` gate op (section 7), so the
  function math has one source of truth.

Function-specific *runtime* parameters (LeakyReLU `alpha`, GELU approximation
flavor) still ride in the config — `TFn` selects the function, the functor carries
the scalar by value. `Gelu` is retained initially as the reference activation
component; it folds into `Activation<…, ActivationType::Gelu>` as the unification
lands. Test-wise this is the typed-sweep methodology applied to the function
parameter.

### 5.1 Operation implementation (ElementwiseActivationOp)

The runtime `ActivationType` becomes a compile-time functor type **once, at the
model factory** (section 5). Below that boundary everything is specialized and the
per-element hot path is branch-free.

1. **Shared functor library.** Each activation is a small POD functor exposing
   `fwd(x)` and `df(x)` (or `df_from_y(y)` where cheaper, e.g. sigmoid / tanh),
   parameterized by value where needed (LeakyReLU `alpha`, GELU approximation). The
   functors are annotated `MILA_HD` (= `__host__ __device__` under nvcc, empty
   otherwise), so the same definitions compile for the MSVC-built CPU op and the
   nvcc-built CUDA op. `MILA_HD` is the one sanctioned preprocessor use, confined to
   the functor header's global module fragment. This library is the single source of
   truth, also consumed by the `Swiglu` gate op (it applies the same functor to its
   gate half).

2. **The op is templated on the functor, not on a traits axis.**
   `OperationTraits<ElementwiseActivationOp, Device, Precision>` resolves the op
   *template* — keeping the four traits axes clean — and the `Activation<…, TFn>`
   component maps `TFn -> functor` and instantiates the op with it
   (`CudaElementwiseActivationOp<Precision, Fn>`). There is no fifth traits axis.

3. **CUDA op.** A kernel templated on the functor and precision; nvcc inlines the
   functor, so every thread runs identical code (no divergence). Elementwise
   activation is memory-bound, so the kernel is a grid-stride loop with vectorized
   access (`float4`, `__nv_bfloat162` pairs for BF16, compute in float). The op
   launches its single specialized kernel directly — no host switch in `forward()`.

4. **CPU op.** The same functor-templated loop; the inlined branch-free loop
   auto-vectorizes (SIMD). One implementation yields all eight elementwise functions.

5. **Backward** uses the same functor: `dx = dy * f'(x)` (or from the cached forward
   output where the derivative is cheaper from `y`).

**Anti-pattern (explicitly avoided):** a single kernel with a per-element
`switch(type)`, or a per-element device function-pointer call. Both defeat
inlining, raise register pressure, and cause warp divergence. The cost of the
chosen design is N thin compiled kernel instantiations; the only enum switch is the
factory bridge, run once per model build.

---

## 6. MLP (Dense FFN)

```
input [..., in] -> fc1 Linear(in -> hidden) -> Activation(fn) -> fc2 Linear(hidden -> in) -> [..., in]
```

- `MLP<TDeviceType, TPrecision, ActivationType TActivation = ActivationType::Gelu>`
  holds an `Activation<…, TActivation>` child (GPT-2 pins GELU). **No activation
  polymorphism, no runtime switch, no `mlp_activation_impl`, no `std::function`
  bridge, no SwiGLU branch, no `fc1` doubling.** Removing this is the `MLP<Cpu>` /
  Bard CPU-test unblock.
- `MLPConfig`: `input_features`, `hidden_size`, `has_bias`, `activation`
  (elementwise enum, serialized metadata; the factory bridges it to `TActivation`).
  The dead LayerNorm-in-MLP remnant is removed.

---

## 7. GatedMLP (Gated FFN)

```
input [..., in] -> fc_gate_up Linear(in -> 2H, fused) -> Swiglu(split . gate_fn . multiply, 2H -> H)
                -> fc_down Linear(H -> in) -> [..., in]
```

- `GatedMLP<TDeviceType, TPrecision, ActivationType TGate = ActivationType::Silu>`
  holds a `Swiglu<…, TGate>` gate. The gate function is a **compile-time** parameter
  (same model as `Activation`'s `TFn`), bridged from `GatedMLPConfig.gate_activation`
  metadata at the factory.
- **Fused gate+up projection** (one `2H`-wide GEMM), matching Llama's existing
  `fc_gate_up_` and converter weight layout. The split happens inside the `Swiglu`
  gate op.
- `Swiglu<…, TGate>` gate op: `[..., 2H] -> [..., H]`, computing
  `TGate(gate) (*) up` via the shared functor library, so the one component expresses
  SwiGLU (SiLU), GeGLU (GELU), ReGLU (ReLU), etc. by `TGate`.
- `GatedMLPConfig`: `input_features`, `hidden_size`, `has_bias` (gated FFNs are
  typically bias-free), `gate_activation` (elementwise enum, serialized metadata).
- `LlamaBlock` delegates its FFN to `GatedMLP`, deleting its inline
  `fc_gate_up -> swiglu -> fc_down` wiring (and the per-step debug `synchronize()`
  calls — see section 11).

---

## 8. Mixture-of-Experts: Execution Model (Decision B)

The expert is a `GatedMLP`. The performant execution path is **decision B: a
grouped GEMM over stacked expert weights**, not a loop over N component instances.

- **`MixtureOfExperts` component** orchestrates routing and combination: a `Router`
  (Linear `hidden -> num_experts` + top-K + softmax over selected experts), token
  gather/dispatch, and the weighted combine of selected experts' outputs. It may
  also host always-on shared experts (DeepSeek style).
- **`MoeOp`**, resolved via `OperationTraits` like every other operation, owns the
  **grouped/segmented GEMM over stacked expert weights** `[E, in, 2H]` and
  `[E, H, in]`. This maps directly onto the vendored CUTLASS grouped-GEMM kernels.
  The hot path operates on stacked *data*, not N `GatedMLP` instances.
- `GatedMLP` remains the **single-expert reference and CPU semantics** — the
  correctness oracle the grouped op is validated against, and the small-`E` / CPU
  fallback. This is the same component-orchestrates / operation-holds-the-kernel
  division `Linear` established (prefill-GEMM vs. decode-matvec).

This decision constrains `GatedMLP`'s weight layout **now**: per-expert gate/up and
down weights must be packable into the grouped tensors, the same way `fc_gate_up`
already fuses gate+up. The converter / `PretrainedReader` fuses E experts into the
stacked tensors at load time.

---

## 9. MoE-Readiness Seams (required from day one)

These constraints apply to `MLP`, `GatedMLP`, and their child `Linear`s
immediately, even though the MoE layer is deferred. Getting them wrong forces a
rewrite when MoE lands.

1. **Shared, injected execution context — no per-expert owned context.** A
   `GatedMLP` constructed with a `DeviceId` currently *owns* an `ExecutionContext`
   (stream + cuBLAS handle). N experts each owning one is a non-starter at MoE
   scale. The expert must accept an **injected** context (the
   `optional<DeviceId> = nullopt` path); the parent (`MixtureOfExperts` / the model)
   owns the single context. Owned-context construction is the standalone-convenience
   case only.
2. **Leading-dimension-agnostic forward.** Operates on `[..., H]`, so it is valid
   whether fed `[B, T, H]` (standalone) or `[num_routed_tokens, H]` (gathered).
   The components already are; do not regress this.
3. **Stackable weight layout.** Per-expert weights laid out so the converter can
   pack `E` experts into the grouped `[E, ...]` tensors (decision B).
4. **Quantization on the expert Linears.** Experts are the bulk of MoE parameters,
   so FP8/FP4 matters most here; the grouped `MoeOp` must carry the same W4A16
   dequant-in-GEMM the `Linear` path already has (see `Quantization.md`).

---

## 10. OperationTraits Additions

| Operation | Used by | Notes |
|---|---|---|
| `ElementwiseActivationOp` | `Activation` | host switch selects function-specialized kernel; CPU + CUDA |
| `SwigluOp` (gate) | `Swiglu` | split + gate_fn + multiply; reuses the elementwise functor library; needs a CPU specialization (currently CUDA-only) |
| `MoeOp` | `MixtureOfExperts` | grouped GEMM over stacked expert weights; CUTLASS grouped kernels; quantization-aware |

A missing specialization remains a hard compile error by design. The CPU `SwigluOp`
gap is the one to file in BACKLOG; until it lands, `Swiglu<Cpu>` / `GatedMLP<Cpu>`
are a compile error only when actually instantiated, no longer dragged in by `MLP`.

---

## 11. Block Integration and Debug-Sync Removal

- `GptBlock` uses `MLP` (dense). `LlamaBlock` uses `GatedMLP` (gated), replacing its
  inline FFN.
- Both `GptBlock::forward()`/`backward()` and `LlamaBlock::forward()`/`backward()`
  carry a `synchronize()` after **every** component step — bring-up debug
  scaffolding. Llama's `prefill()`/`decode()` already comment these out and run
  correctly, proving they are unnecessary: single-stream ordering makes a downstream
  op wait for its upstream on-device, and the caller already synchronizes before the
  host reads logits (the Bard trainer does `forward(); synchronize(); copy(...)`).
  `GptBlock` has only one `forward()` for both training and inference, so its syncs
  were never stripped and burden GPT-2 generation too. Remove all block-internal
  per-step `synchronize()` calls; keep only caller-side host-read boundary syncs.

---

## 12. Sequencing

Ordered so each step is independently buildable and the early steps unblock the
in-flight Bard test revival without waiting on the full redesign.

1. **Immediate (Bard unblock):**
   - De-polymorphize `MLP` to a fixed elementwise `Activation` (dense FFN only).
     Makes `MLP<Cpu>` / `GptBlock<Cpu>` / `GptTransformer<Cpu>` compile.
   - Move `Swiglu` from `Activations/` to `FFN/Swiglu/`.
   - Strip the per-step debug `synchronize()` from `GptBlock` (and `LlamaBlock`).
   - Relocate `MLP` Src + tests into `FFN/MLP/`.
2. **GatedMLP:** add `GatedMLP` (fused gate+up -> `Swiglu` gate -> down) with the
   section-9 seams; `LlamaBlock` delegates to it; delete Llama's inline FFN.
3. **Activation unification:** collapse the eight elementwise activations into
   `Activation` + `ElementwiseActivationOp`; add the CPU `SwigluOp`. `Gelu` folds in.
4. **MoE (deferred):** `Router`, `MixtureOfExperts`, `MoeOp` grouped GEMM, combine,
   load-balance auxiliary loss; shared + routed experts.

---

## 13. Open Decisions

- **`Swiglu` naming vs. generalization.** Once the gate function is config-driven,
  `Swiglu` expresses GeGLU/ReGLU too, so the name (which implies SiLU) is a future
  rename candidate (`Glu` / `GatedActivation`). Retained as `Swiglu` for now per the
  flat-name component convention; revisit when a non-SiLU gate first ships.
- **Shared-expert modeling** (DeepSeek): whether shared experts are `GatedMLP`
  instances composed beside the router or a distinct always-on path. Decide when MoE
  is scheduled.
- **Expert parallelism across devices** is out of scope (Mila targets single-GPU);
  the layout must not preclude it but need not enable it now.
