# Mila Mixture-of-Experts Specification — Gemma 4 26B-A4B

## Status

Design of record for the MoE execution path and its first target, Gemma 4
26B-A4B. **Post-v0.20**; nothing here is committed to a release. It extends
`FfnAndMoE.md` §8/§9 (decision B) with the detail that only a concrete model
forces, and it **corrects three assumptions** in that spec that reading the real
checkpoint falsified (Section 3).

Configuration and weight inventory in Sections 1-2 were read on **2026-09-12**
from `google/gemma-4-26B-A4B-it` — `config.json` and
`model.safetensors.index.json` — not from a summary of them. Parameter
arithmetic in Section 8 is derived from those shapes and is **not measured**;
it is a sizing estimate, and `Chat.Footprint.ixx` remains the only authority on
what a load actually allocates.

---

## 1. Confirmed Configuration

| Property | Value |
|---|---|
| Layers | 30 |
| `hidden_size` | 2816 |
| Query heads | 16 |
| KV heads (sliding) | 8 |
| KV heads (global) | 2 |
| `head_dim` (sliding) | 256 |
| `global_head_dim` | 512 |
| `attention_k_eq_v` | true |
| `intermediate_size` (dense branch) | 2112 |
| `moe_intermediate_size` (per expert) | 704 |
| `num_experts` | 128 |
| `top_k_experts` | 8 |
| `layer_types` | 5 sliding : 1 full, repeating |
| `sliding_window` | 1024 |
| RoPE (sliding) | theta 10000, default |
| RoPE (full) | theta 1e6, proportional, `partial_rotary_factor` 0.25 |
| `vocab_size` | 262144 |
| `max_position_embeddings` | 262144 |
| `final_logit_softcapping` | 30.0 |
| `rms_norm_eps` | 1e-6 |
| `hidden_activation` | `gelu_pytorch_tanh` |

Every row above except the MoE block, `intermediate_size`, and the global KV
geometry is **already expressible in `GemmaConfig`**. The 26B differs from the
12B on two existing fields: 30 layers rather than 48, and
`num_global_key_value_heads` **2** rather than 1. `GemmaConfig` defaults to 1;
the 26B sets 2 through the existing `withNumGlobalKVHeads`. No new axis.

`intermediate_size` 2112 is exactly `3 x 704`. The dense branch is three
experts wide.

---

## 2. Confirmed Weight Inventory

Read from the safetensors index, decoder layer 0 (a **sliding** layer):

```
self_attn.q_proj.weight          self_attn.q_norm.weight
self_attn.k_proj.weight          self_attn.k_norm.weight
self_attn.v_proj.weight
self_attn.o_proj.weight

input_layernorm.weight
post_attention_layernorm.weight

mlp.gate_proj.weight             <- dense branch, width 2112
mlp.up_proj.weight
mlp.down_proj.weight

experts.gate_up_proj             <- STACKED [128, 2816, 1408]
experts.down_proj                <- STACKED [128, 704, 2816]

router.proj.weight
router.scale
router.per_expert_scale

pre_feedforward_layernorm.weight
pre_feedforward_layernorm_2.weight
post_feedforward_layernorm.weight
post_feedforward_layernorm_1.weight
post_feedforward_layernorm_2.weight

layer_scalar
```

Three facts this establishes, none of which were assumed:

1. **The experts ship pre-stacked.** `experts.gate_up_proj` and
   `experts.down_proj` are single tensors with a leading expert dimension. There
   is no per-expert tensor to fuse.
2. **There is no tensor named `shared`**, but there is an always-on dense
   `mlp.*` branch beside the routed experts. The always-on path exists; it is
   spelled as an ordinary MLP.
3. **The router is not a bare `Linear`.** It carries two additional learned
   tensors, `scale` and `per_expert_scale`, beyond `proj.weight`.

---

## 3. Corrections to `FfnAndMoE.md`

Three §8/§13 statements do not survive contact with this checkpoint. They are
recorded here rather than silently diverged from; §8 should be amended to point
at this document.

**(a) "The converter / `PretrainedReader` fuses E experts into the stacked
tensors at load time."** Not on this model's path — upstream already ships
stacked, so the load is a direct upload. The fusion path is still needed, but
for models that ship per-expert (Qwen3-30B-A3B, gpt-oss-20b), not for Gemma 4.
Stacking is therefore a **converter capability, not a load-time step**, and the
Gemma 4 path must not be built through a fuse-then-upload detour it does not
need.

**(b) "This maps directly onto the vendored CUTLASS grouped-GEMM kernels."**
**Correct as written, including for block-scaled FP4 on SM120.** An earlier
revision of this document claimed otherwise; that claim was the error, not §8.

Verified in the vendored tree:
`include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp` (`_array_`
is CUTLASS's name for the Ptr-Array/grouped variant), plus
`builders/sm120_blockscaled_mma_builder.inl`. The official example is
`examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu`
with `ArchTag = cutlass::arch::Sm120`, present since CUTLASS 3.9.0 and in every
version Mila has pinned. The field reports of garbage output on SM120 trace to
the **arch flag** (`compute_120a` versus `compute_120f`), not to a missing or
defective kernel — Section 7.2(b).

**CUTLASS remains the only route for the grouped path.** cuBLASLt has no
grouped/varying-shape matmul in CUDA 13.3 or 13.4 (`cublasLtGemmGroupedBatchedEx`
appears in documentation ahead of both headers), so unlike the dense path in
Section 7.1 there is no library alternative here.

**(c) §13 open decision, "whether shared experts are `GatedMLP` instances
composed beside the router or a distinct always-on path."** **Decided:** the
always-on branch is an ordinary `GatedMLP` composed beside the router, with its
own pre- and post-norms. It is not an entry in the expert bank, it is not
routed, and it must not be folded into the grouped path — its width (2112)
differs from an expert's (704), so folding it would break the uniform expert
extent the grouped kernel depends on.

---

## 4. Block Topology

What is established:

- The dense `mlp` branch and the routed-expert branch are **parallel, then
  summed** — not sequential.
- The router reads the **unnormalized** post-attention residual; the experts
  read `pre_feedforward_layernorm_2(x)`.
- The router is `RMSNorm(no scale) -> root-size scaling -> learnable scale ->
  Linear -> softmax -> top-k -> renormalize`. `router.scale` is the learnable
  scale in that chain.

What is **not** established and must not be guessed:

- The exact assignment of the five feed-forward norms to the two branches. Four
  are accounted for (`pre`/`post` per branch); `post_feedforward_layernorm_1`
  is the fifth and its consumer is unconfirmed.
- What `router.per_expert_scale` does — it is absent from the routing chain
  quoted above.
- What `layer_scalar` multiplies.
- Whether a **global** layer carries a `v_proj` at all, given
  `attention_k_eq_v: true`. Layer 0 is sliding; layers 5, 11, 17, 23 and 29
  were not inspected.

**Resolution method:** read `Gemma4MoEDecoderLayer.forward` in the HuggingFace
`modeling_gemma4.py`, and read the tensor **shapes** (not just names) for one
global layer from the safetensors header. Both are cheap and both are
prerequisites to Section 11 step 1. Nothing below depends on the answers except
the block wiring itself.

---

## 5. Component Decomposition

The chassis does not change. `GemmaBlock<TDeviceType, TPrecision, kGlobal,
TWeightQuantization, TKvCachePolicy>` behind `ITransformerBlock` already solves
heterogeneous layers, and **every** layer of the 26B is an MoE layer — there is
no dense/MoE interleave to model. The sliding/global split is the existing
`if constexpr (kGlobal)` selector and is untouched by this work.

```
GemmaBlock (unchanged attention half)
  |
  +-- pre_feedforward_layernorm    -> GatedMLP<..., Gelu>  (2112)  -+
  |                                                                 |-- sum -> residual
  +-- pre_feedforward_layernorm_2  -> MixtureOfExperts      (704)  -+
                                        |
                                        +-- Router      (proj + scale + top-k)
                                        +-- MoeOp       (stacked [128, ...])
```

New components:

| Concept | Component | Config | Operation |
|---|---|---|---|
| Routing | `Router` | `RouterConfig` | `RouterOp` |
| Expert bank + combine | `MixtureOfExperts` | `MixtureOfExpertsConfig` | `MoeOp` |

`MixtureOfExperts` orchestrates; `MoeOp` holds the kernel. This is the division
`Linear` established and `OperationDispatch.md` requires. There is **no**
`Expert` component and **no** `ExpertBank` — an expert is a row of a stacked
tensor, never an object. `GatedMLP` remains the single-expert CPU semantics and
the correctness oracle the grouped path is validated against, exactly as
`FfnAndMoE.md` §8 specifies.

**Prerequisite:** `GemmaBlock` currently **inlines** its FFN
(`fc_gate_up_ -> geglu_ -> fc_down_`, `Gemma.Block.ixx:232`) rather than
delegating to `GatedMLP` — the same asymmetry `FfnAndMoE.md` §7 removes from
`LlamaBlock`. The FFN slot must become a delegated component before anything
can be swapped into it. That refactor is independently valuable and carries no
MoE risk.

---

## 6. Execution Model

Two paths, split on `M`, mirroring the prefill-GEMM / decode-matvec division
`Linear` already carries.

**Prefill (`M > 1`) — grouped GEMM.** Tokens are permuted into expert-major
order, one segment per expert, and a grouped GEMM walks the segments against
`experts.gate_up_proj`. Segment extents are data-dependent, so the launch is
sized from a device-side histogram of the routing result.

**Decode (`M == 1`, `outer_size == 1`) — gather-matvec.** A single token visits
8 of 128 experts. Permutation, segmentation and grouped launch all cost more
than the arithmetic they organize. The decode path indexes the 8 expert rows
directly out of the `[128, ...]` tensor and runs a fused gather-matvec, never
materializing gathered weights.

This is the decode case Mila actually ships, and it is memory-bound on 8
expert-rows per layer. The prefill path is the one that touches enough weight
mass for a GEMM to matter.

**Routing is a first-class operation** (`RouterOp`), not a subroutine buried in
a Gemma kernel. Its output — expert indices and renormalized weights — is the
interface `MoeOp` consumes, and it is the seam a second MoE family plugs into
without touching the kernel.

---

## 7. Quantization, and the Gate on This Milestone

Experts are **22.8B of the model's 25.2B parameters** — 91%. Quantization
matters here and essentially nowhere else in this model, which is why
`nvfp4_experts_only`-style recipes exist upstream.

### 7.1 NVFP4 as a sibling policy

NVFP4 is **not** `PerGroupFp4<128>` with different constants. It is block-16
E2M1 with an **FP8 E4M3 scale per block** plus a per-tensor FP32 global scale —
a different storage format, so it is a **new `WeightQuantPolicy`**, not a
retuning of the existing one. `PerGroupFp4<128>` (FP32 per-group scales) stays
exactly as it is.

The seam already exists. `Policies.ixx` grew **companion-tensor traits** so
`PerGroupCodebook3` could carry its third-bit plane, with structural detection
and `if constexpr` guards at every consumer. NVFP4's FP8 block-scale plane is
the same shape of problem and rides the same mechanism. This is a real fit, not
an analogy.

Per `Quantization.md`, quantization is **offline**: `Tools/ExportArtifact` is
the only writer, weights declare their policy in
`__metadata__["mila_quantization"]`, and a load refuses a policy that is not
the compiled one. NVFP4 adds an export path and traits rows; it does not add a
runtime mode.

### 7.1a The DENSE path needs no kernel work at all — measured

**cuBLASLt already performs native NVFP4 GEMM on SM120**, and the numbers are
not marginal. `Mila/Profiling/Microbenchmarks/CublasLtScaleModes.cu`, both
cards, CUDA 13.3, Gemma 4 12B prefill shapes at M=1024:

| arm | 4070 (SM 8.9) | 5060 Ti (SM 12.0) |
|---|---|---|
| BF16 reference | 52.5-60.3 | 50.8-51.3 |
| FP8 + `SCALAR_32F` (ships today) | 103.9-119.9 | 183.0-197.2 |
| FP8 + `OUTER_VEC_32F` | no algorithm | no algorithm |
| FP4 + `VEC16_UE4M3` | no algorithm | **329.3-361.3 TFLOP/s** |

361 against the `mxf4nvf4` instruction ceiling of 415.6 is **88% of peak**, and
**1.8x the FP8 path Mila ships**. The mechanism is one descriptor attribute —
`CUBLASLT_MATMUL_DESC_A_SCALE_MODE = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`
with A and B as `CUDA_R_4F_E2M1` — not a kernel project. Every mode that matters
is in **CUDA 13.3**; 13.4 adds only packed MX-style layouts Mila does not use.

Two consequences that reshape the rest of this section:

- The dense expert GEMM, the shared-expert GEMM and every `Linear` in the model
  reach NVFP4 through the library. **CUTLASS is needed only for the grouped
  path** (Section 3(b)), where cuBLASLt has no equivalent.
- The remaining risk is **numerics, not throughput**: this is W4A4, and
  per-tensor FP8 activations already produced incoherent Gemma once. Block-16
  scaling is a far finer instrument than what failed, so it is open rather than
  lost — but it gates on token parity and coherent generation, never on the
  per-layer tolerance alone.

Also measured: `OUTER_VEC_32F` returns no algorithm on **either** card, so the
per-token scale epilogue cannot be folded into the GEMM that way.
`Fp8ActivationPrefill.md` attributes that limitation to Ada; it is not an Ada
limitation.

### 7.2 The SM120 question

The hardware is not in doubt, and Section 7.1a settles the dense path. What
remains in doubt is the **toolchain** for the grouped path, where three separate
issues are routinely conflated into one. They are not the same problem and they
do not have the same answer.

**(a) The `sm_100a` restriction is Python-DSL only — irrelevant to Mila.**
CUTLASS `BlockScaledMmaOp` hard-codes `admissible_archs = [Arch.sm_100a]` in
`python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py`, blocking `sm_120a` and
`sm_121a`. Open since 2025-11-22, no maintainer response. **The C++ API is not
restricted** and is what vLLM already uses for FP4 on SM120. Mila is C++; this
one does not touch us. An earlier reading of this spec treated it as a
general CUTLASS restriction — it is not.

**(b) The build flag is a live trap, and Mila is currently in it.** Plain
`sm_120` does **not** carry the block-scaled MMA capability; it needs the `a`
(arch-conditional) or `f` (family-conditional) suffix. The failure mode is the
dangerous one — not a compile error but a silent omission, with the assertion
surfacing far downstream. PyTorch shipped exactly this bug by stripping the
suffix during arch auto-detection.

Mila's presets set `CMAKE_CUDA_ARCHITECTURES "120"` — no suffix — in
`x64-release-blackwell` and in all four published-artifact lists. Nothing is
broken today (the preset description correctly notes nothing in Mila emits
sm_120 instructions yet), but this is **step 0** of any NVFP4 work, not a
detail to discover later.

Which suffix: dense block-scaled MMA is **family**-specific and available via
`120f` from PTX ISA 8.8; only **sparse** `mma.sp` with `.kind::mxf4nvf4`
requires arch-specific `120a`. Mila needs dense, so `120f` suffices and is
forward-compatible across the whole 12.x family — the better choice for the
published-artifact lists, where `120a` would pin to one architecture. This rig
has **CUDA 13.4** and **CMake 4.0.1**, both of which support the `f` suffix, so
this is expressible today with no toolchain upgrade.

**(c) CUTLASS grouped block-scaled GEMM on SM120 — present and supported; the
field failures are (b) in disguise.** The collective ships in every CUTLASS
version Mila has pinned, and `79d_blackwell_geforce_nvfp4_grouped_gemm.cu` is
an official SM120 example (Section 3(b) lists the verified paths). The reports
of garbage output are builds using `compute_120a`, where the TMA
warp-specialized tactics fail to initialize; `compute_120f` restores them,
measured **14.6 -> 39.0 tok/s** on an NVFP4 MoE model.

An earlier revision of this document called this a broken template with a
non-upstream fix. That was wrong: it is the **same arch-flag trap as (b)**,
which is the third place in this section where that flag turned out to be the
actual story. Read (b) as the root cause and this as a symptom.

Note that those numbers are from **RTX PRO 6000** (188 SMs, 96 GB). The 5060 Ti
has 36 SMs. Do not transfer the throughput.

CUTLASS's documented SM120 constraints do apply and shape any grouped kernel:
**TN layout only** (A row-major, B column-major), **cluster fixed at 1x1x1**
(no multicast), NVFP4 tile shapes limited to `128x128x128`, `256x128x128` and
`128x128x256`, and `EpilogueScheduleAuto` mandatory. The 128-row minimum tile
is an independent reason the decode path cannot use the grouped kernel: at
`M == 1` it would waste 127 of 128 rows, which is Section 6's gather-matvec
split arrived at from the kernel side.

### 7.3 What Mila writes, and what it does not

Nothing in Mila includes a CUTLASS header today; only the include directory is
wired. That is a fact about the present, not a design position, and the split
after Section 7.1a is clean:

- **Dense GEMM — the library.** cuBLASLt reaches 88% of the instruction ceiling
  on NVFP4 and 88-96% on FP8. Mila's own history says what happens to a hand
  kernel here: the WMMA FP4 GEMM measured 4x slower at Stage 1 and 1.1-2x
  slower at Stage 2, both against cuBLASLt. Do not re-litigate this.
- **Grouped GEMM — CUTLASS.** No library alternative exists (Section 3(b)).
- **Hand-written — only where neither serves.** The decode gather-matvec, whose
  shape no GEMM library expresses and which the 128-row minimum tile rules out
  of the grouped kernel anyway.

Useful context if a hand-written kernel is ever reached for: SM120 uses
warp-level `mma.sync`, **not** `tcgen05`/TMEM, so every SM100-derived kernel
(DeepGEMM, CUTLASS SM100 collectives, WGMMA flash attention) fails to compile
or crashes — while the SM8x programming model Mila's kernels already use ports
directly. SM120 also has **99 KB shared memory per SM** against SM100's 228 KB,
so inherited SM100 tile designs do not fit. The instruction is fixed-shape:
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`,
quad-based scale distribution, published SM12x references at ~60% of peak —
which is the number to weigh against cuBLASLt's 88% before writing anything.

**The decision NVFP4 now gates on is numerics, not throughput.** Throughput is
measured and favourable. Validate W4A4 against Gemma token parity and coherent
generation before adopting it; if FP4 activations cannot hold, the dense path
stays on `PerGroupFp4<128>` and MoE lands there unaffected.

### 7.4 Decoupling

MoE and NVFP4 are **two milestones, not one**. MoE lands on the existing
`PerGroupFp4<128>` and is complete and shippable there. NVFP4 is a policy
addition that any `Linear` and the `MoeOp` can adopt afterwards, on any model.

The reason to keep them apart has changed but not weakened. It was "do not make
MoE hostage to an unresolved kernel question"; the kernel question is now
answered and the open one is whether FP4 **activations** hold their quality.
That is a per-model numerics investigation with its own oracle runs, and
nothing about the router, the expert bank or the grouped dispatch depends on
its outcome. Land MoE first.

---

### 7.5 The NVFP4 pipeline — the activation half is what remains

Mila is most of the way to an end-to-end NVFP4 pipeline already. Stating what
ships is the point of this section, because the remaining work is far narrower
than "add NVFP4" suggests and the architecture it plugs into is the one Qwen 3.8
established.

**Shipped:**

- **Quantized weights, several formats.** `PerGroupFp4<128>` (E2M1 with an FP32
  per-group scale), `PerChannelFp8<>`, `PerGroupCodebook2/3`. Offline through
  `Tools/ExportArtifact`, declared in `__metadata__["mila_quantization"]`, and a
  load refuses a policy that is not the compiled one.
- **Companion-tensor traits.** `Policies.ixx` already serves a format whose
  storage spills into a second plane, structurally detected, `if constexpr` at
  every consumer. NVFP4's block-scale plane needs no new mechanism.
- **Per-role bit allocation.** `WeightQuantization::Plan` and
  `Qwen.PrecisionPlan.ixx` — a family's designed allocation read from a
  pre-quantized artifact, with no quantize-on-load path because the codes are
  fitted offline.
- **Runtime activation quantization.** `cuda_quantize_bf16_to_fp8_per_token`
  produces FP8 E4M3 with per-token absmax scales, shipped on and gated green.
- **The GEMM.** Measured at 329.3-361.3 TFLOP/s for FP4 x FP4 with
  `VEC16_UE4M3` (Section 7.1a). No kernel to write.

**What remains is one kernel and one vocabulary extension.**

**(a) An activation quantizer at block-16 granularity.** The existing quantizer
emits one E4M3 scale per token; NVFP4 wants `e2m1` values with a `ue4m3` scale
per 16 elements along K. Same shape of kernel — one pass over the activations,
absmax then quantize — at a different granularity, writing the scale plane
cuBLASLt expects. The layout is a contract, not a choice: the scale tensor must
match what `VEC16_UE4M3` reads, and getting it wrong produces wrong numbers
rather than an error, so it needs a CPU reference codec the way
`CodebookPacking.ixx` has one.

The cost trade is favourable and worth recording, because it inverts the usual
objection to activation quantization. The pass this **deletes**
(`dequantize_fp4_to_fp8`) is `O(weights)`; the pass it **adds** is
`O(tokens x hidden)`. At a 512-token prompt that is the difference between
51.6% of prefill and something near noise — the weight dequant is expensive
precisely because it is paid per forward regardless of how few tokens are in
flight.

**(b) Plans must allocate activation precision, not only weight bits.**
`LanguageModelConfig` is explicit that "a plan allocates WEIGHT bits", and that
is the line NVFP4 crosses. A uniform W4A4 model is the aggressive variant; the
useful one holds the layers that carry accuracy at higher precision and runs the
rest at W4A4. That is the same idea a precision plan already encodes, extended
one axis. The expert bank is 91% of this model's parameters and the obvious
W4A4 candidate; the router, the norms and the embedding are the obvious
holdouts.

**What has to be validated, and why the FP8 result does not predict it.**
Per-tensor FP8 activations failed here once: one outlier token set the tensor
scale, crushed every other token's resolution, and the error compounded over 48
layers. That is an argument about **granularity**, and NVFP4 moves granularity
by three orders of magnitude — one scale per 16 elements against one per tensor.
It is a reason to expect a different outcome, not the same one. The gate is
unchanged and is not the per-layer tolerance: **token parity against the
reference, and coherent generation**, because 30 layers of top-8 routing
compound and a 5e-2 per-layer pass has already proven it can hide that.

Sequencing note: (a) is independently useful. An NVFP4 activation quantizer plus
the `VEC16_UE4M3` GEMM is a **dense-model** win on every `Linear` Mila has, on
any Blackwell card, with no MoE machinery involved. It does not need this
document's model to land.

---

## 8. Memory Footprint

Derived from Section 1-2 shapes. **Not measured.**

Per layer: attention 34.6M (sliding) + dense branch 17.8M + experts 761.3M +
router 0.4M = **814.1M**, of which **761.3M (93.5%) is the expert bank**.

| | Parameters |
|---|---|
| Expert banks (30 layers) | 22.84B |
| Everything else (attention, dense branches, router, embedding) | 2.32B |
| **Total** | **25.16B** |
| Active per token | ~3.0B + head |

Weight residency, 5060 Ti (16 GiB, roughly 15.0-15.5 GiB usable after display
and driver):

| Experts | Rest | Total |
|---|---|---|
| NVFP4 (11.97 GiB) | BF16 (4.32 GiB) | **16.29 GiB — does not fit** |
| `PerGroupFp4<128>` (11.30 GiB) | BF16 (4.32 GiB) | **15.62 GiB — does not fit** |
| NVFP4 (11.97 GiB) | FP8 (2.16 GiB) | 14.13 GiB — fits, ~1 GiB headroom |
| `PerGroupFp4<128>` (11.30 GiB) | FP8 (2.16 GiB) | 13.46 GiB — fits |

**The "experts quantized, everything else BF16" recipe does not fit this card.**
The non-expert mass is only 9% of the parameters but 4.32 GiB at BF16, and the
embedding alone is 1.38 GiB of that. A 26B-A4B build for the 5060 Ti must
quantize the non-expert mass too. This contradicts the upstream
`nvfp4_experts_only` recipe, which targets cards with room to spare.

KV cache, by contrast, is the easy half — the bounded ring cache does the work:

- 25 sliding layers, capped at 1024 entries: **~0.20 GiB, independent of
  context length.**
- 5 global layers at 32K context: ~0.63 GiB (~0.31 GiB if `k_eq_v` halves it).

Only the 5 global layers scale with context. This is the strongest argument for
the bounded-KV `TKvCachePolicy` sibling in `SlidingWindowKvCache.md` landing
before, not after, this model.

**NVFP4 KV does not help this model, and the reason is worth stating so it is
not tried.** An NVFP4 KV cache halves KV against FP8 at under 1% accuracy loss,
which is a real result — but Gemma 4's 5:1 sliding-window pattern has already
made KV the cheap half here. Halving 0.83 GiB saves ~0.4 GiB against a weight
problem measured in whole gigabytes. It is also not free: values are
dequantized from NVFP4 to FP8 *before* attention, so it adds a pass of exactly
the kind Section 7.1a removes elsewhere. Where it would matter is a family with
unbounded KV — Llama 3.1 8B at 49152 context holds ~3.1 GiB of FP8 KV, and
halving that is worth having. File it against Llama, not against this model.

### 8.1 Expert residency — an open design axis

Everything above assumes the whole expert bank is resident. That assumption is
worth challenging rather than inheriting, because the **working set is far
smaller than the resident set**: top-8 of 128 means a token touches 6.25% of
each layer's experts.

The arithmetic bounds it before any design work. Per decode token, 8 experts x
5,947,392 parameters x 30 layers = **1.43B parameters**, or **~0.71 GB at 4
bits**. The 5060 Ti negotiates **PCIe Gen5 x8 (~31.5 GB/s)** — measured, and
better than the 4070's Gen4 x4 (~7.9 GB/s), which corrects a note that had the
slots the other way round. That puts a hard ceiling of ~22.7 ms per token, so
**~44 tokens/s if every active expert is streamed every token** — against 57.0
tok/s measured for a resident 8B FP4 model. Streaming everything is therefore
not a free lunch; it is roughly a halving, before any compute.

Three things decide whether a partial-residency design beats that ceiling, and
none is answered here:

1. **Routing locality.** A resident cache of hot experts only pays if the hit
   rate is high. Nothing in this document knows Gemma 4's routing entropy, and
   it is measurable offline from the router weights plus a corpus — cheap, and
   it gates the whole idea.
2. **The prefetch window is closed by a serial dependency.** Layer N+1's router
   consumes layer N's output, so the next layer's expert set is not knowable
   while the current layer computes. There is no cross-layer prefetch without
   speculating on the routing, which is a different and larger design.
3. **Prefill does not benefit.** A 1024-token chunk with top-8-of-128 touches
   essentially every expert in every layer, so the full bank must be resident
   during prefill regardless. Residency streaming is a **decode-only**
   technique, and it therefore cannot solve the fit problem Section 8 states —
   only the steady-state one.

The honest framing: this is an axis for running a model that **otherwise would
not load at all**, not an optimization for one that fits. Decide it after the
Section 8 table, not instead of it.

`Chat.Footprint.ixx` and `MemoryFootprint.md` currently have **no term for a
sparse layer** — no notion of resident-but-inactive parameters. Adding it is a
prerequisite, not a follow-up: a footprint report that counts an MoE layer as
dense is wrong by a factor of 14 on the only number a user checks before
loading. A residency design would need a second term on top, separating
resident from active from streamed.

---

## 9. Chassis Deltas

Beyond the dense Gemma 4 chassis, which is otherwise reused unchanged:

1. `GemmaBlock` FFN slot delegated to a component (Section 5 prerequisite).
2. `Router` + `RouterOp`, including the `scale` / `per_expert_scale` semantics
   from Section 4.
3. `MixtureOfExperts` + `MoeOp`, both paths of Section 6.
4. Parallel dual-FFN wiring and its five norms.
5. `layer_scalar`, semantics unknown.
6. `GemmaConfig`: expert count, top-k, `moe_intermediate_size`, and the
   dense-branch width as a field distinct from the expert width.
7. Converter: direct stacked upload; `1.0 +` norm-weight convention extended to
   the three new norms.
8. Footprint: a sparse-layer term (Section 8).

Items 1, 2, 3 and 8 are reusable by every future MoE family. Items 4, 5, 6 and
7 are Gemma-specific.

---

## 10. Validation

The oracle discipline is unchanged: token-for-token against HuggingFace on the
same weights, per `Testing.md`.

- `MoeOp` is validated against a loop of `GatedMLP` over the same stacked
  weights — the §8 oracle. This runs on CPU and needs no card.
- `Router` is validated against the HF routing chain on fixed hidden states;
  index agreement is exact, weight agreement is `atol`-bounded. **A
  must-differ threshold here is absolute, never a multiple of `atol`.**
- Routing is discrete, so a near-tie that flips one expert selection changes
  the output materially while the norms stay small. Generation must be
  validated, not just the oracle — 30 layers of top-8 compound.
- The decode gather-matvec is validated against the prefill grouped path at
  `M == 1`, which is the only comparison that isolates it.

---

## 11. Sequencing

Each step is independently buildable and independently valuable.

1. **Resolve Section 4's open questions** — read `modeling_gemma4.py` and one
   global layer's tensor shapes. Hours, no card, and it gates the block wiring.
2. **Delegate `GemmaBlock`'s FFN to `GatedMLP`.** No MoE risk; closes an
   existing `FfnAndMoE.md` §7 asymmetry.
3. **Bounded-KV `TKvCachePolicy` sibling**, if not already landed. Section 8
   shows it is what makes the context story work at all.
4. **Footprint sparse-layer term.**
5. **`Router` + `RouterOp`**, validated against HF routing in isolation.
6. **`MixtureOfExperts` + `MoeOp` prefill path**, validated against the
   `GatedMLP` oracle on CPU.
7. **`MoeOp` decode gather-matvec**, validated against step 6 at `M == 1`.
8. **Converter + `fromPretrainedImpl`**, on `PerGroupFp4<128>`. **The model
   runs here.**
9. **NVFP4** — throughput is measured and favourable (Section 7.1a), and the
   remaining work is scoped in Section 7.5: a block-16 activation quantizer with
   its CPU reference codec, and precision plans extended to allocate activation
   precision. Its own step 0 is the `120f` build-flag correction of Section
   7.2(b), without which none of it compiles.

   **This step does not belong to this model.** An NVFP4 activation quantizer
   plus the `VEC16_UE4M3` GEMM is a dense-model win on every `Linear` in the
   tree; it should land against a model that already works, where token parity
   is a known quantity, rather than against a chassis being brought up at the
   same time.

Steps 1-8 have no dependency on NVFP4 and no dependency on CUTLASS.

---

## 12. Open Decisions

- The four Section 4 unknowns, with the resolution method named there.
- Whether the non-expert mass quantizes to FP8 or FP4 (Section 8 requires one
  of them; which is a quality measurement, not a design choice).
- Whether `RouterOp` earns a CPU specialization, or whether `Router<Cpu>` is a
  compile error like `Swiglu<Cpu>` was.
- Load-balancing auxiliary loss is **out of scope** — Mila does not train this
  model, and the routing weights ship fitted.
- Expert parallelism across devices remains out of scope per
  `FfnAndMoE.md` §13; the stacked layout must not preclude it and need not
  enable it.
