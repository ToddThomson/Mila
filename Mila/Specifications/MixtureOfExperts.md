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
True for BF16, and true for block-scaled FP4 on SM100. **Not** true for
block-scaled FP4 grouped GEMM on SM120, the validation card: those templates
fail to initialize and emit garbage, and the known fix is not upstream
(Section 7.2(c)). Treat CUTLASS as one candidate backend for the BF16 grouped
path, not as the plan for the quantized one.

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

### 7.2 The SM120 question

The hardware is not in doubt. The RTX 5060 Ti (compute capability **12.0**,
16311 MiB, 36 SMs) has the NVFP4 block-scaled tensor-core instruction, and
dense block-scaled NVFP4 GEMM is known to work on SM120. What is in doubt is
the **toolchain and the kernel authorship**, and three separate issues are
routinely conflated into one. They are not the same problem and they do not
have the same answer.

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

**(c) CUTLASS grouped block-scaled GEMM on SM120 — broken, with a known
non-upstream fix.** TMA warp-specialized grouped GEMM tactics fail to
initialize under `compute_120a`, producing garbage output rather than an error.
Identical broken templates in CUTLASS 4.2.1 and 4.4.1. The fix is
`compute_120f` (CUDA 13.0+), which restores the warp-specialized tactics:
measured **14.6 -> 39.0 tok/s** single-user on an NVFP4 MoE model. It is **not
upstream** — it required patching 10+ files across FlashInfer and vLLM.
CUTLASS PR #3082 (`is_family_of()` for the SM12x arch guard) is the in-flight
upstream correction; its status should be re-checked before this work starts.

Note that those numbers are from **RTX PRO 6000** (188 SMs, 96 GB). The 5060 Ti
has 36 SMs. Do not transfer the throughput.

### 7.3 Why this is a redirection rather than a blocker

Mila does not use CUTLASS grouped GEMM. It writes its own W4A16 GEMM and its
own flash-attention kernels, and `FfnAndMoE.md` §8's "maps directly onto the
vendored CUTLASS grouped-GEMM kernels" was a convenience assumption, not a
constraint.

That matters more than it sounds, because the two properties that broke
everyone else's SM120 port are **design inputs** for a kernel written from
scratch rather than porting problems:

- SM120 uses warp-level `mma.sync`, **not** `tcgen05`/TMEM. Every SM100-derived
  kernel (DeepGEMM, CUTLASS SM100 collectives, WGMMA flash attention) fails to
  compile or crashes. This is the SM8x programming model **Mila's existing
  kernels already use**.
- SM120 has **99 KB shared memory per SM**, against SM100's 228 KB. Tile
  designs inherited from SM100 do not fit. A kernel designed against 99 KB from
  the start has no such inheritance.

The instruction is fixed-shape and non-tunable:
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`,
with quad-based scale-factor distribution (threads 0-1 supply SFA, thread 0
SFB). Published SM12x reference kernels reach ~60% of peak, and the working
consumer-Blackwell implementations got there with custom warp-level GEMMs
rather than CUTLASS template inheritance — the same approach Mila would take.

**Decision required before any NVFP4 work is scheduled:** measure, on the 5060
Ti, a hand-written SM120 NVFP4 blockscaled matvec against Mila's existing
`PerGroupFp4<128>` W4A16 path. If NVFP4 does not beat the FP4 kernel Mila
already has, the coupling buys nothing and MoE should land on
`PerGroupFp4<128>` alone. **Measure the baseline before proposing**, and state
the direction the number must move before running it.

### 7.4 Decoupling

MoE and NVFP4 are **two milestones, not one**. MoE lands on the existing
`PerGroupFp4<128>` and is complete and shippable there. NVFP4 is a policy
addition that any `Linear` and the `MoeOp` can adopt afterwards, on any model.
Coupling them makes the MoE milestone hostage to an unresolved kernel question
on a single card. Land MoE first.

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

`Chat.Footprint.ixx` and `MemoryFootprint.md` currently have **no term for a
sparse layer** — no notion of resident-but-inactive parameters. Adding it is a
prerequisite, not a follow-up: a footprint report that counts an MoE layer as
dense is wrong by a factor of 14 on the only number a user checks before
loading.

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
9. **NVFP4** — only after the Section 7.3 measurement says it earns its place.
   Its own step 0 is the `120f` build-flag correction of Section 7.2(b).

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
