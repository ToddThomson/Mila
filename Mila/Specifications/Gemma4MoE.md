# Mila Gemma 4 26B-A4B Implementation Record

## Overview

This is the implementation record for `google/gemma-4-26B-A4B-it`: the phases, the gate each must
pass, and what each phase actually found. **It is not the design.** The design of record is
`MixtureOfExperts.md` — configuration, weight inventory, component decomposition, execution
model, footprint and sequencing all live there, and this document points at its sections rather
than restating them. Where a phase finds that the design is wrong, the design is corrected in the
same change and the evidence is recorded here.

Post-v0.20; nothing here is committed to a release.

**The record ends at `PerGroupFp4<128>`.** NVFP4 is `MixtureOfExperts.md` §11 step 9 and does not
belong to this model: its activation quantizer lands on a dense model whose token parity is
already known, so that a fault in the router, the expert dispatch and FP4 activations is never
debugged all at once with no working arm to bisect against.

---

## Phasing

Phases map one-to-one onto `MixtureOfExperts.md` §11. Each is independently buildable, and
nothing is scheduled.

**Phase 1 — resolve the block topology** (step 1). No code, no card. *Exit:* every tensor in one
global and one sliding layer has a named consumer and a verified shape. **Done 2026-09-12.**

**Phase 2 — delegate `GemmaBlock`'s FFN to `GatedMLP`** (step 2). No MoE code. Split in two,
because the nested component renames every published FFN tensor (`tf_layer_i.fc_gate_up` becomes
`tf_layer_i.mlp.fc_gate_up`) and rc.1 would pay a republish of a published model for work that
ships after 0.20.0:

- **2a — the component and the block wiring, now.** `GatedMLP` gains the block's weight
  quantization, pooled activation slots and a footprint prediction, and is proven equal to the
  block's inline chain. `GemmaBlock` and `GemmaTransformer` gain a trailing
  `bool kDelegatedFeedForward = false`; set, every block's FFN is a `GatedMLP` child named `mlp`.
  `GemmaModel` never sets it, so no published surface moves. *Exit:* `GatedMLP.Cuda.cpp` green,
  including bit-identical output against `Linear -> Swiglu<Gelu> -> Linear` at BF16 and
  `PerGroupFp4<128>`; and `Gemma.DelegatedFeedForward.Cuda.cpp` green — a sliding plus a global
  layer, one saved weight set loaded into both wirings, bit-identical prefill and decode logits at
  FP32, BF16 and `PerGroupFp4<128>`, an unchanged and correctly predicted footprint, and a flat
  vocabulary that differs from the inline one only by the FFN projections moving under `mlp`.
  **Done 2026-09-12**, every gate green on the RTX 4070 and the RTX 5060 Ti, each pinned by UUID, and
  forced to fail once.
- **2b — the switch, after the 0.20.0 tag.** Flip the default, delete the inline branch and the
  flag together, and move Llama in the same pass so both families republish once. *Exit:* Gemma 4
  12B token parity unchanged under `GemmaModel.Parity.Cuda.cpp`, footprint drift gates unchanged,
  packages re-exported and republished, full suite green.

**Phase 3 — bounded-KV `TKvCachePolicy` sibling** (step 3). *Exit:* `SlidingWindowKvCache.md`'s own
gates. **Already landed** before this record began: `GemmaModel` routes `SlidingWindowKvCache` to
the local layers (`GemmaModel.ixx:98`), and the 26B reuses that chassis.

**Phase 4 — footprint sparse-layer term** (step 4). *Exit:* `Chat.Footprint.ixx` reports resident
and active parameters separately for an MoE configuration; every dense family's report is
unchanged.

**Phase 5 — `Router` + `RouterOp`** (step 5). *Exit:* on fixed hidden states and real layer
weights, expert indices agree exactly with `Gemma4TextRouter` and combine weights agree within a
tolerance written down before the run. **Done 2026-09-12**, CPU and CUDA, CUDA on both cards — see
Phase 5 below.

**Phase 6 — `MixtureOfExperts` + `MoeOp` prefill** (step 6). *Exit:* matches a loop of `GatedMLP`
over the same stacked weights, on CPU. **Done 2026-09-12**, CPU and CUDA, CUDA on both cards —
against HuggingFace's eager `Gemma4TextExperts` and a definition in the test instead, because
`GatedMLP<Cpu>` does not compile (see Phase 6 below). The CUDA kernel is the simple two-pass one; the
grouped GEMM is an optimization to gate against it later.

**Phase 7 — `MoeOp` decode gather-matvec** (step 7). *Exit:* matches Phase 6 at `M == 1`. The
Phase 6 CUDA kernel already reads each token's selected expert rows in place, which is the
gather-matvec shape, so Phase 7 now reduces to its exit: gate `M == 1` explicitly.

**Phase 8 — converter + `fromPretrainedImpl`** (step 8). *Exit:* hidden-state parity against the HF
reference on a short prompt at BF16, then a `PerGroupFp4<128>` load on the 16 GiB 5060 Ti with
measured VRAM inside the `MixtureOfExperts.md` §8 row it was built for, and coherent generation.
The BF16 model is ~47 GiB and fits neither card, so the reference runs layer-streamed —
`Qwen3.8.md` §8, *The layer-streamed HF reference*, is the method.

---

## Phase 1 — Block Topology (2026-09-12)

### Sources

- `transformers` **5.12.1**, `models/gemma4/modeling_gemma4.py` (the converter venv's copy):
  `Gemma4TextDecoderLayer`, `Gemma4TextRouter`, `Gemma4TextExperts`, `Gemma4TextAttention`,
  `Gemma4RMSNorm`.
- Tensor **shapes** from the safetensors headers of both shards, for layer 5 (global) and layer 6
  (sliding), read by HTTP range request.
- `config.json` `text_config`.
- **Values** of `layer_scalar` for all 30 layers and of `router.per_expert_scale` for layer 5, read
  from the weights.

`MixtureOfExperts.md` §4 named the class to read `Gemma4MoEDecoderLayer`. There is no such class:
the MoE layer is `Gemma4TextDecoderLayer` with `enable_moe_block` set.

### Unknown 1 — the five feed-forward norms

```
residual = h                                          post-attention residual stream
d = post_feedforward_layernorm_1( mlp( pre_feedforward_layernorm( residual ) ) )
e = post_feedforward_layernorm_2( experts( pre_feedforward_layernorm_2( residual ), router( residual ) ) )
h = residual + post_feedforward_layernorm( d + e )
h = h * layer_scalar
```

`post_feedforward_layernorm_1` is the **dense branch's** post-norm. The unsuffixed
`post_feedforward_layernorm` belongs to neither branch: it normalizes the **sum**. So the §4
accounting ("pre/post per branch, plus a fifth") had the wrong shape, not just a missing entry —
there are two norms per branch plus one on the join.

This is the cheap outcome for Mila. The unsuffixed norm sits exactly where the dense 12B block
already runs `post_ffn_norm_` (`Gemma.Block.ixx:236`), between the FFN output and `res2_`. The MoE
layer keeps that norm, its position and its converter mapping, and inserts a two-branch sum in
front of it. The three new norms are `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1`
and `post_feedforward_layernorm_2`.

### Unknown 2 — `router.per_expert_scale`

```
x    = rms_norm_without_scale( residual ) * router.scale * 2816^-0.5
p    = softmax( router.proj( x ) )              over all 128 experts
w, i = topk( p, 8 )
w    = w / sum( w )
w    = w * per_expert_scale[ i ]
```

`per_expert_scale` multiplies each **selected** expert's combine weight, after renormalization. It
runs after `topk`, so it never changes which experts run — only how much each contributes — and
the final combine weights do **not** sum to 1.

Layer 5 values: min **0.9805**, max **1.0234**, mean **1.0007**, std 0.012 — within about 2% of 1.
Omitting it therefore moves each expert's contribution by at most ~2%: small enough that no
per-layer tolerance would notice and the post-sum norm would absorb most of it, and exactly the kind
of error that only token parity catches.

> **Corrected 2026-09-12.** This section first recorded 1.870-1.878 (mean 1.874) and concluded that
> omitting the scale would halve the routed branch. Those numbers came from a hand-written decoder
> that read the bytes as IEEE float16 (5-bit exponent) when the tensor is bfloat16 (8-bit exponent).
> The values above are torch's decode, captured by the Phase 5 reference script. The `layer_scalar`
> table below had the same error.

The softmax is over **all 128** logits before selection, not a softmax over the top 8. `RouterOp`
must reproduce that order; at 128 wide the cost is negligible.

Two details a `RouterOp` implementer would otherwise guess:

- The router and `pre_feedforward_layernorm_2` both read the **same** unnormalized residual. The
  router applies its own **unscaled** RMSNorm; `pre_feedforward_layernorm_2` has a learned weight.
  They are two different norms of one input, not a shared one.
- `router.scale` is a `[2816]` vector, not a scalar.

**Open, not decided:** because an expert's output is linear in its combine weight,
`per_expert_scale[e]` folds exactly into row `e` of `experts.down_proj`, offline, before
quantization. That would leave `RouterOp` a pure selection operation. Against it: the exported
weights would no longer equal the checkpoint tensor-for-tensor, which is what the Phase 5 and
Phase 6 oracles compare. Keep it in `RouterOp` through Phase 5; decide at Phase 8.

### Unknown 3 — `layer_scalar`

It multiplies the whole layer output, after the second residual add. That is the operation the
dense Gemma 4 chassis already runs — `Gemma.Block.ixx:241` (prefill) and `:315` (decode), loaded to
a host float at `:586`, written FP32 by the converter. **It is not a chassis delta.**

| Layers | `layer_scalar` (torch decode) |
|---|---|
| 0 | 0.0703 |
| 1-2 | 0.2031, 0.1699 |
| 3-28 | 0.5156-0.8164 |
| 29 | 0.1953 |

The first and last layers are scaled down hardest, so the dense 12B's "varies wildly per layer"
(`convert_weights.py`) holds here too.

### Unknown 4 — does a global layer carry `v_proj`

**No.** From the index and headers:

| Tensor | Layer 5 (global) | Layer 6 (sliding) |
|---|---|---|
| `q_proj` | `[8192, 2816]` | `[4096, 2816]` |
| `k_proj` | `[1024, 2816]` | `[2048, 2816]` |
| `v_proj` | **absent** | `[2048, 2816]` |
| `o_proj` | `[2816, 8192]` | `[2816, 4096]` |
| `q_norm`, `k_norm` | `[512]` | `[256]` |

`V = v_norm(k_proj(x))` and `K = RoPE(k_norm(k_proj(x)))` — the identity the dense chassis already
implements (`Gemma.Block.ixx:206`), with the converter already writing `[Q | K]` for global layers
(`convert_weights.py:285`). **Not a chassis delta either.**

It does correct `MixtureOfExperts.md` §8, which offered that `k_eq_v` might halve the global KV
cache. It cannot: K passes through `k_norm` and RoPE, V through `v_norm`, so they differ and both
are cached. `k_eq_v` saves the `v_proj` weights and nothing else.

### Also established

**Expert tensor layout.** `experts.gate_up_proj` is `[128, 1408, 2816]` and `experts.down_proj` is
`[128, 2816, 704]` — PyTorch's `[experts, out, in]`. `MixtureOfExperts.md` §2 had both transposed;
corrected there. `gate_up_proj` chunks **gate first, then up**, and the activation applies to the
gate half — the same `[gate | up]` order the converter already writes for `fc_gate_up`. Each row of
the stacked tensor is therefore laid out exactly like the dense block's fused projection.

**Norm convention.** `Gemma4RMSNorm` multiplies by the **raw** weight; there is no `1 +` offset.
Mila already runs every Gemma norm at unit offset 0 (`Gemma.Block.ixx:838`), so the three new norms
convert raw like the rest. `MixtureOfExperts.md` §9 item 7 said the `1 +` convention would need
extending to them; that was wrong, and it was inherited from three stale comments claiming the
offset — the converter's module docstring, `_rmsnorm_to_numpy`, and `Gemma.Block.ixx`'s file
header — all corrected in the same change.

**Global attention parameters.** A global layer's attention is **49.0M**, not the 34.6M §8 used
for every layer. The total moves from 25.16B to **25.23B** and the non-expert mass from 2.32B to
**2.40B**; the §8 residency table is recomputed. No fit verdict changes.

**What `config.json` rules out.** `hidden_size_per_layer_input = 0` (no per-layer-input branch),
`num_kv_shared_layers = 0` (no KV sharing), `attention_bias = false`, `tie_word_embeddings = true`.
`layer_types` is `sssssF` five times, so the global layers are **5, 11, 17, 23 and 29**.
`use_bidirectional_attention = "vision"` affects image tokens only; the text path is causal. The
checkpoint also carries `model.vision_tower.*`, which the text model does not read.

### The chassis delta after Phase 1

Of the eight items in `MixtureOfExperts.md` §9, one disappears (`layer_scalar`) and two shrink (the
norms, the converter). Attention — both kinds — the embedding, `v_norm` and `layer_scalar` are
reused from the dense chassis untouched.

### Not answered by Phase 1

- **Which experts implementation the HF reference ran.** `Gemma4TextExperts` is wrapped in
  `@use_experts_implementation`, which can substitute a batched kernel for the loop in the source.
  Phases 5-8 must record which implementation produced their reference, or a numerics difference
  between HF's own paths reads as a Mila defect.
- **Routing locality** (`MixtureOfExperts.md` §8.1). Needs a corpus run through the router weights.

---

## Phase 5 — Router

### Design (2026-09-12)

- **`RouterOp` is selection only**: from `[N, 128]` router logits to top-8 expert indices (INT32)
  and combine weights. Softmax over all 128, top-8, renormalize, multiply by
  `per_expert_scale[index]`, which the op receives through `setParameters`. It has a CPU and a
  CUDA specialization. The discrete part of routing lives here, and it is what the gate below
  exercises hardest.
- **`Router` orchestrates** the unscaled norm, the projection and the selection. The projection is
  a `Linear` child named `proj`, so `router.proj.weight` loads under its checkpoint name, and it
  stays unquantized — the router is a precision holdout (`MixtureOfExperts.md` §7.5).
- **Open: where the unscaled norm and `router.scale` live — and `Testing.md` constrains it.** A CPU
  op may only be tested through its component (`Testing.md` §1: CPU op tests are mirrors, and an
  unreachable component is a component bug). So the CPU `RouterOp` is legitimate only if
  `Router<Cpu>` compiles, and `RmsNormOp` has no CPU row (`OperationTraits.Cpu.ixx:23`).
  (a) An `RmsNorm` child whose weight is `scale * 2816^-0.5`, CUDA-only: the reference gate runs in
  a `.Cuda.cpp` and never in CI, and the CPU `RouterOp` has no reason to exist.
  (b) The norm and scale move into `RouterOp`: `Router<Cpu>` is `Linear<Cpu>` plus `RouterOp`, so the
  gate runs on CPU in CI, at the cost of a second RMS kernel on CUDA.
  (c) (a) plus a CPU `RmsNormOp`, the open Llama-lineage contributor gap: the clean decomposition
  and a CI gate, at the cost of `RmsNorm<Cpu>` and its component test.
  **Decided (c), 2026-09-12.**

### Reference and gate

The reference is HF's `Gemma4TextRouter` run standalone on one layer's real router tensors, read by
HTTP range request — no checkpoint download, no card. It records the router logits as well as the
outputs, so the selection op is gated on its own before the norm and projection join it.

**Precision trap, written down before any run.** HF takes the softmax and the top-k on probabilities
held at model precision. Softmax is monotone, so ranking by logits and ranking by probabilities
agree — except where BF16 rounding makes two probabilities equal, and `torch.topk`'s tie order
decides. An op that ranks in FP32 is then correct and still disagrees on an index. So the reference
is captured at both precisions, and the gate is:

- **The selected set** — the eight (expert, weight) pairs of a row — is what is compared, never the
  order: the combine is a sum over the eight, so an op that returns the right experts in another
  order is correct, and an exact tie gives `torch.topk` no order worth matching.
- **Expert indices**: the selected set equals the FP32 reference exactly. Against the BF16 reference,
  a disagreement counts only where the reference probabilities tie.
- **Weights**: FP32 op within `1e-6` absolute of the FP32 reference; BF16 op within `2^-6` relative
  — two steps of bfloat16's 7-bit mantissa.

**First capture, layer 5, 256 seeded tokens (2026-09-12).** 13 rows select a different expert set at
BF16 than at FP32. 12 are exact BF16 probability ties, as predicted. The 13th (row 137) is not: FP32
ranks expert 38 above expert 100 by a logit gap of 7.6e-4, and the BF16 logits themselves come out
in the other order. So the tie rule above describes a ranking fed a capture's own logits, and is too
strict for an **end-to-end BF16 router**, whose projection rounds before anything is ranked.

**Every gate runs through `Router`, never `RouterOp` alone** — `Testing.md` §1 forbids op tests for
behaviour the component reaches, and the component reaches all of it. Written before any Mila run:

- **FP32, end to end** (`Router<Cpu>` and `Router<Cuda>`): fed `hidden_states` with the capture's
  real tensors, each row's selected set equals `fp32`'s, except where the two experts in question
  have FP32 reference logits within `1e-5` of each other; weights within `1e-5` absolute.
- **BF16, end to end** (`Router<Cuda>`): against the FP32 reference, a differing expert is allowed
  only where the two experts' FP32 logits lie within BF16 resolution of each other — defined as
  `|a - b| <= 2^-6 * max(|a|, |b|)`, two steps of the 7-bit mantissa; weights within `2^-6` relative.
  The BF16 router loads the capture's FP32 tensors truncated to bfloat16, so that truncation is
  inside the tolerance, not outside it.
- **Exact ties** (both devices): two identical `proj` rows give exactly equal logits, and the lower
  expert index must be selected — a hand-built case, because seeded rows almost never tie.

### CPU result (2026-09-12)

`Router<Cpu>` is `RmsNorm` (weight derived as `scale * 2816^-0.5`) + `Linear` + `CpuRouterOp`, and
exists because a CPU `RmsNormOp` now does (decision (c)). `Router.Cpu.cpp`:

| Gate | Result |
|---|---|
| FP32 end to end, layer 5, 256 tokens | **0** rows with a different expert set; worst combine-weight error **6.7e-7** (tolerance 1e-5) |
| Exact tie | lower index selected; weights match the definition within 1e-6 |
| Flat vocabulary | exactly `proj.weight`, `scale`, `per_expert_scale` |
| Footprint | predicted equals built |

**The gate can fail.** With the checkpoint's `scale` replaced by ones — which breaks only the derived
norm weight, the least-proven part of the chain — it reports 58 rows with an unexplained different
set and 1990 weights out of tolerance (worst 0.22). The seven gates that do not read `scale` stayed
green, as they should.

### CUDA result (2026-09-12)

`CudaRouterOp`: one thread per row, no shared memory and no thread sync, softmax accumulated in
double, insertion top-k where only a strictly greater logit displaces. `Router.Cuda.cpp`, both cards
pinned by UUID:

| Gate | RTX 5060 Ti (SM 12.0) | RTX 4070 (SM 8.9) |
|---|---|---|
| FP32 end to end, layer 5 | 0 differing sets; worst weight error **1.49e-7** | 0 differing sets; worst **1.04e-7** |
| BF16 end to end, vs FP32 reference | 11 rows differ, **0 unexplained**; worst **2.35e-3** | identical: 11, 0 unexplained, 2.35e-3 |
| Exact tie, FP32 and BF16 | pass | pass |
| BF16 footprint | predicted equals built | predicted equals built |

The FP32 error differs between the cards in the last bits and the BF16 result does not, which is the
expected shape: the FP32 GEMM reduces in a different order per architecture, while BF16 rounding
lands both cards on the same representable values. Every BF16 set difference sits inside the
bfloat16-resolution allowance fixed above, and the worst BF16 weight error (0.23% absolute) is well
inside its 1.56% relative tolerance.

**The CUDA gate can fail, and it fails on the kernel itself.** With the `per_expert_scale` multiply
removed from `Router.cu` (5060 Ti): the FP32 gate reports 1795 weights out of tolerance (worst
5.5e-3), the BF16 gate 487, and both exact-tie gates fail on expert 1, whose scale is 0.5. Expert
sets are unchanged, as they should be — the scale acts after selection — and the footprint gate,
which reads no kernel output, stayed green.

---

## Phase 6 — Expert bank

### Design (2026-09-12)

- **`MixtureOfExperts<Device, Precision, TGate>` is a leaf** owning the checkpoint's stacked tensors
  under their checkpoint names — `gate_up_proj` `[E, 2I, H]` and `down_proj` `[E, H, I]` — and the
  combine. Routing is an input: `forward( input, weights, indices )`, with `Router` a sibling in the
  block (`MixtureOfExperts.md` §5 amended).
- **`MoeOp` dispatches like `ElementwiseActivationOp`:** the CPU row registers `op_for<TFunctor>`,
  and the component passes `functor_of_t<TGate>`. `ActivationType::Gelu` resolves to `GeluTanh`,
  the same function and constants as HuggingFace's `gelu_pytorch_tanh`, so the activation adds no
  term to the tolerance.
- **The `GatedMLP` oracle the spec names cannot run on CPU:** `GatedMLP<Cpu>` does not compile, because
  `Swiglu` has no CPU op by decision. The CPU gates are instead a reference computed from the
  definition in the test, and HuggingFace's eager `Gemma4TextExperts` on synthetic stacked weights —
  which checks the chunk order (gate first), the activation and the combine against the source of
  truth without downloading a gigabyte-scale expert bank.

### Gate, written before any run

- **HuggingFace, CPU, FP32:** every output element within `1e-5` absolute of `Gemma4TextExperts` run
  eagerly on the same synthetic weights, inputs, indices and combine weights.
- **Definition, CPU, FP32:** a hand-built bank where each expert is distinguishable, within `1e-6`
  absolute of a double-precision computation in the test.

### CPU result (2026-09-12)

The reference is HuggingFace's eager loop reached through `forward.__wrapped__`, on 48 tokens, 16
experts of intermediate width 32, top-4, hidden 64. HuggingFace's dispatched forward reproduced the
eager loop exactly, so the reference does not depend on which of its paths ran.
`MixtureOfExperts.Cpu.cpp`:

| Gate | Result |
|---|---|
| HuggingFace eager `Gemma4TextExperts` | worst element error **4.8e-7** (tolerance 1e-5) |
| Definition, double precision | within 1e-6 |
| Out-of-range expert index | refused |
| Flat vocabulary | exactly `gate_up_proj`, `down_proj` |
| Footprint | predicted equals built |

**The gate can fail.** With the gate and up halves of `gate_up_proj` swapped in `CpuMoeOp` — the
chunk-order mistake the HuggingFace reference exists to catch — both numeric gates fail, the
HuggingFace one on all 3072 elements (worst 1.48). The five gates that do not depend on the chunk
order stayed green.

### CUDA design (2026-09-12)

**The simple kernel comes first, and the grouped GEMM becomes an optimization gated against it.**
Two passes over the stacked tensors in place, nothing gathered, nothing shared, no thread sync:

1. One thread per (token, slot, intermediate unit) writes `GeluTanh(gate) * up` into an FP32 scratch
   buffer of `tokens x top_k x I`, owned by the op and sized at build.
2. One thread per (token, output column) writes `sum over slots of weight * (down[e] . gated)`.

A single decoding token visiting 8 experts is exactly this kernel, so it is also Phase 7's decode
path; Phase 8 can run the model on it. The CUTLASS grouped path is not started here, and would first
need a decision on the rc.1 BACKLOG gate that removes CUTLASS from the build. The scratch buffer is
reported through the op's state-memory hooks, which `MixtureOfExperts` now adds to its own footprint.

A kernel cannot throw, and validating indices on the host would cost a synchronization per call. An
out-of-range expert index therefore writes NaN into that token's output instead of reading out of
bounds, so the defect surfaces rather than silently corrupting a neighbour.

### CUDA gate, written before any run

- **FP32** (`MixtureOfExperts<Cuda>`), against HuggingFace's eager experts: within `1e-5` absolute.
- **BF16**, same reference, with the FP32 capture truncated to bfloat16: within
  `0.05 * |reference| + 0.01`. Truncating weights of magnitude ~0.1 to a 7-bit mantissa moves each
  product by up to ~1%, and a sum of many of them moves the output by a comparable fraction; the
  swapped-chunk defect the CPU gate caught produces errors around 1.5, two orders above this line.
- **Definition:** FP32 within `1e-6`, BF16 within the BF16 rule above.
- **Invalid index:** the affected token's output is NaN; every other token is finite.

### CUDA result (2026-09-12)

`MixtureOfExperts.Cuda.cpp`, both cards pinned by UUID, same synthetic reference as the CPU gate:

| Gate | RTX 5060 Ti (SM 12.0) | RTX 4070 (SM 8.9) |
|---|---|---|
| FP32 vs HuggingFace | worst **2.4e-7** (tolerance 1e-5) | worst **2.4e-7** |
| BF16 vs HuggingFace | worst **1.9e-2** (tolerance 0.05·\|ref\| + 0.01) | worst **1.9e-2** |
| Definition, FP32 and BF16 | pass | pass |
| Out-of-range index | that token NaN, the other finite | same |
| BF16 footprint, gated scratch included | predicted equals built | same |

Unlike the router, the two cards agree to the last bit here: no library GEMM is involved, so both run
the same sequential per-thread accumulation. The CPU MoE suite also still passes with the component
now adding operation state to its footprint.

**The CUDA gate can fail, and it fails on the kernel itself.** With the gate and up offsets swapped
in `Moe.cu` (5060 Ti): both definition gates fail, the FP32 HuggingFace gate fails on all 3072
elements (worst 1.48) and the BF16 one on 2860 of 3072 (worst 1.47). The 212 BF16 elements that
stayed inside tolerance are outputs small enough to sit under the rule's 0.01 absolute floor — which
is why the FP32 and definition gates, not the BF16 one, are the ones that bound this defect class.
The poisoning and footprint gates, which do not depend on the chunk order, stayed green.
