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

**Phase 4 — footprint sparse-layer term** (step 4). *Exit:* `MemoryStats` carries resident and
active parameter bytes separately, predicted equal to built for `MixtureOfExperts`; every dense
family's report is unchanged. Reporting it through `Chat.Footprint.ixx` moves to Phase 8, the first
phase with an MoE configuration to report.

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
gather-matvec shape, so Phase 7 reduces to its exit. **Done 2026-09-12**, bit-identical on both cards —
see Phase 7 below.

**Phase 8 — converter + `fromPretrainedImpl`** (step 8). *Exit:* hidden-state parity against the HF
reference on a short prompt at BF16, then a `PerGroupFp4<64>` load on the 16 GiB 5060 Ti with
measured VRAM inside the `MixtureOfExperts.md` §8 row it was built for, and coherent generation.
The BF16 model is ~47 GiB and fits neither card, so both the HF reference and the Mila side run
layer-streamed — `Qwen3.8.md` §8 is the method. The FP4 load needs an FP4 expert bank, which this
phase builds, and group 64 rather than 128, which this phase decides (see Phase 8 below).

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

## Phase 4 — Footprint sparse-layer term

### Design (2026-09-12)

**The term is bytes on `MemoryStats`, not a parameter count on `Component`.** The footprint
pre-flight asks a constructed, unbuilt graph, and there `getRequiredMemory` answers while a
composite's `parameterCount()` throws. `MemoryStats` gains `device_inactive_parameter_bytes`: the
part of `device_parameter_bytes` a single token does not read. It is a subset, never a separate
allocation, so `totalDeviceBytes()` is unchanged, and `activeDeviceParameterBytes()` is the
difference. `operator+=` sums it, so it reaches a transformer through the existing aggregation.

`MixtureOfExperts` is the only component that sets it: of `E` equal expert rows a token reads
`top_k`, so `(E - top_k) / E` of the bank's bytes are inactive. Every dense component leaves it zero.

**Split from its Chat half.** No model configuration carries an MoE layer until Phase 8, so
`Chat.Footprint.ixx` has nothing to report yet; its wiring moves to Phase 8.

### Gate, written before any run

- **Component, CUDA BF16:** predicted equals built for the new field, and the built value is one
  expert's bytes for a top-2-of-3 bank — 48 of 144 parameter bytes — with 96 active.
- **Dense families unchanged:** every existing footprint drift gate passes untouched, and the full
  suite stays green.
- **The gate can fail:** reporting `top_k / E` as inactive instead of `(E - top_k) / E` fails the
  component gate.

### Result (2026-09-12)

RTX 5060 Ti, pinned by UUID. `MixtureOfExperts.Cuda.cpp`'s BF16 footprint test: predicted equals built
for the new field, and the built bank reports **144** parameter bytes, **48** inactive, **96** active.
All 23 MoE tests pass. **Full suite 1933 run, 1932 pass / 0 fail / 1 skipped** (the long-standing
Swiglu BF16 backward skip), with Gemma 4 12B's HuggingFace token parity passing — the dense families'
reports did not move.

**The gate can fail.** With `inactiveBytes` returning the `top_k` share, the CUDA test reports 96
inactive and 48 active and fails on both. Predicted still equals built — one helper feeds both paths —
so the literal byte counts, not the drift comparison, are what bound this defect; the CPU footprint
test, which compares only predicted to built, stayed green. **Reverted.**

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

---

## Phase 7 — Decode at `M == 1`

### Design (2026-09-12)

No new kernel. Both passes of the Phase 6 CUDA kernel compute a token from its own input row and
its own routing row, read the selected expert rows in place, and accumulate in an order that does
not depend on how many tokens are in the call. One decoding token is therefore the gather-matvec
`MixtureOfExperts.md` §6 describes, and Phase 7 is its gate.

### Gate, written before any run

Because nothing in a token's arithmetic depends on the token count, the gate is **bit-identical**,
not a tolerance:

- **Prefill-built bank, decoding.** 48 synthetic tokens (hidden 64, intermediate 32, 16 experts,
  top-4, fixed seed) run as one `[48, H]` call; then each token alone as `[1, 1, H]` through the same
  bank. Every output bit equals its row of the prefill call, at FP32 and at BF16. This is how the
  chassis decodes: the bank is built for the prompt and called one token at a time.
- **Bank built for one token.** The same comparison with a second bank built at `[1, 1, H]`, which
  sizes the gated scratch for exactly one token.
- **The gate can fail.** A stride defect in the combine pass's scratch offset leaves token 0 correct
  in both calls, so it must fail on the other 47 tokens.

### Result (2026-09-12)

`MixtureOfExperts.Cuda.cpp`, `*_DecodeMatchesPrefillRow_*`, each card pinned by UUID:

| Gate | RTX 5060 Ti (SM 12.0) | RTX 4070 (SM 8.9) |
|---|---|---|
| FP32, bank built for prefill | **0 of 3072** elements differ | 0 of 3072 |
| BF16, bank built for prefill | **0 of 3072** | 0 of 3072 |
| FP32, bank built for one token | **0 of 3072** | 0 of 3072 |
| BF16, bank built for one token | **0 of 3072** | 0 of 3072 |

The Phase 6 gates in the same run were unchanged on both cards.

**The gate can fail.** With the combine pass reading the gated scratch at `token * intermediate`
instead of `token * top_k * intermediate` (5060 Ti), all four decode gates fail: FP32 on **3008** of
3072 elements, exactly the 47 tokens after token 0, and BF16 on 3007 — one element of those tokens
landed on the same bfloat16 value either way, which was not examined further. The definition and HuggingFace gates
fail too; the poisoning and footprint gates stay green. **Reverted.**

---

## Phase 8 — Converter and `fromPretrainedImpl`

### Decisions (2026-09-12)

Reading the dense chassis against this checkpoint turned up four things the phasing did not price.
All four were decided before any code:

1. **A `PerGroupFp4<128>` expert bank is required, not an optimization.** `MixtureOfExperts` and
   `MoeOp` carry no weight quantization, and the bank is ~45 GiB at BF16 — the model runs on neither
   card without it. `MixtureOfExperts` gains `TWeightQuantization`, and the FP4 kernel is gated
   against the Phase 6 BF16 kernel on synthetic weights, the way Phase 6 was gated against
   HuggingFace.
2. **The Mila half of the BF16 parity gate is layer-streamed too.** 47 GiB fits neither card on
   either side. The method is `Qwen3.8.md` §8's: construct, load and run one block at a time
   against the layer-streamed HuggingFace reference. It needs the workspace sizing that is private to
   `GemmaTransformer` lifted into a type the transformer and the harness both construct — the cost
   Qwen paid, and the reason it did not duplicate the sizing.
3. **The converter reads safetensors tensor by tensor.** The dense converter materializes the whole
   model through `from_pretrained`; the host has 31.8 GiB against a 48 GiB checkpoint.
4. **Block wiring is a trailing `bool kMixtureOfExperts`** on `GemmaBlock` and `GemmaTransformer`,
   beside `kDelegatedFeedForward`. Set, the block's FFN is the delegated `mlp` plus `Router`,
   `MixtureOfExperts` and the three extra norms of Phase 1's topology. `PretrainedMetadata` gains the
   expert count, top-k and expert width. **`per_expert_scale` stays in `RouterOp`**: folding it into
   `down_proj` would make the exported weights differ from the checkpoint the Phase 5 and 6 oracles
   compare against.

### Wiring gate, written before any run

The reference is a tiny `Gemma4ForCausalLM` with the MoE block enabled — hidden 128, two layers
(sliding with window 8, then global), 8 experts of width 64 at top-2, a dense branch of 128, vocabulary
128 — run with eager attention and eager experts. Every weight is drawn at random, and every norm,
`router.scale`, `per_expert_scale` and `layer_scalar` is drawn away from 1, so a misplaced norm or scale
cannot hide behind an identity. The script saves that model as a HuggingFace checkpoint and converts it
with `Gemma/convert_weights.py`, so one capture holds the converter's names and the block's wiring. The
12-token prompt is longer than the window, so the sliding mask is exercised.

- **FP32:** the prefill's last-position logits and three decode steps' logits, each within `1e-4`
  absolute of HuggingFace's logits before the final softcap.
- **BF16** (the BF16 conversion of the same checkpoint): each step's logits within relative L2 `1e-2`,
  with the same argmax.
- **Names:** a routed transformer loaded from the converted file saves exactly the file's tensor names.
- **Footprint, BF16:** predicted equals built for every category including inactive bytes, and the
  inactive bytes are six of eight experts' share of both layers' banks.
- **Refusals:** a routed config in a dense transformer, and a dense config in a routed one, both throw.
- **Entry point:** `GemmaModel::getDeploymentFootprint` on the converted file reports inactive bytes, so
  the geometry reaches the dispatch from the file's own metadata.
- **Converter:** on a tiny dense checkpoint, the streaming converter's output is byte-identical to the
  `from_pretrained` converter it replaces.
- **The gate can fail:** routing on the dense branch's normalized input instead of the residual must
  fail the FP32 logits gate.

### Wiring result (2026-09-12)

`Gemma.MixtureOfExperts.Cuda.cpp`, both cards pinned by UUID, identical on each except the fifth digit
of one prefill number:

| Gate | RTX 5060 Ti (SM 12.0) | RTX 4070 (SM 8.9) |
|---|---|---|
| FP32 logits, prefill + 3 decode steps | worst **5.8e-6** (tolerance 1e-4), every argmax equal | worst 5.5e-6, same |
| BF16 logits, relative L2 per step | **2.4e-2, 1.4e-2, 2.4e-2, 5.6e-2 — FAILS the recorded 1e-2**; every argmax equal | 2.5e-2, 1.4e-2, 2.4e-2, 5.6e-2, same |
| Names, footprint and inactive bytes, entry point, both refusals | pass | pass |

**FP32 agrees with HuggingFace to about six digits** through prefill and three decode steps across a
sliding and a global layer — the converter's names, the norm placement, the router's input, the
expert bank, K=V global attention and the KV cache all hold.

**The BF16 gate missed the tolerance written before the run, and the tolerance is not changed here.**
HuggingFace's own bfloat16 run of the same checkpoint lands **3.0e-2, 1.4e-2, 2.3e-2, 4.1e-2** from its
float32 run, relative L2 per step, with every argmax equal — the same order as Mila's BF16 figures. So
the recorded 1e-2 underestimated bfloat16 on this model (logits of magnitude ~2, two layers whose
`layer_scalar` rounds from 0.91594 to 0.91406) rather than exposing a Mila defect. What the BF16 gate
should compare against, and at what tolerance, is open.

**The gate can fail.** With the router reading the dense branch's normalized input instead of the
residual (5060 Ti), the FP32 gate fails at every step — worst 0.22-0.57, relative L2 0.14-0.39 — and
the step 2 argmax changes. The names, footprint, entry-point and refusal gates stay green. **Reverted.**

**Full suite after the revert, RTX 5060 Ti:** 1940 run, 1938 pass, 1 skipped (the long-standing Swiglu
BF16 backward skip), 1 failing — the BF16 wiring gate above. Gemma 4 12B's HuggingFace token parity
passes, so the dense chassis did not move.

**`PerGroupFp4<128>` cannot build this model.** `CudaLinearOp::build` refuses an FP4 projection whose
input width is not a multiple of both 16 and the group size, and two of the 26B's widths are not
multiples of 128: the dense branch's `fc_down` reads 2112 (16.5 groups) and every expert's `down_proj`
reads 704 (5.5 groups). `QuantizationDispatch` maps `WeightQuantization::FP4` to `PerGroupFp4<128>`
for every family, so an FP4 load of the 26B would throw at build. Every width is a multiple of 64 —
2816, 2112, 704, 4096 and 8192 — and the FP4 kernels already support group 64. At group 64 a scale costs
0.5 bits per weight instead of 0.25, so the expert bank is 11.97 GiB rather than the 11.30 GiB
`MixtureOfExperts.md` §8 used — the same as its NVFP4 row, whose FP8-rest total of 14.20 GiB fits the
5060 Ti.

**Decided 2026-09-12, both on the evidence above:**

- **The BF16 wiring gate stays against HuggingFace's float32 logits, at relative L2 `1e-1` per step with
  the same argmax.** The original `1e-2` is replaced, not quietly widened: HuggingFace's own bfloat16 run
  is 1.4e-2 to 4.1e-2 from its float32 run on this model, so `1e-2` would fail a correct bfloat16
  implementation. Comparing against HuggingFace's bfloat16 run instead was rejected — that measures two
  independent roundings against each other and is no tighter.
- **The 26B's FP4 weights are `PerGroupFp4<64>`.** The published 12B stays at `PerGroupFp4<128>`.
  `WeightQuantization::FP4` stays one API value — the group is a property of the model's geometry, not
  a choice a caller makes — so the group travels at compile time: `dispatchWeightQuantization` takes a
  trailing `kFp4GroupSize` (default 128), `weightQuantizationName` spells it (`per_group_fp4_64`), and
  `requireStoredQuantizationMatches` compares against that spelling. `GemmaModel` reads the file's
  geometry first and makes one dispatch — dense at 128, routed at 64 — so the build instantiates dense ×
  {none, FP8, FP4<128>} and routed × {none, FP8, FP4<64>}, the same count as before, and a routed
  FP4<128> body that could only throw at build is never compiled.

**Converter on the real checkpoint (2026-09-12).** `--max-layers 6` over the downloaded
`gemma-4-26B-A4B-it`: 122 Mila tensors, 10.50 GiB, 16 s. Every checkpoint tensor was accounted for — 133
consumed, 356 skipped (355 `vision_tower`, 1 `embed_vision`), none unconsumed — and every declared shape
matched the geometry the config implies: pattern 6 from `layer_types`, global rotary width 128 from the
nested `rope_parameters`, 2 global KV heads, 128 experts of width 704 at top-8, text prefix
`model.language_model.`.

**Full conversion (2026-09-13).** All 30 layers at BF16 from the local checkpoint: 602 Mila tensors,
47.00 GiB, 1 min 45 s. 657 checkpoint tensors consumed, the same 356 skipped, none unconsumed. The tensor
count is exact against the map: embedding + 30 x 20 per layer + final norm = 602 Mila tensors; 25 sliding
layers x 22 + 5 global layers x 21 (no `v_proj`) + embedding + final norm = 657 sources. The checkpoint is
already `bfloat16`, so this pass renames, fuses and repacks; only `layer_scalar` changes precision (FP32).

**Layer-streamed HuggingFace reference (2026-09-13).** `Tools/Converters/Gemma/gemma_4_26b_moe/hf_gemma_layer_stream.py`,
the reference half of Decision 2, on `Qwen3.8.md` §8's method: one `Gemma4TextDecoderLayer` resident at a
time, both masks and both rotary tables from the model's own calls, the embedding scale multiplied in the
table's dtype. Its self-test (CPU, FP32) runs a six-layer routed model — window 4 against 10 tokens —
whole and streamed: **every layer and the final norm agree bitwise**. Two negative controls, both required
to diverge: no masks at all (final norm off by 5.0) and the sliding layers given the full causal mask (4.9).

**The prompt is the chat turn, not the bare sentence.** On the real checkpoint `<bos>The capital of France is`
put " DO" (5.69) and " CAP" (4.09) on top at BF16, and the same two (5.23, 4.67) at FP32 — so not
rounding. The config the driver builds matches `AutoConfig`'s on all 47 keys, and transformers applies no
text-weight conversion for `gemma4_unified`. The same sentence in the instruct chat turn with thinking off
— the 12B parity test's prompt — gives "The" at 18.63 and "Paris" at 14.00, the 12B's first generated
token. The driver was right and the bare sentence is out of distribution for an instruct model; a parity
gate on a logit-5 argmax would be measured on noise.

**References captured (2026-09-13, RTX 5060 Ti), 18-token chat prompt, all 30 layers:**
`Data/models/gemma/gemma4_26b_a4b_ref.bin` (BF16, 55 s) and `gemma4_26b_a4b_ref_fp32.bin` (FP32, 51 s). Both put
"The" (818) on top, then "Paris" (50429), "**", "the", " Paris": BF16 18.63 / 14.00, FP32 18.28 / 14.09.

### Layer-streamed parity gate, written before any run

`Tests/Dnn/Models/Gemma/GemmaModel.MixtureOfExperts.Parity.Cuda.cpp`, RTX 5060 Ti pinned by UUID: the 18-token
chat prompt through all 30 layers at BF16 on the converted weights, one block resident at a time, each block
built from `GemmaModel::configFromMetadata` with its workspaces from `makeGemmaBlockWorkspace` and
`makeGqaWorkspace` — the factories `GemmaTransformer` calls. The tolerance is the one the BF16 wiring gate was
decided at, against the same reference.

- **Against HuggingFace FP32:** every layer's last-token hidden state, the final norm and the last-position logits
  within relative L2 `1e-1`, and the same argmax.
- **Against HuggingFace BF16:** the same argmax. Per layer, Mila and HuggingFace BF16 are each reported against
  FP32, not asserted.
- **The gate can fail:** each layer loading its expert bank from the next layer must fail the layer gate from
  layer 0.
- **The shared workspace moved nothing:** the full suite passes, including the Qwen parity and footprint tests and
  every Gemma footprint literal.

The workspace change that precedes the run: `QwenGqaWorkspace` became the family-neutral `GqaWorkspace` in
`Compute.GqaWorkspace` beside `GqaState`, used by Qwen, Gemma and the harness; `GemmaTransformer`'s private
sizing became `makeGemmaBlockWorkspace` and `gemmaBlockWorkspaceWidths` beside `GemmaBlockWorkspace`; and
`GemmaModel::configFromMetadata` became public.

### Layer-streamed parity result (2026-09-13, RTX 5060 Ti)

**The gate passes: the 26B-A4B stack at BF16 predicts HuggingFace's next token through all 30 layers on the real
weights**, 37 s, one block resident at a time.

| Stage | Mila BF16 vs HF FP32 | HF BF16 vs HF FP32 |
|---|---|---|
| Layers 0-1 (sliding) | 4.7e-3, 3.5e-3 | 4.1e-3, 3.4e-3 |
| Layer 2 (sliding) | 6.7e-3 | 2.2e-2 |
| Layer 5 (first global) | 4.8e-3 | 7.6e-3 |
| Layers 10-17 | 9.7e-3 to 1.1e-2 | 1.1e-2 to 1.5e-2 |
| Layer 28 (worst) | **1.50e-2** | 2.49e-2 |
| Layer 29 (last, global) | 5.6e-3 | 9.8e-3 |
| Final norm | 1.22e-2 | 1.79e-2 |
| Logits | **4.2e-3**, argmax 818 | 8.5e-3, argmax 818 |

Every argmax is 818 ("The"): Mila, HuggingFace BF16 and HuggingFace FP32. **Mila's BF16 is closer to FP32 than
HuggingFace's own BF16 run at every layer from 2 on**, and equal within rounding at layers 0 and 1 — the same
direction Qwen measured, with the same likely cause (Mila keeps RoPE's cos/sin cache and rotation in FP32). The
error rises slowly down the stack and falls back at the last layer; it does not compound.

The worst measured error is 1.50e-2 against the recorded 1e-1 bound — the bound would pass an error about six times
larger. It is left as recorded; whether to ratchet it, as Qwen's was to ~45% above its measured peak, is open.

**Full suite with the shared workspaces, pinned RTX 5060 Ti:** 1948 run, 1947 pass, 0 fail, 1 skipped (the
long-standing Swiglu BF16 backward), 174 s. Qwen's layer-streamed parity and Gemma 12B's HuggingFace token parity
both pass, and every Gemma and Qwen footprint test holds, so moving the workspaces changed no geometry.

**The gate can fail.** With every layer loading the next layer's expert bank (layer 29 took layer 0's), the layer gate
fails at every one of the 30 layers from layer 0 (4.6e-1, rising past 1.0 by layer 10), the final norm at 1.07,
the logits at 4.7e-1, and the next token becomes 236779 instead of 818 against both references. **Reverted.**
After the revert every target builds and the harness reproduces the passing run to the printed digit (layer 28
1.503e-2, final norm 1.217e-2, logits 4.215e-3, argmax 818).

### FP4 load gate, written before any run

`Tests/Dnn/Models/Gemma/GemmaModel.MixtureOfExperts.Fp4.Cuda.cpp`, RTX 5060 Ti pinned by UUID:
`GemmaModel::fromPretrained` on the BF16 weights with `WeightQuantization::FP4` at context 8192 — the routed
dispatch at `PerGroupFp4<64>`, quantized on load. The row it was built for is `MixtureOfExperts.md` §8's
`PerGroupFp4<64>` row, 13.54 GiB of weights.

- **The expert term is exact:** the reported inactive parameter bytes equal 120/128 of the packed bank and its
  scales from the layout alone. *Recorded as 12,840,960,000 / 12,038,400,000, which was wrong — see the first run;
  the layout gives 12,846,366,720 / 12,043,468,800.*
- **The weights land in the §8 row:** reported parameter bytes within 3% of 13.54 GiB.
- **Gate A on the real routed model:** predicted parameter and state bytes equal the reported ones exactly.
- **Gate B:** the prediction does not exceed what `cudaMemGetInfo` says the load consumed, and the unmodelled
  residual stays under 25% of it — the 12B's bounds.
- **It fits:** the load consumes less than 15.0 GiB, the low end of §8's usable range for this card.
- **Generation:** greedy decode of the 18-token chat prompt reproduces HuggingFace BF16's greedy tokens
  token-for-token, the HuggingFace side generated by `hf_gemma_layer_stream.py --generate` re-running the whole
  prompt per token. As for the 12B, an FP4 run held to a BF16 reference: a divergence is investigated, never
  re-captured.
- **The gate can fail:** routing on the dense branch's normalized input instead of the residual (the wiring
  gate's negative) must change the generated tokens.

**First run (2026-09-13, RTX 5060 Ti, 14.82 GiB free) — FAILED, and one criterion above was unmeasurable.**
Weights reported 13.536 GiB, inside the §8 row; state 2.075 GiB; predicted = reported = **15.61 GiB against 14.82
GiB free**. The load completed and generated `818 5279 529 7001 563 5213 50429 84750` — "The capital of France is
**Paris**" — but only because WDDM placed the overflow in host memory.

- **"It fits: consumes less than 15.0 GiB" could not fail on this card.** `cudaMemGetInfo` reports the card as full
  once WDDM spills, so consumed read exactly the 14.82 GiB that was free. It passed while the model did not fit.
  Replaced, before the second run, by: the prediction is below the free memory, the card is not left saturated
  (under 256 MiB free), and Gate B's two bounds apply only when it is not. Gate B's "prediction exceeded actual
  consumption" failure in this run is the same saturation, not an overestimate.
- **Inactive bytes 12,043,468,800 against the recorded 12,038,400,000** — 180,224 bytes of bank per layer more than
  the recorded layout. **The recorded figure was wrong, not the code:** the `gate_up_proj` scales are
  128 x 1408 x 44 x 4 = 31,719,424 bytes, and the gate was written with 31,539,200. The second run asked one
  standalone bank for its own prediction — 428,212,224 bytes, which is the layout correctly multiplied — and the
  literal, the test constant and the §8 text are corrected to it.
- **Why the state is 2.08 GiB and not §8's ~0.4 GiB of KV** — two terms, both code-verified, sizes derived:
  every layer's RoPE cos/sin cache is sized by `getMaxSequenceLength()` (`Gemma.Block.ixx:989`), the trained
  262,144 rather than the 8,192 deployment context, so the two deduplicated caches hold 0.75 GiB; and the router
  and expert-bank outputs plus the bank's FP32 gated scratch are allocated per layer rather than pooled, ~0.5 GiB at
  the derived chunk of 512. Either alone leaves the prediction above the free memory; both would not. Neither is
  changed here — both are `Mila/Src` decisions.

**HuggingFace greedy reference (2026-09-13, RTX 4070, BF16, 8 min 41 s re-running the prompt per token):**
`818 5279 529 7001 563 5213 50429 84750` — "The capital of France is **Paris**.". The narrowest top-1 margin is 4.25
logits, at the first token; every later one is at least 14. Mila's FP4 output in the first run is identical, token
for token.

**Third run (2026-09-13, RTX 5060 Ti), corrected literal and the HuggingFace tokens in place — every criterion
passes except the fit:**

| Criterion | Result |
|---|---|
| One bank, from the component | 428,212,224 parameter / 401,448,960 inactive bytes = the layout |
| Model inactive bytes | 12,043,468,800 = 30 x the layout, exact |
| Weights in the §8 row | 13.536 GiB against 13.54 |
| Gate A, predicted = reported | exact, parameters and state |
| Greedy tokens against HuggingFace BF16 | **identical, all 8** — "The capital of France is **Paris**." |
| Fit | **fails** — 15.61 GiB predicted against 14.82 GiB free; the card saturates |
| Gate B | not measurable while saturated |

Prefill chunk 512 of an unconstrained 1024. **The model is correct at FP4 and does not fit this card at context
8192.** What stands between: the RoPE caches sized to the trained maximum (0.75 GiB) and the per-layer routed
buffers (0.49 GiB) — `Mila/Src` decisions, not taken here.

**The recorded negative did not fail the token criterion — the detector is weak, not the edit.** With the router
reading the dense branch's normalized input instead of the residual (RTX 5060 Ti), the FP4 load still generated
`818 5279 529 7001 563 5213 50429 84750`, identical to HuggingFace; only the fit criteria failed. The edit was
live: the BF16 layer-streamed gate, run from the same build, **failed at every layer from layer 2** — 1.2e-1
rising to 3.8e-1 at layer 27, final norm 3.6e-1, logits 1.2e-1 — while its argmax also stayed 818. On real
weights the router's own unscaled RMS norm absorbs most of `pre_feedforward_layernorm`'s per-channel scale, so the
selection moves too little to flip eight tokens whose narrowest margin is 4.25 logits. The tiny wiring model drew
those norm weights from 0.5-1.5, which is why the same edit failed there.

So the eight greedy tokens show coherent generation and nothing finer; the layer-streamed hidden-state gate is the
detector that can see a routing error on this model. The FP4 gate's "can fail" criterion is **not demonstrated** by
this negative, and is not quietly replaced by a stronger one after the fact. **Reverted.** After the revert every
target builds, the BF16 layer-streamed gate reproduces its passing run to the printed digit (layer 28 1.503e-2, logits
4.215e-3, argmax 818), and the FP4 load again fails only the fit.

**What a short prompt actually uses (2026-09-13, RTX 5060 Ti, headless).** The idle card reports 16,046 MiB free of
16,311. The test's `free before load` of 14.82 GiB is read inside the test process after its CUDA context exists, so
the ~0.85 GiB between the two is that process's own runtime — inferred, not isolated. Measured per process through
the WDDM `GPU Process Memory` counters while the FP4 load prefills the 18-token prompt and generates eight tokens:

| Context | Prefill chunk | Mila accounts | Peak dedicated | Peak shared | Peak dedicated + shared, one sample |
|---|---|---|---|---|---|
| 8192 | 512 | 15.61 GiB | 16,040 MiB (card full) | 1,282 MiB | not sampled together |
| 1024 | 1024 | 15.73 GiB | 16,035 MiB (card full) | 1,324 MiB | **17,358 MiB** |

**A shorter context uses more, not less.** The chunk heuristic budgets activations against the KV cache, so a small
context admits the largest chunk, and the unpooled routed buffers scale with the chunk: state rises from 2.075 to
2.197 GiB. Shared usage includes the loader's pinned staging as well as spill, so it overstates the spill by an
unmeasured amount. Tokens were correct at both contexts.

**Group plumbing result (2026-09-12, RTX 5060 Ti).** Loading the tiny routed model through
`GemmaModel::fromPretrained` at FP4 reports `per_group_fp4_64`; the contract test holds that the group is
part of the scheme in both directions; the delegated-FFN gates, including `PerGroupFp4<128>` bit-identity,
are unchanged; the BF16 wiring gate passes at its revised tolerance. **Full suite 1942 run, 1941 pass,
0 fail, 1 skipped** (the long-standing Swiglu BF16 backward skip); Gemma 4 12B token parity passes.

### FP4 expert bank — design (2026-09-12)

- **`MixtureOfExperts` gains `TWeightQuantization`** (default `NoWeightQuant`). Under `PerGroupFp4<g>`
  the bank holds `gate_up_proj` U8 `[E, 2I, H/2]` with `gate_up_proj_scale` FP32 `[E, 2I, H/g]`, and
  `down_proj` U8 `[E, H, I/2]` with `down_proj_scale` FP32 `[E, H, I/g]`. Every expert row is laid out
  exactly as a `Linear` FP4 weight row, so a stacked tensor is `E × rows` output channels of the
  existing per-group quantizer: BF16 reference weights quantize on load through the path `Linear`
  already uses, and a pre-quantized file loads its packed and scale tensors as stored.
- **`CudaMoeOp` keeps its two passes.** Under FP4 each pass decodes the selected expert's nibbles
  against their group scales inline; nothing is dequantized ahead of the matvec. BF16 compute only,
  as for an FP4 `Linear`.
- **The E2M1 decode is stated once:** it moves out of `CudaMatVecBias.Bf16.cu` into a shared kernel
  header that both kernels include.
- `GemmaBlock` passes its `TWeightQuantization` to the bank; the router stays unquantized.

### FP4 expert bank — gate, written before any run

- **Exact, CUDA BF16:** weights built from the E2M1 grid with each group's largest magnitude `6 · 2^k`,
  so the absmax quantizer is lossless and every dequantized value is exact in BF16. The FP4 bank's
  output must be **bit-identical** to the Phase 6 BF16 bank on the same weights, over prefill and
  one-token calls.
- **Round trip:** a bank saved pre-quantized and loaded back produces bit-identical output.
- **Footprint:** predicted equals built, with parameter bytes equal to the packed plus scale sizes
  written as literals, and inactive bytes the unselected experts' share of both.
- **Refusal:** an expert width that is not a multiple of the group throws at build.
- **The gate can fail:** decoding the high nibble as the even column must fail the exact gate.

### FP4 expert bank — result (2026-09-12, RTX 5060 Ti)

`MixtureOfExperts.Cuda.cpp`, a `PerGroupFp4<64>` bank of 16 experts (hidden 128, width 64, top-4, 48
tokens):

| Gate | Result |
|---|---|
| Exact, against the BF16 bank | **0 of 6144** elements differ, prefill and one token at a time |
| Round trip | saved names are the two packed tensors and their `_scale` companions; reload bit-identical |
| Footprint | predicted equals built; 221,184 parameter bytes, 165,888 inactive |
| Width refusal | an expert width of 96 at group 64 throws at build |
| FP8 bank | refused at construction |

The FP8 refusal was written expecting `std::invalid_argument`, which the operation throws, and failed
its first run because `Component::setExecutionContext` rethrows any construction failure as
`std::runtime_error` — its documented contract. The test now expects that type and checks the message
names the refusal.

**The gate can fail.** With the gated pass decoding the high nibble as the even column, the exact gate
fails on 6140 of 6144 elements in both modes; the round trip, which compares the bank against itself,
stays green, so the bit-identity gate is the one that bounds this defect. **Reverted.**

After the revert the exact gate reads **0 of 6144 on both the RTX 5060 Ti and the RTX 4070**, each pinned
by UUID, with all 23 MoE CUDA tests passing on each card. **Full suite, 5060 Ti: 1947 run, 1946 pass,
0 fail, 1 skipped** (the long-standing Swiglu BF16 backward skip); Gemma 4 12B token parity passes.

**Converter.** On a tiny dense checkpoint the streaming converter is byte-identical to the
`from_pretrained` one at FP32, and at BF16 when the checkpoint itself is BF16 — how every published Gemma
checkpoint is stored. Converting an FP32 checkpoint to BF16, the two differ only in the six
`layer_scalar` tensors: the old converter loaded the whole model at bfloat16, so each scalar was rounded
to bfloat16 before being written as FP32 (0.9458286 became 0.9453125); the streaming converter keeps the
checkpoint's value.
