# Mila Gemma 4 Chassis Specification

## Overview

This document specifies the **Gemma 4 12B Unified** dense transformer chassis — a
new `Components/Transformers/Gemma` family modeled on the validated Llama work,
not a modification of it. Gemma 4 is Mila's entry into 2026-era transformer
architecture and the deliberate stepping stone to Mixture-of-Experts: the 12B
dense model and the 26B-A4B MoE model share one chassis, differing only in the
FFN block (see `FfnAndMoE.md`). Proving the chassis on the dense model first
isolates the attention/RoPE/normalization subsystems from the router/grouped-GEMM
risk.

The design rests on one discriminating principle — **template axes are for types
and layouts, runtime config is for arithmetic** — applied to decide which of
Gemma's deltas become compile-time policies and which stay configuration values.
`Linear` remains the reference for the component/operation dispatch pattern (see
`OperationDispatch.md`). Investigation collapsed most of Gemma's deltas onto
**existing seams** rather than new dispatch axes: the global attention geometry
rides the existing GQA op from config (Section 5), the sliding window is a runtime
field (Section 6), and proportional RoPE is a runtime cache-build change (Section
4). The genuinely new compile-time piece is the **`TGate` gate functor** for GeGLU
(Section 7); the bounded-window KV cache (deferred) reuses the existing KV-cache
policy axis (Section 6).

---

## 1. The Model (confirmed `google/gemma-4-12B` config, 2026-06-19)

| Field | Value | Note |
|---|---|---|
| `num_hidden_layers` | 48 | interleaved 5 sliding : 1 full, **final layer global** |
| `hidden_size` | 3840 | residual stream |
| `num_attention_heads` | 16 | query heads |
| `num_key_value_heads` | 8 | sliding layers (GQA group 2) |
| `head_dim` | 256 | **decoupled** from hidden (3840 / 16 = 240 != 256) |
| `global_head_dim` | 512 | full/global layers only |
| `num_global_key_value_heads` | 1 | full layers: single shared KV head |
| `attention_k_eq_v` | true | full layers: V reuses K, no `v_proj` |
| `intermediate_size` | 15360 | GeGLU FFN |
| `hidden_activation` | `gelu_pytorch_tanh` | GeGLU gate (not SwiGLU) |
| `vocab_size` | 262144 | tied embeddings |
| `max_position_embeddings` | 262144 | 256K context |
| `rms_norm_eps` | 1e-6 | RMSNorm; QK-norm per head |
| `final_logit_softcapping` | 30.0 | no attention/query softcap (dropped vs Gemma 2) |
| RoPE (sliding) | theta 10000, `default` | full rotation of head_dim |
| RoPE (global) | theta 1e6, `proportional`, `partial_rotary_factor` 0.25 | rotate first 25% (128 of 512) |

"Unified" = the **encoder-free multimodal** architecture (12B/26B project raw
image patches and audio waveforms directly into the embedding space via linear
layers, dropping the dedicated encoders the E2B/E4B edge models use). The
multimodal projection is out of scope for the initial text port; the **dense text
chassis is the entry target**.

---

## 2. The Discriminating Principle

A variant earns a **compile-time template axis** only when it changes a **type, a
memory layout, or selects a genuinely different specialized kernel** that should
not be branched into per element. A variant that only changes an **arithmetic
value or a loop-invariant branch** inside an otherwise byte-identical kernel is
**runtime configuration**.

Applied to Gemma's eight deltas:

| Delta | Verdict | Mechanism |
|---|---|---|
| Decoupled `head_dim` | runtime field | `GqaConfig` / `RopeConfig` (Section 3) |
| RoPE default vs proportional+partial | runtime cache-build | `rotary_dim` zeroes upper freqs in the existing `Rope` cache (Section 4) |
| Local vs global geometry | runtime config + block wiring | existing GQA op at `NKV=1, HS=512` + `GemmaBlock` instantiation (Section 5) |
| Window size (masking) | runtime field | attention op parameter (Section 6) |
| Bounded KV ring buffer | **template** | `TKvPolicy` sibling (Section 6) |
| GeGLU gate | **template (functor)** | `TGate` (Section 7, exists) |
| Logit softcap | runtime field | block-level scalar |
| QK-norm | structural (always on) | block wiring |

The single new op-level axis is **`TRopePolicy`** (Section 4, pending the Step 3
kernel check). The local/global distinction is *not* an op-level axis: the
existing GQA op already expresses the global geometry from config, so the
distinction is carried by the two `GemmaBlock` instantiations (Section 5), a
block-wiring selector rather than an `OperationTraits` axis.

---

## 3. Decoupled `head_dim` (Step 0 — the root break)

Gemma decouples per-head width from the residual stream: `head_dim` 256 (sliding)
/ 512 (global) is not `hidden_size / num_heads` (3840 / 16 = 240). The question is
*where* the codebase bakes in the `head_dim == hidden / num_heads` coincidence,
and the answer (confirmed by reading the three configs) is **only at the
model/block level**, not in the attention/RoPE leaf configs:

- **`GqaConfig` and `RopeConfig` already decouple.** Both take the **Q-projection
  width** (`num_heads * head_dim`) as their first constructor argument
  (`GqaConfig` `model_dim`, `RopeConfig` `channels`, both documented as such) and
  derive `head_dim = width / num_heads`. Fed `num_heads * head_dim` they are
  correct for Gemma with **no change** — `GqaConfig(8192, 16, 1).getHeadDim()`
  already returns 512. They are left untouched (no retrofit of an explicit
  `head_dim` field — the Q-width contract is already documented, so there is no
  footgun to fix and no reason to touch validated Llama leaf code).
- **`LlamaConfig` bakes in the coincidence** and is the reason a new config is
  needed: it stores only `embedding_dim_` (the residual stream), exposes it as
  `getModelDim()`, and `withNumHeads` validates `embedding_dim % num_heads == 0`,
  hard-deriving `head_dim = embedding_dim / num_heads`. For Gemma that check
  passes silently (3840 % 16 == 0) while producing the wrong head_dim (240).

So Step 0 is **`GemmaConfig` carrying `head_dim` as an explicit, first-class
field**, separate from `embedding_dim` (residual), with **no edits to the
validated leaf configs**. The `GemmaBlock` then:

- feeds `num_heads * head_dim` (the Q-width, 4096 sliding / 8192 global) into the
  `GqaConfig` / `RopeConfig` constructors, and
- wires a **non-square** output projection `Linear(num_heads * head_dim,
  embedding_dim)` = `Linear(4096, 3840)` (sliding) / `Linear(8192, 3840)`
  (global), where Llama's square `Linear(model_dim, model_dim)` o_proj is the
  special case `num_heads * head_dim == embedding_dim`.

The QKV packing trailing dim is `(num_heads + 2 * num_kv_heads) * head_dim`, or
`(num_heads + num_kv_heads) * head_dim` for K=V global layers (Section 5).

Validated tests-first: a `GemmaConfig` with the sliding geometry
(`embedding_dim=3840, num_heads=16, num_kv_heads=8, head_dim=256`) and the global
geometry (`head_dim=512, num_kv_heads=1`) asserting the derived Q-width, QKV
packing dim, and o_proj shape — before any kernel or block exists.

---

## 4. RoPE proportional/partial-rotary — a cache-build change, not a policy

RoPE is a separate `Rope` component applied to Q/K before attention. Gemma needs
two per-layer variants:

- **Sliding layers** — full rotation, theta 10000. Already works today via
  `RopeConfig::withBase` (the cos/sin cache is keyed on `base`).
- **Global layers** — theta 1e6 (already works via `withBase`) plus
  **proportional partial-rotary**: `partial_rotary_factor 0.25`.

The original plan made this a compile-time `TRopePolicy` functor. Reading the HF
reference (`_compute_proportional_rope_parameters`) showed that is unnecessary.
"Proportional" builds the **full** `head_dim/2` inverse-frequency table as
`base^(-2i/head_dim)` (denominator is `head_dim`, not the rotary sub-dim), but only
the first `rotary_dim/2 = int(partial_rotary_factor * head_dim // 2)` pairs (64 of
256) carry real frequencies — **the remaining pairs are padded with zero**.

A **zero frequency means `cos = 1`, `sin = 0`, so the rotation is the identity** on
those dimensions (pass-through). Therefore feeding the **existing** rotation kernel
a cache whose upper frequencies are zeroed produces partial-rotary with **no kernel
change at all**. The work is entirely in the cache build:

1. `build_cache` zeroes the frequency pairs at index `>= rotary_dim/2`.
2. `rotary_dim` is added to the `RopeCacheRegistry` cache key (so the global
   layer's truncated table is a distinct entry).

`rotary_dim` already exists on `RopeConfig` (`withRotaryDim`, default 0 = full) —
the op/kernel simply ignore it today. This completes that field's intent. **No
`TRopePolicy`, no `OperationTraits` change, no new `PRoPE` component, no rotation-
kernel change.** Llama is byte-identical: `rotary_dim = 0` → all frequencies real →
identical cache (no intrinsic shift, unlike GeGLU).

---

## 5. The global geometry rides the existing GQA op

The global layer differs from sliding layers in three coupled ways:

- `head_dim` 512 (vs 256),
- a **single** shared KV head (vs 8),
- **K = V**: no `v_proj`; value states alias key states (`attention_k_eq_v`).

The original plan treated this as a distinct op selected by a `TAttentionKind`
policy through the GQA `OperationTraits` lookup. Reading `CudaGqaOp` (2026-06-19)
showed that is unnecessary — **all three are already expressible through the
existing op**, which derives every dimension from config and takes separate q/k/v
pointers on its live path:

- **head_dim 512** — `CudaGqaOp` reads `HS_ = config_.getHeadDim()`; the cuBLASLt
  plans and kernels take it as a parameter.
- **single KV head** — `num_kv_heads = 1` is **MQA**, an already-supported case
  (`GqaConfig::validate` allows `>= 1`; `GS_ = NH/NKV = 16`, `batch_count = B*NKV`
  fall out).
- **K = V** — `prefill`/`decode` take *separate* q/k/v pointers. Aliasing the V
  pointer to K (`prefill(q, k, /*v=*/k, ...)`) makes `kvcache_write_kv` write K
  into both caches. The `(num_heads + 2*num_kv_heads)*head_dim` packing assumption
  lives only in the stubbed standalone `forward()` and the component's
  `validateConcatenatedQKVShape`, **not** in the live path — so K=V packing is a
  *block* concern (how `GemmaBlock` sizes `qkv_proj` and splits it), never the op's.

So a global layer is just `CudaGqaOp` at `GqaConfig(model_dim=8192, num_heads=16,
num_kv_heads=1)` (head_dim 512 derived) with the block aliasing V to K — the same
class, kernels, and `OperationTraits` row Llama already uses. **No `TAttentionKind`
policy, no new `OperationType`, no new traits row, no new op class, no new template
parameter on `GroupedQueryAttention`, and zero change to the Llama path.**

The local/global distinction survives only as a **`GemmaBlock` wiring selector**
(the two instantiations of Section 8 differ in `qkv_proj` width, the V split, and
the `GqaConfig` they construct — see the table there). If `GemmaBlock` is templated
on a small block-level `GemmaLayerKind { Local, Global }`, that enum is used only
for `if constexpr` in the block's wiring; it never reaches the component, the op,
or `OperationTraits`.

**One runtime check, deferred to Section 9 Step 5:** confirm the hand-written GQA
kernels (`permute_q_compact`, prefill/decode softmax, unpermute) carry no static
`head_dim` assumption that breaks at 512 — Llama only ever runs 128/256. The
cuBLASLt GEMMs handle 512; the custom kernels need a read.

---

## 6. Window and bounded KV cache

These are two separable concerns that are easy to conflate:

- **Masking math is runtime.** A sliding-1024 layer and a global layer have
  *identical* Q/K/V/output shapes; the only difference is a lower bound
  `window_start = max(0, abs_t - window + 1)` added to the softmax loops alongside
  the existing causal upper bound (`Gqa.Prefill.{Bf16,Fp32}.cu`, and the
  `softmax_decode_forward` cache sweep). `window` (0 = global) passes through the
  dispatch beside `position_offset`. No type multiplication is earned — this is a
  **runtime field**.
- **Bounded KV is a `TKvPolicy` sibling.** The payoff of SWA at 256K context is
  that a sliding layer never needs more than `window` cached keys, so its cache is
  a **fixed-capacity ring buffer** (modular decode indexing) instead of a linear
  full-context cache. That changes allocation strategy and the decode kernel's
  indexing — a layout+kernel difference that belongs on the **existing KV-cache
  policy axis** (`Quantization/KvCache/Policy.ixx`), not a new template parameter
  and not conflated with the window number. Adding only the mask gives Gemma's
  numerics with none of its memory win; the ring buffer is the structural prize.

---

## 7. GeGLU FFN

Gemma's FFN is gated with `gelu_pytorch_tanh`, i.e. **GeGLU, not SwiGLU**. This is
the `TGate` generalization already specified in `FfnAndMoE.md` §9 and partially
landed: the `GatedMLP<Device, Precision, TGate>` composite and the `GeluTanh`
functor exist; the remaining work is generalizing `Swiglu<..., TGate>` over the
shared functor library plus the CPU `SwigluOp` so `TGate = GeluTanh` resolves.
The Gemma FFN is `GatedMLP<..., TGate=GeluTanh>` with `intermediate_size 15360`.

---

## 8. Heterogeneous layers — the `IDecoderLayer` boundary

A local layer and a global layer are **different `GemmaBlock` instantiations** —
they differ in `qkv_proj` width, the V split, the `GqaConfig` they construct, and
(Step 3) the RoPE policy:

| | Local block | Global block |
|---|---|---|
| `qkv_proj` | `Linear(3840 -> (16+2*8)*256 = 8192)` | `Linear(3840 -> (16+1)*512 = 8704)` |
| split | Q[4096] . K[2048] . V[2048] | Q[8192] . K[512] . **V := K (alias)** |
| `GqaConfig` | `GqaConfig(4096, 16, 8)` -> HS 256 | `GqaConfig(8192, 16, 1)` -> HS 512 |
| attention | causal + window 1024 (Step 2) | full causal |
| o_proj | `Linear(4096 -> 3840)` | `Linear(8192 -> 3840)` |
| RoPE | full rotation, theta 10000 | proportional partial-rotary (`rotary_dim` 128), theta 1e6 (Step 3) |

Because they are distinct types, Gemma (interleaving them 5:1 across 48 layers,
final layer global) can no longer hold a homogeneous `vector<GemmaBlock>` the way
`LlamaTransformer` / `GptTransformer` hold one block type.

The mechanism is a small **virtual `IDecoderLayer` interface** (`prefill` /
`decode` / `forward`) that both `GemmaBlock` instantiations implement; the
transformer iterates the layer list polymorphically. The cost is **one virtual
call per layer per token-step** — negligible against the per-layer GEMMs. The
`std::variant<LocalBlock, GlobalBlock>` alternative (monomorphic, `std::visit` at
each layer) was rejected: it bloats the variant to the larger block and adds visit
ceremony for no measurable gain.

This boundary is **genuinely new** — every existing Mila model (GPT-2, Llama) is
homogeneous and has no such interface. It is the one real architectural cost of
the compile-time approach, accepted deliberately.

---

## 9. Foundation Sequence

Built tests-first (the MNIST/Bard revival methodology), each step a small
reversible increment on the now-clean compact-NKV GQA op (alpha.6+69):

0. **Decoupled `head_dim`** in `GqaConfig` + `RopeConfig` (Section 3) — config +
   shape test, no kernel.
1. **Global-layer geometry in `GemmaConfig`** (Section 5) — config only: the
   `global_head_dim` / `num_global_kv_heads` / `key_equals_value` fields + the K=V
   packed-width helper. No op, no traits, no `TAttentionKind` policy (the global
   geometry rides the existing GQA op). Block wiring + V-aliasing land in Step 5.
2. **Sliding-window masking** (runtime `window`, Section 6). The **bounded-KV
   `TKvPolicy` sibling** (also Section 6) is **resequenced to after Step 5**: it is
   a memory optimization, not a correctness gate (the mask is correct against the
   full cache), and the prefill ring is the hardest kernel work here — build it
   against the HF-validated full-cache path as the oracle.
3. **Proportional partial-rotary RoPE** (Section 4) — `build_cache` zeroes the
   upper frequency pairs (`rotary_dim`) + `rotary_dim` joins the cache key. No
   kernel/op/component change; extends the existing `Rope`.
4. **GeGLU** via `TGate` (Section 7).
5. **`GemmaBlock`** assembling 0-4 + per-layer kind + `final_logit_softcapping`,
   behind `IDecoderLayer` (Section 8); then `GemmaTransformer` and the converter.

---

## 10. Relationship to MoE

The 26B-A4B MoE model reuses this exact chassis, swapping the per-layer
`GatedMLP` for a `MixtureOfExperts` (Router + grouped `MoeOp` over stacked expert
weights + shared expert), validated against the dense `GatedMLP` oracle. The MoE
machinery is specified in `FfnAndMoE.md` (decision B, §8/§9) and remains a
Future Direction; landing the dense Gemma chassis first is what de-risks it.
