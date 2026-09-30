# Mila Gemma 4 Chassis Specification

## Overview

This document specifies the **Gemma 4 12B Unified** dense transformer chassis — a
new `Components/Transformers/Gemma` family modeled on the validated Llama work,
not a modification of it. Gemma 4 is Mila's entry into 2026-era transformer
architecture and the deliberate stepping stone to Mixture-of-Experts: the 12B
dense model and the 26B-A4B MoE model share one chassis, differing only in the
FFN block (the 26B-A4B is Section 10). Proving the chassis on the dense model first
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

Gemma's FFN is gated with `gelu_pytorch_tanh`, i.e. **GeGLU, not SwiGLU**, with
`intermediate_size 15360`. Every block's feed-forward is one sublayer component
named `ffn`, chosen by `GemmaFeedForward` through `GemmaFeedForwardTraits`: on the
12B, `GemmaDenseFeedForward` — `pre_norm`, a `GatedMLP<..., ActivationType::Gelu>`
named `mlp`, `post_norm` — and on the 26B-A4B `GemmaRoutedFeedForward` (10.3). The
tensor names follow: `tf_layer_i.ffn.pre_norm`, `tf_layer_i.ffn.mlp.fc_gate_up`,
`tf_layer_i.ffn.mlp.fc_down`, `tf_layer_i.ffn.post_norm`. Until
`ModelFamilyParity.md` §8.2 G4 the dense FFN was three inline children of the block
(`tf_layer_i.fc_gate_up`), so weights converted before it do not load.

---

## 8. Heterogeneous layers — the `ITransformerBlock` boundary

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

The mechanism is a small **virtual `ITransformerBlock` interface** (`prefill` /
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
   behind `ITransformerBlock` (Section 8); then `GemmaTransformer` and the converter.

---

## 10. Gemma 4 26B-A4B

`google/gemma-4-26B-A4B-it` reuses this chassis. Each layer runs its dense FFN
as a `GatedMLP` beside a `Router` and a `MixtureOfExperts` over stacked expert
weights, the two branches summed ahead of the post-FFN norm. It landed during
v0.20's `rc.1`, gated against HuggingFace at BF16 and FP4.

**Committed to v0.21.0** under `ROADMAP.md`'s *Gemma 4 Complete*, and **held to
parity**: the 26B-A4B is finished when it matches Gemma 4 12B, Llama 3.1 and 3.2
and Qwen 3.8 in every respect `ModelFamilyParity.md` §3 records (Todd,
2026-09-29). It publishes in Q4_0, the format Google trained it for (10.4).

This section is the model's design of record. The MoE path it runs on is
`MixtureOfExperts.md`; how it was built and gated is the notebook
`Notebooks/Gemma4MoE.md`; the stages that remain are `ModelFamilyParity.md`
§8.2, G4 to G6. Moved here from `MixtureOfExperts.md` on 2026-09-29, where
10.1-10.9 were its Sections 1, 2, 4, 7.6, 8, 9, 10, 11 and 12.

Configuration and weight inventory (10.1, 10.2) were read on **2026-09-12** from
`config.json` and `model.safetensors.index.json`, not from a summary of them;
the topology (10.3) the same day from `modeling_gemma4.py` and the shard
headers. Parameter arithmetic in 10.5 is derived from those shapes and is **not
measured** except where it says so; what a load allocates is priced by the
deployment planner (`Deployment.md`), whose prediction the footprint gates hold
equal to what is built.

### 10.1 Configuration

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

### 10.2 Weight Inventory

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

experts.gate_up_proj             <- STACKED [128, 1408, 2816]  [gate | up]
experts.down_proj                <- STACKED [128, 2816, 704]

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

### 10.3 Block Topology and Wiring

Resolved 2026-09-12 from `Gemma4TextDecoderLayer`, `Gemma4TextRouter` and
`Gemma4TextExperts` in transformers 5.12.1, and from the layer 5 (global) and
layer 6 (sliding) shard headers. The evidence is `Gemma4MoE.md` Phase 1.

```
residual = h                                     post-attention residual stream
d = post_feedforward_layernorm_1( mlp( pre_feedforward_layernorm( residual ) ) )
e = post_feedforward_layernorm_2( experts( pre_feedforward_layernorm_2( residual ), router( residual ) ) )
h = ( residual + post_feedforward_layernorm( d + e ) ) * layer_scalar
```

- The dense and routed branches run **in parallel** from the same residual,
  each with its own pre- and post-norm. Their sum passes through the unsuffixed
  `post_feedforward_layernorm` — the norm the dense block already applies at
  that position — before the residual add.
- The router is `RMSNorm(no scale) -> x router.scale -> x hidden_size^-0.5 ->
  proj -> softmax over all 128 -> top-8 -> renormalize -> x
  per_expert_scale[index]`. `per_expert_scale` acts after selection: it changes
  combine magnitudes, never which experts run, and the combine weights do not
  sum to 1.
- `layer_scalar` multiplies the whole layer output, exactly as in the dense
  chassis (`Gemma.Block.ixx:241`). It is not a delta.
- Global layers carry **no `v_proj`**: `V = v_norm(k_proj(x))`,
  `K = RoPE(k_norm(k_proj(x)))`, as the dense chassis already does. K and V
  differ after their norms, so both are cached.
- Every norm multiplies by its raw weight; there is no `1 +` offset.

`GemmaBlock<TDeviceType, TPrecision, kGlobal, TWeightQuantization,
TKvCachePolicy, GemmaFeedForward::Routed>` behind `ITransformerBlock` already solves heterogeneous
layers, and **every** layer of the 26B is an MoE layer — there is no dense/MoE
interleave to model. The sliding/global split is the existing
`if constexpr (kGlobal)` selector and is untouched.

```
GemmaBlock (unchanged attention half)
  |
  +-- ffn: GemmaRoutedFeedForward
        |
        +-- pre_norm         -> mlp: GatedMLP<..., Gelu> (2112) -> dense_post_norm   -+
        |                                                                             |
        |                                             sum -> post_norm -> residual    |
        |                                                                             |
        +-- experts_pre_norm -> experts: MixtureOfExperts (704) -> experts_post_norm -+
        |                        +-- MoeOp  (stacked [128, ...])
        |                        ^ weights, indices
        +-- router (reads the residual itself; scale, top-k, per-expert scale)
```

HuggingFace's `pre_feedforward_layernorm`, `post_feedforward_layernorm_1`,
`pre_feedforward_layernorm_2`, `post_feedforward_layernorm_2` and
`post_feedforward_layernorm` are `ffn.pre_norm`, `ffn.dense_post_norm`,
`ffn.experts_pre_norm`, `ffn.experts_post_norm` and `ffn.post_norm`
(`ModelFamilyParity.md` §9 item 7). The sublayer is chosen by one template
argument, `GemmaFeedForward { Dense, Routed }`, read from the checkpoint in
`GemmaModel::dispatchChassis`, which refuses a routed network a policy its
expert bank does not implement before instantiating it (`expertBankImplements`,
`MixtureOfExperts.ixx`); `GemmaRoutedFeedForward` carries the same constraint.

### 10.4 Weight Format: Q4_0

**The 26B-A4B runs Q4_0.** A producer's quantization-aware weights run in the
format they were trained for (`ModelFamilyParity.md` §9 item 14), Google
publishes the 26B-A4B quantization-aware at Q4_0, and parity with the 12B, which
already runs Q4_0, requires the same.

Read 2026-09-29 from the tensor table of Google's
`google/gemma-4-26B-A4B-it-qat-q4_0-gguf` (`gemma-4-26B_q4_0-it.gguf`,
14,439,363,584 bytes), by HTTP range request. Each type was confirmed twice:
gguf-py's `GGMLQuantizationType` names it, and the tensor's span in the file is
exactly the size that format gives its shape.

| Tensors, every layer | Type |
|---|---|
| Expert bank, `ffn_gate_up_exps` `[128, 1408, 2816]` and `ffn_down_exps` `[128, 2816, 704]` | **Q4_0** |
| Dense branch (`ffn_gate`, `ffn_up`, `ffn_down`) and every attention projection | Q4_0 |
| Router projection, `router.scale`, `per_expert_scale`, every norm, `layer_scalar` | F32 |
| `token_embd`, tied (the file has no `output.weight`) | Q6_K |

- **Only the instruction-tuned model is quantization-aware.** Google publishes
  `-it-qat-q4_0-gguf`, `-it-qat-q4_0-unquantized` (51.6 GB, the full-precision
  weights out of the QAT pipeline) and a QAT drafter. The pretrained
  `google/gemma-4-26B-A4B` is BF16 only, and there is no compressed-tensors build
  of the 26B-A4B (there is for E2B, E4B, 12B and 31B).
- **Stored as Q4_0 is proven; trained under Q4_0 is inferred.** Google's card
  says the QAT checkpoints hold near-BF16 quality at Q4_0 and does not say tensor
  by tensor what was trained under it. The 12B's check settles it: Q4_0 built
  from the unquantized checkpoint by the reference rounding equals the GGUF in
  every code and scale bit (`Quantization.md`, Q4_0, *Rounding*), and G2 measures
  the cost over BF16.
- **The Q6_K table is the GGUF's conversion, not the trained format**; the
  unquantized checkpoint carries it at full precision. The 12B's Q4_0 build
  holds its tied table at per-row FP8 (`Quantization.md` Part III), and the 26B
  follows the 12B. The llama.cpp comparison states the head's difference or
  removes it.
- **The router stays unquantized** at compute precision, as it does today
  (`MixtureOfExperts.md` §4); the GGUF keeps it F32.
- **The policy is `PerGroupInt4<32>`** in Mila's two-plane layout
  (`Quantization.md`, Q4_0 decision 2). Group 32 divides every width — 2816,
  2112 (66 groups), 704 (22), 4096 and 8192 — so the group question FP4 raised
  (`Gemma4MoE.md` Phase 8) does not arise. Per layer the bank is 285,474,816
  bytes of `gate_up_proj` (253,755,392 of codes, 31,719,424 of FP16 scales) and
  142,737,408 of `down_proj`: **428,212,224 bytes, the same as the
  `PerGroupFp4<64>` bank**, whose FP32 scale per 64 weights costs the same half
  bit per weight as an FP16 scale per 32. The fit analysis of 10.5 carries over.

**Today the routed model cannot run Q4_0.** `GemmaModel::dispatchChassis` refuses
`q4_0` and FP8 for a routed checkpoint because `expertBankImplements` says the bank
has no path for them, and the bank itself refuses any policy but FP4
(`MixtureOfExperts.md` §7.6). Admitting Q4_0 is a change to that one predicate
and the bank's kernels. The `PerGroupFp4<64>`
build (`Gemma4MoE.md` Phase 8) is Mila's fallback format for this model, not what
it publishes. The work is `MixtureOfExperts.md` §7.7, as `Gemma4MoE.md` Phase 9.
Under the parity bar the grouped prefill and the tuned decode are part of the
model, not optional speed-ups: the 12B's prefill and generation rates are in the
llama.cpp comparison, so the 26B's are too.

### 10.5 Memory

Derived from the 10.1-10.2 shapes. **Not measured** except where stated.

Per sliding layer: attention 34.6M + dense branch 17.8M + experts 761.3M +
router 0.4M = **814.1M**, of which **761.3M (93.5%) is the expert bank**. A
global layer's attention is 49.0M (no `v_proj`, but `global_head_dim` 512), for
828.5M.

| | Parameters |
|---|---|
| Expert banks (30 layers) | 22.84B |
| Everything else (attention, dense branches, router, embedding) | 2.40B |
| **Total** | **25.23B** |
| Active per token | ~3.1B + head |

Weight residency, 5060 Ti (16 GiB, roughly 15.0-15.5 GiB usable after display
and driver):

| Experts | Rest | Total |
|---|---|---|
| NVFP4 (11.97 GiB) | BF16 (4.46 GiB) | **16.43 GiB — does not fit** |
| `PerGroupFp4<128>` (11.30 GiB) | BF16 (4.46 GiB) | **15.76 GiB — does not fit** |
| NVFP4 (11.97 GiB) | FP8 (2.23 GiB) | 14.20 GiB — fits, ~1 GiB headroom |
| `PerGroupFp4<128>` (11.30 GiB) | FP8 (2.23 GiB) | 13.53 GiB — fits |
| `PerGroupFp4<64>` (11.96 GiB) | `PerGroupFp4<64>` Linears 0.86 + FP8 table 0.69 + BF16 router 0.02 (1.57 GiB) | **13.54 GiB — what `WeightQuantization::FP4` builds** |
| **Q4_0 (11.96 GiB)** | Q4_0 Linears 0.86 + FP8 table 0.69 + BF16 router 0.02 (1.57 GiB) | **13.54 GiB — the build this model publishes; not built** |

The `PerGroupFp4<64>` row is the build that exists (`Gemma4MoE.md` Phase 8). Every width is a multiple of 64, not of 128, so
the group is 64; `WeightQuantization::FP4` quantizes every `Linear` rather than only the experts, the tied
table takes Gemma's per-row FP8, and the router projection stays unquantized. The expert term is exact from the
packed layout — per layer, `[128, 1408, 1408]` + `[128, 1408, 44]` FP32 scales for `gate_up_proj` and
`[128, 2816, 352]` + `[128, 2816, 11]` for `down_proj`, 428,212,224 bytes, x30 = 12,846,366,720 (11.96 GiB).

The Q4_0 row is derived: the same composition as the 12B's Q4_0 build, and Q4_0 costs every quantized tensor
exactly the bytes `PerGroupFp4<64>` does (10.4), so the two rows are equal. Google's GGUF of the same
model is 13.45 GiB, its Q6_K table 0.56 GiB against Mila's FP8 0.69.

**Measured, and short of the card** (`Gemma4MoE.md` Phase 8, after RoPE's tables were sized to the built
context): the `PerGroupFp4<64>` build at context 8192 predicts 15,982,200,832 bytes — 13.54 GiB of weights and
1.35 GiB of state — exactly what it reports, against 15,894,315,008 bytes free in the process: **83.8 MiB over**
at a fixed chunk of 512. The planner fits it by narrowing the chunk: at 8192 it plans 256 rows, 15,024 MiB,
with 120 MiB free after load, and it refuses 32768 (measured 2026-09-29, `Gemma4MoE.md` Rates Baseline).
The 12B plans 1024 at 8192. What stands between is the routed buffers, allocated per layer rather than
pooled — each layer's `MixtureOfExperts` output and FP32 gated scratch, about 0.49 GiB at chunk 512. Pooling
them is `ModelFamilyParity.md` §8.2 G5, and it holds for Q4_0 unchanged, since the weights are the same bytes.

**The "experts quantized, everything else BF16" recipe does not fit this card.**
The non-expert mass is under 10% of the parameters but 4.46 GiB at BF16, and the
embedding alone is 1.38 GiB of that. A 26B-A4B build for the 5060 Ti must
quantize the non-expert mass too. This contradicts the upstream
`nvfp4_experts_only` recipe, which targets cards with room to spare.

KV cache, by contrast, is the easy half — the bounded ring cache does the work:

- 25 sliding layers, capped at 1024 entries: **~0.20 GiB, independent of
  context length.**
- 5 global layers at 32K context: ~0.63 GiB. `k_eq_v` does not halve it — K
  and V leave their norms different and are both cached (10.3).

Only the 5 global layers scale with context, which is why the bounded-KV
`TKvCachePolicy` sibling in `SlidingWindowKvCache.md` landed before this model.

**NVFP4 KV does not help this model, and the reason is worth stating so it is
not tried.** An NVFP4 KV cache halves KV against FP8 at under 1% accuracy loss,
which is a real result — but Gemma 4's 5:1 sliding-window pattern has already
made KV the cheap half here. Halving 0.83 GiB saves ~0.4 GiB against a weight
problem measured in whole gigabytes. It is also not free: values are
dequantized from NVFP4 to FP8 *before* attention, so it adds a pass of exactly
the kind `MixtureOfExperts.md` §7.1a removes elsewhere. Where it would matter is
a family with unbounded KV — Llama 3.1 8B at 49152 context holds ~3.1 GiB of FP8
KV, and halving that is worth having. File it against Llama, not against this
model.

Whether a partial expert residency could help at all is `MixtureOfExperts.md`
§8; it is a decode-only technique and cannot solve the fit.

### 10.6 Chassis Deltas

Beyond the dense Gemma 4 chassis, which is otherwise reused unchanged:

1. `GemmaBlock` FFN slot delegated to a component (10.3, prerequisite).
2. `Router` + `RouterOp`, including the `scale` / `per_expert_scale` semantics
   from 10.3.
3. `MixtureOfExperts` + `MoeOp`, both paths of `MixtureOfExperts.md` §6.
4. Parallel dual-FFN wiring: three new norms (`pre_feedforward_layernorm_2`,
   `post_feedforward_layernorm_1`, `post_feedforward_layernorm_2`) and a sum
   ahead of the existing post-FFN norm.
5. `GemmaConfig`: expert count, top-k, `moe_intermediate_size`, and the
   dense-branch width as a field distinct from the expert width.
6. Converter: direct stacked upload, the three router tensors, and the three
   new norms written raw like every other Gemma norm.
7. Footprint: a sparse-layer term (`MixtureOfExperts.md` §8).
8. A Q4_0 expert bank, with its grouped INT8 prefill and gather decode
   (`MixtureOfExperts.md` §7.7).

Items 1 to 7 are built (`Gemma4MoE.md` Phases 1-8); item 1 is half done, the
dense 12B still inlining its FFN until G4. Item 8 is not built. Items 2, 3, 7
and 8 serve every future MoE family (`MixtureOfExperts.md` §9). Items 4, 5 and 6
are Gemma-specific. `layer_scalar`, K=V global attention and `v_norm` are
already in the dense chassis and are not deltas.

### 10.7 The Bar

The oracle discipline is `MixtureOfExperts.md` §10's. **The model is finished
when it holds every row of `ModelFamilyParity.md` §3 that the 12B holds.**
Beyond the oracle, that is:

- the published build (Q4_0) against HuggingFace in the suite, and its Q4_0
  tensors equal in code and scale bits to Google's GGUF;
- a publish gate shown to fail on a routing edit: on this model a router fed the
  wrong input still generated HuggingFace's eight greedy tokens at FP4, and only
  the layer-streamed BF16 hidden-state gate failed (`Gemma4MoE.md` Phase 8);
- quality measured across every context length the planner can choose. The
  12B's measure is cost over BF16 to 32K and the model against itself beyond.
  The 26B's BF16 fits neither card, so the first half has no whole-model
  reference; the method is open (`ModelFamilyParity.md` §9 item 18);
- the planner's prediction equal to what is built, and the fit at the context
  it chooses on the 16 GB card;
- decode replay equal to the called path, on a routed network;
- prefill and decode rates in the rates harness and the llama.cpp comparison;
- active parameter bytes shown to the user (`MixtureOfExperts.md` §8);
- the model reached through Chat, the inference server and the Python binding,
  tool calls included.

### 10.8 Sequencing

Steps 1-8 are the path that made the model run; what remains is sequenced in
`ModelFamilyParity.md` §8.2, G4 to G6, and not here.

1. **Resolve the block topology.** Done 2026-09-12 — `Gemma4MoE.md` Phase 1.
2. **Delegate `GemmaBlock`'s FFN to `GatedMLP`.** Phase 2a done 2026-09-12; the
   switch (Phase 2b) landed with `ModelFamilyParity.md` G4, where the feed-forward
   became the `ffn` sublayer on every Gemma block. Llama's half moves to Llama's pass.
3. **Bounded-KV `TKvCachePolicy` sibling.** Landed before this work began
   (`Gemma4MoE.md` Phase 3).
4. **Footprint sparse-layer term.** Done 2026-09-12 — `Gemma4MoE.md` Phase 4.
5. **`Router` + `RouterOp`**, validated against HF routing in isolation. Done
   2026-09-12 — `Gemma4MoE.md` Phase 5.
6. **`MixtureOfExperts` + `MoeOp` prefill path**, validated against the
   `GatedMLP` oracle on CPU. Done 2026-09-12, against HF's eager experts and a
   definition instead — `Gemma4MoE.md` Phase 6.
7. **`MoeOp` decode gather-matvec**, validated against step 6 at `M == 1`. Done
   2026-09-12 — `Gemma4MoE.md` Phase 7.
8. **Converter + `loadImpl`.** Done 2026-09-13, on `PerGroupFp4<64>` rather
   than `<128>` — `Gemma4MoE.md` Phase 8. **The model runs here**, correct at
   BF16 and FP4, and 83.8 MiB short of the 16 GB card at context 8192 at a fixed 512-row
   chunk; the planner later fits 8192 at 256 rows.

Then, by stage (`ModelFamilyParity.md` §8.2):

- **G4** — the feed-forward sublayer becomes a type (step 2's switch with it); done.
- **G5** — the routed buffers pooled, so the model fits; the Q4_0 expert bank
  (`Gemma4MoE.md` Phase 9, `MixtureOfExperts.md` §7.7).
- **G5b** — its kernels at parity: grouped INT8 prefill, gather decode and the
  router, measured in the rates harness and the llama.cpp comparison.
- **G6** — the rest of 10.7's bar, and the publish.

**NVFP4 is not a step of this model** (`MixtureOfExperts.md` §7). No remaining
step depends on NVFP4 or on CUTLASS.

### 10.9 Open Decisions

Open:

- **How quality is measured across the planner's range** when the BF16 model
  fits no card (10.7). `ModelFamilyParity.md` §9 item 18 carries it.

Decided:

- `per_expert_scale` **stays in `RouterOp`** (`Gemma4MoE.md` Phase 8, decision
  4): folded into `down_proj`, the exported weights would no longer equal the
  checkpoint the oracles compare against.
- The non-expert mass is **Q4_0 where Google's GGUF has Q4_0** (10.4), with the
  tied table at the 12B's per-row FP8 and the router unquantized. It was open
  between FP8 and FP4; the trained format settles it.
- `RouterOp` **has a CPU specialization**, reached through `Router<Cpu>`, which
  exists because a CPU `RmsNormOp` does (`Gemma4MoE.md` Phase 5, option (c)).
