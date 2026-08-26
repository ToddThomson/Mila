# Mila Qwen3.8-27B Chassis Specification

## Overview

This document specifies an **experimental** chassis for `Qwen/Qwen3.8-27B` — a 27B dense
hybrid-attention model — targeting a single **12 GiB RTX 4070**. The work sits outside the
v0.20 scope entirely: it is a research track whose purpose is to test whether Mila's
compile-time component design absorbs an architecture it was not built for, and to produce
the evidence for the website's typed-components claim.

Qwen3.8 was released 2026-08-14, after the Gemma chassis existed. That ordering is the point.
Gemma proves the design accommodates the variation it was designed around, which is a
circular claim; Qwen3.8 tests it against a sequence mixer Mila has never run.

The design rests on the same discriminating principle as `Gemma.md` — **template axes are for
types and layouts, runtime config is for arithmetic** — plus one new constraint that comes
from the research goal rather than from the code: **the declaration site must stay legible.**
A per-projection precision scheme is expressible either as a readable named plan or as an
eight-parameter template, and both compile. Only one of them demonstrates anything, so the
plan struct (Section 6) is a requirement, not a preference.

Investigation collapsed most of Qwen3.8's deltas onto **existing seams**. The 48:16
heterogeneous layer list rides `ITransformerBlock` unmodified (Section 7). The full-attention
layers ride the existing GQA op from config, wrapped by one small new mechanism: the
`attn_output_gate` (Section 2). The genuinely new pieces are four: the **Gated DeltaNet
sequence mixer** with no precedent in the tree, **sub-4-bit weight storage**
(`PerGroupCodebook2`, `PerGroupCodebook3`), the **per-projection precision plan** that lets a single block carry more
than one weight format, and the **full-attention output gate** — small, but absent from
Gemma's GQA path.

---

## 1. The Model (published `Qwen/Qwen3.8-27B` config, 2026-08-14)

> **Revision 2026-08-16:** table re-verified against the raw `config.json` (it was first
> transcribed from a summarization, which dropped `attn_output_gate: true` — the source of
> the Section 2 undercount). Licence and the 1M YaRN extension verified on the model card.
>
> **Revision 2026-08-19:** re-verified against the downloaded checkpoint — `config.json` plus
> every shard header — and against the reference implementation, now readable locally
> (`transformers 5.12.1`, `models/qwen3_5/`). Three rows changed: QK-norm exists and was
> missing entirely, the attention output gate is **sigmoid** rather than swish, and the two
> DeltaNet head dimensions are separate fields. See *Phase 3 status* for how each was found.

| Field | Value | Note |
|---|---|---|
| `num_hidden_layers` | 64 | interleaved **3 linear : 1 full**, `full_attention_interval: 4` |
| `hidden_size` | 5120 | residual stream |
| `intermediate_size` | 17408 | SwiGLU FFN, uniform across both layer kinds |
| `vocab_size` | 248320 | large; embedding and head are 4.7% of parameters each |
| `tie_word_embeddings` | false | **untied** — two full-size tables, unlike Gemma |
| `num_attention_heads` | 24 | full-attention layers only |
| `num_key_value_heads` | 4 | full-attention layers, GQA group 6 |
| `attn_output_gate` | true | **q projection is double-width**: [query \| gate], **sigmoid** gate on the attention output (Section 2) |
| `q_norm` / `k_norm` | present | RMSNorm over `head_dim`, per head, **before RoPE**. Not a config field — visible only in the checkpoint and the reference |
| `head_dim` | 256 | **decoupled** (5120 / 24 = 213 != 256) |
| `partial_rotary_factor` | 0.25 | rotary width 64 of 256 |
| `rope_theta` | 1e7 | |
| `rope_parameters` | mrope, sections [11, 11, 10] | interleaved; multimodal positional layout |
| `linear_num_key_heads` | 16 | Gated DeltaNet; 3 value heads per key head |
| `linear_num_value_heads` | 48 | Gated DeltaNet |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 | two separate fields, equal here — do not assume one |
| `linear_conv_kernel_dim` | 4 | short causal depthwise convolution over q/k/v |
| `output_gate_type` | swish | **read nowhere in the implementation.** Its value does describe the DeltaNet value gate (silu = swish); it does **not** describe the attention output gate, which is hardcoded sigmoid. A chassis that keys the attention gate off this field picks the wrong function |
| `mamba_ssm_dtype` | float32 | published dtype of the DeltaNet recurrent state (Section 3) |
| `max_position_embeddings` | 262144 | extensible to 1M via YaRN overrides (model card) |
| `dtype` | bfloat16 | transformers 5.x field name (formerly `torch_dtype`) |
| Vision tower | 27 layers, width 1152, out 5120 | 16x16 patches, 2x2 spatial/temporal merge |
| Licence | Apache 2.0 | republishable |

**Out of scope for the first chassis:** the vision tower and the MTP draft head. Text-only is
also the configuration the upstream vLLM recipe describes as verified.

---

## 2. Parameter Budget

Derived from the published config, **not** read from checkpoint tensor shapes. Reconciled
against the published BF16 GGUFs: 53.8 GB text-only, and the Unsloth artifact at 54.7 GB
carrying the single-layer MTP head (~0.45 B) as well. Both imply ~26.9 B text parameters,
agreeing with the total below to within 0.1%. **Confirm against real tensor shapes before
any of the Section 5 allocation is treated as final** — `intermediate_size` is a single
scalar in the config, so uniformity across both layer kinds is what the config states; the
checkpoint check at converter time remains cheap insurance.

> **Revision 2026-08-16:** `attn_output_gate: true` doubles the full-attention q projection
> to emit [query | gate]. The first version of this table omitted the gate half (full
> attention 73.40 M/layer, text total 26.39 B) and sat 1.9% under the GGUF anchor — that
> residual **was** the gate. The gate is a mechanism as well as a count: swish(gate)
> scales the attention output elementwise before `o_proj`, and Gemma's GQA path has no
> equivalent, so the full-attention block carries this one new piece alongside the stock
> GQA op.

| Group | Per layer | Layers | Total | Share |
|---|---|---|---|---|
| FFN gate + up | 178.26 M | 64 | 11.408 B | 42.4% |
| FFN down | 89.13 M | 64 | 5.704 B | 21.2% |
| DeltaNet v, output-gate, o | 94.37 M | 48 | 4.531 B | 16.8% |
| Full attention q (+gate), k, v, o | 104.86 M | 16 | 1.678 B | 6.2% |
| `embed_tokens` | — | — | 1.271 B | 4.7% |
| `lm_head` | — | — | 1.271 B | 4.7% |
| DeltaNet q, k | 20.97 M | 48 | 1.007 B | 3.7% |
| DeltaNet beta, decay, conv | 0.53 M | 48 | 0.025 B | 0.1% |
| **Text total** | | | **26.90 B** | |

The FFN is just under 64% of the model. Any scheme that fits this card is decided almost
entirely by what happens to those three matrices.

---

## 3. Why the Hybrid Layout Makes This Card Viable

Only 16 of 64 layers hold a KV cache:

```
per token = 16 layers * 2 (K,V) * 4 kv_heads * 256 head_dim = 32,768 elements
          = 64 KiB at BF16, 32 KiB at FP8
```

The other 48 layers hold a **recurrent state whose size is independent of sequence length**:

```
per layer = 48 value_heads * 128 (d_k) * 128 (d_v) = 786,432 elements
total     = 48 layers * 786,432 * 4 bytes (FP32) = 151 MB = 0.141 GiB, flat
conv ring = 48 layers * 10,240 channels * 4 taps * 4 bytes = 7.5 MiB, also flat
```

A dense 64-layer model of this width would cost 4x the KV. At the 16K baseline the
full-attention cache is 1.0 GiB at BF16 (0.5 GiB at FP8, a policy not yet in the tree —
see Section 8) and the recurrent state is 0.141 GiB regardless. This is the property that
makes a 27B model plausible on 12 GiB at all, and it is architectural, not an optimization
Mila supplies.

The state must be **FP32**. It is a recurrence, and BF16 accumulation over a 262K-token
sequence will drift.

---

## 4. There Is No Uniform 4-Bit Path

> **Revision 2026-08-16:** the first version of this table computed decimal bytes and
> labelled them GiB (13.34 and 12.95 "GiB" were GB values), and carried the pre-revision
> 25.12 B resident count. Corrected below; with the context relaxed to 16K the budget is
> Section 5's baseline and the required average moves from 2.74 to ~3.0 bits.

Moving `embed_tokens` to host memory leaves 25.62 B device-resident parameters. Against the
16K-baseline weight budget of 9.06 GiB (Section 5):

| Scheme | Bits/weight | Resident weights |
|---|---|---|
| `PerGroupFp4<128>`, FP32 scales (today's policy) | 4.25 | 12.68 GiB |
| `PerGroupFp4<128>`, FP16 scales | 4.125 | 12.31 GiB |
| Required to fit | **~3.0** | 9.06 GiB |

Uniform FP4 misses by more than a third. Even a uniform 3.25-bit FFN misses: 17.11 B FFN
parameters at 3.25 bits are 6.48 GiB, and the non-FFN residents cost 3.17 GiB in the Section
5 plan — 9.65 GiB total. The gate/up pair has to sit near 2.5 bits. The ecosystem's own
numbers agree: the smallest published GGUF that fits this card is UD-IQ2_XXS at 9.01 GB
(~2.6 bits/weight), and it is labelled as carrying a quality cost; Q4_K_M is 17.1 GB.

**Therefore the FFN — 64% of the model — must sit between 2 and 3 bits.** Only the
full-attention projections and `lm_head` — 11.5% of resident parameters — can afford 4 bits.
This is the finding that shapes everything downstream, and it is why a per-projection
precision plan is required rather than merely interesting.

One thing makes this far less costly than it sounds: **the RTX 4070 is SM 8.9 and has no FP4
hardware.** Every sub-8-bit format already dequantizes to BF16 in the GEMM tile load, so INT2
and INT3 cost nothing extra against the FP4 path Mila already runs. On this card the bit width
is purely a storage decision.

One caution on the IQ2_XXS comparison: that format earns its ~2.6 bits with E8-lattice
codebooks and importance-matrix calibration. Plain per-group asymmetric INT2 at equal bits
is worse. The mitigation is free on this card: since dequantization already goes through a
LUT in the tile load, a non-uniform codebook prices identically to uniform steps — Section 8
makes codebook dequantization the default for the sub-4-bit formats.

---

## 5. Precision Plan

Ranked by sensitivity per byte. One ordering decision is non-obvious and load-bearing: **the
DeltaNet projections outrank the FFN.** Error in an attention layer is recomputed from cache
each step; error in a recurrence enters the state matrix and persists for the remainder of the
sequence.

> **Revision 2026-08-16:** the full-attention row now carries the `attn_output_gate` half of
> the q projection (1.678 B, was 1.174 B); the norms row was unexplained padding and is now
> the derived count (norms + conv + per-head vectors); the average is stated per **resident**
> weight (the earlier 2.78 divided by all text parameters, including the host-resident
> embedding). Resident total moves from 8.55 to 8.77 GiB.

> **Revision 2026-08-16 (post-gate):** the sub-4-bit rows are now the Phase-0-validated
> **codebook** formats. The codebook absorbs asymmetry per tensor, so there is no zero
> point and the 2-bit rows drop from 2.5625 to 2.5 bits; the uniform-step variants lost
> the gate by 15x and are retired. Resident total moves from 8.77 to 8.65 GiB.

| Group | Params | Policy | Bits/w | GiB |
|---|---|---|---|---|
| `embed_tokens` | 1.271 B | BF16, **host-resident** | — | 0.000 |
| FFN gate + up | 11.408 B | `PerGroupCodebook2<32>` | 2.5 | 3.320 |
| FFN down | 5.704 B | `PerGroupCodebook3<64>` | 3.25 | 2.158 |
| DeltaNet v, gate, o | 4.531 B | `PerGroupCodebook2<32>` | 2.5 | 1.319 |
| DeltaNet q, k | 1.007 B | `PerGroupCodebook3<64>` | 3.25 | 0.381 |
| DeltaNet beta, decay | 0.024 B | **BF16 — never quantized** | 16 | 0.045 |
| Full attention q (+gate), k, v, o | 1.678 B | `PerGroupFp4<128>` | 4.125 | 0.806 |
| `lm_head` | 1.271 B | `PerGroupFp4<128>` | 4.125 | 0.610 |
| Norms, conv, misc | 0.003 B | FP32 | 32 | 0.012 |
| **Resident total** | **25.62 B** | | **2.90 avg** | **8.65** |

The bit widths include their metadata, and the packing is part of the contract:
`PerGroupCodebook2<32>` is 2-bit codes plus an FP16 absmax scale per group of 32
(2 + 16/32 = 2.5) indexing a per-tensor 4-entry codebook; `PerGroupCodebook3<64>` is 3-bit
codes plus an FP16 scale per group of 64 (3 + 16/64 = 3.25) with an 8-entry codebook; the
codebooks amortize to nothing. `PerGroupFp4<128>` with FP16 scales is 4 + 16/128 = 4.125.
There is no zero point anywhere in the sub-4-bit path — the codebook carries the
asymmetry, which is both cheaper and measurably better than a per-group zero
(Phase 0 results below).

Rationale for the assignments that are not simply "biggest group gets fewest bits":

- **`embed_tokens` to host.** It is a gather, never a multiply. Cost is 10 KiB per token over
  PCIe at decode. This buys 4.7% of the model for no VRAM and no measurable latency, and it
  is the single cheapest move available.
- **`beta` and `decay` stay BF16.** These control the forget gate. Quantization error in a
  decay rate compounds *exponentially* over the sequence, unlike error in a value projection
  which compounds linearly. They are 0.1% of parameters; protecting them costs 0.045 GiB.
- **FFN `down` above `gate`/`up`.** `down_proj` reads post-SwiGLU activations, which carry the
  heavy outliers. A half-step of extra precision here is the conventional protection and it is
  cheap relative to the gate/up pair.
- **`lm_head` stays at 4 bits.** It drives sampling directly. It is also read in full every
  decode step — 0.61 GiB per token — so it is a bandwidth cost as well as a memory one.
- **Scales move to FP16.** Today's `kScaleDtype` is FP32; at group 128 that is 0.25 bits of
  overhead per weight. FP16 halves it. This is a free win for Gemma and Llama independently
  of this work.

### Memory budget (GiB, as the device reports it)

> **Revision 2026-08-16:** 16K is now the baseline (was 32K). That relaxation retires a
> hidden dependency: BF16 KV fits at 16K, so the FP8 KV-cache policy — which does not yet
> exist in the tree — becomes an upgrade or the price of the 32K stretch, not a
> prerequisite. The 64K column is retired. Weights, KV, and state are derived; the
> activations and CUDA-context rows are estimates informed by the Gemma chassis, and at
> these margins they are where the remaining risk lives.

| | 16K, BF16 KV (baseline) | 16K, FP8 KV | 32K, FP8 KV (stretch) |
|---|---|---|---|
| Weights | 8.65 | 8.65 | 8.65 |
| Full-attention KV (16 layers) | 1.00 | 0.50 | 1.00 |
| DeltaNet recurrent state (FP32) | 0.14 | 0.14 | 0.14 |
| Activations and scratch (estimated) | 0.45 | 0.45 | 0.45 |
| CUDA context, cuBLASLt workspace (estimated) | 0.35 | 0.35 | 0.35 |
| **Total** | **10.59** | **10.09** | **10.59** |

Against 12 GiB, with roughly 11.0-11.3 GiB usable on a display-attached card under WDDM: the
baseline fits with 0.4-0.7 GiB of margin and no KV quantization; FP8 KV widens that to
0.9-1.2 GiB, or buys the 32K stretch at the same total. 64K is out of reach without moving
the desktop off the card. The FP8-KV margin at 16K is worth bits instead of context: each
+0.25 bits across the 17.11 B FFN parameters costs 0.50 GiB, so the margin covers +0.25 to
+0.5 bits on the most-starved rows. Whether to spend it on context or on quality is an open
question (Section 9).

### Decode bandwidth ceiling

Decode is bandwidth-bound and comfortably fast. Each token reads 8.65 GiB of weights
(9.29 GB), reads and writes the 151 MB recurrent state (0.30 GB), and — at a filled 16K BF16
cache — reads 1.07 GB of KV: about 10.7 GB per token against the RTX 4070's 504 GB/s, a
ceiling near 47 tok/s. Even at half that ceiling the experiment is quality-limited, not
speed-limited, which is why margin goes to bits rather than to bandwidth tricks.

Two implementation constraints follow directly and are easy to get wrong:

1. **Prefill must compute logits for the last position only.** The head emits the compute
   precision, so at 248,320 vocabulary a single BF16 logit row is 0.474 MiB; a 512-token chunk
   that materialized all of them would allocate 0.237 GiB.
2. **The FFN must never expand a whole weight matrix to BF16 at once.** Dequantizing one
   5120x17408 matrix is 170 MiB, against the 461 MiB this section allots to all activations
   and scratch. Either the fused tile-load path or a **striped** two-phase pass satisfies
   this; striping was measured cheaper on both axes and is the standing answer (Section 8).
   This constraint bounds the staging buffer, and does not name a kernel.

---

## 6. The Precision Plan Is a Named Struct, Not a Parameter List

Per-projection precision cannot ride the existing single `TWeightQuantization` parameter,
which applies uniformly to every `Linear` in a block. The extension must not widen the block's
template parameter list:

```cpp
// Required: the declaration states the strategy.
using DeltaNetLayer = QwenDeltaNetBlock<DeviceType::Cuda, TensorDataType::BF16, QwenPrecisionPlan>;

// Rejected: compiles, guarantees the same things, demonstrates nothing.
using DeltaNetLayer = QwenDeltaNetBlock<DeviceType::Cuda, TensorDataType::BF16,
    PerGroupCodebook2<32>, PerGroupCodebook3<64>, PerGroupCodebook2<32>, NoWeightQuant, PerGroupFp4<128>, ...>;
```

A precision plan is a struct of named policy typedefs, one per tensor role, satisfying a
concept that requires every role the block will instantiate. A role the plan does not name is
a compile error at the block, not a silent fallback. This preserves the property that makes
the whole approach worth writing about: **the memory strategy of the model is readable at the
declaration site.**

`Gemma.ixx` already does a one-role version of this with `TableQuantizationPolicy`, where the
embedding and head diverge from the body. The plan struct generalizes that from one exception
to a full table.

---

## 7. `ITransformerBlock` Fit (verified 2026-08-16)

The heterogeneous layer list needs no new mechanism. `ITransformerBlock` was introduced for
Gemma's two block types and takes DeltaNet with **five of six methods unmodified**.

| Method | Fit | Note |
|---|---|---|
| `prefill(input, position_offset)` | Fits | Chunked-parallel delta rule. `position_offset` is unused — linear layers carry no RoPE. Cross-chunk state is member data, exactly as the KV cache is. |
| `decode(input, position)` | Fits | The O(1) recurrent update. `position` unused. |
| `resetKvCache()` | Fits | Zeroes the recurrent state and the convolution ring. The transformer loops it over every layer unconditionally, so nothing gates it away. Misnamed for this layer, semantically exact. |
| `rewindKvCache(position)` | Fits | Returns `false`. The interface already anticipated refusal: the transformer ANDs results across layers and documents all-or-nothing, because a bounded sliding-window ring can already refuse. |
| `supportsKvCache()` | Harmless | Conflates "holds a per-token cache" with "holds resettable sequence state". Every call site in the tree is a `toString()` diagnostic or a test, so there is no behavioural consequence. |
| `setState(const GqaState&)` | **Does not fit** | A concrete GQA workspace — seven `ITensor*`, `q_permute`/`preatt`/`att`/`v_out` plus decode variants — in an otherwise generic interface. A DeltaNet block uses none of them and needs entirely different scratch: chunk-parallel delta-rule buffers, convolution staging, gate buffers. |

The single failure is a concrete type where a generic one belonged, not a structural mismatch.
The fix is to make the workspace an associated type of the layer rather than a fixed struct in
the interface. Deliberately deferred until the DeltaNet workspace shape is known from working
code — designing the generalization before there is a second instance to generalize over would
be guessing.

> **Resolved 2026-08-19, and the deferral paid.** With the block built, the generalization
> turned out not to be needed: `QwenDeltaNetBlock` accepts `setState` and ignores it — there is
> no attention transient to wire — and self-allocates its transients. The interface is
> unchanged across all three families. What remains is a second workspace struct for pooling
> the DeltaNet slots, which is a memory optimization rather than an interface question. Had
> the generalization been designed in 2026-08-16, it would have been built for a shape that
> never arrived.

### Consequence: prompt caching cannot work by rewinding

A KV cache rewinds because it stores per-token entries and only the fill pointer moves. A
recurrent state cannot: `S_t` is a lossy accumulation and `S_k` is unrecoverable from it for
`k < t`. With 48 of 64 layers refusing, the all-or-nothing rewind always fails and every
prefix reuse degrades to a full prefill.

Snapshot and restore works where rewind does not, because the state is constant-size: 0.141
GiB for the whole model — plus the 7.5 MiB convolution ring, which is part of the sequence
state and must ride every snapshot — plus the full-attention KV for the cached prefix. This
is a different mechanism from the one `PromptCaching.md` describes and that document should
record it.

---

## 8. What Must Be Built

Ordered by dependency, not priority. Nothing here is scheduled.

0. **The quality gate, in Python, before any CUDA.** Quantize a known-good model with the
   exact planned scheme — packing, group sizes, zero points, codebooks — and compare
   **generation quality against an IQ2_XXS quantization of the same model**, not against
   BF16. The claim under test is parity with the format class, not absence of degradation.
   Llama 3.2 3B first (known-good BF16 output exists to diff against), then the target
   layer stack on CPU. The result can reshape or kill items 1-3 before they exist; it is
   the cheapest de-risking step in the plan.
1. **`PerGroupCodebook2<kGroupSize>` and `PerGroupCodebook3<kGroupSize>` weight policies**,
   plus FP16 scales. Codebook dequantization only — the uniform-step variants lost the
   Phase 0 gate by 15x and are retired; the LUT tile-load path prices a 4-entry or 8-entry
   non-uniform codebook identically to uniform steps (Section 4). Pack the 3-bit codes as
   a **2-bit plane plus a 1-bit plane**, both byte-aligned, rather than 32 values in 12
   bytes: the planes keep tile loads aligned and let both policies share one kernel family.
   Validate against the step-0 oracle. Independently useful; carries no Qwen risk.
2. **W2A16 and W3A16 decode GEMV kernels**, plus a prefill path that respects the Section 5
   staging bound. The GEMV is done. Prefill is a **striped** two-phase pass, not the fused
   tile-load GEMM this item originally named: the fused kernel was priced against the
   alternative and lost on both memory and speed (Section 8). A fused W2/W3 GEMM remains
   worth building for prefill throughput, behind the tile ladder the W4A16 kernel also
   needs, and is no longer part of this item.
3. **The precision plan struct and concept** (Section 6), and threading it through `Linear`
   construction inside a block.
4. ~~**Depthwise causal `Conv1d`**, kernel 4.~~ **Built** (Phase 3): `CausalConv1d`, the first
   convolution component in the tree.
5. **Gated DeltaNet component**: L2 normalization on q/k, swish output gate, chunked-parallel
   prefill kernel and O(1) recurrent decode kernel, FP32 state. Build against a Python
   reference oracle. Validate **generation, not per-layer tolerance** — a per-layer test can
   pass while 48 recurrent layers compound to garbage, and that failure mode is more likely
   here than in an attention stack.
   *Phase 3 built the component, the recurrent kernel and the oracle. Still open: the
   chunked-parallel prefill kernel, and the generation-level validation — which needs a
   converter, so it lands with Phase 4.*
6. ~~**Host-resident `embed_tokens`** path.~~ **Built** (Phase 5): a residency axis on
   `TokenEmbedding`, pinned host memory, gathered in place by the unchanged kernel. See
   *Host-resident embedding* below.
7. **`PerChannelKvFp8` KV-cache policy.** Optional for the 16K baseline (BF16 KV fits) and
   the price of the 32K stretch (Section 5). Independently useful for Gemma and Llama.
8. ~~**Qwen block types, model, config, converter.**~~ **Built**: blocks, model and config
   in Phase 3, the BF16 converter in Phase 4, and the quantizing packer in Phase 5 (see
   *The Qwen packer* below), which has now produced the full 15.09 GiB artifact. The
   checkpoint carries the MTP tensors (~0.45 B); both converters skip them.
9. **Corpus perplexity through Mila's inference path.** Teacher-forced summed
   log-likelihood over a fixed held-out corpus. Nothing in the tree does this —
   `quality_gate.py` is Python fake-quantization and Bard's perplexity is a training loss —
   and without it the Phase 5 gate is a judgement call. Perplexity needs a logit at every
   position, which is exactly what the Section 5 prefill constraint forbids materializing at
   once, so the evaluation accumulates log-likelihood chunk by chunk. Independently useful
   for Gemma and Llama.
   *Built on Qwen:* `LanguageModelNetwork::scoreTokens` returns a `SequenceLogLikelihood`
   (summed log-probability plus the count of scored positions, reported separately so a
   corpus divides once). `QwenTransformer` implements it by evaluating the head in windows of
   `QwenConfig::withLanguageModelHeadPositions` rows inside the prefill chunk loop, reducing
   each window before the next overwrites it. The width defaults to 1 -- the one row
   generation reads -- and `resolveLanguageModelHeadPositions` bounds it by the prefill chunk
   for both `build()` and `getRequiredMemory()`, so a scoring build cannot allocate a head the
   prediction never named. It is a run capacity and deliberately absent from `toMetadata()`.
   The oracle is the generation path itself: scoring position p must equal what a prefill of
   tokens[0..p] says about token p+1, which is what pins the target alignment.
   **The gate is measured and PASSES** (2026-08-25). `DISABLED_QualityGateAcrossContextLengths`
   runs both arms at three context lengths over the same ~31,650 positions of wikitext-2 test,
   head width 1, each cell its own deployment, all six on the 5060 Ti pinned by UUID:

   | Context | Oracle 4.125 b | Plan 2.82 b | Ratio |
   |---------|----------------|-------------|-------|
   | 4096    | 6.439          | 7.089       | 1.101 |
   | 8192    | 6.126          | 6.704       | 1.094 |
   | 16384   | 5.686          | 6.478       | **1.139** |

   **1.139 at 16K against a threshold of 1.25** set before this table was read. A separate
   1024-token run measured 1.137 (6.606 / 7.513) over a different, shorter span of the corpus,
   so its absolute figures do not belong in the table -- but its ratio says the same thing.
   Across 1024 to 16384 the ratio is flat at 1.09-1.14: **the gap does not widen with context**,
   which was the specific failure this sweep existed to catch.

   Two readings the table earns beyond the pass:

   - **Both arms improve with context** -- the oracle from 6.439 to 5.686, the plan from 7.089
     to 6.478. Long-context prediction is working on both, which is a load-bearing sanity check
     on the DeltaNet stack independent of quantization.
   - **The quantized arm captures about half the benefit of the extra context.** From 8K to
     16K the oracle improves 7.2% and the plan 3.4%, which is why the ratio ticks up at the
     last row. Mild, non-monotonic (4K to 8K moved the other way), and well inside the
     threshold -- but it is the compounding signature the recurrent layers make plausible, and
     it points the same way at every larger context. **Re-run this gate before any 32K claim.**

   Cost of record: 33 minutes for six loads and ~190,000 scored positions.

   **The other two criteria** (`DISABLED_DivergenceAgainstTheOracle`, both arms, one card,
   greedy, six fixed prompts, 128-token budget):

   | Prompt                        | Diverges at | KL(oracle\|\|plan) | Top-1 |
   |-------------------------------|-------------|--------------------|-------|
   | The capital of France is      | 38          | 0.1463             | same  |
   | In 1969, humans first walked  | 1           | 0.1609             | same  |
   | def fibonacci(n):             | 28          | 0.2957             | same  |
   | list vs tuple in Python       | 1           | 0.0157             | same  |
   | Once upon a time...           | 0           | 0.5494             | DIFF  |
   | Summarize in one sentence     | >=9 (to EOS)| 0.0066             | same  |

   Mean KL **0.196 nats**, top-1 agreement 5 of 6.

   **The logit divergence is the informative half, and it corroborates the perplexity gate.**
   0.196 nats against the 0.130 that ln(1.139) implies is the same order from an independent
   measurement. The two prompts where the arms behave most differently are narrative (0.549,
   and the only top-1 disagreement) and code (0.296) -- the first because creative
   continuation is a field of near-ties where any perturbation flips the argmax, the second
   because code is the prompt class where the plan is furthest from the oracle while still
   agreeing on the next token.

   **The divergence index is descriptive only, and is REPLACED as evidence by trajectory
   cost.** A bare index cannot discriminate: greedy decoding is chaotic, and this project has
   already measured two *same-precision* builds forking the same way (Llama 3.2 3B, Ada
   against Blackwell: BF16, FP8 and FP4 all fork, at a token index set by the prompt rather
   than the precision). The criterion is now: **take each arm's greedy continuation, score
   both teacher-forced under the ORACLE over the same token count, and report what the plan's
   road gives up in nats per token.** The prompt is common to both and cancels.

   | Prompt                        | Forks at | KL(o\|\|p) | Plan cost/token |
   |-------------------------------|----------|------------|-----------------|
   | The capital of France is      | 38       | 0.1463     | 0.0517          |
   | In 1969, humans first walked  | 1        | 0.1609     | **-0.0508**     |
   | def fibonacci(n):             | 28       | 0.2957     | 0.1643          |
   | list vs tuple in Python       | 1        | 0.0157     | 0.2582          |
   | Once upon a time...           | 0        | 0.5494     | 0.2540          |
   | Summarize in one sentence     | >=9      | 0.0066     | 0.0000          |

   **Mean trajectory cost 0.1129 nats/token.** Two rows make the case for the change on their
   own: *"In 1969"* and *"list vs tuple"* both fork at token 1, and one costs **-0.051** while
   the other costs **+0.258**. Identical index, opposite meaning. The negative is not an
   error -- greedy is locally, not globally, optimal, so the plan's road can lead somewhere the
   oracle scores above its own continuation, and when it does the fork was noise. Only the
   MEAN is asserted non-negative for that reason. *"Summarize"* costs exactly zero: both arms
   produced the same nine tokens to EOS.

   **Three independent measurements now agree**: 0.130 nats implied by the perplexity ratio
   over ~31,650 positions, 0.113 from generated trajectories, and 0.196 from KL at the prompt
   boundary. The first two are the same kind of quantity and land within 15% of each other.
   The KL is the outlier and the reason is visible in the table -- it reads a single position
   right after a short prompt, where context is thinnest. Note it also does not track the
   trajectory cost per prompt: *"list vs tuple"* has the LOWEST KL and the HIGHEST cost. Strong
   agreement about the next token does not imply the road stays close, which is the second
   argument for measuring trajectories rather than positions.

   **What this rules out.** Cost per generated token (0.113) does not exceed what per-token
   prediction predicts (0.130), so quantization damage does not compound along a 128-token
   greedy generation beyond the per-token rate. That is direct evidence against the runaway
   the 48 recurrent layers make plausible -- at this length.

   The test split is load-bearing: the packer calibrates on `wiki.train.raw`, so scoring on
   that text would measure memorization of the calibration set. The protocol -- segmentation,
   head width, corpus -- is part of the measurement, and so is the card: the two GPUs disagree
   in the last digits (7.506 on the 4070, 7.513 on the 5060 Ti), which is float
   non-associativity between architectures and not a defect, so a ratio built from two cards
   is not a measurement. The figures are tokenizer-dependent and comparable only to another
   Mila run under this protocol.

   The FP4 oracle needed one loader change to become reachable: `dispatchQwenWeightPlan` had
   refused every uniform mode, and `validateArtifact` had refused a BF16 artifact for any
   quantized build. The second refusal was too broad -- FP4 and FP8 are DERIVED from the
   weights at load, which this family already does for its own attention and head projections,
   while a plan's codebooks are FITTED offline and genuinely cannot be recovered. The oracle
   now loads from the reference blob with no repack.

   *Still open:* the head runs at width 1 only (see below); the reduction is on the host and
   dominates a corpus run; and the other three families still build their heads at T=1, so
   only Qwen can be scored.

   **The head could not be evaluated at more than one position until 2026-08-26, and the
   reason is worth keeping.** At `outer_size > 1` a `PerGroupFp4` Linear takes the W4A8-FP8
   prefill path, which upcast the whole weight matrix into scratch unstriped: for the head,
   248320 x 5120 = 1212.5 MiB asked for whether the caller wanted 8 output rows or 512. It did
   not fit beside the model and aborted with no diagnostic. This was the Section 5 constraint
   above being broken -- by the head rather than the FFN the constraint was written for, and
   invisibly, because prefill evaluates the head at exactly one position, which takes the
   decode matvec and never reaches that path. Teacher-forced scoring was the first caller to
   ask for more.

   **Fixed by capping every packed format's staging at 256 MiB and striping the W4A8 path**,
   which `runStagedPrefill` already did for the codebook and FP8 per-channel policies. The cap
   is a no-op for every shape previously measured -- the 27B's feed-forward matrices are
   178 MiB and still expand in one pass -- and forces the head into ten strips. Striping the
   FP8 path additionally needed `build_fp8_prefill_plan` to accept an output row stride;
   `build_linear_plan` rejects that parameter on its own quantized branch reasoning that C is
   column-major, which does not follow, since column-major C carries a leading dimension as
   naturally as row-major. Pinned bitwise, striped against unstriped, including a ragged
   trailing strip.

   The head now runs at width 64: perplexity 7.515 against 7.513 at width 1, and 220
   positions/s against 156 -- only 1.4x, because the head was never what scoring spends its
   time on.

   **Where it does go, measured** (`DISABLED_ScoringCostBreakdown`, packed 27B, 7443 positions,
   baseline = a one-token generate over the same segment):

   | Component                          | Time   | Share |
   |------------------------------------|--------|-------|
   | Model forward (prefill)            | 23.2 s | 68%   |
   | Everything scoring adds            | 11.0 s | 32%   |

   Within that 11 s the transfer is not a factor -- 3.7 GB of logit rows is about 0.3 s -- and
   the host `exp` loop is essentially all of it, its 9.2 s prediction landing on 11.0 s
   measured. **So a device-side reduction is capped at 1.45x and is not worth building.** The
   cheap version, parallelising the host loop across cores, captures most of the same 11 s in
   a few lines with no kernel and no numerics risk.

   The number that matters more is the 68%: scoring is dominated by prefill, and prefill on
   this model still runs the DeltaNet recurrence sequentially because the chunked UT-transform
   kernel does not exist. That one kernel owns two thirds of every scoring run, on top of
   being what Section 8 item 5 says the 27B is not shippable without.

### Phasing

The list above is inventory; this is the order it lands, with the gate each phase must pass
before the next is worth starting. **The C++ lands in the normal tree** — policies beside the
other weight policies, the operation beside the other Linear backends, kernels under that
operation, dispatch rows in `OperationTraits.Cuda.ixx`.

**Isolation comes from the work being inert, not from where it lives.** A new policy, codec or
kernel costs nothing until a model instantiates it, and no shipped family does. Two earlier
isolation mechanisms have been retired on that reasoning: `Src/Experimental/` with its
`MILA_ENABLE_EXPERIMENTAL` flag (2026-08-17), and the `qwen3.8-quantization` branch, merged to
`dev` and deleted 2026-08-19. Neither was buying anything the type system was not already
providing, and both cost more than they saved — the branch in divergence, the flag in a second
build configuration.

Two tracks are independent until Phase 4 joins them: the **storage track** (Phases 0-2) touches
no Qwen code, and the **mixer track** (Phase 3) needs no quantization. Nothing here is scheduled.

**Phase 0 — quality gate** (item 0). Python only. *Exit:* the planned scheme — codebooks,
group sizes, zero points — reaches IQ2_XXS-class generation quality on Llama 3.2 3B.
Failure reshapes the Section 5 allocation before any kernel exists; if no allocation near
2.9 bits survives, the experiment's answer is "no" at the cost of a notebook, not a kernel
family.

**Phase 1 — sub-4-bit storage on a proven model** (items 1, 2). *Exit:* Llama 3.2 3B runs
end-to-end on the new policies; kernel dequantization bit-matches the Phase 0 oracle, and
generation quality matches what the oracle predicts — the oracle defines the expected
degradation, so the gate is oracle parity, not absolute coherence. FP16 scales land here
and benefit Gemma and Llama immediately.

**Phase 2 — precision plan struct** (item 3). Proven on the **Qwen family**, built the way the
tree builds every family. *Exit:* per-role policies dispatch; a plan missing a role is a compile
error at the block; the declaration-site alias reads — which is the property the experiment
exists to demonstrate.

> **Revision 2026-08-18:** this phase previously read "proven on an existing block, not a Qwen
> one: instantiate a Llama or Gemma block from a mixed plan." That was wrong on three counts.
> Gemma 4 is the token-parity oracle and the chat default, so editing it risks the thing every
> other gate is measured against. Neither Gemma nor Llama gains anything from per-role precision,
> so the plan would carry no load. And a Gemma block declared with a mixed plan is a *contrived*
> declaration — it compiles and demonstrates nothing, which is the same failure Section 6 rejects
> the variadic form for. Llama is a fallback only if the Qwen route proves impossible.
>
> Model the family on **Gemma**, for an architectural reason rather than recency: Gemma
> interleaves global and sliding-window layers, so it already solves the problem Qwen has — a
> transformer holding heterogeneous layer kinds (Qwen interleaves 3 linear : 1 full,
> `full_attention_interval: 4`). Copy the transformer-level list, where `GemmaTransformer` builds
> two block types and stores both as `ITransformerBlock*`. Do **not** copy Gemma's `bool kGlobal`
> flag: that is right for one mixer with two geometries, and Qwen has two different mixers.
> Llama is a uniform stack and teaches nothing here. This is reading Gemma, not editing it.

**Phase 3 — Gated DeltaNet** (items 4, 5). Runs in parallel with the storage track. Conv1d
first, then the mixer, at BF16 against the Python oracle on real checkpoint weights — ref
from `output_hidden_states`, the Gemma parity method. *Exit:* chunked prefill and
token-by-token decode produce identical state and output for the same input; oracle parity
per layer; state-plus-conv-ring snapshot/restore roundtrips exactly.

**Phase 4 — chassis at reference precision** (items 6, 8). Block types including the
output-gated GQA block, model, config, converter, host-resident embedding. The full model
cannot run unquantized on this card, so the gate runs CPU-resident or layer-streamed.
*Exit:* end-to-end hidden-state parity against the HF reference on a short prompt at BF16,
and matching last-position logits.

**Phase 5 — the plan applied** (join of both tracks). Quantized load per Section 5 on the
12 GiB card at the 16K baseline. *Exit:* measured VRAM within the Section 5 table, decode
rate measured against the 47 tok/s ceiling, and quality measured against the 16 GiB FP4
oracle below — corpus perplexity ratio, the divergence point of matched greedy generations,
and last-position logit divergence on a fixed prompt set. The Section 9 quality question
gets its first real answer here, and only here.

**The perplexity threshold: the ratio must stay under 1.25 at 16K**, the context length the
model is sold at, on the wikitext-2 test split under the protocol item 9 defines. Two things
about how that number was chosen, because a gate settled after the fact is not a gate:

- **It was written down before the sweep that tests it was read.** The 1024-token ratio
  (1.137) was known; whether the gap widens with context was not, and widening is the failure
  this threshold exists to catch. 1.25 leaves the measured 1024 figure roughly half its
  distance to the line, so a gap that grows by half again with context still passes and one
  that doubles does not.
- **It is a ratio against FP4, not against BF16**, because a BF16 27B fits no card here. FP4
  is not free, so the true distance from reference precision is larger than any number this
  gate reports. Do not quote the ratio as the cost of quantization; it is the cost of the last
  1.3 bits.

Failing it does not condemn the allocation — Section 9's first lever is recalibrating on
prose plus code rather than wikitext alone, and a re-pack is 50 minutes. It condemns
*shipping* the allocation unexamined, which is the whole point of writing a number down.

**Perplexity is the floor of this gate, not its ceiling.** It is prose next-token accuracy,
which is the property a quantized model keeps longest; instruction-following, tool calls and
long-horizon coherence degrade earlier and are not measured here (the Phase 0 caveats say the
same). A pass on 1.25 licenses "the allocation is sound", never "the model is undamaged".

**Phase 6 — stretch** (item 7 plus margin). FP8 KV cache; spend the freed margin per
Section 9 — bits or 32K; snapshot/restore prompt caching, recorded in `PromptCaching.md`.

### The 16 GiB oracle (2026-08-17)

An RTX 5060 Ti 16GB (Blackwell, sm_120) joins the rig alongside the 4070, headless. **The
12 GiB card remains the target** — the claim under test is a 27B model on the card most
people already own, and nothing in Sections 4-5 relaxes. The second card's role is
measurement.

It supplies the rung the evidence chain is missing. Today the chain steps from a 3B proxy's
wikitext perplexity straight to a judgement about the 27B, because 53.8 GB of BF16 runs on
neither card. At an estimated 15.0-15.5 GiB usable headless, uniform `PerGroupFp4<128>`
fits — 12.31 GiB of weights, 14.25 GiB total at the 16K BF16-KV baseline — so the chain
becomes:

```
HF reference (Phase 4, CPU-resident or layer-streamed)
  -> 27B uniform FP4 on the 16 GiB card
    -> 27B at 2.90 bits per Section 5 on the 12 GiB card
```

The middle rung is a **relative** oracle, not ground truth. Phase 4 is unchanged and still
gates against the HF reference. What the card buys is a same-model, same-tree, same-kernel
reference at a precision known to be near-lossless, for Phase 5 to measure the Section 5
allocation against.

Two prerequisites, in order:

1. **Cross-arch kernel agreement, established on a model that fits both cards.** The oracle
   runs sm_120 and the target sm_89, so a disagreement between them is ambiguous between
   bits and architecture — and the 27B cannot disambiguate it, since the FP4 build does not
   fit the 4070. Gemma 4 12B FP4 and Llama 3.2 3B FP4, token-for-token across both cards,
   first.
2. **Item 9.** Without it the Phase 5 gate has no number to report.

Two properties of the oracle that are not regressions. It is **slower** than the target:
448 GB/s against the 4070's 504, reading 12.31 GiB of weights per token instead of 8.65,
for a ceiling near 32 tok/s. And Section 4's "the bit width is purely a storage decision" is
a statement about SM 8.9 — sm_120 has FP4 hardware, so it does not hold on the oracle.
Reaching that hardware needs the activation-quantization axis and is outside this track.

### Phase 0 first results (2026-08-16)

Harness: `Mila/Tools/Quantization/quality_gate.py` (fake-quantization on Llama 3.2 3B
Instruct, greedy generation plus a small-text perplexity probe; ratios are against the BF16
reference on the same probe).

| Variant | Avg bits | PPL ratio vs BF16 | Output character |
|---|---|---|---|
| Uniform steps, data-free | 2.82 | 1608x | noise |
| Codebook, data-free | 2.78 | 107x | noise |
| Codebook + calibration | 2.78 | 5.9x | grammatical, repetition loops |
| Codebook + calibration + fp4 edge layers (2+2) | 2.97 | 2.45x | near-coherent, factual slips |

Three findings, each load-bearing:

1. **Calibration is mandatory, and that collides with the load-time quantization design.**
   Data-free round-to-nearest at ~2.8 bits destroys the model outright — and load-time
   `loadParameter()` quantization from a BF16 checkpoint is inherently data-free. The
   sub-4-bit path therefore needs either a converter that emits **pre-quantized codes**
   (breaking "converter always writes BF16" for these policies) or a calibration artifact
   consumed at load. This reshapes item 8 and is the biggest design consequence of the gate
   so far.
2. **The codebook lever is real and large**: 15x perplexity improvement over uniform steps
   at equal bits before calibration, consistent with the Section 4 prediction.
3. **Edge-layer protection is cheap on the real target.** Holding the first and last two
   layers at fp4 cost 0.19 bits on the 28-layer proxy but costs only ~0.24 GiB on the
   64-layer target (4 of 64 layers), inside the 16K margin.

**Gate result (2026-08-16): PASSED.** Measured on wikitext-2 test (first 131,072 tokens),
held-out train-split calibration, identical probe and code path for every row (the GGUF
baseline is dequantized at load, so kernel differences are excluded by construction). The
trajectory table above uses the small built-in probe, which flatters (its 2.45x is 2.99x
on the corpus); only the numbers below are of record:

| Candidate | Bits | PPL ratio vs BF16 |
|---|---|---|
| UD-IQ2_XXS (the format-class pass line) | ~2.6 | 2.57 |
| Codebook + calibration + fp4 edge layers | 2.97 | 2.99 |
| + sequential GPTQ error compensation | 2.97 | **1.67** |

Compensation closed the gap and then some: the planned scheme at its planned bit budget
sits 35% below the IQ2_XXS line, without E8-lattice machinery. Caveats of record: one
proxy model (Llama 3.2 3B, tied lm_head so that row went unmeasured); perplexity plus
short greedy generations only — the long-horizon capabilities Section 9 worries about are
not measured by this gate; and the scheme spends ~0.4 more bits than the IQ2_XXS file,
which is the budget Section 5 allocates.

**What the pass reshapes:** compensation requires calibration activations and sequential
layer-by-layer propagation, so sub-4-bit quantization is definitively an **offline,
converter-side step**. Item 8 gains: the converter emits pre-quantized codes plus
per-tensor codebooks for the sub-4-bit policies, and `loadParameter()` for these policies
uploads codes rather than quantizing. Phase 1's kernel contract is unchanged — LUT
dequantization of codes times FP16 group scales; what changes is where the codes come from.

### Mila's fused tensors constrain what an artifact can carry (measured 2026-08-17)

Mila fuses projections that HuggingFace keeps separate, and the packer quantizes per HF
linear, so the two do not line up:

| Mila tensor | HuggingFace source |
|---|---|
| `tf_layer_N.fc_gate_up` | `gate_proj` + `up_proj` |
| `tf_layer_N.fc_qkv_proj` | `q_proj` + `k_proj` + `v_proj` |
| `tf_layer_N.fc_down` | `down_proj` |
| `tf_layer_N.fc_out_proj` | `o_proj` |

Codes and scales concatenate along the output axis without trouble -- packing is row-major
and rows are independent. **A codebook does not: it is one table per tensor, and a fused
tensor can carry exactly one.**

The map above is executable, in `Tools/Converters/common.py` (`LLAMA_LAYER_TENSORS` and
`expand_llama_tensor_map`), so the packer and the BF16 converter name tensors from one
source rather than two. `is_linear` on each entry is what lets the packer pick the
quantization targets out of it.

For `fc_gate_up` the answer is to fit both projections jointly, and it costs **0.77%**.
Measured on Llama 3.2 3B Instruct, both arms deterministic, back to back on one build at 8
calibration samples and 65,536 eval tokens, identical BF16 reference of 10.988:

| Fit | PPL ratio vs BF16 |
|---|---|
| Per-HF-linear codebooks | 1.817 |
| One codebook across the gate/up pair | 1.831 |

Small, real, and in the direction theory requires -- one table serving two tensors has
strictly less freedom than two fitted independently. Worth paying: it is 0.77% against a
35% margin to the IQ2_XXS line, and the alternative is a format change (a table per row
range) reaching the packing codec, the policy, both kernels and the operation.

**This number only exists because the harness was made deterministic first.** Earlier
non-deterministic runs put the shared fit *ahead* by 0.9%, which is impossible for a
strictly less expressive fit and was the tell that the comparison was broken (next
subsection).

**Fit the pair jointly by sampling each tensor separately and concatenating the samples**,
never the tensors: normalization makes three FP32 copies of its input, so concatenating
first doubles the largest transient in the pass and pushed a 12 GiB card into PCIe spill.

### The gate was not reproducible, and is now (measured 2026-08-17)

Five runs of one identical configuration returned ratios 1.792, 1.807, 1.807, 1.847 and
1.915 -- **sigma 2.7%, range 6.9%**. The seed is set before the layer walk, so the random
subsampling was already identical run to run. The spread came from cuBLAS choosing GEMM
algorithms by available workspace, which changes summation order in the Hessian
accumulation; GPTQ then compounds it, because every layer quantizes against the previous
layer's already-quantized outputs.

`torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` fixes it
completely: three repeats returned 1.831 exactly, with byte-identical generations. No
operation in the path lacked a deterministic implementation. It costs about 35% throughput
(5.6 min per run against 4.1 at these settings), which is the right trade for a harness
whose output is a number of record. `enforce_determinism()` runs unconditionally.

**What this costs the earlier figures.** Every Phase 0 number above -- 1.67 for the
compensated scheme, 2.57 for the IQ2_XXS pass line, 2.99 without compensation -- is a single
run of the non-deterministic harness and carries roughly +/-2.7%. The gaps those numbers
establish are far larger than that (1.67 against 2.57 is about 13 sigma), so **the Phase 0
pass verdict stands unchanged**. What was never resolvable, and must not be quoted as
though it were, is any difference of a few percent. Re-run under determinism before
tightening any of them.

**What it demands of Phase 5.** That gate's exit criterion compares the 2.9-bit build
against the FP4 oracle by perplexity. Determinism is a precondition for that comparison to
mean anything, and the same discipline applies to the on-device perplexity path of item 9.

`fc_qkv_proj` cannot be solved the same way, and does not need to be. The Phase 0 research
scheme assigns q/k to `cb8` and v to `cb4` -- two *formats* in one fused tensor, which no
shared table reconciles. But Section 5 puts full attention at `PerGroupFp4<128>`, and step 5
below already keeps everything at 4 bits and above in BF16 in the artifact, quantized at
load. FP4's level table is format-defined and identical for every tensor, so fusing FP4
projections is trivially valid. **The artifact carries codebook tensors only**; attention
and `lm_head` ride the existing quantize-on-load path. The research scheme that maximizes
compression is not the deployment allocation, and the artifact follows the latter.

**The gate needs a 12 GiB card.** Reserved memory climbs monotonically across the 28 layers
-- 8.5 GB to 11.5 GB, spilling ~0.6 GB on the last layers -- because torch's caching
allocator holds freed blocks that `hessians.clear()` releases only by reference. An
`empty_cache()` per layer bounds it; without it the pass still completes but crawls once it
spills.

### Mila reads a Python-written safetensors file (verified 2026-08-18)

The artifact design rests on Python writing the container and `PretrainedModelReader`
reading it, and that had never been run. It works. A file written by the `safetensors` pip
package (0.7.0, torch 2.11, the Converters venv) carrying the tensor set this artifact
will carry -- U8 codes, FP32 scales, an FP32 codebook, a U8 high plane, a BF16 embedding,
and `__metadata__` holding `mila_config` and `mila_quantization` -- read back through the
container sniff, the header parse, the offset rebasing, `readTensorBlob()` byte-for-byte
including BF16 bit patterns, and `streamTensorBlobs()`. `1e-06` survives the metadata
parser's `stof`, and `num_heads` does not match inside `num_kv_heads`.

Two properties of the Python writer the packer must respect:

- **No I64, F64, U64, or BOOL.** Mila has no wire code for them, and the reject is loud at
  construction (`safetensors: unsupported dtype 'I64'`). Torch defaults integer tensors to
  int64, so any id or index tensor must be cast to int32 before `save_file`.
- **The data region is not in insertion order.** The writer sorts by dtype size descending,
  then by name, so a declared order is not a file order. Mila is indifferent because it
  sorts by offset, but nothing downstream may assume otherwise.

Verified separately because one sample does not cover them: headers with 0 through 7 bytes
of the writer's trailing-space padding all read (the first sample happened to land
8-aligned, the rare case), a rank-0 scalar reads at `ndim == 0`, and FP16 and INT8 both
carry through.

Run as a temporary probe against real generated files, then reverted -- this path has no
regression cover yet. The durable form writes the Python container's byte layout in C++
from the format definition, the way `SafeTensors.Cpu.cpp` already pins the MILA layout, so
no Python is needed at test time and it rides the CPU-only CI gate.

### Converter quantization pipeline (design of record, 2026-08-16)

The sub-4-bit half of item 8, designed now because Phase 1 needs its output as the kernel
oracle. Offline, Python, layer-streaming — the 27B model never resides anywhere in full,
and peak device memory is under 4 GiB, so the conversion itself runs on the target card.

1. Index the HF safetensors checkpoint by mmap; drop the vision tower and MTP tensors at
   the index. Never materialize the whole model.
2. Calibration set: ~32 samples x 2048 tokens, **prose plus code** — the importance
   weights should reflect the traffic the model is valued for, not only encyclopedia text.
3. Run the embedding once over the set -> initial hidden states (~0.7 GB BF16).
4. Walk the 64 decoder layers in order. Per layer: load its tensors to the device
   (~740 MB) -> forward the calibration set through the HF `qwen3_5` layer code with
   pre-hooks accumulating per-linear Hessians (`down_proj` at 17408^2 FP32 is 1.2 GB, a
   whole layer ~1.9 GB) -> per target linear: fit the per-tensor codebook with
   Hessian-diagonal importance, run the compensated column walk, pack -> then
   quantize-dequantize the FP4-at-load tensors **in place**, so calibration for the layers
   downstream sees the damage the deployed network will actually have -> re-run the layer
   with quantized weights to produce the next layer's inputs -> stream the packed records
   to the artifact.
5. Artifact: only the codebook tensors carry the new record type (packed codes, FP16 group
   scales, 4- or 8-entry codebook, policy metadata). Everything at 4 bits and above —
   attention, `lm_head`, edge layers, beta/decay — stays BF16 in the artifact and keeps
   today's quantize-at-`loadParameter()` path, so the serialization change is one record
   kind. Size: ~5.5 GB packed + ~8.4 GB BF16 = **~14 GB**, from 54.7.
6. Validation: reload and bit-match dequantization against the Python oracle; Phase 4
   parity gates the end-to-end result.

Runtime is on the order of an hour on the 4070, dominated by the compensated column walk
(~2.5M columns across ~450 tensors). Two known risks: transformers' `qwen3_5` must expose
a pure-torch DeltaNet forward path (a five-minute check, and the same dependency the Phase
3 oracle carries), and the 54.7 GB checkpoint download happens only when Phase 4/5 needs
the real artifact — the packer is proven on the Llama 3.2 3B proxy first.

### Phase 1 status (2026-08-16, extended 2026-08-17)

Built, green, and proven against each other. Policies and the packed-layout codec sit in
`Mila/Src/Dnn/Quantization/Weight/`, the operation and its kernels in
`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/`, tests mirror both, and the Python
tooling is in `Mila/Tools/Quantization/`:

- **Policies** (`CodebookPolicies.ixx`): `PerGroupCodebook2<32>` / `PerGroupCodebook3<64>`,
  satisfying the production `WeightQuantPolicy` concept, FP16 scale dtype.
- **Normative packed layout + CPU codec** (`CodebookPacking.ixx`): 2-bit codes four per
  byte; 3-bit as two byte-aligned planes; FP16 scale bits; host half conversions with
  subnormals. The Python packer (`packing.py`) is held to it by a generated fixture the
  oracle test bit-matches in both directions.
- **W2A16/W3A16 decode GEMV** (`Kernels/CodebookGemv.cu`): one template kernel for both
  formats on the qfp4 pattern — codebook in registers, 16 codes per thread per iteration,
  one FP16 scale per chunk. Validated on device against the CPU codec.
- **Real packed artifact at model scale**: `quality_gate.py --gptq --emit-artifact` packs
  every codebook tensor of the GPTQ-quantized Llama 3.2 3B and verifies each dequantizes
  bit-for-bit to the weights the evaluated model carried (168 tensors, 0.84 GB payload,
  ~2.78 bits/weight with scales; quality through the artifact path: ratio 1.68).
- **The artifact now carries Mila names in the safetensors container** (2026-08-18): 196
  tensors over 56 fused linears for Llama 3.2 3B, 0.73 GB. Tensor names come from the
  converter's map (`Tools/Converters/common.py`), not a second copy, and every emitted
  dtype and shape is checked against what `Linear::initializeParameters` allocates for
  the policy before the file is written. Verified: all 196 read back through
  `PretrainedModelReader` -- names, dtypes, shapes, the whole `streamTensorBlobs` walk --
  and every primary weight names a tensor the BF16 converter also writes.

  Two consequences worth stating plainly. `--emit-artifact` now turns on
  `--fuse-gate-up-codebook` by itself, since `fc_gate_up` carries one table and is
  otherwise inexpressible; the packer refuses to fuse sources whose fitted tables differ
  rather than trusting the flag. And **the artifact is not the evaluated network**: it
  holds the FFN codebook subset only, while the run's perplexity covers the research
  allocation with attention quantized too.

  Open: `mila_quantization` is one string and this artifact carries two policies, so the
  per-tensor map rides beside it under `mila_codebook_policies`. Which of the two a Mila
  loader should trust is decided when the codebook load path exists.

- **Operation and dispatch** (`CudaLinearOp.ixx`, rows in `OperationTraits.Cuda.ixx`,
  tests in `CudaLinearOp.Codebook.Cuda.cpp`): the policies resolve through the production
  `OperationTraits` table to working decode and prefill forwards on real tensors, and they
  resolve to **the one CUDA Linear operation every other weight format uses**. The policy
  selects the decode kernel, the scale element type and the prefill strategy; a member the
  format has no meaning for is constrained away rather than present and throwing, so
  `quantize()` is uncallable here — the Phase 0 finding expressed in the type — while
  `setCodebookTensors()` is callable only on a format that has a table. Consolidated
  2026-08-18: a `CudaCodebookLinearOp` existed for five commits and was retired, having
  proven the drift a second operation causes (the striped staging below landed in it alone
  and the production FP4/FP8 two-phase path did not get it).
- **Two-phase prefill** (`Kernels/Codebook/CodebookDequantize.cu`): packed codes expand
  into the shared BF16 scratch, then a standard cuBLASLt GEMM — the same structure as the
  proven FP4 baseline, and after consolidation literally the same code, with bias added
  post-GEMM by `cuda_add_bias` as every other staged path does. It is the cheap route to
  end-to-end at 3B and does not scale unstriped: one Qwen3.8 FFN matrix expands to 170 MiB
  per forward, which is the constraint Section 5 already states.

**The traits table is open for extension (measured 2026-08-17).** MSVC accepts a
specialization of a template owned by another module, so a dispatch row can be registered
from outside the module that declares the primary. This was measured while the codebook
rows lived in a separate module and it remains true, but it is no longer load-bearing:
the rows now sit in `OperationTraits.Cuda.ixx`, the documented single registration point,
because the work needs no isolation mechanism of its own — an unused dispatch row costs
nothing. Recorded because it stays useful — it means an out-of-tree
consumer could add a row without patching Mila — not because this code demonstrates it.

**Decode is more accurate than prefill, and no prefill kernel can close the gap
(measured 2026-08-17).** The GEMV multiplies `codebook[code] * scale` in an FP32 register
and never stores it. A tensor-core prefill cannot, and the reason is a chain whose last
link is forced by the first:

1. Activations span a wide dynamic range — Linear's own oracle fixture spans fifteen
   decades, and that is what caught the +98 incoherence.
2. So the activation operand must be BF16; FP16 overflows above 65504.
3. cuBLASLt requires both operands to carry the same type.
4. Therefore the staged weight is BF16 as well — 8 mantissa bits, 2^-9 per weight.

Measured over 4096 comparisons at 3072 columns: decode matched a double-precision
reference everywhere within 1e-4 of the sum of absolute products, prefill needed 1.8e-4,
and 93% of the gap vanished once the reference modelled BF16 weight rounding instead of
FP32. **The fused W2/W3 GEMM of item 2 does not close it** — moving dequantization into
the tile load changes where the BF16 conversion happens, not whether it happens; the MMA
fragment is BF16 either way.

Step 4 is the irritating one, because FP16 would otherwise be the better staging type:
10 mantissa bits instead of 8, same two bytes, same tensor-core rate on Ada, and provably
in range for these weights, since a codebook entry is normalized to [-1, 1] and the group
scale is already FP16. The activations forbid it, not the weights. Reaching FP16 operands
therefore means bounding the activations first — which is the per-token scaling machinery
`Fp8ActivationPrefill.md` already built, and a different design than "two-phase, but
fused". Blackwell moves this the wrong way: FP4/FP6 block-scaled MMA is lower operand
precision, so the gap widens on sm_120 rather than closing.

**This is a parity concern, not a quality one.** A 2^-9 representation error sits against a
scheme whose own quantization error is 1.67x perplexity; it will never surface in
generation. It surfaces when two paths are asked to agree numerically, which is what the
Phase 4/5 gates do — so those must compare like with like, and a prompt reprocessed
token-by-token will not reproduce its batched logits bit-for-bit.

The production FP4 two-phase path has the same property, and already accounts for it:
`Linear.Cuda.cpp`'s prefill-vs-decode oracle derives its budget from a bit-faithful CPU
model of both paths including the BF16 epilogue, and records the BF16-staging worst
deviation at 0.0073 of row absmax.

**Owed:** the codebook op's test budget anchors on the L1 mass (sum of absolute products),
while that same production comment records anchoring on L1 mass instead of row absmax as
"measured and rejected -- same 2.6x row spread, so it buys nothing for a less obvious
quantity." Re-derive the codebook budget against row absmax so the tree has one convention.

**The decode GEMV is instruction-bound, not bandwidth-bound (measured 2026-08-17).**
`CodebookLinearOp.Cuda.cpp`'s `DISABLED_DecodeBandwidth`, on a 3072x8192 projection,
RTX 4070:

| Path | Weight bytes | ms before | ms after | GB/s after |
|---|---|---|---|---|
| `PerGroupCodebook2<32>` | 7.9 MB | 0.0331 | 0.0302 | 260.5 |
| `PerGroupCodebook3<64>` | 10.2 MB | 0.0552 | 0.0322 | 317.0 |
| `PerGroupFp4<128>` (production, control) | 13.4 MB | 0.0270 | 0.0271 | 493.6 |

**Read these figures as relative, not absolute.** At 8-13 MB the weight matrix fits inside
the 4070's 36 MB L2, so the benchmark re-reads it from cache and every number here --
including qfp4's -- overstates what DRAM can sustain. Real decode streams 8.65 GiB per
token and is genuinely DRAM-bound. The comparison stays valid because all paths share the
same residency, and qfp4 is the unchanged control that proves the harness stable.

The cause was found by a variant probe rather than by profiling: a standalone kernel with
each candidate changed one at a time.

| Variant | 2-bit | 3-bit |
|---|---|---|
| baseline | 1.00x | 1.00x |
| codebook lookup removed entirely | 1.87x | 3.13x |
| lookup via `__shfl_sync` | 1.11x | 1.81x |
| activations staged in shared memory | 0.99x | 0.98x |

The runtime-indexed codebook lookup was 46% of the 2-bit kernel and 68% of the 3-bit one.
**The local-memory theory was wrong** -- ptxas reports 0 bytes of stack frame and 0 spill
stores at 37 registers. nvcc keeps the array in registers and lowers each lookup to a
select chain instead, roughly 3 comparisons at 4 entries and 7 at 8, which is why cost
tracked entry count. Holding one entry per lane and reading it back with a single
`__shfl_sync` is worth 1.71x on the 3-bit format in the real harness. **Activation traffic
was not a factor at all** -- staging `x` in shared memory measured neutral to negative, so
that hypothesis is closed rather than parked. FP4 E2M1 never had the problem because its
levels decode arithmetically instead of through a table.

**A full-mask shuffle requires a warp-uniform trip count.** The first version kept the
natural `for (c = threadIdx.x * 16; c < C; c += 512)` bound, under which lanes 0-1 run
twice and lanes 2-31 once at C = 544. Harmless with a select chain; with a shuffle the
surviving lanes wait forever on lanes that have already exited, and the kernel deadlocks
rather than returning a wrong answer. The loop now runs `ceil(C / 512)` iterations on every
lane and masks the tail. `CodebookGemvCuda.TwoBitGroup32WithBiasAndBlockTail` -- written
for tail *correctness* at 544 columns -- is what caught it; every 3072-column test passed,
so without that case the deadlock would have reached the first real layer of awkward width.

**Still short of qfp4, and still instruction-bound.** 3-bit now takes about the same time
as 2-bit (0.0322 vs 0.0302 ms) while moving 30% more bytes, which is the signature of an
instruction limit rather than a memory one. Closing it needs a different kernel structure
-- amortizing the unpack across several output rows per thread, or bucketing activations by
code -- not another peephole.

**What it costs the plan:** Section 5 derives ~47 tok/s from bytes per token, which assumes
decode is bandwidth-bound. These kernels are not, so that ceiling stays **unverified**
until they are measured against a DRAM-resident model rather than an L2-resident matrix.
Correctness is unaffected throughout -- this is throughput, not numerics.

### The GEMV gap closed, and it was not the lookup (measured 2026-08-18)

Reading the paragraph above against a fresh variant probe overturns two of its claims. Both
errors came from comparing numbers taken under different conditions, which is the failure
[[feedback_compare_arms_from_one_build]] names.

**There was no 1.7x sitting in the shuffles.** The `0.0178 ms` lookup-free figure was
measured on the pre-shuffle kernel and then read against the post-shuffle baseline. With
baseline and floor control built together, deleting the codebook lookup entirely is worth
**1.28x at 2 bits and nothing at all at 3 bits** -- the shuffle version had already taken
what the lookup had to give. A shared-memory table indexed by several packed codes at once
was measured as the replacement and **lost at both widths** (0.80x, 0.81x): a
runtime-indexed shared load costs more here than the shuffle it replaces. That closes the
lookup as an avenue rather than parking it.

**The limit was memory-level parallelism.** A warp issued one code word per iteration and
then waited on it, over only six to ten iterations. Walking several adjacent output rows in
one warp issues that many independent code loads before consuming any, and it is worth
1.10-1.32x depending on width and shape. Independent accumulators *within* a row -- the
dependent-FADD-chain theory -- were measured and do nothing (1.00x); ptxas already
reassociates. The counts differ by width because a 3-bit row costs two plane loads instead
of one, so it saturates at 2 rows and regresses beyond, while 2-bit peaks at 4.

Landed in `CodebookGemv.cu`: 4 rows per warp at 2 bits, 2 at 3 bits, 4 warps per block
(was 8). **Bit-identical** to the previous kernel -- the accumulation order within a row is
untouched, only the loads move -- so no numerics gate applies. Existing tail coverage
already exercises the new edge case: 27 rows at 4 per warp and 17 at 2 per warp both leave
a partial group.

In-tree `DISABLED_DecodeBandwidth`, RTX 4070, with the qfp4 control unchanged at 0.0269 ms
proving the harness stable:

| Path | Weight bytes | ms before | ms after | GB/s after |
|---|---|---|---|---|
| `PerGroupCodebook2<32>` | 7.9 MB | 0.0302 | 0.0252 | 312.5 |
| `PerGroupCodebook3<64>` | 10.2 MB | 0.0322 | 0.0292 | 349.8 |
| `PerGroupFp4<128>` (control) | 13.4 MB | 0.0271 | 0.0269 | 497.6 |

**qfp4 was never ahead on time, and the gap to it was an artifact of the metric.** All three
kernels ran within 0.005 ms of each other before this change and qfp4 looked 1.9x better
only because it moves 70% more bytes in that time. W2A16 is now *faster in wall time* than
qfp4 while reading 41% fewer bytes, which is what a decode kernel is actually asked for. A
GB/s figure divided by a time all three share measures the format, not the kernel.

**The right ceiling is 455 GB/s, not the card's 504, and it must be measured
DRAM-resident.** A read-only kernel streaming the same planes with the same layout sustains
452-457 GB/s at the Qwen FFN shape; every table above it is L2-resident, so qfp4's 497.6
sits *above* what DRAM can deliver and is not a bandwidth achievement. Against a rotating
weight set larger than L2 the landed kernels reach **86% of that ceiling** at the Qwen shape
(391 and 394 GB/s) and 82-88% at the Llama 3B shape, up from 60-76%. The remaining 12-18% is
decode arithmetic overlapping the stream, and Section 5's ~47 tok/s ceiling is now
supportable rather than unverified.

**Tuning this on the L2-resident test would have picked the worst production
configuration.** 8 and 16 rows per warp were the fastest L2 variants (1.41-1.44x) and are
*slower than the baseline* under DRAM residency (0.76-0.87x). The two residencies rank the
candidates differently, so a weight matrix small enough to cache is not a valid stand-in for
a model that streams.

### The fused prefill GEMM loses to striping the staging buffer (measured 2026-08-18)

Section 5's second implementation constraint requires the fused tile-load path because
dequantizing one 5120x17408 matrix to BF16 is 170 MiB. Priced properly, the fused kernel is
the wrong way to satisfy it, and a third option nobody had named is better on both axes.

**What fusion could win.** The dequantize pass costs 1.02 ms at 2 bits and 1.11 ms at 3, and
is **chunk-independent** -- it expands the whole matrix whatever `M` is -- while the GEMM is
linear in `M`. So two-phase carries a penalty that grows as chunks shrink, and Section 5's
memory pressure pushes chunks small. cuBLASLt runs the BF16 GEMM at 62 TFLOP/s throughout.

| Chunk `M` | GEMM alone | Two-phase, W2 | Most a perfect fused kernel could win |
|---|---|---|---|
| 128 | 0.42 ms | 1.44 ms | 3.42x |
| 256 | 0.74 ms | 1.76 ms | 2.37x |
| 512 | 1.48 ms | 2.50 ms | 1.69x |
| 1024 | 2.95 ms | 3.97 ms | 1.34x |
| 2048 | 5.94 ms | 6.96 ms | 1.17x |

This is a much better case than the W4A16 attempt had -- there the fused win was ~12% -- for
three compounding reasons: the expansion ratio is 6.4x at 2.5 bits against 3.9x at 4.125, the
Qwen FFN matrix is 89 M elements, and the chunk is small.

**What a fused kernel of this family actually delivers.** The tree's own Stage 2 cp.async
double-buffered WMMA kernel, measured at this shape, runs **18.6 TFLOP/s -- a flat 3.34x
below cuBLASLt** at every chunk. (That confirms the Stage 2 record with a number: chunk
independence achieved, cuBLASLt not approached.) Set against the ceilings above it wins only
at chunk 128 and below. FP4 decodes its levels arithmetically where a codebook must look
them up, so **18.6 TFLOP/s is an upper bound on a codebook version of the same geometry** --
the crossover would sit lower still. Winning at chunk 512 needs ~37 TFLOP/s, twice what this
geometry gives, so the fa-5090 ladder (XOR swizzle, `ldmatrix`, 128x128 tiles) is not polish
here but the entire task, and it is the step the W4A16 attempt stopped before.

**Striping the staging buffer beats it without a new kernel.** Nothing requires expanding
the whole matrix before one GEMM. Split the output into strips of `stripN` rows, dequantize
one strip and GEMM it: staging becomes `stripN x K`, total dequantize work is unchanged,
cuBLASLt keeps the GEMM, and the arithmetic is untouched so results stay bit-identical. At
chunk 512:

| Staging | Total | vs monolithic |
|---|---|---|
| 170.0 MiB (monolithic) | 2.44 ms | 1.00x |
| 85.0 MiB (2 strips) | 2.52 ms | 1.03x |
| 42.5 MiB (4 strips) | 2.80 ms | 1.15x |
| 21.2 MiB (8 strips) | 3.00 ms | 1.23x |
| 10.6 MiB (16 strips) | 3.19 ms | 1.31x |
| ~0 MiB (fused, Stage 2 geometry) | 4.96 ms | 2.04x |

**Eight strips dominate the fused kernel on both axes** -- 1.65x faster and down to 4.6% of
the 461 MiB Section 5 allots to all activations and scratch. cuBLASLt handles `N` = 1088
nearly as well as 17408 (the GEMM term grows only 18% across the whole range), so the cost is
the dequantize pass losing parallelism per launch and the loss of overlap, not GEMM
efficiency. Halving the buffer is nearly free at 1.03x.

**Consequence for Section 5 and for Phase 1.** The constraint reads as a bound on the staging
buffer, not a requirement for a particular kernel: the FFN must not expand a whole matrix at
once. The fused GEMM stops being a Phase 1 exit item and becomes what it is -- a prefill
*performance* project whose real content is the tile ladder, shared with FP4 and Gemma and
Llama, and whose gate is 37 TFLOP/s rather than correctness.

**Landed (2026-08-18).** `CudaLinearOp::runStagedPrefill` walks the output channels in strips
sized to hold staging under a 32 MiB cap: six strips at 28 MiB for the Qwen FFN shape, two at
24 MiB for Llama 3B. The cap is a **constructor parameter** defaulting to a per-policy
constant, because a real budget belongs to whoever owns device memory rather than to one
operation -- and because nothing else can exercise the striped path at a testable shape.

Since the consolidation the same routine serves FP8 per-channel, FP4 E2M1 and both codebook
formats; only the expansion kernel inside the loop differs. **Only the codebook formats set a
real cap.** FP8 and FP4 default to an unbounded one, which yields a single strip and therefore
the byte-for-byte prefill they had before, because capping them trades throughput for VRAM at
shapes where the trade has not been measured. Making that trade is now one constant rather
than a second operation.

Two implementation notes. The dequantize kernel needed **no change**: it addresses every
plane relative to row 0 of the pointers it is handed, so a strip is the row offset folded
into each pointer plus a shorter row count. And each strip's `C` is a column slice of the
full output, which needed one new thing outside the operation -- an optional
`output_row_stride` on `build_linear_plan`, defaulting to `out_features` so every existing
caller is unchanged, and rejected on the quantized FP8 path whose `C` is column-major.

The tests assert **bitwise** equality against the same operation run in one pass, not a
tolerance. A tolerance would pass equally on a version that split the contraction instead,
which is the variant that rounds partial sums to BF16 between strips and is exactly what
striping along the output dimension avoids. Ragged trailing strips get their own case (200
rows into strips of 80 leaves 40) since that width needs its own cuBLASLt plan. Verified by
mutation: dropping the per-strip scale offset fails all four.

Not measured, worth knowing. Overlapping strip `i+1`'s dequantize with strip `i`'s GEMM on a
second stream should recover most of the 1.23x, but the shared scratch buffer's safety
argument rests on one stream per context (`CudaExecutionContext.ixx:224`), so that is a real
change rather than a tuning knob.

The broader point outlives the bug: **fewer bits do not buy proportional decode speed.** The
FMA count per token is fixed by the parameter count, so shrinking the weights raises
arithmetic intensity until something other than DRAM becomes the limit. Any sub-4-bit
decode plan has to show where that crossover sits rather than assume the byte count leads.

Both of the items this paragraph once listed as remaining for the Phase 1 exit gate are
discharged (2026-08-18, both below). The GEMV gap is closed at 86% of the measured DRAM
ceiling. The fused W2/W3 prefill GEMM is **removed from the exit gate**: striping the staging
buffer satisfies the Section 5 constraint that motivated it, measured faster than the fused
kernel and with no numerics change, so the fused path is reclassified as a prefill performance
project. Striping is landed in the codebook operation. The artifact half -- emit the packed
tensors, read them back, load them through `Linear` and decode correctly -- is done and
verified end to end.

**That seam is now closed (2026-08-18).** It was not merely unexercised: `Linear<Cuda,
BF16, PerGroupCodebook2<32>>` did not compile at all. `loadParameter()`'s weight branch
chose between the packed load and `operation_->quantize()` at **runtime**, so the call had
to be well-formed for every quantized policy, and the codebook operations implement no
`quantize()` — deliberately, since their codes come from an offline fit the weights cannot
reconstruct. The fix is the smallest one that states the fact: `if constexpr
( HasCodebookTable<TWeightQuant> )` selects a load-only branch that refuses a
full-precision blob with a policy/artifact mismatch message. Every other policy compiles to
exactly what it did before, and Phase 2 no longer inherits the question.

Proven end to end: `CodebookLinearOpCuda.{TwoBit,ThreeBit}LoadsThroughLoadParameter` build
a real `Linear`, push the four tensors through `loadParameter()` under the names
`quality_gate.py` emits, and match the CPU codec. The emitted 3B artifact itself was loaded
the same way through `PretrainedModelReader` — `tf_layer_13.fc_down`, all 3072 rows within
1.7e-4 of the row L1 mass against a host dequantization of the same bytes. Full suite 1630
passed / 1 pre-existing skip / 0 failed.

Three things that ran differently from expectation:

- **The component's output buffer is sized from the `BuildContext` shape.** A `Linear`
  built for one row throws `View exceeds buffer bounds` on a batch-16 forward. Not a defect
  — the allocation has to come from somewhere — but a decode-shaped component cannot serve
  a prefill call, so any test comparing the two paths must build for the wider one.
- **The decode GEMV really is reached through the component.** Batch-1 and row 0 of a
  batch-16 forward over the identical vector agreed on only 1753 of 3072 outputs, and the
  two paths share no kernel, so the disagreement is the evidence that both ran.
- **The decode-vs-prefill reference split does not separate the paths at 3 bits on real
  weights.** Against the artifact's own tensors the two references landed at 1.68e-4 and
  1.94e-4 of row L1 mass — indistinguishable, because a 3-bit quantization step dwarfs a
  BF16 rounding. The measurement above stands; it was taken on synthetic tensors at 3072
  columns, and that is the condition under which the split is legible.

### Phase 2 status (2026-08-19)

**The exit gate is met.** Per-role policies dispatch, a plan missing a role is a compile error
at the block, and the declaration-site alias reads:

```cpp
using QwenAttentionLayer = QwenAttentionBlock<DeviceType::Cuda, TensorDataType::BF16,
    QwenPrecisionPlan>;
```

Built as a family, the way the tree builds every family: `Components/Transformers/Qwen/` holds
`Qwen.PrecisionPlan.ixx` (the Section 5 table as types), `Qwen.Config.ixx`, `Qwen.AttentionBlock.ixx`
and `Qwen.ixx`; the output gate is a component of its own under
`Components/Attention/OutputGate/`. Suite green at 1681 passed / 1 pre-existing skip.

Five decisions the code now carries, worth stating because none is recoverable from a diff:

- **`QwenTransformer` refused the published interleave at construction**, naming Phase 3.
  Only an all-full-attention configuration built. A transformer that quietly built 16 of 64
  layers is the worst failure available here, and the refusal was checked before any device
  memory was touched so it named the config rather than a build. *Superseded by Phase 3: the
  refusal is deleted and the published interleave builds.*
- **The output gate composes rather than fuses**: the shared elementwise activation op, then
  the TensorOps multiply in place over the gate's own output. Two launches on 16 of 64 layers.
  A fused kernel needs a new operation type and a dispatch row, and nothing has measured it.
- **The fused projection is `[query | gate | key | value]` with query and gate as contiguous
  halves**, and the block performs two splits — 3-way then 2-way. The halves cannot be reached
  as views instead: a row of the combined buffer is [query row | gate row], so the query rows
  are strided by twice the query width. The converter (item 8) owns emitting that order.
- **Untied tables forced a new role.** Gemma's single `TableQuantizationPolicy` exists to keep
  one *shared* table consistent across two consumers; Qwen's tables are independent and
  Section 5 prices them apart (BF16 host-resident against FP4), so the plan machinery gained
  a `LanguageModelHead` role beside `EmbeddingTable`.
- **QK-norm was recorded as absent, on this document's authority, and that was WRONG.** The
  reasoning was that Section 1's table had been re-verified against the raw `config.json` and
  named `attn_output_gate` explicitly while naming no QK-norm field. The flaw is that
  **QK-norm is not a config field in this architecture** — it is unconditional in the
  reference's attention constructor and visible only in the checkpoint's tensor names. No
  amount of re-reading `config.json` could have found it. *Corrected in Phase 3; see there
  for the general lesson.*

`QwenModel` is deliberately absent. Its whole job is loading weights, and with no converter
(item 8) and no DeltaNet layers it would be an entry point to a model that cannot load —
which gets mistaken for one that works. *Superseded: the converter and `QwenModel` both landed
2026-08-19; see below.*

### The converter and `QwenModel` (2026-08-19)

Both halves of item 8's remainder. `Tools/Converters/Qwen/convert_weights.py` streams the
checkpoint shard by shard — `from_pretrained` cannot run at 51.8 GiB against 31.8 GB of RAM —
and emits the four transforms this document settled: the `q_proj` de-interleave, the
`in_proj_qkv` split at `2 * key_dim`, the matching depthwise `conv1d` split, and raw norm
weights. The name map lives in `Converters/common.py` beside the Llama one, so the codebook
packer will name tensors from the same source.

Every rearrangement is verified **bit-identical** to the checkpoint, and the de-interleave
additionally against the reference's own expression
(`q_proj(x).view(*shape, heads, 2 * head_dim).chunk(2, -1)`) — with a negative control
confirming the naive first-half/second-half split fails that check. The full artifact is
**50.10 GiB, 851 tensors**; 851 consumed plus 348 skipped accounts for all 1199 checkpoint
tensors, and 50.10 GiB is the 53.8 GB text-only GGUF anchor Section 2 reconciles against.

**The BF16 27B fits nowhere on this hardware** — 50 GiB against a 12 or 16 GiB card and
31.8 GB of RAM. So reference precision cannot be exercised on the whole stack here at all,
which is not a gap in `QwenModel` but the precise reason Phase 4's parity harness must stream
layer by layer. What can be exercised is the converter's `--max-layers 4` fixture: three
DeltaNet layers and one full-attention layer, every block kind and every transform, at
7.57 GiB. `Tests/Dnn/Models/QwenModel.Load.Cuda.cpp` drives it and skips without it. Four of
sixty-four layers produce meaningless tokens, so nothing there asserts what is generated.

Three properties the model carries, none recoverable from a diff:

- **Reference precision only.** The uniform FP4/FP8 modes are refused by name. Section 5 is a
  per-role plan over codebook formats, so a uniform body is a *different allocation*, not Qwen
  at lower precision, and the artifact carrying the real plan is Phase 5.
- **No prefix-reuse path exists**, rather than Gemma's block present and permanently failing.
  Machinery that can never fire reads as a capability.
- **The recurrent state is self-cleaning at prefill-position-0** (`GatedDeltaRule::prefill`,
  `CausalConv1d::prefill`), so no explicit reset is needed — and that is also why `prefillFrom`
  at a non-zero offset must never be reached on this family. Pinned by a test asserting two
  greedy generations agree.

**One defect fell out of walking the path:** `Activation::getRequiredMemory` did not exist, so
`getDeploymentFootprint` threw for any configuration containing a DeltaNet layer — the path
Chat's GPU-fit verdict and `/context` use. The base throws by design so an unconverted
component cannot be missed, and Qwen's block was the first composite to recurse into
`Activation`. Fixed; nine components on the GPT-2 side still lack it, and are filed.

### The tokenizer converter (2026-08-19)

`Tools/Converters/Qwen/convert_tokenizer.py` plus `BpeVocabulary::loadQwen`. Qwen 3.8 is
GPT-2 style byte-level BPE — explicit merge ranks, no byte fallback, 248,077 pieces
(248,044 learned plus 33 control tokens) — so it takes Mila's merge-by-rank path, not the
max-munch path the Llama loader uses. The file layout is the one Gemma's converter defined,
unchanged. EOS is `<|im_end|>` and there is no BOS and no UNK.

**`transformers` is not a reliable tokenizer reference for this checkpoint.** Version 5.12.1's
`Qwen2Tokenizer` rebuilds the backend from `vocab.json` + `merges.txt` and overwrites the
checkpoint's pre-tokenizer with a hardcoded Qwen2-era pattern (`tokenization_qwen2.py:33`)
omitting `\p{M}`. The checkpoint's pattern admits combining marks into the letter run; the
stale one does not, so on Devanagari, Thai and Arabic the two produce different id sequences —
measured at 7 tokens against 3 on a five-syllable Thai greeting. The checkpoint is right and
the proof is in its own vocabulary: it contains base+mark pieces the stale pattern can never
emit. Both decode back to the input, so the only symptom is worse output. The converter
therefore reads `tokenizer.json` directly and pins the pattern against the constant the
runtime carries (`QWEN3_PRETOKENIZATION_PATTERN`), failing the run on a mismatch — which is
how this was found, on the guard's first execution.

Two limits of the current state, neither blocking:

- **The NFC normalizer is dropped.** Mila has no normalization stage, so text that is not
  already composed may encode differently than in HuggingFace. ASCII is unaffected.
- **The Unicode pattern never compiles under MSVC**, so the ASCII fallback is always what runs
  (the pre-existing `\p{...}` gap, not a Qwen one). Under that fallback Qwen's pattern and
  Llama's produce *identical* pretokens on ASCII, because the digit rule — the only remaining
  difference — is inert here: this vocabulary holds no multi-digit piece and no digit-digit
  merge rule, so the merge loop cannot join digits however they were grouped. **No ASCII test
  can distinguish the two families**, which is worth knowing before writing one that claims to.

### The layer-streamed HF reference (2026-08-19)

`Tools/Converters/Qwen/qwen38_BF16/hf_qwen_layer_stream.py`. The reference half of Phase 4's
parity harness, and it works on the real checkpoint: **64 layers at BF16 on the 12 GiB card,
holding one decoder layer resident at a time.** On "The capital of France is" the last-position
argmax is 11751 = ` Paris` at 17.50 against 14.69 for second place, so the driver is not merely
running — it reproduces the model.

It emits a **MILA `.bin`** (67 tensors, 2.22 MB): the last-token hidden state after every layer,
after the final norm, and the last-position logits. That is the format `PretrainedModelReader`
already reads, so the Mila side can assert against numbers rather than against printed digits.
A `--max-layers 4` run pairs with the converter's 4-layer fixture.

Layers are built on the meta device and installed with `load_state_dict(assign=True)`, so the
default initialization never allocates and the checkpoint tensors become the parameters
directly. Weights come from safetensors mmap; the embedding table is read one row per prompt
token rather than materialized.

**`--self-test` is the reason to trust it, and it earned its place immediately.** It builds a
small random model, runs it both whole (`Qwen3_5TextModel`, `output_hidden_states=True`) and
streamed, and requires bitwise agreement — plus a negative control that drops the causal mask
and must diverge. Three findings came out of the first two runs:

- **`attention_mask=None` is not causal.** `eager_attention_forward` adds nothing when the mask
  is None, so a full-attention layer attends bidirectionally and returns a plausible hidden
  state. The driver therefore calls the model's own `create_causal_mask` rather than
  hand-rolling one, and passes `None` only to the linear-attention layers — which need a 2-D
  padding mask, not a 4-D causal one.
- **`output_hidden_states` does not expose the last layer's raw output.** Its final entry is the
  post-norm state, so the last layer can only be checked through the norm.
- **A causal-mask error in the LAST layer is invisible at the last token**, because the final
  row of a causal mask masks nothing. It becomes observable only once a position-mixing layer
  runs after it. The self-test uses eight layers rather than four for exactly this reason, and
  the same blindness applies to layer 63 of the real stack: last-position comparison does not
  exercise the final attention layer's masking.

**Still owed: the Mila half.** A standalone `QwenAttentionBlock` cannot run prefill or decode —
it needs the shared GQA workspace `QwenTransformer` owns — and `Component::build()` is `final`
with no release hook, so there is no per-layer build/free cycle today. The streamed Mila run
needs one of: a streaming mode inside `QwenTransformer`, or a harness that allocates the shared
workspace itself and constructs one block at a time. This is the open decision.

**Correction, same day: both claims in the paragraph above are wrong.** Kept rather than deleted
so they are not re-derived. `Component::build()` being `final` with no release hook does not
prevent a per-layer cycle — Python does not release a layer either, it *destroys* one, and a
fresh block per layer is built exactly once in its lifetime and freed by its destructor. And a
standalone `QwenAttentionBlock` *can* run prefill and decode: `Qwen.Block.Cuda.cpp`'s header says
the transformer **owns** the shared GQA workspace, which is not the same as saying no one else
may supply one. Every step the Python driver takes has a public counterpart:

| Python | Mila |
|---|---|
| construct the layer on the meta device | construct the block — construction allocates nothing |
| (not applicable) | `installSharedWorkspace()` on the attention block; `setState()`, which the DeltaNet block ignores |
| `load_state_dict( assign=True )` | `PretrainedModelReader::readTensorBlob<MR>( name )` + `Component::loadParameter()` |
| `layer( hidden, ... )` | `prefill()` / `decode()` |
| `del layer` | drop the `shared_ptr` |

`readTensorBlob` is random access by name, so a per-layer load costs one read per tensor rather
than 64 passes over the 50 GiB artifact.

What remains is a cost, not a blocker: the sizing lives in four private members of
`QwenTransformer` — `computeWorkspaceWidths()`, `resolvePrefillChunkSize()`,
`allocateBlockWorkspace()` and `allocateAndWireGqaWorkspace()`. A harness that duplicates them
silently measures a different geometry than the model does, the first time one of them changes.
Lifting them into a workspace type the transformer and a harness both construct is the smaller
change, and it leaves the harness itself outside `Mila/Src`.

### Phase 4's parity gate is MET on next-token agreement (2026-08-19)

`Tests/Dnn/Models/QwenModel.Parity.Cuda.cpp` streams the 50 GiB BF16 artifact one block at a
time on the 12 GiB card and compares against the reference above, layer by layer. **On "The
capital of France is" Mila's last-position argmax is 11751 = ` Paris`, the same token the
HuggingFace reference predicts, through all 64 layers at reference precision on real checkpoint
weights.** Logit relative L2 is 2.24e-2. The run takes 43 s.

That is the Phase 4 exit gate's second clause ("matching last-position logits") and it closes
Phase 3's outstanding "oracle parity per layer, on real checkpoint weights" criterion.

#### The per-layer error profile, and what it says

| Layers | Relative L2 vs the reference |
|---|---|
| 0-2 (DeltaNet) | 5.3e-3 falling to 2.7e-3 |
| 3 (first full attention) | 1.8e-2 |
| 27 (peak) | 7.5e-2 |
| 63 (last) | 3.3e-2 |
| after final norm | 3.1e-2 |

Two properties of that curve matter more than any single number.

**It falls after layer 27 rather than compounding.** Error accumulating across 64 layers would
grow monotonically; this does not. The residual stream grows in magnitude down the stack (the
reference's own L2 goes 11.9 at layer 0 to 141.9 at the final norm), so a roughly fixed absolute
error becomes a shrinking relative one. That is the signature of rounding, not of a wrong
computation.

**Every local maximum sits on a full-attention layer** — 3, 7, 27, 43, without exception, while
the DeltaNet layers between them are flat or falling. **The cause is not yet known.**

The first suspect was wrong and the elimination is worth recording. The materializing BF16
softmax (`Gqa.Prefill.Bf16.cu:94`) stores the *unnormalized* exponentials as BF16 and narrows a
second time after scaling, where the reference narrows once — a real defect, but not this one.
Measured in Python against an FP64 softmax: the second rounding costs **2-22% more relative
error than narrowing once, ~1e-4 absolute**, at every row length from 5 to 4096 and after the AV
matmul. The step it was supposed to explain is ~2e-2, two orders of magnitude larger. BF16
rounding is relative, so storing `exp(x - max)` and later scaling by a shared `inv_sum`
perturbs each element by ~2^-9 either way and the two roundings do not compound.

**Confirmed end to end.** The four materializing BF16 softmax kernels were converted to
recompute the exponential on the store pass, and the parity run was repeated: layers 0-2 are
bit-identical (no attention runs there) and every later layer moves by well under one percent of
its error — layer 27 from 7.523e-2 to 7.565e-2, the logits from 2.238e-2 to 2.314e-2, i.e.
*marginally worse*, which at this magnitude is only which way individual values happened to
round. The argmax is unchanged. The rounding is not the cause, and the store/reload stands as a
**throughput** change whose prefill benefit is so far argued from the decode precedent rather
than measured.

#### Measured against FP32 truth: the gap is real and it is Mila's (2026-08-19)

A third arm settles it. The driver was run at `--dtype float32` over the *same BF16 weights*, so
only the arithmetic width differs, and both BF16 arms were compared against it. Every layer now
reports three numbers.

| | layer 0 (DeltaNet) | layer 3 (first attention) | layer 63 | logits |
|---|---|---|---|---|
| Mila-BF16 vs FP32 | 4.124e-3 | 1.592e-2 | 3.703e-2 | 2.666e-2 |
| HF-BF16 vs FP32 | 1.328e-3 | 4.399e-3 | 1.326e-2 | 1.387e-2 |

**Mila carries roughly 3x the BF16 error HuggingFace does**, at essentially every layer. The
hypothesis that Mila might be *more* accurate — its RoPE keeps the cos/sin cache and the rotation
in FP32 where the reference does both in BF16 — is refuted: whatever that buys is swamped.

Two structural facts fall out, and they point in different directions:

- **The gap is already present at layer 0**, a DeltaNet layer with no attention, no RoPE and an
  input identical on both sides (the embedding). 3.1x there. So its *origin* is not
  attention-specific — something common to both block kinds contributes.
- **But attention is where it compounds.** Across layer 3, Mila's error against truth quadruples
  (3.505e-3 -> 1.592e-2) while HF's rises 18% (3.720e-3 -> 4.399e-3). HF's error against truth
  plateaus near 1e-2 for the whole stack; Mila's keeps climbing to 3.7e-2.

All three arms still choose 11751 (` Paris`), so this is headroom rather than a failure — but the
headroom is smaller than the vs-HF numbers alone suggested, and a quantized body spends from it.

The harness is not the cause: the hidden state crosses the host as FP32 between layers, and
BF16 -> FP32 -> BF16 is exact, so no rounding is added there.

#### FIXED: Qwen's rotary frequencies were spread over head_dim instead of rotary_dim (2026-08-19)

**The cause, and it is a correctness defect, not precision.** `rope_build_cache_kernel`
(`Rope.Fp32.cu:44`) computed `theta_i = base^(-2i / head_dim)`. Qwen's reference
(`compute_default_rope_parameters`) uses `dim = head_dim * partial_rotary_factor` as the
denominator, i.e. `base^(-2i / rotary_dim)`. At 64 of 256 that is a factor of ~29,000 at the last
rotated pair.

A partial-rotary convention has TWO halves — which channel pairs rotate, and what frequency
spectrum they span — and the families differ on both:

| | pairs | frequency denominator |
|---|---|---|
| Gemma (`WholeHead`) | `i` with `i + head_dim/2` | `head_dim` |
| Qwen (`RotaryPrefix`) | `i` with `i + rotary_dim/2` | `rotary_dim` |

Both now hang off `RotaryLayout` on `RopeConfig`, defaulting to `WholeHead` so Gemma and Llama
are byte-identical. The layout is part of the RoPE cache key: two ops with identical geometry
but different layouts build genuinely different tables and must not share an entry.

**Result — the ~3x deficit is gone and reversed:**

| | before | after | HF-BF16 |
|---|---|---|---|
| `q_roped` (layer 3) | 4.127e-1 | **5.064e-3** | 5.113e-3 |
| `gated` | 1.174e-1 | **8.578e-3** | 9.259e-3 |
| `block_output` | 1.117e-2 | **3.604e-3** | 3.614e-3 |
| layer 63 | 3.703e-2 | **1.008e-2** | 1.326e-2 |
| logits vs FP32 | 2.666e-2 | **9.677e-3** | 1.387e-2 |

**Mila is now closer to FP32 truth than HuggingFace's own BF16 run at essentially every layer** —
its cos/sin cache and rotation are FP32 where the reference does both in BF16, an advantage the
frequency bug had been swamping. Layers 0-2 (DeltaNet, no RoPE) are unchanged to the digit, the
control that the fix touches only what it should.

**Why the first attempt made things worse, which is the lesson worth keeping.** Selecting
`RotaryPrefix` while the cache still spread the spectrum across `head_dim` left the pairing and
the frequencies on *different* conventions — worse than either consistent choice. The failed
experiment was still what excluded the "layout alone" theory and forced the search upstream; the
error was reporting a cause before the confirming run, not running it.

**How it was found, in order:** stage attribution on a common input showed `input_norm`
bit-identical and `q_roped` at 41%, bounding the fault to four stages. The projection probe then
cleared two of them -- `split_q`/`split_gate` match HF *to the digit* after the converter's
de-interleave, and `split_k`/`split_v` are marginally better -- leaving `q_norm` and RoPE.
`q_norm` uses the same `rms()` helper as the bit-identical `input_norm`, which left the rotation,
and the only part of it the layout experiment had not touched was the frequency table.

#### Superseded: bounded to four stages; the rotary-LAYOUT-only explanation was refuted

**Read this before the subsection below, which proposed a cause that turned out to be wrong.**
The measurements in it stand; the conclusion does not.

The partial-rotary layout difference between Qwen and Gemma is real — the two references
genuinely disagree, and that is documented below. **It is not the cause of this error.** A
`RotaryLayout` selector was built and Qwen switched to the prefix form; every measured number got
worse:

| | WholeHead (default) | RotaryPrefix |
|---|---|---|
| `q_roped` | 4.127e-1 | 3.575e-1 |
| `gated` | 1.174e-1 | **2.865e-1** |
| `block_output` | 1.117e-2 | **2.653e-2** |
| logits vs FP32 | 2.666e-2 | **4.239e-2** |

Reverted; the baseline reproduces exactly.

**What the failed experiment established, which the successful-looking reasoning had not:**
`q_roped` is ~36-41% wrong under *both* layouts. The rotation touches only 64 of 256 channels, so
it can account for at most ~0.7 relative error and only if the remaining 192 already agree. Had
pre-RoPE `q` been correct, the correct layout would have driven `q_roped` to ~5e-3. It did not
move. **The divergence is upstream of the rotation.**

With `input_norm` bit-identical and `q_roped` badly wrong, the fault is bounded to four stages:
`fc_qkv_proj`, the query/gate split, `q_norm`, and RoPE — and RoPE is now excluded. The next
probe is the cheapest of the three remaining: compare Mila's `q` and `gate` workspace slots
against the reference's `stage_q_proj` *before* any norm or rotation. That isolates the fused
projection and the de-interleaved split — the one transform in this path with no counterpart in
any other family, recorded as verified against the checkpoint but never against a running block.
Both slots already exist in the harness-owned workspace, so this is two more rows, not new
machinery.

**Method note worth keeping.** The per-head uniformity cited below as evidence for the rotary
theory does not discriminate: a bad projection or a bad split produces it equally. Having a
mechanism that *explains* an observation is not having measured that it *causes* it.

#### The superseded reasoning: Qwen and Gemma's partial-rotary conventions do differ

Stage attribution on layer 3, both sides fed the reference's own block input, so the two start
identical and any difference is that block's:

| stage | Mila vs FP32 | HF vs FP32 |
|---|---|---|
| `input_norm` | 2.967e-3 | 2.967e-3 (**bit-identical**) |
| `q_roped` | 4.127e-1 | 5.113e-3 |
| `k_roped` | 3.870e-1 | 5.111e-3 |
| `gated` | 1.174e-1 | 9.259e-3 |
| `o_proj` | 3.826e-2 | 5.438e-3 |
| `ffn_down` | 6.891e-2 | 1.034e-2 |
| `block_output` | 1.117e-2 | 3.614e-3 |

RMSNorm is bit-identical, so the block enters agreeing. **The rotation is where it breaks**, at
40% relative error — far too large for BF16 and uniform across heads (q: 0.41-0.53 over 24
heads; k: 0.33-0.45 over 4), which rules out a head permutation. Everything downstream is this
error decaying as the projections average it away.

**The two sides rotate different dimension pairs.** With `head_dim` 256 and `rotary_dim` 64:

- **The reference** slices the contiguous prefix and rotates *within* it:
  `q_rot = q[..., :64]`, then `rotate_half(q_rot)` pairs `i` with `i + 32`. Rotated pairs are
  `{(0,32) ... (31,63)}`; dims 64-255 are untouched.
- **Mila** pairs across the whole head — `half_dim = head_dim / 2 = 128`
  (`Rope.Bf16.cu:138`), so the kernel pairs `i` with `i + 128` and the cache zeroes the
  frequency for pairs at or beyond `rotary_dim / 2` (`Rope.Fp32.cu:52-57`). Rotated pairs are
  `{(0,128) ... (31,159)}`.

Both rotate 64 of 256 dimensions; they are not the same 64, and not the same pairing. Mila's is
the **proportional** convention the Gemma global layers were built for — the kernel comment says
so — and `QwenAttentionBlock` reuses the shared `Rope` component and inherits it.

This is a correctness defect, not a precision one, and the parity run understates it: at a
5-token prompt the rotation angles are small and the corrupted dimensions are a minority, so the
argmax survives. **It gets worse with context**, which is exactly the failure a short-prompt
parity gate cannot see, and it is the reason to fix this before any long-context or quality
measurement is trusted.

**Mila's convention is CORRECT for Gemma — checked, not assumed.** Gemma 4's reference is
`(x * cos) + (rotate_half(x) * sin)` with `rotate_half` splitting at `x.shape[-1] // 2`
(`modeling_gemma4.py:780-806`): no prefix slice, the rotation spans the whole head pairing `i`
with `i + head_dim/2`, and partial rotary lives entirely in the cos/sin cache. That is Mila's
kernel exactly. Gemma runs this path for real — its global layers carry `global_rotary_dim` 128
of `global_head_dim` 512, read from checkpoint metadata (`GemmaModel.ixx:839` ->
`Gemma.Block.ixx:184`, `:898`) — so the proportional form is validated by Gemma's token-parity
test rather than untested.

So the two families genuinely disagree, and **the fix is a per-family choice on `RopeConfig`**,
never a change to the shared kernel's default. Adding a rotary-layout selector and giving Qwen
the prefix form leaves Gemma and Llama untouched.

*(An earlier revision of this section cited `Gemma.Config.ixx:536` `getRotaryDimForLayer()` as
the evidence Gemma uses partial rotary. That function is dead library code — its only callers
are two assertions in `Gemma.Config.cpp`. The live path is `rotaryDim()` ->
`getGlobalRotaryDim()`. The conclusion was right and the citation was not.)*

The test's per-layer bound (1.0e-1) is a **recorded baseline, not a proof of correctness**; the
argmax equality is the assertion that carries the gate. Tighten the bound if the softmax
rounding is fixed.

#### What the harness cost the tree

Four private `QwenTransformer` members held the workspace sizing, so a harness would either
duplicate them or reach inside. Instead `makeQwenAttentionBlockWorkspace()`, `QwenGqaWorkspace`
and `makeQwenGqaWorkspace()` now live in `Qwen.AttentionBlock.ixx` beside the struct they fill, and the
transformer calls them — one source, no duplication. `QwenModel::configFromMetadata` moved from
private to public so the harness builds blocks from the geometry a real load would use. All 73
Qwen tests stayed green across the change.

Three constraints the harness had to work around rather than change, each in its file header:
`Component::getExecutionContext()` and `setExecutionContext()` are both protected, so
independently constructed components can neither share a stream nor expose the one they own —
the harness falls back to a device-wide synchronize; the hidden state travels between layers
through the host, because a block's output buffer dies with the block; and prefill runs in one
chunk, since chunk-boundary equivalence is already pinned by the Phase 3 block tests.

One defect the run found in the harness itself, worth the note because the symptom was so
readable: a leaf component's tensor name has no path left after its prefix (`temb.wte` -> `wte`),
and requiring a dot made the embedding load silently load nothing. Every layer then reported a
relative error of exactly 1.000e+00 — which is what a zero hidden state looks like, since the
error equals the reference norm. The loader now refuses a prefix that matched no tensor.

### Phase 3 status (2026-08-19)

**The published 27B geometry constructs, builds, and runs** — prefill and decode end to end,
finite logits, all 64 layers in the 3:1 interleave. Suite 1743 passed / 1 pre-existing skip.
Built: `CausalConv1d` (`Components/Convolutions/`), `GatedDeltaRule`
(`Components/DeltaNet/`), `QwenDeltaNetBlock`, and both arms wired into `QwenTransformer`.

**The phase is not finished** — its exit gate has three criteria and one is met outright. See
*The exit gate is NOT met* below before treating this as done.

#### The checkpoint settled four questions this document had guessed at

Enumerating cost ~140 KiB: the safetensors index, then one HTTP **range** request per shard
header (8-byte length prefix, then that much JSON). No tensor data. Worth reusing — it reads
names, dtypes and shapes for a 51.77 GiB checkpoint for the price of a web page.

- **QK-norm exists** — `q_norm`/`k_norm` `[256]`, per head, before RoPE, on every
  full-attention layer and on the MTP layer. Now wired, mirroring `Gemma.Block.ixx`.
- **The attention output gate is sigmoid, not swish.** `output_gate_type` is read nowhere.
- **Both RMSNorm conventions appear in one model.** Every stream norm uses the unit-offset
  form `x_norm * (1 + weight)` with weights stored zero-centered; the DeltaNet mixer's gated
  norm uses the **raw** weight, ones-initialized. Getting these the same way round is silent,
  not loud — the model produces plausible garbage.
- **`intermediate_size` really is uniform** at 17408 across both kinds and MTP, confirming
  Section 9's config-level answer at the checkpoint level.

**The general lesson, worth more than any of the four:** three of these are invisible in
`config.json` and visible in the tensor names or the reference source. *A config file
describes what varies between checkpoints, not what the architecture always does.* Re-reading
it more carefully would never have found them.

#### The precision plan dictated the projection split

Section 5's parameter counts only add up one way, and they are the reason the block does not
mirror the checkpoint's tensor layout:

| Role | Section 5 | Tensors |
|---|---|---|
| `DeltaNetQueryKey` | 1.007 B | q + k |
| `DeltaNetValueGateOutput` | 4.531 B | v + z + out_proj |
| `DeltaNetGating` | 0.024 B | a + b |

The checkpoint fuses q, k and v into one `in_proj_qkv` `[10240, 5120]`, but one tensor cannot
carry two storage policies. So the converter splits it into `[q|k]` at 4096 and `[v]` at 6144,
and splits `conv1d.weight` `[10240, 4]` to match. **That split is exact, not an
approximation**: the convolution is depthwise, so two convolutions over a partition of the
channels compute what one convolution over all of them computes.

#### Kernel shape: the recurrent state lives in registers

A `[head_key_dim, head_value_dim]` state is 64 KiB per head — too large for shared memory, and
streaming it through global memory each step would cost more bandwidth per layer than reading
the whole model. But **one thread per value column makes the recurrence entirely thread-local**:
decay, `kv_mem`, the outer-product update and the output projection each touch only that
column, with no cross-thread exchange. The state therefore stays in registers for the whole
chunk. Only q and k are shared (they are per key-head), so they pass through shared memory and
their L2 norms are recomputed redundantly per thread — uniform control flow, no block
reduction, no partial-warp sync.

Two shortcuts, both licensed by an oracle rather than by reading: q and k are passed at
`num_key_heads` width and the kernel indexes `key_head = value_head / group`, so no
`repeat_interleave` is materialized; and `g`/`beta` are derived inside the rule from `A_log`
and `dt_bias`, keeping a softplus off the public activation enum. The oracle is an independent
Python implementation proven **bit-identical** (max |diff| = 0.0) to
`torch_recurrent_gated_delta_rule`, and the C++ test vectors are its output — so a failure
means divergence from the reference, not from someone's reading of it.

#### Section 7's predictions, checked against working code

Five of six held exactly. The sixth resolved more cheaply than expected: `setState(const
GqaState&)` was recorded as *does not fit*, with the remedy being to make the workspace an
associated type. In practice the DeltaNet block **accepts and ignores it** — there is no
attention transient to wire — and self-allocates its own transients. Its slots share nothing
with `QwenAttentionBlockWorkspace`, so pooling them means a *second* workspace struct rather
than a wider one. That is a memory optimization, still owed, and the interface is unchanged.

The prompt-caching consequence Section 7 derived is now enforced in code:
`QwenDeltaNetBlock::rewindKvCache` always returns `false`, and the transformer ANDs it into a
stack-wide refusal.

#### The exit gate is NOT met — two of three criteria

Phase 3's gate (above) names three. Stating this plainly because the geometry building and
running makes it easy to read the phase as finished, and it is not:

| Criterion | Status |
|---|---|
| Chunked prefill and token-by-token decode produce identical state and output | **Met.** Pinned at three levels — the convolution, the mixer, and the whole block — each with a positive control that fails if the carried state is ignored. |
| Oracle parity per layer, at BF16, **on real checkpoint weights** | **Met 2026-08-19.** All 64 layers compared against the HuggingFace reference on the published checkpoint, and the last-position argmax agrees (` Paris`). See "Phase 4's parity gate is MET" above for the per-layer profile and the one open numerics question it raises. Parity against the Python oracle on synthetic vectors remains bit-identical in FP32. |
| State-plus-conv-ring snapshot/restore roundtrips exactly | **Not started.** Neither the recurrent state nor the convolution window can be snapshotted or restored today. Section 7 identifies this as the mechanism that replaces rewinding for prompt caching, so it is the gate criterion with a product consequence attached. |

#### What Phase 3 does not do

**Prefill runs the recurrence sequentially**, O(T) in sequence steps. The chunked
(UT-transform) formulation is the throughput answer and is not built; the recurrent form is
the oracle it must be validated against. **This tree is not shippable at 27B prefill.**

Two Phase 4 constraints found while enumerating, both filed:

- **The parity harness cannot use `from_pretrained`.** The BF16 reference is 54 GB against
  31.8 GB of system RAM and a 12 GiB card, so it must stream layer-by-layer off the shards via
  safetensors mmap.
- **The harness must tokenize from `tokenizer.json`, not from `AutoTokenizer`.** See the
  tokenizer-converter section above: the two disagree on mark-bearing scripts. A parity run on
  ASCII prompts is unaffected, but the moment one is not ASCII the reference is wrong rather
  than the model.
- **MTP has no HF reference at all.** `transformers 5.12.1` declares
  `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` and implements no MTP class, so that head
  cannot be gated against HF. Its wiring here is read from tensor shapes and family
  convention, not from a reference.

---

**Owed, outside the phases:** `Web/content/blog/expressing-qwen38-in-types.md` is a `draft: true`
post written before the 2026-08-16 revision of this document, so its figures are stale wherever
they touch `attn_output_gate`, the GB/GiB rows, or the 16K baseline — including the "2.78 Bits per
Weight" in its title. Reconcile it against this spec before the draft flag comes off. Tracked here
rather than in `BACKLOG.md` because this document, not the task list, is the record for this track.

### Host-resident embedding, item 6 (built 2026-08-22)

Section 5 spends 1.271 B parameters -- 2.37 GiB of BF16 -- on a table that is gathered from
and never multiplied by, and puts it in host memory. That is now a compile-time axis on the
component, `EmbeddingTableResidency::{Device, Host}`, defaulting to Device so every other
family is byte-identical. `QwenTransformer` selects Host on CUDA.

**The kernel did not change, and that is the whole mechanism.** Pinned host memory is
device-addressable under unified virtual addressing, so the existing gather
(`Y[bt] = Wte[ix]`, 128-bit loads) reads rows across PCIe with the table pointer being the
only difference. `CudaPinnedMemoryResource` already reported `DeviceType::Cuda` and
`is_device_accessible`, so `CudaTokenEmbeddingOp` accepts the tensor through the same
`ITensor*` binding with no new code at all.

Three constraints fell out, each stated as a `static_assert` or a build-time refusal rather
than left implicit:

- **Host residency and table quantization are mutually exclusive.** Both are ways of not
  spending VRAM on the table; combining them prices a quantization error against a cost that
  residency has already removed.
- **Inference only.** The backward kernel accumulates with device-scope `atomicAdd`, which is
  not defined against mapped host memory. Refused at `build()`, not at the first backward.
- **Tying is a compile error by construction**, since the shared handle's type now differs.
  Qwen is untied (`tie_word_embeddings: false`); Gemma keeps Device residency and is untouched.

**One real defect, found by the change and fixed with it.** `CudaTensorOps::copyFromBlob`
issued every load as `cudaMemcpyHostToDevice` regardless of the destination resource. For a
pinned destination both pointers are host, and a copy declaring a direction its pointers do
not have is undefined. It now dispatches on `is_host_accessible`, which is what the sibling
`copy()` has always done. This was latent for any host-accessible parameter, not only this one.

**Measured cost** (RTX 4070, real 248320 x 5120 geometry, `DISABLED_GatherCost` in
`Tests/Dnn/Components/Embeddings/TokenEmbedding.Cuda.cpp`):

| | device-resident | host-resident |
|---|---|---|
| decode, 1 token (10 KiB) | 5.99 us | 5.51 us |
| prefill, 512 rows (5 MiB) | 8.07 us | 229 us |

**Decode is free** -- both figures are launch latency, not bandwidth; 10 KiB is too little
traffic to measure either way, and the host arm being marginally faster is noise. Section 5's
"no measurable latency" claim is now measured rather than argued.

**Prefill pays 28x, at 22.8 GB/s** -- PCIe gen4 x16, as expected. It does not matter at this
scale: 0.22 ms sits against a 512-token chunk that costs order 1 s of GEMM through 64 layers of
a 27B model, so it is under 0.02% of the chunk. Recorded because it is the number that would
change the answer on a wider prefill or a smaller model, and the mitigation is known if it ever
does -- stage the chunk's distinct rows H2D once instead of gathering them in place.

Cover: four cases per precision in section H of the component test (gather equality against
both the stored bytes and the device-resident arm, the decode kernel separately, the memory
split, the training refusal), plus the real 4-layer 27B fixture, which now loads its table
through the reader into pinned memory and generates. Suite 1766 / 1 pre-existing skip.

### The Qwen packer, item 8's offline half (built 2026-08-22)

`Tools/Quantization/pack_qwen.py`. The converter pipeline above, implemented: it holds one
decoder layer resident, calibrates it, quantizes it, emits its Mila tensors, and advances
the calibration set through the quantized layer. It runs on the real checkpoint.

**Measured on the 4070**, four layers at 4 samples x 2048 tokens: **46 s and 8.22 GiB peak
per layer**, the peak flat from layer 2 onward. Extrapolating the full 64-layer run at
32 x 2048 gives **1.5-2 hours** — most of a layer's cost is fixed (shard read, codebook
fit, compensated column walk) and only the two calibration passes scale with sample count.

**8.22 GiB is more than double this document's "under 4 GiB" estimate.** The estimate
counted the layer and its Hessians and not the GPTQ transients: `gptq_quantize_tensor`
clones its weight in FP32 and holds `Q` beside it, which for the fused `fc_gate_up` pair is
713 MiB of the two together, plus the Cholesky inverse. It still fits, with under 3 GiB of
headroom on a display-attached 12 GiB card — so the run wants an otherwise idle GPU.

**Three structural things this needed that the Llama packer did not:**

- **One HF tensor becoming two Mila tensors at two policies.** `in_proj_qkv` splits into
  `fc_in_proj_qk` (cb8) and `fc_in_proj_v` (cb4). Legitimate because GPTQ's column walk
  compensates each output row independently — the update is an outer product of a per-row
  error with a row of the inverse Hessian — so quantizing a row range in isolation is
  identical to quantizing it inside the whole matrix, given the same Hessian. The two
  slices share one, because they share an input.
- **A streaming safetensors writer.** `safetensors.numpy.save_file` takes a dict, so a
  14 GB artifact would be 14 GB resident before a byte reached disk. `streaming_safetensors.py`
  declares every tensor first, writes the header, and seeks to each offset as the bytes are
  produced. It is held to `save_file` byte-for-byte in the data region.
- **The FP4-at-load tensors are damaged in place but never compensated.** They are written
  BF16 and Mila re-quantizes them data-free at load, so a compensated weight would be
  optimized for a quantizer that never sees these codes. `lm_head` is not damaged at all:
  nothing inside the model reads its output, so it would alter no other tensor's calibration.

**Two facts about the Python safetensors writer, both measured:**

- The data region is ordered by **dtype size descending, then name** — confirmed again here,
  and now depended on rather than merely recorded.
- **`__metadata__` key order is not stable between runs.** The Rust side carries the map in
  a hash table, so two `save_file` calls over the same dict produced different orders. A
  whole-file byte comparison against the library is therefore not a valid check; the writer's
  test compares the parsed header and the data region instead.

**What proves it, since an hour-long run must not be trusted on its output alone.**
`--self-test` packs a small random 8-layer Qwen end to end and checks completeness (the
artifact carries exactly the tensor set the BF16 converter emits, modulo the codebook
companions), fidelity (all 69 pass-through tensors bit-identical to the transform applied to
the live layer, which is what exercises the `q_proj` de-interleave and the DeltaNet row
splits), and damage — a negative control, since every other property holds just as well if
quantization silently did nothing. `--verify` audits a produced artifact against the
checkpoint in minutes: on the 4-layer real-weight artifact, 103 tensors complete, packed
shapes matching what `Linear` allocates per policy, 33 untouched tensors byte-identical to
the checkpoint, and the FP4 tensors carrying damage.

**One refactor rode with it, verified as a no-op.** `resolve_qwen_geometry` and
`qwen_mila_metadata` moved out of `convert_qwen` so the packer and the BF16 converter read
one geometry and declare one architecture; a metadata drift between them would load a
quantized artifact into a differently-shaped chassis. The 4-layer fixture regenerated after
the refactor is SHA-256 identical to the one on disk.

**A silent-failure guard worth reusing:** the tensor map derives which layers are full
attention arithmetically from `full_attention_interval`, while the model carries
`layer_types` built by the config, and nothing connected the two. `verify_layer_kinds`
now compares them per layer. A disagreement would have the packer look for
`linear_attn.in_proj_qkv` on a layer holding `self_attn.q_proj`.

#### The full 64-layer artifact exists (2026-08-22)

`Data/Models/Qwen/qwen38_27b_2p9bit.safetensors`, **15.09 GiB from 50.10**, in **49.7
minutes** on the 4070: 24.327 B quantized parameters at **2.819 average bits per weight**,
1603 tensors over 320 packed Mila linears.

**Peak VRAM was flat across all 64 layers** — 8.12 to 8.22 GiB, with only layer 1 lower at
7.43 — which is the one-layer-resident design holding. Nothing like the monotonic creep the
Llama gate showed before it freed its blocks per layer.

The run cost far less than the layer-count extrapolation predicted, and the reason is worth
keeping: per-layer time barely moved with calibration size (32.2 s at 512 tokens, 38.1 s at
8192, 42.7 s at 65536). The calibration forwards are launch-bound at small token counts, so
the fixed work — shard read, codebook fit, the compensated column walk — dominates a layer,
and 64x the calibration data costs about a third more time rather than eight times.

`--verify` against the checkpoint: 1603 tensors complete, packed shapes matching what
`Linear` allocates for each policy, **498 untouched tensors byte-identical to the
checkpoint**, and all 32 FP4-at-load tensors carrying damage. Ten seconds.

**Calibration was wikitext-2 alone**, not the prose-plus-code set step 2 of the pipeline
calls for — that is what `corpus/` holds and what Phase 0 used. Re-running with a
code-bearing set is the first cheap thing to try if the Phase 5 quality numbers come in soft.

**Still not shown:** a read of this artifact by `PretrainedModelReader`. The reader gate of
2026-08-18 proved a `save_file`-written container reads, and this writer is byte-identical in
layout, so it is expected rather than demonstrated — it is demonstrated when the Phase 5 load
path exists.

### Phase 5's load path, and the first real VRAM measurement (2026-08-22)

**Qwen 3.8-27B at 2.82 bits per weight loads and generates on the 12 GiB card.** That is the
claim this whole track exists to make, and it is now a passing test rather than an argument.
It is not yet the Phase 5 gate: the context it runs at is 512, not the 16K baseline.

`QwenModel` reaches the allocation through its own `dispatchQwenWeightPlan`, not the shared
`dispatchWeightQuantization`, for the reason `Qwen.PrecisionPlan.ixx` gives for its own lift:
the shared dispatcher yields a uniform policy and Section 5's allocation is a per-role plan
over roles this family invented. `WeightQuantization::Plan` is the deployment value that
selects it, and the shared dispatcher now refuses `Plan` explicitly rather than falling
through its `default` to `NoWeightQuant` — which would have built Gemma or Llama unquantized
and reported success.

The artifact and the build must agree on storage format **in both directions**, and the load
refuses either mismatch: packed codes read as BF16, or a BF16 blob decoded through a codebook,
both produce a model that loads and runs and is wrong.

#### The measured budget, against Section 5's table

| | Section 5 | measured at 16K |
|---|---|---|
| Weights | 8.65 GiB | **8.69 GiB** |
| Everything else on device | 1.94 GiB | 1.65 GiB |
| Device total | 10.59 GiB | **10.34 GiB** |

**The weights row lands within 0.5% of a budget written before any of this existed**, which
is the strongest evidence so far that the Section 5 allocation is real rather than plausible.

#### Three defects the first load found, all of them latent before Qwen

- **The FP4 quantize-on-load kernel could not quantize a vocabulary-sized tensor at all.**
  Its grid was `dim3(num_groups, out_features)` and `grid.y` is capped at 65535 on every CUDA
  architecture; `lm_head` is 248320 wide, so the launch failed outright. Shipped code, and
  Qwen is simply the first model to FP4-quantize an output axis that large — Llama's `lm_head`
  ignores the weight policy entirely and Gemma's is tied and goes through the FP8 table path.
  The output channel now indexes `grid.x`. The FP8 kernels use a 1-D grid and were never
  affected, checked rather than assumed.
- **Quantize-on-load staged the whole tensor through the grow-only shared scratch**, so
  `lm_head` asked for 2.54 GiB that is never given back. Now staged in row blocks under the
  same 256 MiB ceiling `CudaTokenEmbeddingOp` already applies to its FP8 table, for the same
  recorded reason.
- **Qwen's prefill activation budget was Gemma's, and does not fit this model.** Measured
  across the ladder; the table and the reasoning are in `Qwen.ixx`. The counter-intuitive part
  is that a *generous* budget overruns at SHORT context: with little KV to pay for, the ladder
  takes the 1024-row rung, and 1024 rows of a 27B geometry is what does not fit. 512 MiB fits
  the whole ladder to 16K; 32K remains out of reach, which is what Section 5 already says.

#### What bounds the context: ~92 MB per layer that nothing predicts

**Corrected 2026-08-22, same day.** This section first called the gap "~0.9 GiB" and attributed
it to the CUDA context and allocator rounding. That was an artefact of the instrument:
`cudaMemGetInfo` sees only **dedicated** VRAM, so it reported "10.85 GiB consumed, 0 free" for
a process that had in fact committed 11.21 GB dedicated **plus 8.53 GB shared**. Windows
per-process counters (`\GPU Process Memory(pid_N*)\Dedicated Usage` and `\Shared Usage`) show
the real figure, and on Windows they are the ones a fit decision must use.

Measured predicted-against-actual at 512 context:

| | predicted | actual device-intended | excess |
|---|---|---|---|
| 4-layer packed fixture | 1.61 GiB | 2.33 GiB | 0.72 GiB |
| full 64-layer artifact | 9.94 GiB | ~16.0 GiB | ~6.1 GiB |

Two points, one line: **~92 MB per layer plus ~0.36 GiB fixed**. At 64 layers that is 5.7 GiB
of the 6.1, so the per-layer term is the whole story.

**92 MB is the size of one layer's own transients at chunk 512** — `fc_gate_up`'s output is
[512 x 34816] BF16 = 35.7 MB, SwiGLU's is 17.8 MB, the DeltaNet projections are 4-6 MB each.
Every layer holds its own set rather than sharing a pooled buffer, and `getRequiredMemory` does
not count them.

**So the fit blocker is not the Section 5 allocation, and not a residual.** The weights land at
8.69 GiB exactly as budgeted; it is the transients around them that oversubscribe the card, and
oversubscription is what makes WDDM page the weights and cost 5x on decode. One defect explains
the 512-context cap and the decode rate together.

#### Localized: the prediction promises an installation the build never performs

Not a component under-reporting. `QwenTransformer::getRequiredMemory` (`Qwen.ixx:381`) hands
every block a context carrying `.withInstalledOutput( context.isInferenceMode() )`, whose
documented meaning is *"the parent installs the child's output buffer before calling build(), so
the child skips self-allocating its output… Only prediction reads this."*

Gemma sets that flag truthfully — its `onBuilding` really does `allocateBlockWorkspace` and
`installSharedWorkspace`, which is why Gemma's Gate B can assert predicted == reported exactly.
**Qwen copied the prediction line and not the installation.** `withInstalledOutput` appears
nowhere else in `Qwen.ixx`.

The asymmetry is visible in the built tree, per component at 512 context:

| layer | kind | device state |
|---|---|---|
| 0, 1, 2 | DeltaNet | **138.2 MiB each** |
| 3 | full attention | ~0 per component (plus the shared RoPE cache) |

The attention block installs its children's outputs into `QwenAttentionBlockWorkspace`, so its
components hold nothing and the flag's claim is true. The DeltaNet block does not, so every one
of its components self-allocates: six separate [512 x 5120] stream buffers at 5 MiB each
(`input_norm`, `post_attn_norm`, `res_1`, `res_2`, `fc_down`, `fc_out_proj`), `fc_gate_up` at 34
MiB, SwiGLU at 17. At 48 DeltaNet layers that is **~6.5 GiB**, which is the measured excess.

**Both fixes are needed and they are not alternatives.** Installing the DeltaNet outputs is the
memory win — a second workspace struct, since those slots share nothing with the attention one.
Adding Gemma's Gate B equality assert to Qwen is what stops a 60% under-prediction passing
again; Chat's GPU FIT verdict reads that number, so a wrong prediction is worse than none.

BACKLOG had the pooling half filed as "a memory optimization, sized by the prefill chunk". It is
not an optimization — it is the release gate, and it explains the context cap and the decode
paging as one defect.

#### The packed model is coherent (2026-08-22)

Read, not inferred. Every other assertion in the load tests checks that generated ids are
inside the vocabulary, which catches NaN logits and index errors and nothing else -- a
2.82-bit model emitting plausible garbage passes all of them.

Greedy, temperature 0, through the converted Qwen tokenizer:

> **The capital of France is** " Paris.\nThe capital of Germany is Berlin.\nThe capital of
> Italy is Rome.\nThe capital of Spain is"

> **A gardener explains why compost matters:** " "Compost is a soil amendment that improves
> soil structure, increases nutrient availability, and supports beneficial microbial activity.
> It is not a fertilizer, but it does improve soil fertility over time.""

**The first token is 11751 = " Paris", the same token Phase 4 measured the HF reference and
Mila-at-BF16 both choosing** through all 64 layers on real weights. That is the assertion the
test carries; it ties the 2.82-bit build directly to a number already of record rather than to
a reading. Three further capitals are correct without being asked for, and the second
completion is accurate on a point it would be easy to get wrong -- compost is a soil amendment
rather than a fertilizer.

The second sample then restates itself under a new speaker. Greedy decoding does this at BF16
too, so it should not be attributed to the bit width without a controlled comparison; it is
noted because Phase 0 recorded "grammatical, repetition loops" as the failure mode at these
widths before compensation, and this is far milder than that.

**What this does NOT establish** is the Phase 5 quality clause, which is a perplexity ratio
and a divergence point against the FP4 oracle, not a reading. It establishes that the thing
being measured is a working model.

#### Decode is 4.7 tok/s, and four fifths of that is WDDM paging (2026-08-22)

> **Corrected the same day.** This section first recorded 4.7 tok/s as refuting the 47 tok/s
> ceiling. It does not: the benchmark ran on a model that fills the card exactly, so most of
> what it measured was the driver paging Mila's own weights to host memory. A VRAM-resident
> measurement puts decode near **24 tok/s** — still under the ceiling, but by 2x rather than
> 10x. The attribution below is what separated the two, and is kept because the method is
> reusable; the original conclusion is not.


Measured on the 4070 against the packed 27B, by subtracting an 8-token generation from a
72-token one so the load and the prefill cancel exactly
(`DISABLED_DecodeRate` in `QwenModel.Load.Cuda.cpp`):

| | |
|---|---|
| Decode | **213.45 ms/token, 4.7 tok/s** |
| Implied weight bandwidth | **44 GB/s** |
| Card peak | 504 GB/s |
| Section 5 ceiling | 47 tok/s |

The path is confirmed correct, not a fallback: `outer_size == 1` reaches `launchCodebookDecode`
and the dedicated codebook GEMV.

**Per-kernel attribution** (nsys, 88 decode tokens). This answers the question of which kernel
spends the time, and the answer is not the one this document expected:

| kernel | share of decode | ms/token |
|---|---|---|
| cb4 codebook GEMV | 45% | 87 |
| cb8 codebook GEMV | 27% | 51 |
| FP4 GEMV | 23% | 45 |
| **Gated DeltaNet recurrence** | **2.6%** | 5 |

**The DeltaNet recurrent kernel is not a bottleneck at all**, despite running on 48 of the 64
layers and being the one kernel here with no precedent in the tree. The three GEMVs are 95% of
decode. That closes the suspicion this section previously carried.

#### The variance was the finding, not the mean

| kernel | median | mean | max |
|---|---|---|---|
| cb4 GEMV | 130 us | 434 us | 3.6 ms |
| FP4 GEMV | **84 us** | 1411 us | **30.6 ms** |

A 365x spread between median and max is not a compute characteristic; a compute-bound kernel
has tight variance. It is the signature of residency thrashing — some invocations read VRAM,
some read host memory across PCIe. The instrumentation had already printed the cause without
its significance being noticed: **"load consumed 10.8457 GiB device, 0 GiB still free"**. The
model fills the card exactly, leaving the desktop compositor nothing, so WDDM evicts Mila's
own weights to make room and pages them back per kernel.

**The control that settles it.** The 4-layer packed fixture runs the same kernels on the same
policies at ~1.2 GiB resident, with 8.52 GiB still free — nothing can be evicted:

| | per layer | 64-layer equivalent |
|---|---|---|
| 4-layer fixture, resident | 0.85 ms | ~54 ms/token, 18.4 tok/s |
| full model, card full | ~3.3 ms | 213 ms/token, 4.7 tok/s |

The extrapolation overstates, because dividing by layer count charges each layer a share of
`lm_head` and then multiplies it back sixteenfold; backing that out puts a resident model near
**24 tok/s**, which is independently where the nsys medians land (~41 ms/token). Two methods,
one answer: **paging costs a factor of about five, and the kernels cost a factor of about two
against the ceiling.**

**What this means for Section 5's ceiling.** It is optimistic by roughly 2x, not refuted. The
codebook GEMVs being instruction-bound is real and is what the remaining 2x is; the L2-resident
Phase 1 figures did overstate DRAM, but by far less than the contaminated measurement implied.
"Quality-limited, not speed-limited" survives at 24 tok/s in a way it does not at 4.7.

**Method note worth keeping.** A decode benchmark on a model sized to fill the card measures
the driver's pager, not the kernels. This is the same class of error as benchmarking an
L2-resident matrix and reading it as DRAM bandwidth, in the opposite direction: one flatters,
one penalises, and both are fixed by stating the residency of the thing being measured before
believing the number.

#### Fixed: the DeltaNet block now pools, and the promise is enforced by a test

`QwenDeltaNetBlockWorkspace` is the second workspace struct `QwenAttentionBlockWorkspace`'s
comment anticipated — not a wider version of it, because a DeltaNet layer's slots are shaped
by the mixer's own geometry (fused `[query|key]`, the value stream, two per-head gating
scalars) and share no width with the attention block's beyond the residual stream. It carries
**twenty component output slots plus the q/k split scratch**, allocated once by
`QwenTransformer` and viewed by every DeltaNet layer, on the same sequential-execution
argument the attention workspace makes.

Three aliasing constraints decide the slot count, and each is a slot that could not be
shared: `normed` is read by all five input projections, `z` survives from its projection to
the output gate several stages later, and `res1` is read again at `res_2`. The block's own
input is the previous layer's `stream` slot and is last read at `res_1`, mid-block, before
`res_2` overwrites it — the same argument the attention workspace records.

One component had to gain the capability: **`Activation` had no `installSharedOutput`**, so
the two SiLU stages after the convolutions could not pool. It is now the ninth component to
carry the pattern, mirroring `Swiglu` exactly. `Activation` is a child of no other block in
the tree, so nothing else changes behaviour.

Three things are new in the row-cost model and the prediction, and they must move together
or Gate A separates them: the workspace is counted by `QwenTransformer::getMemoryStats`,
predicted by `deltaNetWorkspaceBytes`, and charged per chunk row by
`computeChunkRowCostBytes` — the last is what stops the rung ladder from choosing a chunk it
cannot afford. All three are gated on `getNumDeltaNetLayers() > 0`, so an all-attention
configuration allocates and predicts nothing.

**What now enforces it.** The gap survived because Gate A only ever ran on an all-attention
configuration, where one block kind and one workspace made the promise true by accident.
`GetRequiredMemory_MatchesBuiltFootprint_HybridInterleave` runs the same equality on the
published 3:1 interleave, and three block-level cases pin the contract where it lives:
pooled output equals self-allocated output value for value (the aliasing test), the installed
slot set is counted by the installer and nobody else, and prediction equals build in **both**
positions of `withInstalledOutput`. The general rule is now recorded in
`MemoryFootprint.md` section 4.5: **a Gate A case is owed per block kind, not per model.**

#### Measured on the 27B: decode 4.7 -> 33.7 tok/s, and the spill is gone (2026-08-22)

Same artifact, same card, same `DISABLED_DecodeRate` subtraction. Two runs agreeing to 0.1%:

| | before | after |
|---|---|---|
| Decode | 213.45 ms/token, **4.7 tok/s** | 29.7 ms/token, **33.7 tok/s** |
| Implied weight bandwidth | 44 GB/s | **314 GB/s** (4070 peak 504) |
| Against Section 5's 47 tok/s ceiling | 10x off | **1.4x off** |

The per-process counters say why, and they are the instrument that matters here because
`cudaMemGetInfo` cannot see WDDM's shared allocation:

| | before | after |
|---|---|---|
| Dedicated | 11.21 GB | 10.58 GB |
| Shared | 8.53 GB | **2.94 GB** |
| of which the pinned embedding, by design | 2.54 GB | 2.54 GB |
| **Unintended spill** | **~5.99 GB** | **~0.40 GB** |

**This retires the "kernels are 2x off the ceiling" reading as well.** That estimate came
from the 4-layer resident control extrapolated to 64 layers, which gives 18.9 tok/s — and
the extrapolation *understates*, because it charges each of four layers a share of `lm_head`
and then multiplies it back sixteenfold. The real resident model is 33.7, so what remains
between the kernels and the ceiling is 1.4x, not 2x. The three GEMVs being instruction-bound
is still the right target; it is a smaller prize than it looked.

The footprint ladder moves with it — `DISABLED_FootprintAcrossContexts` against 10.85 GiB
free:

| context | 512 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|
| device GiB | 9.42 | 9.41 | 9.61 | 9.74 | 10.27 | 11.43 |
| chunk rows | 512 | 256 | 256 | 64 | 64 | 64 |

**16K now fits where the model previously capped at 512.** 32K still does not, which is what
Section 5 already says: the stretch needs the FP8 KV policy, not a bigger activation budget.

One caution on reading that table against the pre-fix one. The prediction was never what
oversubscribed the card — the *allocation* was — so the ladder is not a before/after of the
same quantity. The prediction fell 0.52 GiB at 512 for a reason worth stating, because it is
the whole shape of the defect in one line: the three things the false pooled flag did NOT
cover were counted per layer and are now counted once (the two `Activation` outputs at
0.47 GiB and the q/k split scratch at 0.19 GiB), against 0.14 GiB added for the shared
workspace. Everything else the prediction omitted entirely, and the card paid for it anyway.

---

## 9. Open Questions

- **`head_dim` 256 with 4 KV heads on the full-attention layers.** Gemma's global layers run
  512, so the width is precedented, but the shared-memory budget of the FlashAttention prefill
  kernel needs checking against this specific geometry.
- ~~**`intermediate_size` uniformity**~~ — **closed 2026-08-19.** Resolved at the config level
  in 2026-08-16 and now confirmed against every shard header: 17408 on all 64 layers, both
  kinds, and the MTP layer.
- **Does the DeltaNet chunk size interact with the prefill chunk?** *Half answered
  2026-08-19.* The convolution side is settled and implemented: it carries `kernel_width - 1`
  rows across chunk boundaries, and chunked prefill provably reproduces a single pass. The
  delta rule is currently recurrent, so it has no chunk of its own; the question becomes live
  again — and unresolved — when the UT-transform kernel is built.
- **mRoPE sections [11, 11, 10].** Text-only input gives all three sections the same position
  index, so mRoPE degenerates to standard RoPE regardless of the interleaved layout — the
  single-section fallback is exact, not an approximation. `rotary_dim` already exists on
  `RopeConfig` and covers the 0.25 partial factor; the multi-section layout is a multimodal
  concern and is deferred with the vision tower. *Corroborated 2026-08-19: the sections sum to
  32 = rotary_dim / 2, and `partial_rotary_factor` is honoured by the shared rope init
  (`modeling_rope_utils.py:174`), so the 64-of-256 width is confirmed rather than inferred.*
- **Spend the FP8-KV margin on context or on bits?** At the 16K baseline, landing the FP8 KV
  policy frees ~0.5 GiB — enough for +0.25 bits across the whole FFN or +0.5 bits on its
  most sensitive rows (Section 5). Quality (Section 8 step 0) should decide, not memory.
- **Is the GPTQ compensation slightly mismatched on the prefill path?** Compensation is
  computed offline against exact `codebook[code] * scale` values. Decode reproduces those;
  a tensor-core prefill reproduces them only to BF16 (Phase 1 status). So the correction
  is fitted to weights one of the two paths does not quite multiply. At 2^-9 the residual
  should be negligible against the quantization error it corrects, but that is an
  expectation, not a measurement, and the converter's column walk is where it can be
  checked cheaply.
- **Does the quality survive?** The earliest possible measurement is Section 8 step 0, before
  any CUDA exists; final answer only when it runs. The plan sits at IQ2_XXS class, and the
  capabilities this model is valued for — agentic coding, long-horizon tool use — are the
  ones that degrade first and that a perplexity check will not catch.
