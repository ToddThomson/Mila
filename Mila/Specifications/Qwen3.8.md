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
heterogeneous layer list rides `IDecoderLayer` unmodified (Section 7). The full-attention
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

| Field | Value | Note |
|---|---|---|
| `num_hidden_layers` | 64 | interleaved **3 linear : 1 full**, `full_attention_interval: 4` |
| `hidden_size` | 5120 | residual stream |
| `intermediate_size` | 17408 | SwiGLU FFN, uniform across both layer kinds |
| `vocab_size` | 248320 | large; embedding and head are 4.7% of parameters each |
| `tie_word_embeddings` | false | **untied** — two full-size tables, unlike Gemma |
| `num_attention_heads` | 24 | full-attention layers only |
| `num_key_value_heads` | 4 | full-attention layers, GQA group 6 |
| `attn_output_gate` | true | **q projection is double-width**: [query \| gate], swish gate on the attention output (Section 2) |
| `head_dim` | 256 | **decoupled** (5120 / 24 = 213 != 256) |
| `partial_rotary_factor` | 0.25 | rotary width 64 of 256 |
| `rope_theta` | 1e7 | |
| `rope_parameters` | mrope, sections [11, 11, 10] | interleaved; multimodal positional layout |
| `linear_num_key_heads` | 16 | Gated DeltaNet, head_dim 128 |
| `linear_num_value_heads` | 48 | Gated DeltaNet, head_dim 128 |
| `linear_conv_kernel_dim` | 4 | short causal depthwise convolution over q/k/v |
| `output_gate_type` | swish | on the DeltaNet value path |
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

1. **Prefill must compute logits for the last position only.** At 248,320 vocabulary a single
   FP32 logit row is 0.95 MiB; a 512-token chunk that materialized all of them would allocate
   0.48 GiB.
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

## 7. `IDecoderLayer` Fit (verified 2026-08-16)

The heterogeneous layer list needs no new mechanism. `IDecoderLayer` was introduced for
Gemma's two block types and takes DeltaNet with **five of six methods unmodified**.

| Method | Fit | Note |
|---|---|---|
| `prefill(input, position_offset)` | Fits | Chunked-parallel delta rule. `position_offset` is unused — linear layers carry no RoPE. Cross-chunk state is member data, exactly as the KV cache is. |
| `decode(input, position)` | Fits | The O(1) recurrent update. `position` unused. |
| `resetKVCache()` | Fits | Zeroes the recurrent state and the convolution ring. The transformer loops it over every layer unconditionally, so nothing gates it away. Misnamed for this layer, semantically exact. |
| `rewindKvCache(position)` | Fits | Returns `false`. The interface already anticipated refusal: the transformer ANDs results across layers and documents all-or-nothing, because a bounded sliding-window ring can already refuse. |
| `supportsKVCache()` | Harmless | Conflates "holds a per-token cache" with "holds resettable sequence state". Every call site in the tree is a `toString()` diagnostic or a test, so there is no behavioural consequence. |
| `setState(const GqaState&)` | **Does not fit** | A concrete GQA workspace — seven `ITensor*`, `q_permute`/`preatt`/`att`/`v_out` plus decode variants — in an otherwise generic interface. A DeltaNet block uses none of them and needs entirely different scratch: chunk-parallel delta-rule buffers, convolution staging, gate buffers. |

The single failure is a concrete type where a generic one belonged, not a structural mismatch.
The fix is to make the workspace an associated type of the layer rather than a fixed struct in
the interface. Deliberately deferred until the DeltaNet workspace shape is known from working
code — designing the generalization before there is a second instance to generalize over would
be guessing.

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
4. **Depthwise causal `Conv1d`**, kernel 4. No convolution component exists in the tree.
5. **Gated DeltaNet component**: L2 normalization on q/k, swish output gate, chunked-parallel
   prefill kernel and O(1) recurrent decode kernel, FP32 state. Build against a Python
   reference oracle. Validate **generation, not per-layer tolerance** — a per-layer test can
   pass while 48 recurrent layers compound to garbage, and that failure mode is more likely
   here than in an attention stack.
6. **Host-resident `embed_tokens`** path.
7. **`PerChannelKvFp8` KV-cache policy.** Optional for the 16K baseline (BF16 KV fits) and
   the price of the 32K stretch (Section 5). Independently useful for Gemma and Llama.
8. **Qwen block types, model, config, converter.** The checkpoint carries the MTP tensors
   (~0.45 B); the converter skips them.
9. **Corpus perplexity through Mila's inference path.** Teacher-forced summed
   log-likelihood over a fixed held-out corpus. Nothing in the tree does this —
   `quality_gate.py` is Python fake-quantization and Bard's perplexity is a training loss —
   and without it the Phase 5 gate is a judgement call. Perplexity needs a logit at every
   position, which is exactly what the Section 5 prefill constraint forbids materializing at
   once, so the evaluation accumulates log-likelihood chunk by chunk. Independently useful
   for Gemma and Llama.

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
> two block types and stores both as `IDecoderLayer*`. Do **not** copy Gemma's `bool kGlobal`
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
`Qwen.PrecisionPlan.ixx` (the Section 5 table as types), `Qwen.Config.ixx`, `Qwen.Block.ixx`
and `Qwen.ixx`; the output gate is a component of its own under
`Components/Attention/OutputGate/`. Suite green at 1681 passed / 1 pre-existing skip.

Five decisions the code now carries, worth stating because none is recoverable from a diff:

- **`QwenTransformer` refuses the published interleave at construction**, naming Phase 3.
  Only an all-full-attention configuration builds today. A transformer that quietly built 16
  of 64 layers is the worst failure available here, and the refusal is checked before any
  device memory is touched so it names the config rather than a build.
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
- **QK-norm is absent, on this document's authority.** Section 1's table is recorded as
  re-verified against the raw `config.json` and names `attn_output_gate` explicitly while
  naming no QK-norm field, so the block wires none. The Qwen 3 family does carry it upstream;
  if the published config does too, this is a two-component addition to `createGraph` and a
  per-head view in the forward path, and Section 2's count does not disambiguate (the norm
  weights would land in the "norms, conv, misc" row either way).

`QwenModel` is deliberately absent. Its whole job is loading weights, and with no converter
(item 8) and no DeltaNet layers it would be an entry point to a model that cannot load —
which gets mistaken for one that works.

**Owed, outside the phases:** `Web/content/blog/expressing-qwen38-in-types.md` is a `draft: true`
post written before the 2026-08-16 revision of this document, so its figures are stale wherever
they touch `attn_output_gate`, the GB/GiB rows, or the 16K baseline — including the "2.78 Bits per
Weight" in its title. Reconcile it against this spec before the draft flag comes off. Tracked here
rather than in `BACKLOG.md` because this document, not the task list, is the record for this track.

---

## 9. Open Questions

- **`head_dim` 256 with 4 KV heads on the full-attention layers.** Gemma's global layers run
  512, so the width is precedented, but the shared-memory budget of the FlashAttention prefill
  kernel needs checking against this specific geometry.
- **`intermediate_size` uniformity — resolved at the config level** (2026-08-16): the config
  carries a single scalar, so both layer kinds share 17408 as published. The checkpoint-shape
  check at converter time remains as cheap insurance (Section 2).
- **Does the DeltaNet chunk size interact with the prefill chunk?** The convolution spans
  chunk boundaries and needs a 3-token carry; whether the delta-rule chunk should match the
  prefill chunk or be independent is unresolved.
- **mRoPE sections [11, 11, 10].** Text-only input gives all three sections the same position
  index, so mRoPE degenerates to standard RoPE regardless of the interleaved layout — the
  single-section fallback is exact, not an approximation. `rotary_dim` already exists on
  `RopeConfig` and covers the 0.25 partial factor; the multi-section layout is a multimodal
  concern and is deferred with the vision tower.
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
