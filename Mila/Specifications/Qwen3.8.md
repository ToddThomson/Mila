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
2. **The FFN must use the fused tile-load dequantization path, never the two-phase staging
   buffer.** Dequantizing one 5120x17408 matrix to BF16 is 170 MiB.

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
2. **W2A16 and W3A16 GEMM and decode GEMV kernels**, following the existing W4A16 LUT
   tile-load pattern.
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

### Phasing

The list above is inventory; this is the order it lands, with the gate each phase must pass
before the next is worth starting. The C++ phases land under `Mila/Src/Experimental/` —
outside the v0.20 surface, promoted into the main tree only if the experiment earns it. Two
tracks are independent until Phase 4 joins them: the **storage track** (Phases 0-2) touches
no Qwen code, and the **mixer track** (Phase 3) needs no quantization. Nothing here is
scheduled.

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

**Phase 2 — precision plan struct** (item 3). Proven on an existing block, not a Qwen one:
instantiate a Llama or Gemma block from a mixed plan. *Exit:* per-role policies dispatch; a
plan missing a role is a compile error at the block; the declaration-site alias reads —
which is the property the experiment exists to demonstrate.

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
12 GiB card at the 16K baseline. *Exit:* coherent generation through headless chat, measured
VRAM within the Section 5 table, decode rate measured against the 46 tok/s ceiling. The
Section 9 quality question gets its first real answer here, and only here.

**Phase 6 — stretch** (item 7 plus margin). FP8 KV cache; spend the freed margin per
Section 9 — bits or 32K; snapshot/restore prompt caching, recorded in `PromptCaching.md`.

### Phase 0 first results (2026-08-16)

Harness: `Mila/Tools/Experimental/Qwen38/quality_gate.py` (fake-quantization on Llama 3.2 3B
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

### Phase 1 status (2026-08-16)

Built, green, and proven against each other, in `Mila/Src/Experimental/Quantization/` with
tests under `Mila/Tests/Experimental/Quantization/` and tooling in
`Mila/Tools/Experimental/Qwen38/`:

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

Remaining for the Phase 1 exit gate (end-to-end on the new policies **in Mila**): an
experimental op/Linear path consuming the packed buffers via the decode GEMV (prefill may
start on the two-phase dequant staging pattern at 3B scale; the fused W2/W3 prefill GEMM of
item 2 replaces it before Qwen), the serialization record plus `loadParameter()` upload
path, and a perf sanity of the GEMV against the qfp4 kernel.

**Owed, outside the phases:** `Web/content/blog/expressing-qwen38-in-types.md` is a `draft: true`
post written before the 2026-08-16 revision of this document, so its figures are stale wherever
they touch `attn_output_gate`, the GB/GiB rows, or the 16K baseline — including the "2.78 Bits per
Weight" in its title. Reconcile it against this spec before the draft flag comes off. Tracked here
rather than in `BACKLOG.md`, which is `dev`'s task list and carries nothing about this branch.

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
- **Does the quality survive?** The earliest possible measurement is Section 8 step 0, before
  any CUDA exists; final answer only when it runs. The plan sits at IQ2_XXS class, and the
  capabilities this model is valued for — agentic coding, long-horizon tool use — are the
  ones that degrade first and that a perplexity check will not catch.
