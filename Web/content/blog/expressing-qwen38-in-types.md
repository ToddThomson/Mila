---
title: "Expressing a Model Mila Cannot Yet Run: Qwen3.8-27B, Gated DeltaNet and 2.78 Bits per Weight"
date: 2026-08-16
description: "A 27B hybrid-attention model shipped after Mila's decoder chassis was written. Before writing a kernel, we asked what it would take to express it in the type system - and found five of six interface methods fit unchanged."
draft: true
# TODO(todd): open the Discussion thread and fill this in before publishing.
# discussion: "https://github.com/ToddThomson/Mila/discussions/NN"
---

Mila cannot run Qwen3.8-27B. It has no Gated DeltaNet, no sub-4-bit weight storage, and no
converter for the checkpoint. This post is not about running it.

It is about a narrower question that turns out to be more interesting: **a model was released
on 14 August 2026, after Mila's decoder chassis was written. Can that chassis express it?**

That question is worth asking because the usual demonstration is rigged. Mila's heterogeneous
layer machinery exists because Gemma 4 interleaves sliding-window and global attention layers,
and it was written while we were bringing Gemma up. Showing that it handles Gemma proves
approximately nothing — the abstraction and its one instance grew up together. A sceptical
reader is right to discount it.

Qwen3.8 is not rigged. It has a sequence mixer Mila has never run, a different layer ratio, a
262K context, and a memory budget that no quantization format currently in the tree can
satisfy. If the existing abstractions take it, that means something. If they don't, the place
they break is worth knowing.

We spent a session on the design before touching a kernel. Here is what fell out.

## The architecture delta

Gemma 4 is the right reference point, because it is the most structurally unusual model Mila
already runs.

| | Gemma 4 12B | Qwen3.8-27B |
|---|---|---|
| Layer list | heterogeneous, 5 sliding : 1 global | heterogeneous, **3 linear : 1 full** |
| Layers | 48 | 64 |
| Sequence mixer | attention everywhere | **Gated DeltaNet on 48 of 64 layers** |
| Attention | sliding-window local + full global | full attention on 16 layers only |
| `head_dim` | 256 local / 512 global | 256, `partial_rotary_factor` 0.25 |
| KV heads | 8 local / 1 global | 4 |
| Hidden / FFN | 3840 / 15360 | 5120 / 17408 |
| Vocabulary | 262144, **tied** | 248320, **untied** |
| RoPE | dual base | mrope, sections [11, 11, 10], theta 1e7 |
| Context | 128K | **262144**, extensible to 1M |

The one that matters is the third row. Forty-eight of the sixty-four layers do not compute
attention at all. They run a gated delta rule — a linear-attention recurrence that carries a
fixed-size state matrix forward token by token, instead of attending over a cache of every
token it has seen.

This has a consequence that decides whether the model is viable on a 12 GiB card at all.

## Only sixteen layers hold a KV cache

A KV cache grows with context. A recurrent state does not.

```
Full attention (16 layers):
  per token = 16 * 2 (K,V) * 4 kv_heads * 256 head_dim = 32,768 elements
            = 64 KiB at BF16, 32 KiB at FP8

Gated DeltaNet (48 layers):
  per layer = 48 value_heads * 128 (d_k) * 128 (d_v) = 786,432 elements
  total     = 48 * 786,432 * 4 bytes (FP32) = 0.141 GiB, at ANY context length
```

At 32K context the attention cache is 1.0 GiB at FP8 and the recurrent state is 0.141 GiB. At
262K context the recurrent state is still 0.141 GiB. A dense 64-layer model of this width
would want four times the KV.

The state has to be FP32. It is a recurrence, and BF16 accumulation across a quarter of a
million steps will drift. That is 0.141 GiB well spent.

None of this is a Mila optimization. It is a property of the architecture, and it is the
reason the rest of this post is possible.

## There is no four-bit path

Mila's smallest weight format today is `PerGroupFp4<128>` — FP4 E2M1, packed two nibbles to a
byte, with one FP32 scale per group of 128 input channels. It is what runs Gemma 4 12B and
Llama 3.1 8B on a 12 GiB card.

It does not fit here, and it does not come close.

Deriving the parameter budget from the published config gives roughly **26.4 B** text
parameters, split 65% into the feed-forward network:

| Group | Parameters | Share |
|---|---|---|
| FFN gate + up | 11.41 B | 43.2% |
| FFN down | 5.70 B | 21.6% |
| DeltaNet value, gate, output | 4.53 B | 17.2% |
| `embed_tokens` | 1.27 B | 4.8% |
| `lm_head` | 1.27 B | 4.8% |
| Full attention q,k,v,o | 1.17 B | 4.4% |
| DeltaNet q, k | 1.01 B | 3.8% |

*(Derived from config, not read from tensor shapes. It reconciles with the published BF16 GGUF
of 53.8 GB to within 2%, which is reassuring but not the same as checking.)*

`embed_tokens` is a gather, never a multiply, so it can live in host memory and cost 10 KiB per
token over PCIe. That leaves 25.1 B parameters that must be resident. Against a weight budget
of about 8.6 GiB:

| Scheme | Bits per weight | Resident weights |
|---|---|---|
| `PerGroupFp4<128>`, FP32 scales | 4.25 | 13.34 GiB |
| `PerGroupFp4<128>`, FP16 scales | 4.125 | 12.95 GiB |
| Required to fit | **2.74** | 8.55 GiB |

Uniform four-bit misses by about 50%. The published GGUF quantizations agree: the smallest one
that fits a 12 GiB card is around 9 GB at roughly 2.6 bits per weight, and it carries an
explicit quality warning.

So the FFN — two thirds of the model — has to sit between two and three bits. Only about 14% of
the parameters can afford four. *That is the finding that shapes everything else, and we did
not expect it going in.*

One thing makes it much cheaper than it sounds. The RTX 4070 is SM 8.9, and Ada has no FP4
hardware. Every sub-eight-bit format already dequantizes to BF16 during the GEMM tile load, so
INT2 and INT3 cost nothing extra against the FP4 path we already run. On this card the bit
width is purely a storage decision — which means the interesting question is not "can we
afford two bits" but "where should the bits go."

## Where the bits go

Ranked by sensitivity per byte. One ordering decision is non-obvious and load-bearing.

| Group | Policy | Bits/w | GiB |
|---|---|---|---|
| `embed_tokens` | BF16, **host-resident** | — | 0.000 |
| FFN gate + up | `PerGroupInt2<32>` asymmetric | 2.5625 | 3.403 |
| FFN down | `PerGroupInt3<64>` | 3.25 | 2.158 |
| DeltaNet value, gate, output | `PerGroupInt2<32>` asymmetric | 2.5625 | 1.352 |
| DeltaNet q, k | `PerGroupInt3<64>` | 3.25 | 0.381 |
| DeltaNet beta, decay | **BF16 — never quantized** | 16 | 0.045 |
| Full attention q,k,v,o | `PerGroupFp4<128>` | 4.125 | 0.564 |
| `lm_head` | `PerGroupFp4<128>` | 4.125 | 0.610 |
| Norms | FP32 | 32 | 0.037 |
| **Total resident** | | **2.78 avg** | **8.55** |

The load-bearing decision is that **the DeltaNet projections outrank the FFN**. Quantization
error in an attention layer is recomputed from the cache at every step and does not accumulate.
Error in a recurrence enters the state matrix and stays there for the rest of the sequence.
Those two things are not the same kind of error, and a scheme that ranks matrices purely by
size will get this wrong.

The sharpest case is `beta` and `decay` — the projections that drive the forget gate. They are
0.1% of the parameters. An error in a decay rate compounds *exponentially* over the sequence
rather than linearly, so they stay at BF16 and it costs 0.045 GiB to protect them. That is the
kind of decision that is obvious once stated and invisible if you are allocating bits by
tensor size.

Budgeting the rest, in the GiB the device actually reports:

| | 16K context | 32K context | 64K context |
|---|---|---|---|
| Weights | 8.55 | 8.55 | 8.55 |
| Attention KV (FP8, 16 layers) | 0.50 | 1.00 | 2.00 |
| DeltaNet state (FP32) | 0.14 | 0.14 | 0.14 |
| Activations, scratch | 0.45 | 0.45 | 0.45 |
| CUDA context, cuBLASLt | 0.35 | 0.35 | 0.35 |
| **Total** | **9.99** | **10.49** | **11.49** |

Against 12 GiB with roughly 11.0 to 11.3 usable on a display-attached card, 32K fits and 64K
does not.

## The part this post is actually about

All of the above is arithmetic. The question was whether the *type system* takes it.

Mila's decoder blocks implement `IDecoderLayer`, a six-method virtual interface introduced for
Gemma because a transformer with two structurally different block types cannot hold a
homogeneous `vector<Block>`. It is the one place in the model path where Mila deliberately
chose runtime polymorphism over compile-time dispatch — one indirect call per layer per token,
negligible against the per-layer GEMMs.

A Gated DeltaNet layer is a stranger thing than anything that interface was written for. It has
no KV cache, no positional encoding, and a decode step that is a rank-one state update rather
than an attention computation. We expected it to need a new interface.

It needs five of six methods unchanged.

| Method | Fit |
|---|---|
| `prefill(input, position_offset)` | Chunked-parallel delta rule. `position_offset` is unused — linear layers carry no RoPE. State crosses chunk boundaries as member data, exactly as the KV cache does. |
| `decode(input, position)` | The O(1) recurrent update. This is what linear attention *is*. |
| `resetKVCache()` | Zeroes the recurrent state and the convolution ring. Misnamed for this layer, semantically exact. |
| `rewindKvCache(position)` | Returns `false`, and the interface already handles that. |
| `supportsKVCache()` | Semantically wrong for this layer, but every call site in the tree is a diagnostic string or a test. No behavioural consequence. |
| `setState(const GqaState&)` | **Does not fit.** |

`rewindKvCache` was the pleasant surprise. It returns `bool`, the transformer ANDs the result
across every layer and documents the rewind as all-or-nothing, and the doc comment already
contemplates a bounded sliding-window ring refusing when its stale tail has been overwritten.
A DeltaNet layer refuses too — unconditionally, for a different reason. The interface
anticipated a case it was never written for.

The one failure is `setState( const GqaState& )`. `GqaState` is a bag of seven `ITensor*` —
`q_permute`, `preatt`, `att`, `v_out`, and their decode variants — the shared attention
workspace that the transformer owns and hands to each layer. A DeltaNet block wants none of
them. It needs chunk-parallel delta-rule buffers, convolution staging, and gate buffers.

So the leak is not structural. It is a **concrete type sitting where an associated type
belonged**: a generic interface that names one implementation's workspace. The fix is small and
obvious in hindsight, and we have deliberately not written it yet, because designing the
generalization before there is a second instance to generalize over is guessing. When the
DeltaNet workspace exists and we can see both shapes, the right abstraction will be a
five-minute change. Today it would be a bet.

*Lesson, and it is one we keep relearning: an abstraction with one implementation is a
hypothesis, not a design.*

## The consequence nobody asked for

Prompt caching does not work here, and not for a reason anyone can fix.

Mila reuses a cached prompt prefix by rewinding each layer's KV cache to a position and
continuing. That works because a KV cache stores per-token entries; rewinding moves a fill
pointer and discards nothing that is still needed.

A recurrent state cannot rewind. `S_t` is a lossy accumulation of every token up to `t`, and
`S_k` for `k < t` is not recoverable from it. With 48 of 64 layers refusing, the all-or-nothing
rewind always fails, and every prefix reuse degrades to a full prefill.

But the state is *constant size* — 0.141 GiB for the entire model, at any context length. So
you can snapshot it at a prefix boundary and restore it later. Prompt caching by **snapshot and
restore** works exactly where rewind does not, and the thing that makes it affordable is the
same property that made the model fit on the card in the first place.

That is a different mechanism than the one Mila implements, and it is a better answer than
"linear attention breaks prompt caching."

## What this does not prove

It does not prove the model runs. Gated DeltaNet needs a chunked-parallel prefill kernel and an
O(1) recurrent decode kernel, neither of which exists. The INT2 and INT3 storage formats do not
exist. The converter does not exist.

It does not prove the quality survives. A 2.78-bit average is in the territory the ecosystem
labels as lossy, and the capabilities this model is valued for — agentic coding, long-horizon
tool use — are exactly the ones that degrade first and that a perplexity check will not catch.
We will publish that number when we have it, including if it is bad.

And the parameter budget is derived, not measured. If `intermediate_size` turns out not to be
uniform across both layer kinds — some hybrid designs narrow the FFN on linear layers — the
whole allocation shifts.

What it does show is narrower and, we think, more useful: **a model that shipped after the
chassis was written fits it, and the one place it does not fit is a concrete type where a
generic one belonged.** The types were the easy part. The kernels are months.

The work is behind `-DMILA_ENABLE_EXPERIMENTAL=ON`, off by default, and is not part of any
release. The design is written up in full in `Specifications/Qwen3.8.md`.

Next: sub-four-bit weight storage, measured against Llama 3.2 3B where we have a known-good
oracle to diff against — because the first thing to establish is not whether two bits fits, but
what two bits costs.
