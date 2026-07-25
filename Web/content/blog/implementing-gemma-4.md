---
title: "Implementing Gemma 4 in C++: Sliding-Window Attention, Dual RoPE and Token-for-Token Parity"
date: 2026-06-23
description: "Porting a C++ inference runtime from Llama 3 to Gemma 4: per-layer local/global sliding-window attention, dual RoPE, GeGLU, logit softcap, and the RMSNorm bug that cost token parity."
discussion: "https://github.com/ToddThomson/Mila/discussions/15"
# Renamed from bringing-up-gemma-4 (2026-07-25): "bringing up" reads as vomiting, not board
# bring-up. Nothing is known to link here yet, but the alias costs one redirect stub and covers
# any link already sitting in Discussion #15.
aliases: ["/blog/bringing-up-gemma-4/"]
---

## Where We Left Off

Alpha.2 ended with Mila producing token-for-token matches against HuggingFace for Llama 3.2 1B, and the alphas pushed that to Llama 3.1 8B with FP8 and FP4 weight quantization. The framework had a working Grouped-Query Attention op, a two-phase prefill/decode KV cache, and a clean `OperationTraits` dispatch path. The Llama family was, by then, a *homogeneous* machine: every layer looked like every other layer.

Gemma 4 broke that assumption on contact.

## The Architecture Delta

Gemma 4 12B "Unified" is not a re-skinned Llama. The differences are structural, not cosmetic:

| | Llama 3.1 / 3.2 | Gemma 4 12B Unified |
|---|---|---|
| Layer list | homogeneous | **heterogeneous, 5 sliding : 1 global** |
| Attention | GQA, full causal | sliding-window (1024) local + full global |
| `head_dim` | `hidden / heads` | **decoupled**: 256 local / 512 global |
| KV heads | 8 | 8 local / **1 global (MQA), K=V** |
| RoPE | single base | **dual base** (10k local / 1e6 global) + partial-rotary global |
| Norms | 2x RMSNorm | **sandwich, 4x RMSNorm**, RAW weights |
| Q/K/V norm | none | **per-head q_norm / k_norm / v_norm** |
| FFN | SwiGLU (SiLU) | **GeGLU (GELU)** |
| Per-layer scale | none | **`layer_scalar`** (learned, full-stream) |
| Logits | plain | softcap `30*tanh(z/30)` |
| Embedding | plain | `x sqrt(d)` scale |

Any one of these would be a feature. Together they argued for a new chassis.

## A New Chassis, Not a Bent Llama

The first real decision was *not* to retrofit `LlamaTransformer`. Gemma got its own `Transformers/Gemma` tree, built on a few orthogonal compile-time axes — a `kGlobal` flag selecting per-layer geometry, decoupled `head_dim`, a runtime sliding window — so that a "local" block and a "global" block are two instantiations of the same `GemmaBlock<..., kGlobal>` template that genuinely differ in head width, KV-head count, RoPE base, and attention span.

The heterogeneity needed a home in the type system. Every prior model was a flat list of identical blocks; Gemma interleaves two kinds. We introduced a virtual `IDecoderLayer` interface (one `prefill`/`decode` call per layer) so the transformer can hold a mixed list and drive it polymorphically — one indirect call per layer per token, negligible against the per-layer GEMMs.

## The Parity Hunt

Then came the part that always takes longer than the estimate. The model loaded, ran, and produced *fluent garbage*: coherent-looking tokens, wrong answer, and a residual stream that quietly detonated.

### Red Herring #1: The (1+w) Trap

Gemma 3's RMSNorm applies `x_norm * (1 + weight)`. We trusted that. It was wrong. With `(1+w)`, the QK-norm scaled Q by ~2x and K by ~9x, the scores blew up ~18x, and the softmax went one-hot — sharp where HuggingFace's attention was soft. The fix was counterintuitive: Gemma 4's norm is **raw** (`x_norm * weight`). *Lesson: do not trust the previous generation's modeling code.*

### Red Herring #2: The Hooks That Lied

To localize the blow-up we hung forward hooks on every submodule and compared magnitudes. We chased an "attention permutation" bug for about fifteen turns. The bug did not exist. The **hooks** were the liars — norm-module forward hooks over-capture, and a per-layer hook read a pre-scale tensor that disagreed with the real residual by 50x. The only trustworthy references turned out to be `output_hidden_states`, `output_attentions`, and a hand-written fp32 oracle. *Lesson: pick your ground truth before you debug, not during.*

### The Real Fixes

With reliable references, three genuine deltas surfaced from the loaded-model source:

1. **`layer_scalar`** — Gemma 4 multiplies *each decoder layer's entire output* by a learned scalar (e.g. L0 ~= 0.053). Without it the residual grew unbounded (88 -> 14,357). This was the dominant 18x bug, and it lives at the very end of the layer: `hidden *= layer_scalar`.
2. **`v_norm`** — Gemma normalizes V per head (a pure normalize, no learnable weight), which neither Llama nor our first oracle did.
3. **Global `V = v_norm(k_proj)`** — the global K=V layers have no `v_proj`. V is the *raw* key projection passed through `v_norm` — no `k_norm`, no RoPE — while K is `RoPE(k_norm(k_proj))`. Same projection, two different normalizations.

A subtle trap nearly cost another session: our fp32 oracle was *stale*. It still used `(1+w)`, skipped `layer_scalar`, and read raw V. It faithfully reproduced the *wrong* answer (res2 ~= 1640) and kept pointing us away from the fix. An oracle is only as honest as its last update.

### The Last Bug: Zero

After all of that, layer 0's residual was still 20% low, and the stream that should grow to 304 flatlined near 40. The sub-step dump found it instantly:

```
attn            l2 = 0.0000
o_proj          l2 = 0.0000
post_attn_norm  l2 = 0.0000
res1(+attn)     l2 = 64.08   <- == the input, exactly
```

The attention output was **exactly zero** — across all 48 layers. Not small. Zero. Mila's `RmsNorm` only initializes its weight to 1.0 when building *fresh*; on a pretrained load it leaves the weight at its zero-allocated default and fills it from the checkpoint. The checkpoint had been converted *before* `v_norm.weight` was added to the converter — so `v_norm` was multiplying V by a tensor of zeros. `normalize(V) * 0 = 0`, and an all-zero V annihilates attention.

The fix was a re-convert. The lesson is filed as a real follow-up: a norm that silently zeroes on a missing weight is a footgun, and the loader should *error* on an expected parameter it never sees.

## The Moment

Re-converted. Re-run. Layer 0's `attn` came up at ~50, `res1` jumped from 64 to 333, and the residual climbed 88 -> 220 -> 304 in lockstep with HuggingFace, layer for layer, to the last decimal that BF16 allows.

Then the chat harness, decoding one token at a time through the same path:

> **Hello! I am Mila. How can I help you today?**

Token-for-token parity, prefill **and** decode. Gemma 4 12B runs on Mila.

## What Alpha.6 Built

- A dedicated `Transformers/Gemma` chassis with compile-time per-layer geometry.
- A heterogeneous decoder via `IDecoderLayer` (sliding + global blocks in one list).
- Decoupled `head_dim`, dual-base + partial RoPE, sliding-window attention, K=V/MQA global layers.
- GeGLU FFN, the 4-norm sandwich with raw weights, per-head Q/K/V norm.
- `layer_scalar`, `sqrt(d)` embedding scale, and final logit softcap.
- A HuggingFace -> Mila converter and a layer-by-layer parity harness.

## What's Next

Gemma 4 Unified emits a *channel-structured* response (a reasoning channel and a final channel), and the chat app currently prints those control tokens raw — so the next slice is a channel-aware renderer that separates "thought" from "final." After that: a bounded KV ring cache to make the sliding window pay for itself in VRAM, and the 26B MoE sibling that shares this exact chassis.

The Llama work taught Mila to match a model. The Gemma work taught it to match a *family*.

