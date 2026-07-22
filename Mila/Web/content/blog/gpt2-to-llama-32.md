---
title: "From GPT-2 to Llama 3.2 in C++: RMSNorm, SwiGLU, Grouped Query Attention and RoPE"
date: 2026-04-04
description: "What it actually takes to build a modern transformer from scratch in C++23 and CUDA - swapping LayerNorm for RMSNorm, GELU for SwiGLU, MHA for grouped query attention, and learned positions for RoPE."
discussion: "https://github.com/ToddThomson/Mila/discussions/8"
---

*A show-and-tell of what it actually takes to build a modern transformer from scratch in C++23 and CUDA.*

---

## Where We Left Off

Alpha.1 ended cleanly. GPT-2 inference validated token-for-token against HuggingFace using greedy decoding. The full stack — BPE tokenizer, embeddings, multi-head attention, MLP, KV cache, two-phase prefill/decode — all confirmed correct. The methodology was established: if Mila's greedy output matches HuggingFace's greedy output exactly, the implementation is correct.

Alpha.2 began with a deceptively simple goal: do the same thing for Llama 3.2 1B.

It turned out to be considerably more interesting than that.

---

## The Architecture Delta

On paper, Llama is GPT-2 with a few changes:

| Concern | GPT-2 | Llama |
|---|---|---|
| Positional encoding | Learned (fused into embeddings) | RoPE applied to Q, K in attention |
| Normalization | LayerNorm | RMSNorm |
| MLP activation | GELU | SwiGLU |
| Attention | Multi-Head (MHA) | Grouped Query (GQA) |
| Tokenizer | BPE | SentencePiece |

In practice, each of those differences is a separate implementation and validation campaign. New components built from scratch: `TokenEmbedding`, `RMSNorm`, `RoPE`, `SwiGLU`, `GroupedQueryAttention`, `LlamaBlock`, `LlamaTransformer`, `LlamaModel`. New Python scripts for weight conversion. SentencePiece integrated via CPM.

The hardest of these, by a significant margin, was GQA.

---

## The GQA Campaign

Grouped Query Attention is the attention mechanism used in modern Llama models. Instead of one key/value head per query head, a smaller number of KV heads are shared across groups of query heads. For Llama 3.2 1B: 32 query heads, 8 KV heads.

This sounds like a minor change. It is not. The implementation required:

- A fused QKV projection with asymmetric head counts
- A `split()` primitive to produce separate, contiguous Q, K, V tensors
- A `prefill_permute_q` and `prefill_permute_kv` kernel pair to write into the KV cache with the correct layout
- A `prefill_expand_kv` kernel to broadcast from 8 KV heads to 32 query heads before the matmul
- Chunked prefill (`kPrefillChunkSize=64`) to fit the 12GB VRAM budget for 1B FP32 weights
- Separate cuBLASLt plans for QK and AV matmuls, with carefully computed batch strides
- A `prefill_softmax` kernel with causal masking over the chunk window
- An `unpermute_output_padded` kernel to pack the output back

The GQA prefill debugging session fixed bugs in all of those layers: out-of-bounds softmax writes, incorrect `expand_kv` source strides, `permute_qkv` using fixed rather than dynamic shapes, wrong strides in `unpermute_output_padded`, cuBLASLt batch stride errors in both matmul plans. The debugging methodology was Nsight memory inspection — breaking inside kernels mid-execution to read the active GPU thread's registers.

After all of that, the prefill pipeline was validated. The model's greedy prediction for the prompt *"Once upon a time"* matched HuggingFace exactly: **`,`** — a comma. Not exciting, but numerically identical. That's the gold standard.

---

## The `Tensor::view()` Bug

Before GQA could work correctly, a more fundamental bug surfaced in `Tensor::view()`.

The original implementation computed strides from the view's shape alone, ignoring the parent tensor's row stride. When a view was taken of a slice of a larger allocation — as happens when splitting QKV — the resulting Q and K views had non-contiguous strides. RoPE, operating on those views, was writing rotated values to the wrong memory locations entirely.

The fix was architectural: introduce a `TensorOps::split` primitive that writes Q, K, V into separately allocated contiguous buffers, replacing the view-based slicing. A view is now a read-only window into parent memory. A split is a copy into fresh memory. Clean separation, no silent stride bugs.

---

## Prefill to Decode: "Famous Last Words"

With prefill validated, decode seemed close. The `LlamaModel::generate()` loop was already written. The estimate was 1-3 sessions.

The first run of the decode loop produced:

```
Mila: , the Lord, the sun, the day by the United States of the United States of 
the United States of the United States of the United Statesboro, the United States 
that will be it was a, and . . . . . . . . . . . . . . . . . .
```

So. Not quite there.

The bugs came in layers:

**Bug 1 — `encoder_out_ptr_` was null.** A GPT-2 artifact left over from the copy-paste starting point. `LlamaTransformer::decode()` was trying to feed a null pointer into block 0 instead of the token embedding output. Fixed immediately.

**Bug 2 — EOS token was wrong.** `eos_token_` was set to `2` (the Llama 2 `</s>` token). Llama 3's `<|end_of_text|>` is token `128001`. Without this fix, generation barrelled through every EOS it produced and kept going. The comment in the code even had the right value. It just wasn't assigned. Fixed.

After those two fixes, the decode loop ran and produced output that started coherently then rapidly degenerated into:

```
Mila: , the first time I was a little girl, and I was a little girl, and I was a 
little girl, and I was a little girl, [... 400 more tokens of this ...]
I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I,
```

This is a very specific failure signature. The model starts coherent, then locks into a repetition attractor. Classic KV cache staleness — each decode step was seeing the same context rather than the growing sequence.

**Bug 3 — `expand_kv` vs `prefill_expand_kv`.** The decode path was calling the old `expand_kv` kernel, a GPT-era function that expanded with different semantics than the validated `prefill_expand_kv`. Swapping it in with `position_offset=0` was the right fix in principle, but the degeneration persisted — which led to the real issue.

**Bug 4 — The RoPE convention mismatch.** This was the one that mattered.

The validated prefill RoPE kernel (`rope_rotate_kernel`) used the **split-half** convention: treat the head as two halves, rotating pair `(x[i], x[i + half_dim])` for each frequency index `i`. This is the standard Llama RoPE convention.

The decode RoPE kernel (`rope_decode_kernel`) used the **interleaved** convention: rotate adjacent pairs `(x[2i], x[2i+1])`. Different layout, different rotation, wrong result.

The bug was confirmed by comparing post-RoPE K values against HuggingFace's `apply_rotary_pos_emb` output (captured via a monkey-patched Python hook). Pre-RoPE K matched HF exactly. Post-RoPE K did not. The fix was rewriting `rope_decode_kernel` to match the split-half layout of the prefill kernel.

---

## The Moment

After all of that, the HuggingFace comparison:

```
HF greedy decode: "Once upon a time, there was a man who was very rich..."
```

Mila:

```
You: Once upon a time
Mila: , the world was a place of peace and harmony. But then, something happened. 
Something terrible. Something that changed everything. Now, the world is a place of 
fear and uncertainty. The only thing that can bring peace back is the power of the 
people. The people must come together and fight for what is right.
```

Matching HuggingFace token-for-token up to `max_new_tokens=64`. ✅

The repetition after that point ("The people must be relentless. The people must be relentless...") is not a Mila bug. It is a 1B parameter model running greedy decode past its comfort zone. The model, not the framework.

---

## With Sampling Enabled

Remove the greedy debug overrides, set `temperature=0.8`, and:

```
You: Once upon a time
Mila: , when a young girl named Mairi was left to run the family's croft after their 
father has suffered a stroke, she's a bit of a maverick, but nothing could prepare her 
for the challenges she faced on her first day as a teacher in the tiny village of the 
same name...
```

Four hundred words of coherent, creative fiction. Characters, plot, conflict, a book 
review at the end. Terminated cleanly on `<|end_of_text|>`.

And for completeness:

```
You: In a single word, one plus one is
Mila: two. The world may be divided into those who think it is one plus one is two 
and those who do not. The two groups are at war with each other. This is the most 
difficult question in mathematics...

I. To be able to understand it, you must first believe in it.
II. To believe in it, you must first learn to accept it.
...
XIII. To believe in losing your faith, you must first believe in losing your faith.
```

The model got `two` correct, then spent three hundred tokens constructing a recursive 
philosophical proof that eventually collapsed into self-reference. This is not a Mila bug either.

---

## What Alpha.2 Built

| Component | Notes |
|---|---|
| `TokenEmbedding` | Pure vocabulary lookup, no positional encoding |
| `RMSNorm` | With `rms_norm_eps` from config |
| `RoPE` | Split-half convention, chunked prefill + decode paths |
| `SwiGLU` | Fused gate/up projection with SiLU activation |
| `GroupedQueryAttention` | 32Q/8KV, chunked prefill, KV cache, cuBLASLt |
| `LlamaBlock` | Pre-RMSNorm, GQA, SwiGLU MLP, dual residual |
| `LlamaTransformer` | 16-block decoder stack |
| `LlamaModel` | `fromPretrained()` + `generate()` with sampling |
| `TensorOps::split` | Contiguous QKV splitting primitive |
| SentencePiece tokenizer | Via CPM, validated against HuggingFace |
| Weight converter | `convert_llama_weights.py`, handles GQA layout |
| Validation scripts | Per-component and full-network HF equivalency |

---

## What's Next

Alpha.2 closes into Beta. The remaining work is cleanup: removing debug instrumentation, gating diagnostics behind `MILA_DEBUG`, filling in CPU reference implementations for the new components, and completing test coverage.

After Beta: BF16 and FP8 precision variants to fit larger models into a 12GB consumer GPU. The dispatch infrastructure is already in place. Llama 3.2 3B in BF16 fits comfortably. Llama 3.1 8B in FP8 is the stretch target.

The framework is working. The model is talking. On to Beta.

---

*Mila is open source under the MIT licence. Contributions welcome — see [[CONTRIBUTING.md](https://github.com/ToddThomson/Mila/blob/master/CONTRIBUTING.md)](CONTRIBUTING.md).*
