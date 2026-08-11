---
title: "The Eight-Headed Bug That Hid Behind a System Prompt"
date: 2026-04-24
description: "Adding system prompt support to the chat CLI turned working prompts into fluent, confident nonsense. The cause was eight heads and an offset."
discussion: "https://github.com/ToddThomson/Mila/discussions/9"
---

When I added system prompt support to Mila's chat CLI, something unexpected happened. Prompts that had previously produced perfect responses started generating fluent, coherent, and completely context-oblivious text. Ask for a story about rabbits living on the moon eating green carrots — get a charming tale about Hungarian meadow rabbits instead. The model wasn't broken. It was just ignoring everything I said.

This is the story of how a single innocent constant — `kPrefillChunkSize` — quietly corrupted eight different stages of the GQA prefill pipeline, and why it took a system prompt to wake the beast.

---

## The Setup

Mila's prefill pipeline processes long prompts in chunks of 64 tokens (`kPrefillChunkSize`). For each chunk, the GQA attention op writes K/V entries into the cache, expands them for grouped query attention, computes `Q @ K^T` via cuBLASLt, applies a causal softmax, computes `att @ V`, and unpermutes the output. All pre-allocated buffers are sized to `kPrefillChunkSize` rows.

The key insight — which should have been obvious but wasn't — is that `kPrefillChunkSize` serves two distinct roles:

1. The **buffer allocation size** — how many rows were reserved at build time
2. The **actual row count** — how many rows the current chunk is using

For full chunks these are identical. For partial chunks — the final chunk when `T_prompt` is not a multiple of 64 — they diverge. And every place in the pipeline that confused one for the other became a bug.

---

## The Trigger

Without a system prompt, typical user messages fit comfortably within 64 tokens — a single full chunk. All bugs were dormant. The pipeline had never actually executed a partial chunk followed by meaningful downstream processing.

Adding a system prompt pushed `T_prompt` past 64, activating the partial chunk path for the first time. Eight bugs woke up simultaneously.

---

## The Bugs

Every bug was a variation of the same mistake — `kPrefillChunkSize` used as a stride where `chunk_len` was required:

| Stage | Bug | Fix |
|---|---|---|
| `build_qk_prefill_plan` | `strideC = kPrefillChunkSize * T` | `strideC = chunk_len * T` |
| `build_att_value_prefill_plan` | `strideA = kPrefillChunkSize * T` | `strideA = chunk_len * T` |
| `build_att_value_prefill_plan` | `strideC = kPrefillChunkSize * HS` | `strideC = chunk_len * HS` |
| `prefill_softmax` (BF16) | `row_offset` used `chunk_stride` | `row_offset` uses `chunk_len` |
| `prefill_softmax` (FP32) | `row_offset` used `chunk_stride` | `row_offset` uses `chunk_len` |
| `CudaGqaOp::prefill` | Q pointer not offset by `position_offset * HS` | Added `position_offset * HS` to Q pointer |
| `kvcache_expand_kv` | Expanded current chunk rows only | Expands full `[0..position_offset + chunk_len)` history |
| `prefill_unpermute_output_padded` | `padded_T = kPrefillChunkSize` | `padded_T = chunk_len` |

cuBLASLt wrote `preatt_` with 64-row head strides regardless of actual chunk size. The softmax then read the wrong row of `preatt_` for every head beyond head 0. The model was computing attention over garbage for every layer, every head, every chunk beyond the first.

---

## The Symptom

The corrupted attention heads didn't produce noise. They produced fluent, grammatically correct, contextually irrelevant text. The model was responding from its language prior — *"what statistically follows a system prompt"* — rather than from the actual prompt content.

- Ask about rabbits on the moon → get meadow rabbits
- Ask about the weather in Paris → get a recitation of the system prompt persona
- Ask a question with tool calls active → get chain-of-thought reasoning about how to use a tool

This made the bug particularly deceptive. A model producing garbage output is obviously broken. A model producing polished, confident, wrong answers requires careful prompting to diagnose. The tell was specificity: the model correctly responded to the *shape* of the prompt (a story request, a question) but ignored every specific detail (the moon, green carrots, Paris).

---

## The Diagnosis

The debugging session worked through the pipeline systematically, clearing each stage in turn:

- ✅ KV cache write — correct
- ✅ KV cache expand — correct (incremental, persistent, right index math)
- ✅ RoPE absolute positioning — correct (`abs_pos = position_offset + t`)
- ✅ Block residual stream — correct
- ✅ Chunk loop hidden state — correct (prefill is stateless across chunks by design)
- ❌ `prefill_softmax` row offset — **first fix** (`chunk_stride` → `chunk_len`)
- ❌ cuBLASLt partial plan strides — **remaining seven fixes**

The softmax fix was necessary but not sufficient. After it was applied, the model produced better output on small partial chunks (71 tokens: one full + seven-token partial worked correctly) but still failed on larger inputs with multiple preceding full chunks. The debug print:

```
[GQA::prefill_softmax] B=1 NH=24 T_stride=4096 chunk_stride=64 chunk_len=11 position_offset=64
```

confirmed the softmax parameters were correct — which redirected suspicion to the cuBLASLt plans themselves.

The smoking gun was `strideC` in `build_qk_prefill_plan`:

```cpp
// Before — bug
const long long strideC = static_cast<long long>(prefill_window_size) * max_seq_length;

// After — fix
const long long strideC = static_cast<long long>(chunk_rows) * max_seq_length;
```

cuBLASLt was writing `preatt_` with 64-row head strides. The softmax (now correctly striding by `chunk_len`) was reading with 11-row head strides. Every head beyond head 0 was reading from a completely different location in the buffer.

---

## The Fix

Eight coordinated one-line fixes, all of the form: replace `kPrefillChunkSize` with `chunk_len` in stride and offset calculations.

```cpp
// fix(gqa): correct partial-chunk prefill pipeline for multi-chunk prompts
//
// Eight coordinated bugs caused all heads beyond head 0 to read garbage
// data during partial-chunk prefill (T_prompt > kPrefillChunkSize with
// a non-zero remainder). All bugs shared the same root cause: every
// stage used kPrefillChunkSize as the row stride instead of chunk_len.
```

A comment has been added to `kPrefillChunkSize` at its declaration:

```cpp
// Buffer allocation size only — never use as a stride in matmul plans
// or kernel index arithmetic. Always use the actual chunk_len for the
// current iteration.
```

---

## Lessons

**Single-chunk testing is insufficient for chunked pipelines.** A test suite that never exercises the partial chunk path cannot catch partial chunk bugs — no matter how thorough it is otherwise. Mila's validation suite now includes explicit multi-chunk prefill cases with non-multiples of `kPrefillChunkSize`.

**The most dangerous bugs produce fluent wrong answers.** Numerical corruption that manifests as coherent text is harder to catch than corruption that manifests as garbage. When a model ignores specific details of a prompt while responding correctly to its general shape, suspect attention score corruption before anything else.

**Buffer allocation size and active row count are different things.** In a chunked pipeline with pre-allocated buffers, the temptation to use the allocation constant as a stride is always present and always wrong on partial chunks. Name them differently, document the distinction at the declaration site, and never let them meet in index arithmetic.

---

*Mila is a solo CUDA-accelerated LLM inference framework targeting local inference. Built in C++23 with cuBLASLt and custom CUDA kernels.*

