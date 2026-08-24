# Prompt Caching

KV prefix reuse for Mila's inference paths — as shipped 2026-07-03 (Gemma first).

---

## 1. Overview

Every Mila caller is stateless per generate() call: the full prompt (system prompt +
conversation history + new turn) arrives on every call — the Chat harness rebuilds the
conversation each turn (and each tool round), and the Mila Inference Server's
Anthropic-compatible `POST /v1/messages` endpoint receives the full history per request.
Before this feature, `onGenerating` unconditionally prefilled from position 0, re-paying
the full prefill for tokens whose K/V the caches already held — measured at ~5-6 s for a
full 8K-context re-prefill on the 4070 even after the chunk-512 work
(Gemma4InferenceReview.md 5.4: the biggest user-visible chat win).

This specification describes the shipped zero-device-memory-overhead reuse: the model
transparently skips the prefill of any prompt prefix whose tokens exactly match what the
KV caches already hold.

---

## 2. Key Insight — No Device Memory Overhead Required

`resetKvCache()` only sets `cached_seq_len_ = 0`; it never zeroes device K/V contents.
K/V writes are purely positional (`kvcache_write_kv` takes the absolute
`position_offset`), so the rows written for tokens at positions `[0, n)` remain
physically present and correct after a generate call completes. Reusing them requires
only rewinding `cached_seq_len_` to `n` and prefilling the new tokens from
`position_offset = n`.

No snapshot buffers, no device-to-device copies, no additional device memory. This is
the same approach as llama.cpp / Ollama's `n_past` rewind.

K/V state at `[0, n)` is a deterministic function of the first `n` token ids and the
fixed model weights — so **exact token equality is the sole validity test**, and reuse
can never change outputs.

---

## 3. Design — Transparent Model-Side Reuse

The original draft threaded a `cacheable_prefix_len` hint through the generate surface.
As shipped, there is **no hint and no API change** (decided 2026-07-03): `generate()`
and `GenerateParams` are untouched. The model keeps a private token history and does
the matching itself:

```
GemmaModel::kv_token_history_   // the token ids whose K/V the caches hold, in
                                // position order: the last prefilled prompt plus
                                // every token fed through decode (appended in
                                // lockstep with the decode call)

onGenerating( prompt_tokens ):
    common = longest common prefix( prompt_tokens, kv_token_history_ )
    reuse  = min( common, prompt_len - 1 )     // always prefill >= the final position
                                               // so the sampled logits are fresh
    if reuse > 0 and network.rewindKvCache( reuse ):
        logits = network.prefillFrom( full_prompt_tensor, reuse )   // cache hit
    else:
        logits = network.prefill( full_prompt_tensor )              // full prefill
    kv_token_history_ = prompt_tokens          // then append per decode step
```

Consequences:

- **Every caller wins without changes**: Chat turns, Chat tool rounds (previously up to
  4 full re-prefills per tool-using turn), and MIS requests. MIS needs no handler logic,
  no pybind surface, and can ignore Anthropic `cache_control` blocks entirely.
- **Serial agentic clients (Codex / Claude CLI against MIS) are the best case** —
  monotonically growing prefixes. Interleaved conversations on one instance degrade
  gracefully: tiny common prefix, full prefill, exactly the pre-feature behavior.
- **Retokenization drift is safe**: a harness that rebuilds history from text may
  retokenize the assistant turn differently from the generated ids; the common prefix
  simply ends at the first divergence and the tail re-prefills. Savings become partial,
  correctness is untouched.
- Multi-session paged caching (vLLM-style radix trees) remains out of scope; if MIS
  grows session routing, this single-session policy moves up a layer and the primitives
  below carry over unchanged.

---

## 4. Implementation (as shipped)

### 4.1 `IKvCacheLifecycle` — `rewindKvCache` returns bool

```cpp
virtual bool rewindKvCache( int position ) = 0;
```

Returns `true` when the rewind is valid and was applied. Implementations must refuse
(`false`, state unchanged) when reuse would be incorrect. The bool is the bounded-ring
correction to the original draft (which predated SlidingWindowKvCache.md).

### 4.2 `CudaGqaOp<TPrecision, kBounded>`

Unbounded: any `0 <= position <= cached_seq_len_` is valid. Bounded ring additionally:
the resident rows are the last `cache_capacity_` written positions, and a continuation
from `position` attends down to `position - window_`, so

```
valid  <=>  cached_seq_len_ - position <= cache_capacity_ - window_   (= chunk - 1)
```

At chunk 512 the ring tolerates up to 511 stale tokens past the reuse point; chat
turn-boundary divergence is a few tokens, so hits are the overwhelming case. Pinned by
`RewindKvCache_BoundedRingEnforcesWindowValidity` (CudaGqaOp.Cuda.cpp).

### 4.3 `CudaMhaOp`

Unbounded one-liner for `IKvCacheLifecycle` parity.

### 4.4 Delegation chain (Gemma)

`GroupedQueryAttention::rewindKvCache` (keeps the cache session live — unlike
`resetKvCache` it does NOT clear `cache_initialized_`) -> `ITransformerBlock` /
`GemmaBlock::rewindKvCache` -> `GemmaTransformer::rewindKvCache` (AND across the
heterogeneous layer list; all-or-nothing from the caller's perspective — a refused or
partial rewind needs no cleanup because a full prefill positionally overwrites).

### 4.5 `LanguageModelNetwork` base

Two non-pure virtuals with safe defaults so Llama/Gpt compile untouched:
`prefillFrom(input, start_offset)` (default: throws) and `rewindKvCache(position)`
(default: `false`). Callers only reach `prefillFrom` after a successful rewind.

### 4.6 `GemmaTransformer::prefillFrom`

`prefill(input)` delegates to `prefillFrom(input, 0)` — one chunk-loop implementation.
`input` is the FULL prompt tensor (token index == absolute position); chunking starts at
`start_offset`. `start_offset` must lie inside the prompt (throws otherwise), so at
least one position always prefills and the returned last-position logits are fresh.
Pinned by `PrefillFrom_AfterRewind_MatchesFullPrefill` (Gemma.Cuda.cpp, real-weight
full-vs-incremental logits parity).

### 4.7 `GemmaModel`

`kv_token_history_` + the section 3 pseudocode in `onGenerating`; the history appends in
lockstep immediately before each `decode()` call, so early returns (stop token,
cancellation, context overflow) leave history == cache contents. A cache hit logs
`"KV prefix reuse -- skipped N of M prompt tokens"`.

---

## 5. Invariants

| Invariant | Notes |
|---|---|
| K/V contents at `[0, n)` are never modified by `rewindKvCache` | No zeroing anywhere in this path |
| `rewindKvCache` refuses rather than corrupts | position > fill, or bounded-ring staleness past `capacity - window` |
| A refused/partial rewind needs no cleanup | full prefill positionally overwrites all caches and resets every fill counter |
| `prefillFrom` never writes below `start_offset` | `kvcache_write_kv`'s `position_offset` argument enforces this |
| `kv_token_history_` tracks the caches exactly | assigned after prefill, appended in lockstep with each decode |
| Reuse never changes outputs | token equality + deterministic positional K/V; the only differences are float accumulation across chunk boundaries |
| Thread safety unchanged | the model is not thread-safe; MIS serializes per instance |

---

## 6. Scope and Follow-ups

- **Shipped**: Gemma chain (CudaGqaOp both axes, CudaMhaOp parity, GQA component,
  ITransformerBlock/GemmaBlock, GemmaTransformer, GemmaModel).
- **Follow-up (BACKLOG)**: Llama chain mirror (`LlamaBlock`/`LlamaTransformer`/
  `LlamaModel` — mechanical; the op layer is already done since Llama uses CudaGqaOp).
- **Deferred**: multi-session paged caching; the harness-level `GenerateSession`
  convenience (BACKLOG, Generation API section).
