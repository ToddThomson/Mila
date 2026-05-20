# Prompt Caching

Implementation Contract for Mila's System Prompt KV Cache Reuse

---

## 1. Overview

The Mila Inference Server exposes an Anthropic-compatible `POST /v1/messages` endpoint.
Each HTTP request is stateless: the full prompt (system prompt + conversation history +
new user turn) is sent on every call. `LlamaModel::onGenerating()` unconditionally calls
`LlamaTransformer::prefill()` starting at `position_offset = 0`, paying the full O(n^2)
attention cost for the system prompt tokens on every single request — even though those
tokens are identical across all requests for a given chat session.

This specification describes a zero-device-memory-overhead approach to skip the redundant
system prompt prefill on subsequent requests within the same session.

---

## 2. Key Insight — No Device Memory Overhead Required

`CudaGqaOp::resetKvCache()` only sets `cached_seq_len_ = 0`. It never zeroes or
invalidates the device K/V buffer contents. K/V writes are purely positional:

```cpp
kvcache_write_kv( k_, v_, Xk, Xv, B_, chunk_len, NKV_, HS_, position_offset, T_, stream );
```

The K/V data written for prefix tokens at positions `[0, prefix_len)` is still physically
present and correct in device memory after the decode loop completes. The only thing
needed to reuse it on the next request is to rewind `cached_seq_len_` back to `prefix_len`
before running a partial prefill over the new user turn tokens starting at
`position_offset = prefix_len`.

No snapshot buffers. No device-to-device copies. Zero additional device memory required.

This is the same approach used by `llama.cpp` / Ollama, which rewinds `n_past` to a
common prefix boundary rather than re-running the full prefill.

---

## 3. Design

### 3.1 Request Flow

```
Request N (cache miss / first request for this session)
--------------------------------------------------------
full prefill:  [system prompt | conversation | user turn N]   positions 0..T_N
               --> K/V cache contains [0, T_N)
               --> record prefix_cache_ = { system_tokens, prefix_len }

Request N+1 (cache hit, same session, same system prompt)
----------------------------------------------------------
rewindKvCache( prefix_len )
               --> cached_seq_len_ = prefix_len  (device buffers untouched)
partial prefill: [conversation delta | user turn N+1]   positions prefix_len..T_N+1
               --> K/V cache positions [0, prefix_len) reused as-is
               --> K/V cache positions [prefix_len, T_N+1) written fresh
```

### 3.2 Cache Validity

**Model layer** — a cache hit requires all of the following:
- `cacheable_prefix_len > 0`
- `prefix_cache_.len == cacheable_prefix_len`
- `prefix_cache_.tokens == prompt_tokens[0 .. cacheable_prefix_len)` (exact token equality)

Token equality is the sole identity test. No session ID is required at the model layer
because K/V state at `[0, prefix_len)` is a deterministic function of those tokens and
the fixed model weights.

**Server layer** — the endpoint handler must pass `cacheable_prefix_len = 0` when:
- The request carries no `system` field.
- The `system` field tokenizes to zero tokens.

All other cases let `LlamaModel`'s token equality check determine the hit/miss outcome.

**Stale tail data** — after each request the K/V cache holds positions `[0, T_last)`.
On rewind to `prefix_len`, positions `[prefix_len, T_last)` remain in device memory but
are unreachable: `prefillImpl` addresses only up to `position_offset + chunk_len` and
`decode` addresses only up to `position + 1`. No explicit invalidation is needed.

### 3.3 Scope

This feature targets **single-session, single-model-instance** use — one system prompt
per loaded `LlamaModel`. Multi-session paged caching (vLLM-style radix trees with LRU
eviction) is out of scope for this iteration.

---

## 4. Implementation

### 4.1 `IKVCacheLifecycle.ixx`

Add one new pure virtual method alongside the existing `resetKvCache()`:

```cpp
/// Rewind the logical cache fill position to `position` without modifying
/// device K/V buffer contents. Positions [0, position) remain valid for reuse.
/// Precondition: position <= cached_seq_len_ at call time.
virtual void rewindKvCache( int position ) = 0;
```

No other interface changes are required.

### 4.2 `CudaGqaOp.ixx`

```cpp
void rewindKvCache( int position ) override
{
    cached_seq_len_ = position;
}
```

Both the legacy (`prefillImpl`) and optimized (`prefill_optimized`) paths set
`cached_seq_len_ = position_offset + chunk_len` at the end of each chunk. A rewind
simply sets it back; neither path reads `cached_seq_len_` before writing K/V data, so
the rewind is safe.

### 4.3 `CudaMhaOp.ixx`

Same one-liner as `CudaGqaOp` for interface parity. LLaMA uses GQA exclusively but MHA
must remain consistent with `IKvCacheLifecycle`.

### 4.4 `GroupedQueryAttention.ixx`

Delegate through `positional_op_` using the same pattern as the existing `resetKVCache`:

```cpp
void rewindKvCache( int position )
{
    if ( positional_op_ )
        positional_op_->rewindKvCache( position );
}
```

### 4.5 `Llama.Block.ixx`

```cpp
void rewindKvCache( int position )
{
    if ( attn_ )
        attn_->rewindKvCache( position );
}
```

### 4.6 `Llama.ixx` (LlamaTransformer)

**a. Aggregate rewind across all blocks:**

```cpp
void rewindKvCache( int position )
{
    for ( auto& block : transformer_blocks_ )
        block->rewindKvCache( position );
}
```

**b. `prefillFrom()` — chunked prefill starting at a non-zero offset:**

Identical to the existing `prefill()` chunked loop with `offset` initialized to
`start_offset` rather than `0`. K/V cache positions `[0, start_offset)` are already
populated and are not touched.

```cpp
TensorType& prefillFrom( const TokenIndexType& input, int start_offset );
```

The implementation mirrors `prefill()` exactly — only the initial `offset` value differs.

### 4.7 `LanguageModel.ixx`

Add `cacheable_prefix_len` to `generateStreaming` and the protected `onGenerating` hook
with a default of `0` to preserve all existing call sites unchanged:

```cpp
void generateStreaming(
    const std::vector<int32_t>& prompt_tokens,
    std::function<void(int32_t)> on_token,
    size_t max_new_tokens       = 64,
    float temperature           = 1.0f,
    int top_k                   = 0,
    std::stop_token stop        = {},
    size_t cacheable_prefix_len = 0 );
```

### 4.8 `LlamaModel.ixx`

**a. New private member:**

```cpp
struct PrefixCache {
    std::vector<int32_t> tokens;
    int len = 0;
};

PrefixCache prefix_cache_;
```

**b. Updated `onGenerating` logic:**

```
if cacheable_prefix_len > 0:

    prefix = prompt_tokens[0 .. cacheable_prefix_len)
    tail   = prompt_tokens[cacheable_prefix_len .. end)

    if prefix_cache_.len == cacheable_prefix_len &&
       prefix_cache_.tokens == prefix:

        // Cache hit: rewind and partial prefill only
        getNetwork().rewindKvCache( cacheable_prefix_len )
        logits = getNetwork().prefillFrom( tail_tensor, cacheable_prefix_len )
        Logger::info( "prompt cache hit, skipped {} prefix tokens" )

    else:
        // Cache miss: full prefill, record new prefix boundary
        logits = getNetwork().prefill( full_tensor )
        prefix_cache_ = { prefix, cacheable_prefix_len }
        Logger::info( "prompt cache miss, full prefill {} tokens" )

else:
    // No caching hint: full prefill, invalidate any cached prefix
    logits = getNetwork().prefill( full_tensor )
    prefix_cache_ = {}
```

The `position` counter passed to the decode loop must always be `prompt_tokens.size()`
(total tokens including the cached prefix) regardless of which prefill path was taken.

### 4.9 Inference Server — `v1/messages` Handler

Before assembling the full prompt token sequence, tokenize the `system` field
independently to obtain `prefix_len`. Pass it as `cacheable_prefix_len` to
`generateStreaming`. The system prompt tokens must always occupy
`prompt_tokens[0 .. prefix_len)` — prepended before all conversation history and user
turn tokens — so that `prefix_cache_.tokens` comparison is stable across requests.

---

## 5. Invariants

| Invariant | Notes |
|---|---|
| K/V buffer contents at `[0, prefix_len)` are never modified by `rewindKvCache` | Correctness depends on this — no zeroing permitted in this path |
| `prefix_cache_` is cleared when `cacheable_prefix_len == 0` | Prevents stale reuse after session changes |
| `prefillFrom` must not write K/V positions below `start_offset` | `kvcache_write_kv` `position_offset` argument already enforces this |
| `rewindKvCache(position)` requires `position <= cached_seq_len_` | `LlamaModel` is responsible for satisfying this precondition |
| Thread safety is unchanged | `LlamaModel` is not thread-safe; the server must serialize access per model instance |

---

## 6. Files Modified

| File | Change |
|---|---|
| `Mila/Src/Dnn/Compute/Operations/IKVCacheLifecycle.ixx` | Add `rewindKvCache(int)` |
| `Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx` | Implement `rewindKvCache` |
| `Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/MHA/CudaMhaOp.ixx` | Implement `rewindKvCache` (parity) |
| `Mila/Src/Dnn/Components/Attention/GQA/GroupedQueryAttention.ixx` | Delegate `rewindKvCache` |
| `Mila/Src/Dnn/Components/Transformers/LlaMa/Llama.Block.ixx` | Delegate `rewindKvCache` |
| `Mila/Src/Dnn/Components/Transformers/LlaMa/Llama.ixx` | Aggregate `rewindKvCache`; add `prefillFrom()` |
| `Mila/Src/Dnn/Core/LanguageModel.ixx` | Add `cacheable_prefix_len = 0` to `generateStreaming` / `onGenerating` |
| `Mila/Src/Dnn/Models/LlamaModel.ixx` | Add `PrefixCache` member; update `onGenerating` |
| Inference server `v1/messages` handler | Extract system prompt token count; pass as `cacheable_prefix_len` |

## 7. Files Created

| File | Purpose |
|---|---|
| `Mila/Specifications/PromptCaching.md` | This specification |
