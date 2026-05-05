# GQA Memory Optimization

Implementation Contract for Mila's CUDA Grouped-Query-Attention (GQA) Memory Optimization

---

## 1. Overview

This document specifies the required changes to eliminate the memory waste in Mila's
`CudaGroupedQueryAttentionOp` and `LlamaBlock` for the `LlamaTransformer` inference path.

The current implementation allocates approximately **3.2 GB** of CUDA device memory for
GQA buffers alone on Llama 3.2 3B Instruct at BF16. This is unacceptable given the 12 GB
VRAM budget — the model weights themselves are approximately 6 GB, leaving only ~800 MB
of headroom before accounting for activations and future model growth.

### Scope

**`CudaGroupedQueryAttentionOp` — the primary source of the problem.** The three guilty
buffer groups account for ~3.2 GB and are eliminated in Phase 1 and Phase 2.

**`LlamaBlock` — secondary cleanup.** Per-layer split and residual buffers account for
~28 MB and are eliminated in Phase 3.

**`CudaRopeOp` — no changes required.** RoPE is correctly implemented:

- The cos/sin table is a single shared allocation via `RopeCacheRegistry`, not duplicated per layer.
- Both `prefill` and `decode` write into caller-provided output tensors — RoPE owns no output buffers.
- The shared table for Llama 3.2 at FP32 costs approximately 67 MB total across all 28 layers.

### Key Architectural Invariants

**RoPE produces fully-rotated Q and K before they reach GQA.** The call chain in
`LlamaBlock::prefill` is:

```
qkv_proj_->forward( rms1_out )               // fused QKV projection
split( qkv_out, q_view, k_view, v_view )     // split into separate contiguous tensors
rope_->prefill( q_view, k_view, ... )        // rotate Q and K in-place
attn_->prefill( q_view, k_view, v_view, ...) // GQA receives pre-rotated, contiguous Q, K, V
```

By the time Q and K reach GQA they are in `[B, T_actual, NH*HS]` and
`[B, T_actual, NKV*HS]` layout respectively — separately allocated, contiguous, and
fully RoPE'd. GQA never applies RoPE internally.

**Only K and V are reused across decode steps.** Q for a given token is computed once,
used once, and discarded. The KV cache is exclusively for K and V.

### Implementation History

The current `kvcache_write_q` and `kvcache_expand_kv` kernels were introduced to solve a
non-contiguous memory problem that existed when GQA received a fused `[B, T, (NH+2*NKV)*HS]`
QKV tensor as input. Slicing Q out of a fused buffer produced a non-contiguous layout that
cuBLASLt could not stride correctly — only head 0 computed correctly, with heads 1..NH-1
reading from garbage offsets. The permute kernels solved this by writing into fresh
contiguous buffers that the plans could consume cleanly.

The architectural invariant established since then — separate contiguous Q, K, V tensors
delivered to GQA after the `split()` step — means the non-contiguous memory problem no
longer exists. The permute kernels are now solving a problem that was eliminated by earlier
design work. The cuBLASLt plans were never updated to reflect this, and the expansion
buffers they require have persisted as a result.

---

## 2. Current Allocation Breakdown (Confirmed)

The following tables show the confirmed allocations for Llama 3.2 3B BF16 at inference:
`B=1, NH=24, NKV=8, T=4096, HS=128, kPrefillChunkSize=64, model_dim=3072`, 28 layers.

### 2.1. LlamaBlock — Per-Layer Inference Buffers

| Buffer | Shape | Per-Layer | × 28 Layers |
|---|---|---|---|
| `res1_prefill_` | `[1, 64, 3072]` | 0.4 MB | 11 MB |
| `q_` | `[1, 64, 24*128]` | 0.4 MB | 11 MB |
| `k_` | `[1, 64, 8*128]` | 0.1 MB | 3 MB |
| `v_` | `[1, 64, 8*128]` | 0.1 MB | 3 MB |

**LlamaBlock total: ~28 MB**

Note: `qkv_proj_` owns an internal output buffer at `[1, 64, 40*128]` = 0.6 MB × 28 = 17 MB.
This is tracked inside the `LinearType` component and is a separate concern from this document.

### 2.2. CudaGroupedQueryAttentionOp — Always-Allocated Buffers (inference + training)

| Buffer | Shape | Per-Layer | × 28 Layers |
|---|---|---|---|
| `q_tensor_` | `[1, 24, 4096, 128]` | 25.2 MB | **705 MB** |
| `k_tensor_` | `[1, 8, 4096, 128]` | 8.4 MB | 235 MB |
| `v_tensor_` | `[1, 8, 4096, 128]` | 8.4 MB | 235 MB |
| `k_exp_tensor_` | `[1, 24, 4096, 128]` | 25.2 MB | **705 MB** |
| `v_exp_tensor_` | `[1, 24, 4096, 128]` | 25.2 MB | **705 MB** |
| `preatt_decode_tensor_` | `[1, 24, 1, 4096]` | 0.2 MB | 5 MB |
| `att_decode_tensor_` | `[1, 24, 1, 4096]` | 0.2 MB | 5 MB |
| `v_out_decode_tensor_` | `[1, 24, 1, 128]` | ~0 MB | ~0 MB |

### 2.3. CudaGroupedQueryAttentionOp — Inference-Only Prefill Buffers

| Buffer | Shape | Per-Layer | × 28 Layers |
|---|---|---|---|
| `preatt_tensor_` | `[1, 24, 64, 4096]` | 12.6 MB | **352 MB** |
| `att_tensor_` | `[1, 24, 64, 4096]` | 12.6 MB | **352 MB** |
| `v_out_tensor_` | `[1, 24, 64, 128]` | 0.4 MB | 11 MB |

**GQA total: ~3,310 MB ≈ 3.2 GB**

### 2.4. The Three Guilty Buffer Groups

- `q_tensor_` — **705 MB** — persistent Q cache used only as a permute target within the same call, solving a non-contiguous memory problem that no longer exists
- `k_exp_tensor_` + `v_exp_tensor_` — **1,410 MB** — full NH expansion of the NKV cache, rebuilt on every prefill chunk and every decode step, existing solely because cuBLASLt plans were built against `[B, NH, T, HS]` layout
- `preatt_tensor_` + `att_tensor_` — **704 MB** — allocated permanently per layer, only needed during one layer's forward pass at a time; idle in 27 of 28 layers simultaneously

---

## 3. Root Cause Analysis

### 3.1. `q_tensor_` — Persistent Q Cache

Q is written into the persistent `q_` buffer via `kvcache_write_q` in both prefill and decode:

```cpp
// Prefill
Detail::cuda_gqa_kernels<NativeType>::kvcache_write_q(
    q_, Xq, B_, chunk_len, NH_, HS_, position_offset, T_, stream );

// Decode
Detail::cuda_gqa_kernels<NativeType>::kvcache_write_q(
    q_, Xq, B_, 1, NH_, HS_, position, T_, stream );
```

In both cases the buffer is read back within the same call:

```cpp
const NativeType* q_chunk  = q_ + static_cast<int64_t>(position_offset) * HS_;  // prefill
const NativeType* q_decode = q_ + static_cast<int64_t>(position) * HS_;          // decode
```

Q is never reused across calls. Storing Q in a `[B, NH, T_max, HS]` persistent buffer —
25.2 MB per layer, 705 MB across 28 layers — to read it back within the same call is pure
waste. In the decode path this writes approximately 6 KB of data into a 25 MB buffer slot.

The fragile layout coupling this creates is also worth noting. The offset:

```cpp
const NativeType* q_chunk = q_ + static_cast<int64_t>(position_offset) * HS_;
```

assumes `q_` is laid out as `[NH, T, HS]` flattened and relies on the cuBLASLt plan's
`strideA` to handle the head dimension. This coupling disappears entirely once Q becomes
a transient workspace buffer with an explicit permute step.

### 3.2. `k_exp_tensor_` and `v_exp_tensor_` — NH Expansion Buffers

`kvcache_expand_kv` expands the full KV history from NKV heads to NH heads on every
prefill chunk and every decode step:

```cpp
// Prefill — expands [0..total_kv_len) on every chunk
Detail::cuda_gqa_kernels<NativeType>::kvcache_expand_kv(
    k_exp_, v_exp_, k_, v_,
    B_, total_kv_len, T_, NH_, NKV_, HS_, 0, stream );

// Decode — expands [0..position+1) on every generated token
Detail::cuda_gqa_kernels<NativeType>::kvcache_expand_kv(
    k_exp_, v_exp_, k_, v_,
    B_, actual_len, T_, NH_, NKV_, HS_, 0, stream );
```

This has two compounding problems:

**Memory:** The expanded buffers are `[B, NH, T, HS]` — a 3× expansion over the NKV=8
cache (NH=24), stored persistently per layer. This completely negates GQA's memory
advantage over MHA.

**Compute:** The expansion is O(T²) across prefill (re-expands the full growing history
on each chunk) and O(T²) bandwidth across decode (re-expands a growing sequence on every
generated token). Over a 500-token generation at T=2048 cached, this is 500 full expand
operations of growing size.

The expansion exists solely because the cuBLASLt plans were built against the
`[B, NH, T, HS]` expanded layout. Rebuilding the plans against `[B, NKV, T, HS]` with
grouped head strides eliminates both the buffers and the kernel entirely.

### 3.3. `preatt_tensor_` and `att_tensor_` — Per-Layer Prefill Scratch

The prefill attention score buffers are allocated permanently per layer at
`[B, NH, kPrefillChunkSize, T]`. Since `LlamaTransformer` executes layers sequentially,
only one layer's prefill buffers are active at any point. These 352 MB + 352 MB = 704 MB
of buffers are idle for 27 out of 28 layers at all times. They are natural candidates for
a single shared workspace allocation owned by `LlamaTransformer`.

---

## 4. Secondary Issues

### 4.1. `padded_T` Inconsistency

The current partial-chunk logic sets `padded_T = chunk_len` for non-full chunks:

```cpp
const int padded_T = is_full_chunk ? static_cast<int>(kPrefillChunkSize) : chunk_len;
```

`padded_T` is used as the chunk-dimension stride in `prefill_unpermute_output_padded`.
The `preatt_`, `att_`, and `v_out_` buffers are always allocated with `kPrefillChunkSize`
rows, not `chunk_len` rows. Passing `chunk_len` as the stride on partial chunks is
inconsistent with the buffer geometry and currently works only because the partial plans
happen to write tight-packed output. `padded_T` must always equal `kPrefillChunkSize`.
The valid row count `chunk_len` is a separate argument and must not be conflated with the
buffer stride.

### 4.2. `max_seq_len` vs. RoPE Cache Size

`active_max_seq_len_` is initialized from `parameter.max_seq_len`. Care must be taken to
ensure this resolves to the model's actual context window (4,096 for Llama 3.2 3B, 8,192
for Llama 3.1 8B) and never to the RoPE table size (131,072). The KV cache `T_` dimension
and all cuBLASLt plan geometries must be sized against the model context window only.
The RoPE table size is an independent concern owned entirely by `CudaRopeOp`.

---

## 5. Required Changes

The work is organized into three sequential phases. Phase 1 is the prerequisite gate —
Phases 2 and 3 cannot begin until Phase 1 is validated correct.

---

### Phase 1 — Rebuild cuBLASLt Plans Against NKV Layout (Gate)

Rebuild all cuBLASLt QK and AV plans — prefill full-chunk, prefill partial-chunk, and
decode — to read K and V directly from `[B, NKV, T, HS]` with grouped head strides, where
`q[h]` attends `k[h / group_size]`.

For Mila's `B=1` inference target, each GQA group is a separate batched GEMM with
`batchCount = NKV`, with Q slices strided manually. The general `B>1` grouped GEMM path
is explicitly out of scope for this pass.

Validate the new plans layer-by-layer against HuggingFace reference values using the
established tensor dump methodology before proceeding. Phase 2 cannot begin until
token-for-token correctness is confirmed.

Fix `padded_T` in the same pass — it is a self-contained correctness fix with no
dependency on the plan rebuild.

---

### Phase 2 — GQA Buffer Surgery (Primary Memory Recovery, ~3.2 GB)

Once Phase 1 plans are validated:

**Delete `q_tensor_` and `kvcache_write_q`.**
Remove `q_tensor_`, `q_`, and all calls to `kvcache_write_q` from both `prefill` and
`decode`. Replace with a transient permute of the incoming `Xq` pointer into the shared
workspace Q slot (see Phase 2 workspace below).

**Delete `k_exp_tensor_`, `v_exp_tensor_`, and `kvcache_expand_kv`.**
Remove both expansion buffers and all calls to `kvcache_expand_kv` from both `prefill`
and `decode`. The Phase 1 plans read directly from `k_tensor_`/`v_tensor_`.

**Retain `k_tensor_` and `v_tensor_` unchanged.**
These are the KV cache. No changes to their shape, dtype, or ownership.

**Introduce a shared workspace passed in from `LlamaTransformer`.**
Allocate a single workspace tensor at the `LlamaTransformer` level and pass it into each
`LlamaBlock` and down into `CudaGroupedQueryAttentionOp` for each forward call. The
workspace covers all transient GQA buffers:

| Slot | Shape | Size |
|---|---|---|
| Q permute (prefill) | `[1, 24, 64, 128]` | 0.4 MB |
| Q permute (decode) | `[1, 24, 1, 128]` | ~0 MB |
| `preatt` (prefill) | `[1, 24, 64, 4096]` | 12.6 MB |
| `att` (prefill) | `[1, 24, 64, 4096]` | 12.6 MB |
| `v_out` (prefill) | `[1, 24, 64, 128]` | 0.4 MB |

**GQA workspace total: ~26 MB** — allocated once, reused across all 28 layers sequentially.

Remove `preatt_tensor_`, `att_tensor_`, and `v_out_tensor_` from `initializeState`.
`preatt_decode_`, `att_decode_`, and `v_out_decode_` are minor (~10 MB total) and may be
migrated to the workspace in the same pass or deferred to Phase 3.

---

### Phase 3 — LlamaBlock Buffer Migration (Required Cleanup, ~28 MB)

**Prerequisite:** Phase 2 complete and validated correct.

Migrate the `LlamaBlock` per-layer inference buffers into the shared workspace introduced
in Phase 2. These buffers are only live during one layer's forward pass at a time —
identical to the GQA prefill scratch buffers — and belong in the same workspace:

| Buffer | Shape | Per-Layer | × 28 Layers |
|---|---|---|---|
| `res1_prefill_` | `[1, 64, 3072]` | 0.4 MB | 11 MB |
| `q_` | `[1, 64, 24*128]` | 0.4 MB | 11 MB |
| `k_` | `[1, 64, 8*128]` | 0.1 MB | 3 MB |
| `v_` | `[1, 64, 8*128]` | 0.1 MB | 3 MB |

**LlamaBlock workspace total: ~28 MB recovered.**

The workspace sizing at `LlamaTransformer` level must account for all slots across both
GQA and `LlamaBlock` when Phase 3 is complete. For future models with heterogeneous layer
configurations (e.g. MoE with differing head counts per layer), the workspace must be
sized to the maximum across all layers — still a single static allocation, requiring a
max-reduce over layer configs at build time.

---

## 6. Expected Post-Fix Allocation

For Llama 3.2 3B BF16 (`B=1, NH=24, NKV=8, T=4096, HS=128`, 28 layers):

### After Phase 2 (primary fix)

| Allocation | Owner | Size |
|---|---|---|
| Weights | `LlamaTransformer` | ~6,000 MB |
| K cache (all layers) | `CudaGroupedQueryAttentionOp` × 28 | 235 MB |
| V cache (all layers) | `CudaGroupedQueryAttentionOp` × 28 | 235 MB |
| RoPE cos/sin table | `CudaRopeOp` (shared) | ~67 MB |
| GQA shared workspace | `LlamaTransformer` | ~26 MB |
| Decode scratch (`preatt`, `att`, `v_out`) | `CudaGroupedQueryAttentionOp` × 28 | ~10 MB |
| LlamaBlock buffers (Phase 3 pending) | `LlamaBlock` × 28 | ~28 MB |
| **Total** | | **~6,601 MB ≈ 6.6 GB** |

**Reduction from current: ~3.6 GB — recovering 30% of the 12 GB VRAM budget.**

### After Phase 3 (complete)

| Allocation | Owner | Size |
|---|---|---|
| Weights | `LlamaTransformer` | ~6,000 MB |
| K cache (all layers) | `CudaGroupedQueryAttentionOp` × 28 | 235 MB |
| V cache (all layers) | `CudaGroupedQueryAttentionOp` × 28 | 235 MB |
| RoPE cos/sin table | `CudaRopeOp` (shared) | ~67 MB |
| Shared workspace (GQA + block) | `LlamaTransformer` | ~54 MB |
| Decode scratch (`preatt`, `att`, `v_out`) | `CudaGroupedQueryAttentionOp` × 28 | ~10 MB |
| **Total** | | **~6,601 MB ≈ 6.6 GB** |

The Phase 3 saving is ~28 MB — modest in absolute terms but required to complete the
architectural principle that all transient per-layer inference buffers are owned by a
single shared workspace at `LlamaTransformer` level.

---

## 7. Out of Scope

The following items are deferred. They are correctness or performance improvements
independent of the memory fix and must not be mixed into this work.

- Warp-level softmax kernels
- Fused unpermute kernels
- Paged KV cache
- Full fused attention (QK + softmax + AV)
- `KVCacheDType` / configurable KV cache precision (tracked separately under Alpha.4 FP8 work)
- Runtime statistics logging
- Training path gradient buffer ownership migration (tracked separately under the existing `REVIEW:` comment in `initializeState`)