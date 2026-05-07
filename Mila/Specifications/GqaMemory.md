# GQA Memory Optimization

Implementation Contract for Mila's CUDA Grouped-Query-Attention (GQA) Memory Optimization

---

## 1. Overview

This document specifies the required changes to eliminate the memory waste in Mila's
`CudaGroupedQueryAttentionOp` and `LlamaBlock` for the `LlamaTransformer` inference path.

The original implementation allocated approximately **3.2 GB** of CUDA device memory for
GQA buffers alone on Llama 3.2 3B Instruct at BF16. This is unacceptable given the 12 GB
VRAM budget — the model weights themselves are approximately 6 GB, leaving only ~800 MB
of headroom before accounting for activations and future model growth.

**Phase 1 is complete.** GQA state memory has been reduced from ~3.2 GB to ~1.38 GB
across 28 layers by rebuilding cuBLASLt plans against the compact NKV layout and
introducing a compact Q permute kernel. Phases 2 and 3 will recover the remaining ~1.18 GB.

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
`B=1, NH=24, NKV=8, GS=3, T=4096, HS=128, kPrefillChunkSize=64, model_dim=3072`, 28 layers.

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

### 2.2. CudaGroupedQueryAttentionOp — Legacy Always-Allocated Buffers (inference + training)

These buffers exist in the legacy path only. The Phase 1 optimized path eliminates
`q_tensor_`, `k_exp_tensor_`, and `v_exp_tensor_` entirely. `k_tensor_` and `v_tensor_`
are retained as the KV cache in both paths.

| Buffer | Shape | Per-Layer | × 28 Layers | Status |
|---|---|---|---|---|
| `q_tensor_` | `[1, 24, 4096, 128]` | 25.2 MB | **705 MB** | Eliminated in Phase 1 |
| `k_tensor_` | `[1, 8, 4096, 128]` | 8.4 MB | 235 MB | Retained (KV cache) |
| `v_tensor_` | `[1, 8, 4096, 128]` | 8.4 MB | 235 MB | Retained (KV cache) |
| `k_exp_tensor_` | `[1, 24, 4096, 128]` | 25.2 MB | **705 MB** | Eliminated in Phase 1 |
| `v_exp_tensor_` | `[1, 24, 4096, 128]` | 25.2 MB | **705 MB** | Eliminated in Phase 1 |
| `preatt_decode_tensor_` | `[1, 24, 1, 4096]` | 0.2 MB | 5 MB | Migrate in Phase 2/3 |
| `att_decode_tensor_` | `[1, 24, 1, 4096]` | 0.2 MB | 5 MB | Migrate in Phase 2/3 |
| `v_out_decode_tensor_` | `[1, 24, 1, 128]` | ~0 MB | ~0 MB | Migrate in Phase 2/3 |

**Legacy GQA total: ~3,310 MB ≈ 3.2 GB**

### 2.3. CudaGroupedQueryAttentionOp — Optimized Path Buffers (Phase 1, inference only)

The Phase 1 optimized path (`initializeState_optimized`) allocates a compact scratch set.
These buffers are self-owned during the A/B validation period and will be migrated to
the shared `LlamaTransformer` workspace in Phase 2.

| Buffer | Shape | Per-Layer | × 28 Layers |
|---|---|---|---|
| `k_tensor_` | `[1, 8, 4096, 128]` | 8.4 MB | 235 MB |
| `v_tensor_` | `[1, 8, 4096, 128]` | 8.4 MB | 235 MB |
| `q_permute_tensor_optimized_` | `[1, 24, 64, 128]` | 0.4 MB | 11 MB |
| `preatt_tensor_optimized_` | `[1, 24, 64, 4096]` | 12.6 MB | 352 MB |
| `att_tensor_optimized_` | `[1, 24, 64, 4096]` | 12.6 MB | 352 MB |
| `v_out_tensor_optimized_` | `[1, 24, 64, 128]` | 0.4 MB | 11 MB |
| `preatt_decode_tensor_` | `[1, 24, 1, 4096]` | 0.2 MB | 5 MB |
| `att_decode_tensor_` | `[1, 24, 1, 4096]` | 0.2 MB | 5 MB |
| `v_out_decode_tensor_` | `[1, 24, 1, 128]` | ~0 MB | ~0 MB |

**Phase 1 GQA total: ~1,206 MB ≈ 1.18 GB** (plus ~10 MB decode scratch)

### 2.4. The Three Guilty Buffer Groups

- `q_tensor_` — **705 MB** — persistent Q cache used only as a permute target within the same call, solving a non-contiguous memory problem that no longer exists. **Eliminated in Phase 1.**
- `k_exp_tensor_` + `v_exp_tensor_` — **1,410 MB** — full NH expansion of the NKV cache, rebuilt on every prefill chunk and every decode step, existing solely because cuBLASLt plans were built against `[B, NH, T, HS]` layout. **Eliminated in Phase 1.**
- `preatt_tensor_` + `att_tensor_` — **704 MB** — allocated permanently per layer, only needed during one layer's forward pass at a time; idle in 27 of 28 layers simultaneously. Migrated to shared workspace in Phase 2.

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

### 4.1. `padded_T` and Plan `strideC` Alignment

`padded_T` is the chunk-dimension stride passed to `prefill_unpermute_output_padded`. It
must match the `strideC` used by the cuBLASLt AV plan when writing `v_out`, which in turn
must match the `strideC` used by the QK plan when writing `preatt`.

The correct value is:

```cpp
const int padded_T = is_full_chunk ? static_cast<int>(kPrefillChunkSize) : chunk_len;
```

This is correct because both the `_optimized` plan builders and the fixed legacy plan
builders set `strideC = chunk_rows * T` (using the actual `chunk_rows` argument, not the
allocated `kPrefillChunkSize` capacity). For a partial chunk, cuBLASLt therefore writes
`v_out` with stride `chunk_len * HS` between heads — so `padded_T = chunk_len` is the
correct unpermute stride.

An earlier version of this section incorrectly stated that `padded_T` must always equal
`kPrefillChunkSize`. That was based on the assumption that `strideC` used the allocated
buffer capacity; once the plan builder bug was fixed (`strideC = chunk_rows * T` rather
than `prefill_window_size * T`), that guidance became wrong. The current code is correct.

Note that `preatt_`, `att_`, and `v_out_` buffers are still allocated with
`kPrefillChunkSize` rows as the outer dimension. The `chunk_len` value is a valid-row
count within that allocation, not the allocation size.

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

### Phase 1 — Rebuild cuBLASLt Plans Against NKV Layout (Gate) — COMPLETE

Rebuild all cuBLASLt QK and AV plans — prefill full-chunk, prefill partial-chunk, and
decode — to read K and V directly from `[B, NKV, T, HS]` with grouped head strides, where
`q[h]` attends `k[h / group_size]`.

For Mila's `B=1` inference target, each GQA group is a separate batched GEMM with
`batchCount = NKV`, with Q slices strided manually. The general `B>1` grouped GEMM path
is explicitly out of scope for this pass.

**Implementation approach.** An internal `bool` gate (`kUseOptimizedPath` /
`use_optimized_path_`) selects between the legacy and optimized execution paths without
changing any public API. New methods and plan builders carry an `_optimized` suffix during
the validation period; these suffixes are removed in a mechanical rename once the legacy
path is deleted. The legacy path is preserved verbatim and must not be modified.

New additions for the optimized path:
- `_optimized` plan builders in `CudaGqa.Plans.ixx` — `batchCount = B * NKV`, GS Q heads
  folded into M, no expansion buffers required.
- `permute_q_compact` kernel (`CudaGqa.Permute.cu`, dispatched via `CudaGqa.Dispatch.ixx`)
  — permutes Q from `[B, chunk, NH*HS]` into a compact `[B, NH, chunk, HS]` scratch buffer
  with stride `chunk*HS` between heads, replacing `kvcache_write_q` in the optimized path.
- `initializeState_optimized` — allocates only the KV cache; transient scratch is provided
  externally via `setState(GqaState)` from the shared `LlamaTransformer` workspace.
- `prefill_optimized` / `decode_optimized` — full implementations wired to the new plans
  and `permute_q_compact`.

**Validation.** Token-for-token correctness was confirmed by runtime inference traces
against conversational prompts. Memory reduction confirmed: ~3.2 GB legacy → ~1.38 GB
optimized (model state across 28 layers). Phase 2 gate is cleared.

**`padded_T` fix** was applied in the same pass to both legacy and optimized paths — see
Section 4.1 for the corrected analysis.

---

### Phase 2 — Shared GQA Workspace (Primary Memory Recovery) — COMPLETE

**Prerequisite:** Phase 1 validated correct.

**Introduce `GqaState` and `setState(GqaState)`.**
- New module `Compute.GqaState` — a plain struct of non-owning `ITensor*` pointers covering
  all seven transient scratch slots (four prefill, three decode).
- `CudaGqaOp::setState(GqaState)` wires the raw device pointers from the caller-supplied
  tensors. `initializeState_optimized` no longer allocates any transient scratch.
- `GroupedQueryAttention::setState` and `LlamaBlock::setState` forward the call down to the
  concrete op.

**Allocate the shared workspace in `LlamaTransformer::onBuilding`.**
Seven tensors are allocated once at the transformer level and passed to every block via
`block->setState(gqa_state)` immediately after the block-build loop. All 28 layers share
the same allocation because they execute sequentially.

The workspace covers all transient GQA buffers:

| Slot | Shape | Size |
|---|---|---|
| `q_permute` (prefill) | `[1, 24, 64, 128]` | 0.4 MB |
| `preatt` (prefill) | `[1, 24, 64, 4096]` | 12.6 MB |
| `att` (prefill) | `[1, 24, 64, 4096]` | 12.6 MB |
| `v_out` (prefill) | `[1, 24, 64, 128]` | 0.4 MB |
| `preatt_decode` | `[1, 24, 1, 4096]` | 0.2 MB |
| `att_decode` | `[1, 24, 1, 4096]` | 0.2 MB |
| `v_out_decode` | `[1, 24, 1, 128]` | ~0 MB |

**GQA workspace total: ~26 MB** — allocated once, reused across all 28 layers.

**Validation.** Memory confirmed at 736 MB state / 7.44 GB total for Llama 3.2 3B BF16.
Token-for-token correctness confirmed by runtime inference traces. Phase 2 is complete.

---

### Phase 3 — LlamaBlock Buffer Migration (Deferred to Beta Cleanup, ~28 MB)

**Prerequisite:** Phase 2 complete and validated correct. ✓

**Status: DEFERRED.** The remaining per-layer `LlamaBlock` inference buffers total ~28 MB
across 28 layers — less than 0.3% of the 12 GB VRAM budget. The marginal return does not
justify the disruption prior to the beta milestone. This work is tracked as a beta cleanup
task.

The buffers to be migrated when this phase is undertaken:

| Buffer | Shape | Per-Layer | × 28 Layers |
|---|---|---|---|
| `res1_prefill_` | `[1, 64, 3072]` | 0.4 MB | 11 MB |
| `q_` | `[1, 64, 24*128]` | 0.4 MB | 11 MB |
| `k_` | `[1, 64, 8*128]` | 0.1 MB | 3 MB |
| `v_` | `[1, 64, 8*128]` | 0.1 MB | 3 MB |

**LlamaBlock workspace total: ~28 MB recoverable.**

When undertaken, the `GqaState` workspace in `LlamaTransformer` should be extended to
cover these slots. For future models with heterogeneous layer configurations (e.g. MoE),
the workspace must be sized to the maximum across all layers — still a single static
allocation, requiring a max-reduce over layer configs at build time.

---

## 6. Confirmed Post-Fix Allocation

For Llama 3.2 3B BF16 (`B=1, NH=24, NKV=8, T=4096, HS=128`, 28 layers):

### After Phase 2 (current) — CONFIRMED

| Allocation | Owner | Spec | Actual |
|---|---|---|---|
| Weights | `LlamaTransformer` | ~6,000 MB | **6,720 MB** |
| K cache (all layers) | `CudaGqaOp` × 28 | 235 MB | included in state |
| V cache (all layers) | `CudaGqaOp` × 28 | 235 MB | included in state |
| RoPE cos/sin table | `CudaRopeOp` (shared) | ~67 MB | included in state |
| GQA shared workspace | `LlamaTransformer` | ~26 MB | included in state |
| Component output buffers | various × 28 layers | not modelled | included in state |
| **State total** | | **~601 MB** | **736 MB** |
| **Grand total** | | **~6,601 MB** | **7,440 MB** |

The ~135 MB state delta above the spec estimate is accounted for by component output
buffers (`Linear`, `RmsNorm`, `Residual`, `SwiGLU`) across 28 layers, which were not
modelled in the original allocation breakdown. These are not candidates for the GQA
workspace and are correctly owned at the component level.

**Net VRAM recovered from baseline (~10.2 GB): ~2.76 GB — 23% of the 12 GB budget.**

### After Phase 3 (target, deferred)

| Allocation | Owner | Size |
|---|---|---|
| Weights | `LlamaTransformer` | ~6,720 MB |
| State (KV cache + workspace + components) | various | ~708 MB |
| **Total** | | **~7,428 MB** |

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