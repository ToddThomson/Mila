# GQA Attention Extent: Attended Length vs Physical Stride

Status: implemented (2026-07-07), pending on-GPU parity + tax-gone validation. Scope:
`CudaGroupedQueryAttentionOp` unbounded/global prefill path and its prefill softmax
kernels. Interim optimization ahead of the post-0.20 flash-attention prefill rewrite,
which subsumes it.

---

## 1. Problem

For the unbounded (global) GQA layers, the prefill attention computes over the full
*allocated* `context_length`, not the *used* KV length. The width parameter
`T_stride` passed to the QK GEMM, the softmax, and the AV GEMM is
`cache_capacity_`, which for the unbounded cache equals `T_` (the model's
`context_length`, [CudaGqaOp.ixx:270](../Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx)).
`T_stride` is used simultaneously as the physical row stride *and* the logical key
count, so nothing distinguishes "how wide is the buffer" from "how many keys are
valid".

Consequences:

- **Over-allocation tax.** A prompt far shorter than `context_length` still pays
  attention over `context_length` columns. Measured (Gemma 12B FP4, RTX 4070, fixed
  512-token prompt, varying `context_length`):

  | context_length | prefill (512-token prompt) |
  |---:|---:|
  | 512 | 298.6 ms |
  | 2048 | 343.5 ms |
  | 8192 | 380.6 ms |
  | 16384 | 430.4 ms |
  | 40960 | 639.5 ms |

  2.1x, ~linear in `context_length`. This is paid by the Mila Inference Server, which
  runs `MILA_CONTEXT_LENGTH=40960` to fit Claude Code's ~35.7K-token harness prompt but
  then taxes every *short* conversation.

- **Rectangular, not triangular.** Even at `context_length == seq_len`, each prefill
  chunk computes over the full width rather than only up to its causal position, so the
  GEMMs do ~2x the necessary causal work.

Decode is affected only marginally (measured -8% from ctx 512 -> 40960): decode is
weight-bandwidth-bound (streaming the 12B FP4 weights per token dominates), so the
O(context) attention is a small slice. Decode is therefore descoped here (see 5).

## 2. Root cause

`cache_capacity_` is the physical row count of the compact KV cache
`[B, NKV, cache_capacity_, HS]`. For the unbounded path `cache_capacity_ == T_`
([CudaGqaOp.ixx:270](../Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx)),
and it is passed as the kernels' `T_stride`
([CudaGqaOp.ixx:638-640](../Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx)).
In `prefill_softmax_bf16_kernel`
([Gqa.Prefill.Bf16.cu](../Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Prefill.Bf16.cu)):

- `row_offset = (... * chunk_len + t) * T_stride` uses `T_stride` as the physical
  stride.
- Steps 1-3 (max / exp / normalize) already loop only to
  `max_t2 = min(abs_t, T_stride - 1)` (causal), so the arithmetic is bounded by
  position -- good.
- **Step 4 zeros `[max_t2+1, T_stride)`** -- O(`context_length`) writes per query row.
- The QK GEMM produces `preatt[chunk_len x T_stride]` and the AV GEMM consumes
  `att[chunk_len x T_stride]`, both with the K/N dimension = `T_stride`.

So the width `T_stride == context_length` drives GEMM extents, softmax zeroing, and
memory traffic, independent of the actual prompt length.

The bounded (local) layers are unaffected: `cache_capacity_ = min(T_, window + chunk - 1)`
([CudaGqaOp.ixx:266](../Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx)),
already capped at the window.

## 3. Design

Introduce a per-chunk **attended length**

```
L = position_offset + chunk_len        // causal key count for this chunk's queries
```

and use `L` wherever the code currently uses `T_stride` as an *extent* (a count of
valid keys), while keeping `T_` only as the *physical stride* for addressing. The two
are decoupled by giving the kernels a separate `attended_len` parameter distinct from
the existing row-stride argument.

Per unbounded prefill chunk:

- **QK GEMM:** N (key count) = `L` (was `T_`). Writes `preatt[chunk_len x L]` into a
  buffer whose physical row stride remains `T_`.
- **Softmax:** pass `L` as the logical extent; Step-4 zeroing runs `[max_t2+1, L)`.
  Steps 1-3 are unchanged (already causal-bounded). Row addressing still uses the
  physical stride `T_`.
- **AV GEMM:** K = `L`. Columns `[L, T_)` are never read, so they need no zeroing.

Cost becomes O(`chunk_len x L`) per chunk; summed over chunks it is triangular
(~seq^2 / 2) and **independent of `T_`**. This removes the over-allocation tax and
makes prefill attention causal-triangular in one change.

Invariant: results are identical to today for any input. The columns dropped from the
extent are exactly the ones today's Step 4 zeros and the AV GEMM multiplies by zero, so
no value that reaches the output changes. This is a *where-we-compute*, not a
*what-we-compute*, change -- the basis for the parity test (4).

## 4. Validation

1. **Token-for-token parity.** Run current vs modified at a fixed `context_length`
   (e.g. `context_length == seq_len` and `context_length >> seq_len`); generated tokens
   and per-layer attention output must match exactly. This is the safety net.
2. **Tax-gone check.** Re-run the isolation experiment (fixed 512-token prompt, varying
   `context_length`); prefill time should be ~flat across `context_length` after the fix
   (contrast the table in 1).
3. **Existing GQA prefill/decode parity oracles** stay green (single-chunk, multi-chunk
   across window, partial-final-chunk, prefill-then-decode).

## 5. cuBLASLt plan geometry (the risk)

`L` varies per chunk (`L = (i+1) * chunk` for full chunk `i`), so the QK/AV plans become
shape-per-chunk. The partial-plan cache was generalized to
`getOrBuildPrefillQKPlan_optimized` / `getOrBuildPrefillAVPlan_optimized`, keyed on
`makePlanKey(chunk_len, L)` (both fields, since the bounded partial chunk shares
`L == capacity` across prefills but varies `chunk_len`). Distinct plans ~= `seq / chunk`
(~70 for a 35K prefill), built once and reused across prefills.

The kernels need one new argument: `attended_len` distinct from the physical row stride.
The partial-chunk note at
[Gqa.Prefill.Bf16.cu:54-59](../Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Prefill.Bf16.cu)
shows the `chunk_len` vs `chunk_stride` (extent vs stride) split has caused a garbage-output
bug before -- the same class of error lives here, so the parity test (4) is mandatory,
not optional, and the cuBLASLt leading-dimension / stride must stay `T_` while N/K become
`L`.

## 6. Descoped: decode

Decode reads `cache_capacity_ == T_` columns per token
([CudaGqaOp.ixx:710-712](../Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx),
comment "Both read exactly cache_capacity_ columns"), so it has the same structural tax.
But measured impact is only ~8% (weight-bandwidth-bound), not worth the plan surgery
(`actual_len = position + 1` grows every token, so per-length plans explode). Optional
low-risk freebie: tighten the decode softmax loop bound to the already-passed `actual_len`
-- recovers most of the 8% without touching the decode QK/AV plans. Leave the plans alone.

## 7. Relationship to flash-attention

The post-0.20 flash-attention prefill kernel (BACKLOG, tiled/online-softmax, no
materialized scores) subsumes this: it is inherently causal-triangular and allocation-
independent, and it additionally removes the score-materialization memory traffic this
change leaves in place. This spec is the *interim* win because flash-attention is
post-0.20; when flash lands, these unbounded prefill kernels are replaced and this extent
plumbing retires with them.

## 8. Effort

Kernel `attended_len` parameter + Step-4 bound: low. Plan-cache-by-`L`: medium (cuBLASLt
ld/stride is where bugs hide). Estimated ~half a day of edits plus a VS2026 build and an
on-GPU parity run.
