# GQA FlashAttention: Fused Prefill on the Compact KV Cache

Status: Iteration 1 implemented and validated -- correct (parity green) but a **prefill
regression** (74% of prefill GPU time, ~100x memory-bound; 5.0), so `use_flash_prefill_`
defaults to false and flash stays behind the toggle until the tiled kernel (5.2) lands.
Scope: `CudaGqaOp` prefill path. Subsumes and retires `GqaAttentionExtent.md` only once the
tiled kernel makes flash the faster path across the bounded + global layers.

Note: this specification was written alongside the Iteration 1 implementation, not ahead
of it. It is the design of record; where the code and this document disagree, this
document is correct and the code is the defect.

---

## 1. Motivation

Two problems, one change.

**Prefill TTFT.** Measured ~4x slower than llama.cpp on Gemma 4 (LM Studio, same model
family). The cuBLASLt prefill pipeline (2) materializes the full attention score matrix,
is rectangular rather than causal-triangular on the global layers (`GqaAttentionExtent.md`
reduced this but did not remove it), spends six kernel launches per layer per chunk, and
is throttled to `chunk_len = 256` at ctx 40960 because the score scratch does not fit
12 GB.

**Memory / edge viability.** The materialized scores and the transient permute/output
workspace are exactly what a 12-16 GB edge card cannot afford at long context. A modern
attention kernel never materializes them.

FlashAttention addresses both with one mechanism: an online (streaming) softmax that keeps
all intermediate state on-chip, so transient attention workspace scales to **zero**, the
computation is inherently causal-triangular and independent of the allocated context, and
the prefill chunk size is decoupled from the VRAM budget.

### 1.1 Scope boundary (what flash does not fix)

Gemma 4 interleaves **5 sliding : 1 global** attention layers
(`GemmaConfig::isGlobalLayer`, `(layer_index + 1) % sliding_window_pattern == 0`,
pattern 6). Flash attacks the O(S^2) global layers directly and removes the
all-layers score-materialization tax, but the linear/FFN weight GEMMs (W4A16 FP4, O(S),
weight-bandwidth-bound) are outside this kernel. A prefill attribution -- attention vs
linear GEMM vs launch overhead, captured at 8K and 32K prompts -- is the honest way to
bound what flash can buy before claiming it closes the full 4x. If the linear GEMMs
dominate, that is a separate roadmap item; flash does not recover it.

## 2. Background: the current cuBLASLt prefill pipeline

`CudaGqaOp<TPrecision, kBounded>::prefill_optimized`
([CudaGqaOp.ixx](../Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx))
runs six device steps per chunk:

1. `kvcache_write_kv` -- append K/V into the compact cache `[B, NKV, cache_capacity, HS]`.
2. `permute_q_compact` -- Q `[B, chunk, NH*HS]` -> `[B, NH, chunk, HS]` scratch.
3. QK cuBLASLt plan -> `preatt [B, NH, chunk, attended_len]`.
4. `prefill_softmax_bf16_kernel` -> `att [B, NH, chunk, attended_len]`.
5. AV cuBLASLt plan -> `v_out [B, NH, chunk, HS]`.
6. `prefill_unpermute_output_padded` -> `Y [B, chunk, NH*HS]`.

Steps 2-6 exist only to satisfy cuBLASLt's strided GEMM layout. The KV cache
(`k_tensor_`, `v_tensor_`) is retained; all transient buffers (`q_permute_opt_`,
`preatt_opt_`, `att_opt_`, `v_out_opt_`) are a single shared workspace owned by
`LlamaTransformer` and wired via `setState()`, sized to the maximum over all GQA layers.
`preatt`/`att` are the `O(chunk x cache_capacity)` buffers that force the chunk-size
throttle at long context.

The KV cache is stored in **`TPrecision`** -- BF16 for the validated targets. There is no
KV quantization in this op.

## 3. Premise correction (design decision record)

A pre-existing draft (`GqaFlashAttention.v2.md`, `Gqa.Flash.cu`, `Gqa.Flash.Wmma.cu`)
specified a fused kernel over a **pre-quantized FP4 KV cache** (`K_fp4`, `V_fp4`,
`KV_scales`, `ITensor::getQuantizationScales()`), dequantizing nibbles inside the inner
loop. `CudaGqaOp` has no such cache: it stores K/V in BF16 and no KV-quantization policy
is wired in. That draft therefore couples two independent features -- fused attention and
FP4 KV quantization -- onto a data layout that does not exist.

**Decision: implement flash on the BF16 cache first; treat quantized KV as a later,
separate iteration (5.4).** Rationale:

- It is a pure "where-we-compute, not what-we-compute" change against the existing
  numerics, so it has an exact parity oracle (6). Coupling in a new (lossy) cache format
  would forfeit that oracle.
- It is the apples-to-apples match to the benchmark: llama.cpp / LM Studio run an f16 KV
  cache with flash attention by default; they do not quantize the KV cache. Chasing fused
  FP4-KV first would over-engineer relative to the thing being measured against.

The v2 draft is marked SUPERSEDED in place and retained only as a reference sketch for
the WMMA (5.3) and quantized-KV (5.4) iterations.

## 4. Design (Iteration 1)

### 4.1 Data layout and index arithmetic

All buffers are row-major BF16. `HS` is the per-head dimension of this op instance.
This path serves the **global** layers only, so for Gemma 4 `HS == global_head_dim == 512`
(NOT the sliding `head_dim` of 256 -- sliding layers are bounded and take the cuBLASLt
path); Llama is 128. `GS = NH / NKV` (Gemma global: `NH=16, NKV=1`). The head dim must be
a multiple of 32 and `<= 512` (9).

| Buffer | Shape | Element `(b, t, nh, d)` / `(b, kv, p, d)` linear index |
|---|---|---|
| `Q` (projection output) | `[B, chunk, NH*HS]` | `((b*chunk + t)*NH + nh)*HS + d` |
| `Y` (attention output) | `[B, chunk, NH*HS]` | `((b*chunk + t)*NH + nh)*HS + d` |
| `K`, `V` (compact cache) | `[B, NKV, cache_capacity, HS]` | `((b*NKV + kv)*cache_capacity + p)*HS + d` |

Q and Y share the identical index expression: a query row is read and its result written
at the same offset. The KV head for query head `nh` is `head_kv = nh / GS`. For the
unbounded cache the physical cache row equals the absolute position (`p == kv_pos`, no ring
wrap), so no slot->position translation is needed in Iteration 1.

Flash reads Q directly from the projection output and writes Y directly to the op output;
it uses **neither** `permute_q_compact` (step 2) **nor** `prefill_unpermute_output`
(step 6), and allocates none of `preatt`/`att`/`v_out`.

### 4.2 Online-softmax recurrence

For query row `(b, nh, t)` with absolute position `abs_t = position_offset + t`, attend
over key positions `p in [window_start, abs_t]` (4.4). Maintain running scalars `m`
(row max), `l` (denominator) and vector `o in R^HS` (unnormalized output), initialized
`m = -inf, l = 0, o = 0`. For each key `p` in increasing order:

```
s      = scale * <q, k_p>              // QK score, one scalar
m_new  = max(m, s)
alpha  = exp(m - m_new)                // rescale factor for prior state
w      = exp(s - m_new)                // this key's unnormalized weight
o      = o * alpha + w * v_p           // vector update
l      = l * alpha + w
m      = m_new
```

After the last key, `y = o / l`. This is the standard FlashAttention identity: it computes
`sum_p softmax_p(s) * v_p` in a single streaming pass without materializing the score row
or the softmax row. Causal attention always includes the key at `p == abs_t`, so `l > 0`.

### 4.3 Kernel execution model

`gqa_flash_prefill_bf16_kernel` (`Gqa.Flash.Bf16.cu`). One **warp** owns one query row
`(b, nh, t)`; the flat warp index decomposes as `t = w % chunk`, `nh = (w / chunk) % NH`,
`b = w / (chunk*NH)`. The head dimension is **striped across the 32 lanes**: lane `L`
owns dims `{ L, L+32, ..., L + 32*(HS/32 - 1) }`, held in per-lane register arrays
`q_reg[HS/32]` and `accum[HS/32]`. Striping (rather than contiguous chunks) makes all 32
lanes touch a contiguous 32-element span of a K/V row at each step -> coalesced loads.

Per key position: each lane forms its partial dot product over its owned dims; a butterfly
warp reduction (`__shfl_xor_sync`) gives every lane the full scalar `s`; each lane then
applies the scalar recurrence (4.2) to its own `accum` stripe. All of `m`, `l`, `o` live
in registers -- **no shared memory, no global score traffic**. Block = 4 warps (128
threads); grid = `ceil_div(B*NH*chunk, 4)`.

This warp-per-row, register-resident shape is the Iteration 1 baseline chosen for
correctness and zero-workspace, not peak throughput; the tensor-core variant is 5.3.

### 4.4 Masking

- **Global causal** (`window <= 0`): `window_start = 0`, keys `[0, abs_t]`.
- **Sliding window** (`window > 0`, unbounded addressing):
  `window_start = max(0, abs_t - window + 1)`. The kernel implements this general form,
  but Iteration 1 only routes the **unbounded** cache here, where global layers carry
  `window == 0`. Sliding-window layers use the bounded ring cache, whose column-j-is-a-
  ring-slot masking (`prefill_softmax_ring_bf16_kernel`, absolute position
  `p = end - ((r - j + capacity) % capacity)`) is a distinct kernel -- Iteration 2 (5).

### 4.5 Scale and numerics

`scale` is the config-derived `attention_scale_` (`1/sqrt(HS)` for Llama; `1.0` for Gemma,
where QK-norm controls magnitude). It is **passed in, never recomputed** in the kernel
(the v2 drafts hardcode `1/sqrt(HS)`, which is wrong for Gemma). There is **no
attention-logit softcap**: Gemma's only softcap is the final-logit cap at the sampler;
attention is plain masked scaled-dot-product, and QK-norm is applied upstream of this op.
All accumulation (`s`, `o`, `l`) is FP32; BF16 is widened on load and narrowed only on the
final store, matching `prefill_softmax_bf16_kernel`.

## 5. Iteration ladder

### 5.0 Measured: Iteration 1 is memory-bound (do not ship as default)

Nsight attribution of an 8192-token Gemma 12B FP4 prefill (RTX 4070, 2026-07-10):
`gqa_flash_prefill_bf16_kernel` is **74.1% of prefill GPU time** (16.3 s of ~22 s, 128
instances -- 8 global layers x 16 chunks -- averaging 127 ms each), versus ~26% for
everything else combined (the FP4 linear GEMMs ~15%, the sliding-layer softmax ~4%, RoPE/
GeGLU/RmsNorm the rest). The Iteration 1 kernel is correct but **~100x too slow**: it has
no shared-memory K/V tiling, so every query-row warp re-streams all of K and V from global
memory. Per instance that is ~8192 warps (512 queries x 16 heads) each reading a ~16 MB
K/V working set -> ~100+ GB of redundant global traffic per call, ~450 GB/s -> ~the 127 ms
observed. So Iteration 1 is a **prefill regression** vs the cuBLASLt tensor-core path, and
it has not banked the memory win (7 still allocated). `use_flash_prefill_` therefore
**defaults to false**; flash stays behind the runtime toggle until 5.2 lands.

### 5.1 BF16 unbounded/global prefill (implemented)

Correctness foundation: parity oracle (10.1) + model parity (10.1) green. Not the default
(5.0). This is the reference the tiled/WMMA kernels validate against.

### 5.2 Shared-memory K/V tiling (mandatory next -- the actual flash algorithm)

The redundant-global-read defect (5.0) is the whole point flash-attention's SRAM tiling
exists to fix, and Iteration 1 skipped it. A thread block owns a tile of `Br` query rows
and streams `Bc`-key tiles of K (then V) into shared memory, so K/V are read from global
**once per query tile**, not once per query row -- cutting the ~100+ GB traffic by ~`Br`.
The online-softmax state (4.2) accumulates across key tiles per query row. This is what
makes flash stop being a regression; it is the prerequisite for every iteration below, not
an optional optimization.

### 5.3 Tensor-core / WMMA throughput

On top of 5.2, run the QK and AV tile GEMMs on tensor cores. The `.Wmma.cu` draft is a
non-working sketch (single-slot K/V indexing; no cross-N-tile online softmax rescale).
Design constraint: an `HS = 512`, `BLOCK_M = 64` FP32 accumulator tile far exceeds the
48 KB default shared-memory limit -> opt-in to >48 KB (`cudaFuncAttributeMaxDynamicShared-
MemorySize`, <=99-100 KB on Ada) and/or tile the head dimension.

### 5.4 Bounded sliding-window ring

Reproduces `prefill_softmax_ring_bf16_kernel` masking on the ring cache (5/6 of Gemma's
layers). Orthogonal to 5.2/5.3 -- it reuses the tiled kernel with the ring slot->absolute
mask. Deferred behind the tiling because a tiled global kernel is the bigger, prerequisite
win.

### 5.5 Quantized KV (FP8/FP4)

Dequant-on-load, gated on a KvCache quantization policy wired into the op. Where the v2
draft's nibble-unpack belongs; reuses the tiled structure with a dequant step on each K/V
tile load, validated against the BF16 kernel as oracle.

### 5.6 Reclaim the transient workspace (memory win)

Skip the `preatt`/`att`/`v_out`/`q_permute` allocation on the flash-only path so the memory
saving is physically banked (Iteration 1 leaves them allocated, so the 40960 memory-
pressure bump remains). Interacts with the still-cuBLASLt bounded/FP32 paths sharing the
`LlamaTransformer` workspace, so it lands once flash covers those.

Decode stays on the cuBLASLt path (weight-bandwidth-bound; ~8% context tax per
`GqaAttentionExtent.md`, section 6). A flash-decode kernel is optional and later.

### 5.5 Platform targets (Iterations 3-4 reference)

| | Ada (RTX 4070, sm_89) | Blackwell (RTX 5060 Ti / 5070 Ti, sm_120) |
|---|---|---|
| Shared memory ceiling | ~100 KB per SM (opt-in) | strict ~99 KB hard max per block |
| L2 | 32 MB | ~64 MB |
| Suggested tiling | `BLOCK_M=64, BLOCK_N=64` | asymmetric `BLOCK_M=64, BLOCK_N=32` (keeps smem under the cap) |
| 4-bit (Iteration 4) | emulated: bitwise nibble unpack + scalar dequant | native FP4 (E2M1) via 5th-gen tensor cores |
| Prefill chunk headroom | 1024-2048 | up to 4096 (16 GB) |

Build flags: `-arch=sm_89` (Ada), `-arch=sm_120`/`sm_121` (Blackwell); CUDA 13.0+ for the
Blackwell mappings. These figures are targets to validate on hardware, not measured
results.

## 6. Correctness / parity invariant

The fused kernel must reproduce the current path -- QK (`scale = attention_scale_`) ->
`prefill_softmax_bf16_kernel` -> AV GEMM -- within `atol = 1e-2`.

Argument: for a fixed query row, both paths (a) attend the identical key set
`[window_start, min(abs_t, attended_len-1)]`, (b) apply the identical `scale`, and (c)
accumulate in FP32. The reference zeros the non-attended columns and multiplies them by
zero in the AV GEMM; flash never visits them -- the contributed values are identical. The
only difference is summation order (streaming online-softmax vs materialized max/exp/
normalize), a floating-point reassociation bounded by `atol`. This makes the parity test
mandatory and exact-in-intent, exactly as `GqaAttentionExtent.md` section 5 warns for the
extent/stride class of bug.

## 7. Memory and launch accounting

Per prefill chunk, unbounded cache, shared workspace sized to the max over GQA layers:

| Buffer | cuBLASLt path | Flash path |
|---|---|---|
| `q_permute` `[B, NH, chunk, HS]` | allocated | unused |
| `preatt` `[B, NH, chunk, cache_capacity]` | allocated (the throttle) | **none** |
| `att` `[B, NH, chunk, cache_capacity]` | allocated | **none** |
| `v_out` `[B, NH, chunk, HS]` | allocated | unused |
| device kernel launches / layer / chunk | 6 | **2** (`kvcache_write_kv` + fused) |

The `preatt`/`att` term is `O(B x NH x chunk x cache_capacity)`; at `chunk = 512`,
`cache_capacity = 40960` it is what forces the heuristic down to `chunk = 256`. Flash
removes it entirely, decoupling chunk size from the VRAM budget. (Iteration 1 leaves the
buffers allocated-but-unused; physically reclaiming them is a follow-up once flash is the
sole prefill path.)

## 8. Integration and selection

Flash replaces steps 2-6 of `prefill_optimized`; step 1 (`kvcache_write_kv`) is shared.
Selection is a compile-time `if constexpr ( !kBounded && std::is_same_v<NativeType,
nv_bfloat16> )` gate immediately after the cache write -- so the bounded ring and FP32
never instantiate the flash symbol -- with a **runtime** `use_flash_prefill_` member
inside it choosing flash vs cuBLASLt. The member (**default false**, since the Iteration 1
kernel is a regression -- 5.0) is set via `setUseFlashPrefill()`, which lets a single test
process run both paths back-to-back and diff them (10.1). It was deliberately made runtime
rather than a compile-time constant so that direct flash-vs-cuBLASLt parity is testable in
one build.

## 9. Constraints and preconditions

- `HS % 32 == 0` and `HS <= 512` (per-lane register arrays sized `HS/32 <= 16`). The
  ceiling is load-bearing: this path serves the **global** layers, and Gemma 4's
  `global_head_dim` is **512** (the sliding `head_dim` of 256 belongs to the bounded
  layers, which do not reach this kernel). Llama 128 qualifies. The launcher enforces this
  with a throwing guard -- a violation is silent stack corruption, not a wrong answer, so
  it must fail loud (this was the Iteration 1 parity-failure root cause: the cap was
  wrongly set to 256).
- Unbounded cache only (`cache_capacity == T`, no ring wrap): physical row == absolute
  position.
- `kvcache_write_kv` must complete before the fused kernel reads the cache (same stream,
  ordered -- satisfied by the launch order).

## 10. Validation protocol

0. **Op-level flash-vs-cuBLASLt oracle** (`CudaGqaOp.Cuda.cpp`,
   `CudaGqaFlashPrefillParity.FlashMatchesCublasLt_GemmaGlobalConfig`). Runs the same
   unbounded BF16 op twice via `setUseFlashPrefill(true/false)` at the exact Gemma global
   geometry (HS=512, NKV=1, window=0, multi-chunk with position_offset>0) and diffs the
   outputs within `kGFlashAtol`. This is the fast, isolated regression guard the older
   bounded-vs-unbounded oracles could not provide (they use the unbounded op as their
   reference, only at window 8); the Iteration 1 HS-cap bug proved it was needed.
1. **Token-for-token model parity** vs `use_flash_prefill_ = false`
   (`GemmaModelParityCudaTests.GreedyDecode`), on Gemma 12B. The bounded-ring
   `CudaGqaOpTests` prefill oracles (single-chunk, multi-chunk, partial-final-chunk,
   prefill-then-decode) stay green. This is the end-to-end safety net (6).
2. **Tax-gone sweep** (fixed 512-token prompt, varying ctx): prefill time flat across ctx
   (contrast the `GqaAttentionExtent.md` section 1 table).
3. **Prefill attribution + llama.cpp delta** at 8K/32K: split prefill into attention /
   linear GEMM / launch overhead (bounds what flash can buy, 1.1), and report TTFT vs
   llama.cpp same-ctx, same-prompt, flash-on. Accept that weight quant differs (Mila FP4
   E2M1 vs GGUF Q4_K_M).

## 11. Risks and open questions

- **Warp-per-row occupancy at small `chunk`.** The last prefill chunk and short prompts
  produce few warps; throughput there is not the design point (5.3 addresses peak).
- **BF16 dot-product accumulation order** differs from cuBLASLt's tiling; parity relies on
  `atol`, not bitwise equality. If an oracle needs tighter tolerance, that is a signal, not
  a pass.
- **Reclaiming the unused workspace** interacts with the shared `LlamaTransformer` buffer
  sizing and the still-cuBLASLt bounded/FP32 paths; deferred until flash covers them.

## 12. Relationship to GqaAttentionExtent.md

`GqaAttentionExtent.md` is the interim, cuBLASLt-resident causal-triangular fix (attended
length `L` decoupled from physical stride `T`). Flash subsumes it: the fused kernel is
inherently causal-triangular and allocation-independent **and** removes the score
materialization the extent change left in place. When Iteration 2 lands the bounded ring,
the unbounded and bounded prefill kernels are both flash, the extent plumbing (per-`L`
plan cache, `attended_len` kernel argument) has no remaining caller, and
`GqaAttentionExtent.md` retires with those plans.
