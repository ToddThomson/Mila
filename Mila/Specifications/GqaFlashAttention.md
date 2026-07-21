# GQA FlashAttention: Fused Prefill on the Compact KV Cache

Status: Iteration 2 (shared-memory K/V tiling, 5.2) implemented and **profiled -- it does
NOT fix the regression** (5.2.1). Root cause of Iteration 1 was misattributed: the scalar
warp-per-row kernel is **LSU / shared-memory-instruction bound** (one load per FMA), not
global-DRAM-traffic bound, so tiling K/V into shared memory moved the loads without cutting
their count and bought only ~1.2x (127 -> 103 ms/instance) -- still ~3x slower than
cuBLASLt on the full prefill. `use_flash_prefill_` stays **false**. The real fix is tensor
cores (5.3): only an MMA instruction mix (hundreds of FLOPs per instruction) escapes the
1-FLOP-per-load ceiling that binds the entire scalar family. Scope: `CudaGqaOp` prefill
path. Subsumes and retires `GqaAttentionExtent.md` only once flash is measured the faster
path across the bounded + global layers -- not yet.

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

### 5.0 Measured: Iteration 1 is a regression (do not ship as default)

CORRECTION (see 5.2.1): the "memory-bound / redundant-global-traffic" diagnosis below was
**wrong**. Nsight Compute later showed the 16 MB K/V working set lives in L2 (99.7% hit),
so Iteration 1 was never DRAM-traffic bound -- it is LSU/shared-memory-instruction bound
(one load per FMA). The regression conclusion and the default-false decision stand; only the
*why* changed. The original text is kept for the record.

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
**defaults to false**; flash stays behind the runtime toggle. Iteration 2 (5.2) is the
tiled kernel that removes this regression; the default flips only once 5.2 is profiled
faster than cuBLASLt on hardware.

### 5.1 BF16 unbounded/global prefill (implemented)

Correctness foundation: parity oracle (10.1) + model parity (10.1) green. Not the default
(5.0). This is the reference the tiled/WMMA kernels validate against.

### 5.2 Shared-memory K/V tiling (implemented, but profiled a non-fix -- see 5.2.1)

The redundant-global-read defect (5.0) is the whole point flash-attention's SRAM tiling
exists to fix, and Iteration 1 skipped it. A thread block owns a tile of `Br` query rows
and streams `Bc`-key tiles of K (then V) into shared memory, so K/V are read from global
**once per query tile**, not once per query row -- cutting the ~100+ GB traffic by ~`Br`.
The online-softmax state (4.2) accumulates across key tiles per query row. This is what
makes flash stop being a regression; it is the prerequisite for every iteration below, not
an optional optimization.

**Iteration 2 realization (`Gqa.Flash.Bf16.cu`).** The warp-per-row, lane-striped-HS
layout of Iteration 1 (4.3) is preserved unchanged -- so the QK dot product, the butterfly
reduction, and the V accumulation are byte-for-byte the same code -- and the *only* change
is that K and V are read from a shared-memory tile instead of directly from global. That
keeps the parity oracle (10.0) exact.

- **Block shape `Br = 16` query rows x `Bc = 32` key tile.** One block owns
  `(b, nh, query-tile of 16 tokens)`; the 16 warps each own one query token in the tile and
  share the block's single `(b, nh)` KV head. Grid = `B * NH * ceil(chunk / 16)`. Warps
  past a partial final tile carry no row but still drive the cooperative loads and
  `__syncthreads` (never early-returned) so the block cannot deadlock.
- **`Br = 16` is the ceiling for this layout, not a free parameter.** Traffic drops by
  `Br`, so bigger is better -- but warp-per-row holds HS in registers
  (`q_reg[HS/32] + accum[HS/32] + p_tile[Bc]` ~= 64 floats/thread at HS=512). `Br = 32`
  (1024 threads) overflows the 64 K register file and will not launch; `Br = 16` (512
  threads, ~84 regs) launches at 1 block/SM and already converts the ~100 GB redundant
  traffic to ~6 GB (~16x). Pushing past this needs the register-free HS accumulation of the
  tensor-core layout (5.3), not a larger warp-per-row block.
- **Shared-memory budget: a single reused `Bc x HS` BF16 tile = `32 x 512 x 2 = 32 KB`**,
  under the 48 KB default -- **no `cudaFuncAttributeMaxDynamicSharedMemorySize` opt-in
  needed here**. That opt-in (Ada allows ~99-100 KB) is 5.3's concern, where the FP32
  accumulator tile forces it. `Bc` is a tuning knob: it changes sync/load granularity and
  register pressure (`p_tile[Bc]`), not total traffic (that is set by `Br`).
- **K-then-V two-pass tile load.** Pass 1 fills the tile from K and scores all `Bc` keys;
  pass 2 *overwrites* the same buffer from V and accumulates. Reusing one buffer (vs holding
  K and V resident together) halves smem, at the cost of holding the tile's `p_tile[Bc]`
  probabilities in registers between the two passes. The masked keys skip their V
  contribution.
- **One online-softmax rescale per tile (FlashAttention-2), not per key.** Pass 1 scores
  the whole tile into `p_tile`, finds the masked tile max `m_tile`, then merges once:
  `m_new = max(m_i, m_tile)`, `alpha = exp(m_i - m_new)` (forced to 0 on the first
  contributing tile so `m_i = -inf` never produces a `-inf - (-inf)` NaN), rescale
  `accum *= alpha` and `l *= alpha`, set `p_j = exp(s_j - m_new)`, add `l += sum p_j`.
- **Causal tile-skip.** The tile loop runs only to the block's last row's absolute
  position (`block_max_key`); rows with smaller `abs_t` mask the tail per key. On the
  triangular global layer this halves the key traffic.

The default `use_flash_prefill_` stays **false** until 5.2 is profiled faster than cuBLASLt
on hardware (the profiling-before-commit gate that caught the Iteration 1 trap); the tiled
kernel is validated for correctness first via the 10.0 oracle and 10.1 model parity.

### 5.2.1 Measured: tiling is a non-fix -- the scalar kernel is LSU-bound, not traffic-bound

Nsight of an 8192-token Gemma 12B FP4 prefill (RTX 4070, 2026-07-10), flash-on vs the
cuBLASLt baseline, same config:

| | cuBLASLt | Flash Iter 1 (naive) | Flash Iter 2 (tiled) |
|---|---|---|---|
| 8192 prefill wall | **6465 ms** | ~22 s | **19068 ms** |
| `gqa_flash_prefill_bf16_kernel` | -- | ~16.3 s (74%) | 13.1 s (69.7%) |
| per-instance avg (128 inst) | -- | ~127 ms | 103 ms |

Tiling cut the redundant global K/V traffic by ~16x (its whole design goal) yet bought only
**~1.2x**. That single fact falsifies the 5.0/5.2 premise that Iteration 1 was
*global-DRAM-traffic* bound. Nsight Compute on a full-context launch shows why:

- **Local Memory Spilling Requests: 0** (48 reg/thread) -- the per-lane arrays are true
  registers, not spilled. The spill hypothesis was wrong.
- **Achieved occupancy 65%** (register-limited to 66.7%) -- healthy, not the limiter.
- **Mem Busy 87% / Max Bandwidth 82% at only ~10 GB/s actual, L2 hit 99.7%.** The memory
  *pipes* (L1 / shared / LSU) are pegged near saturation while moving almost no useful
  bytes; **Compute (SM) throughput 44%.**

That is the signature of an **LSU / shared-memory-instruction-throughput bound** kernel: the
scalar warp-per-row formulation issues one shared-memory load per FMA (1 FLOP per 2-byte
load), and it is the *instruction issue rate*, not bandwidth, that saturates. At 8K the
16 MB K/V working set already lived in the 32 MB L2 (L2 hit 99.7%), so Iteration 1 was never
DRAM-bound either; tiling moved the loads from L2 to shared memory without changing their
count -- so Iter1 and Iter2 land at the same ~100 ms/instance.

**Conclusion.** The entire scalar warp-per-row family (Iter 1 register-streaming, Iter 2
shared-memory-tiled) is structurally LSU-bound and cannot approach cuBLASLt's tensor-core
GEMMs -- no amount of tiling, occupancy tuning, or spill avoidance escapes the 1-FLOP-per-
load ceiling. The only lever that changes the instruction mix is **tensor cores (5.3)**,
where each MMA does hundreds of FLOPs per load. The Iteration 2 kernel is retained (correct,
parity-green) as the **shared-memory tiling scaffold the WMMA kernel builds on** (5.3 keeps
the block/tile/two-pass/online-softmax structure and swaps the scalar inner loops for MMAs),
not as a shippable path. `use_flash_prefill_` remains false; flash is not yet a win.

### 5.3 Tensor-core / WMMA throughput (mandatory next -- the actual fix per 5.2.1)

This is the iteration that makes flash a win, not an optimization on top of a working one:
5.2.1 proved the scalar inner loop is the ceiling, and only tensor cores lift it. It reuses
5.2's block/tile/two-pass/online-softmax scaffold verbatim and replaces the scalar QK and AV
loops with tile MMAs. On top of 5.2, run the QK and AV tile GEMMs on tensor cores.
`Gqa.Flash.Wmma.cu` now holds the working Stage 1 (single-warp, retired) and Stage 2a
(multi-warp HS-split) kernels; the original `.Wmma.cu` sketch (single-slot K/V indexing; no
cross-N-tile online softmax rescale) was replaced. Design constraint: an `HS = 512`,
`BLOCK_M = 64` FP32 accumulator tile far exceeds the
48 KB default shared-memory limit -> opt-in to >48 KB (`cudaFuncAttributeMaxDynamicShared-
MemorySize`, <=99-100 KB on Ada) and/or tile the head dimension.

### 5.3.1 WMMA kernel design (proposed, pre-implementation -- 2026-07-10)

Concrete layout for the Gemma global config (`HS=512, NKV=1, NH=16, window=0`, BF16). Goes
in `Gqa.Flash.Wmma.cu` (rewriting the non-working sketch), selected alongside the scalar
kernel via the same `cuda_gqa_flash_prefill_bf16` launcher signature (so no cuh/Dispatch/Op
change) and validated against the same `CudaGqaFlashPrefillParity` oracle (10.0).

**The two GEMMs and why HS=512 is the crux.** Per query tile of `Br` rows, per key tile of
`Bc` keys:

- **QK^T**: `S[Br x Bc] = Q[Br x HS] . K[Bc x HS]^T`, contraction over `HS=512`. Accumulator
  `[Br x Bc]` is **small** and transient (one key tile's scores).
- **PV**: `O[Br x HS] += P[Br x Bc] . V[Bc x HS]`, contraction over `Bc`. Accumulator
  `O[Br x HS]` is **large** and **persistent** -- it is the online-softmax output, alive for
  the whole key loop. At `HS=512` this is the entire problem: for `Br=32` it is
  `32 x 512 x 4 = 64 KB` of FP32 state. It cannot sit in one warp's registers
  (`[16 x 512]` alone is 32 accumulator fragments = 256 f32/thread) and does not fit smem
  beside the Q/K/V tiles.

**Layout decision: split HS across the warps (not the query rows).** Standard
FlashAttention-2 splits `Br` across warps and gives each warp the full head dim -- which is
exactly what explodes at `HS=512`. Instead, a block of `W` warps partitions the **head
dimension**: warp `w` owns the `HSt = HS/W` output columns `[w*HSt, (w+1)*HSt)`. Its slice
of the persistent accumulator is `O_w[Br x HSt]` -- e.g. `W=8 -> HSt=64 -> O_w[32 x 64]` =
8 WMMA accumulator fragments = **64 f32/thread**, register-resident. This directly divides
the oversized dimension instead of fighting it.

The cost of that choice: the QK contraction now runs over a dimension (`HS`) that is split
across warps, so QK becomes a **split-K** GEMM.

**Per-phase flow (one key tile):**

1. **Load** K then V into a single reused smem tile (the 5.2 two-pass), Q resident in smem
   for the whole query tile.
2. **QK split-K.** Each warp computes a *partial* `S_w[Br x Bc]` over only its `HSt` slice
   (`HSt/16` k-steps of `mma`), then the `W` partials are summed to the full `S[Br x Bc]`
   through a small smem reduction (`S` is `32x32x4 = 4 KB`; the reduction is cheap next to
   the GEMMs). Orientations: `A=Q` row-major, `B=K` col-major yields `K^T`.
3. **Online softmax in smem.** With full `S` visible to all warps, compute the masked tile
   max, the running-max merge, `alpha`, and `P = exp(S - m_new)` cooperatively; keep
   `m[Br]`, `l[Br]` in smem. Causal/window mask applied here per `(row, key)` exactly as the
   scalar kernel.
4. **PV per-warp HS-slice.** Warp `w` does `O_w += P . V[:, w-slice]` (`Bc/16` k-steps),
   accumulating into its persistent fragments -- no cross-warp reduction (each warp owns
   disjoint output columns).
5. **Rescale** the persistent `O_w` fragments by the per-row `alpha` before step 4's add
   (FA-2 one-rescale-per-tile).

**The load-bearing risk: per-row rescale on a tensor-core accumulator.** The online softmax
multiplies each *row* of `O` by `alpha_row` every tile. The `nvcuda::wmma` API hides which
fragment element maps to which row, so a per-row scale through that API is not cleanly
expressible. This -- not the GEMMs -- is the hard part, and it is exactly what the `.Wmma.cu`
sketch omitted. Resolution: **drop to the `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32` PTX**
(Ada sm_89), whose C/D fragment->(row,col) thread mapping *is* documented, so each thread
knows the rows of its accumulator registers and can apply `alpha_row`. This is how CUTLASS /
FlashAttention-2 do it; the `wmma` convenience API is not sufficient here.

**Bring-up (correctness first, then throughput):**

- **Stage 1 -- DONE + VALIDATED 2026-07-10.** `CudaGqaFlashPrefillParity` green. Simplest
  correct shape: **one warp** per `(b, head, 16-row tile)` (not the W=8 split -- single warp
  is even simpler), Q/K/V and the FP32 `O` accumulator all in smem, QK/PV via the safe
  `wmma` API only (no fragment `.x[]`), online softmax + per-row rescale by trivial smem
  indexing. Slow by design (single warp, `O` streamed through smem with a store+add per
  HS-subtile) but the oracle geometry is tiny, so it only had to be correct. Proves the
  WMMA GEMM orientations, the cross-tile online softmax, and the causal mask.
- **Stage 2a -- multi-warp, still no PTX (IMPLEMENTED 2026-07-11, pending build + oracle).**
  Go to the `W`-warp HS-split (warp `w` owns `HSt=HS/W` output columns) with split-K QK +
  smem `S` reduction, but keep each warp's `O_w` slice in **smem** and rescale it by per-row
  smem indexing. Because the slices are disjoint, the PV/rescale needs no cross-warp sync --
  only the K/V load and the split-K `S` reduction do -- which collapses Stage 1's
  ~64-syncs-per-tile to ~6 and adds `W`-way MMA parallelism. This captures most of the
  throughput **without** the `mma.sync` risk. `Br=16` keeps `O` smem at 32 KB (fits beside Q
  + two-pass K/V under the ~99 KB opt-in). Realized in `Gqa.Flash.Wmma.cu` (rewrites the
  Stage 1 kernel; same `cuda_gqa_flash_prefill_bf16` launcher signature -- no cuh/Dispatch/Op
  change). `W = min(8, HS/16)` reduced to the largest power-of-2 keeping `HSt = HS/W` a
  multiple of 16 (HS=512 -> W=8, HSt=64; block = 256 threads); `Bc=16` single WMMA key tile
  (a tuning knob). Measured smem: 81 KB at HS=512. Gate: `CudaGqaFlashPrefillParity` at
  `atol` (10.0). Profile against cuBLASLt once green; 2a may already win, deferring 2b.
- **Stage 2b -- register-resident `O` + `mma.sync` PTX (IMPLEMENTED 2026-07-11, pending build
  + oracle).** Move `O_w` off smem into `mma.sync.m16n8k16` accumulator registers and apply
  the per-row `alpha` directly on those registers (the documented fragment layout). 2a was
  profiled (below) and does NOT beat cuBLASLt, so 2b is the active work. **Risk-scoped
  realization:** only PV is raw PTX -- QK keeps 2a's proven `wmma` split-K path (its `S`
  accumulator is transient and goes to smem for the softmax regardless, so registers buy it
  nothing), confining the `mma.sync` surface to one GEMM. PV uses
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`; each warp's `O_w[Br x HSt]` is
  `HSt/8` accumulator tiles held as `float o_acc[nt][4]` per thread (32 regs at HS=512), A(P)
  and B(V) operands loaded manually from smem per the documented fragment layout (not
  `ldmatrix`, for reviewability), the per-row `alpha` applied as two scalars per thread
  (`c0,c1` -> row `g`; `c2,c3` -> row `g+8`, `g = lane/4`). Dropping `O` and the PV scratch
  from smem cuts the block footprint **81 KB -> 41 KB** (HS=512), which should lift the
  occupancy cap from 1 -> 2 blocks/SM. Re-validate the oracle (parity exact-in-intent within
  `atol`), then re-profile: expect occupancy and Compute-throughput % up, and the PV
  store->sync->add chain gone.

**Stage 2a measurement (8192 prefill, Gemma 12B FP4, RTX 4070, 2026-07-11).** Oracle green.
A real step but not yet a win: wall 11702 ms (vs cuBLASLt 6465, Iter2 scalar 19068);
`gqa_flash_prefill_wmma_bf16_kernel` = 6.08 s (52% of prefill), 47.5 ms/instance average --
**2.2x faster than the scalar Iter2 (103 ms/inst) with tensor cores confirmed engaged**, but
still ~3.5x heavier than the cuBLASLt global attention it replaces (~1.7 s), so the wall stays
above cuBLASLt. ncu on a heavy (81 ms) instance: **occupancy 16.67%, smem-limited to 1 block/SM**
(83.65 KB dynamic), Compute (SM) 22% / Memory 16% / DRAM 0.9% / L2 94% / 0 spills / 55 regs --
i.e. **latency/stall-bound at low occupancy, a different regime than the scalar family's LSU
ceiling** (that pegged Mem Busy 87%; here both compute and memory sit idle while warps stall on
the dependency chain). Root cause, two coupled effects with one fix: the **32 KB `O[Br x HS]`
FP32 accumulator in shared memory** (a) caps occupancy at 1 block/SM and (b) forces the PV
`store_matrix_sync -> __syncwarp -> smem-add -> __syncwarp` serial chain (low ILP). Stage 2b
removes both -- register-resident `O_w` frees ~32 KB smem (occupancy up) and eliminates the
round-trip/sync chain (ILP up). The measurement vindicates 2b as the lever, for the right reason.

**Budgets and starting config (Ada):** `W=8` warps (256 threads), `Br=32`, `Bc=32`,
`HSt=64`. Smem (two-pass K/V): `Q 32 KB + KV 32 KB + S/P ~8 KB ~= 68-72 KB` -> requires the
`cudaFuncAttributeMaxDynamicSharedMemorySize` >48 KB opt-in (<=99 KB Ada). Registers: ~64
f32/thread for `O_w` + transient fragments + scalars ~= 1 block/SM (smem-limited). Occupancy
is low (~8 warps/SM) but WMMA kernels are designed for it -- MMA issue rate, not occupancy,
carries throughput. **Tunables** for the profiler: `W` (16 -> `HSt=32`, lighter `O_w`, more
warps to hide latency, more split-K partials), `Br` (64 -> 2x reuse, 2x `O_w` registers),
`Bc` (smem vs sync granularity). None of these change total K/V traffic (set by `Br`) or
correctness.

**What it must beat and why it should.** The win over cuBLASLt is *not* the QK/AV FLOPs
(cuBLASLt already runs those on tensor cores) -- it is fusing the softmax so the `[Br x
attended_len]` score matrix never touches global memory (the ~10.3 s / 29% at 32K, 5.2.1 /
the 32K attribution). WMMA brings QK/PV to tensor-core throughput on top; the floor of value
is the removed score-materialization, so unlike 5.2 this is not a pure-regression risk.

**Open decisions to confirm before Stage 2:** (a) accept the >48 KB smem opt-in (yes --
unavoidable at `HS=512`); (b) commit to `mma.sync` PTX for the rescale (yes -- the `wmma`
API cannot express it); (c) starting `(W, Br, Bc)` above, tuned after first parity-green.

### 5.4 Bounded sliding-window ring

Reproduces `prefill_softmax_ring_bf16_kernel` masking on the ring cache (5/6 of Gemma's
layers). Orthogonal to 5.2/5.3 -- it reuses the tiled kernel with the ring slot->absolute
mask. Deferred behind the tiling because a tiled global kernel is the bigger, prerequisite
win.

### 5.5 Quantized KV (FP8/FP4)

Dequant-on-load, gated on a KvCache quantization policy wired into the op. Where the v2
draft's nibble-unpack belongs; reuses the tiled structure with a dequant step on each K/V
tile load, validated against the BF16 kernel as oracle.

### 5.6 Reclaim the transient workspace (memory win) -- IMPLEMENTED 2026-07-11 (partial, Gemma)

**This is the PRIMARY justification for flash, not the ~18% speed** (32K crossover measurement,
5.3.1): flash removes the `O(chunk x T_ctx)` score materialization (`preatt`/`att`), which is
the memory wall that OOMs a 12-16 GB card at long context. The 32K run banked no memory (the
buffers were still allocated), leaving 309 MiB free; the reclaim is what makes the 64K target
fit.

**Realization (partial, Gemma only).** The `preatt`/`att` buffers are shared across all GQA
layers and were sized to the *global* layer's `T_ctx` width. Global attention is now on flash
(Stage 2b) and never touches them, so they shrink to the *sliding* layers' exact need:
`score_width = min(T_ctx, window + prefill_chunk_size - 1)` (matching `CudaGqaOp` bounded
`cache_capacity_`) -- ~1535 vs 65536 at 64K, reclaiming ~1 GB per buffer. This does NOT require
5.4 (bounded-ring flash): the sliding layers stay on cuBLASLt but only need window-bounded width.

**Coupling (the correctness constraint).** The buffer width MUST match the op's flash decision
or the cuBLASLt global path overflows a narrow buffer. Both now derive from one source of truth,
`GemmaTransformer::useFlashPrefillForContext(T_ctx)` (`Gemma.ixx`): flash-on when
`T_ctx >= kGemmaFlashPrefillMinContext` (default 16384) and BF16. The transformer sets it on the
global blocks (`setUseFlashPrefill` passthrough: `GemmaBlock` -> `GroupedQueryAttention` -> op)
AND sizes `preatt`/`att` via `prefillScoreWidth(T_ctx)`. `CudaGqaOp::use_flash_prefill_` default
restored to **false** (safe): a standalone op never flashes into a narrow shared buffer. Below
the threshold cuBLASLt runs (faster, and its full-width score workspace still fits); at/above it
flash runs and the buffer is reclaimed -- best of both. `computeChunkRowCostBytes` also uses
`prefillScoreWidth`, so the chunk heuristic is no longer throttled by `T_ctx` at long context
(chunk-size decoupled from the score span).

**Still open:** decode buffers stay `T_ctx` (chunk=1, small; decode is cuBLASLt). `q_permute`/
`v_out` stay `HS_max` (small; could shrink to the sliding head_dim). LlamaTransformer has no
sliding layers -- all-global, so flash there could reclaim `preatt`/`att` *entirely*, but the
same wiring must be added (follow-up). Full zero-allocation reclaim on a pure-flash model waits
on 5.4.

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
