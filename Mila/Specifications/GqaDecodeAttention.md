# GQA Decode Attention on the Tensor Cores

Status: implemented at `0.21.0-dev+20` (`Kernels/Gqa.Decode.Mma.cu`). The CUDA-core kernel it replaces
(`Kernels/Gqa.Decode.Bf16.cu`) is retired from the build and kept in-tree. Scope: the fused decode path of
`CudaGqaOp`, both caches (BF16, FP8 `PerTokenKvFp8`), unbounded and ring. The prefill sibling is
`GqaFlashAttention.md`; the replay contract every decode op keeps is `DecodeGraph.md`.

This is the design of record; where the code and this document disagree, this document is correct and the code is
the defect.

---

## 1. Why

After decode replay (`DecodeGraph.md` section 7), attention over a long cache was the only generation gap left against
llama.cpp on the same weights (then a `BACKLOG.md` item; since `+20` parked in `Mila/Issues/Future.md`). The measurement that
located it, one decode call at depth, DRAM-resident, replayed as a graph (`CudaGqaDecodeRate`, section 5):

| Old kernel, RTX 5060 Ti | 8K | 32K |
|---|---|---|
| Llama 3.1 8B (HS 128, group 4) | 109 us, 308 GB/s | 414 us, 324 GB/s |
| Llama 3.1 8B, FP8 cache | 86 us, 201 GB/s | 317 us, 219 GB/s |
| Gemma 4 12B global (HS 512, group 16) | 118 us, 142 GB/s | 361 us, 186 GB/s |

The card's matvecs stream at about 415 GB/s. Nsight Compute on one launch at 8K: Llama's kernel issued on 52% of
cycles with DRAM at 78% of peak; Gemma's on 67% with DRAM at 37%, one 512-thread block per SM (registers), so its 128
blocks ran in four waves. Both were bound by instruction issue, not bandwidth. The kernel gave each warp one query row
and did every multiply-add on the CUDA cores: widening each cached value, a five-step shuffle reduction per key per
warp, and every exponential in every lane. At a group of 16 each cached value was widened and multiplied 16 times.

## 2. Design

**The query-head group as the MMA rows.** A block owns one (KV head, split, batch). Its group's query rows, up to 16,
fill the rows of one `m16n8k16` BF16 tile; rows past the group are zero. QK and PV run on the tensor cores with FP32
accumulation, following the packed prefill's fragment layout (`Gqa.Flash.Packed.cu`): Q straight into A fragments,
K through `ldmatrix`, the score accumulators repacked into PV's A fragments in registers, V through
`ldmatrix.trans`. A KV head is read once for its whole group, and the per-key cost no longer scales with the group.

**Warps.** Head dimensions split into 128-wide slices, one warp each (1, 2 or 4); the key tile splits into groups of
16 keys (one PV k-step), each carrying its own online-softmax state. With more than one slice, the slices' partial
scores meet in shared memory behind a named barrier of that key group, summed in slice order. Every head size runs
four warps: 4 key groups at HS 128 (64-key tiles), 2 at 256 (32), 1 at 512 (16). The key groups merge through shared
memory at the end of the block. Measured (section 5): one or two key groups read short bands a few percent faster
and long bands a few percent slower, so four warps stays.

**Staging.** K and V tiles double-buffer through `cp.async` into padded rows (`ldmatrix` conflict-free). A key outside
the split is staged as zeros and scored -inf, so no unwritten row is read and no 0 x NaN reaches PV. Cache row is
`position % capacity`, the identity for the unbounded cache.

**Splits.** `decodeSplits( band, target, tile )` is one function for the host and both kernels (the replay rule of
`DecodeGraph.md` 4.2): at most `target` splits, none shorter than 64 keys or one tile, each a whole number of tiles,
none empty. The target fills whole waves: the least split count for which KV heads x batch x splits is a multiple of
the device's resident blocks (occupancy x SMs, measured once per device and kernel). On the 5060 Ti a block holds one
SM, so Llama (8 KV heads) takes 9 splits = 72 blocks = two full waves; Gemma's global layer (1 KV head) 36. The grid is
sized from the longest band the op can hold; blocks past the live count exit. With one split the block writes Y;
otherwise it writes the unnormalized (O, m, l) partial and the fixup launch merges them. Scratch is unchanged: B x NH x
128 x (HS + 2) floats.

**FP8 cache.** E4M3 codes and their scales double-buffer; each tile widens, unscaled and sixteen codes a thread a
step, into one BF16 stage (two would pass the 99 KB a block holds). The next tile's load is issued before the widening,
so it stays in flight through it. Each key's K scale multiplies its score, each key's V scale its probability as P is
packed, and l sums the unscaled probabilities (`Quantization.md`, Part III). On lossless values the FP8 path equals the
BF16 path bit for bit (`CudaGqaFp8CacheTests.LosslessValues_MatchTheBf16CacheBitForBit`).

## 3. Numerics

P is rounded to BF16 before PV, as in every flash kernel here; the old kernel kept it in FP32. Exponentials are
`__expf`. Both change decode output in the last bits, so the hand-run references of `DecodeReference.Cuda.cpp`
(`DecodeGraph.md` gate A1, gitignored and per card) must be rewritten from `+20`. The parity tests against the cuBLASLt
pipeline (`CudaGqaDecodeParity`, 3e-2) and the FP8 exactness tests pass unchanged.

## 4. Requirements

Compute capability 8.0 or later (the library's floor), head size 128, 256 or 512, group size 1 to 16. Other
geometries take the cuBLASLt pipeline, as before.

## 5. Result

`CudaGqaOp.Cuda.cpp`, `CudaGqaDecodeRate.DISABLED_*`: one decode call (cache write, attention, merge) at a fixed
position after a prefill to that depth, replayed as one graph, with enough ops decoded in turn that their bands exceed
L2 several times over. RTX 5060 Ti, both kernels from one build, microseconds a call:

| | 1K | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|
| Llama 3.1 8B, old | 21.1 | 58.4 | 109.0 | 214.5 | 414.0 |
| Llama 3.1 8B, new | 17.7 | 46.7 | 86.8 | 166.9 | 325.4 (412 GB/s) |
| Llama 3.1 8B FP8 cache, old | 20.4 | 48.3 | 86.1 | 163.8 | 316.6 |
| Llama 3.1 8B FP8 cache, new | 13.3 | 30.2 | 50.5 | 91.8 | 175.8 (394 GB/s) |
| Qwen 3.8 full attention, old | 18.9 | 63.8 | 117.2 | 227.6 | 437.0 |
| Qwen 3.8 full attention, new | 17.1 | 47.5 | 87.7 | 167.8 | 326.0 |
| Gemma 4 12B global, old | 26.9 | 56.6 | 118.3 | 197.9 | 360.9 |
| Gemma 4 12B global, new | 15.5 | 32.1 | 52.4 | 93.5 | 175.4 |

Gemma's sliding layer (window 1024) is flat: 27.3 us old, 28.6 new (27.4 with one key group). Key groups, same
build: Llama at 32K 325.4 / 326.0 / 332.3 us with 4 / 2 / 1; at 1K 17.7 / 16.3 / 16.1.

Generation through `ProfileModel` (`generate()`, 129 tokens, three runs, spread under 0.1), tokens a second, old ->
new from one build; llama.cpp b11216 on the same card from the `+19` comparison run (`DecodeGraph.md` 7.2):

| Depth | Llama 3.1 8B Q4_0 | llama.cpp | Gemma 4 12B Q4_0 | llama.cpp |
|---|---|---|---|---|
| 0 | 80.3 -> 80.4 | 79 | 53.5 -> 53.7 | 54 |
| 8K | 63.3 -> 66.5 | 65 | 49.1 -> 50.4 | 51 |
| 32K | 39.1 -> 44.2 | 39 | 44.9 -> 48.1 | 48 |

## 6. What remains

At a short band the call is dominated by fixed cost. nsys of the Gemma sliding call (28.5 us): attention 24.6, the
fixup 2.0, the cache write 1.2, gaps 0.7. Gemma pays it on 48 layers a token. Two fusions would remove the two small
launches:

1. **The merge into the attention kernel.** The last split block to finish merges, found by an atomic counter it
   resets for the next replay. The counter needs device memory that persists across calls and starts at zero, which
   the shared scratch is not; it would be a few bytes of op state, which `getStateMemorySize` and the footprint
   tests account exactly. A design decision, not taken.
2. **The cache write into the attention kernel.** Only the split holding the current position reads its row, so that
   block could take the new key and value from the projection output and write the row itself. For the FP8 cache it
   would also quantize the row.

Gemma's depth-0 row also carries the output-head difference `DecodeGraph.md` section 8 states: FP8 in Mila's package,
Q6_K in Google's GGUF.
