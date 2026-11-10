# Metal: Decode Attention

**Status:** agreed 2026-11-05 (section 6) and built (section 7). Carried to `Metal/MetalBackend.md` section 4.4. On agreement the decisions are carried to
`Metal/MetalBackend.md` section 4.4, which owns decode.

**Question:** Phase 4 measured attention's decode at about 22 times llama.cpp's cost for each key in the cache
(`MetalFirstModel.md` section 5). What replaces it?

## 1. What is wrong today

`GroupedQueryAttention.metal`, `gqa_decode_softmax` then `gqa_decode_values`, after the cache write:

- **24 SIMD-groups on the whole GPU** for Llama 3.2 3B's scores: one per query head. Each lane walks whole 256-byte key rows
  by itself, so no two lanes read the same row and no read combines.
- **Each query head reads its KV head's keys again.** Three heads share each KV head in Llama 3.2 3B, four in 3.1 8B.
- **Scores pass through memory as FP16**, written by the first kernel and read twice, then the weights read again by the
  second.

Measured on the model, decode after 4096 adds 74 ms a token to Llama 3.2 3B, linear in depth. The probe below gives the
same, 75.4 ms for 28 layers at 4096, so it measures the cost the model pays.

## 2. The design

Two passes, after the cache write, which does not change: still three dispatches a layer. The shape is MLX's two-pass decode
(`ml-explore/mlx`, `mlx/backend/metal/kernels/sdpa_vector.h` and `mlx/backend/metal/scaled_dot_product_attention.cpp`,
read 2026-11-05, MIT licence; ideas taken, no code), and the split of the keys with a log-sum-exp merge is Flash-Decoding's
(Tri Dao, Daniel Haziza, Francisco Massa, Grigory Sizov, *Flash-Decoding for long-context inference*, Stanford CRFM, 2023).
The kernels are Mila's own.

- **Partial pass.** A grid of 32 blocks by every KV head of every batch. A threadgroup is one SIMD-group for each query
  head of its KV head, so the heads of a group read the same rows together. A lane holds head size / 32 channels of the
  query and of its running output, so 32 lanes read one row whole and a dot product is a `simd_sum`. Block b takes keys
  b, b + 32, b + 64 up to the position, which spreads every depth over every block. Softmax runs online in 32 bits: the
  largest score so far, the sum, the output rescaled as the largest grows. Each block writes its output, largest score
  and sum, in FP32.
- **Merge pass.** One SIMD-group per query head: the blocks' largest score, then each block's output and sum rescaled to
  it, the output divided by the sum and written in the run's precision.
- **The decode rule holds** (`CLAUDE.md`, `DecodeGraph.md`): the grid is a constant of the operation, the position is read
  from memory, and how many keys a block takes follows from the position inside the kernel. A block with no keys writes
  a sum of zero and drops out of the merge.
- **A window** (Gemma's sliding layers) starts the keys at the window's first and interleaves from there.
- **Head sizes** up to Gemma 4's 512. The sizes the models use, 64, 128, 256 and 512, have kernels of their own, 2 to 16
  channels a lane in a contiguous run; any other size runs a slower kernel (section 7).
- **FP16 and BF16** are one template. No score is rounded to the run's precision before the merge; today every score is.
- **Scratch.** The partials are 32 x heads x (head size + 2) FP32 values a batch, 0.4 MB for Llama 3.2 3B and 2.1 MB for a
  head of 512 at 32 heads (arithmetic). They come from the execution context's scratch, asked for by the operation as
  Linear's padding does, and fetched when the step is encoded, never kept. Dispatches on one encoder run in order, so a
  layer's merge reads its own partials before the next layer writes them.
- **Not changed:** prefill, the cache write, and several tokens in one decode, which stay Phase 6's.

**Why 32 blocks, fixed.** Measured below over three shapes and seven depths: 32 is on average within 11% of the best
count for each case, and 27% at worst (a head of 256 at depth 512). From depth 4096 it is the best on Llama 3.2 3B.
Choosing the count from the depth, a grid of 256 of which the kernel uses as many as the depth needs, was tried twelve
ways and beat the fixed count nowhere. A count per shape would gain up to 25% on Llama 3.1 8B at 32K. The question of how
to choose a split count for decode with few KV heads is open on CUDA too (*Sequence-Aware Split Heuristic to Mitigate SM
Underutilization in FlashAttention-3 Low-Head-Count Decoding*, arXiv 2604.00028, abstract read only).

## 3. Evidence

`D:\Claude\metal\attention\decode_probe.cpp`, metal-cpp outside the library, on the M1 Pro, 2026-11-05, on AC. FP16. Every
layer its own cache of 32768 positions filled with values in [-1, 1). GPU time from the command buffer, all layers of a
step in one, median of 7. Each result checked against a double-precision reference for layer 0, every head; the same
comparison against the next head's reference must miss, and missed every time at 0.8 to 1.05. Floor: K and V read once at
180 GB/s, Phase 2's rate for Linear. Logs `probe1.log` (block counts, both orders of keys), `probe2.log` (adaptive counts).

Milliseconds for a whole decode step's attention:

| Shape (heads, KV heads, head, layers) | Depth | Floor | Today | Two-pass, 32 blocks | Faster |
|---|---|---|---|---|---|
| Llama 3.2 3B (24, 8, 128, 28) | 512 | 0.33 | 8.83 | 0.86 | 10.2x |
| | 4096 | 2.61 | 75.5 | 3.17 | 23.8x |
| | 32768 | 20.9 | 636 | 20.6 | 30.9x |
| Llama 3.1 8B (32, 8, 128, 32) | 512 | 0.37 | 11.1 | 1.19 | 9.3x |
| | 4096 | 2.98 | 106 | 5.26 | 20.2x |
| | 32768 | 23.9 | 862 | 36.6 | 23.5x |
| Head 256 (16, 8, 256, 14) | 512 | 0.33 | 8.02 | 0.68 | 11.8x |
| | 4096 | 2.61 | 68.2 | 3.27 | 20.9x |
| | 32768 | 20.9 | 560 | 23.8 | 23.6x |

The two-pass values for 32 blocks are read from `probe2.log`'s rows for that count. Relative L2 against double precision:
2.0e-4 to 2.1e-4 at every depth, against today's 3.0e-4, rising to 5.9e-4 at 32768.

Interleaved keys beat a contiguous run per block at most points and never lost by much; a contiguous run lost badly at
32768 with few blocks (56.7 ms against 28.2 at 16 blocks on Llama 3.2 3B).

**What this cannot say.** Random keys and values, not a model's peaked scores; one chip; FP16 only; attention alone, not
the model's step around it.

## 4. What it should give the model

Arithmetic, from the probe and section 5 of `MetalFirstModel.md`: Llama 3.2 3B FP16 after 4096 tokens at 125.8 ms a token,
less today's 75.5 and plus 3.2, is about 54 ms, about 18.6 tokens/s against 7.95 now and llama.cpp's 22.4. What remains is
depth 0's gap, about 10 ms a token, which is not attention's.

## 5. Tests and the gate

- The Metal attention tests as they are, against double precision in both precisions, and the block test's prefill
  equal to its decode.
- New: a depth past 32 keys with blocks left empty below it; a window that cuts mid-block; a depth at the cache's end; a
  head of 256 and of 512.
- **Forced failures:** the merge without rescaling; a block that skips its last key; the wrong KV head for a query head;
  the window's first key one late. Each must fail a test.
- **Gate:** Phase 4's HuggingFace gate on Llama 3.2 3B passes in FP16 and BF16 at its bars; `mila-metal-model-timing`'s
  decode after 4096 runs again and the result goes in `MetalFirstModel.md`.

## 6. Choices

Todd, 2026-11-05: continue with the attention work on MLX's path. Within it:

1. The two decode kernels are replaced by the two passes, both precisions, every head size, as section 2.
2. 32 blocks, fixed, until a measurement on a model shows a count per shape pays. **Tried on Gemma 4 12B, 2026-11-09, and not kept:** MLX's count by keys held moved its global layers' decode by under 5%; what held them was each lane's value-at-a-time loads (`MetalRates.md` 7.2, `Metal/MetalBackend.md` 4.4).
3. The partials come from the execution context's scratch, as Linear's padding does.
4. The shared workspace's decode buffers (`preatt_decode`, `att_decode`, from `Compute/Operations/GqaWorkspace.ixx`) are no
   longer read on Metal and are still allocated for every device, about 0.8 MB on Llama 3.2 3B at 8192 (arithmetic). They
   stay: skipping them per device would change code CUDA shares, for under a megabyte.

## 7. Built

2026-11-05, in `GroupedQueryAttention.metal` and `MetalGqaOp.ixx`. The two kernels and the shared workspace's decode buffers
are gone from the operation, which asks the context for its partials as section 2 says.

**Head sizes.** The probe's kernels hold a contiguous run of HS / 32 channels a lane, fixed at compile time, so need a head
size that is a multiple of 32. A form that takes any size, lane l taking channels l, l + 32 and so on with the size read at
run time, was timed in the same probe (`probe3.log`): about three times slower at every depth, 9.9 ms against 3.2 at 4096
on Llama 3.2 3B. So the head sizes the models use, 64, 128, 256 and 512, have their own kernels, and any other size up to
512 runs the slower one; past 512 the build refuses. No model Mila runs uses the slower one; the tests' heads of 13, 24
and 48 do.

**Tests.** Today's `GroupedQueryAttention.Metal.cpp` cases all pass unchanged but one, `AsksForNoScratch`, whose contract
changed: it is now `AsksForScratchOnlyForADecodesPartials`. New: `EveryDepthAroundTheBlockCount` (1 to 34 keys),
`TheHeadSizesTheModelsUseRunTheirOwnKernels` (64 to 512), `EveryGroupingOnTheModelsKernels` (groups of 1, 3, 4 and 8,
and two batches), `AWindowThatCutsABlock` (37), `ADecodeAtTheCachesLastPosition`, `AHeadPastWhatADecodeHoldsIsRefused` and
`MoreQueryHeadsToAKvHeadThanAThreadgroupHoldsAreRefused`. The last is a check the kernel report found missing: a partial
threadgroup holds a SIMD-group per query head of a KV head, and the compiled kernel takes 896 threads at a head of 512 and
832 in the any-size one, so a build now refuses more query heads to a KV head than the kernel takes, where the dispatch
would have failed. On the M1 Pro, with the block tests, 46 of 46 pass in both precisions. The full suite: Metal 65,
`MilaTests` 1482 pass, 1 skip and 1 fail, BF16's log-likelihood as filed; Agents 10, Chat 33, Desktop 14.

**Forced failures**, `D:\Claude\metal\attention\mutate_attention.py` and `mutate_guard.py`: 12 of 12 caught, each failing 2
to 49 tests.

| Change | Failed |
|---|---|
| The merge without rescaling | 45 |
| A block skips its last key | 43 |
| Every query head reads the first group's KV head | 45 |
| The window's first key one late | 5, the window tests |
| The online sum not rescaled | 21 |
| The online output not rescaled | 25 |
| The batch offset dropped | 7, the batch tests |
| The any-size kernel takes channels past the head | 33 |
| The merge's largest score not shared across lanes | 13 |
| The scratch understated | 49 |
| The query not scaled | 44 |
| The threadgroup check removed | 2, the check's own test |

**Compiling.** The library has 79 kernels now, against Phase 3's 63, and compiles in 578 ms on a first load
(`mila-metal-report`), about 400 ms before.

**The gate**, Llama 3.2 3B as Phase 4 ran it (`MetalFirstModel.md` section 1), 2026-11-05, on AC:

| | Worst logit step | Log-likelihood from HF FP32 | Greedy | One position early |
|---|---|---|---|---|
| Bar | 2.185e-2 | within 8.97e-4 | ties only | must miss |
| FP16, before | 1.93e-3 | +6.8e-5 | 32 of 32 | 2.75e-1 |
| **FP16, two passes** | **1.74e-3** | **+6.76e-5** | **32 of 32** | **2.75e-1** |
| BF16, before | 2.15e-2 | +1.25e-3, misses | 32 of 32 | 2.75e-1 |
| **BF16, two passes** | **1.48e-2** | **+1.25e-3, misses** | **32 of 32** | **2.77e-1** |

The logits move closer to HuggingFace in both precisions, BF16's most, since no score is rounded to BF16 before the merge.
BF16's log-likelihood is scored over a prefilled sequence, which this change does not touch, and misses as before
(`Issues/Untriaged.md`).

**On the model**, `mila-metal-model-timing`, the same session. Rates as Phase 4's: three loads of five runs, median of 15.

| Tokens/s | Before | Two passes | llama.cpp b12138 |
|---|---|---|---|
| FP16 decode 128 | 19.4 | 19.6 | 24.3 |
| FP16 decode 128 after 4096 | 7.95 | **18.65** | 22.4 |
| BF16 decode 128 | 19.2 | 19.3 | 24.1 |
| BF16 decode 128 after 4096 | 7.85 | **18.21** | 19.6 |
| FP16 prefill 512 | 555 | 573 | 649 |

FP16 decode by depth, three runs each:

| Depth | 0 | 1024 | 2048 | 4096 | 8000 |
|---|---|---|---|---|---|
| Before, tokens/s | 18.8 | 14.0 | 11.1 | 7.95 | 4.93 |
| Two passes, tokens/s | 19.6 | 19.3 | 19.2 | 18.65 | 17.75 |
| Two passes, ms a token | 51.0 | 51.8 | 52.2 | 53.6 | 56.3 |

**Attention now costs about 0.66 us a token for each key, 24 ns a key in each layer** (arithmetic), against Phase 4's
670 ns before and about 30 ns for llama.cpp. After 4096 Mila decodes at 0.83 of llama.cpp in FP16 and 0.93 in BF16;
what remains is depth 0's gap, which is not attention's (section 4).

**The build.** Xcode's install moved CMake's sysroot to Xcode's SDK while the frameworks `find_library` had cached stayed
under the Command Line Tools', so one compile mixed the two and Apple's own headers failed. A fresh configure fixed it.
Since then, with the Metal Toolchain downloaded, `xcrun metal` compiles a kernel file alone in about 3 s.
