# Metal Phase 2: the matrix path

Phase 2 of `Specifications/Metal/MetalBackend.md`, on the M1 Pro. It answers two questions: which path does Linear use per
shape, and why did TensorOps run BF16 at half of FP16's time at a decode-like shape? The gate is written before the run.

True as of its date, on one chip. Not a model run, and it says nothing about FP4 or FP8 weights, which are Phase 5's.

**Machine:** MacBook Pro 2021, Apple M1 Pro, 14-core GPU, 16 GB, macOS 27.2, on AC, MSL 4.1, default math mode.

## 1. The gate

**Shapes.** Linear multiplies activations, M rows by K, by weights stored row-major as N outputs by K inputs. The columns are
Llama's, read from its presets in the tests and its published configs, since Llama's operations are Phase 3:

| Name | K | N |
|---|---|---|
| 8B mlp in | 4096 | 14336 |
| 8B mlp out | 14336 | 4096 |
| 8B query | 4096 | 4096 |
| 8B key and value | 4096 | 1024 |
| 8B head | 4096 | 128256 |
| 3B mlp in | 2048 | 8192 |
| 3B mlp out | 8192 | 2048 |
| 3B query | 2048 | 2048 |
| 3B key and value | 2048 | 1024 |

**Rows.** M = 2048, the largest prefill chunk; 128, the smallest; 8, the most a multi-token decode passes; and 1, decode. The
head runs at M = 8 and 1 only.

**Data.** Activations normal with sigma 1 and weights normal with sigma 0.02, rounded to each format on the host. Uniform in
[-1, 1], as the earlier timing used, is run too, for the anomaly.

**The two paths are each tuned before they are compared.**

- TensorOps: tile rows, tile columns and SIMD-groups per threadgroup.
- Own kernels: SIMD-group tiles, a per-column dot product, the order of the grid, and blocking over K, M and N.

Tuning a path stops when two further changes in a row each move its best time at that shape by under 10%, or sooner if it
reaches the other path's time.

**Accuracy bar.** Relative L2 error of 16 sampled rows against a double-precision reference built from that arm's own rounded
inputs. Own FP16 at a shape must reach no more than 2x TensorOps FP16's error at that shape and data. That is the spec's
"accumulates wider, or does not ship". A kernel over the bar is out, whatever its time.

**The choice, per shape, in FP16, the M1's model precision.** The spec keeps own kernels where the result is close, so own
stays unless TensorOps is more than 10% faster at the median. BF16 is timed and checked, and decides nothing here, since it is
emulated on this chip.

**FP16 inputs with FP32 output.** Reported beside FP16 output, at normal data and at small data (activations sigma 0.01, weights
sigma 0.001), for its cost and its error. Whether a model should use it is for Phase 4.

**The anomaly.** At M = 8, K = N = 8192, on the earlier timing's layout, reproduce first. Then vary one factor at a time: tile
rows, tile columns, SIMD-groups, K and N, M, the data, and the weight layout. It is explained when one factor makes the
ratio flip, in all three runs. If none does, this notebook says so, and no decision rests on it.

**Validity.**

- GPU timestamps, one command buffer per run of an arm, waited on.
- 3 warm-up runs per arm, then 15 rounds with every arm once per round, order rotated each round.
- Three separate process runs. A result differing between them by more than 15% is reported as unstable.
- An arm's timing counts only if its error is reported and meets its bar. A deliberately misaligned output must fail the check.
- Mac on AC, lid open, nothing else running on it.

**What this cannot say.** Real activations and weights are not normal. It is a chip with no M5 accelerators, so a choice here
says nothing about M5, and Phase 7 may make it per family. Every shape is a multiple of the tile, so remainders are Phase 3's.

**Extended 2026-11-05 to every weight format, before its runs (Todd).** Same machine, same rules of validity, and section 2's choice rule: own stays unless a TensorOps path is more than 10% faster.
It covers the formats in `MetalBackend.md` section 1 that a Linear reads, stored as `Lib/Src/Dnn/Quantization/Weight` lays them out: FP8 per channel; FP4 at groups 128 and 64; Q4_0; Q6_K;
2-bit codebook at groups 32 and 64; 3-bit at 64 and 128. Weights are normal with sigma 0.02, encoded by each format's own rule; a codebook's table is a fixed spread, since it is fitted offline. Results are in section 6.

- **Shapes:** 8B mlp in, mlp out and query, and 3B mlp in, at M = 2048, 8 and 1; the head at M = 8 and 1 in FP4 and Q6_K, the formats a tied head uses.
- **Paths:** own lanes in 32 bits reading the weights once for all rows, up to M = 8; TensorOps on FP16 weights expanded into a scratch buffer, a block of outputs at a time, with the expansion in the time; and TensorOps reading FP8 directly, with the row scales applied after.
- **Accuracy bar:** an arm's error against a double-precision reference on the decoded weights must be no more than 2x the best arm's at that shape and format, or it is out.
- **TensorOps' own formats:** the header decides what it can carry. Reported whether any can take Mila's scales, and read back, not assumed.
- **What this does not cover:** the FP8 KV cache and TokenEmbedding, which are not a matrix multiply by weights. Mixture of experts reads FP4 and Q4_0 as Linear does; its routing is Phase 6's. Phase 5 tests all of them on models.

## 2. The choice

**Decode, M = 1: Mila's own kernel. Every other shape measured, M = 8 to 2048: TensorOps.** All 34 shapes follow the gate's rule, and
each choice is the same taken from any single process run.

- **M = 1.** Own is faster on all nine shapes, by 6% to 54%, and has TensorOps' accuracy. TensorOps also needs a padded input: `matmul2d`
  on device tensors, as used here, refuses to compile a tile of 1 row and its header asks for a multiple of 8, so at M = 1 the arm multiplies seven rows of zeros, and the small copies in and out that a
  real kernel would need are not in its time.
- **M = 8 to 2048.** TensorOps is 1.6 to 4.8 times the fastest own tile, which misses the accuracy bar by far (6.6e-3 to 1.7e-2 against
  2e-4). The closest own attempt at the bar, spilling to 32 bits every 8 inputs, reaches 4.9e-4 against a bar of 4.1e-4, and takes 2.3 to 9.1 times
  TensorOps' time. At M = 8 the own kernel that does pass takes 3.2 to 7.8 times. So the choice does not rest on where the bar was set.
- **M = 2 to 7 was not measured.** The own kernel's time grows with M (7.8 times at M = 8), and TensorOps pads to 8 at the same cost as M = 8,
  so TensorOps from M = 2 is arithmetic. Phase 3 checks it.

The own kernel, `dotLanes`: one SIMD-group for every few outputs, its lanes walking the same rows of weights side by side, multiplying and
summing in 32 bits from 16-bit storage, with a `simd_sum` at the end. It reads the weights at 182 GB/s (arithmetic: 117 MB in 0.644 ms),
against the chip's 200 GB/s. Four outputs per SIMD-group serves every shape: on the large ones the number makes no measurable difference,
and on the smallest four is the best. The earlier form, one thread per output with each lane on its own row, took 2.5 ms on that shape.

**Tiles.** A tile choice matters: 1.6 to 8.9 times between the slowest and fastest at a shape, and FP16 can fall behind BF16 at the wrong one
(section 4). Two serve everything: 128 rows, 16 columns, 4 SIMD-groups at M of 128 and more, within 9% of each shape's best; and 8, 16, 2 at M = 8,
within 15%. The head at M = 8 was swept on five tiles only, and not on that one.

## 3. Results

Median milliseconds, the middle of three process runs' medians, FP16, normal data. A kernel's error is relative L2 against the double-precision
reference; TensorOps' is 2.0e-4 to 2.1e-4 everywhere, which is one rounding of the output to FP16.

**M = 2048**

| Shape | TensorOps tile (rows, columns, groups) | TensorOps | Fastest own | Its error | Own that meets the bar | Choice |
|---|---|---|---|---|---|---|
| 8B mlp in | 128, 32, 4 | 58.857 ms | 187.874 ms | 0.0094 | none | TensorOps |
| 8B mlp out | 128, 32, 4 | 62.742 ms | 303.421 ms | 0.017 | none | TensorOps |
| 8B query | 128, 32, 4 | 16.405 ms | 43.565 ms | 0.0094 | none | TensorOps |
| 8B key and value | 128, 32, 4 | 4.169 ms | 10.710 ms | 0.0094 | none | TensorOps |
| 3B mlp in | 128, 32, 4 | 17.170 ms | 42.925 ms | 0.0066 | none | TensorOps |
| 3B mlp out | 64, 64, 4 | 16.720 ms | 48.007 ms | 0.013 | none | TensorOps |
| 3B query | 64, 32, 2 | 4.101 ms | 10.522 ms | 0.0066 | none | TensorOps |
| 3B key and value | 128, 32, 4 | 2.091 ms | 5.331 ms | 0.0067 | none | TensorOps |

**M = 128**

| Shape | TensorOps tile (rows, columns, groups) | TensorOps | Fastest own | Its error | Own that meets the bar | Choice |
|---|---|---|---|---|---|---|
| 8B mlp in | 64, 32, 2 | 3.644 ms | 11.590 ms | 0.0093 | none | TensorOps |
| 8B mlp out | 128, 16, 4 | 4.261 ms | 18.445 ms | 0.017 | none | TensorOps |
| 8B query | 64, 16, 2 | 1.094 ms | 3.025 ms | 0.0093 | none | TensorOps |
| 8B key and value | 64, 16, 2 | 0.315 ms | 0.770 ms | 0.0093 | none | TensorOps |
| 3B mlp in | 128, 16, 4 | 1.074 ms | 2.757 ms | 0.0066 | none | TensorOps |
| 3B mlp out | 32, 16, 1 | 1.164 ms | 2.973 ms | 0.013 | none | TensorOps |
| 3B query | 64, 16, 2 | 0.295 ms | 0.728 ms | 0.0066 | none | TensorOps |
| 3B key and value | 64, 16, 2 | 0.162 ms | 0.381 ms | 0.0066 | none | TensorOps |

**M = 8**

| Shape | TensorOps tile (rows, columns, groups) | TensorOps | Fastest own | Its error | Own that meets the bar | Choice |
|---|---|---|---|---|---|---|
| 8B mlp in | 8, 64, 4 | 0.697 ms | 1.507 ms | 0.0094 | 5.054 ms | TensorOps |
| 8B mlp out | 8, 32, 4 | 0.893 ms | 2.019 ms | 0.017 | 5.062 ms | TensorOps |
| 8B query | 8, 16, 1 | 0.216 ms | 0.347 ms | 0.0093 | 1.399 ms | TensorOps |
| 8B key and value | 8, 16, 2 | 0.061 ms | 0.146 ms | 0.0094 | 0.209 ms | TensorOps |
| 3B mlp in | 8, 32, 1 | 0.197 ms | 0.349 ms | 0.0066 | 1.354 ms | TensorOps |
| 3B mlp out | 8, 16, 1 | 0.256 ms | 0.406 ms | 0.013 | 1.342 ms | TensorOps |
| 3B query | 8, 16, 2 | 0.038 ms | 0.097 ms | 0.0066 | 0.215 ms | TensorOps |
| 3B key and value | 8, 16, 2 | 0.033 ms | 0.078 ms | 0.0066 | 0.105 ms | TensorOps |
| 8B head | 8, 64, 4 | 6.000 ms | 46.474 ms | 0.00021 | 46.474 ms | TensorOps |

**M = 1**

| Shape | TensorOps tile (rows, columns, groups) | TensorOps | Fastest own | Its error | Own that meets the bar | Choice |
|---|---|---|---|---|---|---|
| 8B mlp in | 8, 64, 4 | 0.702 ms | 0.644 ms | 0.0002 | 0.644 ms | own |
| 8B mlp out | 8, 32, 4 | 0.895 ms | 0.654 ms | 0.0002 | 0.654 ms | own |
| 8B query | 8, 16, 1 | 0.213 ms | 0.191 ms | 0.0002 | 0.191 ms | own |
| 8B key and value | 8, 16, 2 | 0.079 ms | 0.036 ms | 0.0002 | 0.036 ms | own |
| 3B mlp in | 8, 32, 2 | 0.203 ms | 0.185 ms | 0.00021 | 0.185 ms | own |
| 3B mlp out | 8, 16, 1 | 0.268 ms | 0.215 ms | 0.0002 | 0.215 ms | own |
| 3B query | 8, 16, 2 | 0.051 ms | 0.033 ms | 0.00021 | 0.033 ms | own |
| 3B key and value | 8, 16, 2 | 0.048 ms | 0.028 ms | 0.0002 | 0.028 ms | own |
| 8B head | 8, 32, 2 | 6.020 ms | 5.675 ms | 0.00021 | 5.675 ms | own |

**FP16 inputs, FP32 output, TensorOps.** No measurable cost, 0.98 to 1.01 times FP16 output on all but the smallest decode shapes, whose timings are noise
(down to 0.85). Error 8e-7 to 2.1e-6 against 2.1e-4. At small data the same: 58.68 against 58.79 ms at the 8B mlp, error 1.14e-6 against 2.08e-4.
So it costs nothing, and removes the output's rounding. Small data as run here did not reach FP16's limits: FP16 output kept 2.1e-4, so
this shows no range problem being solved. The inputs are FP16 either way. Whether a model should use it is for Phase 4.

**BF16, emulated here, decides nothing.** TensorOps BF16 takes 1.4 to 2.9 times FP16, and error 1.7e-3. At M = 1 the own kernel takes the same time as FP16
(0.642 against 0.644 ms), since it works in 32 bits either way, and has the same error of 1.7e-3 from BF16's output rounding. So on this chip BF16's cost is at
prefill, and there is none at decode. Phase 4 measures both on a model.

**Own tiles, for the record.** Tiles of 8 x 8 at 2048 x 4096 x 14336 took 408 ms. Loading fewer matrices than are multiplied, by holding a patch of tiles, was the
one change that moved it: 2 x 2 took 197 ms, 2 x 4 188 ms, and 3 x 3 146 ms, the last on a shape it does not divide, so about 0.5% short of the work. At 4 x 4 and at 3 x 4 it took 1460 and 1580 ms,
inferred to be the registers running out. Grid order and blocks of M and N each moved it about 1%, which ended the tuning by the gate's rule. Blocks of K made it slower. FP16's error
in a tile grows with K, 9.4e-3 at K = 4096 and 1.7e-2 at K = 14336.

**Stability.** 18 of 212 arms differ between runs by more than 15%, 17 of them under 0.3 ms at M = 1, where both paths tended to slow together in the same run; the other is BF16 at M = 128. None changes a choice. The misaligned
output scored 1.37 or more in every shape, and true outputs 2e-4. The Mac stayed on AC, with no thermal warning.

## 4. The anomaly, explained

TensorOps BF16 did not run fast at the decode-like shape. **FP16 ran slow, on one tile.** At M = 8, K = N = 8192, on the earlier program's layout, weights one row per input:

| Varied | FP16 | BF16 | BF16 / FP16 |
|---|---|---|---|
| As the earlier timing: tile 8 x 32, 4 groups | 2.77 | 1.34 | **0.48** |
| 16 columns | 1.11 | 1.38 | 1.24 |
| 64 columns | 1.24 | 1.96 | 1.57 |
| 1 group | 0.85 | 1.10 | 1.29 |
| 2 groups | 1.07 | 1.55 | 1.46 |
| Normal data | 2.78 | 1.33 | 0.48 |
| Weights one row per output, Linear's layout | 0.90 | 1.89 | 2.11 |
| K and N 2048 | 0.12 | 0.29 | 2.50 |
| M = 64, tile 8 rows | 31.8 | 8.93 | 0.28 |
| M = 64, tile 64 rows | 8.79 | 7.37 | 0.84 |

It reproduces, in all three runs at each row, and data is not the cause. Changing the tile columns or the groups takes FP16 from 2.77 to about 1 ms and leaves BF16 near where it was, so the ratio flips with
any of them: the slow case is that tile, with that layout, on large K and N. **In Linear's layout it does not occur**: FP16 is 2.1 times BF16's speed, as at every other shape. Why that tile is slow is not known, and Apple
documents nothing about it. No decision rests on it. What it does change: `Notebooks/MetalPrecisionTiming.md`'s FP16 TensorOps figure at its shape D is this slow tile, and not a fair number for TensorOps.

## 5. What went wrong on the way

- **An own kernel could pass the check on another arm's output.** Arms shared an output buffer, and a patch that does not divide the shape leaves elements unwritten, which then held the last arm's correct values. Seen
  from 3 x 3's error matching 2 x 2's exactly. The output is now cleared before each arm's check, and `tile` refuses a shape its patch does not divide. Sweeps from there on are clean. The first looks were rerun, and the 3 x 3 times are reported as short of the work.
- **The first gate run produced no results for most shapes.** `dot` is not defined on `bfloat4`, so BF16 arms failed to compile, which the script reported and the table script caught. Rerun in full.
- **A status command on the Mac never returned**, `pmset -g thermlog`, which streams. Stopped. It ran after the run, so it touched no number.

## 6. Every weight format

**The choice, per format and rows, in FP16:**

| Rows (M) | Unquantized | FP8 | FP4, Q4_0, Q6_K, codebooks |
|---|---|---|---|
| **1** | Own lanes | Own lanes | Own lanes |
| **8** | TensorOps | TensorOps on the bare codes | Own lanes |
| **128** | TensorOps | TensorOps on the bare codes | TensorOps, expanded |
| **2048** | TensorOps | TensorOps, expanded | TensorOps, expanded |

All 308 arms of the gate ran three process runs, none differed between runs by more than 11%, and every choice is the same taken from any single run. The misaligned output scored 1.397 or more. Mac on AC, no thermal warning. The only rows not
measured are M = 2 to 7 and M between 8 and 128, and the head is at M = 8 and 1 only.

**TensorOps cannot carry Mila's scales, with one exception.** This is the SDK's header, `MPPTensorOpsMatMul2dImpl.h`, not a measurement: a tensor's block scale must be `metal_fp8_ue8m0_format`, a power of two, with a block of exactly 32
(a `static_assert` in each case). That is the shape of NVFP4, and not of Mila's FP4 (FP32, per 128 or 64), Q4_0 or Q6_K (FP16, per 32). The type table has `int4`, `int2`, FP4 and FP8 weights, and nothing for six bits or a codebook's table.

**FP8 is the exception.** Its scale belongs to one output, so it comes out of the sum: TensorOps multiplies FP16 activations by the bare FP8 codes into 32 bits, and a second pass applies the scales, with Mila's bytes used as they are and nothing expanded. Its output matches the
reference, so Metal's E4M3 is Mila's (2.1e-4). That pass is the cost: it loses to expanding at M = 2048 (68 against 61 ms), and wins below it (4.2 against 6.0 ms at M = 128, 0.46 ms at M = 8, where expanding takes 1.9).

**Prefill costs little.** Everything else is expanded to FP16 in a scratch buffer, 1024 outputs at a time, then multiplied by TensorOps. Over plain TensorOps on the same shape, that costs at most 5% at M = 2048 and 20% to 74% at M = 128. The scratch is 8 MB at K = 4096 and 29 MB at K = 14336, since a block of 1024 takes
no longer than expanding everything at once (63.2 against 62.6 ms) where 256 takes 7% longer. So every format prefills at nearly FP16's speed, from half its memory or less. Expanding rounds each weight to FP16, which brings the error from 2.1e-4 to between 2.6e-4 and 3.0e-4: inside the bar, and well below BF16 output's 1.7e-3.

**Decode is where the formats pay.** At M = 1 expanding takes 3.7 to 17 times the own kernel's time, and the own kernel is 20% to 36% faster than TensorOps on FP8's codes. Tuned for three things in turn, each measured on 8B mlp in:

| Own kernel, FP16, 1 x 4096 x 14336 | FP4-128 | Q4_0 | Q6_K | 2-bit codebook | 3-bit codebook | FP8 |
|---|---|---|---|---|---|---|
| One weight at a time, in the first look (noisy) | 0.96 | 1.19 | 1.31 | 1.18 | 0.86 | 0.92 |
| Eight weights from one load | 0.302 | 0.380 | 0.456 | 0.304 | 0.391 | 0.366 |
| And activations read once for 2 outputs | 0.222 | 0.306 | 0.390 | 0.230 | 0.314 | 0.338 |
| For 4 outputs | 0.204 | 0.261 | 0.356 | 0.201 | 0.275 | 0.339 |
| For 8 outputs | 0.208 | 0.242 | 0.338 | 0.198 | 0.277 | 0.348 |

Unquantized takes 0.63 to 0.67 ms on the same shape. Reading activations once for 4 outputs moved every format by more than 10% but FP8, and 8 moved every format by under 10%, which ends it by the gate's rule. FP4 and FP8 read through a table, here 1.7 to 3 times as fast as arithmetic on the bits at M = 1, and about level at M = 8. Why arithmetic is so much slower is not known. The sections below were run on the second row's form, so they understate the own kernel; no choice changes.

These are limited by arithmetic, not memory: FP4 at 0.204 ms is 288 billion weights a second, but only 144 GB/s of its codes. **Arithmetic, as a ceiling on Linear alone:** at Q4_0's 243 billion weights a second, Llama 3.1 8B's weights take 33 ms a token and Gemma 4 12B's 49 ms, so about 30 and 20 tokens a second before attention and everything else. llama.cpp's published 35.5 tokens a second for Llama 2 7B Q4_0 on this chip implies about 250 billion. Measured on a model in Phase 5.

**At M = 8, own lanes win for all but FP8,** where expanding takes 1.1 to 2.4 times as long, and over 3 times on the head, in every run. They read the weights once for all eight rows. Unquantized stays with TensorOps, 2 to 2.2 times faster, so a kernel's choice depends on the format as well as the shape.

**BF16, emulated here, decides nothing.** Expanded at M = 2048, 120 ms against 61, with error 2.3e-3. Own lanes at M = 1: Q4_0 0.402 against 0.380 ms, 3-bit codebook 0.433 against 0.391, error 1.7e-3. FP4 was run by arithmetic only.

**What it means for Phase 3 and 5.** `Linear` gets three kernels per format, chosen by M: lanes, expand and multiply, and for FP8, bare codes and scale. Expanding needs a scratch buffer that belongs to the execution context, as on CUDA. Nothing here passes a per-token value, so the decode rule holds. Phase 2's unquantized choice stands: rerun with the repeats below, own at M = 1 beats TensorOps on the five shapes run, by 5% to 25%.

**Measurement changed here.** Run alone, a sub-millisecond kernel on this GPU varied from 0.30 to 1.0 ms: inferred to be the GPU's clock, since it does not when the GPU is kept busy. A decode step is many kernels in a row, so M = 1 ran 32 times in one command buffer, M = 8 16 times, the head 8 and 4, and the time is per run. Spreads fell under 1%. Section 3's M = 1 numbers are from single runs, but agree.

**What this does not settle.** Weights are normal, and a codebook's table a fixed spread, not fitted. Tiles were carried over from section 2's sweep, and not swept again on expanded weights. FP8's scale pass was not tuned. The FP8 KV cache, TokenEmbedding and mixture of experts were not run; they read these same layouts, and Phase 5 and 6 test them on models.

**What went wrong.** A progress note to Todd read FP4's 0.301 ms as 195 GB/s, at the chip's limit, counting two bytes a weight where FP4 has half a byte; that was the weight rate, and corrected the same night. The first look at decode (the table's first row) is noisy for the reason above.

### Results

Milliseconds, FP16, at each path's form in the gate: eight weights from one load, one output to a SIMD-group. Own lanes are the better of table and arithmetic for FP4 and FP8.

**M = 2048**

| Shape | Format | Own lanes | TensorOps, expanded | TensorOps, FP8 direct | Error, own / expanded | Choice |
|---|---|---|---|---|---|---|
| 8B mlp in | unquantized | - | 58.726 | - | | TensorOps |
| 8B mlp in | fp8 | - | 61.059 | 68.229 | - / 0.00029 | expanded |
| 8B mlp in | fp4-128 | - | 61.465 | - | - / 0.0003 | expanded |
| 8B mlp in | fp4-64 | - | 61.478 | - | - / 0.00029 | expanded |
| 8B mlp in | q4 | - | 61.127 | - | - / 0.00027 | expanded |
| 8B mlp in | q6 | - | 61.129 | - | - / 0.00028 | expanded |
| 8B mlp in | cb2-32 | - | 61.106 | - | - / 0.00026 | expanded |
| 8B mlp in | cb2-64 | - | 61.027 | - | - / 0.00027 | expanded |
| 8B mlp in | cb3-64 | - | 61.064 | - | - / 0.00027 | expanded |
| 8B mlp in | cb3-128 | - | 61.100 | - | - / 0.00028 | expanded |
| 8B mlp out | unquantized | - | 62.715 | - | | TensorOps |
| 8B mlp out | fp8 | - | 63.311 | 68.242 | - / 0.0003 | expanded |
| 8B mlp out | fp4-128 | - | 63.569 | - | - / 0.0003 | expanded |
| 8B mlp out | fp4-64 | - | 64.300 | - | - / 0.0003 | expanded |
| 8B mlp out | q4 | - | 64.002 | - | - / 0.00027 | expanded |
| 8B mlp out | q6 | - | 63.971 | - | - / 0.00028 | expanded |
| 8B mlp out | cb2-32 | - | 63.653 | - | - / 0.00026 | expanded |
| 8B mlp out | cb2-64 | - | 63.559 | - | - / 0.00027 | expanded |
| 8B mlp out | cb3-64 | - | 63.819 | - | - / 0.00027 | expanded |
| 8B mlp out | cb3-128 | - | 63.929 | - | - / 0.00028 | expanded |
| 8B query | unquantized | - | 17.215 | - | | TensorOps |
| 8B query | fp8 | - | 17.841 | 19.960 | - / 0.00029 | expanded |
| 8B query | fp4-128 | - | 17.893 | - | - / 0.00029 | expanded |
| 8B query | fp4-64 | - | 17.295 | - | - / 0.00029 | expanded |
| 8B query | q4 | - | 17.559 | - | - / 0.00027 | expanded |
| 8B query | q6 | - | 17.478 | - | - / 0.00028 | expanded |
| 8B query | cb2-32 | - | 17.714 | - | - / 0.00026 | expanded |
| 8B query | cb2-64 | - | 17.676 | - | - / 0.00027 | expanded |
| 8B query | cb3-64 | - | 17.778 | - | - / 0.00027 | expanded |
| 8B query | cb3-128 | - | 16.880 | - | - / 0.00028 | expanded |
| 3B mlp in | unquantized | - | 17.079 | - | | TensorOps |
| 3B mlp in | fp8 | - | 17.192 | 20.142 | - / 0.00029 | expanded |
| 3B mlp in | fp4-128 | - | 17.384 | - | - / 0.0003 | expanded |
| 3B mlp in | fp4-64 | - | 17.977 | - | - / 0.00029 | expanded |
| 3B mlp in | q4 | - | 17.222 | - | - / 0.00027 | expanded |
| 3B mlp in | q6 | - | 17.824 | - | - / 0.00028 | expanded |
| 3B mlp in | cb2-32 | - | 17.803 | - | - / 0.00026 | expanded |
| 3B mlp in | cb2-64 | - | 17.803 | - | - / 0.00027 | expanded |
| 3B mlp in | cb3-64 | - | 17.815 | - | - / 0.00027 | expanded |
| 3B mlp in | cb3-128 | - | 17.219 | - | - / 0.00028 | expanded |

**M = 128**

| Shape | Format | Own lanes | TensorOps, expanded | TensorOps, FP8 direct | Error, own / expanded | Choice |
|---|---|---|---|---|---|---|
| 8B mlp in | unquantized | - | 3.646 | - | | TensorOps |
| 8B mlp in | fp8 | - | 6.022 | 4.203 | - / 0.0003 | direct |
| 8B mlp in | fp4-128 | - | 6.337 | - | - / 0.0003 | expanded |
| 8B mlp in | fp4-64 | - | 6.331 | - | - / 0.00029 | expanded |
| 8B mlp in | q4 | - | 5.902 | - | - / 0.00027 | expanded |
| 8B mlp in | q6 | - | 5.917 | - | - / 0.00028 | expanded |
| 8B mlp in | cb2-32 | - | 5.883 | - | - / 0.00026 | expanded |
| 8B mlp in | cb2-64 | - | 5.893 | - | - / 0.00027 | expanded |
| 8B mlp in | cb3-64 | - | 5.890 | - | - / 0.00027 | expanded |
| 8B mlp in | cb3-128 | - | 5.894 | - | - / 0.00028 | expanded |
| 8B mlp out | unquantized | - | 4.260 | - | | TensorOps |
| 8B mlp out | fp8 | - | 5.462 | 4.499 | - / 0.0003 | direct |
| 8B mlp out | fp4-128 | - | 5.559 | - | - / 0.0003 | expanded |
| 8B mlp out | fp4-64 | - | 5.555 | - | - / 0.00029 | expanded |
| 8B mlp out | q4 | - | 5.223 | - | - / 0.00027 | expanded |
| 8B mlp out | q6 | - | 5.333 | - | - / 0.00028 | expanded |
| 8B mlp out | cb2-32 | - | 5.132 | - | - / 0.00026 | expanded |
| 8B mlp out | cb2-64 | - | 5.133 | - | - / 0.00027 | expanded |
| 8B mlp out | cb3-64 | - | 5.165 | - | - / 0.00027 | expanded |
| 8B mlp out | cb3-128 | - | 5.153 | - | - / 0.00028 | expanded |
| 8B query | unquantized | - | 1.097 | - | | TensorOps |
| 8B query | fp8 | - | 1.461 | 1.274 | - / 0.00029 | direct |
| 8B query | fp4-128 | - | 1.545 | - | - / 0.0003 | expanded |
| 8B query | fp4-64 | - | 1.545 | - | - / 0.00029 | expanded |
| 8B query | q4 | - | 1.424 | - | - / 0.00027 | expanded |
| 8B query | q6 | - | 1.428 | - | - / 0.00028 | expanded |
| 8B query | cb2-32 | - | 1.421 | - | - / 0.00026 | expanded |
| 8B query | cb2-64 | - | 1.422 | - | - / 0.00027 | expanded |
| 8B query | cb3-64 | - | 1.422 | - | - / 0.00027 | expanded |
| 8B query | cb3-128 | - | 1.424 | - | - / 0.00028 | expanded |
| 3B mlp in | unquantized | - | 1.073 | - | | TensorOps |
| 3B mlp in | fp8 | - | 1.518 | 1.273 | - / 0.00029 | direct |
| 3B mlp in | fp4-128 | - | 1.602 | - | - / 0.0003 | expanded |
| 3B mlp in | fp4-64 | - | 1.600 | - | - / 0.00029 | expanded |
| 3B mlp in | q4 | - | 1.474 | - | - / 0.00027 | expanded |
| 3B mlp in | q6 | - | 1.481 | - | - / 0.00028 | expanded |
| 3B mlp in | cb2-32 | - | 1.468 | - | - / 0.00026 | expanded |
| 3B mlp in | cb2-64 | - | 1.469 | - | - / 0.00027 | expanded |
| 3B mlp in | cb3-64 | - | 1.471 | - | - / 0.00027 | expanded |
| 3B mlp in | cb3-128 | - | 1.471 | - | - / 0.00028 | expanded |

**M = 8**

| Shape | Format | Own lanes | TensorOps, expanded | TensorOps, FP8 direct | Error, own / expanded | Choice |
|---|---|---|---|---|---|---|
| 8B mlp in | unquantized | 1.531 | 0.695 | - | | TensorOps |
| 8B mlp in | fp8 | 1.507 | 1.884 | 0.461 | 0.00021 / 0.0003 | direct |
| 8B mlp in | fp4-128 | 1.420 (table) | 2.161 | - | 0.00021 / 0.0003 | own |
| 8B mlp in | fp4-64 | 1.421 (table) | 2.155 | - | 0.00021 / 0.00029 | own |
| 8B mlp in | q4 | 1.466 | 1.736 | - | 0.00021 / 0.00027 | own |
| 8B mlp in | q6 | 1.487 | 1.763 | - | 0.00021 / 0.00028 | own |
| 8B mlp in | cb2-32 | 1.377 | 1.695 | - | 0.00021 / 0.00026 | own |
| 8B mlp in | cb2-64 | 1.381 | 1.702 | - | 0.00021 / 0.00027 | own |
| 8B mlp in | cb3-64 | 1.419 | 1.712 | - | 0.00021 / 0.00027 | own |
| 8B mlp in | cb3-128 | 1.418 | 1.712 | - | 0.00021 / 0.00028 | own |
| 8B mlp out | unquantized | 1.791 | 0.911 | - | | TensorOps |
| 8B mlp out | fp8 | 1.590 | 2.195 | 0.586 | 0.00021 / 0.0003 | direct |
| 8B mlp out | fp4-128 | 1.426 (table) | 2.354 | - | 0.00021 / 0.00029 | own |
| 8B mlp out | fp4-64 | 1.443 (table) | 2.349 | - | 0.00021 / 0.0003 | own |
| 8B mlp out | q4 | 1.551 | 2.023 | - | 0.00021 / 0.00027 | own |
| 8B mlp out | q6 | 1.575 | 2.105 | - | 0.00021 / 0.00028 | own |
| 8B mlp out | cb2-32 | 1.389 | 1.958 | - | 0.00021 / 0.00026 | own |
| 8B mlp out | cb2-64 | 1.424 | 1.959 | - | 0.00021 / 0.00027 | own |
| 8B mlp out | cb3-64 | 1.459 | 1.988 | - | 0.00021 / 0.00027 | own |
| 8B mlp out | cb3-128 | 1.452 | 1.977 | - | 0.00021 / 0.00028 | own |
| 8B query | unquantized | 0.431 | 0.205 | - | | TensorOps |
| 8B query | fp8 | 0.417 | 0.504 | 0.154 | 0.00021 / 0.00029 | direct |
| 8B query | fp4-128 | 0.370 (table) | 0.585 | - | 0.00021 / 0.0003 | own |
| 8B query | fp4-64 | 0.372 (table) | 0.579 | - | 0.00021 / 0.00029 | own |
| 8B query | q4 | 0.404 | 0.462 | - | 0.00021 / 0.00027 | own |
| 8B query | q6 | 0.410 | 0.471 | - | 0.00021 / 0.00028 | own |
| 8B query | cb2-32 | 0.362 | 0.454 | - | 0.00021 / 0.00026 | own |
| 8B query | cb2-64 | 0.362 | 0.450 | - | 0.00021 / 0.00027 | own |
| 8B query | cb3-64 | 0.378 | 0.457 | - | 0.00021 / 0.00027 | own |
| 8B query | cb3-128 | 0.377 | 0.458 | - | 0.00021 / 0.00028 | own |
| 3B mlp in | unquantized | 0.396 | 0.198 | - | | TensorOps |
| 3B mlp in | fp8 | 0.376 | 0.734 | 0.151 | 0.00021 / 0.00029 | direct |
| 3B mlp in | fp4-128 | 0.345 (table) | 0.819 | - | 0.00021 / 0.0003 | own |
| 3B mlp in | fp4-64 | 0.345 (table) | 0.813 | - | 0.00021 / 0.00029 | own |
| 3B mlp in | q4 | 0.354 | 0.690 | - | 0.00021 / 0.00027 | own |
| 3B mlp in | q6 | 0.365 | 0.696 | - | 0.00021 / 0.00028 | own |
| 3B mlp in | cb2-32 | 0.334 | 0.683 | - | 0.00021 / 0.00026 | own |
| 3B mlp in | cb2-64 | 0.334 | 0.677 | - | 0.00021 / 0.00027 | own |
| 3B mlp in | cb3-64 | 0.349 | 0.683 | - | 0.00021 / 0.00027 | own |
| 3B mlp in | cb3-128 | 0.348 | 0.682 | - | 0.00021 / 0.00028 | own |
| 8B head | unquantized | 13.534 | 6.126 | - | | TensorOps |
| 8B head | fp4-128 | 12.738 (table) | 45.197 | - | 0.00021 / 0.0003 | own |
| 8B head | q6 | 13.165 | 41.424 | - | 0.00021 / 0.00028 | own |

**M = 1**

| Shape | Format | Own lanes | TensorOps, expanded | TensorOps, FP8 direct | Error, own / expanded | Choice |
|---|---|---|---|---|---|---|
| 8B mlp in | unquantized | 0.671 | 0.710 | - | | own |
| 8B mlp in | fp8 | 0.366 (table) | 1.879 | 0.459 | 0.00021 / 0.00029 | own |
| 8B mlp in | fp4-128 | 0.302 (table) | 2.161 | - | 0.00021 / 0.0003 | own |
| 8B mlp in | fp4-64 | 0.302 (table) | 2.162 | - | 0.00021 / 0.00029 | own |
| 8B mlp in | q4 | 0.380 | 1.735 | - | 0.0002 / 0.00027 | own |
| 8B mlp in | q6 | 0.456 | 1.758 | - | 0.00021 / 0.00028 | own |
| 8B mlp in | cb2-32 | 0.304 | 1.698 | - | 0.00021 / 0.00026 | own |
| 8B mlp in | cb2-64 | 0.304 | 1.693 | - | 0.00021 / 0.00027 | own |
| 8B mlp in | cb3-64 | 0.391 | 1.714 | - | 0.00021 / 0.00027 | own |
| 8B mlp in | cb3-128 | 0.391 | 1.717 | - | 0.00021 / 0.00028 | own |
| 8B mlp out | unquantized | 0.691 | 0.918 | - | | own |
| 8B mlp out | fp8 | 0.386 (table) | 2.183 | 0.605 | 0.00021 / 0.0003 | own |
| 8B mlp out | fp4-128 | 0.308 (table) | 2.349 | - | 0.00021 / 0.0003 | own |
| 8B mlp out | fp4-64 | 0.308 (table) | 2.349 | - | 0.00021 / 0.00029 | own |
| 8B mlp out | q4 | 0.392 | 2.025 | - | 0.0002 / 0.00027 | own |
| 8B mlp out | q6 | 0.462 | 2.104 | - | 0.00021 / 0.00028 | own |
| 8B mlp out | cb2-32 | 0.313 | 1.962 | - | 0.00021 / 0.00026 | own |
| 8B mlp out | cb2-64 | 0.314 | 1.952 | - | 0.00021 / 0.00027 | own |
| 8B mlp out | cb3-64 | 0.394 | 1.984 | - | 0.00021 / 0.00028 | own |
| 8B mlp out | cb3-128 | 0.409 | 1.978 | - | 0.0002 / 0.00028 | own |
| 8B query | unquantized | 0.188 | 0.210 | - | | own |
| 8B query | fp8 | 0.105 (table) | 0.525 | 0.152 | 0.00021 / 0.00028 | own |
| 8B query | fp4-128 | 0.092 (table) | 0.593 | - | 0.00021 / 0.0003 | own |
| 8B query | fp4-64 | 0.092 (table) | 0.612 | - | 0.00021 / 0.00029 | own |
| 8B query | q4 | 0.116 | 0.464 | - | 0.00021 / 0.00027 | own |
| 8B query | q6 | 0.136 | 0.498 | - | 0.00021 / 0.00028 | own |
| 8B query | cb2-32 | 0.094 | 0.457 | - | 0.00021 / 0.00026 | own |
| 8B query | cb2-64 | 0.094 | 0.467 | - | 0.00021 / 0.00027 | own |
| 8B query | cb3-64 | 0.117 | 0.467 | - | 0.00021 / 0.00027 | own |
| 8B query | cb3-128 | 0.117 | 0.460 | - | 0.00021 / 0.00028 | own |
| 3B mlp in | unquantized | 0.176 | 0.194 | - | | own |
| 3B mlp in | fp8 | 0.101 (table) | 0.553 | 0.143 | 0.0002 / 0.00029 | own |
| 3B mlp in | fp4-128 | 0.092 (table) | 0.644 | - | 0.00021 / 0.0003 | own |
| 3B mlp in | fp4-64 | 0.092 (table) | 0.642 | - | 0.00021 / 0.0003 | own |
| 3B mlp in | q4 | 0.115 | 0.506 | - | 0.00021 / 0.00027 | own |
| 3B mlp in | q6 | 0.137 | 0.514 | - | 0.00021 / 0.00028 | own |
| 3B mlp in | cb2-32 | 0.093 | 0.498 | - | 0.00021 / 0.00026 | own |
| 3B mlp in | cb2-64 | 0.093 | 0.498 | - | 0.00021 / 0.00027 | own |
| 3B mlp in | cb3-64 | 0.118 | 0.491 | - | 0.00021 / 0.00027 | own |
| 3B mlp in | cb3-128 | 0.118 | 0.493 | - | 0.00021 / 0.00028 | own |
| 8B head | unquantized | 5.794 | 6.116 | - | | own |
| 8B head | fp4-128 | 2.747 (table) | 45.642 | - | 0.00021 / 0.0003 | own |
| 8B head | q6 | 4.126 | 41.738 | - | 0.00021 / 0.00028 | own |
