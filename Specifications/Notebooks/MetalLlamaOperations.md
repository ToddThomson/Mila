# Metal: Llama's Operations

Phase 3 of `Metal/MetalBackend.md`. **Question:** does each operation give correct values in FP16 and in BF16?

## 1. The gate, written before any run

**Machine and rules.** The M1 Pro, macOS 27.2, on AC under `caffeinate -dimsu`, with the macOS preset. Section 6 of the design applies.

**What passes.** Every operation below, in FP16 and in BF16, matches a double-precision reference computed in its test, with these bars. Each is checked by values, since a Metal binding has already failed once by writing zeros with no error.

| Operation | Bar |
|---|---|
| Linear | Relative L2 error against the reference on the weights as stored: FP16 no more than 5e-4, BF16 no more than 4e-3. Phase 2 measured 2.1e-4 and 1.7e-3; the bars are about twice that |
| TokenEmbedding | Exactly the table's row as stored, since nothing is computed |
| RmsNorm, Swiglu, Residual | The same two bars as Linear, on normal data and on data that overflows FP16 inside the operation: a sum of squares past 65504 for RmsNorm, a gate of about -100 for Swiglu |
| Rope | The same two bars, at positions up to 2048. Its angle is single precision, as on CUDA, so its error grows with position; a far position is Phase 4's question |
| Grouped-query attention | FP16 no more than 2e-3, BF16 no more than 2e-2, prefill and decode, against the reference over the same cache. Wider than Linear's: a score's rounding goes through an exponential, and these are estimates of that effect, written before any run |
| Sampling | Greedy exact. A draw is the token a double-precision CDF gives at that uniform, skipping a uniform within 5e-5 of a token's edge, where the two may differ by a rounding of their running sums |
| NextTokenLogProbability | Within 1e-3 of the host reference. Metal has no 64-bit float, so the exponentials sum in 32 bits |
| LlamaBlock, prefill against decode | FP16 no more than 2e-3, BF16 no more than 2e-2 relative L2 between the two ways to the same token |

**Weights arrive as a BF16 blob, through `loadParameter`, in both precisions.** No published model stores FP16. A value FP16 cannot hold is refused with the parameter's name, and never clipped.

**Shapes.** Rows at 1, 2, 3, 7, 8, 9, 15, 16, 127, 128, 129, 136, 143 and 256; 136 and 143 are where tiles of 8 follow tiles of 128, which three forced failures showed the first list never reached. Outputs on and off a multiple of 16. Inputs on and off a multiple of 4. A view of fewer rows than were built. A bias.

**A missing specialization is a compile error.** Only FP16 and BF16 rows exist, and the tests say so by `OperationSupported`.

**Forced failures.** Every kernel gets at least one mutant that the tests must catch, listed in section 3 with its result. A mutant that survives is a missing test, and is closed before the batch counts.

**The decode rule.** A decode test sets the position first, through the execution context, and no kernel takes a position as a launch argument.

**One check beyond the design's list.** A small `LlamaBlock` on random weights: prefilling N tokens equals prefilling N - 1 and decoding one. It catches a wrong cache or position before Phase 4 has a whole model to read.

**At the end of the phase.** First-load compile cost again, which section 4.3 of the design asks once real kernels exist.

**What this cannot say.** Nothing here is a model, and normal test data does not reach the ranges a real model's activations might. That is Phase 4's gate.

## 2. Batches

1. Dispatch, BF16 weights on load, and Linear.
2. TokenEmbedding, RmsNorm, Rope, Swiglu, Residual.
3. Grouped-query attention, prefill and decode, and the `LlamaBlock` check.
4. Sampling, NextTokenLogProbability.

## 3. Results

**Every operation passes its bar in both precisions,** on the M1 Pro, macOS 27.2, on AC, 2026-11-05. Metal tests 194 of 194. `MilaTests` on the Mac 1431 pass, 1 skip (a CUDA buffer test, rightly), Agents 10, Chat 33, Desktop 14. Weights reached every test as a BF16 blob. Not a model run.

**Windows, `x64-claude-verify`:** all targets with 0 warnings, and the suite at 2475 pass, 12 skip and the 4 failures already filed (one Gemma 12B drafter test, three on a partial Gemma 26B file). Commit checks pass. **Not run:** Todd's `x64-profile`.

**No component needed a `#if` for Metal,** which is the claim `MetalBackend.md` section 4.1 sets out to test. The one change outside the backend is by data type and not by device: `Component::loadParameterFromBlob` lets an FP16 parameter read BF16. `Llama.ixx`'s load synchronization and the `requires Cuda` members of `Linear` and `GroupedQueryAttention` were not reached by this phase's tests, and are Phases 4 and 6.

**Three parts of the backend were missing, and are built:** `split`, `scale` (TokenEmbedding's embedding scale calls it even when a model never scales) and `fill_uniform` and `fill_normal`.

### Forced failures

111 in all, each a change on the Mac's copy, built and run against the Metal tests. Scripts `D:\Claude\metal\phase3\mutate1.py` to `mutate4.py`.

| Batch | Caught first | What the survivors were |
|---|---|---|
| 1, Linear and BF16 weights | 19 of 22 | Three offsets a tile of 8 takes after tiles of 128. The row list stopped at 129, so that path had never run. 136 and 143 rows added; all three then caught |
| 2, TokenEmbedding, RmsNorm, Rope, Swiglu, Residual, scale, random | 28 of 31, and the three of batch 1 | See below |
| 3 and 4, attention, Sampling, NextTokenLogProbability, split | 54 of 58 | `split` had no test of its own, since prefill and decode both go through it and agree when it is wrong. Every test head was one tile wide. Several tokens in a decode are refused first by the component. Tests added; all four then caught |

Three survivors of batch 2 were not gaps in the tests:

- **A decode with no position set** is still refused with the check removed, by the allocation map, which refuses a null address.
- **SwiGLU computed in the element type** passes the bar in both precisions. Its 32-bit arithmetic is what CUDA's kernel does, and no measurement here earned it. Left as it is, and a candidate if Phase 4 shows it costs.
- **A residual's scaling factor** was refused twice. `ResidualConfig::validate` already does it, so the operation's check was deleted.

### Which path each row count really takes

Values cannot say, since both paths are right, so a run printed each tile dispatch. 1, 2, 3 and 7 rows take no tile; 8, 9 and 15 take 8 rows of tile; 16 takes 16; 127 takes 120; 128 and 129 take 128; 256 takes 256.

### M = 2 to 7, own kernel against TensorOps padded to 8 rows

Milliseconds, FP16, normal data, the middle of three process runs of 15 rounds each. The padded arm multiplies zero rows and leaves out the small copies in and out that a real kernel would need, as in `MetalMatrixPath.md`. Phase 2's rule: own stays unless TensorOps is more than 10% faster.

| Shape, K x N | M = 2 own | M = 2 TensorOps | M = 7 own | M = 7 TensorOps | Own over TensorOps, M = 2 to 7 |
|---|---|---|---|---|---|
| 4096 x 14336 | 1.280 | 0.707 | 4.454 | 0.721 | 1.81 to 6.18 |
| 14336 x 4096 | 1.310 | 0.925 | 4.494 | 0.923 | 1.42 to 4.87 |
| 4096 x 4096 | 0.472 | 0.389 | 1.104 | 0.186 | 1.21 to 5.94 |
| 2048 x 8192 | 0.325 | 0.193 | 1.121 | 0.191 | 1.68 to 5.87 |
| 8192 x 2048 | 0.309 | 0.210 | 1.023 | 0.207 | 1.47 to 4.94 |
| 2048 x 2048 | 0.124 | 0.106 | 0.391 | 0.105 | 1.17 to 3.72 |

**By the rule TensorOps wins from M = 2, as Phase 2's arithmetic expected.** 10 of the 36 shapes differed between runs by more than the 15% that makes a result unstable, all at small shapes, up to 2.4 times. The own kernel's error matches: 2.0e-4 to 2.1e-4. This table leaves padding's own cost out; the next section has it in.

### M = 2 to 7 through the real path, copies and all

After Phase 3, Todd agreed that Linears pad, and that attention pads only if timing earns it. `Lib/Tools/MetalMatrixTiming/ProductTiming.cpp` times `encodeMatrixProduct` itself: 32 products to a command buffer, as a network chains its layers, each padded product with its copy in, zeroing, tile, and copies out. Milliseconds per product, FP16 middle of three process runs of 15 rounds, BF16 one run. Own over padded: more than 1 means padding wins. Scripts and full logs are `D:\Claude\metal\padding` (`timing.sh`, `timing2.sh`, `table.py`).

Linears, outputs x inputs:

| Shape | M = 2 own | M = 2 padded | M = 7 own | M = 7 padded | Own over padded, M = 2 to 7, FP16 | BF16 |
|---|---|---|---|---|---|---|
| 28672 x 4096 (8B's gate and up) | 2.52 | 1.55 | 8.69 | 1.55 | 1.63 to 5.60 | 1.04 to 3.57 |
| 4096 x 14336 (8B's down) | 2.27 | 0.99 | 9.30 | 1.00 | 2.29 to 9.33 | 0.93 to 3.69 |
| 6144 x 4096 (8B's fused projection) | 0.59 | 0.41 | 1.89 | 0.41 | 1.42 to 4.58 | 1.03 to 3.34 |
| 4096 x 4096 (8B's output projection) | 0.33 | 0.29 | 1.14 | 0.30 | 1.14 to 3.87 | 0.93 to 2.87 |
| 8192 x 2048 | 0.34 | 0.27 | 1.15 | 0.28 | 1.25 to 4.13 | 0.86 to 2.75 |
| 2048 x 8192 | 0.31 | 0.30 | 0.76 | 0.31 | 1.03 to 2.47 | 0.65 to 1.63 |
| 2048 x 2048 | 0.057 | 0.081 | 0.160 | 0.085 | 0.71 to 1.89 | 0.43 to 1.11 |

Attention's shapes, N = 128. Q x K^T is outputs K x inputs 128. Attention x V is timed here as outputs 128 x inputs K, the same multiplies, on `linear_lanes`; its real leftover path is `gqa_value_elements`, so that row is a guide:

| Shape | M = 2 own | M = 2 padded | M = 7 own | M = 7 padded | Own over padded, M = 2 to 7, FP16 | BF16 |
|---|---|---|---|---|---|---|
| Q x K^T, K = 64 | 0.012 | 0.059 | 0.018 | 0.060 | 0.20 to 0.30 | 0.22 to 0.30 |
| K = 256 | 0.018 | 0.058 | 0.021 | 0.060 | 0.31 to 0.35 | 0.30 to 0.35 |
| K = 1024 | 0.024 | 0.058 | 0.028 | 0.060 | 0.36 to 0.49 | 0.23 to 0.48 |
| K = 4096 | 0.031 | 0.060 | 0.043 | 0.062 | 0.45 to 0.69 | 0.45 to 0.74 |
| K = 16384 | 0.047 | 0.064 | 0.132 | 0.066 | 0.73 to 2.02 (M = 2, 4, 7) | 0.59 to 1.68 |
| K = 65536 | 0.150 | 0.114 | 0.484 | 0.123 | 1.32 to 3.93 (M = 2, 4, 7) | 0.89 to 2.77 |
| Attention x V, K = 4096 | 0.049 | 0.099 | 0.049 | 0.104 | 0.47 to 0.49 | 0.23 to 0.26 |
| K = 16384 | 0.166 | 0.237 | 0.163 | 0.238 | 0.69 to 0.70 (M = 2, 4, 7) | 0.29 |
| K = 65536 | 0.661 | 0.876 | 0.666 | 0.882 | 0.75 to 0.76 (M = 2, 4, 7) | 0.28 |

**Linears pad from 50 million multiplies.** Padding costs about 60 microseconds at its smallest, and its cost then follows K x N, not M. By rows x inputs x outputs, FP16 passes Phase 2's 10% at every shape from 50.3 million up, the lowest timed being 1.42 at 2 x 6144 x 4096 and 1.46 at 3 x 2048 x 8192. Below that the answer turns on the shape: 33.5 million gives 1.03 to 1.25, and 12.6 million gives 1.02. So 2 x 4096 x 4096 stays on the own kernel and 3 x 4096 x 4096 pads. No shape was timed between 33.5 and 50.3 million, so the point is measured no closer than that. 50 million leaves BF16 at 0.93 to 1.04 for two rows of 8B's Linears, and at 0.94 at 3 x 2048 x 8192, which is under emulation on this chip and no worse than 7% against. Timed on the M1 Pro at Linear's shapes alone. Spread between runs: at most 1.10 on the Linear rows.

**What it gives (arithmetic from these rows):** on Llama 3.1 8B, over 32 layers and its four Linears, once per prompt, a partial chunk with 2 rows left over saves 78 ms and with 7 saves 568 ms. A full chunk gains nothing. Phase 3's estimate was about 260 ms at 7, from unchained timings of fewer Linears.

**Attention does not pad.** Up to K = 4096 padding loses at every row count in both precisions, with the ratios at most 0.74, and attention x V loses at every length. Q x K^T wins from K = 16384 at 4 rows and more. Llama 3.1 8B's groups are 4 rows, so at K = 65536 that is about 0.17 ms a product over 8 groups and 32 layers: **about 42 ms once per prompt at 64K, arithmetic.** It would also mean reserving about 1 MB of scratch, and attention x V would keep the own kernel either way. So attention stays with the own kernel, and Phase 4's long-context runs can reopen this. The own arm's spread reaches 1.87 at these small shapes; the padded arm's stays within 1.04.

**Padded rows give the same bits as the tile they would have joined:** rows 2 to 7, 10, 15 and 127 alone equal the same rows of a 128-row product exactly, in both precisions. Scratch is 8 x (inputs + the outputs of whole tiles) elements, for example 295 KB at 8B's 4096 x 14336. A Linear built for one row, as a generating model's output layer is, reserves none.

**35 forced failures on padding, all caught** (`D:\Claude\metal\padding\mutate.py`). 33 on the first run. The two survivors were gaps in the tests: a context's scratch not freed at its destruction, which the test could not see because its last buffer was too small for the device's count to show it; and the own kernel running the padded rows again after the tile, which small shapes cannot show since the two paths give the same bits there. Both tests were changed, and both failures are now caught. Nothing here has run on a model; a network's reservation is Phase 4's to exercise, and the footprint test holds a single Linear to it.

### First-load compile

63 kernels, up from 9. Five rounds of the real source with a comment no cache has seen: **380 to 416 ms** to compile, and 1.4 to 2.4 ms for every pipeline. Those pipeline times follow earlier runs of the same kernels, so they may be a cache's. Once the OS has the source, 0.9 ms. Against about 35 ms in Phase 1. It is paid once per Mac and kernel source, so no `metallib` is earned yet. Phase 4 can reopen it on a first run.

### What the tests do not reach

- **Rope's angle is single precision, as on CUDA.** Its error grows with position, and the bar holds to 2048. At 8192 the angle's own rounding is about 5e-4 (arithmetic), the size of FP16's bar. Filed in `Issues/Untriaged.md`.
- **Decode ahead does not pipeline here.** Metal has no event for one small readback, so `awaitToken` finishes everything encoded so far, and setting a decode position does too. The tokens are the same; the speed is Phase 4's.
- **The sampler is one SIMD-group,** with the reference kernel's 40-pass searches, not CUDA's histogram pipeline. Correct, and untimed.
- **A sliding-window cache, the FP8 cache and every quantized weight** are Phase 5's. **Several tokens in one decode or sampling call** are Phase 6's, and are refused.
