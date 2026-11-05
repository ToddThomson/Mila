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

**By the rule TensorOps wins from M = 2, as Phase 2's arithmetic expected.** 10 of the 36 shapes differed between runs by more than the 15% that makes a result unstable, all at small shapes, up to 2.4 times. The own kernel's error matches: 2.0e-4 to 2.1e-4. **Linear does not use this yet.** Padding needs a scratch buffer and two copies, and it was agreed that these rows go through the own kernel until this was timed. What it costs today (arithmetic): on Llama 3.1 8B, a partial chunk of 7 rows takes about 8 ms more per layer in its two MLP Linears and its fused projection, so about 260 ms once per prompt over 32 layers. A full chunk costs nothing.

### First-load compile

63 kernels, up from 9. Five rounds of the real source with a comment no cache has seen: **380 to 416 ms** to compile, and 1.4 to 2.4 ms for every pipeline. Those pipeline times follow earlier runs of the same kernels, so they may be a cache's. Once the OS has the source, 0.9 ms. Against about 35 ms in Phase 1. It is paid once per Mac and kernel source, so no `metallib` is earned yet. Phase 4 can reopen it on a first run.

### What the tests do not reach

- **Rope's angle is single precision, as on CUDA.** Its error grows with position, and the bar holds to 2048. At 8192 the angle's own rounding is about 5e-4 (arithmetic), the size of FP16's bar. Filed in `Issues/Untriaged.md`.
- **Decode ahead does not pipeline here.** Metal has no event for one small readback, so `awaitToken` finishes everything encoded so far, and setting a decode position does too. The tokens are the same; the speed is Phase 4's.
- **The sampler is one SIMD-group,** with the reference kernel's 40-pass searches, not CUDA's histogram pipeline. Correct, and untimed.
- **A sliding-window cache, the FP8 cache and every quantized weight** are Phase 5's. **Several tokens in one decode or sampling call** are Phase 6's, and are refused.
