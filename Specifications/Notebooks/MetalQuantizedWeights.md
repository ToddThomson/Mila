# Metal: Quantized Weights and the KV Cache

Phase 5 of `Metal/MetalBackend.md`. **Question:** does every format in the design's section 1 run correctly, and can
TensorOps' formats carry Mila's?

**Scope (Todd, 2026-11-05).** Model runs here are Llama's: 3.2 3B in FP8 and FP4, 3.1 8B in FP4. The formats only Gemma and
Qwen use get their Metal operation and its own test against a reference here, and their model runs in Phase 6 with those
models. The FP8 KV cache is reached through `OperationSupported`, as weight formats were in Phase 4.

## 1. The gate, written before any Metal run

**Machine and rules.** As Phase 4 (`Notebooks/MetalFirstModel.md` section 1): the M1 Pro, macOS 27.2, on AC under
`caffeinate -dimsu`, the macOS preset, design section 6.

### 1.1 Each operation

Every Metal specialization this phase adds is tested on its own in FP16 and BF16 against a double-precision reference
computed on **the values as stored**: the codes and scales the operation holds, decoded on the host. So the bar measures the
operation's arithmetic, not the format's rounding, and stays Phase 3's: `kBar` in `Lib/Tests/Dnn/Components/MetalReference.h`,
5e-4 in FP16 and 4e-3 in BF16.

| Operation | Checked at |
|---|---|
| Linear, each weight format | 1 to 10, 15, 16, 127 to 130, 135, 136, 143 and 256 rows; outputs off a tile; with a bias; a pre-quantized blob and a full-precision one |
| Quantize-on-load, FP8 and FP4 | The codes and scales equal what CUDA's quantize writes for the same tensor, byte for byte, read from a file `ExportArtifact` wrote; a value to quantize that is not finite refused with the tensor's name |
| Attention with an FP8 cache | Prefill against decode; each against the reference on the cache as stored; every head size the models use; a cache written at one position read at another |

**The forced failures.** Each check above is run once against a deliberate fault in the kernel or its dispatch, and must
fail: a scale read from the wrong group, a nibble read from the wrong half, the K scale left off a score, a code rounded the
other way. Their list and results go in section 3.

### 1.2 Each model and format

**The reference.** The Phase 4 capture, `llama32_3b_instruct_reference.safetensors`: HuggingFace's FP32 run of Llama 3.2 3B
on its BF16 weights.

**The bars are Mila's CUDA run of the same weights, measured on the 5060 Ti before the Metal run.** CUDA is where these
formats are published and accepted. A quantized model is further from HuggingFace FP32 than its BF16 weights are, and by an
amount the format sets, so a bar has to come from a run of that format. `LlamaModel.HuggingFaceReference.Cuda.cpp` runs the
same checks on each package. Metal, in either precision, passes when it is no further from HuggingFace FP32 than CUDA is:

| Check | Bar |
|---|---|
| Logits, 33 steps | None not finite. Worst step's relative L2 no more than CUDA's on the same weights. An argmax differs from HuggingFace's only at a tie, as Phase 4 defines it |
| Log-likelihood, 1024 tokens | The mean negative log-likelihood's distance from HuggingFace FP32 no more than CUDA's on the same weights |
| Greedy, 32 tokens | Through `LlamaModel::load` and `generate`. HuggingFace's tokens until the first divergence, and that at a tie, or none |
| Activations | Every block's output finite over the 1024 tokens, the largest magnitude per layer recorded against FP16's 65504 |

**The weights.**

| Model | Format | Weights |
|---|---|---|
| Llama 3.2 3B | FP4 per group of 128 | The published package, `llama32_3b_instruct_fp4.safetensors` |
| Llama 3.2 3B | FP8 per channel | Written by `ExportArtifact --quantization fp8` from the BF16 weights; no package is published |
| Llama 3.2 3B | FP8 per channel, quantized at load | The BF16 weights, quantized by Metal at load; the same bars as the exported FP8, since section 1.1 holds the bytes equal |
| Llama 3.2 3B | BF16 weights, FP8 KV cache | The BF16 weights. Bars from CUDA's BF16 run with an FP8 cache |
| Llama 3.1 8B | FP4 per group of 128 | The published package, `llama31_8b_instruct_fp4.safetensors` |

**Llama 3.1 8B has no HuggingFace capture,** and none can be made here: its FP32 run needs about 32 GB, and the PC has
31.8 GB of memory and a 16 GB card. So its reference is Mila's CUDA run of the same package, captured the same way, and its
bar is the distance this phase measures between Metal and CUDA on 3.2 3B FP4, in each precision. That number is carried
from another model, not measured on this one, and the result says so.

**The forced failure.** As Phase 4: the logits run again with each decoded token one position early, and must miss.

**Then reading.** A reply to a chat prompt at the model's recommended sampling, per format, read for sense.

**Measures, after the gate.** Prefill of 512 and decode, at depth 0 and after 4096, per format, in FP16; memory against the
planner's budget. llama.cpp on the same Mac, tuned, at its nearest formats, Q8_0 for FP8 and Q4_0 for FP4: these are not
Mila's formats, and the comparison says so. Method as `Notebooks/MetalFirstModel.md` section 4.

**What this cannot say.** As Phase 4: one family, short prompts, a 1024-token sequence. A format whose model is Gemma's or
Qwen's is proven here as an operation only.

## 2. Results

### 2.1 The operations

2026-11-05, on AC. Every check of section 1.1 passes, in FP16 and BF16: 147 tests, `Linear.Quantized.Metal.cpp`,
`GroupedQueryAttention.Metal.cpp` and `QuantizationDispatch.cpp`.

**Linear on FP8 and FP4, groups of 128 and 64.** Within the bar at every row count from 1 to 1031, with outputs off a tile
and off a block of 1024, with a bias, and after any earlier call. The paths:

| Rows | FP8, FP16 | FP8, BF16 | FP4 |
|---|---|---|---|
| 1 | Own lanes, 4 outputs to a SIMD-group | as FP16 | as FP8 |
| 2 to 7 | Own lanes, 8 rows to a SIMD-group | as FP16 | as FP8 |
| 8 to 1023 | TensorOps on the bare codes in whole tiles, each output's scale after; own lanes for the rest | Own lanes up to 63, then expanded | Own lanes up to 63, then expanded |
| 1024 up | Expanded a block of 1024 outputs at a time, then TensorOps | Expanded | Expanded |

The two crossings, 64 and 1024, are where Phase 2's measurements put them within its gaps. Measured on these shapes they
are 16 and 384 (section 4).

**TensorOps refuses BF16 activations times FP8 codes** at compile time (a `static_assert` in `MPPTensorOpsMatMul2dImpl.h`);
FP16 times FP8 into FP32 compiles and is right. So BF16 FP8 takes FP4's paths. BF16 is not what the M1 selects.

**Quantize at load writes CUDA's bytes.** On Llama 3.2 3B's own weights, three tensors from each end of the network, Metal's
codes and scales equal what `ExportArtifact` wrote on CUDA, byte for byte, in FP4 (the published package) and FP8; and so on
the test's own weights, past the staging limit in two blocks, and with a row of zeros and a row whose other weights fall in
E4M3's smallest codes. **The first build missed by one code in 0.04% of the 3B's codes**, every miss at a code's boundary,
with every scale equal: Metal compiles with fast math by default, which may turn `w * (1 / scale)` into `w / scale`. Each
quantize kernel's body is now compiled in safe math (`#pragma METAL fp math_mode(safe)`, MSL section 1.6.3), scoped to the
body since the library's sources are compiled as one. A weight that is not finite is refused with the tensor's name;
CUDA's quantize writes a code for it (`Issues/Untriaged.md`).

**Attention with an FP8 cache.** Within the attention bar against a reference over the cache as it stores each row, in
prefill and decode, at every head size the models use and two that are off a tile, every grouping, three batches, a
window, the cache's last position, and rows of zeros. Decode reads the codes and each row's two scales in its two-pass
kernel. **Prefill expands instead of multiplying the bare codes,** which section 1 left open: one KV head's rows go to
the compute precision in the context's scratch, then through the uncompressed path. Folding V's scale into the
probabilities, as bare codes would need, puts them below FP16's smallest normal value, 6.1e-5: a row whose largest value is
9 has a scale of 0.02, and a probability of 1e-3 then becomes 2e-5 (arithmetic on an assumed range, not measured). The scratch is one head's rows, 2 MB at 8192 positions and a head of
128. The cache holds 1 byte a channel and 8 bytes a row of scales, against 2 bytes a channel.

### 2.2 The bars: CUDA's runs of the same weights

2026-11-05, the 5060 Ti, BF16 compute, `LlamaModel.HuggingFaceReference.Cuda.cpp`. Distances from HuggingFace FP32 on Llama
3.2 3B's capture; its BF16 run repeats Phase 4's to every digit.

| Weights | Worst logit step | Mean | Log-likelihood from HF FP32 | Greedy | Activations |
|---|---|---|---|---|---|
| BF16 (Phase 4) | 1.825e-2 | 9.86e-3 | -5.51e-4 | 32 of 32 | finite |
| FP4, the published package | 2.952e-1 | 1.94e-1 | +4.681e-2 | 32 of 32 | finite |
| FP8, exported | 7.396e-2 | 4.44e-2 | +5.774e-3 | 32 of 32 | finite |
| FP8, quantized at load | the exported FP8's, to every digit | | | 32 of 32 | finite |
| BF16, FP8 cache | 5.299e-2 | 2.69e-2 | +9.795e-4 | 32 of 32 | finite |

No argmax differs from HuggingFace's at any step, in any format. `LlamaModel.CudaReference.Cuda.cpp` wrote CUDA's FP4 runs
of the 3B and the 8B to `*_cuda_reference.safetensors` beside the weights: the 3B capture's prompt and sequence, CUDA's own
greedy continuation, its logits and its mean negative log-likelihood (3B 1.865132, 8B 1.665260). The prompt is the 3B's,
rendered by Llama 3.2's template, which the 8B's own template would render differently; for comparing two engines on the
same tokens that does not matter.

### 2.3 Each model and format on Metal

2026-11-05, on AC, `LlamaModel.HuggingFaceReference.Metal.cpp`. Llama 3.2 3B, against HuggingFace FP32; CUDA's bar beside each.

| Weights | Precision | Worst logit step (bar) | Mean | Log-likelihood from HF FP32 (bar) | Greedy | Activations |
|---|---|---|---|---|---|---|
| FP8 | FP16 | **7.599e-2** (7.396e-2) | 4.41e-2 | +5.693e-3 (5.774e-3) | 32 of 32 | finite |
| FP8 | BF16 | **7.828e-2** (7.396e-2) | 4.51e-2 | **+6.618e-3** (5.774e-3) | 32 of 32 | finite |
| FP8 at load | both | the exported FP8's, to every digit | | | 32 of 32 | finite |
| BF16, FP8 cache | FP16 | 4.447e-2 (5.299e-2) | 2.32e-2 | **+1.002e-3** (9.795e-4) | 32 of 32 | finite |
| BF16, FP8 cache | BF16 | **5.363e-2** (5.299e-2) | 2.61e-2 | +6.686e-4 (9.795e-4) | 32 of 32 | finite |

| FP4 | FP16 | 2.876e-1 (2.952e-1) | 1.93e-1 | +3.469e-2 (4.681e-2) | 32 of 32 | finite |
| FP4 | BF16 | 2.897e-1 (2.952e-1) | 1.92e-1 | +3.486e-2 (4.681e-2) | 32 of 32 | finite |

FP4 is read from the PC's later export of the published package, the same tensors with a newer config: today's tree refuses
the published package's own (`BACKLOG.md` *A model installed under v0.20 stops loading*). Decoding one position early
missed the FP4 bar in both precisions (worst step 3.93e-1 and 3.89e-1).

**FP4 on Metal scores the sequence 0.012 better than CUDA does on the same weights** (`LlamaModel.CudaReference.Metal.cpp`:
-1.21e-2 in FP16, -1.20e-2 in BF16, worst logit step 4.8e-2 and 5.0e-2 from CUDA's, greedy 32 of 32). Inferred, not tested:
CUDA's FP4 prefill multiplies activations rounded to FP8 (`CudaLinearOp.ixx`, `kUseFp8ActivationPrefill = true`), and
Metal's multiplies them in FP16 by expanded weights. So on FP4 the two engines do not compute the same thing, and the
distance between them is not an engine's noise.

**Llama 3.1 8B FP4 against CUDA** (no HuggingFace capture): worst logit step 9.77e-2 in FP16 and 1.004e-1 in BF16, mean
3.7e-2 and 3.8e-2; log-likelihood -6.43e-3 and -3.47e-3 from CUDA's 1.665260; every argmax CUDA's, greedy 32 of 32. The
bar written in 1.2, the 3B's distance from CUDA, would fail its worst step by twice, and that distance holds CUDA's FP8
activations as above.

No argmax differs from HuggingFace's at any step. **As written, the gate fails, by 1% to 15%, on one number of each FP8 run
in bold**; FP4 passes. Step by step, Metal FP16 on FP8 is further from HuggingFace than CUDA at 12 of 32 steps and closer at 20, the worst
step is step 10 in both, and the two differ there by 3%. Both runs carry the format's same rounding, which is most of the
distance; what separates them is each engine's own, which falls on either side. So "no further than CUDA" compares two
draws of that small difference, and is close to a coin flip at the worst step. Holding Metal to CUDA's run of the same
weights instead fails on FP4 for CUDA's FP8 activations, above. Put to Todd, 2026-11-05: keep the bar and record the
misses; or capture HuggingFace FP32 on each format's weights decoded to full precision, the exact model a format defines,
and hold every engine to Phase 4's bars against it. Todd: the second "is likely the best choice"; noted for a later session.

## 3. Forced failures

2026-11-05, `D:\Claude\metal\phase5\mutate.py`: each change made alone in the Mac's copy, built, and the tests of section
1.1 run. **27 of 29 caught.**

| Caught | Changes |
|---|---|
| Codes | FP4 nibbles from the wrong half; FP8 ties rounded away from even; FP8's smallest values coded as normal; an FP4 midpoint moved |
| Products | Lanes or expansion reading the first group's scale; lanes reading one row's activations for all; expansion ignoring its block's first output; FP8's scales not applied after bare codes; the outputs or the rows past the tiles left undone; an expanded block written at output 0; expansion's scratch not counted |
| Quantize | Fast math in the kernels; a row of zeros scaled 0; weights that are not finite not counted; an odd code from two lanes on; every staging block reading or writing the first block's rows |
| FP8 cache | K's scale left off a score; V's off a weight; a row written at its token and not its position; prefill expanding with the first row's scale, or half the rows; prefill's scratch not counted; the scales not counted as state |
| Dispatch | The FP8 cache dispatch asking CUDA's old question |

**The two survivors are both in the cache's row write.** Fast math there changes nothing the tests see: the pragma was
there to round as CUDA's write does, no gate holds the cache's bytes to CUDA's, and a code's difference in one channel is
far below attention's bar. It was taken out. An inverse of 1 for a row of zeros is equivalent, since every value in the
row is zero; the guard stays, because 1 / 0 would make 0 times infinity.

## 4. The crossings

2026-11-05 and 2026-11-06, on AC, FP16, `D:\Claude\metal\phase5\crossings.py`: each path forced for every row count by
building `mila-metal-quantized-timing` against changed constants in `MetalLinearOp`, on Llama 3.2 3B's four shapes and 3.1
8B's two, at 8 to 16, 32 to 128 and 256 to 2048 rows. Logs `phase5-crossings.log` and `phase5-crossings2.log` beside it.

**Own lanes against expanded weights (FP4, and FP8 in BF16): expanded wins from 16 rows, not 64.** At 8 to 15 rows own lanes
are 8 to 68% faster; at 16 expanded is 1.4 to 1.8 times faster on every shape, and at 32 and 48 rows 2.1 to 2.5 times.

| Rows | 3B gate_up lanes / expanded, ms | 8B down lanes / expanded, ms |
|---|---|---|
| 15 | 2.37 / 3.46 | 2.91 / 4.88 |
| 16 | 2.39 / 1.56 | 2.92 / 2.15 |
| 32 | 4.65 / 2.17 | 5.88 / 3.14 |
| 48 | 6.99 / 2.94 | 8.87 / 4.25 |

**FP8's bare codes against expanded weights: expanded wins from 384 rows, not 1024,** by 4 to 13% from 384 to 2048 on
five of the six shapes, and bare codes win by 2 to 5% at 320. On the 8B's down projection bare codes win at every row
count measured, by 7% at 384 and 0.4% at 768.
