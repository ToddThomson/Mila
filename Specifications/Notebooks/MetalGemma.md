# Metal: Gemma 4 12B

Phase 6 of `Metal/MetalBackend.md`, its first model. **Question:** does FP16 hold up on Gemma 4 12B, the first agentic
model on Metal, in the format it was trained for?

**Scope (Todd, 2026-11-06).** Gemma 4 12B Instruct in Google's QAT Q4_0, Mila's package
`gemma4_12b_it_qat_q4_0.safetensors` (6.95 GB). The operations it needs that Metal does not have are built here: Linear on
Q4_0 (`PerGroupInt4<32>`), the six-bit table (`PerGroupInt6<32>`) in the token embedding and the tied head, GeGLU, and the
sliding-window ring cache its local layers hold. The drafter, Qwen 3.8 27B and the 26B mixture of experts come later.
Chat on macOS follows this gate (design Phase 8).

## 1. The gate, written before any Metal run

**Machine and rules.** As Phase 4 (`Notebooks/MetalFirstModel.md` section 1): the M1 Pro with 16 GB, macOS 27.2, on AC
under `caffeinate -dimsu`, the macOS preset, design section 6.

### 1.1 The reference: the package's own weights, decoded

**Decided (Todd, 2026-11-06):** a quantized model is held to HuggingFace FP32 running that format's weights decoded to full
precision, the exact model the format defines, with Phase 4's bars against it. Comparing to CUDA's run of the same weights
compared two engines' small differences, and failed FP8 by chance (`Notebooks/MetalQuantizedWeights.md` 2.3). For Gemma
it is also the model as trained: Google's QAT fitted these weights to Q4_0.

`Lib/Tools/Converters/Gemma/gemma_4_BF16/hf_gemma4_12b_q4_0_reference.py` decodes every Q4_0 and six-bit tensor of the
package on the device and runs HuggingFace's Gemma 4 at FP32, one decoder layer at a time through
`hf_gemma_layer_stream.py`'s driver: the FP32 model is about 48 GB, past the PC's 31.8 GB. Run once on the 5060 Ti, and
again at BF16 on the same tokens.

- A chat prompt rendered by the model's own template at its defaults (34 tokens, thinking off), HuggingFace FP32's greedy
  continuation of 32 tokens, and the logits at the prompt's last position and at each continued position, the
  continuation teacher-forced.
- A PG-19 test book, 30312, as the G2 harnesses give it to an instruct model: `<bos>`, a user turn "Continue this book.",
  the model's turn with thinking off, then the book with its line wraps joined, 2048 tokens in all. The local layers'
  window is 1024, so the sequence runs past it. Recorded: the log-probability of each next token; the logits at the
  last position of each 32-token piece, the 64 positions a prefill in pieces of 32 returns; and the logits at positions
  992 to 1055, either side of the window's edge.

**Not wikitext (Todd, 2026-11-06).** The first capture read 5.89 nats on wikitext, and raw wikitext cannot judge a
precision change on this model: rounding that flattens its distributions raises the probability of the tokens that occur
(`ModelFamilyParity.md` 8.2, G1). The G2 decision of 2026-09-26 put Gemma's measurements on PG-19 with KL divergence, and
this gate follows it. The distributions are kept at 64 positions, not all 2048: a Mila prefill returns its last
position's logits only, so 64 pieces measure prefill's own arithmetic with no library change, where every position
would need a decode at each.

Logits are recorded before the final softcap, and log-probabilities after it, as HuggingFace computes them.

**The decoder is checked first** (`--check`): decoded Q4_0 tensors of a local and a global layer are 4.9e-2 to 5.3e-2
relative L2 from the unquantized QAT weights the package was quantized from, Q4_0's own rounding, and the table's rows
1.3e-2. The same tensors with each byte's nibbles swapped are 1.41 from them. 2026-11-06.

### 1.2 Each operation

Every Metal specialization this phase adds is tested on its own in FP16 and BF16 against a double-precision reference on
the values as stored, at Phase 3's bar (`kBar` in `Lib/Tests/Dnn/Components/MetalReference.h`, 5e-4 in FP16, 4e-3 in
BF16), as Phase 5's were.

| Operation | Checked at |
|---|---|
| Linear, Q4_0 | Phase 5's row counts, 1 to 10, 15, 16, 127 to 130, 135, 136, 143 and 256; outputs off a tile; Gemma's shapes, the global layers' 8704-row QKV and 4096-column output among them |
| Linear, six-bit | As Q4_0, and the tied head's 3840 inputs at 32768 of its 262144 outputs, at 1 and 16 rows; the whole head's weights would take 8 GB of host memory to check |
| Token embedding, six-bit table | Rows gathered from either end of the table and a repeated id, with the embedding scale |
| GeGLU | Against GELU's tanh form, the one Gemma 4 names; Gemma's 15360 hidden, prefill and decode; gates large either way |
| Attention, ring cache | Prefill shorter than the window, equal to it and past it; decode before, across and after the ring's wrap; heads of 256 and 512; the global layers' K equal to V |

**The forced failures.** Each check runs once against a deliberate fault and must fail: a Q4_0 nibble from the wrong half,
the offset of 8 left off, a six-bit code's high bits from the wrong pair, GELU without its tanh term, the ring written at
the position and not modulo the window, a key past the window attended. Their list and results go in section 3.

### 1.3 The model

**The bars are two accepted BF16 runs of these weights, measured,** as Phase 4's: Mila on CUDA at BF16, the device and
format Gemma 4 12B is published for, through `GemmaModel` on the 5060 Ti; and HuggingFace's BF16 run on the decoded
weights. Each check's bar is the further of the two from HuggingFace FP32, measured before the Metal run and written here.

| Check | Bar |
|---|---|
| Logits, 33 steps | None not finite. Worst step's relative L2 no more than the further BF16 run's. An argmax may differ from HuggingFace's only where its two tokens are closer than that step's largest error |
| Log-likelihood, 2048 tokens | The mean negative log-likelihood within the further BF16 run's distance of HuggingFace FP32's, in either direction |
| KL divergence, 64 positions | Prefill in pieces of 32: the mean KL from HuggingFace FP32's distribution to the engine's, after the softcap, no more than the further BF16 run's |
| Across the window's edge | Prefill 992 tokens, then decode 64 teacher-forced: worst row's relative L2 no more than the logits bar |
| Greedy, 32 tokens | Through `GemmaModel::load` and `generate`, planned as an application plans it. HuggingFace's tokens until the first divergence, and that at a tie as above, or none |
| Activations | Every block's output finite over the 2048 tokens, the largest magnitude per layer recorded against FP16's 65504. This is FP16's question on this family |

**The bars, measured 2026-11-06** (`hf_bf16_distances.py` beside the logs in `D:\Claude\metal\gemma`, and
`GemmaModel.HuggingFaceReference.Cuda.cpp` on the 5060 Ti). HuggingFace FP32 reads 3.714956 nats over the book.

| Check | HuggingFace BF16 | Mila CUDA BF16 | Bar |
|---|---|---|---|
| Logits, worst step | 2.711e-2 (mean 1.535e-2) | 2.399e-2 (mean 1.475e-2) | 2.711e-2 |
| Log-likelihood from HF FP32 | -3.984e-3 | +2.657e-3 | 3.984e-3 |
| KL, mean over 64 positions | 7.608e-4 | 1.869e-3 | 1.869e-3 |
| Across the window's edge, worst row | 2.540e-2 | **3.339e-1** (mean 2.713e-2) | 2.540e-2 |
| Greedy | | 32 of 32 | ties only |
| Activations | | finite; largest 238, after layer 25 | finite |

No argmax differs from HuggingFace's in any CUDA check. **Across the window's edge, CUDA's worst is not a bar:** rows 19
and 57 of 64 (positions 1011 and 1049) are 2.9e-1 and 3.3e-1 from HuggingFace FP32 and as far from its BF16 run, where
every other row is at most 3.4e-2 and HuggingFace BF16's worst is 2.5e-2. So that bar is HuggingFace BF16's alone, and the
two rows are filed (`Issues/Untriaged.md`) for CUDA. A bend of the gate as written, put to Todd.

**The forced failure.** As Phase 4: the logits run again with each decoded token one position early, and must miss.

**Then reading.** A reply to a chat prompt at Gemma's recommended sampling, in each precision, read for sense; then the
same through `mila-chat` once Chat runs on macOS.

**Measures, after the gate.** Prefill of 512 and decode at depth 0 and after 4096, in FP16; memory against the planner's
budget and `recommendedMaxWorkingSetSize`; first-load compile cost. llama.cpp on the same Mac on Google's own Q4_0 GGUF,
tuned, like for like: the body's codes are the same, and its table is Q6_K where Mila's is six-bit per 32, which the
comparison says. Method as `Notebooks/MetalFirstModel.md` section 4.

**What this cannot say.** One model, a short prompt and a 2048-token sequence, far below the 64K the agentic bar asks.
Long context on Metal is measured separately, once the model runs.

## 2. Results

### 2.1 The operations

2026-11-06, on AC. Every check of section 1.2 passes in FP16 and BF16: 123 tests in `Linear.Integer.Metal.cpp`,
`TokenEmbedding.Metal.cpp`, `Geglu.Metal.cpp` and `GroupedQueryAttention.Metal.cpp`, the attention's earlier tests
among them unchanged.

- **Q4_0 and six-bit Linear** take the paths FP4 takes: own lanes below 16 rows, weights expanded a block of 1024 outputs
  at a time from 16. Their scales stay FP16 as the package stores them. Both load only as stored: a full-precision
  weight is refused with its name. CUDA quantizes both at load, so this is a gap between the backends (section 4).
- **The six-bit table** is decoded as it is gathered, each output rounded once from 32 bits, as CUDA's gather does,
  and equals the library's own decode (`Int6Packing.ixx`) to the bit.
- **GeGLU** computes GELU's tanh form in 32 bits with the precise tanh, narrowed at the store.
- **The ring** holds the window and a prefill chunk less one rows, as on CUDA; position p is in row p % rows, and the
  softmax and decode map each row back to the position it holds. For a cache of every position the map is the identity,
  so one set of kernels serves both. The reference holds every position, so a row read from the wrong turn of the ring
  misses it.

**The family's other operations (Todd, 2026-11-06: Metal provides the operations CUDA does).** Gemma's network also
instantiates the 26B's router and expert bank and the drafter's sampler, so they are built here too, each checked against a
double-precision reference in FP16 and BF16, 40 tests in `Router.Metal.cpp`, `MixtureOfExperts.Metal.cpp` and
`Sampling.Metal.cpp`:

- **The router** selects as CUDA's does, one SIMD-group a row, the lower index on a tie; its exponentials are 32-bit, where
  CUDA's are 64, since Metal has no double.
- **The expert bank** runs CUDA's two passes, unquantized, FP4 per group and Q4_0, one SIMD-group a cell, so every token's
  output is the one it gets alone, through `forward` and a multi-token decode of up to eight, to the bit. FP4 is quantized at
  load through Linear's own quantizer (`MetalLinearOp::quantizeRows`); Q4_0 loads as stored only, as Linear's does.
- **The sampler** draws up to eight rows in one call, each as the one-row entry draws it, with each row penalizing the
  drafts before it, and walks a speculative round's acceptance on the device.

The multi-token decode through the rest of the network (Linear, attention, the drafter's shared-cache attention) is still
CUDA's only, so speculative decoding does not yet run on Metal.

### 2.2 The model

2026-11-06, on AC, `GemmaModel.HuggingFaceReference.Metal.cpp`, the bars of section 1.3. **All 12 checks pass in FP16 and
BF16.** Log `D:\Claude\metal\gemma\gate-metal.log`.

| Check | Bar | Metal FP16 | Metal BF16 |
|---|---|---|---|
| Logits, worst step (mean) | 2.711e-2 | 2.617e-3 (1.649e-3) | 1.778e-2 (1.366e-2) |
| Log-likelihood from HF FP32 | 3.984e-3 | +2.420e-4 | +8.456e-4 |
| KL, mean over 64 positions (worst) | 1.869e-3 | 8.442e-6 (7.704e-5) | 5.758e-4 (4.105e-3) |
| Across the window's edge, worst row (mean) | 2.540e-2 | 2.654e-3 (1.682e-3) | 2.019e-2 (1.258e-2) |
| Greedy, 32 tokens | ties only | 32 of 32 | 32 of 32 |
| Activations | finite | finite; largest 238.8, after layer 25 | finite; largest 240, after layer 25 |

No argmax differs from HuggingFace FP32's in any check. **FP16 is ten times closer to HuggingFace FP32 than either BF16
run** on the logits and across the window, and its KL is ninety times below HuggingFace BF16's: this family's activations,
largest 239 against FP16's 65504, leave FP16 its full precision. Metal BF16 is as close as the two accepted BF16 runs.

**Across the window's edge Metal shows nothing of CUDA's two outlier rows** (section 1.3): its worst row is no further
than its worst logits step in either precision. So those rows are CUDA's own, not the ring's design nor the positions'
tokens (`Issues/Untriaged.md`).

**The forced failure misses:** each decoded token one position early puts the worst step at 6.74e-1 in FP16 and 6.73e-1 in
BF16, 25 times the bar.

## 3. Forced failures

2026-11-06, `D:\Claude\metal\gemma\mutate_gemma.py` (Phase 5's engine): each change made alone in the Mac's copy, built,
and the tests of section 1.2 run. **20 of 21 caught.**

| Caught | Changes |
|---|---|
| Q4_0 and six bits | Nibbles from the wrong half; the offset of 8, or of 32, left off; high bits from the wrong pair; the high plane read from the wrong offset; an FP16 scale read as its row's first; six bits' row length taken as Q4_0's |
| The table | High bits from the wrong pair; the scale from the wrong group; the row length taken as Q4_0's |
| GeGLU | Without its tanh term; without its cubic term; gate and up swapped |
| The ring | Written at the position, not modulo the ring; decode reading a key at its position; prefill taking a row as its position; a key past the window attended; the ring only as long as the window; prefill's columns the positions; the ring's state counted as the context's |

**The survivor is equivalent through the component:** the attention operation refuses a prefill longer than the chunk
its ring was built for, but `GroupedQueryAttention`'s prefill output holds one chunk, so a longer one is refused there
first. The operation's guard stays, for a caller that drives it directly.

## 4. Open

- **Q4_0 and six bits have no quantize-at-load on Metal**, where CUDA has both (`CudaLinearOp`, `CudaTokenEmbeddingOp`).
  The gate needs only the stored package. Told Todd 2026-11-06: the design given that day said CUDA loads them stored
  only, and that was wrong.
