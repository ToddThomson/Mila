# Metal: Qwen 3.8 27B

Phase 6 of `Metal/MetalBackend.md`, its second model. **Question:** does FP16 hold up on Qwen 3.8 27B, with its DeltaNet
layers' recurrence, in the formats its packages carry?

**Scope (Todd, 2026-11-06).** Ahead of Phase 8, because an agent on Metal opens a model through every family's type, and
Qwen's needs operations Metal did not have. Built here, each in FP16 and BF16:

- the elementwise activation (every functor CUDA's carries; Qwen uses SiLU and sigmoid);
- the depthwise causal convolution and its state update;
- the gated delta rule, its state FP32 in either precision, in the recurrent form for prefill and decode;
- Linear on two- and three-bit codebook weights, at the group sizes CUDA carries;
- the tensor operations CUDA's math partition has: add, subtract, multiply, divide and sum.

And one library change (Todd, 2026-11-06): Qwen's weight dispatch builds a plan wherever its Linears exist on the device
at the precision (`OperationSupported`), where it built quantized plans for BF16 compute only. CUDA builds what it built.

**What fits this Mac.** The GPU's working set is 11.84 GiB (`Notebooks/MetalDeviceAndMemory.md`). The packages on the PC:
`fp4` 15.09 GB and `fp4-cb3` 14.30 GB do not fit; `cb2-3` is 11.05 GB, 10.29 GiB, and fits at a short context only. The
agentic build is `fp4-cb3` (`Qwen3.8.md`, 2026-10-19), so on a 16 GB Mac Qwen 3.8 27B can be gated for correctness and
not offered as an agentic model.

## 1. The gate, written before any Metal run

**Machine and rules.** As Phase 4 (`Notebooks/MetalFirstModel.md` section 1).

### 1.1 Each operation

Against a double-precision reference on the values as stored, at Phase 3's bar (`kBar`, 5e-4 in FP16, 4e-3 in BF16),
except where a row says otherwise.

| Operation | Checked at |
|---|---|
| Elementwise activation | Every functor, at a decode row, a prefill and Qwen's widths; inputs large either way |
| Causal convolution | Kernel 4 at dilation 1 and 2, with and without bias; a prefill from position 0, a prefill continuing one, and decodes after it, each against the whole sequence convolved at once; the state after each, exactly; Qwen's 10240 channels |
| Gated delta rule | CUDA's oracle vectors (2 key heads, 6 value heads, dims 4), whose bar is CUDA's BF16 one, 2e-2, in both precisions; a double-precision recurrence on Qwen's geometry (16 key heads, 48 value heads, dims 128) over a prefill and decodes after it, output and the FP32 state |
| Linear, codebook | Two bits at 32 and 64, three bits at 64 and 128; Phase 5's row counts; Qwen's shapes, 5120 by 17408 and back |
| Tensor math | Each operation on a few elements and at Qwen's widths, in place too; sum against a double sum |

**The forced failures.** Each check runs once against a deliberate fault and must fail; the list and results go in
section 3.

### 1.2 The model

As Gemma's (`Notebooks/MetalGemma.md` 1.1 and 1.3): HuggingFace FP32 on the package's weights decoded to full precision,
Phase 4's bars measured from HuggingFace BF16 and Mila CUDA BF16 on the same weights and tokens, and the activation ranges
per layer against FP16's 65504, which is FP16's question on this family. Two runs:

- **The gate, the four-layer build** `qwen38_27b_l4_2p9bit.safetensors` (5.38 GB), the one CUDA's load tests use, three
  DeltaNet layers and one attention layer. Fits with room.
- **One full run, `cb2-3`,** at a short context, read for sense, its activations recorded. Near the working set, so
  macOS's paging is watched and reported.

**The capture** is `Lib/Tools/Converters/Qwen/qwen38_BF16/hf_qwen38_l4_plan_reference.py`: the chat prompt, HuggingFace
FP32's greedy continuation of 32 tokens with the logits teacher-forced, and a PG-19 book (30312) in a user turn, 2048
tokens, with each next token's log-probability and the logits at the last position of every 32-token piece. The weights
are the package's as Mila runs them under its plan: the codebook tensors decoded as stored, and the roles the plan holds
in FP4 but the package stores in BF16 (the attention layer's projections and the head) quantized to FP4 as the load does.
Its `--check` passes, 2026-11-07: the decode equals `Tools/Quantization/packing.py`'s on three tensors, the FP4 round trip
equals packing.py's quantize and dequantize, the tensors the plan leaves unquantized map back to the BF16 checkpoint
exactly, and dropping the three-bit high plane moves a tensor by 1.73. The FP4 rounding runs on the host: CUDA's torch
divides by a constant through its reciprocal, which moved some scales by their last bit.

**The bars, measured 2026-11-07** (`D:\Claude\metal\qwen\hf_bf16_distances.py` and
`QwenModel.HuggingFaceReference.Cuda.cpp` on the 5060 Ti). HuggingFace FP32 reads 9.443303 nats over the book.

| Check | HuggingFace BF16 | Mila CUDA BF16 | Bar |
|---|---|---|---|
| Logits, worst step (mean) | 2.844e-3 (2.391e-3) | 3.884e-3 (2.422e-3) | 3.884e-3 |
| Log-likelihood from HF FP32 | +5.076e-4 | +8.226e-4 | 8.226e-4 |
| KL, mean over 64 pieces (worst) | 1.930e-5 (5.498e-5) | 5.869e-5 (1.209e-4) | 5.869e-5 |
| Greedy, 32 tokens | | 32 of 32 | ties only |
| Activations | | finite; largest 60, after layer 3 | finite |

No argmax differs from HuggingFace's in any CUDA check.

**What this cannot say.** Whether `fp4-cb3` holds on Metal: it does not fit here. Four layers of 64 read 9.44 nats and
continue the chat with 32 spaces, so this gate compares engines number for number and says nothing of the model's
quality; the greedy check in particular is weak here, every step the same token. The largest activation is 60, far
inside FP16's range, but over four layers only: FP16's range on the whole model is the `cb2-3` run's question.

## 2. Results

### 2.1 The operations

2026-11-07, on AC. Every check of section 1.1 passes in FP16 and BF16: 53 tests in `Activation.Metal.cpp`,
`CausalConv1d.Metal.cpp`, `GatedDeltaRule.Metal.cpp`, `Linear.Codebook.Metal.cpp` and `Metal.TensorOps.Metal.cpp`.

- **The gated delta rule** runs one threadgroup per value head and block of 32 columns; four lanes of a SIMD-group hold
  one column's FP32 state between them, their partial sums met by two shuffles. The recurrent form serves prefill and
  decode; CUDA's chunked form is not built, and is built only if a timing earns it. A head of keys up to 256.
- **Codebook Linear** reads two-bit codes eight to a ushort and three bits' high plane eight to a byte, the table in
  registers, on FP4's paths: own lanes below 16 rows, weights expanded a block of 1024 outputs at a time from 16. Both load
  only as stored, as on CUDA.
- **The tensor math** (add, subtract, multiply, divide, sum) was missing on Metal, and Qwen's attention output gate
  multiplies through it.
- **The four-layer build loads and generates** through `QwenModel::load` in both precisions (`QwenModel.Load.Metal.cpp`),
  the attention layer's projections and the head quantized to FP4 at load.

### 2.2 The model

2026-11-07, `QwenModel.HuggingFaceReference.Metal.cpp`, the bars of section 1.2. **All 10 checks pass in FP16 and BF16.**
Log `D:\Claude\metal\qwen\gate-metal.log`.

| Check | Bar | Metal FP16 | Metal BF16 |
|---|---|---|---|
| Logits, worst step (mean) | 3.884e-3 | 3.544e-4 (2.741e-4) | 3.064e-3 (2.423e-3) |
| Log-likelihood from HF FP32 | 8.226e-4 | +1.429e-5 | +4.104e-4 |
| KL, mean over 64 pieces (worst) | 5.869e-5 | 3.290e-7 (1.311e-6) | 1.782e-5 (4.788e-5) |
| Greedy, 32 tokens | ties only | 32 of 32 | 32 of 32 |
| Activations | finite | finite; largest 60.19, after layer 3 | finite; largest 60, after layer 3 |

No argmax differs from HuggingFace FP32's in any check. **FP16 is about eleven times closer to HuggingFace FP32 than
either BF16 run** on the logits, and its KL is sixty times below HuggingFace BF16's.

**The forced failure misses, but narrowly in FP16:** each decoded token one position early puts the worst step at
4.395e-3 in FP16 and 4.947e-3 in BF16, 13% and 27% past the bar. Only one of the four layers is attention, and a position
moves nothing in the three DeltaNet layers, so the shift reaches the logits through that one layer. On the whole model it
would cross sixteen.

## 3. Forced failures

2026-11-07, `D:\Claude\metal\qwen\mutate_qwen.py` (Gemma's engine): each change made alone in the Mac's copy, built, and
the tests of section 2.1 run. **30 of 31 caught.**

| Caught | Changes |
|---|---|
| Activation | SiLU without its x; sigmoid of -x; GELU's cubic coefficient; Mish's softplus cut over at 2; leaky ReLU without its alpha, or with an alpha not the config's |
| Convolution | The taps reversed; the dilation ignored; the retained rows ignored; the bias ignored; the retained rows stored reversed |
| Delta rule | The decay never applied; k not normalized; q not scaled by 1 / sqrt(dk); the output's parts not summed; one shuffle of the parts' sum missing; value heads grouped over key heads the wrong way; beta of -b; dt_bias ignored; the state written back halved; the second block of columns overlapping the first |
| Codebook | Two-bit codes from the wrong pair; three bits' high bit ignored; the high plane's row at the codes' stride; the table read in reverse; a row's length taken as a nibble format's |
| Tensor math | Multiply adds; divide inverted; the sum of the first SIMD-group only |
| Dispatch | Every quantized plan refused |

**The survivor is a branch no test reaches:** softplus's small-input form, taken where `a + dt_bias` is below -9.2, set to
zero. The tests' inputs are normal with unit spread, and there the change moves a step's decay by at most 1e-4. Not
closed here.
