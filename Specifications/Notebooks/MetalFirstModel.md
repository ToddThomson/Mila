# Metal: The First Model

Phase 4 of `Metal/MetalBackend.md`. **Question:** does FP16 hold up on a real model, and what does each way cost?

## 1. The gate, written before any Metal run of the model

**Machine and rules.** The M1 Pro, macOS 27.2, on AC under `caffeinate -dimsu`, with the macOS preset. Section 6 of the
design applies.

**The model.** Llama 3.2 3B Instruct on its BF16 weights (`llama32_3b_instruct_bf16.bin`, 7.2 GB), as
`LlamaTransformer<Metal, FP16>` and `<Metal, BF16>`, at a context of 1024 and the prefill chunk the library's own rule
chooses. FP16 reads the BF16 weights and converts them once at load (design section 5).

**The reference.** `Lib/Tools/Converters/Llama/hf_llama32_3b_reference.py`, run once on the 5060 Ti: HuggingFace's FP32 run,
and its BF16 run on the same tokens.

- A chat prompt rendered by the model's own template (52 tokens, date fixed), HuggingFace FP32's greedy continuation of
  32 tokens, and the logits at the prompt's last position and at each continued position, the continuation teacher-forced.
- The log-probability of each next token over the first 1024 tokens of the wikitext-2 test split.

HuggingFace's own BF16 run sits at a worst step of 2.19e-2 relative L2 from its FP32 one, mean 1.03e-2, and at a mean
negative log-likelihood of 1.819224 against FP32's 1.818327.

**The bars are two accepted BF16 runs of this model, measured.** `LlamaModel.HuggingFaceReference.Cuda.cpp` runs the same
checks at BF16 on CUDA, the format and device Llama 3.2 3B is published for, on the 5060 Ti:

| | Worst logit step, relative L2 | Mean negative log-likelihood, from HuggingFace FP32 | Greedy |
|---|---|---|---|
| Mila CUDA BF16 | 1.825e-2 (mean 9.86e-3) | -5.51e-4 | 32 of 32 |
| HuggingFace BF16 | 2.185e-2 (mean 1.03e-2) | +8.97e-4 | not run |

Metal in either precision passes when it is no further from HuggingFace FP32 than the further of the two:

| Check | Bar |
|---|---|
| Logits, 33 steps | No logit not finite. Worst step's relative L2 no more than 2.185e-2. An argmax may differ from HuggingFace's only where HuggingFace's two tokens are closer than that step's largest error, since logits off by that much anywhere cannot order them |
| Log-likelihood, 1024 tokens | The mean negative log-likelihood within 8.97e-4 of HuggingFace FP32's 1.818327, in either direction |
| Greedy, 32 tokens | Through `LlamaModel::load` and `generate`, which plan the deployment as an application does. HuggingFace's tokens until the first divergence, and that divergence at a tie as above, or none |
| Activations | Every block's output finite over the 1024 tokens. The largest magnitude per layer is recorded against FP16's 65504. On CUDA BF16 it is 5.75 after layer 0, 418 from layer 1 to 26 and 126 after the last |

**The forced failure.** The logits run again, each decoded token one position early: it must miss the bar. The tiny
model's version of it (below) missed by 3 to 10 times its bar.

**First, the tiny model.** `Llama.HuggingFaceReference.Metal.cpp` runs the whole network on the tiny HuggingFace Llama
the CUDA tests use, unscaled and with Llama 3's frequency scaling, against the CUDA test's own BF16 bar of 1e-1, in both
precisions. It is a wiring check before the real one.

**Then reading.** A reply to a chat prompt at the model's recommended sampling, in each precision, read for sense.

**Measures, after the gate.** Prefill and decode rate both ways; memory against the planner's budget and how near
`recommendedMaxWorkingSetSize` the model runs; weights wrapped without a copy; llama.cpp on the same Mac.

**What this cannot say.** One model, short prompts, and a 1024-token sequence. A range problem that only shows at long
context or on another family is Phase 6's, and the gate is per model.

## 2. Results

### The tiny model

All six pass, 2026-11-05: FP16 at about 1e-2 relative L2 from HuggingFace FP32 at every step, BF16 at about 2e-2, unscaled
and scaled. The one argmax disagreement, at step 1, is the tie the CUDA test already records (HuggingFace's two tokens
0.0087 apart). Decoding one position early missed at every step but the first, by 0.3 to 1.0 relative L2.

The first build needed `Llama.ixx`'s `backward` kept from compiling on Metal (section 3).

### Llama 3.2 3B

2026-11-05, on AC at both ends, no sleep in the log during the run. Built with the three `Lib/Src` changes of section 3.

| | Worst logit step | Mean | Argmax | Log-likelihood from HF FP32 | Greedy | Largest activation |
|---|---|---|---|---|---|---|
| Bar | 2.185e-2 | | ties only | within 8.97e-4 | ties only | finite |
| **Metal FP16** | **1.93e-3** | 1.14e-3 | 33 of 33 | **+6.8e-5** | 32 of 32 | 420, all finite |
| Metal BF16 | 2.15e-2 | 9.85e-3 | 33 of 33 | **+1.25e-3, misses** | 32 of 32 | 418, all finite |
| Mila CUDA BF16 | 1.83e-2 | 9.86e-3 | 33 of 33 | -5.5e-4 | 32 of 32 | 418, all finite |
| HF BF16 | 2.19e-2 | 1.03e-2 | | +9.0e-4 | | |

**FP16 holds on this model, and decision 3.2 stands for it.** It is about ten times closer to HuggingFace FP32 than any
BF16 run, by logits and by log-likelihood. Its activations peak at 420, after the first layer, about 156 times below
FP16's largest value, and nothing overflowed.

**Metal BF16 misses the log-likelihood bar**, at +1.25e-3 against 8.97e-4: 1.4 times HuggingFace's own BF16 distance,
and on the other side of FP32 from Mila's CUDA BF16. Its logits pass, at the same distance as the other BF16 runs. BF16
is not what the M1 selects, so this does not block the model, but it says Metal's BF16 path loses more over a sequence
than CUDA's. Where is not found; the bar is not moved.

**The forced failure was caught** in both precisions: each token decoded one position early put the worst step at
2.75e-1, 13 times the bar.

**Load and first call.** Each test loads the 7.2 GB file and builds the network; the logits test took 8 to 16 s whole.
Not yet timed apart.

## 3. What the library needed

Found by building the model, tried first on the Mac's copy, and agreed by Todd (2026-11-05). None is a `#if` for Metal:

1. **`LlamaTransformer::backward` compiled every operation's `backward`**, since it is virtual, and Metal's operations
   declare none (Phase 3's decision). `DeviceTypeTraits` now says whether a device trains (`supports_training`: CUDA
   yes, Metal and the host no), and Llama's and GPT's `backward` throw where it does not. Gemma and Qwen are
   inference-only and threw already.
2. **`loadParameters` synchronized only on CUDA.** On Metal that was not a defect, though it was proposed as one: the copy
   into a Metal parameter, the BF16 to FP16 conversion included, runs on the host and is complete when it returns
   (`MetalTensorOps.Transfer.ixx`, Phase 3), and the two forced failures that put the CUDA-only condition back changed
   nothing. All five networks now synchronize on every device, so the reader's reuse of its staging slot is safe by the
   network's own contract and not by how a device happens to copy. What the extra waits cost a Metal load is not timed.
3. **`dispatchWeightQuantization` decided a format by "BF16 compute"**, so at BF16 on Metal it instantiated FP4, FP8 and
   Q4_0 Linears that Metal does not have until Phase 5. It now asks `OperationSupported` for the device, precision and
   format, the test the rule stood in for, and on CUDA every format gets the same answer as before. A refusal names
   the format, the device and the precision.

### Forced failures

Twelve, each a change built and run against the tests that must fail. Scripts `D:\Claude\metal\phase4\mutate.py` (Mac)
and `mutate_windows.py`.

| Change | Caught by |
|---|---|
| Metal says it trains | The Mac build: Metal's Linear has no `backward` |
| The refusal of `backward` names no device | `Backward_IsRefused`, both precisions |
| CUDA says it does not train | GPT's and Llama's CUDA training tests |
| FP4, or FP8, chosen by "BF16 compute" again | The Mac build: Metal BF16 has no FP4 or FP8 Linear |
| FP8 asks CUDA whatever the device | `BuildsEachFormatExactlyWhereItsLinearOperationExists`, on Windows only, since the Mac has no CUDA operations to answer yes |
| Q4_0 never built | The same test, on Windows |
| The format refusal names no device, or no precision | `TheRefusalNamesTheDeviceAndThePrecision` |
| Unquantized weights refused | `BuildsUnquantizedWeightsOnEveryDevice` |
| Each tensor's load, or the load's end, waited for on CUDA only | **Survived, both.** Metal's copy is complete on return (item 2), so on Metal these change nothing |

### Gates of the change

2026-11-05, with the three changes in the tree.

- **Mac**, clean build from the macOS preset: `MilaTests` 1468 pass, 1 skip (a CUDA buffer test, rightly), 1 fail, the BF16
  log-likelihood bar above (filed in `Issues/Untriaged.md`); Agents 10, Chat 33, Desktop 14. Then the tiny model's tests
  with `Backward_IsRefused`, and the dispatch's, 12 of 12.
- **Windows**, `x64-claude-verify`: all targets with no warnings; the full suite 2480 pass, 12 skip and the 4 failures already
  filed (the Gemma 12B drafter test, three on a partial Gemma 26B file), the CUDA 3B test among the passes; Agents 10,
  Chat 33, Desktop 14.
- **Not run:** Todd's `x64-profile`.
