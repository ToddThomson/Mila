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
`recommendedMaxWorkingSetSize` the model runs; weights wrapped without a copy; llama.cpp on the same Mac. Their method is
section 4.

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

## 4. The measures, written before their run

Same machine and rules as section 1, the same weights, and nothing else running on the Mac.

**Rates.** `mila-metal-model-timing` (`Lib/Tools/MetalModelTiming`) loads the model through `LlamaModel::load` and times
`generate`, as an application runs it, greedy, since this compares engines on the same weights. Its shapes are
llama-bench's, so the two read alike:

- **Prefill:** a 512-token prompt of fixed random ids, answered with one token. Rate = 512 / time to the first token.
- **Decode:** a one-token prompt answered with 128 tokens, end-of-sequence ignored, at depth 0 and after a 4096-token
  prompt. Rate = 127 / time from the first token to the last.

A context of 8192, fixed, so both depths fit. Three rounds, each loading FP16 and then BF16 or the reverse, one model
resident at a time, since two pass the working set. Per load one unmeasured run of each shape, then five; the median of
the fifteen is reported with the lowest and highest. Wall clock around the callbacks, which is what an application gets
and what llama-bench reports. Design section 6 asks for GPU timestamps; the execution context does not report a command
buffer's GPU times, and adding that is `Lib/Src`, so they are not taken here.

**Load.** Wall clock of each load. The first after a change of kernel source also compiles them (design section 4.3);
the first load of the run is reported apart.

**Memory.** For each load: the plan's footprint, the device's allocation after the load (`currentAllocatedSize`, through
the device's own reading) and the process's physical footprint after it and at its peak, against
`recommendedMaxWorkingSetSize`. Then the planner's own context, automatic: the length it chooses, what limited it, and
its footprint against the working set. That plan is loaded and runs one prefill and decode; the same readings, and the
Mac's swap and page-outs before and after, say whether it really runs there.

**No-copy weights.** A probe outside the library: the weights file mapped read-only, the whole mapping wrapped by
`newBufferWithBytesNoCopy`, and a kernel reading words at the start, the middle and the end, compared with the host's.
Recorded: accepted or not, the time to wrap, and the physical footprint before and after the kernel runs. BF16 is the
run form of these weights at BF16; at FP16 it is not, so the answer bears on BF16 here and on FP8 and FP4 in Phase 5.

**llama.cpp.** Its macOS arm64 release, with the source of the same tag for the converter. GGUF files at F16 and at BF16,
converted by its own `convert_hf_to_gguf.py` from the HuggingFace checkpoint the reference was captured from. Tuned:
`llama-bench -p 512 -n 128 -d 0,4096 -ngl 99 -r 5`, flash attention off and on, the faster taken per cell, the KV
cache at the run's own precision. Mila's arms and llama.cpp's are separate programs, run back to back on the same
inputs' shapes. Should a format not run on the M1's GPU, what llama.cpp does instead is recorded and not timed as its
rate.

**What this cannot say.** One model, one chip, two depths, and no GPU time apart from the host's.

## 5. The measures' results

2026-11-05, on AC at both ends, no sleep in the log. Logs `D:\Claude\metal\phase4\measures`.

### Rates, against llama.cpp

Tokens per second. Mila: median of 15 (three loads of five runs), `mila-metal-model-timing`. llama.cpp b12138: mean of 5,
`llama-bench`, flash attention on, which was the faster in every cell; GGUF F16 and BF16 converted from the same
HuggingFace checkpoint, KV cache at the run's precision.

| | Prefill 512 | Decode 128 | Decode 128 after 4096 |
|---|---|---|---|
| Mila FP16 | 555 (488 to 598) | 19.4 (18.3 to 19.6) | 7.95 (7.91 to 7.98) |
| llama.cpp F16 | 649 | 24.3 | 22.4 |
| Mila / llama.cpp | 0.85 | 0.80 | **0.35** |
| Mila BF16 | 299 (297 to 301) | 19.2 (19.1 to 19.3) | 7.85 (7.83 to 7.86) |
| llama.cpp BF16 | 510 | 24.1 | 19.6 |
| Mila / llama.cpp | 0.59 | 0.80 | **0.40** |

llama.cpp keeps TensorOps off before the M5 ("tensor API disabled for pre-M5 and pre-A19 devices"), and reports the
M1 Pro as having BF16; both formats ran on the GPU. With flash attention off it was 4% slower at prefill and 18% slower
decoding after 4096 in F16.

**Decode at depth 0** is 51.5 ms a token for Mila, about 124 GB/s over the 6.4 GB of weights, against 41 ms and about
156 GB/s for llama.cpp (arithmetic). Phase 2's own kernel read weights at about 180 GB/s alone.

**Decode after 4096 is where Mila loses most.** Attention adds about 74 ms a token, 2.6 ms a layer, where reading that
layer's K and V once takes about 0.1 ms at Phase 2's rate (arithmetic); llama.cpp adds 4 ms a token in all. Read, not
timed apart: Mila's decode scores run one SIMD-group per query head, 24 on the whole GPU, each lane walking whole key rows
alone, and each of the three query heads sharing a KV head reads its keys again (design section 4.4).

**BF16 prefill** is 0.59 of llama.cpp's, against 0.85 in FP16.

### Load

6.7 to 11.1 s for the 7.2 GB file, median about 8.5 s; the first load of the run was the slowest. Kernels were already
in the OS's cache from a smoke run, so this does not time their first compile.

### Memory

At a context of 8192 the plan prices 8.470 GiB and the device allocates 8.472 GiB, 71.6% of a working set of 11.840 GiB.
The process's footprint is 6.7 GiB after the load and 8.49 GiB after the runs: the KV cache and scratch become resident
when first written.

**The planner's own context runs too near the working set.** Automatic, it chooses 24576, limited by the prefill chunk
rule, at 11.721 GiB, 99.0% of the working set. The model loads and decodes at its usual rate (FP16 19.6), but FP16
prefill ran at 119 tokens/s and BF16 at 173, and macOS's swap grew from 0.72 to 1.07 GiB, about 79,000 pages swapped
out during the FP16 arm. That arm ran no warm-up, so part of the slower prefill may be a first run's; it is to be run
again with one.

**Run again with a warm-up and three runs**, the same day: FP16 prefill 489 to 504 and decode 18.7 to 18.8, BF16 278
to 288 and 18.5 to 18.6. In the same session at a context of 8192, FP16 prefill was 508 to 529 and decode 18.8 to 18.9.
So the 119 was the first run's, and at 99% of the working set the model runs about 3% slower at prefill and 1% at
decode. macOS swapped out about 214,000 pages in the FP16 arm and 114,000 in the BF16 one, and swap in use grew from 0.78
to 1.12 GiB: the model runs there by pushing the rest of the Mac out.

### Decode by depth

FP16, three runs at each depth, one load each, the same session:

| Depth | 0 | 1024 | 2048 | 4096 | 8000 |
|---|---|---|---|---|---|
| Tokens/s | 18.8 | 14.0 | 11.1 | 7.95 | 4.93 |
| ms a token | 53.2 | 71.2 | 90.3 | 125.8 | 202.8 |

**Attention costs about 18.7 us a token for each key in the cache, linear in depth**: 0.67 us a key in each of the 28
layers, for the key's 4 KB of K and V, about 6 GB/s. llama.cpp's F16 adds 3.5 ms a token at 4096, about 30 ns a key in
each layer, about 136 GB/s (arithmetic, from section 5's table). Mila's is about 22 times that. Linear in the keys says
the cost is in reading them, not in the three dispatches a layer, which do not grow with depth.

**Rebuilt the same day:** attention's decode in two passes (`MetalDecodeAttention.md`) took FP16 decode after 4096 from 7.95
to 18.65 tokens/s, 0.83 of llama.cpp, and after 8000 from 4.93 to 17.75.

### Weights without a copy

`D:\Claude\metal\phase4\measures\nocopy.cpp`, metal-cpp outside the library, on the 7.2 GB weights file:

- **Accepted.** `newBufferWithBytesNoCopy` wrapped the whole read-only mapping, rounded up to the 16 KB page, in 0.22 ms,
  under `maxBufferLength` (9.53 GB). The buffer's contents are the mapping.
- **Read right.** A kernel read 64 words, the start, the middle, the end and 61 at random, and all equalled the host's;
  compared with the word before each, all 64 differed. A kernel reading one word from every one of 440,279 pages gave
  the host's sum, in 1 to 3 ms.
- **Counted as allocated.** The process's footprint stayed at 0.009 GiB, since the pages are the file's, but the device's
  `currentAllocatedSize` rose by 6.718 GiB at the wrap. The planner reads free memory from that figure, so a wrapped
  file is charged to the working set as an allocation would be.
- **Not measured:** a cold read. The file was in the OS's cache from the runs before, so the page pass reads no disk;
  how fast a first decode pages the weights in from the SSD is open.

At BF16 the stored form is the run form, so these weights could load without the copy that takes about 8 s now; at FP16
they cannot, since FP16 is converted at load. FP8 and FP4 are Phase 5's.
