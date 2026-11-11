# Metal Backend

How Mila runs on Apple M-series GPUs: which chips, how a capability is checked, which precision runs where, the kernels,
memory, the build, and the phases that get there with their tests and measurements.

**Status:** agreed 2026-11-04 (Todd, decisions in section 10). Phases 0 to 4 have passed; Phase 5's formats are built, with Llama's FP8 and FP4 tests not yet on the decided gate; Phase 6 has Gemma 4 12B with its drafter, and Qwen 3.8's operations and four layers (section 9). In v0.21.0, experimental, as `ROADMAP.md` *Mila on Apple Silicon* (Todd, 2026-11-05), limited to what the development Mac builds and tests: Llama and Gemma 4 12B. Decode replay, Qwen 3.8 whole, the Gemma 26B-A4B and its formats, and Phase 7 are in `Issues/Vnext.md` (Todd, 2026-11-09). Evidence:
`Notebooks/MetalPrecisionTiming.md`, `Notebooks/MetalDeviceAndMemory.md`, `Notebooks/MetalMatrixPath.md`, `Notebooks/MetalLlamaOperations.md`, `Notebooks/MetalFirstModel.md`, `Notebooks/MetalQuantizedWeights.md`, `Notebooks/MetalDecodeAttention.md`, `Notebooks/MetalGemma.md`, `Notebooks/MetalQwen.md`, `Notebooks/MetalRates.md`.

---

## 1. Scope

**In:** M-series chips, M1 onward, on macOS. Inference, for the families Mila runs on CUDA.

**Every compute path Apple silicon offers is open to the backend** (Todd, 2026-11-09): the GPU and the CPU over the
same unified memory, and where an operation runs is chosen by measuring it, for the best path overall. The GPU is
where the work began, not a limit.

**Every format Mila's components use on CUDA runs on Metal** (Todd, 2026-11-05). FP16 weights cannot carry the models Mila is for: Gemma 4 12B
alone would take about 22 GiB, against this Mac's 11.84 GiB working set (arithmetic). The list is `OperationTraits.Cuda.ixx`:

| Where | Formats |
|---|---|
| **Linear** | Unquantized; FP8 per channel; FP4 per group, 128 and 64; Q4_0; Q6_K; codebook 2-bit, groups 32 and 64; codebook 3-bit, groups 64 and 128 |
| **TokenEmbedding** | Unquantized; FP8 per channel; Q6_K, shared with a tied head |
| **Mixture of experts** | Unquantized; FP4 per group; Q4_0 |
| **Attention's KV cache** | Uncompressed; FP8 per token; a sliding window of each |
| **A family's designs** | Qwen 3.8's per-role plans, `Plan` and `Fp4Codebook3` |

These are storage: the precision a kernel computes in is section 3's. FP32 compute stays out (section 10), so CUDA's FP32 specializations are not on this list.

**Not on Metal in v0.21.0** (Todd, 2026-11-09, `Issues/Vnext.md`): the FP8 token embedding and the FP8 sliding window, which no model's load builds on CUDA either; quantizing to Q4_0 and six bits at load, which serves the exporter and the Gemma 26B-A4B; and the mixture of experts' grouped prefill, the 26B-A4B's alone.

**Out:**

- **Intel Macs** (Todd, 2026-11-04).
- **The Neural Engine and Core ML.** Mila's kernels are its own, and neither fits a stack you can read.
- **MPSGraph.** It is a hidden execution engine, which Mila's design rules out.
- **Training.** Its operations can follow; nothing here depends on it.

**The development machine is one M1 Pro.** Every number in this document comes from that chip or from Apple's
documents, and says which. M2 and later cannot be measured here (section 9, Phase 7).

## 2. What is known, and how

| Source | Means |
|---|---|
| **Apple** | Stated in the Metal Feature Set Tables (2026-10-23), the Metal Shading Language spec 4.1 (2026-06-04), the macOS 27.2 SDK headers, or an Apple talk |
| **Measured** | Run on the M1 Pro, macOS 27.2: `Notebooks/MetalPrecisionTiming.md`, or compile and pipeline creation only where it says so |
| **Todd** | Todd's statement, not found in Apple's documents |
| **Inferred** | Follows from the above, and has not been tested |

### 2.1 Three checks, all needed

An M-series chip has no capability report for inference. What it reports is what can be used.

**1. The OS and the language version** decide what can compile. Apple:

| Language | Needs | Brings |
|---|---|---|
| MSL 3.1 | macOS 14 | `bfloat`, `simdgroup_bfloat8x8` |
| MSL 4.0 | macOS 26 | Tensors and TensorOps (`matmul2d`). `bfloat` in `matmul2d` needs OS 26.1, `int4` OS 26.4 |
| MSL 4.1 | macOS 27 | `int2`, and the FP4 E2M1 and FP8 formats in `matmul2d` |

Measured: the M1 Pro refuses `bfloat` at MSL 3.0, and accepts it at 3.1.

**2. The GPU family, through `supportsFamily`.** Apple: a feature is available from the family its table lists, so the
check is "this family or later". Use `MTLGPUFamilyApple7` and above; the SDK deprecates `Mac2` and the `Common` families.

| Chip | Family | Metal |
|---|---|---|
| M1 series | Apple7 | 3 and 4 |
| M2 series | Apple8 | 3 and 4 |
| M3 and M4 series | Apple9 | 3 and 4 |
| M5 series | Apple10 | 3 and 4 |
| M6 series | Apple11 | 3 and 4 |

M3 and M4 cannot be told apart by family. All of them support everything this document needs, on Apple's tables:
simdgroup matrix multiply, tensors, residency sets (Apple6 or Apple7).

**3. The device's own numbers, read at run time and never tabulated.**

| Query | M1 Pro, 16 GB (measured) | Used for |
|---|---|---|
| `recommendedMaxWorkingSetSize` | 12.7 GB | The planner's memory budget |
| `maxBufferLength` | 9.5 GB | The largest single allocation |
| `maxThreadgroupMemoryLength` | 32 KB | Own kernels' tile size |
| Pipeline's `threadExecutionWidth` | 32 | Threads per threadgroup |
| Pipeline's `maxTotalThreadsPerThreadgroup` | 1024 | TensorOps' scope |
| `hasUnifiedMemory` | true | Refused if false: nothing here is written for discrete memory |

### 2.2 What no API reports

| Question | Reported? | What there is instead |
|---|---|---|
| Is BF16 native on this GPU? | No. Neither Apple document mentions native or emulated | A table Mila keeps, filled by measurement (section 3.3) |
| Does it have Neural Accelerators? | No query in the SDK headers, the spec or the tables. Apple names M5 and A19 | Apple10 and later is inferred. Metal System Trace has a utilization counter, for profiling only |
| Which matrix path is faster? | No | A table Mila keeps, filled by measurement (section 4.2) |
| How many GPU cores? | Not in Metal. IORegistry `gpu-core-count` (14, measured) | Log only |
| What generation exactly? | `architecture.name` (`applegpu_g13s`, measured). Its values are undocumented | Log only |

**Three traps:**

- **A kernel compiling proves nothing about speed.** The M1 Pro accepts every `bfloat` kernel tried, and FP4 `matmul2d`.
- **`hw.optional.arm.FEAT_BF16` is the CPU.** It reads 0 on the M1 Pro, whose GPU accepts `bfloat`.
- **The tables and the compiler can disagree.** `matmul2d` with `char`, `char`, `int` and a plain tensor output was
  refused, though MSL Table 7.3 lists it. Not pursued, since Mila has no use for it. A kernel is proven by compiling it.

## 3. Precision

### 3.1 What holds it up

- **Todd:** the M1's GPU does not support BF16 natively, and it runs through an emulation layer transparently.
  Measured, in support and not as proof: BF16 accumulates with less error than FP16 despite 3 fewer significand bits.
- **Apple:** `bfloat` exists from Apple6, so on every M chip, with add, subtract, multiply and divide correctly rounded.
- **Measured:** at prefill size, the fastest kernel measured for each format takes 25 ms in FP16, 48 ms in BF16 and 85 ms in
  FP32. FP32 costs 3.0 to 3.4 times FP16 there, on both kernels.

### 3.2 The decision

Metal's model precision is **FP16 on the M1, if it passes correctness (3.4)**. BF16 kernels are written and tested
alongside, and are not selected on the M1. **There is no FP32 model path.** On a chip with native BF16, BF16 is used, once a
measurement has shown which chips those are (3.3).

Until then **every Metal build selects FP16 on every chip**, so nothing depends on the table in 3.3 being filled. A chip
running FP16 that could run BF16 faster loses speed, and gives no wrong answers.

**An operation may widen inside itself.** `OperationDispatch.md` section 2 has the op own its internal precision, and
registers cost 1.14 to 1.44 times FP16. A norm's sum of squares or a matmul's accumulator may be wider where FP16's range
or accuracy needs it, each with the measurement that earned it. This is not a way back to FP32 math: nothing is stored in
FP32 for it, and storage is what cost 3 times.

### 3.3 How a chip gets its precision

Precision stays compile-time inside a model. The choice is made once at load, the way `Models/QuantizationDispatch.ixx`
already turns a runtime setting into a compile-time policy: a table keyed on GPU family names a precision, and the
dispatch reaches `LlamaModel<Metal, FP16>` or `LlamaModel<Metal, BF16>`. Both instantiations are built in a Mac build.

| Family | Chips | Native BF16 | Selected |
|---|---|---|---|
| Apple7 | M1 | No (Todd) | FP16 |
| Apple8 | M2 | Unknown | FP16 |
| Apple9 | M3, M4 | Unknown | FP16 |
| Apple10 | M5 | Unknown. Inferred candidate: Apple added `bfloat` to TensorOps in OS 26.1 and calls it critical; it does not say it is accelerated | FP16 |
| Apple11 | M6 | Unknown | FP16 |

A family above the table is treated as the highest row. If M3 and M4 ever differ, the family cannot express it and the
table needs the device name; that is a defect to be faced then, not designed for now.

**An outside AI answer put to Todd said M1 to M4 lack native BF16 and M5 has it.** It gave no source and none was found.
It is not in this table. If it is right, FP16 is the path most Macs in use today run, so FP16's correctness gate should be
treated as a main path, not a fallback.

### 3.4 FP16's correctness is the real gate

| | Significand bits | Largest value | Smallest normal |
|---|---|---|---|
| BF16 | 8 | about 3e38 | about 1e-38 |
| FP16 | 11 | 65504 | about 6e-5 |

**Weights.** Arithmetic, not measured: BF16 to FP16 is exact in FP16's normal range, loses bits below it and overflows
above it. A value above 65504 is refused with the tensor's name, never clipped.

**Activations are the risk.** Logits, attention scores and sums of squares can overflow, and small values can vanish. It
would show as wrong answers on some model and not as a slow run. So each model gets a correctness gate before its speed
counts (Phase 4), against its BF16 run on the same weights and against the HF reference. A model that fails has its op
widened (3.2), if the measurement shows where; otherwise it runs BF16 on that chip and the cost is stated. **FP16 is
expected, and the gate can overturn it.**

**Accumulation.** Measured: FP16 by hand loses accuracy with K (1.32e-2 at K = 8192), about 60 times TensorOps, whose output
shows the error of a wide accumulator (inferred). An own FP16 matrix kernel accumulates wider, or does not ship.

## 4. Kernels

### 4.1 Dispatch

`OperationTraits.Metal.ixx` is a partition of specializations beside `:Cuda`; a missing one is a hard compile error, as
now. This is the design claim the ROADMAP names, and this port is its first test across vendors: a component that needs
a `#if` for Metal is a finding.

Kernels are MSL source using MSL templates, one source for FP16 and BF16 where that serves. Each lives in a `Kernels/` folder beside the operation that launches it, at the same path as its CUDA counterpart.

### 4.2 Two matrix paths

| | TensorOps `matmul2d` | Own SIMD-group kernels |
|---|---|---|
| **Same code on every chip** | Yes. Apple: from M1 to M5, using accelerators where they exist and shader implementations elsewhere | Yes, without the accelerators |
| **M1 Pro, 2048 x 4096 x 14336, FP16** (measured) | 59 ms, error 2.1e-4 | 188 ms tuned, error 9.4e-3 |
| **M1 Pro, 1 x 4096 x 14336, FP16** (measured) | 0.70 ms, padded to 8 rows | 0.64 ms, error 2.0e-4 |
| **Its precedent in Mila** | cuBLASLt, on CUDA | The hand kernels, on CUDA |

**Measured on the M1 Pro (Phase 2, `Notebooks/MetalMatrixPath.md`): Linear uses its own kernel at M = 1, and TensorOps from M = 8 up, with
both paths tuned.** **M = 2 to 7 depends on the shape, so a product says when it pads.** Padding copies the rows into scratch, zeroes the rest of a tile of 8, runs the tile and copies the rows back, and that costs at least about 60 microseconds. Timed through the real path (`Notebooks/MetalLlamaOperations.md`): **a Linear pads rows left over from 50 million multiplies up**, rows x inputs x outputs, the point above which every shape timed won by more than Phase 2's 10% in FP16. **Attention does not pad, and asks for no scratch.** The own kernel beats padding at every context up to 4096, and attention x V loses at every length timed. Q x K^T wins from 16384 positions and at least 4 rows, which is about 40 ms once per prompt on Llama 3.1 8B at 64K (arithmetic). One row stays with the own kernel, as above. The own kernel reads the weights at nearly the chip's
bandwidth, widening inside itself to 32 bits, which needs no storage. TensorOps needs at least 8 rows in a tile. Own tiles do not ship: they take 1.6 to
4.8 times TensorOps, and FP16's sum in a tile misses the accuracy bar. Phase 4 compares
the two on a model. Beyond the M1, *BaseRT: Advancing Best-in-Class LLM
Inference with Apple M5 Neural Accelerators*, arXiv 2607.19438 (abstract read only), routes compute-bound matmuls through
the accelerators and keeps memory-bound decoding on existing kernels. Apple also says prefill is compute-bound and decode
is bandwidth-bound (`mlx-distributed-inference`, machinelearning.apple.com).

**Every other format (`Notebooks/MetalMatrixPath.md` section 6).** TensorOps cannot carry Mila's scales: a block scale must be `metal_fp8_ue8m0_format`, a power of two per exactly 32 (Apple's header),
which is NVFP4's shape, and it has no six-bit or codebook weights. FP8 is the exception, since its scale belongs to one output and comes out of the sum.

| Rows (M) | FP8 | FP4, Q4_0, Q6_K, codebooks |
|---|---|---|
| 1 | Own lanes | Q4_0, six bits: MLX's one-row shape. FP4, codebooks: own lanes |
| 8 | TensorOps on the bare codes, scales after | Own lanes |
| 128 | TensorOps on the bare codes, scales after | TensorOps, on FP16 expanded a block of outputs at a time |
| 2048 | TensorOps, expanded | TensorOps, expanded |

Expanding costs 1% to 5% at M = 2048, and a scratch buffer of 8 to 29 MB. Decode reads less memory than FP16 and is limited by arithmetic: 0.20 ms for FP4 and 0.24 ms for Q4_0 against 0.65 ms unquantized, at 1 x 4096 x 14336. These are weights in isolation, on normal values. In Gemma 4 12B's decode the Q4_0 lanes read their weights at about 117 GB/s, two thirds of a token's time (`Notebooks/MetalRates.md`).

**One row of Q4_0 takes MLX's one-row shape** (Todd, 2026-11-09, on the bar of section 6; `linear_quantized_row` in `LinearQuantized.metal`). It is MLX's `qmv_fast_impl` (`mlx/backend/metal/kernels/quantized.h`, ml-explore/mlx v0.32.4, MIT) on Mila's layout: a threadgroup of two SIMD-groups, four outputs a SIMD-group, each lane sixteen consecutive inputs of every 512, the activations read once into registers for all four outputs. The codes are masked in place and never shifted, against activations prescaled by 1, 1/16, 1/256 and 1/4096, and a code's offset comes off once through the activations' sum. A row whose inputs are not a multiple of 512, as Gemma's 3840 is, ends in a pass the lanes past it leave. The pipeline is resolved when the op is built. FP4 was built in the same shape and was slower than the own lanes, with its codes decoded by arithmetic as MLX's `fp4.h` does or by a table, so it keeps the lanes, as do two to eight rows, FP8 and the codebooks. Results: `Notebooks/MetalMatrixPath.md` section 7.1.

**One row of six bits takes the same kernel, both planes masked in place** (Todd, 2026-11-11). With its codes shifted it had lost to the lanes. A six-bit row is Q4_0's nibbles and then a plane of each code's high pair, eight pairs to a ushort, so the nibbles are masked as Q4_0's and the pairs against sixteen more activations, prescaled by 16 / 4^j for a pair at bit 2j; the offset of 32 comes off through the activations' sum. The kernel takes its group from the format, 32 for both, and finds each output's code and scale addresses once: its scale index had divided by the shape's group size every pass. Alone, Q4_0's product is within 2 to 7% of a kernel that only reads its bytes. Results: `Notebooks/MetalMatrixPath.md` section 7.2, `Notebooks/MetalRates.md` 7.4.

**Binding.** Measured: a tensor bound alone at a buffer index gave all-zero output with no error, and a struct of resource
ids at one index worked. So every kernel test checks values, and a test that only checks completion is not a test here.

### 4.3 Compiling kernels

Kernels are compiled when Mila is built, by Xcode's Metal compiler, as CUDA's are by nvcc (Todd, 2026-11-11). A build of Mila on a Mac
needs Xcode and its Metal Toolchain, and the configure says so when it finds no compiler. Until `+134` the source was embedded and compiled on
first use, chosen in Phase 1 because the development Mac had no full Xcode; that ended on 2026-11-05, and it left a kernel error to be
found by a test and a compiler warning to be found by nobody.

**One list, one library.** `Lib/CMakeLists.txt` lists the kernel files once. A generated source includes them in that order, and
`xcrun metal` compiles it into one Metal library, which `Implementation/MetalKernelLibrary.cpp` alone embeds (`#embed`, told where by
`--embed-dir`: an include directory does not serve it, and a miss gave an empty array with no error, which a `static_assert` now refuses).
A kernel error stops the build and names its file and line; a warning prints in every build. A kernel edit recompiles the kernels and
that one file, and nothing that imports the library: 6 build steps in 3 s on the M1 Pro. The files still compile as one source: a file
may use what an earlier one declares, `QuantizedCodes.metal` is first, and every name shares one scope, kept apart by its operation's
prefix. The options are stated, not left to the compiler's defaults: language 4.1, macOS 27.0, fast math and fast math functions, which
the kernels are written for; a kernel that needs exact arithmetic or a precise function says so itself (`WeightQuantization.metal`,
`Geglu.metal`).

**Pipelines are resolved when an operation is constructed** (Todd, 2026-11-11). An operation asks the library for every kernel it may
dispatch, by the format, precision and configuration it was built with, and holds the pipelines; a dispatch names none. A name Mila has no
kernel for is then refused when a model loads, and no pipeline is created inside a timed prefill or decode. The names are still strings
formatted from template arguments, so a missing instance is a refusal at load, not the compile error a missing `OperationTraits`
specialization is. A matrix product's kernels are its caller's to resolve (`matrixProductPipelines`). TensorOps' functions are stateless
and still look their kernel up by name at each call.

**Cost, on the M1 Pro** (`mila-metal-report`, 217 kernels, one reading each). Loading the embedded library takes 1 to 3 ms. Preparing
a pipeline is where the OS compiles a kernel for the GPU, and it keeps the result per kernel: all 217 took 12.4 s the first time this
Mac saw the library and 7 ms on every run after, and after one kernel's arithmetic was changed the next run took 72 ms. So a Mac pays
once for each kernel it has not seen, when the first model that uses it loads, and again only for kernels a Mila update changed. Not
measured: what one model's first load pays, which is its own kernels and not all 217. Compiling from source at run time, as before
`+134`, took 1,245 and 1,204 ms on a first run and 1.8 ms after (578 ms with 79 kernels, `Notebooks/MetalDecodeAttention.md` 7); whether
its pipelines were ever prepared cold in those readings is not known.

### 4.4 Decode

`IExecutionContext::createDecodeRecording()` stays null, so a decode step is called every step. The rule in `CLAUDE.md`
still binds, so that replay remains possible: read the position from the execution context, no per-token launch argument.
Indirect command buffers (Apple3 and later) are the candidate for replay, which is not in v0.21.0 (Todd, 2026-11-09, `Issues/Vnext.md`).

`synchronize()` commits and waits, and an operation only encodes. Measured: a commit and wait is about 220 us and a further
dispatch about 3 us, so a decode step commits once.

**Attention's decode is own kernels, reading the position from memory, in two passes** (`Notebooks/MetalDecodeAttention.md`, Todd, 2026-11-05). Three dispatches of a fixed grid: the cache write; a partial pass that splits the keys over 32 blocks, a threadgroup holding the query heads of one KV head so that they read its rows together, each lane a contiguous run of a head's channels, the softmax online in 32 bits; and a merge that rescales each block to the largest score of all. The shape is MLX's two-pass decode (`ml-explore/mlx`, `sdpa_vector.h`) and the split and merge Flash-Decoding's (Dao, Haziza, Massa, Sizov, 2023). The partials are FP32 in the context's scratch. The heads the models use, 64, 128, 256 and 512, have their own kernels; any other size up to 512 runs a slower one. It replaced one SIMD-group per query head walking whole key rows, which Phase 4 measured at about 22 times llama.cpp's cost a key; on Llama 3.2 3B decode after 4096 went from 7.95 to 18.65 tokens/s in FP16, 0.83 of llama.cpp, and a key now costs about 24 ns in each layer against llama.cpp's 30. The decode position is written by the host once the context's open work has finished.

**Each lane reads its run of channels in wide loads** (Todd, 2026-11-09; `Notebooks/MetalRates.md` 7.2). Gemma 4 12B's global layers, one KV head of 512 for 16 query heads, read their cache at about 11.5 GB/s, and the GPU counters showed why: the kernel's load unit busy 91% while memory was barely read, each lane loading its 16 channels one value at a time. A lane now reads its run 16 bytes a load where it allows, two loads at a head of 512 and one at 256, the sums in the order they were. MLX's block count by keys held, and four keys read ahead (`sdpa_vector_2pass` and `sdpa_vector_2pass_1`, ml-explore/mlx v0.32.4), were built first and did not move the limit: in alternated runs, 4.5% faster after 16,384 held tokens and about 1% slower below, so not kept (`MetalRates.md` 7.2). The build checks the thread limit of the kernel it dispatches, which it had not for an FP8 cache. Sharing each tile of rows across the query heads through threadgroup memory was built after it and was slower on both of Gemma's shapes, so it was not kept either (`MetalRates.md` 7.2).

**No per-token kernel gives a long row to one SIMD-group (Todd, 2026-11-07; `Notebooks/MetalRates.md`).** One SIMD-group
alone waits on memory through every step of a row, and a decode step runs hundreds of these kernels. The sampler, which
gave one SIMD-group 85 passes over Gemma's 262,144 logits and was the whole of Chat's 2 tokens a second, now runs in the
stages of CUDA's multi-block pipeline, its row across many threadgroups. Where CUDA narrows each truncation over float
histogram rounds, Metal's cut is radix selection on the scaled logits' bits, three rounds of 12, 12 and 8, so the k-th
value and the nucleus edge are exact. Greedy takes a 256-thread threadgroup a row. Every operation resolves its
pipelines once, when constructed (section 4.3).

**Nor one value a load where a thread makes several (Todd, 2026-11-11; `Notebooks/MetalRates.md` 7.4).** A thread's loads
run one after another, each waiting on memory, so a kernel's time is the loads its busiest thread makes. RMSNorm at 256
threads and a value a load took 13.9 us on a row of 3840, 15 loads a thread in each of two passes, and ran 334 times a
token. It reads four values a load, keeps a thread's first four in registers between its passes, and gives a slice wider
than 1024 a threadgroup of 1024 threads, so a row up to 4096 wide is read once: 4.1 us alone. A slice of any width starts
where its width puts it, so the loads are packed ones. A device whose limit for the kernel is under 1024 threads keeps
256 for every slice.

**A decode step is timed kernel by kernel by the library (Todd, 2026-11-07).** `setDispatchProfiling` on the model encodes
each dispatch apart with a GPU timestamp at its start and end, at the encoder's boundary, and `dispatchProfile()` returns
each kernel's GPU time; the command buffer's own GPU time converts ticks to seconds. Slower, values unchanged, for
profiling only; a no-op on CUDA, whose kernels Nsight Systems times one by one. The library being the authority on its
own statistics is Todd's leaning (`Issues/Vnext.md`); this is the first piece of it.

### 4.5 Fused prefill attention

**Agreed (Todd, 2026-11-08), on backend parity (section 10).** Prefill attention on Metal is one fused kernel a layer, as
CUDA's flash prefill is (`GqaFlashAttention.md`), over every cache CUDA's covers: the whole cache, the sliding window's
ring and the FP8 cache, at head sizes 128, 256 and 512, in FP16 and BF16. The unfused pipeline of Phase 3 -- Q x K^T
through the Linear product, a causal softmax over the scores, attention x V -- stays built as the reference the fused
kernel is tested against, and runs only where the op is told not to fuse.

**Why it is a memory change first.** The unfused pipeline keeps two score buffers in the stack's workspace, each a chunk's
query rows by the whole context by every query head. Arithmetic, not measured: on Gemma 4 12B's global layers, 1024 rows
by a 57,344 context by 16 heads in FP16 is 1.9 GB a buffer, 3.8 GB of the M1 Pro's 11.8 GB working set. Fused, the
transformer asks the op, the score width is zero, and the planner prices no score buffer (`makeGqaWorkspace`), so the
same Mac plans a longer context or a longer prefill chunk. Speed follows where the scores' traffic was the cost: after
16,384 held tokens on Llama 3.1 8B, the softmax pass alone is 0.94 s of a 4.9 s prompt (`Notebooks/MetalRates.md` 5).

**The kernel** (`Attention/GQA/Kernels/GroupedQueryAttention.Flash.metal`):

- Q is read in the projection's layout `[B, tokens, NH * HS]` and the output written in it, so neither permute runs.
- A threadgroup owns one KV head and a block of query rows, its query-head group packed into the rows position-major (row
  `f` is position `f / G`, head `f % G`), as CUDA's packed kernel (`GqaFlashAttention.md` 5.7): one K and V tile in
  threadgroup memory serves every head of the group.
- Q x K^T and P x V run on 8 x 8 SIMD-group matrices, operands in the compute type, accumulation in FP32; the softmax is
  online in FP32 (FlashAttention-2, Dao, arXiv 2307.08691), in base 2 with the scale carrying log2(e), as MLX's
  `steel_attention.h` (`ml-explore/mlx` v0.32.4) does. P is rounded to the compute type for P x V.
- A key's position is its cache row's: the row for the whole cache, the latest position mapping to it for the ring (the
  rule `gqa_prefill_softmax` uses), and a row of the FP8 cache is widened with its scale as its tile loads.
- Tile sizes start from MLX's M1 choices (32 query rows; keys 32 below head size 128 and 16 at and above it) and are
  timed against neighbours before they are fixed.

**As built (2026-11-08).** The products are TensorOps scoped to one SIMD-group, as the Metal Shading Language
recommends over SIMD-group matrices, whose layout across lanes it leaves unspecified: the scores are a cooperative
tensor, masked through `get_multidimensional_index`, reduced by `reduce_rows`, and P reaches P x V through threadgroup
memory. **No product is wider than 128 channels.** Measured on the M1 Pro: Q x K^T and P x V as one
TensorOp over a head of 256 or 512 returned wrong values with no error (relative error 0.7 to 1.3 in FP16), where 128
matched the reference; with Q x K^T summed over 128-channel slices and P x V written 128 columns at a time, every
geometry passes. Which of the two products, and which dimension, breaks was not isolated, and Apple's documentation
states no such limit (`Issues/Untriaged.md`). At head size 512 the kernel is given at most 384 threads a threadgroup, so
Gemma's group of 16 query heads is split over threadgroups, halved until it fits what the device reports. The cache is
cleared when built and holds at least one tile of rows, since a tile reads whole rows and some are not yet written; an
FP8 cache expands a KV head's K and V side by side in the context's scratch, twice what the unfused pipeline expanded. A
tile is left unmasked only when every row attends it whole. In a ring of a window and a chunk less one, those conditions
already make each of its rows hold its own position, so the kernel's "not yet wrapped" condition is redundant, and a forced
failure that removes it is caught by no test: the one equivalent mutant of 17.

**Measured, first build (`Notebooks/MetalRates.md` 6).** The memory goal is met: a 16 GB Mac plans Gemma 4 12B at its
trained maximum of 262,144 where it planned 57,344, and Llama 3.1 8B at 49,152 where it planned 24,576. The speed is not:
on Llama 3.1 8B after 4,096 held tokens the fused kernel takes 1.7 times the unfused pipeline's attention time, 127 tokens
a second against 158. Put to Todd, 2026-11-08: keep it on and tune it (Todd, "Go ahead with A").

**Tuned (`Notebooks/MetalRates.md` 6.1).** A SIMD-group takes 16 query rows (8 at head size 512) and 64 cache rows a tile,
a threadgroup at most 8 SIMD-groups. A tile every row attends whole is not masked; the ring's mapping runs only once it
has wrapped; the row maximum is taken before scaling, so the scale joins the exponent; the output is rescaled only when a
row's maximum moved. Measured and not kept: P in registers (the conversion keeps FP32, and FP32 x FP16 products were
slower), K and V shared through threadgroup memory (the tile that fits is too small), and 32 query rows a SIMD-group. From an
empty cache the fused prefill is 13 to 16% faster than the unfused pipeline; after a held prompt on Llama 3.1 8B it is 7%
slower at 4,096 and 15% at 16,384, one run each.

**What MLX measured that this may meet.** MLX v0.32.4 does not fuse every head size on a chip without its neural
accelerators (`scaled_dot_product_attention.cpp`, `use_fallback`): head size 256 goes unfused after a held prompt, and
512 runs blocked and unfused, scores in blocks of 256 rows. If the fused kernel is slower than Phase 3's pipeline at a
head size on the M1, the numbers go to Todd before anything else is decided; the unfused pipeline is not kept on quietly.

**A head of 512 prefills unfused, a block of query rows at a time, where the plan has memory for a block of 32
positions or more; under that the fused kernel runs** (decided 2026-11-10, Todd; built at `0.21.0-dev+133`, over an
uncompressed cache: an FP8 cache was not measured unfused and stays fused). The softmax runs in place, so a block
costs one score buffer. The numbers the paragraph above
asked for, one session on the M1 Pro: Gemma 4 12B's global layers take 1190 ms a prompt of 512 fused after 16,384 held
and 756 unfused in one block, 1008 in blocks of 32 and 1239 in blocks of 8; beside llama.cpp the prompt goes from 0.88 to
0.97. The fused kernel is held to 384 threads a threadgroup by the output it carries in registers, where the matrix
product, which carries nothing, is given 832, and the GPU's occupancy follows that ratio. Heads 128 and 256 stay fused.
This departs from CUDA's fused kernel at this head size on this chip, on measurement, and is MLX's choice on chips
without neural accelerators (`scaled_dot_product_attention.cpp`, `use_blocked_d512`, ml-explore/mlx v0.32.4). Design,
evidence and what `Lib/Src` changes: `MetalGlobalAttentionPrefill.md` sections 7 and 8; the planner's part:
`Deployment.md` section 5.

**Gate.** Parity with the unfused pipeline, the invariant of `GqaFlashAttention.md` section 6, at Llama's geometry (128,
four query heads to a KV head), Gemma's sliding (256, two, window 1024) and global (512, sixteen) and Qwen 3.8's full
attention (256, six), in FP16 and BF16, uncompressed and FP8, at 0, 4096 and 16,384 held tokens, each test failing when
the kernel is broken on purpose; then the HuggingFace gates on Metal of Llama 3.2 3B, Gemma 4 12B Q4_0 and Qwen 3.8's four
layers; then prefill at the depths of `MetalRates.md` 5, and the context a 16 GB Mac plans, before and after.

### 4.6 Speculative decoding

**Agreed (Todd, 2026-11-09); built the same day.** Gemma 4's drafter runs on Metal as on CUDA (`Gemma4Mtp.md`), on backend parity
(section 10): the same loop, the same drafter component, the same rule for what is kept. No measurement is asked to earn
it (Todd, 2026-11-09: "the math is not on our side yet, but we will get there"). It is designed for every M-series chip,
and only the M1 Pro can be run; a kernel choice the M1 measures is its row of a table per family, not a rule.

**What already runs on Metal, unchanged.** The loop, `GemmaModel::generateWithDraft`: it calls `draftTokens`,
`decodeTokens`, the speculative sampler and `rewindKvCache`, none of them device-specific. The speculative sampler draws up
to eight rows and walks acceptance on the device (`Notebooks/MetalGemma.md` 2.1). RoPE's decode already rotates T tokens
from the context's decode position, the token embedding and the norms are row-count blind, and the router and expert bank
take several tokens. `KvCacheView` is plain addresses, which Metal's dispatch already resolves to a buffer and offset.

**What is CUDA only today, and becomes Metal's too:**

1. **The gates in the components.** `Linear::decode`, `GatedMLP::decodeTokens`, the dense and routed feed-forwards'
   `decodeTokens`, `Router::decodeTokens`, `GemmaBlock::decodeTokens`, `GemmaTransformer::onDecodeTokens`,
   `GroupedQueryAttention::getKvCacheView`, `GemmaBlock::kvCacheView` and the transformer's two cache views are each
   `requires Cuda`. They become CUDA or Metal. `kDraftModelRuns` is CUDA at BF16 today, because CUDA's shared-cache
   attention is BF16 only; it becomes "the device has a shared-cache attention at this precision", which on Metal is FP16
   and BF16, and the load's refusal says which.
2. **`MetalLinearOp::decode`, up to 8 rows,** each at decode's arithmetic, every weight read once for all of them. Own
   lanes already serve 2 to 15 rows in `forward`, so `decode` is that path held below the expanded one, for every format
   the drafter and the 12B use: Q4_0, the six-bit head, and the drafter's unquantized weights (BF16 stored, FP16 or BF16
   compute). One row keeps its own kernel, so a verify row is not bit-identical to a one-row decode: CUDA's choice (b) of
   `Gemma4Mtp.md` 4.7, with the same gate. Unquantized weights keep `forward`'s matrix product, which pads 2 to 7 rows
   for TensorOps where that was measured to win. Which kernel serves which row count is a per-family choice: on the M1
   own lanes won at 8 rows for Q4_0 and six bits and TensorOps for unquantized weights (`Notebooks/MetalMatrixPath.md`
   6), at about six times one row's time for Q4_0; TensorOps, which Apple says runs on the M5's neural accelerators,
   stays the other arm wherever it can carry the format.
3. **`MetalGqaOp::decode` of T tokens,** T up to 8: the cache write of T rows from the decode position; the partial pass
   with each token's band ending at its own position, so token r attends through p + r; the merge per token. Over the
   whole cache, the ring and the FP8 cache, at head sizes 256 and 512 (and the others the one-token decode serves). On the
   ring, T rows written must evict no key a token still attends, the check `CudaGqaOp::decode` makes. At T = 1 the
   kernels compute today's decode exactly.
4. **`MetalSharedKvAttentionOp`,** the `OperationTraits` specialization the drafter's layers need: one query at the
   decode position over another layer's cache, through position p - 1, writing nothing. The partial and merge kernels of
   3 with the band ending one before the position, derived on the device as CUDA derives it, so a step stays a pure
   function of device memory. FP16 and BF16, uncompressed and FP8 caches, ring and whole.

**The drafter in FP16.** On the M1 the drafter computes in FP16, and it was trained in BF16. FP16's gate is per model
(section 3.4), so the drafter gets its own: its activations' largest magnitude recorded, and its parity with HuggingFace
as stage 1 of `Gemma4Mtp.md` 5.1 holds it, in FP16 and BF16.

**Not in this change.** Decode replay: Metal records nothing (4.4) and the loop runs without it, as one-token decode
does today. A drafter in a package: `BACKLOG.md`, *Gemma 4 packages do not carry Google's drafters*, unchanged; the gate
loads the drafter's own file as the CUDA tests do. The 26B-A4B compiles with it, and does not fit a 16 GB Mac.

**Gate, written before any code.**

- **Each operation against R one-row runs,** R = 1 to 8, FP16 and BF16, at Phase 3's bar (`kBar`): `Linear::decode`
  per format at Gemma's and the drafter's shapes; attention of T tokens over the whole cache, the ring before, across and
  after its wrap, and the FP8 cache, against T one-token decodes at their own positions; the shared-cache attention
  against the one-token decode at the same position with its write left out. Each with a forced failure: a row's band
  ending at the round's position and not its own, the ring's row taken as its position, the shared read's band ending at
  p and not p - 1, a weight row read per row and offset wrongly.
- **The drafter against HuggingFace,** as `Gemma4Mtp.md` 5.1 gated it on CUDA (`Tools/Drafting drafter-parity`, then
  HuggingFace on the same explicit inputs): argmax equal at every step, the mean logit difference within twice
  HuggingFace's BF16 run's, in FP16 and BF16. The tool runs on CUDA today, so this needs it on Metal first.
- **The loop.** Greedy: every token the plain loop's except at a near-tie, as on CUDA. Sampled: the fraction of rounds
  whose first draft is kept within its standard error of the mean of p(d1) (`Gemma4Mtp.md` 4.4).
- **Then measured, not judged:** decode tokens a second with and without the drafter on chat, code and prose, at K = 1
  to 4, Gemma's own sampling, one run each, on the M1 Pro. The result is recorded whichever way it falls. Every run in
  this gate is under ten minutes or is put to Todd first.

**As built.** The decode kernels of 4.4 take the token as part of the grid, and a field of the geometry says how many
positions before its own a query's keys end: 0 for the layer that owns the cache, 1 for a reader of another's. So the
verify's attention and the drafter's are the one-token decode's kernels, and a token of several computes exactly what its
own one-token decode computes. `MetalLinearOp::decode` runs own lanes for quantized weights at 1 to 8 rows and
`forward`'s matrix product for unquantized ones. The ring now refuses a rewind past what its rows still hold, as CUDA's
does; it accepted any before, safe only because nothing rewound that far.

**Results, 2026-11-09, the M1 Pro, FP16 and BF16.**

- Every operation check passes: 228 tests in the attention and Linear suites, the earlier ones among them, and 236 with
  the model tests below. A decode of
  1 to 8 tokens equals the same tokens decoded one at a time to the bit, over the whole cache, the window, the ring
  before and after its wrap and the FP8 cache, at the models' head sizes and groupings, and the decode after them reads
  the rows the one call wrote. The reader of another layer's cache holds the attention bar against the reference at the
  position before its own, over the whole cache, the ring and the FP8 cache, and writes nothing.
- The tiny Gemma (`DecodeTokens.Metal.cpp`, `Gemma.DraftModel.Metal.cpp`): a decode of 5 tokens equals five one-token
  decodes to the bit, dense and routed, where CUDA's is within its rounding; rejected tokens leave nothing after the
  rewind; the drafter is priced as built and leaves the caches as they were.
- **Gemma 4 12B Q4_0 with Google's drafter, FP16: greedy generation gives the same 128 tokens with the drafter as
  without it**, on CUDA's test prompt at K = 4.
- **Forced failures, 10 of 10 caught** (`D:\Claude\metal\drafter\mutate.sh`): a token's band ending at the round's
  position; every token on token 0's query head; the reader's band ending at its own position; the KV head taken from
  the whole grid row; the ring's eviction check removed; the cache write or the merge over one token's rows only; the
  ring's rewind check removed; the reader's band offset left at 0; a Linear decode of its first row only. Two needed a
  second run: the first text matched the prefill's query permute earlier in the file, and the second was not applied.
  The multi-token tests now hold their setup prefills to the reference as well.
- Not yet run: the drafter against HuggingFace, the sampled check, and the rates.

## 5. Memory

**Storage mode:** shared. The allocator returns the buffer's `contents` as the pointer a tensor holds, and the scaffold's
bridged handle goes. **`is_host_accessible` is false, though the memory is shared.** A host read has to wait for the GPU, and
`Tensor`'s host accessors do neither that nor the right element type: they read FP16 and BF16 as `float`. Being false takes
those accessors away at compile time and sends every copy through Metal's own path, which waits. The host still reads and
writes this memory, inside Metal's tensor operations.

**Binding by pointer.** A Mila tensor holds a pointer; Metal binds a buffer and an offset, or a GPU address. The
device keeps the map from address range to buffer, since a `TensorBuffer` makes a memory resource of its own and a kernel
binds buffers from many, and every operation binds through it. Measured at 300 allocations: the map costs nothing visible,
and one residency set with `gpuAddress` saves 4% of the overhead of 200 dispatches. So each buffer is bound on the encoder,
which also lets Metal order dependent dispatches. Phase 4 can reopen this on a real decode step.

**Three modules include metal-cpp:** `MetalDevice`, `MetalKernelLibrary` and `MetalExecutionContext`, in the global module
fragment, with their code in the `.ixx` like the rest of the library. Every other module names its types by forward
declaration, or goes through `MetalExecutionContext::dispatch` and needs none. The count is what matters: each adds about
14 MB of Clang's 2048 MB of source locations to every importer of Mila, and these three take a file that imports Mila from
3.1% to 5.1% (`Notebooks/MetalDeviceAndMemory.md` section 4). Mila's own count on CUDA, 68 files with its runtime header,
would be about 950 MB here (arithmetic).

**Limits.** `maxBufferLength` is below the working set on this machine (9.5 GB against 12.7 GB, both measured), so a tensor
above it is refused by the planner with the number, not split. `dim_t` and `narrowToKernelIndex()` carry over.

**The planner** (`Deployment.md`) budgets from `recommendedMaxWorkingSetSize`. Apple calls this an approximation, and unified
memory is shared with everything else the Mac is doing, so Phase 4 measures how near it a model can really run. **Measured:** at 99% of it Llama 3.2 3B runs within 3% of its rate, and macOS swaps the rest of the Mac out to make room (`Notebooks/MetalFirstModel.md` 5).

**Weights.** A file whose stored form is the run form (FP4, FP8) is mapped and wrapped without a copy, which is the question
`Issues/Future.md` *Compute backends beyond CUDA* raises, if `newBufferWithBytesNoCopy` accepts a file mapping of this size;
Phase 4 tries it. **It does:** the whole 7.2 GB file, mapped read-only and wrapped in 0.22 ms, read right by a kernel, and counted by the device as allocated though not by the process (`Notebooks/MetalFirstModel.md` 5). A cold first read from the SSD is not measured. **No published Mila model stores FP16 weights** (Todd, 2026-11-05), so an FP16 parameter always reads BF16 weights, converted once at load, and that conversion is part of Phase 3 rather than of the first model. Published packages do not change:
`ModelDistribution.md` *Compatibility* holds.

## 6. Tests and measurement

- **Green on Metal** is the macOS preset, on the M1 Pro, all targets and the full suite, on top of Todd's Windows
  environment, which stays green. Neither stands in for the other.
- **A Metal test compares values with a reference.** Per operation, a double-precision reference in the test. Per model, the
  HF reference and the token ids.
- **Every phase's gate is written in its notebook before its run**, with a forced failure to show the check can fail.
- **Arms come from one program**, on the same inputs, order rotated, with GPU timestamps.
- **On the Mac:** on AC, under `caffeinate -dimsu`, power source recorded at both ends and the sleep log checked. Its
  battery profile sleeps after 1 minute, and a pulled cable has already cost a run. One large job at a time.
- **llama.cpp is the rate reference, as on CUDA,** run on the same Mac, tuned, like for like. It has no other bearing.
- **MLX is the bar a Metal path is built to, and a design source** (Todd, 2026-11-09): its rate on the same Mac, model
  and shape is the target, and its published source may be taken and cited. It is not a dependency; its scheduler is
  an execution engine Mila does not hide behind. **The bar is 0.85 of MLX's rate**, at prefill and at decode. Work goes
  to the largest gap first and stops at the bar; a change that arithmetic does not show moving a model by about 5% is
  not started without Todd. A measurement that points past MLX is followed.

## 7. Build and the scaffold

**The scaffold is replaced, not adapted.** `Compute/Devices/Metal/` predates the current direction: `#ifdef` and Objective-C
in module units, Intel and discrete paths, one type defined twice (`Issues/Vnext.md`), `MTLCopyAllDevices`, and capability
checks that ask the wrong questions (`Issues/Untriaged.md`). `Lib/CMakeLists.txt`'s `MILA_HAS_METAL` block goes with it.

**A Mac build needs Xcode's Metal Toolchain** (`xcodebuild -downloadComponent metaltoolchain`), for the kernels (section 4.3).

**Host code is C++ through metal-cpp** (`github.com/apple/metal-cpp`, header-only, tagged to macOS 27.0.1, with
`MTLTensor.hpp` and `MTLResidencySet.hpp`), pinned as other dependencies are. No Objective-C in module units. Measured in
Phase 0: it works in a module's global fragment beside `import std;` under Clang, and `newTensor` returns null until the
descriptor's strides are set.

**CUDA in the wrong places.** 75 files outside `Devices/Cuda` name it (a text search, so some are comments and some are
legitimate), 11 under `Components/Transformers`, and `Llama.ixx` imports `Compute.CudaPinnedMemoryResource`. Each is either a
trait or a finding. A build with CUDA off compiles and passes; running on Metal is what tests the rest.

## 8. What this changes elsewhere

- `Issues/Future.md` *Remove FP16*: FP16 stays for Metal. Whether CUDA's FP16 rows still go is that entry's own question,
  now separate from this.
- `Quantization.md`'s out-of-scope line and `CLAUDE.md`'s "FP16 is not used" say so, in this change.
- `MilaDesktop.md` and `ChatConfiguration.md` both wait on this backend for macOS; Phase 8 reaches them.

## 9. Phases

No durations: a time estimate follows a timing run. Each phase's question must be answered before the next starts.

### Phase 0 -- The Mac builds the tree

Setup only. **Question:** does the tree build and pass on macOS with CUDA off, from a clean sync?

- A macOS preset in `CMakePresets.json`, and a sync that removes deleted files.
- The three filed defects: `ModelHandle.ixx:286`, `cmake/StdModule.cmake:21`, `mila-store` with CUDA off.
- metal-cpp fetched, and one module unit including it beside `import std;`.

**Gate:** all targets and the full suite on the Mac with no local patches; Windows unaffected.

**Passed, 2026-11-04, at `0.21.0-dev+117`.** On the Mac, from a clean build directory and no local patches: configure 93 s,
build 69 s, `MilaTests` 1237 pass and 1 skip (a CUDA buffer test, rightly), Agents 10, Chat 33, Desktop 14. On Windows,
`x64-claude-verify`: all targets, and the suite at 2475 pass, 12 skip and the 4 failures already filed (one Gemma 12B drafter
test, three on a partial Gemma 26B file). With CUDA off, `mila-store` builds and lists the store. metal-cpp in a module unit
beside `import std;`: a kernel dispatched correctly, 0 of 1024 values wrong, and a tensor was created. **Not run:** Todd's
`x64-profile`.

### Phase 1 -- Device, memory, one kernel

**Question:** can Mila allocate, dispatch and read back through its own device and memory types?

- `MetalDevice`, `ExecutionContext<Metal>` and the memory resource, replacing the scaffold.
- The three checks of 2.1. A Mac below the floor is refused at startup in plain words.
- A capability report tool in `Lib/Tools/`: everything in 2.1 and 2.2, plus compile-and-create of every kernel Mila ships.
- `TensorOps` fill and copy.

**Tests:** every refusal with a forced failure. Values read back. **Measures:** first-load compile cost; command buffer commit
cost per dispatch; binding per buffer against a residency set.

**Gate:** tensor tests pass on `DeviceType::Metal`, and the report runs.

**Passed, 2026-11-04.** `Notebooks/MetalDeviceAndMemory.md` has the gate, its 12 forced failures and the measurements. The
report is `Lib/Tools/MetalReport`.

### Phase 2 -- The matrix path

**Question:** which path does Linear use, per shape, and what is the anomaly?

- The timing program brought under `Lib/Tools/`, on real Linear shapes, decode's M = 1 included.
- The own kernel tuned, with accumulation wider than FP16, before any comparison counts.
- TensorOps with FP16 inputs and FP32 output: does it ease FP16's range at no cost, as it did here?
- **The unexplained result:** TensorOps BF16 at the decode-like shape took half of FP16. Tile size, shape or data? Until it is
  explained, no decision rests on it.

**Gate:** a written choice of path per shape, with the numbers, in the notebook.

**Passed, 2026-11-04.** `Notebooks/MetalMatrixPath.md`, with `Lib/Tools/MetalMatrixTiming`, on 34 of Llama's shapes, three process runs each. Own at M = 1, by 6% to 54%; TensorOps at
M = 8 to 2048, where no tuned own kernel came within 1.6 times, or met the accuracy bar short of 3.2 times. FP16 inputs with FP32 output cost nothing and took the error from 2.1e-4 to about 1e-6.
The anomaly is one slow FP16 tile on the earlier program's weight layout, which Linear's layout does not show; why that tile is slow is not known. Not a model run.

**Extended 2026-11-05 to every weight format that a Linear reads** (Todd, section 1), 308 arms, three process runs each, none differing by more than 11%. The choice per format is in section 4.2. Not run: FP8's KV cache, TokenEmbedding, mixture of experts, M = 2 to 7, and M between 8 and 128.

### Phase 3 -- Llama's operations

**Question:** does each operation give correct values in FP16 and in BF16?

In order: Linear, TokenEmbedding, RmsNorm, Rope, Swiglu, Residual, grouped-query attention (prefill and decode), Sampling,
NextTokenLogProbability. BF16 weights first, quantized in Phase 5. BF16 kernels are tested here through the emulation layer:
that shows correctness, and says nothing about the speed of native BF16.

**Agreed for this phase (Todd, 2026-11-05).** Weights reach a test as a BF16 blob through `loadParameter`, in both precisions, since that is what a model hands them. Attention's decode is one own kernel, which reads the position from memory; its prefill is TensorOps, a softmax kernel, then TensorOps. Phase 4 measured prefill only at 512 tokens, so the measurement a fused prefill waited on never came; it is now decided on parity with CUDA (Todd, 2026-11-08, section 10). TensorOps writes FP16, not FP32. Rows and outputs off a multiple of the tile go through the own kernel, and M = 2 to 7 is timed against padding. Metal operations declare no `backward`. Left to later phases: `Llama.ixx`'s load synchronization (Phase 4) and `Linear::decode` over several rows (Phase 6). The scratch buffer, planned for Phase 5, came with padding after this phase (Todd, 2026-11-05).

**Tests:** each against a double-precision reference in both precisions, with the decode rule (`CLAUDE.md`) held by a test that
sets the position first. **Notebook:** `Notebooks/MetalLlamaOperations.md`.

**Gate:** all pass, and a missing specialization is a compile error, not a skip.

**Passed, 2026-11-05.** `Notebooks/MetalLlamaOperations.md` has the gate, its 111 forced failures and the results. Every operation holds its bar in FP16 and BF16 with weights read as BF16 blobs, a Llama block's prefill equals its decode, and no component needed a `#if` for Metal. **Not a model run.**

### Phase 4 -- First model, both ways

**Question:** does FP16 hold up on a real model, and what does each way cost? (Todd: test both ways on the M1.)

Llama 3.2 3B on BF16 weights, at `LlamaModel<Metal, FP16>` and `LlamaModel<Metal, BF16>`, at its defaults. The gate is written
first, and may overturn FP16.

**Tests:** token ids and logits against HF; generation by reading it; activation ranges per layer, overflow and flush counted.

**Measures:** prefill and decode rate both ways; memory against the budget; how near `recommendedMaxWorkingSetSize` it runs;
no-copy weights; llama.cpp on the same Mac.

**Gate:** the 3.2 decision confirmed or changed for this model, in the notebook and here.

**Correctness passed, 2026-11-05: FP16 is confirmed for Llama 3.2 3B.** `Notebooks/MetalFirstModel.md` has the gate, whose bars are two accepted BF16 runs of the model (Mila's on CUDA and HuggingFace's), and the results. FP16 is about ten times closer to HuggingFace FP32 than either: worst logit step 1.9e-3 against a bar of 2.2e-2, log-likelihood over 1024 tokens within 6.8e-5, greedy 32 of 32, activations at most 420 and all finite. Metal BF16 passes its logits and greedy, and misses the log-likelihood bar at +1.25e-3 against 8.97e-4; it is not what the M1 selects, and the cause is not found. The library needed three changes, none of them a `#if` for Metal: the network's `backward` is kept from compiling where `DeviceTypeTraits` says a device does not train, every network's load synchronizes on every device, and `dispatchWeightQuantization` builds a format where `OperationSupported` says its Linear exists.

**Measures run, 2026-11-05** (`Notebooks/MetalFirstModel.md` sections 4 and 5). Against llama.cpp b12138 on the same weights, tuned: FP16 prefill of 512 at 0.85 of its rate (555 tokens/s), decode at 0.80 (19.4), decode after 4096 at 0.35 (7.95); BF16 0.59, 0.80 and 0.40. Attention's decode costs about 22 times llama.cpp's for each key in the cache, linear in depth. At a context of 8192 the plan prices what the device allocates, to 0.03%. The planner's own context, 24576, is 99% of the working set: it runs within 3%, while macOS swaps the rest of the Mac out. A file mapping of the whole weights is wrapped without a copy and read right, and counts as allocated. GPU timestamps were not taken: the execution context does not report them. Attention's decode was rebuilt the same day (section 4.4): FP16 decode after 4096 is now 18.65 tokens/s, 0.83 of llama.cpp.

### Phase 5 -- Quantized weights and the KV cache

**Question:** does every format in section 1 run correctly, and can TensorOps' formats carry Mila's?

Llama 3.2 3B FP8 and FP4, Llama 3.1 8B FP4, then the rest of the list on the model that uses each, with the paths of section 4.2. Per-token KV compression, and its sliding window.

`dispatchWeightQuantization` builds a format wherever its Linear operation exists on the device at the precision, so each Metal specialization this phase adds is reached as it lands (Phase 4 made it so). `dispatchKvCacheCompression` now does the same for the FP8 cache, through `OperationSupported` on attention (Todd, 2026-11-05).

**Agreed for this phase (Todd, 2026-11-05).** Model runs are Llama's: 3.2 3B in FP8 and FP4, 3.1 8B in FP4. A format only Gemma or Qwen uses gets its operation and its own test here, and its model run in Phase 6 with that model. Quantize-on-load is a Metal kernel per format, held to CUDA's bytes. Notebook: `Notebooks/MetalQuantizedWeights.md`, gate written first.

**Gate:** as Phase 4, per format, and a Metal specialization for every entry in section 1, with its own test against a reference.

**FP8 and FP4 Linear and the FP8 cache, as operations, 2026-11-05** (`Notebooks/MetalQuantizedWeights.md` 2.1). Linear on FP8 and FP4 at 64 and 128 holds its bar at every row count on the paths of section 4.2. TensorOps refuses BF16 activations times FP8 codes, so BF16 FP8 takes FP4's paths. Quantize at load writes the bytes CUDA's `ExportArtifact` writes, on Llama 3.2 3B's own weights, once each kernel is compiled in safe math: Metal's default fast math rounded 0.04% of codes differently. Attention's FP8 cache holds its bar in prefill and decode; decode reads the codes, and prefill expands one KV head's rows at a time, since folding V's scale into the probabilities would put them below FP16's normal range.

### Phase 6 -- The other families

**Question:** do Gemma and Qwen hold FP16 too? FP16's gate is per model, and they are the agentic ones.

Gemma 4 12B and Qwen 3.8 27B, on what 16 GB of unified memory allows, with the agentic quality bar unchanged, in their own formats: Gemma's Q4_0 with its Q6_K
table, Qwen's FP4 and its plans, and mixture of experts. Family parity holds: build order justifies no gap. Apple states Qwen 3.8 is native BF16.

**Gate:** as Phase 4, per model, against HuggingFace FP32 on the model's weights decoded from its format.

**Gemma 4 12B first, in Q4_0 (Todd, 2026-11-06),** with the operations only it needs: Q4_0 and six-bit Linear, the six-bit
table, GeGLU, the sliding-window ring cache. Notebook: `Notebooks/MetalGemma.md`, gate written first.

**Qwen 3.8's operations next, ahead of Phase 8 (Todd, 2026-11-06):** opening a model by name builds every family's type,
so an agent on Metal needs Qwen's operations to compile. The elementwise activation, the causal convolution, the gated
delta rule in its recurrent form with the state FP32, two- and three-bit codebook Linear, and the tensor math; Qwen's
weight dispatch builds a plan wherever `OperationSupported` says its Linears exist. Notebook: `Notebooks/MetalQwen.md`.

**Passed, 2026-11-07, on the four-layer build:** every operation holds its bar in FP16 and BF16, 30 of 31 forced failures
are caught, and the build passes the gate against HuggingFace FP32 on its weights as Mila runs them, FP16 about ten times
closer than either BF16 run. No Qwen 3.8 agentic build fits this Mac's working set, and the whole model has not run here.
Qwen 3.8 whole on a Mac is v0.22.0's (Todd, 2026-11-09, `Issues/Vnext.md`).

### Phase 7 -- Other chips

**Question:** which chips have native BF16, and which matrix path wins on each?

Needs hardware Mila does not have: the report and the timing program, on each generation that can be borrowed or rented. A
row of 3.3 changes on a measurement from that family and nothing else. Accelerators are measured on M5 here, and the path
choice of Phase 2 may become per family.

**Gate:** each row filled from that chip's own run, and a row with no run stays "Unknown". Not in v0.21.0, which is
what the development Mac builds and tests (Todd, 2026-11-09).

### Phase 8 -- Apps and distribution

Chat, Mila Desktop's session process and the server on macOS; how a Mac user installs Mila; what the model cards may claim.
Planned when Phase 6 is done. An app adds no capability to the library. Today's binaries link Homebrew's libc++, libunwind
and OpenSSL, and the linker warns that those were built for macOS 27.2 against the preset's 27.0. Both are fine on the
development Mac and neither can reach a user's.

## 10. Decisions

| Decision | By |
|---|---|
| M-series only, no Intel | Todd, 2026-11-04 |
| Kernels are compiled when Mila is built, by Xcode's Metal compiler, and embedded as one Metal library; a build on a Mac needs the Metal Toolchain, as a CUDA build needs nvcc. Every operation resolves its pipelines when it is constructed (section 4.3) | Todd, 2026-11-11 |
| Decode attention reads each lane's channels in wide loads; MLX's block count and read-ahead, built and measured, are not kept (section 4.4) | Todd, 2026-11-09 |
| MLX is the target and a design source, not a dependency: 0.85 of its rate on the same Mac, model and shape, largest gap first, nothing under about 5% started without Todd (section 6). Decode is first: at `+127` it was 0.58 to 0.79 of MLX, prefill 0.87 and above (`Notebooks/MetalRates.md` 2) | Todd, 2026-11-09 |
| Not restricted to the GPU: every compute path Apple silicon offers, each operation placed by measurement for the best path overall. Replaces "GPU only, no CPU compute" (2026-11-04) and "Metal stays GPU only for now" | Todd, 2026-11-09 |
| BF16 kernels where the user's hardware has BF16 natively, a fallback where it does not | Todd, 2026-11-04 |
| The M1 has no native BF16, and runs it through an emulation layer | Todd, 2026-11-04 |
| On the M1, test both ways and ultimately fall back to FP16 | Todd, 2026-11-04 |
| No FP32 math: too slow, measured at 3.0 to 3.4 times FP16 at prefill size | Todd, 2026-11-04; `Notebooks/MetalPrecisionTiming.md` |
| FP16's correctness gate can overturn it, per model | Todd, 2026-11-04 |
| FP16 selected on every chip until a chip's own run says otherwise | Todd, 2026-11-04 |
| An operation may widen inside itself where FP16's range needs it, with the measurement that earned it; nothing is stored in FP32 for it | Todd, 2026-11-04 |
| Work from a spec with phases, tests and measurements; a new backend starts fresh, with no earlier findings carried | Todd, 2026-11-04 |
| The minimum macOS is 27. It brings the FP4 formats and the current SDK, and the floor can drop later if Phase 5 does not use the 4.1 formats. Whether every M1 model runs macOS 27 has not been checked | Todd, 2026-11-04 |
| "A model runs in the format it was trained for" is about weights. FP16 compute on BF16-trained weights does not break it, and FP16's gate stands for the rest | Todd, 2026-11-04 |
| The first model is Llama 3.2 3B, smallest with the smallest op set. Gemma 4 12B is the first agentic model, in Phase 6 | Todd, 2026-11-04 |
| Host code uses metal-cpp. Phase 0 showed it works beside `import std;` | Todd, 2026-11-04 |
| Whether a kernel calls TensorOps is what Phase 2 measures, with own kernels kept where the result is close | Todd, 2026-11-04 |
| Every format Mila's components use on CUDA runs on Metal (section 1) | Todd, 2026-11-05 |
| A quantized model is held to HuggingFace FP32 on its format's weights decoded to full precision, with Phase 4's bars, not to CUDA's run of the same weights | Todd, 2026-11-06; `Notebooks/MetalGemma.md` 1.1 |
| No published model stores FP16, so an FP16 parameter reads BF16 weights, from Phase 3 on (section 5) | Todd, 2026-11-05 |
| Qwen 3.8's operations before Phase 8, and Qwen's weight dispatch builds a plan where `OperationSupported` says its Linears exist (section 9, Phase 6) | Todd, 2026-11-06; `Notebooks/MetalQwen.md` |
| Pad 2 to 7 rows before Phase 4, with the scratch buffer here and not in Phase 5, and the rows that pad a tile zero. Attention pads only if its timing earns it (section 4.2) | Todd, 2026-11-05 |
| Phase 4's library changes: `DeviceTypeTraits` says whether a device trains, every network's load synchronizes on every device, and the weight dispatch builds a format where `OperationSupported` says its Linear exists (`Notebooks/MetalFirstModel.md` section 3) | Todd, 2026-11-05 |
| Attention's decode in two passes on MLX's path, 32 blocks fixed, the partials in the context's scratch (section 4.4, `Notebooks/MetalDecodeAttention.md`) | Todd, 2026-11-05 |
| Gemma 4's drafter runs on Metal: the multi-token decode, the reader of another layer's cache, every M-series chip in view, no measurement asked to earn it (section 4.6) | Todd, 2026-11-09 |
| Backend parity: what CUDA runs fused or replayed, Metal does too, with no measurement asked to earn it. First, prefill attention fused, over the whole cache, the sliding-window ring and the FP8 cache, as CUDA's flash prefill is (section 4.5); its rates before the change at 0, 4096 and 16384 held tokens are in `Notebooks/MetalRates.md` | Todd, 2026-11-08 |
| Metal in v0.21.0 is what the development Mac, an M1 Pro with 16 GB, builds and tests: Llama and Gemma 4 12B. Decode replay, Qwen 3.8 whole, the Gemma 26B-A4B and the formats only it or no model uses (section 1), and Phase 7 move to `Issues/Vnext.md`. Signing the Mac app stays in v0.21.0 | Todd, 2026-11-09 |
| Phase 5's model runs are Llama's (3.2 3B FP8 and FP4, 3.1 8B FP4); Gemma's and Qwen's formats are operations here and model runs in Phase 6. The FP8 cache dispatch asks `OperationSupported` (section 9, Phase 5) | Todd, 2026-11-05 |
