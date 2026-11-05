# Metal Backend

How Mila runs on Apple M-series GPUs: which chips, how a capability is checked, which precision runs where, the kernels,
memory, the build, and the phases that get there with their tests and measurements.

**Status:** agreed 2026-11-04 (Todd, decisions in section 10). No Metal code exists beyond the
scaffold in `Lib/Src/Dnn/Compute/Devices/Metal/`, which section 7 replaces. No release is attached. Evidence:
`Notebooks/MetalPrecisionTiming.md`.

---

## 1. Scope

**In:** the GPU on M-series chips, M1 onward, on macOS. Inference, for the families Mila runs on CUDA.

**Out:**

- **Intel Macs, and CPU compute** (Todd, 2026-11-04).
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
table needs the device name, as llama.cpp's guard does; that is a defect to be faced then, not designed for now.

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

Kernels are MSL source using MSL templates, one source for FP16 and BF16 where that serves.

### 4.2 Two matrix paths

| | TensorOps `matmul2d` | Own SIMD-group kernels |
|---|---|---|
| **Same code on every chip** | Yes. Apple: from M1 to M5, using accelerators where they exist and shader implementations elsewhere | Yes, without the accelerators |
| **M1 Pro, prefill size, FP16** (measured) | 25 ms | 101 ms, untuned |
| **M1 Pro, decode-like, FP16** (measured) | 2.79 ms | 2.58 ms, untuned |
| **Its precedent in Mila** | cuBLASLt, on CUDA | The hand kernels, on CUDA |

**The own kernel is not tuned, so this is not a verdict.** llama.cpp keeps TensorOps off before M5 on its own measurements
(its source comment: M2 Ultra about 5% slower, M4 no significant difference). Phase 2 tunes before choosing, and the
choice is per shape: prefill and decode need not use the same path. Beyond the M1, *BaseRT: Advancing Best-in-Class LLM
Inference with Apple M5 Neural Accelerators*, arXiv 2607.19438 (abstract read only), routes compute-bound matmuls through
the accelerators and keeps memory-bound decoding on existing kernels. Apple also says prefill is compute-bound and decode
is bandwidth-bound (`mlx-distributed-inference`, machinelearning.apple.com).

**TensorOps and FP4.** The M1 Pro compiles `matmul2d` with FP4 E2M1 weights at MSL 4.1. Whether that runs correctly and fast,
and whether it can take Mila's per-group scales or only Metal's own block-scale planes, is Phase 5's question.

**Binding.** Measured: a tensor bound alone at a buffer index gave all-zero output with no error, and a struct of resource
ids at one index worked. So every kernel test checks values, and a test that only checks completion is not a test here.

### 4.3 Compiling kernels

Compile at run time from source embedded in the library, first. It needs nothing but the OS; offline `metallib`s need full
Xcode, which the development Mac does not have. Phase 1 measures what run-time compilation costs a first load. A
`metallib` or a pipeline cache follows only if that cost earns it.

### 4.4 Decode

`IExecutionContext::createDecodeRecording()` stays null, so a decode step is called every step. The rule in `CLAUDE.md`
still binds, so that replay remains possible: read the position from the execution context, no per-token launch argument.
Indirect command buffers (Apple3 and later) are the candidate for replay, and only after Phase 4 shows what a call costs.

`synchronize()` commits and waits. How much a command buffer costs per token is measured in Phase 1, since a decode step
with a dozen operations should not pay for a dozen commits.

## 5. Memory

**Storage mode:** shared. The allocator returns the buffer's `contents` as the host pointer, so `is_host_accessible` is
true. The scaffold's bridged handle is not a host pointer, and goes.

**Binding by pointer.** A Mila tensor holds a pointer; Metal binds a buffer and an offset, or a GPU address. The memory
resource keeps the map from address range to buffer, and every operation binds through it. Phase 1 decides between
binding each buffer on the encoder and one residency set with `gpuAddress` (residency sets are Apple6 and later).

**Limits.** `maxBufferLength` is below the working set on this machine (9.5 GB against 12.7 GB, both measured), so a tensor
above it is refused by the planner with the number, not split. `dim_t` and `narrowToKernelIndex()` carry over.

**The planner** (`Deployment.md`) budgets from `recommendedMaxWorkingSetSize`. Apple calls this an approximation, and unified
memory is shared with everything else the Mac is doing, so Phase 4 measures how near it a model can really run.

**Weights.** A file whose stored form is the run form (FP4, FP8) is mapped and wrapped without a copy, which is the question
`Issues/Future.md` *Compute backends beyond CUDA* raises, if `newBufferWithBytesNoCopy` accepts a file mapping of this size;
Phase 4 tries it. BF16 weights running as FP16 are converted once at load. Published packages do not change:
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

## 7. Build and the scaffold

**The scaffold is replaced, not adapted.** `Compute/Devices/Metal/` predates the current direction: `#ifdef` and Objective-C
in module units, Intel and discrete paths, one type defined twice (`Issues/Vnext.md`), `MTLCopyAllDevices`, and capability
checks that ask the wrong questions (`Issues/Untriaged.md`). `Lib/CMakeLists.txt`'s `MILA_HAS_METAL` block goes with it.

**Host code is C++ through metal-cpp** (`github.com/apple/metal-cpp`, header-only, tagged to macOS 27.0.1, with
`MTLTensor.hpp` and `MTLResidencySet.hpp`), pinned as other dependencies are. No Objective-C in module units. Whether it
coexists with `import std;` under Clang is the first thing Phase 0 tries.

**CUDA in the wrong places.** 75 files outside `Devices/Cuda` name it (a text search, so some are comments and some are
legitimate), 11 under `Components/Transformers`, and `Llama.ixx` imports `Compute.CudaPinnedMemoryResource`. Each is either a
trait or a finding. A build with CUDA off compiles today, apart from `mila-store`; running on Metal is what tests the rest.

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

### Phase 1 -- Device, memory, one kernel

**Question:** can Mila allocate, dispatch and read back through its own device and memory types?

- `MetalDevice`, `ExecutionContext<Metal>` and the memory resource, replacing the scaffold.
- The three checks of 2.1. A Mac below the floor is refused at startup in plain words.
- A capability report tool in `Lib/Tools/`: everything in 2.1 and 2.2, plus compile-and-create of every kernel Mila ships.
- `TensorOps` fill and copy.

**Tests:** every refusal with a forced failure. Values read back. **Measures:** first-load compile cost; command buffer commit
cost per dispatch; binding per buffer against a residency set.

**Gate:** tensor tests pass on `DeviceType::Metal`, and the report runs.

### Phase 2 -- The matrix path

**Question:** which path does Linear use, per shape, and what is the anomaly?

- The timing program brought under `Lib/Tools/`, on real Linear shapes, decode's M = 1 included.
- The own kernel tuned, with accumulation wider than FP16, before any comparison counts.
- TensorOps with FP16 inputs and FP32 output: does it ease FP16's range at no cost, as it did here?
- **The unexplained result:** TensorOps BF16 at the decode-like shape took half of FP16. Tile size, shape or data? Until it is
  explained, no decision rests on it.

**Gate:** a written choice of path per shape, with the numbers, in the notebook.

### Phase 3 -- Llama's operations

**Question:** does each operation give correct values in FP16 and in BF16?

In order: Linear, TokenEmbedding, RmsNorm, Rope, Swiglu, Residual, grouped-query attention (prefill and decode), Sampling,
NextTokenLogProbability. BF16 weights first, quantized in Phase 5. BF16 kernels are tested here through the emulation layer:
that shows correctness, and says nothing about the speed of native BF16.

**Tests:** each against a double-precision reference in both precisions, with the decode rule (`CLAUDE.md`) held by a test that
sets the position first.

**Gate:** all pass, and a missing specialization is a compile error, not a skip.

### Phase 4 -- First model, both ways

**Question:** does FP16 hold up on a real model, and what does each way cost? (Todd: test both ways on the M1.)

Llama 3.2 3B on BF16 weights, at `LlamaModel<Metal, FP16>` and `LlamaModel<Metal, BF16>`, at its defaults. The gate is written
first, and may overturn FP16.

**Tests:** token ids and logits against HF; generation by reading it; activation ranges per layer, overflow and flush counted.

**Measures:** prefill and decode rate both ways; memory against the budget; how near `recommendedMaxWorkingSetSize` it runs;
no-copy weights; llama.cpp on the same Mac.

**Gate:** the 3.2 decision confirmed or changed for this model, in the notebook and here.

### Phase 5 -- Quantized weights and the KV cache

**Question:** do FP8 and FP4 run correctly, and can TensorOps' formats carry Mila's?

Llama 3.2 3B FP8 and FP4, Llama 3.1 8B FP4. Whether `metal_fp4_e2m1_format` can take Mila's per-group scales, or the kernels
dequantize. Per-token KV compression.

**Gate:** as Phase 4, per format.

### Phase 6 -- The other families

**Question:** do Gemma and Qwen hold FP16 too? FP16's gate is per model, and they are the agentic ones.

Gemma 4 12B and Qwen 3.8 27B, on what 16 GB of unified memory allows, with the agentic quality bar unchanged. Family parity
holds: build order justifies no gap. Apple states Qwen 3.8 is native BF16.

**Gate:** as Phase 4, per model.

### Phase 7 -- Other chips

**Question:** which chips have native BF16, and which matrix path wins on each?

Needs hardware Mila does not have: the report and the timing program, on each generation that can be borrowed or rented. A
row of 3.3 changes on a measurement from that family and nothing else. Accelerators are measured on M5 here, and the path
choice of Phase 2 may become per family.

**Gate:** each row filled from that chip's own run, and a row with no run stays "Unknown".

### Phase 8 -- Apps and distribution

Chat, Mila Desktop's session process and the server on macOS; how a Mac user installs Mila; what the model cards may claim.
Planned when Phase 6 is done. An app adds no capability to the library.

## 10. Decisions

| Decision | By |
|---|---|
| GPU only, M-series only. No CPU compute, no Intel | Todd, 2026-11-04 |
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
| Host code uses metal-cpp, subject to Phase 0 showing it works beside `import std;`. The alternative is Objective-C++ implementation units behind a C++ interface | Todd, 2026-11-04 |
| Whether a kernel calls TensorOps is what Phase 2 measures, with own kernels kept where the result is close | Todd, 2026-11-04 |
