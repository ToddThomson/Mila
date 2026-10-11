# Retiring the CPU Compute Operations

**Status:** Decided 2026-10-11 (Todd): the CPU compute operations are retired. Retirement is deletion; git history
is the archive. **Blocked on coverage** (section 4): nothing is deleted until every behaviour only a CPU test proves
is proven on CUDA, and two of the operations are Qwen 4's only implementation today. **Admitted to v0.21.0**
2026-10-11 (Todd) as an internal fix (`BACKLOG.md`).

## 1. What is retired, and what is not

| | Fate |
|---|---|
| The CPU compute operations: the 15 `OperationTraits.Cpu` specializations and their kernels under `Compute/Devices/Cpu/Operations/` | **Retired** |
| The components' CPU instantiations, the GPT-2 model on CPU, `CpuAdamWOptimizer`, and their `.Cpu.cpp` tests | **Retired** with them |
| The CPU **device**: host memory resources, staging, host tensors and their `TensorOps` (fill, zero, copy, random, math) | **Kept** — it is the host |
| The **CUDA-off build** (`MILA_ENABLE_CUDA=OFF`) | **Kept** — it guards the device boundary and builds `mila-store` and the store tests without the toolkit |
| The `TDeviceType` axis | **Kept**, with one compute value until a second backend arrives |
| Device-independent tests that happen to be named `.Cpu.cpp` (Distribution, Quantization packing, Serialization, Apps) | **Kept**, untouched |

## 2. Why

No end user runs a CPU model, and none of the CPU-only needs looked at serves one:

- **Running models:** an edge case (Todd), and the real CPU-only edge user — home automation on a Raspberry Pi or
  mini PC — needs 4-bit weights read on the CPU, NEON/AVX2 kernels and threads. These FP32 operations are not a
  step toward that; it would be a new backend, started from a measured quantized Linear, and it is breadth.
- **Testing the families on CPU:** a CPU arm tests a second implementation of the math, not the CUDA kernels,
  formats, replay or planning a user runs. Cheaper routes cover what it would: device-independent protocol and
  store tests, a CUDA compile-only CI job, and a GPU runner.
- **Reference:** no CUDA test checks against a CPU operation (one GPT-2 test is the exception); each computes its
  own exact reference on the host.
- **Cost:** every change to the `Operation` interface — decode, the decode position, scratch — is carried into 15
  operations nothing ships.

Edge hardware that matters for language models is mostly CUDA (Jetson, NVIDIA's ARM desktop boxes), which is a
direction for the CUDA path, not a reason to keep this one.

## 3. Method

For each CPU test file that goes, every behaviour it proves is either proven by a CUDA test already, moved to a
CUDA test, or judged not worth keeping by name. A first pass counted `backward`/`gradient` references per file;
section 4 is the per-test reading that replaced it.

## 4. Coverage, test by test

Read 2026-10-11, each CPU test against its CUDA counterpart. Every CPU test not listed below is proven by a
CUDA test of the same name or the same assertion. Duplicates of `GetType`, `ToString`, accessor and
parameter-count tests are not carried: the component code they run is the same in the CUDA instantiation, and
the CUDA files hold their own.

**The finding that matters: on CUDA, backward is mostly checked for shape and finiteness, not against a
reference.** Where a CUDA test does compare, it often drives backward with an output gradient of all ones, which
a kernel that ignored the output gradient would pass (`Gelu.Cuda`, `Activation.Cuda`). Linear and Residual are
the exceptions: both compare against an exact reference with a non-uniform gradient.

### 4.1 Move to CUDA: correctness the CPU alone proves

| CPU test | What only it proves | CUDA test to write |
|---|---|---|
| `Network.Cpu` `Forward_MatchesReference`, `Backward_PropagatesGradientsThroughGraph`, `CreateOptimizer_RegistersAllParameters`, `TrainingLoop_ReducesRegressionLoss`, `Backward_ThrowsWhenBuiltForInference`, `GetGradients_EmptyWhenBuiltForInference`, `Build_PropagatesToChildrenAndAggregatesParameters`, `GetParameters_AggregatesChildrenInOrder` | The training spine: a composite's backward chains its children and lands each layer's parameter gradients, the optimizer is wired to all of them, and a loop reduces a loss | `Network.Cuda`. The network under test is already templated on the device; the work is host-device copies |
| `MultiHeadAttention.Cpu` `Backward_ReturnsInputShapedGradient`, `Backward_MatchesNumericGradient` | Attention backward at all: CUDA only checks that it refuses an inference build | Backward against a numeric gradient (FP32), non-uniform output gradient |
| `LayerNorm.Cpu` `Backward_MatchesReferenceGradients`, `Backward_MatchesNumericGradient`, `Forward_MatchesReferenceWithoutBias` | LayerNorm backward at all (input, weight and bias gradients), and the no-bias forward | Backward against the reference and a numeric gradient; forward without bias |
| `RmsNorm.Cpu` `Backward_MatchesNumericGradientWithUnitOffset`, `Grouped_BackwardMatchesNumericGradient`, `Forward_MatchesReferenceWithoutBias`, `Forward_AppliesUnitOffset`, `Forward_NarrowerInputThanBuildMatchesReference`, `Grouped_DiffersFromOneGroupOverTheSameExtent`, `Grouped_GroupSizeThatDoesNotDivideTheExtentIsRefused`, `GetRequiredMemory_MatchesBuiltFootprint` | RmsNorm backward (plain, unit offset, grouped); unit offset forward; narrower input than the build; the grouped refusal; the footprint | One test per row |
| `Softmax.Cpu` `Backward_MatchesReferenceGradient` | Softmax backward at all | Backward against the reference, non-uniform output gradient |
| `Lpe.Cpu` `Backward_ScattersGradientIntoTables` | Positional and token tables receive the gradient at the right rows | Same, on CUDA, with repeated tokens so rows must accumulate (the CPU test's tokens never repeat) |
| `Gelu.Cpu` `Backward_ChainsNonUniformOutputGradient`, `Backward_MatchesNumericGradient` | Backward multiplies by the output gradient | Non-uniform output gradient in `Gelu.Cuda` and `Activation.Cuda`'s backward tests |
| `MLP.Cpu` `Forward_MatchesComposedReference`, `Forward_MatchesComposedReferenceWithoutBias`, `Backward_InputGradientMatchesFiniteDifference`, `Backward_ProducesNonZeroParameterGradients`, `Backward_MatchesNumericGradient` | MLP output and gradients are right, not only finite | Composed reference forward; backward against finite differences |
| `GptBlock.Cpu` `Backward_InputGradientMatchesFiniteDifference`, `Backward_ProducesNonZeroParameterGradients` | Block backward is right, not only finite and non-zero | Finite-difference input gradient (FP32); non-zero gradient for every parameter |
| `GptTransformer.Cpu` `SaveThenLoad_RestoresEveryParameterAcrossNestedBlocks`, `SaveThenLoad_RecoversTheConfigThatWroteIt`, `Load_ThrowsWhenTheArchiveDescribesADifferentNetwork`, `Backward_ThrowsWhenForwardNotCalled` | GPT-2's save and load round trip, and the refusals | Same, on CUDA, like Llama's `SaveThenLoad_RestoresEveryParameterAcrossBlocks` |
| `GptModel.Cpu` `SaveCheckpoint_RoundTripsEveryBlobUnchanged`, `FromCheckpoint_ThrowsOnAnArchiveItDidNotWrite` | The checkpoint round trip and its refusal (`GptModel.Cuda` loads from a checkpoint but never writes one) | Same, on CUDA |
| `AdamW.Cpu` `Convergence_MinimizesQuadraticToTarget`, `BiasCorrection_EarlyStepsDiffer`, `EdgeCase_MixedSignGradientStaysFinite` | AdamW reaches a minimum, and early steps are bias-corrected. `AdamW.Cuda` checks only the direction of each update | Same, on CUDA, plus one step checked against the AdamW update computed on the host |
| `CausalConv1d.Cpu` `RestoredSnapshotContinuesTheSequence`, `StateDepthIsKernelMinusOneTimesDilation`, `DilationOneDiffersFromTheTorchDilatedReference` | The conv's state snapshot and restore, alone (CUDA reaches it only through `Qwen.DeltaNetBlock`), its depth under dilation, and that dilation is applied | Same, on CUDA |

Writing these tests found two CUDA defects the CPU tests had hidden, both fixed 2026-10-11: RmsNorm backward
ignored the unit offset (every Qwen norm's gradient was wrong), and Softmax backward threw instead of calling its
kernels, which existed but took rows and a row width where the operation passed an element count and an axis.

### 4.2 Move to CUDA: refusals a user can reach

These are component code shared by both devices, but only the CPU instantiation exercises them. They guard
errors a caller sees, so they move.

| CPU test | CUDA test to write |
|---|---|
| `Linear.Cpu` `LoadParameter_ThrowsOnDtypeMismatch`, `LoadParameter_ThrowsOnShapeMismatch`, `LoadParameter_ThrowsOnUnknownName`, `Forward_ThrowsOnInputFeatureMismatch` | Same |
| `LayerNorm.Cpu` `LoadParameter_SetsWeightAndBias`, `LoadParameter_ThrowsOnShapeMismatch`, `Build_ThrowsOnTrailingDimMismatch`, `Forward_ThrowsBeforeBuild` | Same |
| `MultiHeadAttention.Cpu` `Build_ThrowsOnWrongTrailingDim`, `Build_ThrowsOnNonRank3`, `Forward_ThrowsBeforeBuild` | Same |
| `Softmax.Cpu` `Build_ThrowsOnAxisOutOfBounds`, `Forward_ThrowsBeforeBuild` | Same |
| `Lpe.Cpu` `Build_ThrowsOnNonRank2Input`, `Build_ThrowsWhenSeqExceedsMax`, `Forward_ThrowsBeforeBuild` | Same |
| `MixtureOfExperts.Cpu` `Build_ThrowsOnHiddenSizeMismatch`, `FlatNames_AreTheCheckpointVocabulary` | Same |
| `Router.Cpu` `Build_ThrowsOnHiddenSizeMismatch`, `Forward_ThrowsBeforeBuild`, `ParameterCount_ExcludesDerivedNormWeight`, `FlatNames_AreTheCheckpointVocabulary` | Same |

### 4.3 Covered, or not kept

- **Covered on CUDA:** `Activation.Cpu` (apart from the output gradient above), `Residual.Cpu` (the CUDA
  backward test is exact, with a non-uniform gradient; the CPU numeric gradient adds nothing for a sum),
  `Linear.Cpu`'s forward and backward, `MixtureOfExperts.Cpu` and `Router.Cpu`'s references, `GptModel.Cpu`'s
  reconstruction, and the rest of `AdamW.Cpu`.
- **Not built, deleted:** `Model.Cpu` (its `CMakeLists.txt` line is commented out, its mocks stub the training
  loop, and one assertion is a `FIXME`) and `CrossEntropy.Cpu` (empty, commented out).
- **Qwen 4:** `GatedResidual.Cpu` and `NgramEmbedding.Cpu` move with their CUDA operations, gated against the
  same captured reference (`EveryResidualOfEveryStepMatchesTheCapture`, `EveryIdAndEveryEmbeddingEqualsTheCapture`).
  Their validation and metadata tests carry over unchanged.

Two findings change the order of work:

1. **The training spine is proven only on CPU.** `Network.Cpu` holds the network-level training tests — build for
   training, backward through the graph, optimizer registration, and a training loop that reduces a regression
   loss — and nothing on CUDA proves the same. Multi-head attention, LayerNorm, RmsNorm and Softmax backward are
   checked against a reference only on CPU (4.1). With training returning, these move to CUDA before the CPU
   path goes.
2. **Qwen 4 is mid-flight on CPU.** `GatedResidual` and `NgramEmbedding` exist only as CPU operations; Qwen 4's
   Phase 2 gated them on CPU against the tiny reference (`Qwen4.md` section 9). Retiring them now deletes the only
   implementation. Their CUDA operations — which Qwen 4 needs anyway — come first, gated against the same reference.

## 5. Next

1. **Done 2026-10-11:** every test in 4.1 and 4.2 is on CUDA and passes. Each new reference test was
   mutation-checked: a GELU backward that ignores its output gradient, a LayerNorm backward missing its mean
   term, an AdamW update without its second-moment bias correction, and a Softmax backward that ignores its
   output gradient each fail the tests that should catch them. `AdamW.Cuda.cpp` had never been built (it sat
   disabled in `Tests/CMakeLists.txt`); it is revived on the exported `AdamWOptimizer`, and of its 41 tests only
   the new host-reference step caught the bias-correction mutation.
2. **Done 2026-10-11:** the CUDA operations for `GatedResidual` and `NgramEmbedding`, FP32 and BF16, gated against
   the same capture (`Qwen4.md` Phase 2 progress). The CPU gates had been skipping on the development machine,
   since the capture had only been made in a cloud session; it now exists there and both CPU and CUDA gates run.
3. The retirement change: delete the operations, their `OperationTraits.Cpu` entries, the CPU instantiations and
   the retired tests; reword "CPU-practical" text in specs and comments; the CPU-only CI job then proves the build
   and the device-independent tests, and says so. Outside the `.Cpu.cpp` files the change also reaches the CPU
   `static_assert`s in `Tests/Dnn/Compute/Operations/OperationTraits.cpp` and the `--device cpu` option of the
   MNIST and Bard samples. (`GptModel.Cuda.cpp` wrote its checkpoint with a CPU network; it now uses CUDA.)

The CPU device's memory reading (`Untriaged.md`, 2026-10-11) goes at triage: with no CPU compute backend, the
planner never offers the CPU.
