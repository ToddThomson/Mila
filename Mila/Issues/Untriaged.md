# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined after ninety days**, and anything user-reported is a pointer to its
GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## A consumer that enables CMake's own `import std` may get two modules named `std`

`cmake/StdModule.cmake`, `mila_add_std_module_library` @ `b2bdcc9`

Found 2026-10-12 converting the library to `import std;`. Mila links its own `MilaStd` target, which builds `std`
and `std.compat`, PUBLIC from `Mila`, so every FetchContent consumer gets it. A consumer that also sets
`CXX_MODULE_STD` (CMake's experimental support) would have CMake build a second `std` from the same source in the
same build. Not tested: whether CMake, MSVC or Clang reject that, pick one, or build both silently.

## Gelu and Residual zero a gradient buffer the CUDA kernels overwrite

`Mila/Src/Dnn/Components/Activations/Gelu/Gelu.ixx` and `Connections/Residual.ixx`, `backward` @ `4db0309`

Found 2026-10-11 retiring the CPU operations. Both components `zero()` their input gradient before calling the
operation, for the CPU operations that accumulated into it. The CUDA kernels assign every element in range
(`Gelu.Fp32.cu`, `Residual.Fp32.cu` / `.Bf16.cu`), so the zeroing is a memset per backward that nothing reads.
Not checked: whether any caller reads past the range a narrower-than-built input writes, which the zeroing keeps
at zero today.

## The Qwen 4 tiny reference never reaches the n-gram hash's negative branch

`Mila/Tools/Converters/Qwen4/hf_qwen4_tiny_reference.py` @ `44778cd`

Found 2026-10-11 mutation-checking the CUDA n-gram hash: a truncating `%` in place of the floor mod passed the capture
gate, and only the hand-built `Hash_NegativeMixedValueMapsIntoTheHead` caught it. The capture's largest token id is 239
and its largest multiplier 3.5e16, so every product stays below 2^63 and no mixed value is negative. Qwen 4's real
vocabulary (about 250K ids) wraps them. The header says the n-gram base near 1,000 exercises the floor mod; it
exercises collisions, not the negative branch. Token ids above about 2^63 / max(multiplier), 260 here, would.

## The Qwen 4 tiny reference generator refuses its own capture under transformers 5.19

`Mila/Tools/Converters/Qwen4/hf_qwen4_tiny_reference.py:392` @ `44778cd`

Found 2026-10-11 regenerating the capture locally (it had only been made in a cloud session, so the CPU gates had been
skipping here). With transformers 5.19.0 the dense-exact prefix check requires the selected and dense passes to agree
exactly and they differ by 1.192e-7, one float ulp; every file is written before the check raises. The capture used
5.16. Whether the check is too strict for a different version or machine, or 5.19 changed the indexer, is not known.
5.19 did change `qwen4_exp`: its config names the sparse layer type `indexed_attention` where 5.16 wrote
`qwen_sparse_attention` (`convert_weights.py` treats every non-linear type as full attention, so the converter is
unaffected). The tracked `hf_checkpoint/*.json` were left at their 5.16 contents; the local capture is 5.19's.

## The loss GPT-2 and MNIST train with has no test that runs

`Mila/Tests/CMakeLists.txt` (the parked `SoftmaxCrossEntropy.Cuda.cpp` line) @ `44778cd`

Found 2026-10-11 moving the CPU tests to CUDA. `SoftmaxCrossEntropy.Cuda.cpp` sits in the CMake list's parked
section ("loss-on-device milestone"), and the empty `CrossEntropy.Cpu.cpp` went with the CPU operations, so no
built test reaches the softmax cross-entropy component or its CUDA operation. `AdamW.Cuda.cpp` sat in the same section until
2026-10-11 and no longer compiled against the optimizer it tested.

## No model Mila runs can be post-trained, even with LoRA on a 16 GB card

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx:366` @ `44778cd`

Found 2026-10-11 when Todd asked about post-training on 16 GB. Todd's intent: triage this into a full
specification, written from the discussions still to come. Training runs only along the GPT-2 line (MHA,
LayerNorm, GELU, AdamW). A full AdamW fine-tune costs about 16 bytes a parameter, which caps 16 GB below 1B
parameters. LoRA over a frozen base fits: Llama 3.2 3B with a BF16 base (~6.4 GB), and Llama 3.1 8B (~4.5 GB) or
Gemma 4 12B (Q4_0 QAT, ~7 GB) with a 4-bit base (QLoRA). The two ~27B models leave no room for activations.
Missing pieces: GQA backward (it throws; the retired expanded-layout sketch in that file is the cheap way back),
BF16 training (SwiGLU's BF16 backward is a known skip, `Future.md` "Training (advanced)"), LoRA adapters on
`Linear`, and the input gradient through quantized weights for a QLoRA base. DPO with reference log-probs
computed offline adds no memory; GRPO also needs generation. A LoRA trained over Gemma's Q4_0 QAT base fits the
"run in the format it was trained for" direction.

## CI never compiles the CUDA build, so Clang portability defects in the code users run go unseen

`.github/workflows/build-pipeline.yml:198` @ `98ac393`

Found 2026-10-11 discussing the CPU backend. The CPU-only job compiles a different module graph -- the CUDA
modules are absent -- so a missing `#include` that MSVC resolves transitively in CUDA-side or model code reaches
Linux only when someone builds the GPU configuration there. nvcc and Clang compile CUDA code on a runner with no GPU;
a job that configures `MILA_ENABLE_CUDA=ON` and builds without running catches that class in the code users get.
Needs the CUDA toolkit on the runner (the dev image already carries it, `Docker/Dockerfile`).

## No model-level test runs anywhere but Todd's machine

`Mila/Tests/Dnn/Models/` @ `98ac393`

Found 2026-10-11 discussing the CPU backend. Every family's model tests -- generation, log-likelihood, decode
replay, prefix reuse, multi-token decode -- are `.Cuda.cpp`, and the hosted runners have no GPU, so CI never runs a
Llama, Gemma or Qwen forward pass. A GPU runner would: self-hosted on Todd's machine, or sponsored compute, which
the sponsorship plan already asks for (compute only). A CPU path for the families was considered and rejected
(`Specifications/Notebooks/CpuOperationsRetirement.md` section 2).

## Code retired under the old keep-it-in-the-tree rule still sits in the tree

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/` and the files below @ `98ac393`

Found 2026-10-11. Todd's rule changed that day: retired code is deleted, and git history is the archive. Wholly
retired files, out of the build: `Gqa.Decode.Bf16.cu`, `Gqa.Flash.Wmma.cu`, `Gqa.Flash.Bf16.cu`,
`CudaRopeOp.Cache.ixx`, `MLP.Dispatch.ixx`, `CpuLinearOpTypeMap.ixx`, Chat's `Chat.GemmaToolCallParser.ixx` and
`Chat.QwenProtocol.ixx`, MIS's `gemma_protocol.py` (also a BACKLOG item), and the tests
`CudaRopeOp.Calculated.Cuda.cpp` and `Gemma.DelegatedFeedForward.Cuda.cpp`, with their commented-out entries in
`Mila/CMakeLists.txt:276` and `Mila/Tests/CMakeLists.txt:253`. Retired functions inside live modules:
`Tools/Quantization/pack_qwen.py:162` and `artifact.py:181`. Specs that cite a retired file as a reference need
rereading when it goes.

## The CPU device reports no memory, so no plan can be priced against it

`Mila/Src/Dnn/Compute/Device.ixx:78`, `Mila/Src/Deployment/DeviceReading.ixx:43` @ `98ac393`

Found 2026-10-11 reviewing what the `mila` command line should carry (a "will this model fit" verb). Only
`CudaDevice` overrides `getMemoryInfo`; the CPU device inherits the base's empty reading, and `DeviceReading`
treats zero as "not a full device". The planner reads devices through the registry without naming a backend, so
on a machine with no CUDA device it has nothing to plan against. Separate from, and needed alongside, the families
having no CPU path (grouped-query attention has none).

## An evaluation's two arms cannot run either side of midnight

`Mila/Tools/Evaluation/run_arm.py` `lm_eval_arguments`, `compare_arms.py` `check_pairing` @ `6dcac3b`

Found 2026-10-10 when Todd asked for evaluations to run overnight. Llama 3's chat template writes today's date
into every prompt (transformers renders it with `strftime_now`), so a pair whose arms start on different days
has no matching prompts and `compare_arms.py` refuses it. Phase 1 takes about 2.7 hours an arm on the 5060 Ti, so
an overnight pair crosses midnight. The template takes a `date_string` when one is given.

## The Qwen 3.8 2.82-bit model card says "residency"

`Mila/Tools/ExportArtifact/ModelCards/Qwen3.8-27B-cb2-3/README.md:53` @ `539396e`

Found 2026-10-07 reading the cards' quality sections for the evaluation discussion. "The cost of the smaller
residency" -- a term `CLAUDE.md` lists as one only Mila uses. The section's numbers are also against Mila's own FP4
build, not the upstream model.

## Qwen 4 27B is expected around November 2026 and Mila has no Qwen 4 chassis

`Mila/Specifications/Qwen4.md` @ 539396e

Compared Qwen3.8-Flash-Next, the Qwen 4 architecture preview, against the Qwen 3.8 chassis and wrote the
spec. Its section 9 phases 0 to 2 (the tiny reference, the converter skeleton, grouped RmsNorm, dilated
CausalConv1d, the DeltaNet gate activation, and the gated residual, n-gram, PLE and QSA indexer components)
need no 27B checkpoint. 2026-10-08: Phases 0 and 1 are built and their CPU gates passed; Phase 2 has
`GatedResidual` and `NgramEmbedding` on CPU, gated against the tiny reference.
