# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined after ninety days**, and anything user-reported is a pointer to its
GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

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
