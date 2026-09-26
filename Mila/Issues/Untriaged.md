# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined at the production release tag**, and anything user-reported is a
pointer to its GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## A fix to the published site sits in the repo until someone dispatches the workflow

`.github/workflows/publish-site.yml` (`on: workflow_dispatch:`) @ `c60c100a`

Found reading Google Search Console for `mila.toddt.me`: 141 pages reported **"Excluded by 'noindex'
tag"**, against 1 indexed page and 14 others crawled-or-discovered-but-not-indexed.

The `noindex` post-processing step was removed from the workflow at `0589673f` on 2026-08-13. The
workflow only runs on dispatch, and the next dispatch was 2026-09-11 — so the live site served
`noindex` across the whole `/api/` tree for 29 days after the repository had stopped saying it.
Verified on the live site 2026-09-22: no `robots` meta and no `X-Robots-Tag` on `/api/index.html`,
`/api/annotated.html`, `/api/files.html` or `/api/classes.html`. GSC validation is running.

Nothing reconciles the site source in the repository against what is actually deployed, and the
workflow's own verification step reads one file (`build/site/api/index.html`) out of the tree it
assembles.

## The scratch reservation test measures growth on the card that drives the display

`Mila/Tests/Dnn/Models/ScratchReservation.Cuda.cpp:150` @ `1f823df4`

`ScratchReservationCudaTests.Gemma4_12B_Fp4_Context8192` bounds device memory growth during
generation at 64 MiB, measured on the current CUDA device -- ordinal 0, the RTX 4070 that drives
the display, shared with 18 desktop processes. Same binary, three runs on 2026-09-23: 2.0, 1.9 and
116.6 MiB; a full-suite run earlier the same day read 375.3 MiB. Scratch predicted and reported were
identical in every run. The file's own note (`:40`) records noise reaching 47.6 MiB.

## A skipped test sends the reader to a backlog entry that no longer exists

`Mila/Tests/Dnn/Components/FFN/Swiglu/Swiglu.Cuda.cpp:358` @ `1f823df4`

`SwigluCudaTests/Bf16.Backward_MatchesReferenceGradients` skips with "BF16 SwiGLU backward
grad-dtype mismatch (FP32 kernel grads vs BF16 tensors) -- see BACKLOG". `BACKLOG.md` at `+3` has
no SwiGLU entry, so the skip's only record of why it exists points at nothing.

## The deployment types repeat their namespace in their names

`Mila/Src/Deployment/` @ `0.21.0-dev+6`

Moved to the top level in `Mila::Deployment` at `+6` (Todd, 2026-09-24: `Dnn/` held it inconsistently with its
siblings, which each build or run a network), with the type names `Deployment.md` 12.7 recorded left as they
were: `Mila::Deployment::DeploymentPlan`, `DeploymentPlans`, `DeploymentRequest`, `DeploymentRefusal`,
`DeploymentRefusedError`. The namespace would let them drop the prefix, as `Mila::Dnn::Conversation` did for
`Turn` and `ToolCall`. One layering fact the move made visible: unlike `Distribution/`, which is independent
of `Dnn` both ways, deployment imports `Dnn.Component`, `Serialization.WeightsReader` and `Compute.DeviceId`,
and `Dnn/Models`' three families import it. See the `DeviceReading` entry below.

## `DeviceReading` is the wrong name for what a plan was decided against

`Mila/Src/Deployment/DeviceReading.ixx` @ `0.21.0-dev+6`

Todd, 2026-09-23: the name is poor; kept for Phase 3 and to be replaced. What the type carries: one
device's identity, its free and total memory, and its allocation granularity, taken once after the graph is
constructed (`DeviceReading::take`). Every plan and refusal holds one (`DeploymentPlan::reading()`,
`DeploymentRefusal::reading()`), Chat's `-p` JSON reports its fields, and `Deployment.md` 3.3 and 9 use the word.
Decision 12.5 (planning for a device that is described rather than read) bears on the name, since a
described device is not a reading of anything.

## The /model list column prices capacity, which the planner cannot yet be asked about

`Mila/Adaptors/Chat/Src/Chat.ModelCatalog.ixx` (`largestFittingContext`) @ `0.21.0-dev+6`

Deployment Phase 3 moved Chat's load and `/context` onto `planDeployment`, but the listing's GPU FIT column
still walks its own context ladder through `getDeploymentFootprint` and `gradeFootprint`. It answers "could
this card run it at all" against the card's TOTAL memory, deliberately, so a resident model is not charged
to every row. A plan is decided against a live reading of FREE memory, so moving the column onto plans needs
a described reading -- `Deployment.md` 12.5, undecided. `Deployment.md` section 8 says the column renders
plans; it does not yet.

## A find_package consumer may be told C++20 where Mila's modules need C++23

`Mila/CMakeLists.txt:71-75` @ `a8d30790`

When CMake's feature table lacks `cxx_std_23` (the comment names Clang 21.x), the exported `Mila` target
advertises `cxx_std_20` to consumers, who compile Mila's installed module units under it. Those units
already use C++23 library facilities -- `std::ranges::fold_left` at `Mila/Src/Dnn/Tensors/Tensor.ixx:806`
-- which libc++ and libstdc++ provide only in C++23 mode. Not reproduced: whether CMake's synthetic target
then compiles at C++20, or the compiler's default wins, was not tested. Found while checking whether
`std::expected` (C++23) could appear in the public API for `Deployment.md` 3.3.

## A process's second automatic load chooses a shorter context than its first

`Mila/Src/Deployment/DeviceReading.ixx` (`take`) @ `0.21.0-dev+7`

Found verifying Deployment Phase 4 through the Python binding on the RTX 4070: four `GemmaModel.from_store(
"gemma-4-12b-it-fp4")` loads in one process, each deleted before the next, chose 121856, 120832, 120832, 120832.
Stable after the first, so not a leak -- memory the first load leaves held in the process (lazily loaded kernel
modules or a library workspace are candidates; not measured). A fresh process always chooses 121856. It reaches
any caller that loads twice in one process: Chat's `/model` switching and a Python program. Not measured how many
bytes, or which.

## The expert bank works out for itself whether its policy is FP4, and so does its operation

`Mila/Src/Dnn/Components/MixtureOfExperts/MixtureOfExperts.ixx:82` @ `0.21.0-dev+8`

`MixtureOfExperts::kIsFp4`, `CudaMoeOp::kIsFp4` (`CudaMoeOp.ixx:58`, the same lambda verbatim) and
`CudaLinearOp::kIsFp4Weight` (`CudaLinearOp.ixx:178`) each probe `TWeightQuantization::kIsFp4E2M1` for
themselves. A trait on the policy in `Quantization/Weight/Policies.ixx` would state it once. Found reviewing the MoE
path for the Gemma parity pass.

## The 26B-A4B normalizes the same residual twice in every layer

`Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.Block.ixx` (routed branch) @ `0.21.0-dev+8`

The router's norm and `pre_feedforward_layernorm_2` both RMS-normalize the unnormalized residual
(`Gemma4MoE.md` Phase 1), so one reduction could feed both. Unmeasured; 30 layers per token.

## The build targets plain `120`, and block-scaled FP4 needs `120f`

`CMakePresets.json:118`, `Mila/CMakeLists.txt:31` @ `0.21.0-dev+8`

`MixtureOfExperts.md` 7.2(b) requires the family-specific target for SM120 block-scaled FP4. Every preset and the
library default list plain `120`; the comment beside it says nothing in Mila uses sm_120 instructions yet, which
is still true. It becomes a blocker the day a CUTLASS grouped kernel or an NVFP4 path lands, and the list is kept
in step across both wheel presets and `scripts/dockerhub/publish-image.sh`.

## Training has no row in the family parity matrix, and only GPT-2 trains

`Mila/Specifications/ModelFamilyParity.md` section 3 @ `0.21.0-dev+8`

Todd, 2026-09-26: training is first-class, and asked for a second model with full training in 0.21.0; shelved
the same day to keep the Gemma pass moving. `ROADMAP.md` Future places "Training (advanced)" after v0.21.0
because GQA backward does not exist (`CudaGqaOp::backward` throws). Candidates discussed:
- SmolLM2-135M, with 360M as a step up: Llama architecture, about 2.2 and 5.8 GB of mixed-precision state.
  Recommended, since training would land in the Llama chassis.
- Qwen3-0.6B: about 9.6 GB, and a new chassis.
- Pythia-160M: MHA, so no GQA backward needed, with published loss curves.
- GPT-2 124M at full scale.

Llama 3.2 1B full training is about 20 GB, so it does not fit 16 GB; on that card it is a LoRA model. SmolLM2's
licence is unverified. Proposed matrix rows: backward, gradient parity against PyTorch, mixed precision, optimizer,
checkpoint save and resume, loss-curve parity. The choice also decides FP32's future; the kernel measurement is in
`Future.md` "Remove FP16".

## An out-of-range token id is an illegal memory access, not an error

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Embeddings/Kernels/TokenEmbedding.Fp8.cu:122` @ `0.21.0-dev+8`

Found through a test bug on 2026-09-26: 21 uninitialized ints read past a vector reached the FP8 embedding gather as
token ids, and the CUDA context died with `cudaErrorIllegalAddress`, reported first as a cuBLAS internal error in
the next GEMM. Nothing between a caller's ids and the gather checks them against the vocabulary, and a dead context
takes the whole process's CUDA state with it. `sequenceLogLikelihood` checks targets, but only on the host after
the forward has already run.

## Gemma's FP4 prefill with FP8 activations switched off lands far from decode

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/CudaLinearOp.ixx:171` (`kUseFp8ActivationPrefill`) @ `0.21.0-dev+8`

Set to false as an experiment on 2026-09-26, the Gemma 4 12B FP4 prefill takes the BF16 staging path and its logits
move far from both the default prefill and decode: argmax against decode 20 of 32 (31 with the switch on), mean KL
1.2 (6e-2 on), and the prompt's own next-token argmax changes. The switch is on by default, so no shipped build takes
this path for Gemma; it is unvalidated there, and it is the path the switch's own comment names as the fallback.
Reverted after the run. Later the same day, exact FP64 recomputation showed that W4A16 -- what this path should
compute -- is what decode computes, so this path should have landed *nearer* decode than W4A8, not far from it.
Read that as a probable defect in the FP4 BF16-staging prefill, not rounding.

## Gemma's FP4 prefill computes 2-4% away from its decode, and nothing measured what that costs

`Mila/Specifications/Fp8ActivationPrefill.md` @ `0.21.0-dev+8`

W4A8 -- FP8 activations per token, and the FP4 weights re-rounded to FP8 under one scale per tensor -- is
2.0e-2..3.6e-2 relative L2 from W4A16 per projection on Gemma 4 12B layer 0, measured against exact FP64
recomputation; Mila implements it to 1e-4. So a prompt's KV cache is built by different arithmetic than the tokens
generated after it, and a log-likelihood measures the prefill's. It shipped on token parity for short prompts
and a coherent chat. The one quality number available (raw-wikitext perplexity) favours W4A8 for a reason unrelated
to quality, so whether it costs quality on text the model is built for is unknown. The design record's own
"escalate to per-channel weight scales if parity fails" was never triggered. It buys 1.285x prefill.

## The FP8 head's batched path rounds every weight before it scales

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Fp8Prefill/CudaFp8Prefill.cu:79-82` @ `0.21.0-dev+8`

`dequantize_fp8_to_bf16_kernel` stores `fp8 * scale` as BF16, so every staged weight is rounded before the GEMM;
the decode matvec multiplies exact FP8 values and applies the per-channel scale once, in FP32. An E4M3 value is
exact in BF16, so the rounding comes only from folding the scale in early. Measured against exact logits from
`temb.wte` (ModelFamilyParity.md 8.2, G1 result): the staged path's extra error is this rounding alone, reproduced
to four digits by a model of it. Every per-channel FP8 Linear at more than one row takes the same path.
Decided 2026-09-26 (Todd): the fix -- scale after the dot product -- is its own change with a Linear-level gate
against exact FP64, and it sets `Gemma.LogLikelihood.Cuda.cpp`'s window bound back from 2e-3 to 1e-3.

## The site's crawl signals lag the site

`Web/hugo.toml:13-17` @ `0.21.0-dev+8`

From a cloud review, partly verified: the `hugo.toml` comment still says the `/api/` tree is marked noindex by a
workflow step that was removed at `0589673f` (verified), and `enableRobotsTXT = false` with no layout supplying a
sitemap line (verified). Unverified: the home page's sitemap `lastmod` reads 2026-08-15 because git info is off and
the workflows check out shallow; two blog posts link duplicate GitHub Discussions (`flash-decoding-mqa-cuda.md:8`,
`sponsor.md:23`); old `toddthomson.github.io/Mila/` Doxygen URLs have no redirect. See the noindex entry above.

