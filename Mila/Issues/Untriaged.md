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
themselves. A trait on the policy (`Quantization/Weight/PerGroupFp4.ixx`) would state it once. Found reviewing the MoE
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


## Loading from Python before `mila.initialize()` blames the GPU

`Mila/Bindings/Mila_py.Wrappers.cpp:447` @ `0.21.0-dev+9`

`mila.GemmaModel.from_store( name, "auto", 1 )` without a prior `mila.initialize()` raises "CUDA device 1 does not
report its free memory, so no context length can be chosen ... Pass context_length as a number" -- on both cards,
both `load` and `from_store`. With `initialize` first, the same call opens at 262144. The refusal is
`DeviceDoesNotReportMemory` from `DeviceReading::take` (`Deployment/DeviceReading.ixx:61`), which swallows why it
could not read the device, so the message names the wrong cause and its advice (pass a number) is not the fix.

## Gemma 4 12B FP4 loses most of what long context should buy it

`Mila/Specifications/ModelFamilyParity.md` 8.2, G2 result @ `0.21.0-dev+9`

On PG-19 book 30312 inside a model turn, HuggingFace on BF16 weights reads 3.641, 3.339, 3.433, 3.487 nats per token
by 8K band through 32K; on the FP4 weights as Mila quantizes them, 3.830, 3.812, 4.024, 4.173, and Mila's own FP4 run
agrees with the FP4 row. The FP4 cost grows from +0.19 to +0.69 nats per token with context. Through 262144 Mila's FP4
keeps rising to 5.306. One book. The attention projections carry it: FP4 on attention alone costs +0.11 rising to
+0.47 by 32K; FP4 on the feed-forward alone a flat +0.08 to +0.13. Query and key carry most of that: FP4 on q_proj
and k_proj alone costs +0.15 rising to +0.38. Google's QAT weights at Q4_0 cost +0.03, confirmed in llama.cpp.

## Direction has no principle for which 4-bit format a model runs in

`Mila/Specifications/Direction.md` @ `0.21.0-dev+9`

Agreed in discussion 2026-09-27, not yet written anywhere in the repo: a model runs in the format it was trained for.
Producers now ship quantization-aware checkpoints, each fitted to one grid -- Gemma 4 QAT to llama.cpp's Q4_0,
gpt-oss to MXFP4, NVIDIA's tooling to NVFP4 -- and the benefit does not transfer across grids: Gemma 4's QAT weights
cost +0.03 nats per token at 32K in Q4_0 and +0.24 in Mila's FP4 (`ModelFamilyParity.md` 8.2, G2 result). Mila's
own `PerGroupFp4<128>` becomes the fallback for models without a QAT release. Where it is recorded (Direction,
Quantization.md, or both) is the triage call.

## The planner could choose which published variant of a model a device runs

`Mila/Specifications/Deployment.md` @ `0.21.0-dev+9`

`planDeployment` already chooses the context and prefill chunk per device from one reading of it. Where a model
publishes several variants, choosing the variant is the same kind of decision: quality depends on running
quantization-aware weights in the format they were trained for, and speed on whether the device runs that format
natively -- for example Gemma 4's QAT Q4_0 on an Ada card, an NVFP4 build on Blackwell if one exists. Raised
2026-09-27 in the G2 discussion; not in scope for v0.21.

## Mila republishes weights that producers already publish in their trained format

`Mila/Specifications/ModelDistribution.md` @ `0.21.0-dev+9`

Agreed with Todd 2026-09-27 as a 0.21 direction: recipes rather than gigabyte packages. Where a producer publishes an
official quantized release -- Google's Gemma 4 QAT Q4_0, and (Todd, looked up the same day, unverified here)
NVIDIA's `nvidia/Gemma-4-31B-IT-NVFP4` and `nvidia/Gemma-4-26B-A4B-NVFP4`, calibrated with Model Optimizer, which
is post-training quantization rather than QAT -- Mila installs it (fetch, then transcode into Mila's layout once
at install, the loader untouched) instead of republishing the weights. Mila still publishes a package where it
adds a format no producer ships (Qwen 3.8 at 2.82 bits). Bears on G6: the 26B-A4B may need no Mila package if
NVIDIA's measures well on the G2 harness. **Refined the same day (Todd):** a package also where the producer's only
complete source is full precision -- Gemma 4 QAT's is 22 GiB for the 12B, and its small GGUF lacks the image and
audio weights -- so the Gemma QAT builds ship as Mila packages (`ModelFamilyParity.md` §9, item 14). Google's 26B-A4B
QAT now outranks NVIDIA's NVFP4 build, which is post-training quantization.

## Direction does not say what Mila is at the edge next to llama.cpp

`Mila/Specifications/Direction.md` @ `0.21.0-dev+9`

Faced in discussion 2026-09-27, once recipes meant Mila and llama.cpp could load the same files. Todd: "it's not a
shift, it's where Mila must go to be a useful tool to an engineer or researcher." llama.cpp owns breadth (every
device and format, an app ecosystem); Mila does not race it. What Mila is: a readable, typed, composable library
for harnessing intelligence rather than a runtime you call; training as well as inference; the product family on
top; depth on NVIDIA over breadth; and measurement as a first-class feature -- the G2 protocol could not run on
llama.cpp's own tools, and Mila's harness found a long-context loss in a published build. The latest direction
documents already point this way; the statement is what is missing.

## Google ships the 26B-A4B quantization-aware too, and Mila's routed expert bank has no Q4_0

`Mila/Specifications/ModelFamilyParity.md` 8.2, G5 and G6 @ `0.21.0-dev+10`

Google's QAT announcement (blog.google, "quantization-aware-training-gemma-4", read 2026-09-27) covers E2B, E4B, 12B
and the 26B MoE, in Q4_0 (GGUF for llama.cpp, compressed-tensors for vLLM) and a mobile format. The Hub lists
`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, `-qat-q4_0-unquantized` and `-qat-q4_0-unquantized-assistant` (a QAT
drafter; the 12B has one too). Under §9 item 14 -- trained format where the producer ships one -- the 26B-A4B would
run Q4_0 as well, which puts the Q4_0 policy into the routed expert bank as well as `Linear`, and changes the bytes
G5's 16 GB fit is decided on. The announcement gives no benchmark numbers and says nothing about long context.

## Mila has no desktop app with a window

`Mila/Adaptors/Chat/` @ `0.21.0-dev+11`

Raised by Todd 2026-09-27 while naming the tools (`Future.md`, the `ExportArtifact` entry): a Windows GUI app named
`MilaStudio.exe`, whose core would come from the chat app (`MilaChat.exe` after the rename). Chat's session logic --
model resolution against the store, the deployment request, the turn loop, tool dispatch and the streaming display's
channel routing -- currently lives in console-bound modules (`Chat.ixx`, `Chat.Renderer.ixx`), so a second front end
would share that core only once it is separated from the console. Not in `ROADMAP.md`, `BACKLOG.md` or
`MilaProductFamily.md`.

## A unified model loads text only, whatever its package could do

`Mila/Tools/Converters/Gemma/convert_weights.py` (`SKIPPED_PREFIXES`) @ `0.21.0-dev+11`

Raised by Todd 2026-09-27: for a unified model such as Gemma 4, the deployment should say which of text, vision
and audio to enable, where the VRAM budget allows. Discussed the same day, no decision taken: modality as an axis
of the deployment request beside context length (`Deployment.md`), the package's manifest declaring what it carries
(`ModelHandle.md` section 10 names the manifest as a capability source), each encoder a component built only when
enabled rather than a template flag, and "auto" enabling a modality only while the text context stays above a floor.
Facts found along the way: the Gemma converter drops every modality tensor today (`model.vision_tower.`,
`model.embed_vision.`, `model.vision_embedder.`, `model.audio_tower.`, `model.embed_audio.`), so no Mila package
carries them; the 12B QAT checkpoint's vision stack is a patch embedder only (`patch_dense`, `patch_ln1`, `patch_ln2`,
`pos_embedding`, `pos_norm`), so its cost is image tokens in the context rather than weights, while `audio_tower` is an
encoder; and the Q4_0 12B package was chosen over Google's GGUF partly because the GGUF lacks these weights
(`ModelFamilyParity.md` section 9, item 14).

## An FP4 KV cache would halve FP8's, and no family here was trained for one

`Mila/Specifications/Quantization.md` Part III, "Decisions, 2026-09-28" @ `0.21.0-dev+14`

Raised by Todd 2026-09-28: DeepSeek-V4.1-Flash (arXiv 2609.19969) stores its main KV cache as E2M1 with one E4M3
scale per 16 channels, quantized after RoPE, dequantized before attention, and keeps its sliding-window cache at FP8
for sensitivity. It introduced that format by quantization-aware training during post-training, and publishes no
quality comparison against FP8 or BF16 KV. Llama, Gemma and Qwen were not trained with it, and Gemma's long-context
loss traced to rounding in queries and keys (`ModelFamilyParity.md` 8.2, G2), the scores an FP4 key cache perturbs.
Measurable with the G2/L3 protocol once the FP8 cache exists.

## The KV cache could be compressed below 4 bits by a transform fitted offline

`Mila/Specifications/Quantization.md` Part III, "Decisions, 2026-09-28" @ `0.21.0-dev+14`

Raised by Todd 2026-09-28: "KV Cache Compression Through the Lens of Transform Coding" (Laus, Mayrink Verdun, Wang,
du Pin Calmon and Krahmer, arXiv 2608.14191). Offline, each layer's K and V projections are factored W = A B by an
activation-whitened SVD; the cache holds X A, quantized uniformly with per-channel and per-token scales, bits per
channel allocated by reverse waterfilling on attention-weighted importance (zero-bit channels dropped); K and V are
rebuilt as (X A) B at read time. The first 4 and last 128 tokens stay full precision. Calibration: 256 sequences of
2048 tokens. On Llama 3.1 8B Instruct at about 5.8x (about 2.75 bits): RULER to 32K at baseline (0.778 against
0.775 at 32K), LongBench 0.558 against 0.570, cache 1.07 GB to 184 MB. Baselines FP16 only, nothing past 32K, no
CUDA implementation or speed numbers; how RoPE is applied after reconstruction is not stated in what was read. The
shape of the work is the codebook pipeline's (`Tools/Quantization`, packaged tables) plus attention kernels that
reconstruct and rotate inside flash. The FP8 cache is the baseline it would have to beat on the G2/L3 protocol.

## Gemma's prefill falls behind llama.cpp as the prompt grows

`Mila/Profiling/ProfileModel/ProfileModel.ixx` @ `0.21.0-dev+11`

Measured 2026-09-27 on the RTX 5060 Ti, Gemma 4 12B, prefill only (`ProfileModel --phase prefill`, context 131072;
`llama-bench -fa 1 -ngl 99`, Google's Q4_0 GGUF): at an 8K prompt Mila's FP4 package runs 2,590 tokens/s against
llama.cpp's 2,385; at 32K, 1,389 against 1,960. From 8K to 32K Mila loses 46% of its rate and llama.cpp 18%. Scaling
the 8K time linearly puts about 11 s of Mila's 23.6 s at 32K outside the linear layers, against about 3 s in
llama.cpp -- attention being the likely remainder, which a profile has to confirm. Todd, the same day: Mila cannot be
far behind llama.cpp, and prefill time is what decides whether agentic workloads are usable.

Profiled and fixed the same day for the FP4 build (`GqaFlashAttention.md` 5.7): the global layers' attention was 14.7 s
of the 23.6 s at 32K; the packed kernel takes it to 6.0 s, and the FP4 prefill runs 3,134 tokens/s at 8K and 2,183 at
32K, ahead of llama.cpp at both. What remains behind is the Q4_0 build's staged GEMM (1,296 tokens/s at 32K). The
flash threshold is gone in every family, and Llama has flash (`ModelFamilyParity.md` 8.4, L4).

Llama 3.1 8B, measured 2026-09-28 on the RTX 4070 (12 GB), prefill only, one build: Mila Q4_0 3,030 tokens/s at 8K
and 2,640 at 16K; Mila FP4 4,244 and 3,536; llama.cpp on bit-identical Q4_0 weights (`llama31_8b_instruct_q4_0.gguf`,
`-fa 1`) 4,370 and 3,675. Mila's Q4_0 trails llama.cpp by 1.4x on the same weights and its own FP4 by the same. nsys
of the 8K Q4_0 prefill: the three BF16 GEMM kernels cuBLASLt picks (`cutlass_80 ... 256x128`, `ampere ... 256x128`,
`ampere s1688 ... 128x128 stages_32x1`) 68% of GPU time, `dequantize_int4_to_bf16_kernel` 11%, packed flash 15%,
RoPE 3%. Removing the dequantize alone bounds the gain at about 11%; the gap is the GEMM itself. Options not yet
weighed against each other: INT8 activations per 32 with integer MMA (a change to Q4_0 decision 4 in `Quantization.md`),
a fused BF16 kernel that dequantizes in the tile loader, or steering cuBLASLt off the s1688 kernel.

Weighed 2026-09-28. The BF16 GEMMs already run at the BF16 ceiling: 1.14e14 FLOP of linear layers in about 1.84 s is
~62 TFLOPS on the 4070, where `MmaInstructionPeak` (INT8 arm added) measures BF16 `mma.sync` at 58.9 and
`CublasLtScaleModes` banks 52-60. Neither a fused BF16 kernel nor a different cuBLASLt pick can pass that, so both
are bounded by the 11% dequantize. INT8 `m16n8k32` issues at 232.5 TFLOPS on the 4070 and 207.5 on the 5060 Ti -- 4x
BF16 on both cards, 2x FP8 `mma.sync` -- and one k32 MMA is exactly one Q4_0 block. nsys of `llama-bench -p 8192 -fa 1`
on the same GGUF (4070, per prefill): `mul_mat_q<Q4_0>` 1,360 ms (~84 TFLOPS, 36% of the INT8 ceiling),
`quantize_mmq_q8_1` 62 ms, flash attention 308 ms. llama.cpp is ahead because it multiplies in INT8 with activations
quantized per 32-element block, not because its BF16 is better.

Built the same day (Q4_0 decision 4 changed, `Kernels/Int4/CudaInt4Gemm.cu`): Gemma 4 12B Q4_0 prefill on the 5060 Ti
2,688 tokens/s at 8K and 1,950 at 32K against llama.cpp's 2,385 and 1,960; Llama 3.1 8B on the 4070 4,748 and 3,862 at
8K/16K against 4,370 and 3,675. nsys of one 32K Gemma 4 12B Q4_0 prefill on the RTX 4070, both on the same card (ms per
prefill, Mila / llama.cpp): weight GEMMs with activation quantize 7,214 / 8,499; global attention (HS 512, 8 layers,
`gqa_flash_prefill_packed_bf16_kernel` against `flash_attn_ext_f16<512>`) 4,839 / 3,097, about 29 against 45 TFLOPS;
sliding attention (40 layers) 1,130 / 555; RoPE 578 / ~150 (llama.cpp fuses it into the norm before it); all kernels
14,521 / 13,639. The GEMM now leads, and the global-attention kernel alone carries the gap.
Nsight Compute, one late HS-512 launch (4070): tensor pipe 25.7% of ncu's peak, which on GeForce counts FP16
accumulation (`MmaInstructionPeak`: FP16-accumulate 116.1 TFLOPS, FP32-accumulate 58.9), so about half the usable rate;
8 warps per SM (registers and shared memory each allow one block); top stalls math_pipe_throttle, wait,
short_scoreboard, barrier. At HS 512 a key tile is 16 keys, so each warp's 32 MMAs per tile carry a block barrier,
the split-K score exchange, a 16-key softmax and a 64-register rescale, which the second warp per scheduler cannot hide.
The reported 4.0-way shared-load conflict is likely `ldmatrix.x4`'s four wavefronts (the 1040-byte row stride is
conflict-free for it) -- unverified. The redesign has to amortize the per-tile work: more keys per tile or no
exchange, inside 99 KB and with Q at 512 dims being 128 registers per thread.
Same day, two fixes outside that kernel (4070, 32K, ms per prefill): the sliding layers moved from the FA-2 ring kernel,
which reads each KV head once per query head, to the packed kernel with ring addressing, 1,130 -> 775 (llama.cpp 555);
RoPE's lanes ran along heads, so every access was uncoalesced -- swapped to run along the pair index, 582 -> 63. All
kernels 14,521 -> 13,771 against llama.cpp's 13,639. Ring parity and RoPE tests not yet re-run (MilaTests held by G2).
