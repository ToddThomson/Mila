# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined at the production release tag**, and anything user-reported is a
pointer to its GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## Each device tensor larger than 1 MiB is its own allocation, and its rounding costs up to 7% of a model's weights

`CudaDeviceMemoryResource::do_allocate` (one `cudaMalloc` per tensor), `DeviceAllocation.ixx` @ `0.21.0-dev+32`

Measured 2026-10-03. Rounding each allocation to the 2 MiB granule costs Qwen 3.8 27B 2.82-bit 679 MiB of its
weights, Qwen FP4 497, Gemma 4 12B Q4_0 318, the 26B-A4B 225 (priced by `DISABLED_ParameterRounding_26B_Q4_0`), Llama
64-70. Packing changes nothing else measured: `Profiling/Microbenchmarks/AllocationLayout.cu` reads every layout at the
same bandwidth, quiet and under a second process's pressure, which on WDDM spills the newcomer rather than evicting
the resident. Where it would change a user's limit: the 26B-A4B at 96K, 87 MB over today. A memory manager behind
`CudaDeviceMemoryResource` would take it with no component change, at the cost of rebuilding the planner's exactness
around allocation order; discussed with Todd the same day and judged over-engineering for the bytes alone, worth
revisiting when per-session KV memory needs an owner (`.internal/Ideas/AgentStreams.md`).

## Qwen's FP4 load consumes about 40 MiB more than its planned footprint

`Mila/Specifications/ModelFamilyParity.md` 8.3, Q4 @ `0.21.0-dev+32`

Measured 2026-10-03, RTX 5060 Ti, `ProfileModel --model qwen --quantization fp4`: from the post-initialization
baseline, the load consumed 14,962 MiB at context 8192 against a footprint of 14,920. At 16384 the plan left a
26 MiB margin, and the card read zero free after the load -- the condition the planner's exactness exists to prevent.
Not yet attributed; the cb2-3 build was not checked.

## Compaction summarizes a conversation, and the detail it drops cannot be retrieved afterwards

`BACKLOG.md`, Mila::AI, "A conversation that fills its context stops" @ `0.21.0-dev+32`

Raised 2026-10-03 by Luca (ctx, `github.com/ctxrs/ctx`) about agent harnesses built on Mila: a summary keeps the
conclusion and loses the evidence -- why an approach failed, what a number was measured on. The planned text compaction
summarizes the history into a fresh context and keeps only the instructions verbatim. `Mila::AI` already holds the
conversation it compacts, so the original turns could stay addressable after compaction -- for example as a tool the
model calls to search them. No decision; ctx is cited as the source of the idea.

## FP8 per row loses to six-bit integer blocks on Gemma's head, and Mila's FP8 weight packages use the same format

`Mila/Specifications/Quantization.md` Part II, "The tied table -- six bits per 32" @ `0.21.0-dev+30`

Measured 2026-10-01 for the tied head: INT6 per 32 at 6.5 bits sits 2.8x closer to BF16 by KL than FP8 per row at 8,
and Q8_0 at 8.5 bits 8.6x closer (26B-A4B; 9.5x on the 12B). E4M3 keeps three significand bits whatever the row's
scale. `PerChannelFp8<>` is also the body format of Mila's FP8 packages (Llama 3.2 3B and 3.1 8B FP8); whether an
integer block format at equal or fewer bytes would serve those bodies better, as it does the head, is unmeasured. A
random 512 x 2816 matrix, before the head run: relative RMS error FP8 per row 2.6e-2, INT6 per 32 2.1e-2, Q8_0 5.6e-3.
Gemma's head moved to INT6 per 32 at `0.21.0-dev+31`; the FP8 bodies are what remains.

## A multi-line paste into mila-chat becomes one turn per line

`Mila/Adaptors/Chat/Src/Chat.ixx:242` @ `0.21.0-dev+28`

Raised by Todd 2026-10-01. The chat loop reads input with `std::getline`, so a pasted block arrives as one turn per
line: the model answers each line before the next is read, empty lines are skipped, and a pasted line that starts
with `/` runs as a command. A paste needs to arrive as one message -- and how a user writes a multi-line message by
hand is the same question.

## The FP8 cache's prefill widens every key to BF16 because no tensor-core instruction mixes BF16 and FP8

`Mila/Specifications/Quantization.md` Part III, "The FP8 prefill penalty" @ `0.21.0-dev+28`

Asked by Todd 2026-10-01: why is there an FP8 penalty at all. The attention MMAs are BF16 x BF16 and an MMA's operands
share one format, so every K and V tile widens from E4M3 in shared memory first; prefill is compute-bound, so the
halved read saves little and the widening is added work, repeated by every block sharing a KV head. Part III computes
in BF16 so that storage is the only thing FP8 changes (lossless codes match the BF16 cache bit for bit), and decision
6's evidence was measured that way. The way past it is FP8 attention compute -- Q quantized per row and P to E4M3,
FP8 x FP8 MMAs, which Ada and Blackwell have -- as FlashAttention-3's FP8 mode does ("FlashAttention-3: Fast and
Accurate Attention with Asynchrony and Low-precision", arXiv 2407.08608); SageAttention quantizes Q and K to INT8 for
QK instead ("SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration", arXiv 2410.02367).
Two new roundings with three mantissa bits each, so a change to the accuracy contract that would need its own
decision-6 arm. FP8 MMA throughput on the 4070 and the 5060 Ti is unmeasured:
`Profiling/Microbenchmarks/MmaInstructionPeak.cu` covers INT8 and BF16 only, and is the first measurement.

## The small kernels between Gemma's large ones are 4 to 7% of its time, and fusing them is the only way to recover it

`Mila/Src/Dnn/Components/Transformers/Gemma/` (RMSNorm, residual, GeGLU, activation quantize, split, RoPE) @ `0.21.0-dev+27`

Raised by Todd 2026-10-01: if RoPE is fused into attention, fuse more -- the norms and the rest. Measured the same
day (nsys, kernel time, Q4_0, RTX 5060 Ti, a 32,512-token prompt then 256 tokens at that depth; 26B-A4B with the FP8
global cache at chunk 1024, 12B with BF16 caches at chunk 1024). Every kernel that is neither a weight GEMM nor
attention's main kernel, summed -- the ceiling on what fusing them could save, since fused work still runs:

| | 26B-A4B | 12B |
|---|---|---|
| Prefill | 422 ms of 8.28 s, 5.1% | 995 ms of 14.14 s, 7.0% |
| Decode | 0.64 ms of 9.0 ms a token, 7.1% | 0.75 ms of 20.5 ms a token, 3.7% |

The largest parts: in decode, RMSNorm, launched about 330 times a token at about 1.4 us each (5.0% of the 26B-A4B's
decode, 2.2% of the 12B's), a cost that is mostly fixed per launch; in prefill, the 12B's GeGLU (2.7%), whose BF16
output the down projection's INT8 quantize (1.1%) reads straight back, so GeGLU -> quantize and RMSNorm -> quantize
are the natural pairs. RoPE is 0.3 to 0.7% everywhere (`RopeInAttention.md` 4.5). Larger levers the same captures
show, none of them fusion: the HS-512 global prefill attention kernel (38.5% of the 26B-A4B's prefill, 28.2% of the
12B's), the tied FP8 vocabulary head in decode (1.73 ms a token, 19% of the 26B-A4B's), and the 26B-A4B's expert
gathers (25% of its decode). Method: ProfileModel under `nsys profile --cuda-graph-trace=node`, decode read as the
kernels after the last prefill kernel.

## A user cannot measure Mila's rates on their own card

`Mila/Profiling/ProfileModel/` @ `0.21.0-dev+27`

Raised by Todd 2026-10-01: promote ProfileModel to `mila-bench`. ProfileModel is developer tooling and does not ship:
it takes weight paths (by default a BF16 checkpoint quantized on load, `MODELS_DIR` compiled in), and carries nsys
capture ranges, NVTX, a VRAM high-water sampler and load-throughput reporting. Discussed the same day, no decision:
a separate shipped binary beside `mila-chat`, not a rename -- `mila-bench <model>` naming a store model, planned
through `planDeployment`, prefill and generation rates at a depth, `--kv-cache`, a machine-readable output -- with
ProfileModel kept as the profiler and the timing core (salted prefill, decode timed first token to last) shared. A
binary rather than a `mila` verb by the August rule (`mila` keeps install/models/serve; a verb earns its place by the
impedance it hides). It would let a reader reproduce the published comparison rows. A user surface: end-user prose,
the wheel and the image, tests.

## Llama 3.1 8B forgets a system instruction once a book fills 64K of the context

`Mila/Tests/Dnn/Models/LlaMa/Llama.InstructionRetention.Cuda.cpp` @ `0.21.0-dev+27`

Found 2026-10-01 running decision 6's behavioral arm (`Quantization.md`, Part III), Q4_0 weights, RTX 5060 Ti. With
"Begin every reply with the word BANANA" or "reply in exactly three words, all in capital letters" as the system turn
and a PG-19 book in the user turn, the instruction holds at 2048 and 16384 and is gone at 65536 with the BF16 cache
as with the FP8 one: all twelve replies on two books open "The main character of this book". At 130048, FP8 cache, the
same. Both Gemma 4 models keep BANANA to 64K on the same prompts. L3 scores the whole book better than 1024 tokens of
it out to 131072, so the model does read far context; what is lost is the instruction at position 0. **It is the
model's, measured 2026-10-01:** HuggingFace with the BF16 weights loses all six at 65536 the same way and agrees with
Mila on 11 of 12 verdicts at 16384 (`Quantization.md` Part III, decision 6's behavioral arm;
`Tools/Converters/Llama/hf_llama_instruction_retention.py`). So nothing in Mila to fix; what is left is what a Llama
user is told, since the planner may give Llama 3.1 8B up to 131072 tokens.

## RoPE is a separate pass, so a Gemma global layer caches its rotated keys and its values as two tensors

`CudaRopeOp` (`Rope.Rotation.cuh`), `GemmaBlock` global layers (K = V checkpoints) @ `0.21.0-dev+26`

Todd's RoPE policy had three forms: table, calculated, fused into the attention kernels. Calculated replaced the table
(`MemoryFootprint.md` 8.4); fused is not built. On Gemma's global layers K = RoPE(k_norm(x)) and V = v_norm(x), so the
cache holds two tensors where the checkpoint has one projection. Rotating keys inside the flash and decode attention
kernels -- from the same angle function, `Rope.Angle.cuh` -- would leave the cache unrotated keys, the precondition for
storing that tensor once. Both deciding measurements passed on 2026-10-01 -- one stored tensor serves both, and
rotating 64 pairs on read costs 0.69 of today's two-tensor read at 64K -- and the planner puts the 26B-A4B at 128K with
about 251 MB spare once keys are stored once, against 80K today (2026-10-03). Its quality gate, the kernels' shape and
whether it is admitted to v0.21 are open: `Specifications/RopeInAttention.md`.

## A masked key's zero probability still multiplies whatever an unwritten cache row holds

`Gqa.Flash.Packed.cu` (`p * v_scale` before PV; the PV MMA), the clamped and ring rows of every GQA cache @ `0.21.0-dev+24`

Flash prefill loads whole key tiles. Keys past the last written position -- the clamped tail of an unbounded cache,
the not-yet-written rows of a ring -- read rows no write has reached, and device memory is not zeroed
(`TensorBuffer`). Their scores are masked to -inf after scaling, so their probability is exactly 0, but that 0
then multiplies the row's V values in the PV MMA, and in the FP8 caches the row's V scale too. A row holding a NaN or
infinity -- an E4M3 NaN code, a stale non-finite scale -- makes 0 x NaN = NaN in every output that tile touches.
Nothing has been seen to fail; whether fresh or recycled device memory can hold such patterns is not measured.

## Gemma 4 26B-A4B does not fit a 16 GB card at context 32768, and llama.cpp runs it there with an FP16 cache

`GemmaModel::load` (`FixedContextDoesNotFit`), `benchmark_comparison.py` row `gemma-4-26b-a4b-q4_0` @ `0.21.0-dev+24`

On the RTX 5060 Ti at Q4_0 with an FP8 cache, context 32768 needs 16,075,330,560 bytes against 15,904,800,768 free
(weights 14,770,483,712), so both 32K cells of the comparison row are refused; at 33792 (32K depth plus the
generated tokens) 16,113,079,296. llama.cpp on Google's GGUF prefills 32K at 3,321 tokens a second and generates at
depth 32K at 99, with an FP16 cache. Its weights are 13.43 GiB (14.42 GB): a Q6_K tied table where Mila's is FP8,
worth about 0.13 GB of the 0.35 GB weights difference. The FP4 build was refused the same way
(`Gemma4MoE.md`, Rates Baseline, 16,178,589,696 bytes).

Ceilings measured 2026-09-30 on the same card (llama.cpp b11216 by generation at depth, a spill into host memory
read as the rate collapsing): llama.cpp with an FP16 cache at `-ub 512` runs at 72K (77 tokens a second at 64K, 68K
and 72K); at `-ub 1024` it spills past 64K; with q8_0 it spills at 96K (44) and 128K (26). Mila at Q4_0 loads at
24576 only by narrowing the prefill chunk to 128 rows (state 1,030 MiB, against 530 at 8192 and a 1024-row chunk),
and refuses 28672 (15,955,530,752 bytes). About 26K against 72K.

Read from the code, same day, arithmetic not yet measured component by component:

- **No Gemma layer has an FP8 cache.** The sliding layers use `SlidingWindowKvCache`, an uncompressed ring; the
  global layers are hardwired `NoKvCompression` (`GemmaModel.ixx`, `GemmaSlidingKvPolicy`). The preset's FP8
  setting is ignored. "Mila at Q4_0 with an FP8 cache" above is wrong: BF16.
- **The global layers keep K and V both**, though `attention_k_eq_v` makes them equal: 5 layers x 2 x 2 heads x 512
  x 2 bytes = 20.5 KB a token, the term that grows with context. FP8 with K alone would be about 5 KB.
- **The sliding ring** is window + chunk - 1 rows, BF16: 419 MB at a 1024-row chunk, 236 MB at 128, whatever the
  context.
- **The prefill score buffers** (`preatt`, `att`) are chunk x heads x (window + chunk) and stay allocated when every
  layer prefills through flash and never reads them (`Gemma.ixx`, `prefillScoreWidth`): about 134 MB at a 1024-row
  chunk if BF16.
- Weights are 0.35 GB over llama.cpp's, 0.13 GB of it the FP8 tied table against Q6_K. At `0.21.0-dev+31` the
  table is six bits per 32, the same 0.56 GiB as Q6_K.

Per token, about 23 KB (global K and V, the RoPE tables, decode scores). The 37.7 MB difference between the refusals
at 32768 and 33792 overstates it: each of those tensors rounds up to the 2 MiB granularity, and a 1K step crosses a
boundary on most of them.

Measured on the prediction the same day, and two terms removed (`MemoryFootprint.md` 8.3, working tree): state grew
29.2 KB a token, 5.6 KB of it the token embedding's output sized to the whole context; that and Gemma's unused
prefill score buffers are gone, and 32768 fits at a 128-row chunk. The FP8 global cache passed its quality arm and a
caller can ask for it (`withKvCacheCompression( FP8 )`); the FP8 ring failed on the 26B-A4B and is not offered
(`Quantization.md` decision 6). RoPE no longer holds tables (`MemoryFootprint.md` 8.4): with the FP8 global cache,
65536 fits at a 128-row chunk with 93 MiB spare.

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
2026-09-29, `+19` tree, the 4070 pinned alone by UUID: the reading is bimodal, 262.6 or 10.0 MiB, and never between.
Under `CUDA_MODULE_LOADING=LAZY` (the default) six fresh processes read 262.6 five times; under `EAGER` every run reads
10.0, with about 450 MiB less free before generation. So the ~253 MiB step is the driver loading kernels lazily inside
the measured window, not a Mila allocation; it reproduced with either decode attention kernel. History: the test
measured on the display-free 5060 Ti from `rc.1+15` and on the current device (the 4070 when both cards are visible)
from `rc.1+28`, when NVML left the tests. Not measured: the suite with both cards visible, Todd's validation
condition; why identical runs land in different modes; why the 5060 Ti never shows the step. The ~256 MiB
driver block is inferred from the readings, not found in NVIDIA's documentation.

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
against exact FP64, and it sets `Gemma.LogLikelihood.Cuda.cpp`'s window bound back from 2e-3 to 1e-3. At
`0.21.0-dev+31` Gemma's head left this path: its table is INT6 per 32, whose batched GEMM scales each group's exact
dot product. The FP8 Linears of Llama's FP8 packages still take it.

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
Producers now ship quantization-aware checkpoints, each fitted to one grid -- Gemma 4 QAT to Q4_0,
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
**Verified 2026-09-29 from the GGUF's tensor table:** experts, dense branch and attention Q4_0; router, scales and
norms F32; tied table Q6_K; instruct only. The bank is the same bytes as `PerGroupFp4<64>`, so the fit is unchanged.
Taken into the specs the same day (`Gemma.md` §10.4, `Gemma4MoE.md` Phase 9, `ModelFamilyParity.md` G5).

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
`quantize_mmq_q8_1` 62 ms, flash attention 308 ms. The gap is the multiply's arithmetic, not BF16 GEMM efficiency: an
INT8 path with activations quantized per 32-element block is what closes it.

Built the same day (Q4_0 decision 4 changed, `Kernels/Int4/CudaInt4Gemm.cu`): Gemma 4 12B Q4_0 prefill on the 5060 Ti
2,688 tokens/s at 8K and 1,950 at 32K against llama.cpp's 2,385 and 1,960; Llama 3.1 8B on the 4070 4,748 and 3,862 at
8K/16K against 4,370 and 3,675. nsys of one 32K Gemma 4 12B Q4_0 prefill on the RTX 4070, both on the same card (ms per
prefill, Mila / llama.cpp): weight GEMMs with activation quantize 7,214 / 8,499; global attention (HS 512, 8 layers,
`gqa_flash_prefill_packed_bf16_kernel` against `flash_attn_ext_f16<512>`) 4,839 / 3,097, about 29 against 45 TFLOPS;
sliding attention (40 layers) 1,130 / 555; RoPE 578 / ~150; all kernels
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
kernels 14,521 -> 13,771 against llama.cpp's 13,639.
Next day, the HS-512 redesign (`Gqa.Flash.WideHead.cu`, `GqaFlashAttention.md` 5.8: keys on the MMA's M dimension, eight
query rows per warp over the whole head, no exchange): global attention 4,773 -> 3,348 against 3,097; all kernels
13,771 -> 12,422 against 13,639. 8% remains on global attention; accumulating PV in FP16 (twice the FP32 rate on
GeForce) would be a precision trade, unmeasured.

## Gemma printed an image-closing token in the middle of a text reply

No location: nothing masks modality tokens out of sampling today.

Todd, Chat on Gemma 4 12B, 2026-09-28, the evening `+17` was committed: asked for puns, the reply ran "Why did
the<image|>Animals:" -- the rest of that pun lost, the reply continuing coherently after it. The model sampled
`<image|>` (258882) mid-sentence in a text-only conversation. Gemma 4 has seven modality markers (`<|image>`
255999, `<|audio>` 256000, `<|image|>` 258880, `<|audio|>` 258881, `<image|>` 258882, `<audio|>` 258883,
`<|video|>` 258884); Chat now hides all seven, where it printed five of them and stored them in the history.

Not the HS-512 prefill kernel that landed the same day: on `+17`, 257 targets of PG-19 book 30312 scored by
decode (which that kernel does not run) and by prefill agree as closely as before it -- 3.559 vs 3.583 nats/token
after 4096 tokens, 4.108 vs 4.140 after 32768, against gaps of 0.042 and 0.029 before `+17`
(`DISABLED_DecodeAgainstPrefillAlongTheBook`). What remains is sampling at temperature landing in the tail, and
whether a text-only session should mask the modality markers out of sampling -- a sampler change, open.

## A model under a Windows path with characters outside the ANSI code page cannot be opened

`Mila/Src/Dnn/Serialization/WeightsReader.ixx:189`, `SafeTensors.ixx:184`

Both open with `std::fopen( filepath.string().c_str(), ... )`. On Windows `path::string()` converts to the ANSI code
page, so a path it cannot represent fails to open. The store lives under `%LOCALAPPDATA%`, so a user whose profile
name has such characters cannot load any model. Found 2026-09-29 while clearing the `fopen` deprecation warning;
not reproduced. A fix opens by `path::c_str()` (wide on Windows) without an `#ifdef` in a module.

## A Debug CPU-only MSVC build cannot compile the weights metadata module

`Mila/Src/Dnn/Serialization/WeightsMetadata.ixx` (`import nlohmann.json;`) @ `0.21.0-dev+19`

`out/build/x64-claude-cpuonly` (MSVC 14.51.36231, `CMAKE_BUILD_TYPE=Debug`, `MILA_ENABLE_CUDA=OFF`) fails on this module
with `json.hpp(20512): error C2678: binary '!=': no operator found which takes a left-hand operand of type 'nullptr'`
-- a `unique_ptr != nullptr` comparison that the Release CUDA builds of the same file compile. Neither file changed
in `+19`. Not isolated: whether Debug or CUDA-off is the variable, and whether it is the MSVC module/header
interaction `Vnext.md` already records, was not tested. A Release build of the same directory compiles it and passes
1266 tests. Found 2026-09-29 verifying DecodeGraph on the CPU-only configuration.

## The decode position is device memory the footprint does not predict

`Mila/Src/Dnn/Compute/Devices/Cuda/CudaExecutionContext.ixx` (`setDecodePosition`) @ `0.21.0-dev+19`

DecodeGraph Phase A: the context `cudaMalloc`s one `int` on its first decode step, outside `reserveScratch` and outside
every `getMemoryStats` a plan is priced from. Four bytes requested; what the driver reserves for it was not measured,
and a small `cudaMalloc` can cost a whole allocation granule. Every other device allocation a load makes is predicted
(`Deployment.md` section 9), and the scratch reservation throws on growth for that reason. Found 2026-09-29 auditing
the change for limitations met along the way.

## Two compile-time A/B toggles in CudaLinearOp keep branches no build takes

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/CudaLinearOp.ixx` (`kUseW8A16Gemm`, `kUseFusedFp4Gemm`,
`kUseFp8ActivationPrefill`)

`kUseW8A16Gemm` and `kUseFusedFp4Gemm` are `false` and `kUseFp8ActivationPrefill` is `true`, so `cuda_w8a16_gemm`,
the fused FP4 GEMMs (`cuda_fp4a16_gemm`, `cuda_fp4a16_gemm_wmma`) and `use_wmma_fp4_gemm_` are reached by no build.
Found 2026-09-29 while splitting `forward()` into one method per path, which kept them (`runCublasLtPrefill`,
`runFusedFp4Prefill`) so that change stayed a restructure. Retiring the toggles means deciding whether the fused
kernels keep a measured purpose.

## CUTLASS's fix for SM120 block-scaled MMA was closed unmerged

`Mila/Specifications/MixtureOfExperts.md` section 7.2(c) @ `0.21.0-dev+20`

Checked 2026-09-29 for the Gemma 4 26B-A4B grouped expert GEMM on Blackwell: CUTLASS 4.8.0 is the latest release
(2026-09-22); its notes list grouped and grouped block-scaled GEMM improvements (B collector reuse, SMEM-staged TMA
descriptor updates, less prologue) and NVFP4 with a UE5M3 scale type, under Blackwell headings whose reach to the
GeForce SM120 path was not confirmed. PR #3082 (`is_family_of()` for the SM12x guard in `MmaSM120BlockScaledOp`) is
closed and not merged. The spec does not name it (corrected 2026-09-29: this entry first said it did); whether
section 7.2(c)'s `120f` route builds on 4.8.0 without it is untried in Mila, and is now recorded in section 7.3. CUTLASS's block-scaled grouped GEMM serves NVFP4 and MXFP4,
not Q4_0, which bears on the expert-format question in the 26B-A4B QAT entry above: Q4_0 experts would take Mila's
own INT8 GEMM made grouped, NVFP4 experts CUTLASS. Todd, 2026-09-29: Gemma 4 MoE on Blackwell with current CUTLASS
may be a better use of time than more decode-attention work.

## A published post promises a tool-calling design that was never built

`Web/content/blog/mis-with-claude-code-and-codex.md` "What's Coming: Tool Calling" @ `0.21.0-dev+20`

The post (2026-05-14) announces a pybind11 `ToolCallParser` and a `MILA_TOOL_CALLING_ENABLED` flag, and points at
`specifications/toolcalling.md`. Neither was built: the server parses Llama's calls in `tool_bridge.py` and Gemma's and
Qwen's through the library's grammar bindings, and `ToolCalling.md` was archived in the spec index 2026-09-29 as
superseded. Dated and forward-looking, so accurate about the plan it announced; a reader following it today finds
neither the flag nor the design, and not the tool calling that shipped.

## Llama and Qwen rebuild the library's gated feed-forward inside their blocks

`Mila/Src/Dnn/Components/Transformers/LlaMa/Llama.Block.ixx:27`, `Qwen/Qwen.AttentionBlock.ixx:244` @ `0.21.0-dev+24`

Raised by Todd 2026-09-30 ("we've broken our symmetry"). `Components/FFN/GatedMLP` is a fused `fc_gate_up`, a gate
activation and `fc_down`, with any gate, any weight policy, backward, and one `installSharedOutputs` for pooling.
Gemma uses it inside its two feed-forward sublayers (`GemmaDenseFeedForward`, `GemmaRoutedFeedForward`). Llama and Qwen
compose the same three children inline -- `Llama.Block.ixx:27` says "no MLP composite" -- each with its own slot
installation and footprint code, so it exists three times. Moving them onto `GatedMLP<Silu>` as a child `mlp` renames
their flat tensors (`fc_gate_up` to `mlp.fc_gate_up`), so every Llama and Qwen weights file and published package would
be reconverted, as Gemma's were at G4. Two smaller asymmetries found the same day: `Router` and `MixtureOfExperts` are
feed-forward functions but sit in `Components/MixtureOfExperts/` beside `FFN/` rather than in it; and the Gemma sublayers
report their child's `ComponentType` (`GatedMlp` for the dense one, `MixtureOfExperts` for the routed one, the same as
its bank). The distinction discussed, not decided: `FFN/` holds feed-forward functions, a family directory holds the
sublayer -- the norms around the function, which differ per family.
