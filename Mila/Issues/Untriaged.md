# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined after ninety days**, and anything user-reported is a pointer to its
GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## Hermes Agent refuses every Qwen 3.8 27B build but the 2.82-bit one on a 16 GB card

`Mila/Apps/Server/docs/harnesses.md` *Tested With* @ `f610285`

Found 2026-10-18 starting the Hermes task set on Qwen 3.8 27B FP4: Hermes 0.22.2 refuses a model whose reported
window is under 64,000 tokens (`MINIMUM_CONTEXT_LENGTH` in its `agent/model_metadata.py`, enforced in
`agent/agent_init.py`), and all eight tasks ended at start-up. FP4 at 65,792 with the FP8 KV cache is refused on the
5060 Ti (needs 15.65 GB, 14.81 GB free); 32,768 fits. The 26B-A4B holds 27,648 with the BF16 cache and 65,536 with
the FP8 one (`ContextProfile.md`), not yet run under Hermes. So the guide's +69 "yes" for Hermes on both was wrong: Hermes's own log for that run shows every request going out as `gemma-4-12b-it` with a 65,536-token
`context_length` pin from the WSL `~/.hermes/config.yaml`, set 2026-07-06 to clear this check. Rows corrected the
same day; Gemma 4 12B re-run with no pin (full 262,144 window) and passed. Consequence for evaluation: there is no FP4
arm for the Hermes task set on a 16 GB card.

## Mila's website, README and marketing do not yet say what Mila is

`Web/`, `README.md`, model cards, blog @ `f610285`

Raised 2026-10-19 (Todd): "This message needs to be published to the Mila website in the next 2 releases ... It is
the whole Mila positioning message written for the end user. We've refined what Mila is over the last number of
months and I feel it now consistent and achievable by 0.22. The website, README and other marketing messages need
to catch up." The message as refined: a C++ library for compiling modern transformer architecture recipes; only a
few models represent the best of local AI -- Gemma 4 and Qwen 3.8, then Qwen 4 -- and Mila serves those few deeply
rather than many thinly; each published model carries a small set of measured builds; agentic use is quality-first on
a 16 GB card with 64K context as the floor, pushed up along a measured quality curve; developers building on Mila can
choose their own build; the 0.22 harness serves a model, a card and a workload with the best build the measurements
support. Timing is Todd's (v0.21.0 and v0.22.0); what each release carries is for triage, and copy claims only what
the release it ships in can do. Copy follows the end-user prose rules and the site's marketing voice. Todd, same day:
"Each model card should have our quality measurements and the reasoning for our final model recipe" -- the
measurements answer "is it any good", the recipe and what it holds on 16 GB "will it run", the reasoning under its
own heading below the first screen, stated as measured fact; `.internal/Marketing/Positioning.md` is revised first,
the other marketing docs from it.

## The deployment planner cannot choose a model's bit allocation for the context asked of it

`Mila/Specifications/Deployment.md`, `Mila/Src/Deployment/` @ `f610285`

Raised 2026-10-19 (Todd), with the agentic formula: 64K is the floor, the context is pushed up along a measured quality
curve, and the bit-allocation recipe is picked from it -- "it would be incredible if this could be dynamically chosen
with the planner". Today a precision plan is a compile-time type and a package carries one allocation; the planner
chooses context, chunk and KV format for it. Notes at capture: FP4 and FP8 roles quantize at load, so recipes that
differ only there need no second package; codebook roles are fitted offline, so a recipe that differs there does;
`dispatchKvCacheCompression` is the existing runtime-to-compile-time bridge such a choice would follow. Todd, same
day: "This also ties directly into the vnext and overall Mila direction of a model harness for AI" -- a harness that
serves a model, a card and a workload with the best recipe the measurements support. Not decided.

## Qwen 3.8 27B's embedding table is stored at 16 bits where a 2-bit copy serves

`Mila/Src/Dnn/Components/Transformers/Qwen/Qwen.PrecisionPlan.ixx` `EmbeddingTable` @ `f610285`

Found 2026-10-18 reading both files: our 2.82-bit build keeps `temb.wte` (248,320 x 5,120) in BF16, 2.54 GB of its
11.87 GB; Unsloth's Q2_K_XL stores the same table as Q2_K, 0.42 GB of 9.83 GB, and still passed the Hermes short set
8/8 twice. Both keep it in host memory -- ours pinned, llama.cpp's `CPU_Mapped` (its load log: q2_K "cannot be used
with preferred buffer type CUDA_Host, using CPU instead") -- so the saving is 2.1 GB of download, disk and pinned host
RAM, not VRAM: on the device the builds are 8,894 MiB (ours, file less the table) against 8,631 MiB. The model reads
one row of the table per token. The same plan keeps the head at FP4 (4.25 bits) against Q4_K's 4.5. Not measured:
what a 2- or 4-bit table costs Qwen's quality, and whether the other families' tables are BF16 for a reason.

## The inference server answers a request for any model name

`Mila/Apps/Server/src/mila_llm_server/` @ `f610285`

Found 2026-10-18: nothing compares a request's `model` with the model loaded, so a harness configured for one model is
served by another without a word. It is how the false Hermes pin above went unnoticed -- requests named
`gemma-4-12b-it` were answered by Qwen 3.8 27B FP4 and Gemma 4 26B-A4B. Not checked: what OpenAI and Anthropic return
for an unknown model (OpenAI documents `model_not_found`), and whether a harness sends a name MIS could match.

## Hermes Agent's session-title request is refused for its response format

`Mila/Apps/Server/src/mila_llm_server/routes/factory.py` @ `f610285`

Found 2026-10-18 in the Hermes task-set runs: once per task Hermes 0.22.2 sends a Chat Completions request with
`response_format` `json_schema` (name `session_title`, `reasoning_effort` `none`), and MIS refuses it because it has no
constrained decoding. The task itself goes on; not checked what Hermes does with the session after the refusal.

## Model evaluations are not published, and a user cannot rerun them

`Mila/Tools/Evaluation/`, `Web/`, `Mila/Specifications/ModelEval.md` @ `2d6f61e`

Raised 2026-10-18 (Todd), asked as "mila-evals as a release artifact for 0.21?": publish the results on the website,
and give end users a tool that reproduces them. Today the evaluations live in `Tools/Evaluation`, which never ships,
and the results only in specs and `Data/Evaluation`. Costs noted at the time: a shipped tool is a new app under
`Mila/Apps/` with a `mila-` command, help text and tests; the lm-eval, bfcl-eval and inspect-ai arms bring heavy
dependencies and the hf and llama.cpp arms need transformers or a llama.cpp build, while the Hermes task set is
standard library only and talks only to MIS. Not decided: which release, which suites the tool carries, and whether
v0.21.0's ROADMAP has a criterion it serves.

## An agent system around one main model with worker agents across several cards has no design

`Mila/Specifications/LayerSplit.md`, `Mila/AI/` @ `4804dbc`

Raised 2026-10-18 (Todd): a system built around a main model that directs worker agents, each on its own card, in a
multi-card machine -- the counterpart to trading time for quality with one large model on a 16 GB card. Related: the
layer split across devices (`LayerSplit.md`, after v0.21.0), `Mila::AI`'s agent core, NVIDIA Dynamo's KV-as-agent-
resource framing (memory reference). Not designed: which model directs and which work, what crosses between cards,
and how a worker's context is managed.

## Library doc comments still describe the retired operation registry and the removed operation bases

`Mila/Src/Dnn/Components/Connections/Residual.ixx:8`, `Normalization/Softmax.ixx:44`, `Encodings/Lpe/Lpe.ixx:53`, `Losses/SoftmaxCrossEntropy.ixx:11` @ `4804dbc`

Found 2026-10-18 reviewing `CLAUDE.md`. `Mila.ixx:108` records the runtime `OperationRegistry` as retired and no
`UnaryOperation`, `BinaryOperation` or `PairedOperation` class exists, but those components' Doxygen still says they
obtain a backend from the registry or derive from those bases (Residual, Softmax, Lpe, SoftmaxCrossEntropy, Gelu and
others name them; a grep counts 28 mentions of the bases). Public Doxygen describing code that is not there.

## Nobody can tell a user when a smaller model beats a larger one quantized harder on their card

`Mila/Specifications/ModelEval.md` section 12 @ `4804dbc`

Found 2026-10-18 (Todd's question): for a given use case and card, when is a smaller model of a family the better
choice than a larger one at aggressive quantization, Gemma 4 26B-A4B against 12B for one? Captured as ModelEval's
choice matrix: fit at the use case's minimum context first (the planner's answer), then quality at that context at
the model's defaults, then rate by bytes read per token. Only one card answers it today, for one pair (Qwen 3.8 at
12 and 16 GB). Not measured: any smaller model as an agent at 64K, or a smaller model at higher precision against a
larger one at low bits on the same tasks. Open: the use cases, their thresholds, the candidate set.

## No 26B-or-larger model at FP4 or better reaches the 64K agentic minimum on a 16 GB card

`Mila/Specifications/TieredKvCache.md` @ `4804dbc`

Found 2026-10-18 (Todd). Qwen 3.8 27B FP4 holds 32,768 with an FP8 cache at a 512-row chunk and is about 1 GiB short
of 64K; Gemma 4 26B-A4B is not priced at 64K. Todd's proposal, captured as a preliminary spec: one KV pool over the
device and pinned host memory, device first, the oldest positions in the host tier, prefetched under the layers
between attention layers. The weight arena (497 MiB measured on Qwen FP4) and FP8 group scales would shrink the host
tier to about 140 MiB by arithmetic. Not measured: the bus rate on the 5060 Ti, the arena on either model, Gemma at
64K, the decode rate Todd will accept.

## A model installed before capabilities were recorded reads to clients as having no reasoning and no tools

`Mila/Apps/Server/src/mila_llm_server/protocols/openai/models.py:32` @ `4804dbc`

Found 2026-10-18 serving Qwen3.8-27B-fp4 from Todd's store: `/v1/models` says `supports_reasoning: false` and
`capabilities.tools: false`, for a model that reasons by default and calls tools. The record predates the
capability fields, and an absent field reads as false rather than unknown. The BACKLOG entry "What a model can do is
inferred from its family name, not read from its manifest" republishes every model with the fields; until then
every older install reports the wrong answer.

## The evaluation tools assumed greedy decoding and short replies

`Mila/Tools/Evaluation/reference_server.py:138`, `:149`; `run_bfcl.py` (until 2026-10-18) @ `4804dbc`

Found 2026-10-18 moving Phase 8 to the model's defaults. `run_bfcl.py` sent temperature 0 and capped a reply at
4,096 tokens; it now sends no sampling knob (MIS applies the model's recommended setting), asks for 32,768, and
greedy is `--greedy`. `reference_server.py`, the `hf` arm for BFCL, refuses any temperature above 0 and defaults a
reply to 256 tokens: right for Phase 1's engine parity on one set of weights, but it cannot measure a model at its
recommended setting. lm-eval's served arm sends the task's own `gen_kwargs` (greedy for IFEval, GSM8K and RULER), so
`run_arm.py` measures greedy too. Results recorded before this with thinking off: `Data/Evaluation/qwen38-*-thinking-off*`.

## A deployment request cannot fix the prefill chunk; the planner always chooses it

`Mila/Src/Deployment/DeploymentRequest.ixx`, `Mila/Specifications/Deployment.md` section 5 @ `4804dbc`

Found 2026-10-18 setting up Qwen 3.8 FP4 at 32,768 for the Phase 8 sample. Deployment.md decides "The chunk is not a
request knob: the planner always chooses it", so a caller fixes the context and takes whatever chunk fits;
there is no way to ask for a chunk, or a floor under one (the agentic target is at least 64K at a chunk of at least
512, `Qwen3.8.md`). Todd, the same day: the planner needs to work around provided settings -- a value the caller
gives is a constraint the plan honours, as the fixed context already is. Not checked: what the request, Chat, the
server and the binding would each carry.

## A greedy reply through the server can depend on the request before it

`Mila/Bindings/Mila_py.Wrappers.cpp` (reuse off only with a seed), `Mila/Specifications/PromptCaching.md` section 5
@ `2d6f61e`

Found 2026-10-18 fixing the seed: KV prefix reuse rounds a prompt differently from a whole prefill (Llama 3.2 3B
BF16 departs at the tenth sampled token), and a seeded call now prefills whole. An unseeded greedy request still
reuses, so at a near-tie its reply depends on what the server ran before -- the case an engine comparison at greedy
through MIS is in. Not checked: how often greedy flips on real prompts, and whether greedy calls should also
prefill whole.

## A 64-bit sampling seed keeps only its low 32 bits

`Mila/Src/Dnn/Samplers/TokenSampler.ixx` `reseed` @ `2d6f61e`

Found 2026-10-18 reading the reseed for the MIS seed. `reseed( std::uint64_t )` casts to `std::mt19937::result_type`
and `mt19937::seed` keeps 32 bits, so seeds 2^32 apart -- and OpenAI's -1 and 4294967295 -- give the same stream
while the API says each is a different seed. A `std::seed_seq` of both halves would keep all 64.

## GPT-2 samples on the host with its own sampler, which ignores top-p

`Mila/Src/Dnn/Models/Gpt/GptModel.ixx` `onGenerating`, `sampleFromLogits` @ `52045ab`

Found 2026-10-17 adding the presence penalty to the device sampler. `GptModel` copies the logits to the host and
samples with temperature and top-k only; `params.sampling.top_p` is never read, so a top-p request runs the full
distribution. It now refuses a presence penalty rather than drop it. Llama, Gemma and Qwen sample on the device
through `TokenSampler`.

## Claude Code assumes a 200,000-token window for a model served by Mila

`Mila/Apps/Server/docs/harnesses.md` *Context Window* @ `+70`

Found 2026-10-15 running Claude Code 2.1.300 against MIS on Qwen 3.8 27B FP4 at 32,768: its result reports
`"contextWindow":200000` for `Qwen3.8-27B-fp4` and logs `unrecognized_model`. It does not read the `context_window`
`/v1/models` returns, so it would not compact before MIS refuses a request for length.
Read from the 2.1.300 binary the same day: a model whose name does not start `claude-` takes its window from
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` (default 200,000), and compaction fires at window - min(max output tokens, 20,000)
- 13,000; `CLAUDE_CODE_AUTO_COMPACT_WINDOW` is clamped to the window. Set to 32,768 it compacts on every turn --
measured: request 20,490 tokens, summary 21,946, resume 20,152, summary 21,661, ~70 s each, until killed. Claude
Code's own request is ~20.5K, so a session with working room needs a window of about 33.5K + the output reserve
(`CLAUDE_CODE_MAX_OUTPUT_TOKENS`) + the room. Not checked: how Codex and Hermes learn their window.

## Text parts of one message are joined with nothing between them

`Mila/Apps/Server/src/mila_llm_server/conversation.py` `text_of` @ `+68`

Found 2026-10-15 capturing the harnesses. Claude Code 2.1.300's `system` field is three text blocks, and the first
(`x-anthropic-billing-header: ...;`) is joined straight onto the second ("You are a Claude agent..."); Codex
0.162.1 sends its skills and permissions text as two parts of one `developer` message, joined the same way. Not
checked: how Anthropic and OpenAI join a message's parts, or whether the billing line reaches Anthropic's models.

## The inference server's Chat Completions endpoint returns a tool call as reply text

`Mila/Apps/Server/src/mila_llm_server/protocols/openai/chat.py` `format_chat_response` @ `bb27731`

Found 2026-10-13 serving Qwen3.8-27B-fp4 through the inference server. `/v1/chat/completions` puts the request's
`tools` into the prompt, and the model answers with a well-formed call, but the reply carries it as raw
`<tool_call>` text in `message.content` with `finish_reason: stop` -- no `tool_calls`, and no
`finish_reason: tool_calls`. The adapter has no tool path at all, for any family: nothing parses the reply, nor
renders an assistant message's `tool_calls` or a `tool` role message from the history. `/v1/responses` and
`/v1/messages` parse, resume and stream the same call correctly. Not checked: streaming, and which clients reach
the server through Chat Completions.

## A consumer build warns that CudaTensorOps.Random's cudaMalloc is local to another translation unit

`Mila/Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Random.ixx:266` @ `2e3e44a`

Found 2026-10-13 building QuickStart inside the devel image with Clang 22: four
`-Wreference-tu-local-entity-in-other-tu` warnings, from `fill_uniform` and `fill_normal` instantiating
`cudaMalloc<float>`, the CUDA runtime header's template, which the module's global fragment declares. Not checked:
whether Clang 21 warned too, and whether it is only a warning or a real ODR hazard.

## Gelu and Residual zero a gradient buffer the CUDA kernels overwrite

`Mila/Src/Dnn/Components/Activations/Gelu/Gelu.ixx` and `Connections/Residual.ixx`, `backward` @ `4db0309`

Found 2026-10-11 retiring the CPU operations. Both components `zero()` their input gradient before calling the
operation, for the CPU operations that accumulated into it. The CUDA kernels assign every element in range
(`Gelu.Fp32.cu`, `Residual.Fp32.cu` / `.Bf16.cu`), so the zeroing is a memset per backward that nothing reads.
Not checked: whether any caller reads past the range a narrower-than-built input writes, which the zeroing keeps
at zero today.

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

## Qwen 4 27B is expected around November 2026 and Mila has no Qwen 4 chassis

`Mila/Specifications/Qwen4.md` @ 539396e

Compared Qwen3.8-Flash-Next, the Qwen 4 architecture preview, against the Qwen 3.8 chassis and wrote the
spec. Its section 9 phases 0 to 2 (the tiny reference, the converter skeleton, grouped RmsNorm, dilated
CausalConv1d, the DeltaNet gate activation, and the gated residual, n-gram, PLE and QSA indexer components)
need no 27B checkpoint. 2026-10-08: Phases 0 and 1 are built and their CPU gates passed; Phase 2 has
`GatedResidual` and `NgramEmbedding` on CPU, gated against the tiny reference.
