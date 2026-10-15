# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined after ninety days**, and anything user-reported is a pointer to its
GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

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

## The Gemma 4 26B-A4B leaves Claude Code about 7,000 tokens of working room on a 16 GB card

`Mila/Apps/Server/docs/harnesses.md` *Tested With* @ `+68`

Found 2026-10-15 in the harness rerun. Claude Code 2.1.300's first request renders to about 20,400 tokens and the
26B-A4B loads on the 5060 Ti at 27,648. Not measured: the 12B's room, or whether the 26B-A4B's context is held back
for a full prefill chunk as Qwen's is (Qwen moved to `BACKLOG.md`, its KV-cache entry).

## Text parts of one message are joined with nothing between them

`Mila/Apps/Server/src/mila_llm_server/conversation.py` `text_of` @ `+68`

Found 2026-10-15 capturing the harnesses. Claude Code 2.1.300's `system` field is three text blocks, and the first
(`x-anthropic-billing-header: ...;`) is joined straight onto the second ("You are a Claude agent..."); Codex
0.162.1 sends its skills and permissions text as two parts of one `developer` message, joined the same way. Not
checked: how Anthropic and OpenAI join a message's parts, or whether the billing line reaches Anthropic's models.

## Qwen's prompt renderer differs from Qwen 3.8's own chat template in six places

`Mila/Src/Dnn/Models/Qwen/Qwen.Protocol.ixx` `formatPrompt`, `formatTurn`, `serializeToolSignatures` @ `ec6aca9`

Found 2026-10-14 recording the template's goldens (`Tools/ChatTemplates`). The template's default effort is
`xhigh`, where the free functions call `Medium` the checkpoint's default; it groups consecutive tool results into one
user turn; it separates a turn's second and later calls with `\n`; it writes tool signatures and non-string
arguments through `tojson` (`", "`/`": "`, keys in the order given) where the free functions write compact, sorted
JSON; it trims every message; and with `preserve_thinking` on, its default, it renders past reasoning back into
history. `QwenChatProtocol` renders all six as the template does; Chat and the server still call the free
functions. Not measured: how far each moves the model's behaviour.

## Gemma's applications distill a JSON tool result that Google's template renders as text

`Mila/Src/Dnn/Models/Gemma/Gemma.Protocol.ixx` `formatToolResponse` @ `ec6aca9`

Found 2026-10-14 recording the template's goldens. The template renders a result given as text under `value:`,
whole; `formatToolResponse` parses a JSON envelope and keeps one output field (added for Codex's envelope, whose
chunk id the model echoed). `GemmaChatProtocol` renders it as the template does, so the distillation becomes the
server's wire work, for the one client whose envelope needs it, when the server switches. In the same template, a
calling turn's text renders after its results and the turn closes with no prompt after it, so a resume whose
assistant turn carries both text and calls -- Claude Code sends that shape -- gets no open model turn.

## Chat's Llama tool-call parser reads a key Meta's template does not ask for

`Mila/Apps/Chat/Src/Chat.ToolCallParser.ixx:142` `parseJson` @ `ec6aca9`

Found 2026-10-14 folding Llama's grammar into the library. Meta's 3.1 and 3.2 templates ask for
`{"name": ..., "parameters": ...}` and replay a call that way; Chat's parser accepts only `"arguments"`, so a call the
model writes as the template asks is not a call to Chat. Not checked: what Chat's own system prompt asks the model to
write, and which key Llama 3.x emits under it. Two template facts a Llama consumer meets on switching: a plain-text
tool result is written through `tojson`, so the model sees it JSON-quoted, and an assistant turn that made a call
renders the call alone, its text dropped.

## Gemma's drafter generation test fails on the RTX 4070 and passes on the 5060 Ti

`Mila/Tests/Dnn/Models/Gemma/Gemma.DraftModel.Cuda.cpp:367` `Gemma4_12B_GreedyGenerationIsUnchanged` @ `ec6aca9`

Found 2026-10-14 running the full suite at `+68` (a change that touches no model code). Pinned by UUID, the same build
passes on the 5060 Ti and fails on the 4070: drafted and plain greedy tokens part within the first 256, where the
test's comment says the reply first parts at token 402, measured 2026-10-05. Not checked: on which card that was
measured, and the token at which they part on the 4070.

## Stopping a Gemma turn at its first closed call keeps only one of several calls

`Mila/Apps/Server/src/mila_llm_server/model_worker.py:56` `_stop_markers_for` @ `ec6aca9`

Found 2026-10-14 writing `ProtocolMarkers`. Generation stops at `<tool_call|>`, since left running the model repeats
the call to the budget (`routes/factory.py` `_generate_buffered`), so a turn that would have made two calls makes
one. `InferenceServerWire.md` W2 asks for both. Not checked: whether the model, unstopped, writes a second call or
always repeats the first.

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
