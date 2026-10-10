# Mila — Backlog

**Work committed to the release in flight, and nothing else.** Narrative and success criteria are in
[ROADMAP.md](ROADMAP.md); everything upstream of the commitment is in
[`Mila/Issues/`](Mila/Issues/README.md). Completed work is in the git history.

**Admission:** name the ROADMAP success criterion that fails if this never ships. If you cannot,
it belongs in `Mila/Issues/`.

Each `###` bucket is a theme of the release in flight, its name matching the ROADMAP section — the
only join. **This file describes a publish (a minor); a patch tag has no backlog.**

**Entry shape**, one level deeper than the same shape in `Mila/Issues/`:

```markdown
#### Llama throws away its long-context scaling factor

`open` · `llama`

The load path reads `rope_scaling` from the model metadata and discards it — the
`.withRoPEScalingFactor()` call at `Llama.ixx:703` is commented out, for a reason recorded as
unclear. 3.1 8B cannot reach the context length it advertises.
```

The **heading states the problem** and has to read cold: no term that exists only inside Mila.
The **metadata line** carries status — `open`, `in progress`, `done` — then area tags from
[Tags.md](Mila/Issues/Tags.md); status never appears in the prose. The **body** carries whatever
detail the work needs and ends in an anchor, unless the finding is an absence, in which case say so
rather than inventing a location.

**The gate is the entry count, and it only goes down.** A release in flight burns down, so an
addition is paired with a removal, or it is a deliberate admission that scope grew. **What bounds
this file is the release it describes and the date that release carries, not a fixed number.**
v0.21.0 opened against 2026-10-31 and was widened on 2026-09-23, before its work started, to take in
`Direction.md` section 5, re-dated to **2026-12-15** — a deliberate admission that scope grew, made
once at the start of the cycle. Whatever has not landed by then goes back to
[`Vnext.md`](Mila/Issues/Vnext.md) and the release ships narrowed, rather than the date moving to
accommodate the list. A backlog that grows while the date holds is the signal to drain it early.

Scope grew a second time on 2026-09-29: the Gemma 4 26B-A4B's bar rose from a published package to parity
with every other model Mila publishes, admitting three entries under Gemma 4 Complete with no removal. It
is scope the parity survey found rather than scope invented, and it is held to the same date.

Scope grew a third time on 2026-10-03: installed models now survive an upgrade
(`ModelDistribution.md` *Compatibility*), admitting two entries under Model Handle with no removal,
held to the same date. Later the same day (Todd): agentic quality measured for every published model in every
family, admitting one entry under Mila::AI, and the 2.82-bit Qwen build's prefill and decode efficiency,
admitting two under Qwen 3.8 Complete -- each with a success criterion written first, none with a removal,
all held to the same date.

Scope was re-weighted on 2026-10-04 (Todd), the date held: on the 16 GB card the main effort is Gemma 4,
and Qwen's own tuning waits for a 24 or 32 GB card. Admitted, each against a criterion written first:
deployment by use case and the allocation lever under Deployment Planning, use-case presets under
Mila::AI, and under Gemma 4 Complete the 26B-A4B's vision tower, keys stored once, the sliding ring
kept across a rewind, and the drafter's loop where its measurement says it pays. Removed to `Vnext.md`:
four Qwen 3.8 entries -- the 2.82-bit build's prefill and decode efficiency, its 16K perplexity gate and
the head-width note -- with the FP4 build's long-context loss beside them. Capabilities stay shared across
the families; how deeply each is optimized follows the card it is for. If the date presses, the 26B's
vision tower drains first, then audio.

Scope grew on 2026-10-05 (Todd), the date held: the token sampler's synchronous entry moves onto the model's
stream, admitted under Internal fixes with no removal.

Scope grew again on 2026-10-05 (Todd: "Chat defaults must be model based"), the date held: a model's recommended
sampling is read from its manifest, admitted under Model Handle with no removal.

Scope grew on 2026-10-06 (Todd), the date held: each Gemma 4 package carries its drafter, and the planner and Chat
turn it on by default where it is measured to pay, written into the Deployment Planning feature-selection entry with
no removal.

Scope grew on 2026-10-07 at the triage of `Mila/Issues/Untriaged.md` (Todd), the date held: under Mila::AI, a program
that skips initialization fails as if it had no device; under Deployment Planning, the device allocations no plan
prices; and under Internal fixes, the FP8 batched path's early scale, model paths outside the Windows code page, and
the deployment type names. None with a removal.

Scope grew on 2026-10-10 at the MIS triage (Todd), the date held: under Internal fixes, MIS's finish reasons and
streamed stop sequences. No removal.

Scope grew on 2026-10-10 (Todd), the date held: under Applications, the inference server published to PyPI beside
the binding, with a criterion clause added to that theme. No removal.

**Split on 2026-10-10 (Todd): the release was re-dated to 2026-11-20 and drained.** `Direction.md` section 5 — the
model handle, the deployment request, `Mila::AI`, and the applications rebuilt on it — became v0.22.0, and its 15 entries
went to `Vnext.md` with its narrative in ROADMAP's Future tail. What stayed is the families finished and measured: Gemma 4,
Qwen 3.8 and a new Llama 3 theme, each with its evaluation and context profile in its criteria; the applications' work
that survives the rebuild (Applications Ship); the two upgrade entries; the llama.cpp comparison; and Internal fixes,
which took the two allocation entries from Deployment Planning. Gemma's drafter stays as its own entry, the package and
the default, with choosing it in a request left to v0.22.0. Admitted with the split: Llama's grammar in the library, and
benchmark evaluation of every published model. The 12B's image and audio are committed; the 26B-A4B's tower is
tentative and drains first.

Scope grew on 2026-10-10 (Todd), the date held: under Internal fixes, the move to CUDA 13.4.2 for the images,
the wheels and CI. No removal.

Scope grew again on 2026-10-10 (Todd), the date held: under Internal fixes, the release script finished through
publishing. No removal.

**Done means deleted**, in the same commit as the work — `done` is a working-tree marker and is
never committed.

---

## Current release (v0.21.0)

The families finished and measured, in the themes ROADMAP gives them. `Direction.md` section 5 — the
handle, the deployment request, `Mila::AI` and the applications rebuilt on it — is v0.22.0, and its items
wait in [`Vnext.md`](Mila/Issues/Vnext.md).

### Gemma 4 Complete

#### Gemma 4 12B is multimodal and Mila drops its image and audio weights

`in progress` · `gemma` · `mila-src` · `adaptors`

The 12B is **encoder-free** (`Gemma4UnifiedForConditionalGeneration`): no vision tower and no audio
encoder. An image is resized and cut into 48-pixel patches, at most 280 soft tokens, each through a
small embedder (about 50M parameters) into the decoder; an audio clip is 40 ms frames of raw 16 kHz
samples, each through one projection. `convert_weights.py` skips both. The image tokens of one picture
attend each other in both directions, and every Mila attention path is causal: that mask change, in every
prefill path, is the largest piece. Work, in `Gemma4Modality.md` section 7's order: the mask; the image
embedder, soft-token placement, a prompt that carries media, and prefix reuse keyed on the media rather
than its token ids; audio; Chat accepting an image or a clip (`/image`, `/audio`, Todd 2026-10-04), the
inference server's content blocks in both protocols, the binding; converter, manifest and footprint.
Image and audio reach applications through `Mila::AI`, not around it. Each is a feature a deployment
selects (Deployment Planning).

A deployment without a modality should not sample its markers. On 2026-09-28 the 12B sampled `<image|>` (258882)
mid-sentence in a text-only Chat session, losing the rest of a pun; Chat now hides all seven of Gemma 4's modality
markers from display, but nothing masks them out of sampling, so the model can still spend a token on one.

Gate: embedder parity against HuggingFace, then token for token on an image prompt and on an audio
prompt; Chat answers about an attached image and an attached clip.

`ROADMAP.md`, Gemma 4 Complete success criteria · `Mila/Specifications/Gemma4Modality.md`

#### The 26B-A4B's vision tower is dropped when it loads

`open` · `gemma` · `mila-src`

The 26B-A4B checkpoint carries a vision tower -- 27 layers, 1,152 wide, about 411 million parameters,
about 820 MB at BF16 and 410 at FP8 -- and the loader skips its 355 tensors (`Gemma4MoE.md`). The image
path the 12B's work builds (soft-token placement, templates, the application surface) carries over; the
tower is the new piece. On the 16 GB card its bytes are about 39K tokens of the 26B's context, so it is
selected by a deployment rather than always built, and the memory levers are what make it worth
selecting. First to drain if the date presses. Gate: tower parity against HuggingFace, then an image
prompt token-for-token, at a context the plan states.

`ROADMAP.md`, Gemma 4 Complete success criteria · `Mila/Specifications/Gemma4Modality.md` section 4

#### Gemma caches each global layer's keys and values separately, though its checkpoint makes them equal

`open` · `gemma` · `mila-src` · `measured`

On Gemma 4's global layers K = RoPE(k_norm(x)) and V = v_norm(x) from one projection, and the cache holds
both. Storing one tensor and rotating the keys inside the attention kernels -- from the same angle
function, `Rope.Angle.cuh` -- returns about half the global cache. Both deciding measurements passed on
2026-10-01: one stored tensor serves both, and rotating 64 pairs on read costs 0.69 of today's two-tensor
read at 64K (`RopeInAttention.md`). The context profile then gave it its reason (2026-10-04): the
26B-A4B is reliable to 96K on recall and loses the middle of a conversation at 128K, while it fits 64K
today -- so the bytes buy the 96K it is good for, or its vision tower at a useful context, rather than
128K. Applies to the 12B's global layers as well. The kernels' shape is open.

Gate: G2's books and recall at depth equal to storing both; the 26B-A4B plans its measured reliable
depth on the 16 GB card.

`ROADMAP.md`, Gemma 4 Complete success criteria · `Mila/Specifications/RopeInAttention.md`

#### A Gemma reply longer than about 1,024 tokens costs the next turn a full prefill

`open` · `gemma` · `mila-src` · `measured`

Gemma's sliding layers keep a ring of `window + prefill_chunk - 1` rows (2,047 at chunk 1024,
`SlidingWindowKvCache.md` D2). Writing more than about 1,024 tokens past a position overwrites the window
its continuation attends to, so a rewind to it is refused -- correctly -- and the caller prefills from 0:
`GemmaModel` reuses a prefix only when the rewind is accepted. Thinking-off replies never reach it;
reasoning replies run to thousands of tokens (ContextProfile's thinking runs, 2026-10-04), and a full
prefill is about 100 s at 128K on the 12B. The window is the architecture's; the finite rewind horizon is
the ring's, a memory optimization. Work: what Qwen does for its recurrent state -- `savePosition` copies
the sliding layers' last window, and a rewind to the saved position puts it back -- bounded by the window,
not the context. Size unmeasured.

Gate: a turn after a reply of any length resumes without a full prefill, measured on the 12B at 128K;
tokens equal to a fresh prefill's.

`ROADMAP.md`, Gemma 4 Complete success criteria

#### Announce the Gemma 4 26B-A4B mixture of experts

`open` · `gemma` · `distribution`

Landed in `Mila/Src` during rc.1 — the router and expert bank on CPU and CUDA, wired into
`GemmaModel` with FP4 and Q4_0 expert banks and a streaming converter, gated against HuggingFace at BF16,
FP4 and Q4_0. It appears in no README capability row, no CLAUDE.md target and no release note, because
no published package uses it, so a user cannot run it.

Held out of v0.20.0 (Todd, 2026-09-21). A package alone would have published it below every other model Mila
ships; its kernels now run at llama.cpp's rates or above (`ModelFamilyParity.md` §8.2, G5b), and it waits on the
rest of `Gemma.md` §10.7's bar: quality
across the planner's range, by the method `ModelFamilyParity.md` §9 item 18 settles; decode replay gated
equal to the called path on a routed network; active parameter bytes in the plan and in Chat's display;
the model run through Chat, the inference server and the Python binding, tool calls included. Then the
package, its card and the capability row. A first publish, so it is not held for the 12B's republish.

`Mila/Specifications/ModelFamilyParity.md` §8.2, G6

#### Gemma loses its own reasoning between tool calls in a turn

`open` · `gemma` · `adaptors`

Google's multi-turn rule is to strip thoughts from *prior* turns and keep the current turn's.
`extractAnswer` (`Gemma.Protocol.ixx:1288`) removes every channel span from a response rather than a
leading run, so a model working through a multi-step tool sequence starts each step without the
reasoning that led to it. Moved out of v0.20 at `rc.1+24` (Todd): a behaviour change inside Gemma's
protocol is too late in the cycle.

#### A malformed Gemma tool call parses as a call with no arguments instead of failing

`open` · `gemma` · `mila-src`

`parseArguments` (`Gemma.Protocol.ixx:480`) breaks out of its loop at the first key not followed by
`:` and returns what it has accumulated, so a partial parse is indistinguishable from a call that
genuinely took no arguments. Seen driving Codex through MIS: the model emitted
`call:exec_command{cmd="cat line_count.txt"}` — `=` and plain quotes, off the trained grammar — and
`gemma_parse_tool_call` returned `{'name': 'exec_command', 'arguments': '{}'}`. Codex rejected the
empty call and the model retried correctly, so that flow recovered; a client that executes `{}`
would not. Qwen's bridge treats a malformed call as prose, which is the behaviour to match.

Held for the same reason as the entry above (`rc.1+24`): a behaviour change inside Gemma's protocol
is too late in this cycle. `Chat.ToolCallParser.ixx`'s over-eager `[` test — "Any response
containing a bracket enters the tool-call parser" in `Vnext.md` — is the same failure shape in the
application rather than the library.

#### `gemma_protocol.py` is dead and can be deleted

`open` · `gemma` · `binding`

Its 856 lines are superseded by `Gemma.Protocol.ixx` plus `gemma_bridge.py`, nothing imports it, and
it carries a header saying so. Kept on disk under the retire-don't-delete rule, which is the correct
state for now; removing it is a one-file deletion whenever the reconciled grammar has been driven
long enough to be sure.

#### Nobody knows whether Google's drafters would make Gemma 4 decode faster

`in progress` · `gemma` · `perf` · `measured`

Every Gemma 4 size ships a dedicated draft model for speculative decoding (ai.google.dev/gemma/docs/core,
read 2026-09-17), and the QAT targets ship QAT drafters (`google/gemma-4-12B-it-qat-q4_0-unquantized-assistant`),
so the pair measured is that drafter and the Q4_0 12B. The drafter is four layers and its own head, about
0.85 GB at BF16, and keeps no cache: it attends the 12B's last sliding and last global layer's caches.
Three measurements predict the speedup: what a drafter step costs, what a verify of K + 1 tokens costs
against one decode on the 5060 Ti, and how many drafts the 12B accepts. Acceptance is measured in Mila,
with the drafter's forward built and gated first, because the BF16 12B does not fit the card that
HuggingFace's loop would need. If too few drafts are accepted, or a verify costs too much, there is no
win, and the recorded result is "not worth doing".

The loop is in this release where the measurement says it pays (Todd, 2026-10-04), as a feature a
deployment selects: draft, verify, accept, rewind. Sampled as well as greedy, since Chat samples by
default. The rewind is a few tokens and the sliding ring already allows it. The verify is a decode of K + 1
tokens with one tensor-core product per Linear, which reads the weights once for all of them
(`Gemma4Mtp.md` 4.7). Gate: with the drafter selected, the 12B's verify logits are within its decode's own
numerical noise -- token-for-token equality holds only if decode moves to the same kernels, 4.7's option
(a) -- and it decodes faster by the measured amount; if the measurement says "not worth doing", that result
is recorded and the loop stays out. The 26B-A4B gets its own drafter by the same measurement and gate (Todd,
2026-10-04): Google says a mixture of experts gains little at batch 1, and a community report measured 1.46x
on it, so it is measured on the 16 GB card, FP8 global cache included.

`ROADMAP.md`, Gemma 4 Complete success criteria · `Mila/Specifications/Gemma4Mtp.md`

#### Gemma 4 12B at 4 bits predicts worse the longer the context, and Mila cannot run the weights Google trained to prevent it

`in progress` · `gemma` · `quantization` · `mila-src` · `measured`

Measured on one PG-19 book inside a model turn: Mila's FP4 package costs +0.19 nats per token over BF16
in the first 8K, rising to +0.69 by 32K and still rising through 262144. Google's quantization-aware
weights rounded to Q4_0 (32-element groups, FP16 scale) cost +0.03 through 32K, and Google's own Q4_0
GGUF in llama.cpp agrees within 0.007. The same weights in Mila's FP4 cost +0.24, so the benefit holds
only in the format they were trained for. The measurement's question is answered by per-token loss
against BF16, not the KL divergence first planned; the gap is large enough that KL would not change it.

Work: a `PerGroupInt4<32>` symmetric weight policy -- OperationTraits rows, decode matvec and prefill
GEMM, footprint at 4.5 bits per weight -- selected through `LanguageModelConfig` and exposed in Chat's
quantization modes and the binding. The prefill quantizes its activations to INT8 with one scale per 32
values and multiplies on the INT8 tensor cores, since a BF16 prefill cannot keep pace with llama.cpp
(`Quantization.md`, Q4_0 decision 4); the long-context curve gates that rounding. Mila's FP4 prefill
rounds activations to FP8 with one scale per token, which the QAT weights never trained for. The source is
Google's BF16 checkpoint `google/gemma-4-12B-it-qat-q4_0-unquantized`, read by the existing Gemma converter
and rounded by the Q4_0 reference rule -- the rule Google's own GGUF was made with, bit for bit -- in the
same code that serves `ExportArtifact` and quantize-on-load. The tied
embedding goes to six bits per 32 from BF16 (`Quantization.md` Part II, "The tied table"). Delivered as a Mila package, not rounded at install: Google's only complete
source is 22 GiB, and its 6.5 GiB GGUF lacks the image and audio weights (`ModelFamilyParity.md` §9, item 14).
Ships in G4's single republish of the 12B.

Gate: every Q4_0 tensor bit-identical to `google/gemma-4-12B-it-qat-q4_0-gguf`; then G2 re-run in Mila on the
Q4_0 build, cost over BF16 by band, on a second book as well as 30312.

`Mila/Specifications/ModelFamilyParity.md` §9, item 14 · `Mila/Specifications/Quantization.md`, "Q4_0"

#### Gemma's quality above 131072 tokens has never been measured, and the planner may choose up to 262144

`in progress` · `gemma`

Gemma 4 12B's weights declare 262144 (`Gemma.md` §2). Chat caps it at 131072
(`Chat.FamilyTraits.ixx:60`), but since `0.21.0-dev+7` the binding and the inference server plan
against the weights, so on a card with room the same model can open at different contexts depending
on which application loaded it: the binding opens it at 262144 on the RTX 5060 Ti, and Chat at 131072.

Loss by context length on the RTX 5060 Ti, from 8K to 262144, over at least two PG-19 test books long
enough to fill the whole window (`ModelFamilyParity.md` 8.2, G2, which holds the protocol and the gate).
Measured on the published FP4 package, the loss rises from 16K; the gate's run is on the Q4_0 package
(the entry above). The gate is three tests: the whole book predicts no worse than the sliding window alone
at every length, the cost over BF16 stays within 0.05 nats per token to 32K, and Mila agrees with
llama.cpp on Google's GGUF to 262144. The result decides `ModelHandle.md` 10.3: if all three hold, Chat's
cap is deleted; if not, the ceiling comes down to the last band where they hold, in the library, not in an
application.

#### Chat renders Gemma's prompt itself, and its template and the library's have drifted apart

`open` · `gemma` · `mila-src` · `adaptors`

`formatGemmaPrompt` (`Chat.ixx:1359`) renders thinking on — the `<|think|>` trigger in the system
turn — and the library's `Gemma::formatPrompt` (`Gemma.Protocol.ixx:1132`), which the inference
server uses, cannot: it always primes the empty thought channel. The library's has the tool
declarations and `continue_open` that Chat's lacks. Each is missing half of the other.

One template, in the library: `Gemma::formatPrompt` gains thinking on, proven byte-identical to Chat's
output on recorded fixtures (thinking on and off, with and without tools, multi-turn) before Chat calls
it — the `+34` fold's recipe. Chat's effort sentences stay Chat's text: Gemma's `<|think|>` has no
trained budget, so they are a prompt, not grammar (`ModelHandle.md` 3.4). The Gemma half of
`ModelHandle.md` Phase 2.

#### Gemma 4 packages do not carry Google's drafters, so no load can use them

`open` · `gemma` · `distribution` · `mila-src`

Decided 2026-10-06 (Todd): every Gemma 4 package carries its own drafter, so one install
gets both. A load gets it with the measured K wherever its speedup is
measured and the planned context still fits with it loaded -- on the 16 GB card the 26B-A4B's drafter is context
it gives up. Chat takes that default from the deployment, as it takes the model's sampling, with a setting to turn
it off. The drafter's format is measured before a package ships it: BF16 as Google ships it, against its head at the
12B's six-bit table format (`Gemma4Mtp.md` decision 2).

Gate: each Gemma 4 package exported on this tree carries its drafter as a manifest role; a load decodes with it
token-for-token as without it, at the measured speedup, wherever the planned context fits with it resident; and
Chat's setting turns it off.

`ROADMAP.md`, Gemma 4 Complete success criteria · `Mila/Specifications/Gemma4Mtp.md` 4.7

### Qwen 3.8 Complete

#### Publish the 2.82-bit Qwen 3.8 27B so a 12 GB card can run a 27B model

`open` · `distribution` · `quantization` · `qwen`

The package exists and is validated; nobody outside can fetch it.
`https://huggingface.co/api/models/mila-llm/Qwen3.8-27B-cb2-3` answers 401 anonymously, where
every other published model answers 200. It is 11.1 GiB against 15.1 GiB for the FP4 build of the
same model, which is what would let a 12 GB card run a 27B model at all — the FP4 build needs 16 GB.

Held out of v0.20.0 deliberately (Todd, 2026-09-21): a capability with no published package is not
announced, the same rule that kept the Gemma 4 MoE work unannounced. The cost is that
**`ROADMAP.md`'s v0.20 headline claim was written around this model** and had to be narrowed to the
FP4 build at 16 GB. Publishing it is what makes the stronger claim sayable, so it wants to land
early in the cycle rather than at the end. The fitted source is gitignored and not reproducible
from the repo, so no reader can work around its absence.

#### Qwen's KV cache is uncompressed on a card its weights already fill

`in progress` · `qwen` · `quantization` · `mila-src`

`PerTokenKvFp8<>` is built in the shared attention operation, gated on Llama and Gemma, and reached by
their loads on request (`dispatchKvCacheCompression`); Qwen refuses it (`QwenModel::validateDeployment`)
and its layers do not use it. On a 27B at FP4 the weights take the card, so the KV cache is what
buys context back, and halving it is the difference between the context length that fits and the one
the model is sold at.

The freed margin is then a decision rather than a windfall — more context, or more bits where the
quality gate says they are worth most. v0.20 deliberately did not pre-empt that, so it is part of
this work rather than a consequence of it. The compressed cache has to be priced exactly by
`getRequiredMemory`, since the planner reads that price.

#### Qwen answers in one block after a long silence

`open` · `qwen` · `adaptors`

`FamilyTraits::streaming_capable` is false for Qwen (`Chat.FamilyTraits.ixx`), because the harness
routes tokens by Gemma's four control-token ids and nothing else has them. Qwen has one marker pair,
`<think>`/`</think>`, which is enough to separate reasoning from answer; the per-token router has
simply not been written for it.

Not a gap against any other model — Llama and GPT-2 are buffered too, and Gemma is the only family
that streams. It matters most on Qwen because a 27B is the longest wait to sit through with nothing
on screen.

### Llama 3 Measured

#### Llama's prompt and tool grammar live in Chat and the inference server, not the library

`open` · `llama` · `mila-src` · `adaptors`

Gemma's and Qwen's templates and tool grammars are the library's, and both applications render through them;
Llama's are written out twice instead — in Chat (`Chat.MessageFormatter.ixx`, `Chat.SystemPrompt.ixx`) and in the
server (`prompt.py` `_build_llama_prompt`). So the library cannot render a Llama tool conversation, and Llama's
context profile cannot ask its tool-call question.

The server's copy differs from Meta's template: a Chat
Completions request with no system message gets "You are a helpful assistant." as its system turn, where
Meta's writes a "Cutting Knowledge Date / Today Date" header. `Tools/Evaluation` measures through token ids
and does not see it; a score taken through `/v1/chat/completions` — every Hub benchmark score
(`ModelEval.md` section 11.2) — carries it.

Gate: the library renders Meta's template exactly, against its own Jinja rendering, for plain, system and tool
turns; Chat and the server render through it; and a Llama tool call parses through the library as Gemma's and
Qwen's do.

`Mila/Adaptors/Inference/Server/src/mila_llm_server/prompt.py` `_build_llama_prompt` · `Mila/Adaptors/Chat/Src/Chat.MessageFormatter.ixx`

#### Nobody can say whether a published model scores like its original on standard benchmarks

`open` · `models` · `measured`

What a card says about quality today is fidelity alone — perplexity ratios and KL against a reference build --
which a reader cannot act on. `Tools/Evaluation` runs a model on public benchmarks (IFEval, GSM8K, BFCL, RULER,
MMLU-Pro) on Mila, HuggingFace and llama.cpp from the same token ids, and compares them document by document
(`ModelEval.md`). It ran on CUDA for the first time at `+55`, as a 20-document smoke run.

Work, in `ModelEval.md`'s order: Phase 1, Mila BF16 against HuggingFace BF16 on Llama 3.2 3B, which sets the
noise floor; Phase 2, each quantized build of that model against the same reference; then every published model
of every family (Phase 6), which serves the Gemma 4 and Qwen 3.8 criteria as well as Llama's. Every model but the
Llama 3B has BF16 weights larger than the 16 GB card, so where their reference runs is decision 1, owed before
their arms run. The card's table and its sentence follow section 8.

Gate: Phase 1's — every paired interval contains zero and every task's median divergence is past the first
sentence; then each published build's scores beside the original's, its flips read against that floor.

`Mila/Specifications/ModelEval.md` · `Mila/Tools/Evaluation/README.md`

#### Nobody knows how deep into a conversation each model still finds a fact, obeys its instructions and calls a tool correctly

`in progress` · `models` · `ai` · `gemma` · `llama` · `qwen` · `gate`

The quality evidence Mila has is perplexity by context band (`ModelFamilyParity.md` 3.4, 8.2 G2) and
two single-family instruction-retention tests (`Llama.InstructionRetention.Cuda.cpp`,
`Gemma.InstructionRetention.Cuda.cpp`). None of it says where an agent fails, and no family is measured
across the contexts its planner can choose. Admitted 2026-10-03 for every family in the parity matrix
(Todd); the design is `ContextProfile.md`, a tool under `Mila/Tools/ContextProfile`.

In scope: Phase 1 (fit, loss by band, recall at depth) and Phase 2 (instruction retention, tool-call
fidelity), for every model the release publishes -- Llama 3.2 3B and 3.1 8B, Gemma 4 12B and 26B-A4B
in Q4_0, Qwen 3.8 27B at 2.82 bits -- at every band each fits on the 16 GB card, against section 9's decisions and
thresholds, from 16K (Todd, 2026-10-03; section 3). Added 2026-10-04 (Todd): thinking as a configuration
axis, profiled off and on (section 4.3), and a recovery arm beside Phase 2's -- whether an agent with the
tool declared ends up right -- since recall alone measures one unaided lookup.

Every question is asked from the end of one conversation, which the pricing shows is the difference
between about 3 hours for all six models and about 73 (`ContextProfile.md` section 9). Three library
changes make that possible, agreed with Todd the same day, all three in `+33`: Llama's transformer
rewinds and continues a prefill as Gemma's does, and `LlamaModel` reuses a matching prefix; Qwen returns
to a saved position (`savePosition`), and `QwenModel` resumes each turn from the end of the previous
prompt; and `sequenceLogLikelihoodFrom` scores after a cached prefix on all three families. Building them
exposed a flash prefill defect -- a masked key's zero probability multiplied unwritten cache rows, NaN on
fresh memory -- fixed in the same change. The recall arm's answer score forces the value through decode
steps rather than differencing two of those scores (`ContextProfile.md` 4.3).

Llama's tool-call arm needs its grammar in `Mila/Src`, which "Llama's prompt and tool grammar live in Chat
and the inference server, not the library" moves. Turn cost and the llama.cpp column (Phases 3 and
4) are not required by the criterion. Storing Gemma's global keys once (K = V, `RopeInAttention.md`)
waited on the 26B-A4B's profile, to return for discussion only if its reliable depth ran past the 80K that
fits today; the profile puts it at 96K and not 128K (`ContextProfile.md` section 8), so it returns, framed
as reaching 96K rather than 128K.

Gate: section 8's Phase 1 gate before any profile is recorded, then one profile per model, and the
reliable depth each reports is the one compaction reads.

Llama 3.1 8B loses a system instruction by 65536 in BF16 and FP8 cache alike, and HuggingFace on the BF16 weights
loses it the same way (`Quantization.md` Part III, decision 6's behavioral arm), while its planner may choose up to
131072. It is the model's, so its profile is where a Llama user is told it.

`ROADMAP.md`, Mila::AI success criteria · `Mila/Specifications/ContextProfile.md`

### Applications Ship

#### The inference server's README says `pip install mila-llm-server`, and no such package is published

`in progress` · `adaptors` · `ci` · `distribution`

RELEASING publishes the `mila-llm` wheels and nothing else: no step builds or uploads MIS, and it reaches users only
from a checkout or inside the runtime image (`Docker/build-mis.sh`). Its version already derives from `Version.txt`,
so what is missing is the release path itself:

- MIS requires the runtime release it was built with, not `mila-llm>=0.20.0b2` — it now calls `version()` and
  `cuda_devices()`, which an older runtime lacks.
- One `py3-none-any` file from `python -m build`, rehearsed on TestPyPI at release step 1 and uploaded beside the
  wheels at step 8, with a row in RELEASING's onboarding table.
- A clean-room check: `pip install mila-llm-server` from the index into an empty environment, then `mila-server`
  starts and refuses cleanly with no model installed.

Gate: the clean-room install passes from TestPyPI on the release-step-1 snapshot.

`Mila/Adaptors/Inference/Server/pyproject.toml` · `RELEASING.md` *Publishing the wheels*

#### What a model can do is inferred from its family name, not read from its manifest

`open` · `distribution` · `adaptors`

`Chat.FamilyTraits.ixx:55` answers "does this model have a reasoning channel" and "how much context
can it address" with a switch on the family, so two models of one family cannot differ and a new
reasoning model of another family reads as having no channel. `instruct` is already declared in the
record and proves the pattern, and the manifest tolerates unknown fields, so adding fields is
additive.

Moves to the manifest: `thinking_capable`, `max_context`, and a new modality field that Gemma's image
path declares. **Stays in Chat:** `streaming_capable`, which the file's own comment (`:32`) records as
a fact about the display rather than the weights, and `default_context`, layer 1 of Chat's own
configuration. Gate: a manifest omitting every new field still loads.

Also the root of Codex's "Model metadata not found" warning against `/v1/models` — all three
validated flows pass regardless, so that is a symptom of this gap rather than separate work.

#### Chat samples every model at one fixed setting, not at the one its publisher recommends

`open` · `distribution` · `adaptors` · `binding`

Chat samples every model at temperature 0.8, top-k 40 and no top-p (`Chat.Config.ixx:104`), MIS at
temperature 0.6 (`config.py:55`). No family's publisher recommends either. Their `generation_config.json`
files say: Gemma 4, every size and both drafters, 1.0, top-k 64, top-p 0.95; Llama 3.1 and 3.2 Instruct 0.6,
top-p 0.9, no top-k; Qwen 3.8 27B 1.0, top-k 20, top-p 0.95. `ChatConfiguration.md` section 5 already places
the answer in the manifest -- `temperature`, `top_p` and `top_k`, optional, from the model card -- and none
of it is built: no record carries them, `ExportArtifact` does not write them, and no adaptor reads them.
Decided by Todd 2026-10-05: an adaptor's sampling defaults come from the model, and Chat's own are wrong.

The exporter copies them from the source checkpoint's `generation_config.json`; Chat, MIS and the binding
open a session at them, below the user's own layers (`/set`, the session file, the request), which still
override. A manifest without them falls back to its family's published settings, section 5's family layer,
not to an adaptor's constant; the adaptor-wide 0.8 / 40 and 0.6 are removed.

Gate: a Gemma 4 package exported on this tree carries 1.0, 64 and 0.95, and Chat, MIS and the binding open
at them; an installed package without the fields loads and opens at its family's settings.

`Mila/Adaptors/Chat/Src/Chat.Config.ixx:104` · `Mila/Specifications/ChatConfiguration.md` section 5

### Upgrades Keep Your Models

#### A model installed under v0.20 stops loading when Mila is upgraded

`open` · `distribution` · `gemma` · `quantization` · `mila-src`

v0.20 published `gemma-4-12b-it-fp4` with its feed-forward tensors outside `ffn` and its tied
embedding and head in FP8. This tree reads neither: the block names its feed-forward `.ffn`
(`Gemma.Block.ixx:867`), and the planner refuses the FP8 table (`GemmaModel.ixx:595`). Both stay
readable through v0.21 (`ModelDistribution.md` *Compatibility*): the names as an alias at load, the FP8
table as its own decode path again -- row gather, decode matvec and batched head. Not a conversion to
six-bit codes at load, which quantizes twice.

Gate: every model `mila-llm` published at v0.20 loads on this tree as installed, and answers a factual
prompt in a piped Chat session -- the 12B by the two readers above, Llama and Qwen by confirming that
nothing they carry changed.

#### A Mila one release behind the newest build of a model cannot install it

`open` · `distribution` · `mila-src`

A pull refuses when the manifest on `main` needs a newer Mila (`requireCompatibleMilaVersion`,
`ModelManifest.ixx:267`), even when an earlier commit in the same repository holds a build this Mila
reads. Two halves. `Mila/Tools/Publishing/publish_model.py` tags the commit it replaces
`mila-<major>.<minor>`, after that commit's own minimum, before uploading a package that raises it. The
pull lists the repository's tags on that refusal and installs the highest one its version satisfies;
`IModelHub` has no query for a repository's tags today (`ModelHub.ixx:93` lists models only).

Gate: against a hub whose `main` needs 9.0 and that carries a `mila-0.21` tag, a pull installs the
tagged commit and records it; with no tag it refuses as it does now. The 12B's republish with the
six-bit table is the first real use, and tags its v0.20 build `mila-0.20`.

### Performance Next to llama.cpp

#### The website cannot show how Mila performs next to llama.cpp on the models people run

`in progress` · `perf` · `models` · `docs`

The page is evidence of quality: performance as good as llama.cpp's, measured the same way, is what shows Mila is
professional work. The site's Fast claim is a 4070 measurement against llama.cpp under LM Studio at Q4_K_M, from
before this release's kernels. The measurements that replace it come from the comparison script
(`Mila/Profiling/Benchmarks/benchmark_comparison.py`), which runs Llama 3.1 8B and Gemma 4 12B today; the
Llama 3.2 3B row needs its Q4_0 GGUF and the Qwen 3.8 rows a GGUF of their own (llama.cpp knows the
architecture).

One script in the repository drives both engines and writes the data the site shows. The target, fixed 2026-09-29:

- **Rows:** every model published at the tag -- Llama 3.2 3B, Llama 3.1 8B, Gemma 4 12B, Qwen 3.8 27B in each of
  its builds, and the Gemma 4 26B-A4B.
- **Prefill:** 512, 2K, 8K and 32K tokens, and the longest context both engines fit.
- **Generation:** 128 tokens (llama-bench's `tg128`) with an empty context, at 8K and 32K, and at the longest.
- **The longest cell** is the largest multiple of 8K that both engines load on the card with that cell's KV
  format: Mila's figure from its own planner, llama.cpp's from whether it loads. It is not the model's own maximum,
  which no 16 GB card holds for most of these models.
- **The page:** per model, a table of every cell and two graphs, prefill tokens a second against prompt length and
  generation tokens a second against context depth, both engines on each. The site renders them from the data file
  the script writes, so a rerun is the page's update and no number is copied by hand.

Each cell is averaged over at least three runs on the 16 GB reference card (RTX 5060 Ti). llama.cpp runs
`llama-bench` with flash attention, every layer on the GPU and an FP16 KV cache. Where both run the same weights
in the same format (Gemma 4 12B and Llama 3.1 8B in Q4_0) the cell is a head-to-head; otherwise each engine runs
the format a user would pick for that model on that card, named in the table. Whether llama.cpp runs Qwen 3.8's
architecture is checked before its row is promised. Every cell is shown as measured, ahead or behind; a cell where
Mila is behind is never dropped from the page, and closing it is not a condition of the release (Todd,
2026-09-29: faster is the aim, not the criterion). The page is what the landing page's performance claim links to,
with the card, the method and the command that reruns it; its public copy is written in the website's voice once
the numbers exist, says only what the cells show, and is measured once on the CUDA toolkit the tag ships.

The comparison is as fair as the script can make it, because a win that comes from the setup is not a win. Every
cell holds these, and the page states them:

- **Like against like.** The same weights where both engines load them; any tensor stored differently -- Gemma's
  output head is six bits per 32 in Mila and Q6_K in Google's GGUF -- is named in the cell. The KV cache matches too: where Mila
  runs FP8 KV, llama.cpp runs its 8-bit cache (`-ctk q8_0 -ctv q8_0`) in the same cell; BF16 against FP16 otherwise.
- **llama.cpp at its best.** Its batch and micro-batch sizes (`-b`, `-ub`) are swept per cell and its fastest
  setting kept and recorded; Mila runs the plan it chooses for itself, with nothing a user could not set.
- **The same conditions.** One card, one driver, both builds' versions and CUDA runtimes recorded; each engine sized
  to the test; a warm-up and at least three runs in both, mean and spread shown; nothing else running on the machine.
- **The same work timed.** What each engine's timer includes is stated; where one counts work the other does not --
  Mila's generation includes on-device sampling and the token's readback -- it is measured and either removed or
  shown.

Gate: the script reruns every cell unattended and enforces the rules above, the site's tables and graphs render
every cell from the script's data, and the landing page's performance claim links to that page and claims nothing
the cells do not show.

### Internal fixes

Defects in the library the user admitted to this release with no ROADMAP criterion behind them.

#### Each weight tensor over 1 MiB is its own allocation, and the rounding costs a model up to 679 MiB of its card

`open` · `models` · `mila-src` · `measured`

`CudaDeviceMemoryResource::do_allocate` makes one `cudaMalloc` per tensor, each rounded to the device's 2 MiB
granule: Qwen 3.8 27B 2.82-bit loses 679 MiB of its weights to it, Qwen FP4 497, Gemma 4 12B Q4_0 318, the
26B-A4B 225 (priced by `DISABLED_ParameterRounding_26B_Q4_0`), Llama 64-70. Layout changes nothing else
measured (`Profiling/Microbenchmarks/AllocationLayout.cu`). Judged over-engineering for the bytes alone on
2026-10-03; on 2026-10-04 the bytes became features -- the 26B-A4B is 87 MB short of the 96K its profile
measures it reliable to, and every feature a deployment selects brings more such tensors.

Work: an arena behind the weights' memory resource, one allocation per model or per layer, so tensors are
placed rather than rounded. The planner's exactness is the constraint: what it prices must stay what the
build allocates, so the arena's own layout is priced the same way. Gate: every published model's weights
within one granule of their bytes, measured per model before and after; `PlanEqualsBuild` unchanged.

`Mila/Profiling/Microbenchmarks/AllocationLayout.cu`

#### A load can take more of the card than its plan priced

`open` · `models` · `mila-src` · `measured`

Measured 2026-10-03 on the RTX 5060 Ti (`ProfileModel --model qwen --quantization fp4`): from the post-initialization
baseline, Qwen 3.8 27B FP4 consumed 14,962 MiB at context 8192 against a footprint of 14,920; at 16384 the plan left a
26 MiB margin and the card read zero free after the load -- the condition the planner's exactness exists to prevent.
Not yet attributed, and the 2.82-bit build, the Qwen this release publishes, was not checked.

Two device allocations are known to sit outside every price. `CudaExecutionContext` allocates the decode position
(`setDecodePosition`, one `int`) on its first decode step and the prefill key bounds (`key_bounds_`, `:456`) on first
use, each by its own `cudaMalloc`, outside `reserveScratch` and outside every `getMemoryStats` a plan reads; a small
allocation can cost a whole granule. Work: attribute the Qwen gap, measure the 2.82-bit build the same way, and bring
each allocation a load or a decode step makes under the plan's price.

Gate: free memory after a load and the first decode steps is at least the margin the plan states, on Qwen at both
builds and on Gemma 4 12B; `PlanEqualsBuild` covers the context's own allocations.

`Mila/Specifications/ModelFamilyParity.md` 8.3, Q4

#### The synchronous token sampler is ordered after the model's work only by an accident of how its stream was created

`open` · `ai` · `architecture` · `mila-src`

`CudaSamplingOp::forward()` and `forwardReference()` launch on stream 0, and `TokenSampler::sample()` reads the
token back with a copy given no context. They see finished logits only because the execution context creates its
stream blocking (`cudaStreamDefault`, `CudaExecutionContext.ixx:650`), which makes stream 0 wait for it -- nothing
states or checks it, and a non-blocking stream or a per-thread default stream would let `sample()` read logits
still being written. Every other entry of the op runs on the context's stream. The change: both entries dispatch on
the context's stream, `sample()` copies with the context and synchronizes it, the "Phase A, default stream"
comments are rewritten to the contract, and `Sampling.Cuda.cpp`'s tests that read the token after `forward()`
synchronize explicitly. `LanguageModel::sampleNext()` and `TokenSampler::sample()` have no caller outside tests --
whether they stay is a separate decision.

Gate: the sampling suite passes with the context's stream created non-blocking.

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Sampling/CudaSamplingOp.ixx:94` · `Mila/Src/Dnn/Samplers/TokenSampler.ixx:74`

#### The FP8 batched path rounds every weight before it scales

`open` · `quantization` · `llama` · `mila-src`

`dequantize_fp8_to_bf16_kernel` (`CudaFp8Prefill.cu:79-82`) stores `fp8 * scale` as BF16, so every staged weight is
rounded before the GEMM, where the decode matvec multiplies exact FP8 values and applies the per-channel scale once,
in FP32. An E4M3 value is exact in BF16, so the error comes only from folding the scale in early; measured against
exact logits (`ModelFamilyParity.md` 8.2, G1 result), the staged path's extra error is this rounding alone,
reproduced to four digits by a model of it. Every per-channel FP8 Linear at more than one row takes the path: since
`0.21.0-dev+31` Gemma's head no longer does, and the Linears of Llama's FP8 packages still do.

Decided 2026-09-26 (Todd): scale after the dot product, as its own change. Gate: a Linear-level test against exact
FP64, and `Gemma.LogLikelihood.Cuda.cpp`'s window bound set back from 2e-3 to 1e-3.

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Fp8Prefill/CudaFp8Prefill.cu:79`

#### A model under a Windows path with characters outside the ANSI code page cannot be opened

`open` · `distribution` · `build` · `mila-src`

`WeightsReader` and `SafeTensors` open with `std::fopen( filepath.string().c_str(), ... )`. On Windows
`path::string()` converts to the ANSI code page, so a path it cannot represent fails to open, and the store lives
under `%LOCALAPPDATA%`: a user whose profile name has such characters cannot load any model. Found 2026-09-29
clearing the `fopen` deprecation warning; not reproduced. The fix opens by `path::c_str()` (wide on Windows) without
an `#ifdef` in a module. `SafeTensors` and `TokenSequenceLoader` can take `<fstream>`, which also removes the
library's only C4996 and unblocks the warnings ratchet (`Vnext.md`); `WeightsReader` keeps positioned reads beside
its mapping, since faulting a large model through the mapped view throttles below disk bandwidth.

Gate: a model installed under a store path with characters outside the code page loads.

`Mila/Src/Dnn/Serialization/WeightsReader.ixx:193` · `Mila/Src/Dnn/Serialization/SafeTensors.ixx:188`

#### The published binaries and images are built on CUDA 13.3 while 13.4 is current

`in progress` · `build` · `ci` · `distribution`

Move the declared toolkit to 13.4.2, between pieces of native work. Docker Hub has carried
`nvidia/cuda:13.4.2-{base,runtime,devel}-ubuntu{22,24,26}.04` since 2026-09-29 (read on Docker Hub that
day). The move was decided for 13.4.1 on 2026-09-24 (Todd) -- the minor is where behaviour and the driver
floor move, so it should surface early in the cycle, with 13.4.2 a later patch bump -- but it never
happened, and 13.4.2 now exists: one move to it replaces both. The local toolkit is 13.4.1 and needs
13.4.2 installed first.

Moves together (RELEASING.md, toolkit paragraph): `$cudaVersion` in
`scripts/pypi/build-wheel-windows.ps1:59`, `Docker/Dockerfile.wheel:23`, `Docker/Dockerfile.runtime:21`,
`Docker/Dockerfile:9`, `:18`, `Docker/build-chat.sh:12`, `Docker/build-all.sh:26`,
`build-pipeline.yml:54`, `:57`, and the docs naming 13.3 — `README.md:276`, `:286`, `:327`, `:341`,
`getting-started.md:25`, `:33`, `:116-131` (the WSL installer URL and filename change with the
patch level), `:183-191`, `:238`, `CONTRIBUTING.md:46`, `:113`, `:140`, `Docker/README.md:15`,
`:20`, `Web/content/start.md:15`, `RELEASING.md:508`. `RELEASING.md:529` records what the
`0.20.0b3` wheels were built on and stays.

Consequences to carry into the work. Wheel users see nothing — the `nvidia-*` dependencies and
minor version compatibility, and this is now checked rather than assumed: a 13.3-built wheel loads
and generates correctly against the `>=13.0` floor those dependencies declare, on Windows
empirically and on Linux by symbol (`RELEASING.md`, *What the declared toolkit is not*). Moving the
build to 13.4 does not disturb that, but re-check the floor if the cuBLASLt surface grows. Image
users' driver floor rises: the base image's `NVIDIA_REQUIRE_CUDA`
becomes `cuda>=13.4`, and the container toolkit refuses a GeForce driver below it. Every local build
directory is configured against v13.3 while `CUDA_PATH` names v13.4 (13.4.1 installed), so a fresh
configure already drifts — reconfigure all of them deliberately. Published tok/s figures and the
cuBLASLt findings in the specs are 13.3 measurements; re-measure them once, just before the
release, on the 13.4.x that ships. CI's first run
starts with a cold ccache. The patch levels already differ today: Windows pins resolve to 13.3.1,
the Linux images to 13.3.0.

Admitted 2026-10-10 (Todd): the Docker Hub images move to 13.4 for v0.21.0, and with them every site above, as one move.

Gate: every image `publish-image.sh` pushes is built `FROM` `nvidia/cuda:13.4.2-*` and passes `verify-image.sh`; the
wheels are built on 13.4.2 and pass the clean room; no document names 13.3 except `RELEASING.md:529`'s record.

`RELEASING.md`, the toolkit paragraph · `Docker/Dockerfile.runtime:21`

#### Releasing Mila is still a day of manual steps, and the release script stops before anything is published

`open` · `ci` · `build` · `distribution`

`scripts/release/release.py` (`+2`) builds and verifies a release locally from a `git archive` export — the
Windows and Linux wheels, their install tests, both images and their checks, nine stages unattended in about an
hour — and `prepare` / `finish` set the version, merge `dev` into `master` and tag, all in the local repository.
What it does not do is everything a publish is (`RELEASING.md`, steps 1 and 8 to 11), and what has changed since:

- The server's wheel, a fifth file since `+56`, and the CUDA 13.4.2 base images are not in its stages.
- No Linux clang build and full suite from the export (ReleaseAutomation.md ring 2), so a patch tag builds nothing.
- Publishing: the TestPyPI upload and the wait for the index before the clean room is dispatched, the PyPI
  upload, the Docker Hub push behind a real approval rather than a piped prompt, the Hub overview through its
  API, the tag push, the GitHub Release and Discussion at step 11, and the site's publish dispatch.
- The release still validates a `.devN` snapshot at step 1 and ships a different build at step 8; staging the
  exact files, verifying them, then promoting them closes that.

`ReleaseAutomation.md` is still marked Draft, with its section 8 decisions open. Admitted 2026-10-10 (Todd), to
be finished in this release.

Gate: v0.21.0 is released by the script, from one command, with Todd's approvals the only manual steps.

`scripts/release/release.py` · `Mila/Specifications/ReleaseAutomation.md` · `RELEASING.md`

#### The deployment types repeat their namespace in their names

`open` · `api` · `mila-src` · `breaking`

`Mila::Deployment::DeploymentPlan`, `DeploymentPlans`, `DeploymentRequest`, `DeploymentRefusal` and
`DeploymentRefusedError` kept the names `Deployment.md` 12.7 recorded when the planner moved to the top-level
namespace at `0.21.0-dev+6`. The namespace lets them drop the prefix, as `Mila::Dnn::Conversation` did for `Turn` and
`ToolCall`, and v0.21 is the release that first publishes them, so renaming before the tag costs no user a
migration. `DeviceReading` is not part of this: its replacement name waits on `Deployment.md` 12.5 (`Vnext.md`).

Gate: the types, the binding's projection and `Deployment.md` use the short names.

`Mila/Src/Deployment/`
