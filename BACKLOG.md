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
The release's date and what drains first are in [ROADMAP.md](ROADMAP.md). Whatever has not landed by that date
goes back to [`Vnext.md`](Mila/Issues/Vnext.md) and the release ships narrowed, rather than the date moving to
accommodate the list. A backlog that grows while the date holds is the signal to drain it early.

**Done means deleted**, in the same commit as the work — `done` is a working-tree marker and is
never committed.

---

## Current release (v0.21.0)

The families finished and measured, in the themes ROADMAP gives them. `Direction.md` section 5 — the
handle, the deployment request, `Mila::AI` and the applications rebuilt on it — is v0.22.0, and its items
wait in [`Vnext.md`](Mila/Issues/Vnext.md); what tool calling stands on -- the model's tool grammar, the
handle, `Mila::AI` and its tool loop -- comes forward under *The Desktop Agent* (Todd, 2026-10-25).

### Gemma 4 Complete

#### Gemma 4 12B is multimodal and Mila drops its image and audio weights

`in progress` · `gemma` · `mila-src` · `apps`

The 12B is **encoder-free** (`Gemma4UnifiedForConditionalGeneration`): no vision tower and no audio
encoder. An image is resized and cut into 48-pixel patches, at most 280 soft tokens, each through a
small embedder (about 50M parameters) into the decoder; an audio clip is 40 ms frames of raw 16 kHz
samples, each through one projection. `convert_weights.py` skips both. The image tokens of one picture
attend each other in both directions; that mask, the largest piece, is built (`+35`, `Gemma4Modality.md` 5.4):
Gemma's two flash prefill kernels take per-row key bounds, and every other prefill path refuses an image run.
Work left, in section 7's order: the image embedder (about 50M parameters), soft-token placement, a prompt that
carries media, prefill chunks that never split an image's run, and prefix reuse keyed on the media rather than
its token ids; the resize, matched to torchvision's; audio, a 40 ms framing and one projection; Chat accepting an
image or a clip (`/image`, `/audio`, Todd 2026-10-04), the inference server's content blocks in both protocols,
the binding; converter, manifest and footprint. In this release the applications pass media to the model
directly; reaching it through `Mila::AI`, and selecting a modality in a deployment request, are v0.22.0's.

A deployment without a modality should not sample its markers. On 2026-09-28 the 12B sampled `<image|>` (258882)
mid-sentence in a text-only Chat session, losing the rest of a pun. The sampler now never draws any of Gemma 4's
seven markers (`GemmaModel::suppressedTokens`, `TokenSampling.md` 3.3); a deployment that selects a modality takes
its own markers out of that list.

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

`in progress` · `gemma` · `distribution`

Landed in `Mila/Src` during rc.1 — the router and expert bank, wired into `GemmaModel` with FP4 and Q4_0
expert banks and a streaming converter, gated against HuggingFace at BF16, FP4 and Q4_0. It appears in no README
capability row, no CLAUDE.md target and no release note, because no published package uses it, so no end user
can install it.

Held out of v0.20.0 (Todd, 2026-09-21). A package alone would have published it below every other model Mila
ships; its kernels now run at llama.cpp's rates or above (`ModelFamilyParity.md` §8.2, G5b), and it waits on the
rest of `Gemma.md` §10.7's bar: quality
across the planner's range, by the method `ModelFamilyParity.md` §9 item 18 settles; decode replay gated
equal to the called path on a routed network; active parameter bytes in the plan and in Chat's display;
the model run through Chat, the inference server and the Python binding, tool calls included -- the server with
Codex and Hermes (`+79`, `+80`) and Chat (`+82`) have run it live; the binding is not checked. Then the
package, its card and the capability row. A first publish, so it is not held for the 12B's.

`Mila/Specifications/ModelFamilyParity.md` §8.2, G6

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
Ships as a model of its own, `gemma-4-12B-it-q4_0`, which supersedes the FP4 package rather than republishing
it (`ModelDistribution.md`, *A model replaced by another*).

Gate: every Q4_0 tensor bit-identical to `google/gemma-4-12B-it-qat-q4_0-gguf`; then G2 re-run in Mila on the
Q4_0 build, cost over BF16 by band, on a second book as well as 30312.

`Mila/Specifications/ModelFamilyParity.md` §9, item 14 · `Mila/Specifications/Quantization.md`, "Q4_0"

#### Gemma's quality has not been gated across the 262144 tokens the planner may choose

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

#### Gemma 4 packages do not carry Google's drafters, so no load can use them

`open` · `gemma` · `distribution` · `mila-src`

The speculative loop is built and measured for both models (`Gemma4Mtp.md` 4.7, 4.8): draft, verify as one
multi-row decode, accept, rewind, greedy and sampled, replayed. With the drafter's head at the six-bit table format
the 12B decodes 1.59 / 2.02 / 2.15x faster greedy on prose / code / chat and 1.45 / 1.88 / 1.96x sampled, the
26B-A4B 1.20 / 1.28 / 1.36x and 1.11 / 1.27 / 1.29x, each at its best K (RTX 5060 Ti). The six-bit head keeps
acceptance within 0.02 of BF16's and makes a draft step a third cheaper, so it is the format a package carries
(`Gemma4Mtp.md` decision 2), about 0.53 GB rather than 0.85. Greedy output parts from plain greedy only at
near-ties; the RTX 4070 parts earlier than that, under Model fixes and parity.

What is not done: decided 2026-10-06 (Todd), every Gemma 4 package carries its own drafter, so one install gets
both. A load gets it with the measured K wherever its speedup is measured and the planned context still fits with
it loaded -- on the 16 GB card the 26B-A4B's drafter is context it gives up -- so the planner prices it and
`PlanEqualsBuild` holds with it built. Chat takes that default, as it takes the model's sampling, with a setting to
turn it off. Selecting it in a deployment request is v0.22.0's.

Gate: each Gemma 4 package exported on this tree carries its drafter as a manifest role; a load decodes with it
token-for-token as without it, at the measured speedup, wherever the planned context fits with it resident;
`PlanEqualsBuild` holds with it; and Chat's setting turns it off.

`ROADMAP.md`, Gemma 4 Complete success criteria · `Mila/Specifications/Gemma4Mtp.md` 4.7

### Qwen 3.8 Complete

#### End users cannot install the Qwen 3.8 27B build that holds 64K of context on a 16 GB card

`open` · `distribution` · `quantization` · `qwen`

`fp4-cb3` is the Qwen 3.8 this release publishes (Todd, 2026-10-20): FP4 with two measured cuts to three bits,
FFN down and DeltaNet query/key (`Qwen3.8.md` section 5). With the FP8 KV cache and the weight arena it plans 65,792 on
the RTX 5060 Ti, loses 2.3% of the closing brackets BF16 gets right (FP4 2.1%, the 2.82-bit build 10.9%), and
passes the tool-name probes, recall to 64K and the Hermes, Codex and Claude Code sessions. The package is built and lives only on this machine. No card exists:
the cards under `Tools/ExportArtifact/ModelCards/` are FP4's and the 2.82-bit build's. The 2.82-bit build is not
published (Todd, 2026-10-20; `Future.md`).

Work: its card, its repository under the model naming rule, the upload, and its manifest. Not decided: whether it
supersedes the v0.20 FP4 model (`ModelDistribution.md`, *A model replaced by another*). The benchmark and context
profile criteria name this build.

Gate: `https://huggingface.co/api/models/mila-llm/<its repository>` answers 200 anonymously, and a clean store
installs and runs it at 64K on the 16 GB card.

`ROADMAP.md`, Qwen 3.8 Complete success criteria · `Mila/Specifications/Qwen3.8.md` section 5

### Llama 3 Measured

#### Nobody can say whether a published model scores like its original on standard benchmarks

`open` · `models` · `measured`

What a card says about quality today is fidelity alone — perplexity ratios and KL against a reference build --
which a reader cannot act on. `Tools/Evaluation` runs a model on public benchmarks (IFEval, GSM8K, BFCL, RULER,
MMLU-Pro) on Mila, HuggingFace and llama.cpp from the same token ids, and compares them document by document
(`ModelEval.md`). It ran on CUDA for the first time at `+55`, as a 20-document smoke run.

Work, in `ModelEval.md`'s order: Phase 1, Mila BF16 against HuggingFace BF16 on Llama 3.2 3B, which sets the
noise floor; Phase 2, each quantized build of that model against the same reference; then every published model
of every family (Phase 6), which serves the Gemma 4 and Qwen 3.8 criteria as well as Llama's, and `fp4-cb3`, the
Qwen 3.8 this release publishes, for agentic work (Phase 8). Every model but the
Llama 3B has BF16 weights larger than the 16 GB card, so where their reference runs is decision 1, owed before
their arms run. The card's table and its sentence follow section 8.

Gate: Phase 1's — every paired interval contains zero and every task's median divergence is past the first
sentence; then each published build's scores beside the original's, its flips read against that floor.

Two harness limits found on the way. The tools assumed greedy decoding and short replies: `run_bfcl.py` now sends no
sampling knob and asks for 32,768 tokens (greedy is `--greedy`), but `reference_server.py`, BFCL's `hf` arm, refuses
any temperature above 0 and defaults a reply to 256 tokens -- right for Phase 1's engine parity, unable to measure a
model at its recommended setting -- and lm-eval's served arm sends each task's own greedy `gen_kwargs`; results
recorded before 2026-10-18 with thinking off are under `Data/Evaluation/qwen38-*-thinking-off*`. And Llama 3's
template writes today's date into every prompt, so `compare_arms.py` refuses a pair whose arms start either side of
midnight; Phase 1 takes about 2.7 hours an arm, and the template takes a `date_string` when one is given.

`Mila/Specifications/ModelEval.md` · `Mila/Tools/Evaluation/README.md` · `reference_server.py:138` ·
`run_arm.py` `lm_eval_arguments`

#### Nobody knows how deep into a conversation each model still finds a fact, obeys its instructions and calls a tool correctly

`in progress` · `models` · `ai` · `gemma` · `llama` · `qwen` · `gate`

The quality evidence Mila has is perplexity by context band (`ModelFamilyParity.md` 3.4, 8.2 G2) and
two single-family instruction-retention tests (`Llama.InstructionRetention.Cuda.cpp`,
`Gemma.InstructionRetention.Cuda.cpp`). None of it says where an agent fails, and no family is measured
across the contexts its planner can choose. Admitted 2026-10-03 for every family in the parity matrix
(Todd); the design is `ContextProfile.md`, a tool under `Mila/Tools/ContextProfile`.

In scope: Phase 1 (fit, loss by band, recall at depth) and Phase 2 (instruction retention, tool-call
fidelity), for every model the release publishes -- Llama 3.2 3B and 3.1 8B, Gemma 4 12B and 26B-A4B
in Q4_0, Qwen 3.8 27B as `fp4-cb3` -- at every band each fits on the 16 GB card, against section 9's decisions and
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

Llama's tool-call arm can ask its question now that Llama's grammar is in `Mila/Src` and the scratch packages
declare their version (`+94`). Turn cost and the llama.cpp column (Phases 3 and
4) are not required by the criterion. Storing Gemma's global keys once (K = V, `RopeInAttention.md`)
waited on the 26B-A4B's profile, to return for discussion only if its reliable depth ran past the 80K that
fits today; the profile puts it at 96K and not 128K (`ContextProfile.md` section 8), so it returns, framed
as reaching 96K rather than 128K.

Gate: section 8's Phase 1 gate before any profile is recorded, then one profile per model, and the
reliable depth each reports is the one compaction reads.

Llama 3.1 8B loses a system instruction by 65536 in BF16 and FP8 cache alike, and HuggingFace on the BF16 weights
loses it the same way (`Quantization.md` Part III, decision 6's behavioral arm), while its planner may choose up to
131072. It is the model's, so its profile is where a Llama user is told it.

`ROADMAP.md`, each family's success criteria · `Mila/Specifications/ContextProfile.md`

### Applications Ship

#### The inference server's README says `pip install mila-llm-server`, and no such package is published

`in progress` · `apps` · `ci` · `distribution`

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

`Mila/Apps/Server/pyproject.toml` · `RELEASING.md` *Publishing the wheels*

#### What a model can do is inferred from its family name, not read from its manifest

`in progress` · `distribution` · `apps` · `mila-src`

`Chat.FamilyTraits.ixx:55` answers "does this model have a reasoning channel" and "how much context
can it address" with a switch on the family, so two models of one family cannot differ and a new
reasoning model of another family reads as having no channel. `instruct` is already declared in the
record and proves the pattern, and the manifest tolerates unknown fields, so adding fields is
additive.

Decided 2026-10-12 (Todd), `ModelDistribution.md` *Capabilities and modalities*: the manifest declares
capabilities (`instruct`, `thinking`, `tools` from the checkpoint, and `agentic`, Mila's measured verdict) and
modalities (text, image, video,
audio -- only those Mila supports); MTP is reported where the package carries a drafter Mila supports; each can be
turned off. Read through the installed-model path (Internal fixes, below). The context the model addresses comes
from its weights, not the manifest. **Stays in Chat:** `streaming_capable`, a fact about the display rather than
the weights, and `default_context`, layer 1 of Chat's own configuration. Gate: a manifest omitting every new field
still loads; every published model republished with them.

Also the root of Codex's "Model metadata not found" warning against `/v1/models` — all three
validated flows pass regardless, so that is a symptom of this gap rather than separate work.

Found 2026-10-18 serving Qwen3.8-27B-fp4 from an install that predates the fields: `/v1/models` says
`supports_reasoning: false` and `capabilities.tools: false` for a model that reasons by default and calls tools,
because an absent field reads as false rather than unknown. Until every model is republished, each older install
reports the wrong answer. `Mila/Apps/Server/src/mila_llm_server/protocols/openai/models.py:32`

#### The website does not tell a user how to run Chat, the inference server or the model store

`open` · `docs` · `apps`

Someone who wants a local model behind Claude Code, Codex or Hermes Agent, or a chat in their terminal, finds
nothing written for them: the server appears only in two blog posts, which go stale, and Chat only as one line of the
build instructions. Admitted 2026-10-18 (Todd); the shape proposed the same day:

- An Applications section in the site's navigation, with a page per application: Chat (built from source or run from
  the image), the inference server (`pip install`), and `mila-store` (installing a model is step one for both).
- The server's page covers each harness it is tested with -- install, start the server, the harness's own settings,
  the models it is tested with and their limits. The harness instructions stay in one file,
  `Mila/Apps/Server/docs/harnesses.md`, mounted into the site rather than copied.
- A Get started tab for a user who wants a local model behind a coding agent, pointing at the server's page.
- End-user prose throughout (`CLAUDE.md` *End-User Prose*), written after the harness baselines so every tested
  version and limit it states is measured.

`Web/hugo.toml` (navigation), `Web/layouts/index.html` (Get started tabs), `Mila/Apps/Server/docs/harnesses.md`

#### The inference server answers a request for any model name with the model it has loaded

`open` · `apps`

Nothing compares a request's `model` with the model loaded, so a harness configured for one model is served by
another without a word. It is how a stale Hermes setting went unnoticed for three months (found 2026-10-18):
requests naming `gemma-4-12b-it` were answered by Qwen 3.8 27B FP4 and the Gemma 4 26B-A4B, and the harness guide
recorded both as passing. Not checked: what OpenAI and Anthropic return for an unknown model (OpenAI documents
`model_not_found`), and whether every harness sends a name the server could match.

`Mila/Apps/Server/src/mila_llm_server/`

#### The inference server joins the text parts of one message with nothing between them

`open` · `apps`

Claude Code 2.1.300's `system` field is three text blocks, and the first (`x-anthropic-billing-header: ...;`) is
joined straight onto the second ("You are a Claude agent..."); Codex 0.162.1 sends its skills and permissions text
as two parts of one `developer` message, joined the same way (found 2026-10-15 capturing the harnesses). Not
checked: how Anthropic and OpenAI join a message's parts, or whether the billing line reaches Anthropic's models.

`Mila/Apps/Server/src/mila_llm_server/conversation.py` `text_of`

### The Desktop Agent

#### Three places turn a model's name into the type that loads it, each its own way

`open` · `ai` · `architecture` · `api`

The architecture-to-concrete erasure exists three times in two languages -- Chat's `ModelVariant`
(`Chat.Session.ixx`, ten `std::visit` sites), the binding's per-family session classes, and the inference server's
`ModelFamily` enum (`model_worker.py:38`). The bridge from a store name to a family, a weight format and
capabilities is the library's since `0.21.0-dev+61` (`InstalledModel`, `Deployment.md` 8.1); what remains is the
erasure itself. One handle and one factory in `Mila::AI`, with the architecture's identity in `Mila/Src` beside
the manifest reader; the factory takes a deployment request; a network composed from components meets the handle
through a C++ concept, with no registration. Tool calling's loop drives this handle, so it is built here; the
server and the binding move onto it in v0.22.0. Moved from `Vnext.md`. Gate: adding an architecture is an edit
in one place, and decode throughput through the handle is unchanged against the direct type.

`Mila/Specifications/ModelHandle.md` section 7 (Phase 3) · `Mila/Specifications/Direction.md` 3.2, 8

#### Running a model from a program means naming its concrete C++ type

`open` · `ai` · `api`

The application-facing object does not exist: a program names `GemmaModel<Cuda, BF16>::load` or a sibling, and
the tool loop, streaming and conversation state are its own to write. `Mila::AI`, the object an agent is, on
`Direction.md` 4.2's rules: small; never decides a deployment; `plan()` and `model()` part of the contract; one
virtual call per `respond`; a composed model first-class; named by its store name. Moved from `Vnext.md`. Gate:
one program runs Gemma, Llama and Qwen by changing only the name, streaming and calling a tool, and a sample
creates it over a network it composed itself.

`Mila/Specifications/Direction.md` section 4

#### The tool loop exists only inside Chat

`open` · `ai` · `apps`

Parse a call, run it, return the result, continue -- written once, in `Chat.Session.ixx`, so a developer's program
that wants tools rewrites it. It moves into `Mila::AI` (`ToolCalling.md` section 4): limits per turn on rounds,
time and reply budget; a stopped reply cancels a waiting or running call; a result larger than the context left
is cut or refused and the model told which; the conversation is the agent's, its history and cache agreeing.
Moved from `Vnext.md`. Gate: neither Chat's session nor the desktop app's holds a loop, and both run their calls
through `Mila::AI`.

`Mila/Apps/Chat/Src/Chat.Session.ixx` (`answerToolCall`)

#### A tool result is re-rendered and re-tokenized with the whole conversation before generation continues

`open` · `ai` · `models` · `mila-src`

Decided in v0.20 and deferred (`MilaProductFamily.md` Decided 1): a tool result's tokens are appended to the live
KV cache, with no re-render of the conversation and no re-tokenize. Today Gemma recovers the prefill through
transparent prefix reuse (`GemmaModel.ixx:364`) but still renders and tokenizes everything; Qwen prefills from 0
every turn (`QwenModel.ixx:355`). The grammar that frames a result as tokens is the model's; the loop that decides
when to splice is `Mila::AI`'s, and reads from the handle which kind of prefix reuse a model offers. Moved from
`Vnext.md`. Gate: across a multi-turn tool session, prefill tokens per turn equal the tokens the turn added,
measured, for every family that permits prefix reuse, Qwen from the position it saved.

`Mila/Specifications/Direction.md` 5.3

#### A program cannot run a model's tool calls under the person's rules without writing the rules itself

`open` · `ai` · `api`

`ToolCalling.md` sections 3.1 to 3.3: a tool with its schema and declared needs, registered as a C++ callable;
in-process execution with the arguments checked against the schema first, a timeout, cancellation and a result
cap, a failure returned to the model as an error result; and the gate every call passes -- allow, ask or deny with
a grant, rules matching on arguments, the application's hook for asking, the question built from Mila's data and
never the model's text, once / this conversation / deny, no hook meaning deny, a denial reaching the model as a
result, and a record of each call as an event. The code built for the per-user broker (`Mila/Apps/Tools/Broker`,
`Mila/AI/`) is material for it; what the refit does not use is deleted, with its tests. Gate: a test for each
outcome, including a question whose model-written summary is not what the person is shown.

`Mila/Specifications/ToolCalling.md` section 3

#### Nothing keeps a tool inside the folder it was given, or off local addresses

`open` · `ai` · `gate`

`ToolCalling.md` section 3.4: the grant a tool opens files and makes connections through. On Windows one file has
many names: `..`, links, junctions, mount points, short 8.3 names, case, alternate data streams, device names, the
`\\?\` and `\\.\` forms, network paths. The grant compares the opened file's final path with the granted folder's,
and refuses the non-path forms before opening; it refuses private, loopback, link-local and unspecified addresses
on the address actually connected to, and on every redirect. Gate: a test per route and per address kind, by name,
by literal address and through a redirect, each first shown reaching its target with the check removed. Gates
every item below that runs a tool.

`Mila/Specifications/ToolCalling.md` section 3.4

#### A model has no tools to read or edit files or to read a web page

`open` · `apps`

The desktop app's own tools, written through grants (Mila ships none): reading a file or a range of its lines,
listing a folder, finding files by name, searching text, writing a file and editing one exact passage, in the
folder the person chose and its `.mila` folder; and fetching a web page as readable text, size-capped, from public
addresses. Reads allowed by default, writes and fetches asked. Absent today: Chat's only tool is a `get_weather`
stub (`Mila/Apps/Chat/Src/main.cpp:515`, and the same stub in `Mila/Apps/Desktop/Session/Src/main.cpp`).

`Mila/Specifications/ToolCalling.md` section 5

#### The desktop app runs only against a mock

`in progress` · `apps`

`ChatDesktop.md` phases 3 and 4: the session process that speaks section 4 over stdio on a real model, then the
window starting it in place of the mock, with tool calls through `Mila::AI`. Gate: a script driving the session
process runs the equivalents of scenarios 1 to 8 and 11 on a real model; the window runs the theme's task on
every published model that calls tools. The session process is built (`mila-desktop-session`) and its gate passes
on Llama, Gemma and Qwen, with a real install and removal. The window starts it in place of the mock and runs
scenarios 1 to 11 on Llama, Gemma and a real install; tool calls through `Mila::AI` and the theme's task remain.

`Mila/Specifications/ChatDesktop.md` sections 6.1, 8

#### Tool calling has never been driven end to end on Llama

`open` · `llama` · `apps`

Gemma 4 is validated through both applications across plain-chat, single-tool and tool-result-resume flows. Llama
3.1 8B has never been driven through a single tool call, from Chat or over the wire; the framework is family-neutral,
and what is missing is the evidence that Llama's grammar reaches it. Moved from `Vnext.md`, where moving it out had
narrowed the Models criterion and `README.md`'s Llama section; the theme's criterion now restates it.

### Upgrades Keep Your Models

#### A model installed under v0.20 stops loading when Mila is upgraded

`open` · `distribution` · `gemma` · `quantization` · `mila-src`

v0.20 published `gemma-4-12b-it-fp4` with its feed-forward tensors outside `ffn` and its tied
embedding and head in FP8. This tree reads neither: the block names its feed-forward `.ffn`
(`Gemma.Block.ixx:867`), and the planner refuses the FP8 table (`GemmaModel.ixx:595`). Both stay
readable through v0.21 (`ModelDistribution.md` *Compatibility*): the names as an alias at load, the FP8
table as its own decode path again -- row gather, decode matvec and batched head. Not a conversion to
six-bit codes at load, which quantizes twice.

The published Llama packages are not untouched either: `Llama-3.2-3B-Instruct-fp4`, installed fresh from
the hub on 2026-10-13, is refused because its manifest records no position scaling ("packaged by an earlier
version of Mila, which did not record how the model's positions are scaled"). v0.20 read it, so it stays
readable through v0.21 under the same rule.

Gate: every model `mila-llm` published at v0.20 loads on this tree as installed, and answers a factual
prompt in a piped Chat session -- the 12B by the two readers above, the Llama packages likewise, and Qwen
by confirming that nothing it carries changed.

#### A Mila one release behind the newest build of a model cannot install it

`open` · `distribution` · `mila-src`

A pull refuses when the manifest on `main` needs a newer Mila (`requireCompatibleMilaVersion`,
`ModelManifest.ixx:267`), even when an earlier commit in the same repository holds a build this Mila
reads. Two halves. `Mila/Tools/Publishing/publish_model.py` tags the commit it replaces
`mila-<major>.<minor>`, after that commit's own minimum, before uploading a package that raises it. The
pull lists the repository's tags on that refusal and installs the highest one its version satisfies;
`IModelHub` has no query for a repository's tags today (`ModelHub.ixx:93` lists models only).

Gate: against a hub whose `main` needs 9.0 and that carries a `mila-0.21` tag, a pull installs the
tagged commit and records it; with no tag it refuses as it does now. The 12B no longer exercises it: its
Q4_0 build is a new model, and marking the FP4 package superseded leaves its minimum alone.

#### A user with the Gemma 4 12B FP4 package is never told that a better 12B replaces it

`open` · `distribution` · `gemma` · `apps` · `mila-src`

Nothing in a manifest, the store or Chat can say a model has been replaced. The Gemma 4 12B in Q4_0 of
Google's quantization-aware weights is a different model from the FP4 package v0.20 published, and the
better one by a wide margin at length (`ModelFamilyParity.md` §9, item 14), so it is published as its own
repository and the FP4 one is superseded (`ModelDistribution.md`, *A model replaced by another*; Todd,
2026-10-20). Work: a `superseded_by` hub coordinate on the manifest and the store record, read and written
like `base_model`; `mila-store list`, `describe` and a `pull` of the model name the successor, and so does
Chat where it shows the model; `publish_model.py` commits the field alone to the replaced repository, and its
card's first line names the successor. A superseded model still installs and loads.

Gate: a manifest carrying `superseded_by` is read by this tree and by v0.20.0's reader alike, its minimum
unchanged; `list`, `describe`, `pull` and Chat name the successor for it and say nothing for a model without
it; `mila-llm/gemma-4-12b-it-fp4` carries the field and the card line once `gemma-4-12B-it-q4_0` is published.

`Mila/Src/Distribution/ModelManifest.ixx:330` · `Mila/Src/Distribution/ModelStore.ixx:1253` ·
`Mila/Apps/Tools/Store/Store.ixx:346` · `Mila/Apps/Chat/Src/Chat.ModelCatalog.ixx:796` ·
`Mila/Tools/Publishing/publish_model.py`

### Performance Next to llama.cpp

#### The website cannot show how Mila performs next to llama.cpp on the models people run

`in progress` · `perf` · `models` · `docs`

The page is evidence of quality: performance as good as llama.cpp's, measured the same way, is what shows Mila is
professional work. The site's Fast claim is a 4070 measurement against llama.cpp under LM Studio at Q4_K_M, from
before this release's kernels. The measurements that replace it come from the comparison script
(`Mila/Profiling/Benchmarks/benchmark_comparison.py`), which runs Llama 3.1 8B and Gemma 4 12B today; the
Llama 3.2 3B row needs its Q4_0 GGUF and the Qwen 3.8 rows a GGUF of their own (llama.cpp knows the
architecture).

For Qwen 3.8 (agreed 2026-10-19, Todd): `fp4-cb3` and FP4 against Unsloth's UD-IQ4_XS, about the same size on the
card, with the 8-bit cache on both sides; both GGUFs are in the HF cache and llama.cpp b11216 loads them. Missing: an
`fp4-cb3` row, Qwen GGUF paths, and a KV column -- the script always runs llama.cpp with an FP16 cache, so an
FP8-cache cell cannot be matched with `-ctk q8_0 -ctv q8_0`; `ProfileModel` has no `fp4-cb3` quantization, and its
help says Qwen refuses an FP8 cache, which the code no longer does. Run internally first, against b11216.

One script in the repository drives both engines and writes the data the site shows. The target, fixed 2026-09-29:

- **Rows:** every model published at the tag -- Llama 3.2 3B, Llama 3.1 8B, Gemma 4 12B, Qwen 3.8 27B in FP4 and
  `fp4-cb3`, and the Gemma 4 26B-A4B.
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

### Model fixes and parity

A model's own defects, and gaps between the families, the user admitted to this release with no ROADMAP criterion
behind them.

#### Qwen 3.8's loss on chat conversations rises with context, where on prose it falls

`open` · `qwen` · `quantization` · `measured`

On 31 Hermes tool-call conversations rendered through Qwen 3.8's own template and laid end to end
(`Tools/Quantization/calibration_text.py --held-out-tool`, `MILA_QWEN_CORPUS`), the FP4 build's perplexity is 60.0,
223.2 and 534.5 at 4K, 8K and 16K, and the 2.82-bit build's 0.87-0.93 of that, below FP4 at every length, which the
test asserts against. The same runs on wikitext-2 test score 6.47, 6.15 and 5.69. Ruled out: line endings (CRLF and
LF both), and the tokenizer (Mila's ids equal HuggingFace's over the first 30,635 tokens). Scored one conversation at
a time from its own start (`DISABLED_QualityOnConversations`), FP4 reads 63.8 and HuggingFace's BF16 model 89.7 on the
same conversations, so the level is the model's own on this text; transformers' own forward and Mila's BF16 stack
both agree with that reference. Not measured: BF16 laid end to end at 8K and 16K, which is what says whether the rise
with context is the model's or FP4's.

`Mila/Tests/Dnn/Models/Qwen/QwenModel.Load.Cuda.cpp` `DISABLED_QualityGateAcrossContextLengths`

#### Qwen 3.8 27B at four bits predicts a long text worse with all of it in context than with its last 1,024 tokens

`open` · `qwen` · `quantization` · `measured`

Found 2026-10-03 by the first ContextProfile run of `qwen-3.8-27b-fp4` (RTX 5060 Ti, files `D:\Claude\context_profile\
provisional`, `qwen_loss`). Five PG-19 books at 16384, window 64: gate 1 fails in 9 of 10 book-bands, pooled 2.835 /
2.739 at 0-8K and 3.013 / 2.685 at 8K-16K (whole book / 1024 tokens of it); book 10356 reads 3.61 / 2.49 at 8K-16K.
The 2.82-bit build on the same books passes but for 0.015 on one band -- 3.341 / 3.459 on book 10321 at 8K-16K, where
FP4 reads 3.39-3.46 -- although FP4 has more bits on every role where the two plans differ (feed-forward, DeltaNet
query/key, value/gate/output); both share FP4 attention and head and BF16 gating. Not the prefill chunk: 64, 256 and
1024 move it by 0.05. Not the exported file: quantized on load from the BF16 blob it scores identically. Not mainly
FP8 activations: `QwenPackedArtifactTests.DISABLED_DecodeAgainstPrefillAlongTheBook` (added `+34`) scores 257
targets by decode (BF16 activations) and by prefill -- book 10356 after 12288 reads 5.10 decode / 5.25 prefill for
FP4 against 2.59 / 2.59 for the 2.82-bit build; after 2048, 2.87 / 2.87 against 3.03 / 3.02. In August FP4 improved
with context on wikitext to 16K (`Qwen3.8.md` section 8, 5.686 at 16K), segments scored cold, no chat turn. Recall
at 11K and 16K is 99-100% on the same build. One difference left between the plans besides bits: the codebooks
were fitted by GPTQ on calibration text, FP4 is round to nearest. The reference that would separate format from
defect is HuggingFace BF16 and HuggingFace with the same weights rounded to FP4, layer-streamed
(`hf_qwen_layer_stream.py` scores one prompt's last token today), at those positions.

The agentic build carries it at the length it is chosen for (2026-10-19, `qwen-3.8-27b-fp4-cb3-fp8-kv`, 5060 Ti, same
five books at 65,536, chunk 512): whole book / 1024 tokens of it 2.908 / 2.866 at 0-8K, 2.945 / 2.768 at 8K-16K,
3.061 / 2.773 at 16K-32K and 3.452 / 2.760 at 32K-64K, where the 2.82-bit build falls from 2.980 to 2.809 over the
same bands. Recall on the same run is 100% at every depth and band, and Hermes, Codex and the tool-name probes pass;
the longest of those sessions reached about 24K tokens. Related: the chat-conversation loss entry above, on
different text.

`QwenOraclePrecisionPlan` (`Qwen.PrecisionPlan.ixx:141`) · ContextProfile's loss arm

#### Qwen 3.8's BF16 parity test fails on a longer prompt, though Mila is as close to FP32 as HuggingFace is

`open` · `qwen` · `measured`

`LayerStream_MatchesHuggingFaceReference` bounds Mila's distance from HuggingFace BF16 at 2.0e-2, and on the
28-token prompt (`MILA_QWEN_PARITY_REFERENCE`) layers 43 to 63 fail it. Against an FP32 reference for the same prompt
(`MILA_QWEN_PARITY_TRUTH`), the two BF16 stacks are equally far from it, layer by layer and position by position, to
1,024 tokens (Qwen3.8.md, "The drift past the bound is the prompt's, not Mila's"). The bound measures the prompt.
What is left is a gate that judges Mila against FP32 relative to HuggingFace's own BF16, and the ratio it allows.
The FP32 reference has to exist wherever the test runs.

`Mila/Tests/Dnn/Models/Qwen/QwenModel.Parity.Cuda.cpp` `kHiddenStateTolerance`

#### The Gemma 4 26B-A4B leaves Claude Code about 7,000 tokens of working room on a 16 GB card

`open` · `gemma` · `perf` · `measured`

Claude Code 2.1.300's first request renders to about 20,400 tokens, and the 26B-A4B loads on the RTX 5060 Ti at
27,648 -- under half of the 64K Todd set as the minimum context for an agentic workflow (`Qwen3.8.md` section 5), on
one of the two models Mila calls agentic. Not measured: the 12B's room, or whether the 26B-A4B's context is held back
for a full prefill chunk as Qwen's was.

`Mila/Apps/Server/docs/harnesses.md` *Tested With*

#### The Python binding and Mila's context measurements render Qwen 3.8's prompt differently from its own template in six places

`open` · `qwen` · `apps` · `mila-src`

The template's default effort is `xhigh`, where the free functions call `Medium` the checkpoint's default; it groups
consecutive tool results into one user turn; it separates a turn's second and later calls with `\n`; it writes tool
signatures and non-string arguments through `tojson` (`", "`/`": "`, keys in the order given) where the free functions
write compact, sorted JSON; it trims every message; and with `preserve_thinking` on, its default, it renders past
reasoning back into history. `QwenChatProtocol` renders all six as the template does, and the inference server
and Chat (since `+82`) render through it; the binding's wrappers (`Mila_py.Wrappers.cpp:98`) and
`Tools/ContextProfile` (`ContextProfile.ixx:379`, `:885`) still call the free functions, so a context profile is
measured on a prompt the model's template would not write. Not measured: how far each moves the model's behaviour.

`Mila/Src/Dnn/Models/Qwen/Qwen.Protocol.ixx` `formatPrompt`, `formatTurn`, `serializeToolSignatures`

#### The Python binding and Mila's context measurements cut a Gemma tool result down to one field, where Google's template gives the model the whole text

`open` · `gemma` · `apps` · `mila-src`

The template renders a result given as text under `value:`, whole; `formatToolResponse` parses a JSON envelope and
keeps one output field (added for Codex's envelope, whose chunk id the model echoed). `GemmaChatProtocol` renders it
as the template does, and the inference server and Chat (since `+82`) render through it, so the distillation, for
the one client whose envelope needs it, belongs in the server's wire work; not checked whether the server does it
since it switched. The binding's wrappers (`Mila_py.Wrappers.cpp:147`, `:179`) and `Tools/ContextProfile`
(`ContextProfile.ixx:863`) still distil.

In the same template, a calling turn's text renders after its results and the turn closes with `<turn|>`; because
the last thing rendered was a result, the next reply opens with only `<|channel>thought\n` (or nothing, with
thinking off) and no `<|turn>model`. So a resume whose assistant turn carries both text and calls -- Codex and
Claude Code send that shape -- shows the model its own text again after the results and starts the reply outside
any turn (found 2026-10-20 in a Codex session on the 26B-A4B, thinking on). Mila's port matches the template, and
llama.cpp rendered Gemma prompts byte-identically on 2026-10-19. Not checked: whether Google's own serving renders
it this way, and whether moving the text before the call, or dropping it, changes Gemma's agentic results.

`Mila/Src/Dnn/Models/Gemma/Gemma.Protocol.ixx` `formatToolResponse`

#### Gemma 4 12B's greedy reply changes early with its drafter on the RTX 4070

`open` · `gemma` · `mila-src`

Pinned by UUID, the same build at `+68` (a change that touches no model code) passes on the 5060 Ti and fails on the
4070: drafted and plain greedy tokens part within the first 256, where the test's comment says the reply first parts
at token 402, measured 2026-10-05. Not checked: on which card that was measured, and the token at which they part on
the 4070.

`Mila/Tests/Dnn/Models/Gemma/Gemma.DraftModel.Cuda.cpp:367` `Gemma4_12B_GreedyGenerationIsUnchanged`

### Internal fixes

Defects in the library the user admitted to this release with no ROADMAP criterion behind them.

#### A load can take more of the card than its plan priced

`open` · `models` · `mila-src` · `measured`

Measured 2026-10-03 on the RTX 5060 Ti (`ProfileModel --model qwen --quantization fp4`): from the post-initialization
baseline, Qwen 3.8 27B FP4 consumed 14,962 MiB at context 8192 against a footprint of 14,920; at 16384 the plan left a
26 MiB margin and the card read zero free after the load -- the condition the planner's exactness exists to prevent.
Not yet attributed, and `fp4-cb3`, the Qwen this release publishes, was not checked.

Two device allocations are known to sit outside every price. `CudaExecutionContext` allocates the decode position
(`setDecodePosition`, one `int`) on its first decode step and the prefill key bounds (`key_bounds_`, `:456`) on first
use, each by its own `cudaMalloc`, outside `reserveScratch` and outside every `getMemoryStats` a plan reads; a small
allocation can cost a whole granule. Work: attribute the Qwen gap, measure `fp4-cb3` the same way, and bring
each allocation a load or a decode step makes under the plan's price.

Gate: free memory after a load and the first decode steps is at least the margin the plan states, on Qwen at both
builds and on Gemma 4 12B; `PlanEqualsBuild` covers the context's own allocations.

`Mila/Specifications/ModelFamilyParity.md` 8.3, Q4

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

The declared toolkit is 13.4.3 in the tree; what remains is building and verifying on it. Docker Hub has
carried `nvidia/cuda:13.4.3-{base,runtime,devel}-ubuntu{22,24,26}.04` since 2026-10-17 (read on Docker Hub
2026-10-20), and the local toolkit is 13.4.3 (nvcc 13.4.92). The move was decided for 13.4.1 on 2026-09-24
(Todd) -- the minor is where behaviour and the driver floor move, so it should surface early in the cycle --
and each later patch replaced the one before it rather than adding a second move.

The sites (RELEASING.md, toolkit paragraph): the base images in `Docker/Dockerfile.wheel`,
`Docker/Dockerfile.runtime`, `Docker/Dockerfile` and `build-pipeline.yml` name the patch; `$cudaVersion` in
`scripts/pypi/build-wheel-windows.ps1` names the `v13.4` directory, so the Windows wheel takes whatever 13.4
patch is installed there. The docs name the minor, except `getting-started.md`'s WSL installer URL and
filename, which carry the patch. `RELEASING.md` records what the `0.20.0b3` wheels were built on and stays.

Consequences to carry into the work. Wheel users see nothing — the `nvidia-*` dependencies and
minor version compatibility, and this is now checked rather than assumed: a 13.3-built wheel loads
and generates correctly against the `>=13.0` floor those dependencies declare, on Windows
empirically and on Linux by symbol (`RELEASING.md`, *What the declared toolkit is not*). Moving the
build to 13.4 does not disturb that, but re-check the floor if the cuBLASLt surface grows. Image
users' driver floor rises: the base image's `NVIDIA_REQUIRE_CUDA`
becomes `cuda>=13.4`, and the container toolkit refuses a GeForce driver below it; the driver a 13.x
toolkit needs under minor version compatibility is 580 or later on every patch. `x64-validate`,
`x64-wheel`, `claude-gate` and `x64-claude-validate` are still configured against v13.3 while
`CUDA_PATH` names v13.4 — reconfigure them deliberately. Published tok/s figures and the cuBLASLt
findings in the specs are 13.3 measurements; re-measure them once, just before the release, on the
13.4.3 that ships. CI's first run starts with a cold ccache.

Admitted 2026-10-10 (Todd): the Docker Hub images move to 13.4 for v0.21.0, and with them every site above, as one move.

Gate: every image `publish-image.sh` pushes is built `FROM` `nvidia/cuda:13.4.3-*` and passes `verify-image.sh`; the
wheels are built on 13.4.3 and pass the clean room; no document declares a toolkit other than 13.4 except
`RELEASING.md`'s records of what earlier wheels were built on.

`RELEASING.md`, the toolkit paragraph · `Docker/Dockerfile.runtime:21`

#### Releasing Mila is still a day of manual steps, and the release script stops before anything is published

`open` · `ci` · `build` · `distribution`

`scripts/release/release.py` (`+2`) builds and verifies a release locally from a `git archive` export — the
Windows and Linux wheels, their install tests, both images and their checks, nine stages unattended in about an
hour — and `prepare` / `finish` set the version, merge `dev` into `master` and tag, all in the local repository.
What it does not do is everything a publish is (`RELEASING.md`, steps 1 and 8 to 11), and what has changed since:

- The server's wheel, a fifth file since `+56`, and the CUDA 13.4.3 base images are not in its stages.
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

#### The same installed model loads through one app and is refused through another, because each app decides for itself how to load it

`in progress` · `api` · `distribution` · `apps` · `binding` · `mila-src`

The library installs and stores models, but nothing in it loads one by its store name. Chat
(`Chat.ModelCatalog.ixx:130-260`), the binding (`Mila_py.Wrappers.cpp:296-381`, two parsers) and the inference
server (`model_worker.py:66`) each map a record's `architecture` to a family and its `variant` to a weight format,
and the copies disagree: Chat loads `cb2-3` and `q4_0` where the binding's shared parser refuses both and its Qwen
parser refuses `q4_0`. All of them key on `variant`, which the record calls descriptive, rather than
`weight_quantization`. The quantize-on-load rules exist only in Chat, and so does "will it fit": a context ladder
of Chat's own beside the planner, which it has been seen to disagree with. Root causes: `Deployment.md` 12.6 (the
request's weight format defaults to the package's own) was decided and never built, so every caller must restate
what the weights already are; and 12.4 left the family switch to every caller.

Decided 2026-10-12 (Todd): settled now, in `Mila::Deployment` -- an installed model is looked up and planned by
name through the library (`Deployment.md` 8.1), each app loading it through its family's own type, and an FP32
package is refused, since FP32 is GPT-2's precision and no instruct model has one. Absorbs `Vnext.md`'s "the model list prices against total memory" (12.5).

Gate: no app or the binding parses `architecture` or `variant`; a model loads or is refused identically through
Chat, the binding and the inference server; `/model list` and `mila-store` report the planner's answer; a request
naming no weight format loads every published variant.

`Mila/Specifications/Deployment.md` 8.1 · `Mila/Apps/Chat/Src/Chat.ModelCatalog.ixx`

#### Chat and the inference server live under a directory named for what they are not, and Mila's commands follow three naming styles

`in progress` · `apps` · `build` · `breaking`

`Mila/Adaptors/` held Chat and the inference server, which are applications. The commands a user types are
`mila-chat`, `mila-server`, `mila` and `ExportArtifact`. Decided 2026-10-11 (`MilaCommandLineReview.md` 4.1):

- `Mila/Adaptors/Chat` to `Mila/Apps/Chat`, `Mila/Adaptors/Inference/Server` to `Mila/Apps/Server`, with the CMake
  option, the wheel and image paths, `CLAUDE.md`, and the area tag in `Tags.md` (now `apps`) following.
- The store tool leaves `Mila/Tools/Cli` for `Mila/Apps/Tools/Store` and is renamed `mila-store`: the website's Get
  Started, Chat's shell hints, the image's link and `install` verb, and the image checks.
- Every command a user types is lowercase with hyphens and the `mila-` prefix, the same on every platform.
  `ExportArtifact` keeps its name and place: it does not ship in v0.21.0 (notebook section 4.2).

Gate: a clean configure of every preset, the full suite, the runtime image's checks, and no tracked file naming
`Mila/Adaptors/` or the `mila` command except where it records the move.

`Mila/Specifications/Notebooks/MilaCommandLineReview.md` 4.1 and 6 · `Mila/Apps/`

#### Clang can no longer compile Chat, because every Mila module repeats the standard library's text

`in progress` · `build` · `mila-src` · `api` · `ci`

The runtime image build of 2026-10-12 (`0.21.0-dev+61`, Clang 21.1.8) stops at `mila-chat`'s `main.cpp`: "ran out of
source locations". Clang addresses everything a translation unit reads in a 32-bit space, and `main.cpp` used 99%
of it, 2.13 GB of that loaded from module files, because every module's global module fragment includes the
standard headers again (libstdc++'s `version.h` entered 7,820 times). `Chat.ixx` compiled just under. Every module
added moves the next importer closer, so no image, and no Linux build of the apps, can be made until it is fixed.

The fix is `import std;` in the library: measured on a 60-module probe with CUDA headers in a quarter of them,
the space one importer loads falls from 609 MB to 113 MB, and the remainder is mostly CUDA headers, about 6 MB
per module that includes them (`D:\Claude\import-std-probe`, 2026-10-12). It also clears GitHub #30's two
failures on MSVC, which Microsoft has fixed in a release not yet out. Settled by the probe: modules that keep
CUDA headers in their global module fragment still work, though on MSVC an importer keeps its `#include`s before
`import Mila;`; and CMake's `import std` support is still experimental through 4.4.4, so a library using it forces
every FetchContent consumer to set the gate itself. Decided 2026-10-12 (Todd): Mila builds its own `std` and
`std.compat` modules from the source the toolchain ships, found through the manifest each one publishes, until
CMake's support is no longer experimental; a consumer that sets no gate built and ran on MSVC with CMake 4.0.1 and
4.3.1 and on Clang with 4.2.3, and gets `import std;` through Mila. Clang's floor becomes a standard library that ships a `std`
module (libstdc++ 15, libc++ 17).

Clang's floor also becomes 22 (decided 2026-10-12, Todd). Clang 21 rejects libstdc++'s `shared_ptr` friends with
"cannot befriend target of using declaration" in a module that instantiates one while a textual libstdc++ header and
the `std` module are both reachable (LLVM #156604, #138558), which every module holding a `Tensor` member meets; the
fix landed after 21 branched, and 22.1.2 builds `mila-chat`. Measured on Clang 22 with every module
converted: `mila-chat`'s `main.cpp` uses 0.475 GB of the 2.15 GB space, from over it, which Clang 22 alone leaves
unchanged. Every module interface imports `std` now; left are the test and sample translation units, which keep
their includes before `import Mila;`, and the gate's own checks.

Checked 2026-10-13: `QuickStart/Cpp` builds through FetchContent with no `import std` setting and without its
`<sstream>` workaround on MSVC (and runs) and on Clang 22; the published sample keeps the workaround while it pins
`v0.20.0`, and loses it when the release moves the tag. A consumer that also turns on CMake's own `import std` builds
and runs on both compilers: each target compiles against its own `std`. GitHub #30's workarounds in the binding and
the NgramEmbedding test are deleted. Both images build on Clang 22 and generate on a model this tree reads; their
scripted checks fail only on the published packages, which this tree refuses ("A model installed under v0.20 stops
loading when Mila is upgraded"), and pass once that lands. The Linux wheel stays on Ubuntu 24.04 with gcc-15's
libstdc++ from the Ubuntu toolchain PPA (decided 2026-10-13, Todd), measures `manylinux_2_38` as before, and gains a
cp314 wheel for Ubuntu 26.04's Python, which runs end to end there. Mila uses `import std;` alone, with the
standard library's C names qualified (Todd, 2026-10-13).

Work, each step leaving the tree building on MSVC and Clang:

1. The `std` module target in the tree, as the probe built it.
2. The plumbing in the tree and one converted module; a WSL Clang build of `mila-chat` reports its source
   location use (`-Xclang -print-stats`), and that number is the measure for every step after.
3. The modules in batches, by directory, the heaviest importers first, until `mila-chat` and the images build.
4. The rest of the modules, Chat, `mila-store`, the binding, the tests and the samples, then GitHub #30's
   workarounds deleted where the conversion removes their cause.

Gate: both images build and pass their checks; every target builds on MSVC and Clang with the full suite
passing; `QuickStart/Cpp` builds through FetchContent with no `import std` setting of its own; `CLAUDE.md`,
`getting-started.md` and the images state the new Clang floor.

`Mila/CMakeLists.txt` · GitHub #30

#### A 64-bit sampling seed keeps only its low 32 bits

`open` · `mila-src`

`reseed( std::uint64_t )` casts to `std::mt19937::result_type` and `mt19937::seed` keeps 32 bits, so seeds 2^32
apart -- and OpenAI's -1 and 4294967295 -- give the same stream while the API says each is a different seed. A
`std::seed_seq` of both halves would keep all 64.

`Mila/Src/Dnn/Samplers/TokenSampler.ixx` `reseed`

#### A greedy reply through the inference server can depend on the request before it

`open` · `mila-src` · `binding` · `apps`

Reusing a cached prompt prefix rounds a prompt differently from a whole prefill (Llama 3.2 3B BF16 departs at the
tenth sampled token), so since 2026-10-18 a seeded call prefills whole. An unseeded greedy request still reuses, so
at a near-tie its reply depends on what the server ran before -- the case an engine comparison at greedy through the
server is in. Not checked: how often greedy flips on real prompts, and whether greedy calls should also prefill
whole.

`Mila/Bindings/Mila_py.Wrappers.cpp` (reuse off only with a seed) · `Mila/Specifications/PromptCaching.md` section 5

#### Library doc comments still describe the retired operation registry and the removed operation bases

`open` · `docs` · `mila-src`

`Mila.ixx:108` records the runtime `OperationRegistry` as retired, and no `UnaryOperation`, `BinaryOperation` or
`PairedOperation` class exists, but the Doxygen of Residual, Softmax, Lpe, SoftmaxCrossEntropy, Gelu and others still
says they obtain a backend from the registry or derive from those bases; a grep counts 28 mentions of the bases.
This is the bounded part of `Vnext.md`'s "Public Doxygen still describes a world that was refactored away".

`Connections/Residual.ixx:8` · `Normalization/Softmax.ixx:44` · `Encodings/Lpe/Lpe.ixx:53` ·
`Losses/SoftmaxCrossEntropy.ixx:11`

#### Code retired under the old keep-it-in-the-tree rule still sits in the tree

`open` · `mila-src` · `build`

Retired code is deleted and git history is the archive (Todd, 2026-10-11). Wholly retired files, out of the build:
`Gqa.Decode.Bf16.cu`, `Gqa.Flash.Wmma.cu`, `Gqa.Flash.Bf16.cu`, `CudaRopeOp.Cache.ixx`, `MLP.Dispatch.ixx`,
Chat's `Chat.GemmaToolCallParser.ixx` and `Chat.QwenProtocol.ixx`, and the tests `CudaRopeOp.Calculated.Cuda.cpp` and `Gemma.DelegatedFeedForward.Cuda.cpp`, with
their commented-out entries in `Mila/CMakeLists.txt:276` and `Mila/Tests/CMakeLists.txt:253`. Retired functions
inside live modules: `Tools/Quantization/pack_qwen.py:162` and `artifact.py:181`. Specs that cite a retired file as a
reference are reread when it goes.

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/`

#### A consumer build on Clang warns that a random-fill operation uses a cudaMalloc local to another translation unit

`open` · `mila-src` · `build`

Building QuickStart inside the devel image with Clang 22 (2026-10-13) gives four
`-Wreference-tu-local-entity-in-other-tu` warnings, from `fill_uniform` and `fill_normal` instantiating
`cudaMalloc<float>`, the CUDA runtime header's template, which the module's global fragment declares. Not checked:
whether Clang 21 warned too, and whether it is only a warning or a real ODR hazard.

`Mila/Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Random.ixx:266`

#### CI never compiles the CUDA build, so Clang portability defects in the code users run go unseen

`open` · `ci` · `build`

The CPU-only job compiles a different module graph -- the CUDA modules are absent -- so a missing `#include` that
MSVC resolves transitively in CUDA-side or model code reaches Linux only when someone builds the GPU configuration
there. nvcc and Clang compile CUDA code on a runner with no GPU, so a job that configures `MILA_ENABLE_CUDA=ON` and
builds without running catches that class in the code users get. It needs the CUDA toolkit on the runner; the dev
image already carries it (`Docker/Dockerfile`).

`.github/workflows/build-pipeline.yml:198`

#### Three Gemma 4 26B-A4B tests fail instead of skipping when its BF16 weights are a partial file

`open` · `gemma`

Found 2026-10-20 in the full suite: `Fp4Load_...`, `Q4_0Load_...` and `GemmaMixtureOfExpertsParityCudaTests.
LayerStream_MatchesHuggingFaceReference` fail in milliseconds with "Failed to read rank of 'tf_layer_24...'".
`Data/Models/Gemma/gemma4_26b_a4b_it_qat_bf16.bin` is 32,734 bytes, written 2026-10-19 22:44 by a 26B conversion
that crashed (0xC0000005 in 5 s, empty log, `D:\Claude\gemma26b-gates\convert-26b.log`) while the Gemma BF16
checkpoints were being restored; the 12B's re-conversion succeeded. The tests decide to run on the file existing,
so a partial file turns a skip into a failure. Not known: why the converter crashed (the HF cache holds the 49 GB
QAT checkpoint). Admitted 2026-10-20 (Todd), out of `Untriaged.md`.

Work: the 26B's BF16 weights reconverted; the three tests skip, naming the reason, when the file is not a complete
weights file; the Gemma converter writes beside its destination and renames on success, so a converter that dies
leaves no file.

Gate: the full suite passes with the 26B's BF16 weights absent, partial and complete, and a conversion killed part
way leaves no file at its destination.

`Mila/Tests/Dnn/Models/Gemma/GemmaModel.MixtureOfExperts.Load.Cuda.cpp:76` · `Mila/Tools/Converters/Gemma/`

#### The inference server logs a client's dropped connection on Windows as an error with a traceback

`open` · `apps`

Found 2026-10-19 in the Codex session against Qwen 3.8 `fp4-cb3`: between two requests asyncio logged `ERROR
Exception in callback _ProactorBasePipeTransport._call_connection_lost()` with a `ConnectionResetError` [WinError
10054] traceback, and the next request was served normally; Codex's five turns all completed. It is the Windows
proactor loop reporting a peer that reset an idle keep-alive connection. Not checked: whether uvicorn's own loop
setting or a handler for that one exception removes it, and whether it appears on Linux. Admitted 2026-10-20 (Todd),
out of `Untriaged.md`.

Gate: a client that resets an idle keep-alive connection leaves no error and no traceback in the server's log, and
an exception in a request still logs one.

`Mila/Apps/Server/src/mila_llm_server/app.py` (event loop)
