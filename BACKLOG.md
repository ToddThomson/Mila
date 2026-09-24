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
accommodate the list. Gemma 4 image input is the first item to drain. A backlog that grows while the
date holds is the signal to drain it early.

**Done means deleted**, in the same commit as the work — `done` is a working-tree marker and is
never committed.

---

## Current release (v0.21.0)

The whole arc, implemented in five stages in the order ROADMAP gives them: the contracts, the planner
and the handle, `Mila::AI`, the applications, then the families finished. The buckets below follow
the themes, not the stages. The stages are a sequence, not a partition — an item is workable
whenever its own blockers are gone.

### Model Handle

#### Three places turn a model's name into the type that loads it, each its own way

`open` · `ai` · `architecture` · `api`

The architecture-to-concrete erasure exists three times in two languages — Chat's `ModelVariant`
(`Chat.ixx:72`, eleven `std::visit` sites), the binding's per-family session classes, and the
inference server's `ModelFamily` enum (`model_worker.py:38`). Each consumer also writes its own
bridge from the manifest's architecture string to a family (`familyFromArchitecture` in
`Chat.ModelCatalog.ixx`, `architecture == "gemma"` in `Mila_py.Wrappers.cpp`).

One handle and one factory in the new `Mila/AI/` library (`Direction.md` §3.2, §8 decision 2). The
architecture's *identity* — the set of names and the concrete type each resolves to — is the
library's and lives in `Mila/Src` beside the manifest reader; the handle is its one consumer. The
factory takes a deployment request rather than a device, so it is built on the planner from the
start. A network composed from components meets the handle through a C++ concept, with no
registration (§8 decision 7).

Gate: adding an architecture is an edit in one place, and decode throughput through the handle is
unchanged against the direct type.

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

### Deployment Planning

#### Automatic context length is decided inside Chat, where no other caller can reach it

`done` · `models` · `mila-src` · `breaking`

`resolveAutomaticContext` (`Chat.Footprint.ixx:448`) and the fit grading in `Chat.ModelCatalog.ixx`
are the only automatic deployment decisions Mila makes, and they belong to one application.

`Deployment.md` Phase 3: `planDeployment` on one device, returning ranked plans with a limit per value,
or a refusal with its reason (§3.3, §6). Record Chat's choices first (G2 — Gemma 4 12B
FP4, Llama 3.1 8B FP4, Qwen 3.8 27B FP4 and cb2-3, each card pinned by UUID, with the free memory
each run saw), then move the rule and delete Chat's copy. **Breaking:** an explicit context length
that does not fit is refused naming the reason, where Chat today warns and tries (§12.2);
`--help` and `ChatConfiguration.md` §6 change with it. Exit: G2, G4, N2 and N4.

#### A Python or inference server user has to know the context length in advance

`open` · `binding` · `adaptors`

The binding and the server pass explicit values and have no automatic choice at all
(`Deployment.md` §1). Phase 4: `from_store( name, context_length="auto" )`, the server's `config.py`
taking `auto` and passing it through, and the QuickStart samples and `getting-started.md` updated
with it. Exit: a Python QuickStart session and a server request each run with `auto`.

### Mila::AI

#### Running a model from a program means naming its concrete C++ type

`open` · `ai` · `api`

Today the entry point is `GemmaModel<Cuda, BF16>::load` or its siblings, so a program that wants to
change models changes types, and the tool loop, streaming and conversation state are the program's
to write. The finding is an absence: the application-facing object does not exist.

`Mila::AI` in module `Mila.AI` (`Direction.md` §4, §8 decision 1), a `Mila/AI/` library target that
the wheel and a `FetchContent` consumer both receive. The contract is `Direction.md` §4.2's seven
rules: small; never decides a deployment; descent through `plan()` and `model()` is part of the
contract; one virtual call per `respond`; other languages project it; a composed model is
first-class; a model is named by its store name. Gate: the ten-line program runs Gemma, Llama and
Qwen by changing only the name, and a sample creates an `AI` over a network it composed itself.

#### The tool loop exists only inside Chat

`open` · `ai` · `adaptors`

Parse a call, dispatch it, return the result, continue — written once, informally, in `Chat.ixx`,
so a developer's program that wants tools rewrites it. It moves into `Mila/AI/` as the agent core
(`MilaProductFamily.md`, Native Agent Core), and Chat keeps only its human approval gate. Tools are
compiled-in functions registered on the `AI`, a callable plus a schema (`Direction.md` §8 decision
6). The autonomy policy is not part of this.

#### A tool result is re-rendered and re-tokenized with the whole conversation before generation continues

`open` · `ai` · `models` · `mila-src`

Decided in v0.20 and deferred (`MilaProductFamily.md` Decided 1): a tool result's tokens are appended
to the live KV cache, with no re-render of the conversation and no re-tokenize. Today Gemma recovers
the prefill through transparent prefix reuse (`GemmaModel.ixx:364`) but still renders and tokenizes
everything; Qwen prefills from 0 every turn (`QwenModel.ixx:355`). The grammar that frames a tool
result as tokens is model-intrinsic and belongs in `Mila/Src`; the loop that decides when to splice is
the agent core's.

Gate: across a multi-turn tool session, prefill tokens per turn equal the tokens the turn added,
measured, for every family that permits prefix reuse.

#### Qwen refuses prompt-prefix reuse and nothing reports it

`open` · `qwen` · `ai` · `mila-src`

`QwenDeltaNetBlock::rewindKvCache` always returns false — correctly, since a recurrent state is a
lossy summary and cannot be rewound — and `QwenTransformer::rewindKvCache` ANDs that into a refusal
for the whole stack. The per-block mechanism exists (`snapshotState`/`restoreState`); a whole-model
statement of the property does not. It was inert while prefix reuse lived inside each model's
`generate`, and becomes live now that the agent core manages the splice itself: the core reads the
refusal from the handle and reports it, rather than meeting it as a failed rewind.

### Applications

#### The inference server chooses each model's loader and grammar by a family enum of its own

`open` · `adaptors` · `binding`

`model_worker.py` maps `ModelFamily` to a session class (`:38-41`), branches on it for stop markers
(`:56`, `:59`) and for tool support (`:168`), and `/v1/models` does the same. Three latent `else means
llama` sites in this shape were fixed at `rc.1+31`, each correct only while there were exactly two
families. It serves the same three families Chat runs today, so the gap is the second copy of the
erasure rather than a model it refuses.

GPT-2 is not in it, and not by oversight here: it is a base model, Chat refuses base models by
decision, and the binding has no GPT-2 session (`model_worker.py:71`). Whether the server serves a
base model is not this release's question. Gate: the enum is gone and the server reads what it needs
from the handle.

#### The Python binding carries one session class per family

`open` · `binding` · `api`

The per-family session types under `Mila/Bindings/` are the second of the three bridges, and the one
a Python consumer actually meets. They are replaced by `mila.AI`, projecting the same contract and the
same plan, without the binding gaining a component-level surface — it is consumer-blind by design and
stays that way.

The gate is that adding a family adds no binding type. The finding is a duplication rather than a
defect at one line, so it has no single anchor.

#### Chat and the inference server each hold code that knows which model they are running

`open` · `adaptors` · `build` · `breaking`

Rebuilt as consumers of `Mila::AI`, reaching anything beneath it only through `model()`: Chat keeps
terminal rendering and the approval gate, the server keeps the wire shapes and per-request
statelessness. Both leave `Mila/Adaptors/` for `Mila/Applications/Chat` and
`Mila/Applications/Server` (`Direction.md` §8 decision 4), with the CMake option, the wheel and image
paths, and the `adaptors` tag in `Tags.md` following. Gate: neither contains model-specific code, and
the Codex CLI and Claude Code CLI tool flows still pass unchanged.

#### MIS tool calling beyond the three flows the release names

`open` · `gemma` · `adaptors`

N sequential distinct tool calls within one turn, and channel-content parser polish. Moved from the
v0.20 backlog at `rc.1+21`: the release criterion names plain-chat, single-tool and
tool-result-resume only.

### A Developer Can Start

#### Neither QuickStart shows a program putting a model to work

`open` · `docs` · `binding`

`Samples/QuickStart/Cpp` loads a typed model and `Samples/QuickStart/Python` opens a per-family
session; neither creates an object, calls a tool or shows what was decided about the deployment. Both
are published surfaces the website's Get Started tabs link to.

The C++ one becomes a `FetchContent` project that creates an `AI`, calls a tool and prints the plan;
the Python one moves to `mila.AI`. Each keeps a path that builds a network from components
(`Direction.md` §7). Gate: both run from a clean machine with only the documented prerequisites.

#### Every public surface describes Mila as a reference implementation first

`open` · `docs`

`README.md`, the website under `Web/` and `getting-started.md` lead with the v0.20 message, and
several use "adaptor". They move to the trait — a library for developers to harness intelligence — in
this release and not before, which means written against what the tag actually ships
(`Direction.md` §5.7; `.internal/Marketing/Positioning.md` is superseded by it). Gate: no public
surface describes Mila as a reference implementation first, or uses "adaptor".

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

`open` · `qwen` · `quantization` · `mila-src`

`PerChannelKvFp8<>` is specified and not built; `QuantizationDispatch.ixx:103` records that KV-cache
compression is not a live knob. On a 27B at FP4 the weights take the card, so the KV cache is what
buys context back, and halving it is the difference between the context length that fits and the one
the model is sold at.

The freed margin is then a decision rather than a windfall — more context, or more bits where the
quality gate says they are worth most. v0.20 deliberately did not pre-empt that, so it is part of
this work rather than a consequence of it. The compressed cache has to be priced exactly by
`getRequiredMemory`, since the planner reads that price, and `Deployment.md` §2's "KV-cache
compression: not a knob" row changes with it.

#### Qwen answers in one block after a long silence

`open` · `qwen` · `adaptors`

`FamilyTraits::streaming_capable` is false for Qwen (`Chat.FamilyTraits.ixx`), because the harness
routes tokens by Gemma's four control-token ids and nothing else has them. Qwen has one marker pair,
`<think>`/`</think>`, which is enough to separate reasoning from answer; the per-token router has
simply not been written for it.

Not a gap against any other model — Llama and GPT-2 are buffered too, and Gemma is the only family
that streams. It matters most on Qwen because a 27B is the longest wait to sit through with nothing
on screen.

#### The smaller dense Qwen members were never built

`open` · `qwen` · `mila-src`

The family shipped as the 3.8-27B hybrid alone, so "Qwen 3.8" names one model rather than a family.
The dense members reuse the Llama blocks rather than the DeltaNet chassis, which is what makes them
cheap next to everything already landed. Built after the handle, it is the first model added through
it, and so the first test of "a new architecture is added in one place".

The finding is an absence — there is no partial implementation to point at. Gate: a dense member
decodes token-for-token against HuggingFace at BF16 and FP8.

#### Qwen's perplexity gate has only been run to 16K

`open` · `qwen` · `measured`

From 8K to 16K the FP4 oracle improves 7.2% while the 2.82-bit plan improves only 3.4%, so the
quantized arm captures about half the benefit of the extra context — the compounding signature the
recurrent layers make plausible. It was held out of v0.20 because nothing claimed a context above
16K: the model card stopped there and recorded the ratio as flat from 1K.

**The condition it was waiting on has fired.** KV-cache compression above buys context back, and a
release that advertises a longer one has to gate quality at that length rather than at the length
the old card stopped at. The table and caveats are in `Qwen3.8.md` §8 item 9;
`DISABLED_QualityGateAcrossContextLengths` is the harness.

#### The head's two paths disagree in the third decimal, so perplexity must fix the width

`open` · `qwen` · `measured`

Same weights, same corpus: width 1 (the decode matvec) and width 64 (the W4A8-FP8 GEMM) do not
produce identical numbers. Small, but head width is part of the measurement protocol rather than a
free performance knob, so both arms of a quantization comparison have to use the same one.

Probably already recorded at `Qwen3.8.md:509` and `:546` — verify, and if so this entry is a
duplicate and should be deleted rather than worked.

#### A Qwen load test discards a `[[nodiscard]]` status and warns on every build

`open` · `qwen` · `ci`

`QwenModel.Load.Cuda.cpp:205` calls `model->generate(...)` for its side effects inside a lambda,
producing C4834. The status is the only channel reporting why generation stopped, so a test that
ignores it cannot tell a completed run from an aborted one — and the other call sites in the same
file (`:168`, `:523`, `:813`) already bind it. The build at `+3` reports three more sites of the same
warning: `QwenModel.Load.Cuda.cpp:1051`, `:1259` and `Tests/Common/GenerationRates.h:67`.

Assert it instead of casting it away. Also one entry on the warnings-as-errors ratchet's bill.

### Gemma 4 Complete

#### Two questions decide the size of Gemma's image path, and neither is answered

`open` · `gemma` · `mila-src`

Answer both before building, because one of them reaches every attention path in the library. This
is first-stage work — reading the HuggingFace implementation, not building — and its answers decide
whether image input stays in the release, since it is the first item to drain.

**Do image soft tokens attend bidirectionally within a prefill?** Every attention path in Mila is
causal. If a bidirectional span is required, that is a mask change across the attention machinery
rather than a component added beside it, and it resizes everything downstream of it.

**What is the position scheme for an image span?** 280 soft tokens arrive at `model_patch_size` 48
inside a sequence whose positions are otherwise one per token.

The finding is an absence: no code answers either question today, and the entry below cannot be
honestly sized until they do. Whatever the answers are, they are recorded in `Gemma4.md` rather than
in the implementation that assumes them.

#### Gemma 4 12B is multimodal and Mila drops its image and audio weights

`open` · `gemma` · `mila-src` · `adaptors`

The 12B is **encoder-free** (`Gemma4UnifiedForConditionalGeneration`): no vision tower. Images enter
through `vision_embedder` (`patch_dense`, `patch_ln1`, `patch_ln2`, `pos_norm`) and
`embed_vision.embedding_projection` into the decoder itself — `patch_size` 16, `model_patch_size` 48,
280 soft tokens, `mm_embed_dim` 3840; audio through `embed_audio` (`audio_embed_dim` 640).
`convert_weights.py:116` skips `model.embed_vision.` and `model.embed_audio.`. Open before sizing:
whether image soft tokens attend bidirectionally within a prefill (every Mila attention path is causal),
the position scheme for image spans, and the audio front end. Work: patch embedding component, soft-token
placement before layer 0, image decode/resize/normalize (a vendored decoder is a NOTICE entry; decode
belongs in the applications), template image tokens, Chat attach, MIS image content blocks for both
protocols, converter keeps the embedders, manifest declares modality, footprint counts image prefill.
Image input reaches applications through `Mila::AI`, not around it. Gate: embedder parity against
HuggingFace, then token-for-token on an image prompt.

#### Announce the Gemma 4 26B-A4B mixture of experts

`open` · `gemma` · `distribution`

Landed in `Mila/Src` during rc.1 — the router and expert bank on CPU and CUDA, wired into
`GemmaModel` with an FP4 expert bank and a streaming converter, gated against HuggingFace at BF16
and FP4. It appears in no README capability row, no CLAUDE.md target and no release note, because
no published package uses it, so a user cannot run it.

Held out of v0.20.0 deliberately (Todd, 2026-09-21) on the same rule as the entry above. What it
needs to become announceable is a published package and a capability row, not more implementation.
Expect this to be rediscovered by anyone grepping the tree for MoE and wondering why it is silent.

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

#### The Gemma parity script compares two different precisions and calls it parity

`open` · `gemma`

`gemma_greedy_parity.py:70` loads Mila through the binding's FP4 default and diffs it against a
BF16 HuggingFace reference, so any divergence it reports mixes quantization error with a real
defect and a clean run proves less than it appears to. `GemmaModel.load` now takes `quantization=`,
so the honest comparison is one argument away — on a card that can hold a BF16 12B. Either way the
script should state which precision it ran.

Full path: `Mila/Tools/Converters/Gemma/gemma_4_BF16/gemma_greedy_parity.py`.

#### `gemma_protocol.py` is dead and can be deleted

`open` · `gemma` · `binding`

Its 856 lines are superseded by `Gemma.Protocol.ixx` plus `gemma_bridge.py`, nothing imports it, and
it carries a header saying so. Kept on disk under the retire-don't-delete rule, which is the correct
state for now; removing it is a one-file deletion whenever the reconciled grammar has been driven
long enough to be sure.

#### Nobody knows whether Google's drafter would make Gemma 4 12B decode faster

`open` · `gemma` · `perf` · `measured`

Every Gemma 4 size ships a dedicated draft model for speculative decoding (ai.google.dev/gemma/docs/core,
read 2026-09-17). FP4 decode is bandwidth-bound and a verify goes through the prefill GEMM, so the
question is a measurement: what a K-token verify forward costs against K decodes on the 5060 Ti with
today's prefill path. If K=4 costs near 4 decodes there is no win, and the recorded result is "not
worth doing".

This release is the measurement only. The loop it would justify — draft/verify/accept/rewind, the
drafter's KV cache, wrap-safe rewind on the sliding ring — is in `Vnext.md`, and `SpeculativeDecoding.md`
is the draft design.

#### Nobody knows whether Google's quantization-aware 4-bit Gemma 4 beats Mila's FP4

`open` · `gemma` · `quantization` · `measured`

`google/gemma-4-12b-it-qat-w4a16-ct` is int4, symmetric, group 32 (compressed-tensors, read
2026-09-17), where Mila's FP4 is E2M1 at group 128 — so re-quantizing it onto Mila's grid discards
the training that fitted it. The question is wikitext perplexity of the QAT checkpoint, run in
HuggingFace, against Mila's published FP4 and BF16. If QAT does not beat FP4, stop, and the recorded
result is "not worth doing".

This release is the measurement only. The `PerGroupInt4<32>` policy and the compressed-tensors import
it would need are in `Vnext.md`.

#### A parity script tells the reader to diff against a debug flag that no longer exists

`open` · `gemma` · `docs` · `observability`

`kGemmaDumpActivations` is gone from `Mila/Src`, but
`Mila/Tools/Converters/Gemma/gemma_4_BF16/hf_gemma_activation_dump.py:4` still names it as the
thing to compare its output with. Anyone following the script for a parity investigation starts by
looking for a flag that is not there.

The replacement is `LanguageModel::observe` over `"*.tf_layer_*"`.
`GemmaModel::fingerprintPrefill` is **not** the substitute — it localizes a NaN rather than
comparing per-layer activations.
