# Mila — Roadmap

Where Mila is going — the durable narrative of each release and what it means.

- **Open tasks** -> [BACKLOG.md](BACKLOG.md) · **Completed work** -> the
  [release notes](https://github.com/ToddThomson/Mila/releases)
- **How versions, branches, and releases work** -> [RELEASING.md](RELEASING.md)
- **Design rationale** -> `Mila/Specifications/`

The roadmap shows the **release in flight**, plus a **Future** tail. A release is reached through the
**themed workstreams** below; their tasks live in [BACKLOG.md](BACKLOG.md).

---

## v0.21.0 — The Families, Finished and Measured

**Release Date:** _Target — 2026-11-20_

**The release makes one claim: every model Mila publishes is finished, and you can see how good it
is.** Gemma 4, Qwen 3.8 and Llama 3 each run every capability their checkpoints carry that the
release names, at the rates and the quality a reader can check against the original model and
against llama.cpp, from packages that install with `pip` and keep loading when Mila is upgraded.

**Finished means what v0.20 left over.** Qwen 3.8 and Gemma 4 shipped with parts missing — a 2.82-bit
build nobody can fetch, a KV cache left uncompressed on a card the weights already fill, a checkpoint
whose image and audio weights the loader drops, a 26B-A4B that runs at a fraction of llama.cpp's rates
and cannot load the weights Google trained it for. Most of that work is built or under way, and this
release ships it rather than holding it for the API that will sit on top of it.

**Measured means every family, by the same tools.** Each published model is scored on standard
benchmarks against its original BF16 weights under one public harness, read against the noise floor
between two correct BF16 engines (`ModelEval.md`), and profiled for how deep into a conversation it
still finds a fact, obeys its instructions and calls a tool (`ContextProfile.md`). A quantized build is
reported beside the original and never alone. Evaluation is part of each family's own work below, not
a theme of its own: a family is not finished until it is measured.

**On the 16 GB card the main effort is Gemma 4** (Todd, 2026-10-04), the family that can spend that
card's budget on context, modality and a draft model at once. Qwen 3.8 finishes what every family
shares, and its own tuning waits for a 24 or 32 GB card (`SPONSORING.md`).

**`Direction.md` section 5 is the next release, v0.22.0** — the model handle, the deployment planner's
request, `Mila::AI`, and Chat, the inference server and the binding rebuilt on it (**Future**, below).
It was this release until 2026-10-10, when it was split out (Todd): the family work had run ahead of the
API, and a release of finished, measured models need not wait for it. The applications therefore move
once, in v0.22.0, as that plan intended; what they gain here is what survives that move — a published
server, correct replies, and defaults read from the model.

**The date is fixed and the scope drains.** What has not landed by 2026-11-20 returns to
[`Vnext.md`](Mila/Issues/Vnext.md) and the release ships with what it has, its claim stated as
narrowed. **The 26B-A4B's vision tower is tentative and drains first**; the 12B's image and audio are
committed.

Pre-1.0 still holds: breaking changes to the API are acceptable. **A model a user installed is not one
of them.** From this release on, upgrading Mila by one minor never stops an installed model loading,
and an older Mila can still install a build of a model that it can run. So every format this release's
packages carry — the six-bit tied table, the drafter's place in a package, the manifest's new fields —
is one v0.22.0 must read, and each is decided before it ships.

### Gemma 4 Complete

*The family Mila validates most closely, finished — including the half of the checkpoint the loader
currently discards.*

Gemma 4 is the family with the most Mila behind it and the most left on the table. The 26B-A4B
mixture of experts **runs** — router, expert bank, FP4 expert bank and streaming converter, gated
against HuggingFace at BF16 and FP4 — but as a correctness baseline: it reaches 8192 tokens on the 16 GB
card only by prefilling 256 tokens at a time, cannot load the Q4_0 weights Google trained it for, and
prefills and generates at a small fraction of llama.cpp's rates on the same card. **It ships level with every other model Mila publishes, or not at all** —
the trained format, the reference card, rates beside llama.cpp's, quality across the planner's range,
and every application. Two protocol defects survive into multi-step tool use: a model working through a
sequence loses the reasoning that led to each step, and a malformed call parses as a call with no
arguments rather than as prose, which a client that executes `{}` would act on.

**Modality is the largest single item in the release, and the two models differ in it.** The 12B is a
unified transformer and encoder-free — no vision tower — so images enter through `vision_embedder` and
an embedding projection directly into the decoder at up to 280 soft tokens, and audio through
`embed_audio`, a single projection of raw 40 ms frames rather than an encoder. The converter skips both
today. Image tokens take ordinary positions, but the tokens of one picture attend each other in both
directions, while every attention path in Mila is causal: that mask change, in every prefill path, is the
work (`Gemma4Modality.md`). Chat accepts both an image and an audio clip. **The 26B-A4B is a mixture of
experts with a vision tower** — 27 layers, about 411 million parameters, about 410 MB at FP8 — which its
loader skips today (`Gemma4MoE.md`), and on the 16 GB card those bytes are context: about 39K tokens of the
26B's cache. Everything the 12B's image path builds carries over, leaving the tower as the new piece. **The
12B's image and audio are committed; the 26B-A4B's tower is tentative**, the first work to drain if the
date presses.

**The 4-bit 12B has to hold its quality at the context lengths agentic work runs at, and today it
does not.** Mila's FP4 package predicts text worse the more of it the model has read -- about twice
BF16's perplexity by 32K -- and the loss is in the weights, not the implementation. Google's
quantization-aware weights, run in the Q4_0 format they were trained for, stay within 0.03 nats per
token of BF16 through the same range. So the 12B runs those weights in that format: a model runs in
the format it was trained for, and Mila's own formats are the fallback for models whose producer ships
no quantization-aware release.

**Google's drafters ship in the packages.** Each Gemma 4 model's draft model was measured against a
speculative loop's cost before anything was built, as this release's narrative required, and the loop
pays on both: about 2x on the 12B's chat and code and 1.2 to 1.4x on the 26B-A4B (`Gemma4Mtp.md` 4.7),
the drafter's own format in a package being that spec's decision 2. So every Gemma 4 package carries its drafter, a load turns it on with its measured
draft length wherever the planned context still fits with it resident, and Chat turns it off on request.
Choosing it, or any other feature, in a deployment request is v0.22.0's.

**Gemma's memory is spent on what a use case needs.** Two levers are Gemma's own. Its global layers
are trained with keys equal to values (K = V), and Mila caches both; storing them once returns about
half the global cache, which on the 26B-A4B is the difference between the 64K it fits today and the
96K its context profile measures it reliable to (recall alone; 128K loses the middle of the
conversation, `ContextProfile.md` section 8), or room for its vision tower at a useful context. And its
sliding layers keep a ring of the last window, so a reply longer than about a thousand tokens -- any
reasoning reply -- leaves nothing to rewind to, and the next turn pays a full prefill: about 100 s at
128K on the 12B. Saving the ring's last window when a position is saved, as Qwen saves its recurrent
state, removes that.

**Finished also means level with the other families** (`ModelFamilyParity.md`). Gemma is the
furthest along of the three, and its gaps are few but one of them is sharp: it cannot score a text,
so its quality above 131072 tokens has never been measured, while the planner may now choose up to
the 262144 its weights declare. And Chat still renders Gemma's prompt with its own copy of a
template the library already has, so the two can drift. Both models' BF16 weights exceed the 16 GB
card, so where their benchmark reference runs is `ModelEval.md` decision 1, as it is for Qwen.

**Success criteria:** Gemma 4 12B accepts an image and answers about it, with the embedder gated
against HuggingFace and then token-for-token on a mixed prompt; Gemma 4 12B accepts audio, gated against
HuggingFace as its image path is, and Chat accepts both an image and an audio clip; modality is declared
in the manifest; the 26B-A4B is published in Google's
quantization-aware Q4_0 and holds every parity row the 12B holds (`ModelFamilyParity.md` §3.6) — it
runs 8192 tokens on the 16 GB card with the prefill chunk the 12B plans there, its prefill and generation rates stand beside llama.cpp's on
Google's GGUF, its quality is measured across the planner's range, and Chat, the inference server and
the Python binding run it with tool calls; reasoning survives across tool calls within a turn and a malformed call is refused
rather than executed as empty; Gemma 4 12B runs Google's quantization-aware weights in Q4_0, and its
cost over BF16 is measured at every context length the planner can choose for it; each Gemma 4 package
carries its drafter, a load decodes with it token-for-token as it does without it, faster by the measured
amount, wherever the planned context fits with it resident, and Chat can turn it off; Gemma's quality is
measured at every context length the planner can choose for it, by a measurement that lives in the harness
rather than in the public API; and Chat renders Gemma's prompt with the library's template, not its own;
the global layers store each key once, gated equal to storing both on G2's books and on
recall at depth, and the 26B-A4B fits its measured reliable depth on the 16 GB card; a turn that
follows a reply of any length resumes the cached conversation without a full prefill, measured on the
12B at 128K; each published Gemma 4 build states its benchmark scores beside the original's, from the
same harness, read against the floor Llama 3.2 3B measures, and has a context profile on the 16 GB card;
and, tentatively, the 26B-A4B accepts an image through its tower, at a context the plan states.

### Qwen 3.8 Complete

*Everything the family was specced to be, including the parts v0.20 shipped around.*

v0.20 promoted Qwen 3.8 out of a research track and into the release, and the promotion was earned —
but it shipped the model rather than the family. Two things were left: the 2.82-bit build is finished,
validated and **unpublished**, so the claim the release was written around had to be narrowed to the FP4
build at 16 GB; and the KV cache is uncompressed on a card the weights already fill, which is what bounds
the context the model is actually sold at. The dense members, which reuse Llama blocks rather than the
DeltaNet chassis, are new building rather than finishing, and wait for v0.22.0, where a member arriving
after the handle is the first test of the claim that a new model is added in one place.

The user-visible one is smaller and worse: a 27B answers in a single block after a long silence,
because the per-token router was written against Gemma's four control tokens and Qwen's
`<think>`/`</think>` pair was never wired to it. It is the longest wait in Mila with nothing on
screen.

**This release finishes what Qwen shares with the other families, not its tuning.** The 2.82-bit
build's prefill and generation are measurably less efficient than they could be (`ModelFamilyParity.md`
8.3, Q9), and the FP4 build predicts a book worse with more of it in context from 8K on, for a reason
not yet known (`ContextProfile.md` section 8). Both are work on a model that does not fit a 16 GB card
with room to spare, and both wait in `Vnext.md` for the 24 or 32 GB card where a 27B has that room.
The 2.82-bit build is healthy at every context it fits, 74,752 on the 16 GB card, and it is the Qwen
this release publishes; the FP4 build v0.20 published stays installable and is not published again.

**Publishing the 2.82-bit build is not release work** — weights publish on their own schedule and a
release never re-publishes them — but the claim below depends on it, which is why it is named here.

**Measured as the other families are.** A 27B's BF16 weights are about 54 GB, so its benchmark
reference cannot run on the card Mila measures on; where that reference runs is `ModelEval.md`
decision 1, and the comparison below waits on it rather than substituting a Mila build for the original.

**Success criteria:** a 27B model runs on a 12 GB card from a package a stranger can fetch;
FP8 KV cache compression measured against BF16 at the context the model is claimed for, by the same
protocol the weight allocation used, priced exactly by the footprint the planner reads, and the freed
margin spent deliberately rather than absorbed; Qwen streams its reasoning and its answer as separate
channels; the published build states its benchmark scores beside the original's, from the same harness,
read against the floor Llama 3.2 3B measures; and it has a context profile on the 16 GB card.

### Llama 3 Measured

*The family Mila has run longest, measured as the others are.*

Llama 3.2 3B and 3.1 8B are the models Mila was first validated on, and the least measured for the work
people now ask a local model to do. Their prompt and tool grammar are written out in Chat and in the
inference server rather than in the library, as Gemma's and Qwen's are, and the server's copy already
differs from Meta's template; so the library cannot render a Llama tool conversation, and a Llama
context profile cannot ask its tool-call question. The 3B at BF16 is also where every evaluation is
anchored: it is the one published model whose original weights fit the 16 GB card beside a reference
engine, so its BF16 comparison against HuggingFace is the noise floor every quantized build is read
against.

**Success criteria:** Llama's prompt and tool grammar are the library's, and Chat and the server render
through it, matching Meta's template; Mila BF16 scores Llama 3.2 3B as HuggingFace does on IFEval and
GSM8K, every paired interval containing zero (`ModelEval.md` Phase 1), and the flip rate between them is
recorded as the floor; every published Llama build states its scores beside the original's, from the same
harness, above that floor; and every published Llama build has a context profile on the 16 GB card.

### Applications Ship

*Chat, the inference server and the Python binding run every model this release publishes, correctly,
from packages a user can install.*

The inference server's README tells a user to `pip install mila-llm-server`, and no such package has
ever been published: the release procedure uploads the runtime's wheels and nothing else, so the server
reaches users only from a checkout or inside the container. A client is told every reply finished on
its own, including one cut off at its token limit, and a streamed reply runs past the stop sequences a
buffered one honours. Every application samples at a constant of its own rather than at the setting the
model's publisher recommends, and infers what a model can do — a reasoning channel, its context limit,
and now whether it takes an image — from its family's name rather than reading it from the model.

None of this is the rebuild on `Mila::AI` (v0.22.0), and all of it survives it: the server's package,
its wire behaviour, and the fields a manifest declares are the same either side of the move.

**Success criteria:** the server installs from PyPI beside the binding, at the same version, as the
README tells a user it does; Chat, the server and the binding open a session at the sampling settings
the model's manifest carries, and at its family's published settings when it carries none; a model's
reasoning channel, context limit and modality are read from its manifest, and a manifest that omits
them still loads; and the foreign-harness tool flows established in v0.20 — Codex CLI and Claude Code
CLI over the OpenAI and Anthropic wire shapes — still pass unchanged.

### Upgrades Keep Your Models

*A model installed under v0.20 keeps loading under v0.21, and an older Mila can still install a model
it can run.*

v0.20 published the Gemma 4 12B with its feed-forward tensors named as this tree no longer names them
and its tied table in FP8, which the planner now refuses: an upgrade would stop the model a user
installed. And a pull refuses outright when the newest build of a model needs a newer Mila, even when
an earlier build in the same repository is one this Mila reads. The promise in `ModelDistribution.md`
*Compatibility* is made in this release, so both are kept here.

**Success criteria:** every model v0.20 published loads on v0.21 as installed and answers a factual
prompt; and a pull whose newest build needs a later Mila installs the newest one this Mila can read.

### Performance Next to llama.cpp

*How Mila performs next to the engine most local users already run, on the models they run.*

The website shows Mila against llama.cpp on Gemma 4, Qwen 3.8 and Llama, on a 16 GB consumer card, for
prompt processing and for generation. It is there as evidence of quality: performance as good as the
engine most local users already run, measured the same way, shows that Mila is professional work. It is
not the lead, and not a race: being faster is welcome, not the condition. What the release commits to is
a comparison anyone can rerun, with every cell shown as measured, beside the quality numbers each family
publishes from the same models.

**Success criteria:** the website shows, in tables and graphs, Mila against llama.cpp on every published
model, prefill from 512 tokens to the longest context both fit on the card and generation from an empty
context to that same length, from one script in the repository that reproduces every number.

---

## Future

Uncommitted work — no release, no date. An item **promotes** into the Current release, acquiring its
own version, date, and tag, when it is scheduled.

### v0.22.0 — Intelligence Your Program Owns

*Next, and not yet committed: it is promoted when v0.21.0 ships, and its items wait in
[`Vnext.md`](Mila/Issues/Vnext.md) under these themes. The design of record is
[`Direction.md`](Mila/Specifications/Direction.md) section 5.*

**The release makes one claim, and it is a pair.** A local model is at work inside your program in
ten lines of C++, **and** one call takes you from there to the kernel. The first half without the
second is a wrapper; the second without the first is v0.20. It was v0.21.0's claim until 2026-10-10,
when the finished families were split out to ship first; it is proven on those families.

**One release rather than three.** The model handle's entry point is the deployment planner, and
`Mila::AI` is created over the handle. Built in separate releases, Chat, the inference server and the
Python binding would each move twice — onto the handle, then onto `Mila::AI` — and Chat's automatic
context would move twice with them. Together, each moves once.

**The arc runs in four stages, in dependency order**, a sequence rather than a partition:

1. **The contracts.** Pricing a deployment stops reading the device it is bound to, which changes no
   behaviour.
2. **The planner and the handle.** The build executes the prefill chunk it is given, `planDeployment`
   decides on one device, and the handle's factory takes a deployment request rather than a device.
3. **`Mila::AI`.** The object, and the in-process tool loop beneath it with the token-level splice.
4. **The applications.** Chat, the inference server and the binding rebuilt on `Mila::AI`, the
   QuickStarts rewritten, and the public message moved.

It breaks the API in several places — the per-family session classes, the `Mila/Adaptors/` directory,
and Chat refusing an explicit context length that does not fit where today it warns and tries — and
reads every model format v0.21.0 publishes.

#### Model Handle

*One place that turns a model's name into the type that loads it, reading what the model declares
rather than inferring it from the family.*

It starts at the manifest. A model's capabilities — reasoning channel, context limits, modality and a
draft model — are declared in its own record from v0.21.0, which is additive rather than breaking
because the manifest already tolerates unknown fields and `instruct` proves the pattern. They are what
a deployment selects from (Deployment Planning, below). Today
`Chat.FamilyTraits.ixx` derives those same facts from the family, which is correct only while the set
of families is the set it was written against. **Streaming is not one of them**: whether a display can
route a model's output token by token is a fact about the application that renders it, not about the
weights, and it stays with the application.

The factory then reads the record. A family enum remains, because a model still resolves to a
concrete compile-time type and that is the point of Mila; what goes is every *second* place that
re-derives a model's behaviour from which family it belongs to. The handle lives in the new
`Mila/AI/` library, a peer of `Mila/Src` that depends on it and is never imported by it. Its factory
takes a deployment request, so the planner is its entry point from the first commit rather than a
signature change one release later. A network a developer composed from components meets the handle
through a C++ concept — checked at compile time, with no registration — so a published model and a
composed one pass through the same check.

**The thesis is not what is being traded away.** The erasure is one call when a session opens, not
one per layer or per token. Everything inside the forward pass stays exactly as explicit as it is
now — a handle that hid the forward pass would be a wrapper, and the previous release spent itself
proving Mila is not one.

**Success criteria:** a new architecture is added in one place; no dispatch site outside the handle
carries a per-family branch; a model's declared capabilities come from its manifest, and a model
whose manifest omits a field loads rather than failing; a network composed from components meets the
handle's contract with no registration; and the erasure costs no measurable decode throughput, since
it happens once per session and not once per token.

#### Deployment Planning

*How a model runs on the hardware in front of it is decided once, in the library, and the decision
is a value the caller can read.*

Every "auto" in Mila is decided somewhere different today, and one is decided twice: the prefill
chunk is predicted from one reading of free memory and built from another, and on a card that drives
a display the two can disagree. A Chat user gets an automatic context length; a Python or inference
server user has to know the number. [`Deployment.md`](Mila/Specifications/Deployment.md) replaces
all of it with one decision — a plan, priced before anything is allocated, which the load then
executes without deciding anything again.

This release takes Phases 1 to 4: pricing without a bound device, the build executing the chunk it is
given, `planDeployment` on one device, then the binding and the server. Splitting a model across
devices is Phase 5 and a later release. The weight format stays the caller's choice; headroom is zero
in the library and each application chooses its own; an explicit value that does not fit is refused,
naming what bound it.

**A deployment is a model and the features its use case needs.** A model's package declares what it
can do — modalities, a draft model, the contexts and cache formats it supports — and a request selects
from that, each feature fixed or left to the planner like any other value. The planner prices the
selection exactly and refuses it in its own terms: an image path that costs the context the request
fixed is refused as that, with what does fit beside it, rather than surfacing as an allocation failure.
What is not selected is not built and costs nothing. The weight format remains the caller's, as it
always was.

The bytes a selection spends are recovered in v0.21.0, where each weight tensor stops being an allocation
rounded to the device's 2 MiB granule.

**Success criteria:** `Deployment.md` gates G1 to G4 and negatives N1 to N4, each forced to fail
once; Chat's automatic choices are reproduced by the planner on the measured models before Chat's own
code is deleted; and a Python session and an inference server request each run with an automatic
context length; a request selects every feature a package declares — modality, draft model, context,
cache format — and the plan prices each one exactly, by the same build-equals-plan gate as the rest;
an unaffordable selection is refused naming the feature and what fits without it; and an unselected
feature allocates nothing.

#### Mila::AI

*The object a program creates to put a model to work, and the in-process tool loop beneath it.*

`Mila::AI` is a contract, not a catalogue: create, respond whole or streamed, tools, the conversation
it holds, `plan()` and `model()`. It lives in `Mila/AI/` beside the handle, as its own library target
that the wheel and a `FetchContent` consumer both receive. It never decides a deployment — creation
calls the planner and keeps the plan — and descent is part of the contract rather than a debugging
aid: from `model()` a developer reaches the typed model, and from there every component, operation
and named activation v0.20 already exposes.

Beneath it is the tool loop, which exists today only inside Chat: parse a call, dispatch it, return
the result, continue. It moves into `Mila/AI/` so that no application holds a private copy, and it
gains the depth decided in v0.20 and deferred: a tool result's tokens are appended to the live cache,
with no re-render of the conversation and no re-tokenize. Tools are registered as compiled-in
functions. Qwen cannot rewind its recurrent state to an arbitrary position, but it can return to a
position whose state it saved, which is all a turn boundary needs; the loop reads which kind of reuse a
model offers from the handle rather than discovering it as a failed retry.

A conversation that grows toward the depth where its model stops being reliable is compacted rather
than ended: reasoning from earlier turns is dropped, and the history is summarized into a fresh context
that keeps its instructions verbatim, reusing the system prompt's cached prefix. The depth comes from
measurement of that model and format (`ContextProfile.md`), not a guess. Compaction is mechanism; what a
summary keeps is a policy the application may replace.

**Every model is measured for agentic work, in every family, from v0.21.0.** Perplexity
averages over a whole text and hides the failure an agent meets first: a fact far back in the
conversation that is no longer found, an instruction no longer obeyed, a tool call that no longer
parses. So each model and format is profiled at every context its plan can choose -- recall of a fact
planted at depth, retention of a system instruction, a correct tool call -- and its reliable depth is
the deepest context at which all three still hold. That depth is what compaction triggers on, and it
is the measured reason, or the measured absence of one, for spending further work on a model's
longest contexts.

`Mila::AI` deploys a model for a use case. A use case is a named selection of features over the
same request -- a coding agent, a vision assistant, a long-document reader -- and its defaults come
from measurement: what a context is worth from the model's profile, what a draft model buys from its
measured speedup. A preset is a convenience over the request, never a second path to a deployment,
and the plan it produced is as readable as any other.

The autonomy policy is not in this release. Chat keeps its human approval gate as an application
concern.

**Success criteria:** one ten-line program runs Gemma, Llama and Qwen by changing only the name,
streaming and calling a tool; `plan()` and `model()` reach the objects that actually ran; a sample
creates an `AI` over a network it composed from components itself; and across a multi-turn tool
session, prefill tokens per turn equal the tokens the turn added, measured, for every family that
permits prefix reuse, Qwen from the position it saved at the end of the previous turn; and a tool session run past
its model's measured reliable depth continues after compaction with its system instructions still
obeyed, measured by the instruction-retention arm; the reliable depth compaction uses is the one each model's context profile
reports; and an `AI`
created for a named use case runs with exactly that use case's features resident, its plan showing
each one's bytes, and a program can change any of them before creation.

#### Applications

*Chat, the inference server and the Python binding consume `Mila::AI`, in the same position as any
developer's program.*

The architecture-to-type erasure exists three times in two languages today — Chat's `ModelVariant`,
the binding's per-family session classes and the server's family enum — and the tool loop once, in
Chat. When those are gone, what makes each application distinct is its own concern and nothing more:
terminal rendering and the human approval gate for Chat, the HTTP wire shapes and per-request
statelessness for the server. The server keeps its second job as the conformance oracle.

"Adaptor" is retired with the move. It named Chat and the server by what they were not, and it is a
pattern name that fits one of them. Both leave `Mila/Adaptors/` for `Mila/Applications/Chat` and
`Mila/Applications/Server`. The standing rule carries over with a new subject: a gap between what
`Mila::AI` can do and what an application reaches is a defect in the application.

**Success criteria:** neither Chat nor the server contains model-specific code; the server serves
every architecture Chat does and its family enum is gone; the binding's per-family session classes
are replaced by `mila.AI` with the same contract and the same plan, with no component-level surface
added; and the foreign-harness tool flows established in v0.20 — Codex CLI and Claude Code CLI over
the OpenAI and Anthropic wire shapes — still pass unchanged.

#### A Developer Can Start

*The application developer is a new reader, and the toolchain is their barrier first.*

A reader will fight a toolchain to read something remarkable; an application developer will not.
The toolchain itself does not change — C++23 module interfaces are not portable between compilers,
so there is no prebuilt C++ distribution — and source consumption through CMake is this release's
answer. The C++ QuickStart becomes a program built against Mila with `FetchContent` that creates an
`AI`, calls a tool and prints the plan; the Python QuickStart moves to `mila.AI`. Each keeps a path
that builds a network from components, because a new entry point at the top draws every sample
toward itself and the component path would erode by neglect without anyone deciding it should.

**The public message moves in this release and not before.** The README and the website describe
Mila by the trait — a library for developers to harness intelligence — once the release that makes
it true is tagged. Whether a C ABI is needed at the `AI` boundary is decided during this release
against the QuickStart experience; one would ship no earlier than the next.

**Success criteria:** both QuickStarts run from a clean machine with only the documented
prerequisites, and each keeps a component-built path; and no public surface describes Mila as a
reference implementation first, or uses "adaptor".

- **Intelligence that acts — `Direction.md` section 6.** An agent finishes a multi-step task in your
  process without re-reading its own history, and every step it took can be audited: the autonomy
  policy on `Mila::AI` (Chat becomes the same object under the policy "escalate before every tool"),
  splitting a model across devices (`Deployment.md` Phase 5, `LayerSplit.md`), a session saved and
  restored with its KV cache and no prefill, and two `AI`s planned jointly on one machine. The
  measurement that decides it is `MilaProductFamily.md`'s two-loop comparison — the same model and
  tools driven in process and through the inference server by a foreign harness.
- **Muse Glimmer 30B — the named model target, and not the next release.** Meta's Apache 2.0,
  ungated 30B, chosen for *why it exists* rather than for what it resembles: it is tuned for tool
  use, long tasks, and failure recovery, which is the model an on-device agentic loop actually needs.
  It arrives after v0.22.0, which is `Direction.md` section 5 and adds no new family, because this is
  a second architecture. The
  [product definition](Mila/Specifications/MilaProductFamily.md) already reserves the **Agentic
  adaptor** — the loop closing on itself, on-device — as the post-release member of the family, and
  this is the model that makes it real rather than aspirational.
  Its text tower is close enough to Gemma 4 that the chassis carries over: a repeating
  local/local/local/global attention pattern, final logit softcapping, GQA, RMSNorm with a post-norm,
  a SiLU-gated FFN, a bounded sliding-window ring, and per-layer RoPE. It is dense, so MoE is not a
  prerequisite. Three details are new and each is silent if assumed away — a non-standard QK scale, an
  output multiplier, and RoPE disabled on the global layers rather than merely retuned.
  **The real work is that it is a vision-language model.** A fifty-layer ViT with window attention, 2D
  position embeddings and its own RoPE feeds a projector into the text model. That used to mean a
  second architecture from nothing, which is what made this a tentpole rather than a chassis
  extension. **v0.21.0's Gemma 4 modality work changes the sum**: Gemma 4 12B is encoder-free, so
  building its image path delivers patch embedding, soft-token placement, image templates, the
  application surface on both wire protocols, manifest modality and footprint accounting — leaving the
  tower itself as the only genuinely new piece here.
  **The binding constraint is hardware, not code.** Around 31B parameters is roughly 16 GB at FP4
  before any KV cache, against 12 GB on the card every current Mila claim is validated on. Mila's bar
  is token-for-token agreement with the HuggingFace reference, and that cannot be established on
  hardware which cannot load the model — so this target and the compute ask in
  [SPONSORING.md](SPONSORING.md) are one decision, not two.
  Success bar: greedy text decode matches the reference token-for-token; image-conditioned generation
  validated against the same oracle; tool calling driven end-to-end through MIS; and the Agentic
  adaptor closing a multi-step task on-device.
- **v0.20 library-frozen tails** — the Generation API surface tail (SamplerConfig rename, Llama/Gpt
  seedable sampling, eager sampler, accessor propagation), the Sample-API device-sampler migration for
  Llama/Gpt, a second module-compiler oracle (GCC 16) with a broadened Linux compiler matrix, and the
  ungated GPT-2 zero-auth quick-start (a first-run HTTPS weights fetch). These are library-side, which
  is why they wait; adaptor work does not.
- **Ministral** — Ministral transformer with Sliding Window Attention; 3B Instruct (BF16) and 8B
  Instruct (FP8). Builds on the Llama foundation and the Qwen 3 tool-calling pipeline, reusing the SWA
  mask + bounded-KV ring cache from Gemma 4.
- **Training (advanced)** — the second training release, and large enough to be one: **BF16
  training** and **GQA training**, plus a full LLaMA fine-tuning pipeline, loss-function GPU
  migration, gradient checkpointing, and checkpoint save/restore. Sized honestly, because v0.20's
  training scope was narrowed on the evidence that this half was further out than it looked: GQA
  backward does not exist (`CudaGqaOp::backward` throws, which is what the one deliberate compiler
  warning reports), the loss path is still host-side in both samples, and BF16 needs gradient checks
  at its own tolerance. The BF16 optimizer machinery is in the tree and guarded by
  `AdamW.MixedPrecision.Cuda.cpp` — dormant and tested, in the same spirit as the GQA
  expanded-layout substrate — so this release starts from working parts rather than from repair.
  **Sequencing:** v0.22.0 is `Direction.md` section 5; Muse Glimmer follows it. Training (advanced),
  Qwen 3 and MoE all come after it rather than compete with it — MoE in particular is no longer a
  prerequisite for anything on the critical path, since the Muse Glimmer decoder is dense.
- **Architecture** — additional attention variants. The Mixture-of-Experts components this entry used
  to describe — the `GatedMLP` reusable gated FFN, the grouped `MoeOp`, `Router` and
  `MixtureOfExperts`, specified in `Specifications/FfnAndMoE.md` — **landed during v0.20's `rc.1`**
  and are validated against HuggingFace on the Gemma 4 26B-A4B; what remained was a published package,
  which v0.21.0 takes. Speculative decoding moved with it, since Google's published drafter is what
  makes it concrete rather than general.
- **Performance** — Gemma 4 prefill/decode competitiveness levers (the fused W4A16 prefill GEMM, the
  flash-attention global prefill kernel, the FP4 decode-matvec bandwidth campaign), the codebook path's
  own two (staging to FP8 so the sub-4-bit projections reach the same tensor-core GEMM the FP4 path
  already uses, and closing the codebook GEMV's bandwidth gap), tensor parallelism, and deterministic
  gradient accumulation. Each of these is a measured gap rather than a suspicion; the numbers behind
  them live in `Specifications/Qwen3.8.md` and BACKLOG.
- **Native low-precision compute (Blackwell+)** — microscaling data-path support, finer per-arch gating
  (sm_120, CUTLASS 4.x), and the "compute precision as a first-class axis" design question.
- **Compute backends beyond CUDA** — ROCm (AMD) and Metal (Apple silicon) device backends. Both are
  reserved in `DeviceType` and neither is implemented; Mila is CUDA and CPU today. Because the device
  type is a compile-time template parameter and dispatch resolves through an explicit `OperationTraits`
  table, a backend should be a new partition of specializations rather than conditional compilation
  threaded through the components — that is the design claim, and a port is the first honest test of
  whether it holds across a second GPU vendor. Gated on hardware access (see SPONSORING.md).
  Success bar: an existing validated model path reproduces its token-for-token reference result on the
  new backend, with the component sources unchanged.
- **Platform portability** — `aarch64` as a build and correctness target (Mila is x86-64 Windows and
  Linux today), broadening the Linux compiler matrix, and a third compute-capability gate alongside
  sm_89 and sm_120. Grace-Blackwell-class hardware also puts a coherent unified-memory model in front
  of memory resources and the weight loader, both of which assume discrete device VRAM with explicit
  host-to-device staging — a design question, not only a port.
- **Model loading** — the load-time FP4 sidecar cache and concurrent read I/O.
