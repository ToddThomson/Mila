# Mila — Roadmap

Where Mila is going — the durable narrative of each release and what it means.

- **Open tasks** -> [BACKLOG.md](BACKLOG.md) · **Completed work** -> the
  [release notes](https://github.com/ToddThomson/Mila/releases)
- **How versions, branches, and releases work** -> [RELEASING.md](RELEASING.md)
- **Design rationale** -> `Specifications/`

The roadmap shows the **release in flight**, plus a **Future** tail. A release is reached through the
**themed workstreams** below; their tasks live in [BACKLOG.md](BACKLOG.md).

---

## v0.21.0 — The Model Families, Finished and Measured

**Release Date:** _To be set_ (2026-12-04 was withdrawn on 2026-11-02, Todd, when the rest of the agents API came into
this release; the date follows a sizing of that work)

**The release makes one claim: every model Mila publishes is finished, and you can see how good it
is.** Gemma 4, Qwen 3.8 and Llama 3 each run every capability their checkpoints carry that the
release names, at the rates and the quality a reader can check against the original model and
against llama.cpp, from packages that install with `pip` and keep loading when Mila is upgraded.

**Finished means what v0.20 left over.** Qwen 3.8 and Gemma 4 shipped with parts missing — a KV cache left uncompressed on a card the weights already fill, a checkpoint
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

**This release also makes the claim `Direction.md` section 5 was written for: a local model at work inside your
program in ten lines of C++, and one call from there to the kernel.** Section 5 was split out as v0.22.0 on
2026-10-10, so that finished models need not wait for the API, and *The Desktop Agent* then brought the model handle
and the agent with its tool loop back. Much of v0.22.0 was therefore already here, so this release ships it (Todd,
2026-11-02): *A Developer Can Start*, below, adds what the claim still lacks. Mila is one library with one install,
whose models API and agents API a program reaches as `Mila::Dnn` and `Mila::Agents`. Chat and Mila Desktop are built
on the agents API. The inference server stays as it is in this release, on the binding's sessions; what it gains here
is what survives its rebuild -- a published server, correct replies, and defaults read from the model. Choosing a
deployment's features, use cases and that rebuild are v0.22.0's (**Future**, below).

**Once it is set, the date is fixed and the scope drains, except for *The Desktop Agent*.** What has not landed by then
returns to [`Vnext.md`](Issues/Vnext.md) and the release ships with what it has, its claim stated as
narrowed. **The 26B-A4B's vision tower is tentative and drains first**, with *Mila on Apple Silicon*, which is
experimental; the 12B's image and audio are committed. *The Desktop Agent* neither drains nor narrows: its code is not compromised to hold the date, and if
it is not done the date moves (Todd, 2026-10-25).

Pre-1.0 still holds: breaking changes to the API are acceptable. **A model a user installed is not one
of them.** From this release on, upgrading Mila by one minor never stops an installed model loading,
and an older Mila can still install a build of a model that it can run. So every format this release's
packages carry — the six-bit tied table, the drafter's place in a package, the manifest's new fields —
is one v0.22.0 must read, and each is decided before it ships.

### Gemma 4 Complete

*The family Mila validates most closely, finished — including the half of the checkpoint the loader
currently discards.*

Gemma 4 is the family with the most Mila behind it and the most left on the table. The 26B-A4B
mixture of experts **runs** — router, expert bank in FP4 and in the Q4_0 Google trained it for, and a
streaming converter, gated against HuggingFace at BF16, FP4 and Q4_0, its kernels at llama.cpp's rates or above
(`ModelFamilyParity.md` §8.2) — but no end user can install it yet. **It ships level with every other model Mila
publishes, or not at all** — the trained format, the reference card, rates beside llama.cpp's, quality across
the planner's range, and every application.

**Modality is the largest single item in the release, and the two models differ in it.** The 12B is a
unified transformer and encoder-free — no vision tower — so images enter through `vision_embedder` and
an embedding projection directly into the decoder at up to 280 soft tokens, and audio through
`embed_audio`, a single projection of raw 40 ms frames rather than an encoder. The converter skips both
today. Image tokens take ordinary positions, but the tokens of one picture attend each other in both
directions; that mask, the largest piece, is built, and the embedders, the prompt that carries media and the
applications remain (`Gemma4Modality.md`). Chat accepts both an image and an audio clip. **The 26B-A4B is a mixture of
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
furthest along of the three, and its gaps are few but one of them is sharp: the planner may choose up
to the 262144 its weights declare, and Gemma's quality has not been gated across that range. Both models' BF16 weights exceed the 16 GB
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
but it shipped the model rather than the family. Its KV cache was uncompressed on a card the weights already
fill, which is what bounded the context the model was actually sold at. With the cache at FP8 and two
measured cuts to three bits, `fp4-cb3` holds 65,792 tokens on the 16 GB card and stays close to FP4 on
agentic code (`Qwen3.8.md` section 5): 64K is the floor for agentic work, and quality comes first. The
dense members, which reuse Llama blocks rather than the DeltaNet chassis, are new building rather than
finishing, and wait for v0.22.0, where a member arriving after the handle is the first test of the claim
that a new model is added in one place.

The user-visible one is smaller and worse: a 27B answers in a single block after a long silence,
because the per-token router was written against Gemma's four control tokens and Qwen's
`<think>`/`</think>` pair was never wired to it. It is the longest wait in Mila with nothing on
screen.

**This release finishes what Qwen shares with the other families, not its tuning.** The 2.82-bit
build's prefill and generation are measurably less efficient than they could be (`ModelFamilyParity.md`
8.3, Q9), and that waits in `Vnext.md` for the 24 or 32 GB card where a 27B has room. The FP4 build
predicts a book worse with more of it in context from 8K on, for a reason not yet known
(`ContextProfile.md` section 8), and `fp4-cb3` carries it to 64K; finding the reason is this release's,
under the model fixes. `fp4-cb3` is the Qwen this release publishes. The 2.82-bit build, which a 12 GB
card would need, is not published: on agentic code it drops a closing bracket of a nested pair. The FP4
build v0.20 published stays installable.

**Publishing `fp4-cb3` is not release work** — weights publish on their own schedule and a release never
re-publishes them — but the claims below depend on it, which is why it is named here.

**Measured as the other families are.** A 27B's BF16 weights are about 54 GB, so its benchmark
reference cannot run on the card Mila measures on; where that reference runs is `ModelEval.md`
decision 1, and the comparison below waits on it rather than substituting a Mila build for the original.

**Success criteria:** end users install Qwen 3.8 27B from a published package and run it at 64K on the 16 GB card;
FP8 KV cache compression measured against BF16 at the context the model is claimed for, by the same
protocol the weight allocation used, priced exactly by the footprint the planner reads, and the freed
margin spent deliberately rather than absorbed; Qwen streams its reasoning and its answer as separate
channels; `fp4-cb3` states its benchmark scores beside the original's, from the same harness,
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
and now whether it takes an image — from its family's name rather than reading it from the model. And
the website, where a user arrives, says nothing about running either application: the server appears
only in blog posts, and Chat only as one line of the build instructions.

None of this is the rebuild on `Mila::Agents` (v0.22.0), and all of it survives it: the server's package,
its wire behaviour, and the fields a manifest declares are the same either side of the move.

**Success criteria:** the server installs from PyPI beside the binding, at the same version, as the
README tells a user it does; Chat, the server and the binding open a session at the sampling settings
the model's manifest carries, and at its family's published settings when it carries none; a model's
reasoning channel, context limit and modality are read from its manifest, and a manifest that omits
them still loads; the foreign-harness tool flows established in v0.20 — Codex CLI and Claude Code
CLI over the OpenAI and Anthropic wire shapes — still pass unchanged; and the website has a page for each
application — Chat, the server with every harness it is tested with, and the model store — that takes a
user from install to a running model.

### The Desktop Agent

*The models this release finishes call tools, and Mila Desktop, lets you put them to work on your own
files and the web, each call under your rules and kept to the folder you choose -- built from the library alone,
so a developer's program can do the same.*

Family parity gave this release models that call tools: Gemma 4 and Qwen 3.8 are driven through tool calls today,
and Llama never has been, which this theme closes. The only harnesses that exercise them are other people's, over
the inference server, and Mila's own Chat runs one stub tool. Mila Desktop's window was built in October against
a mock; this theme puts it on a real model and makes it an agent harness: the model reads and edits files in a
folder the person chooses and reads web pages, and every call is allowed, asked or refused by the person's rules.

**The library provides everything an application like this needs** (`ToolCalling.md`, decided 2026-10-25 with
Todd, replacing the per-user tool broker of 2026-10-24). So the agent is built in `Mila::Agents`, the library's agents
API, which nothing else in Mila imports, and the app adds only its window, its rules' storage and its tools. That brings forward
from v0.22.0 what tool calling stands on: the model's tool grammar on the model handle, the handle and factory that
let one loop drive every family, and `Mila::Agents` itself, whose tool loop appends each result's tokens to the live
cache instead of reading the conversation again. On it: tools that declare what they touch, permissions every call
passes, with the app's hook for asking the person, and a grant through which a tool opens files and makes
connections, which holds it to the folder and to public addresses. The models API has no tools. The agents API
provides the file and web tools both apps offer, written through grants, and Chat offers them as Mila Desktop does: a person names a folder or
allows the web, and is asked before each call and shown what it can reach. And an agent holds a session -- what the program
gave it and the conversation -- apart from its model, so a program changes model or context without losing either;
Mila Desktop and Chat each build on the library alone for that, with no family-specific code of their own (`Direction.md` 4.3, Todd, 2026-10-27). One program-running tool comes in as an
experiment, off by default: the grant runs a program in Microsoft's execution container, held to the folder, the
network as granted and a desktop of its own, and Mila Desktop's Build & Code gains a shell through it
(`ContainedPrograms.md`, Todd, 2026-10-27). MCP and other people's tool servers come later, through the same
container. The grant's checks are Windows first: on any other platform the app runs without tools until they are
tested there.

**Success criteria:** on Windows, Mila Desktop runs every model this release publishes that calls tools through a
task that reads files in the chosen folder, edits one, and reads a web page, each call allowed or asked by the
person's rules and shown in the conversation; a denied call reaches the model as declined and the reply continues;
a tool working through its grant cannot read or write outside the chosen folder by any name that leads there, and
reaches no private or local address by name, by literal address or through a redirect, shown by a test per kind of
name and address that is refused; Mila Desktop and Chat run every tool call through `Mila::Agents`,
and neither runs a call itself nor holds a model's tool grammar; the console Chat asks before a tool runs, as the
window does, and edits a file in a folder the person named and reads a web page through the same tools the window
offers; a program built against the library alone registers a tool and runs it under `Mila::Agents`'s permissions
on Gemma, Llama and Qwen by changing only the model's name, `plan()` and `model()` reaching the objects that ran;
the agents API is in the Mila library, built on the rest of it, and nothing else in the library imports it; a program changes its model, or its context, and keeps its tools, rules, grant and conversation; Mila Desktop and
Chat each build on `Mila::Agents` alone, neither linking the other's code, and neither holds code that knows which family
a model belongs to; no dispatch site in `Mila::Agents` carries a
per-family branch; and across a multi-turn tool session, prefill tokens per turn equal the tokens the turn added,
measured, for every family that permits prefix reuse, Qwen from the position it saved at the end of the previous
turn; and with the experimental setting on, on Windows 11 26H2, Mila Desktop runs a command in the chosen folder,
asked first, and a test per kind of access the container refuses -- a file outside, a name inside for a file
outside, a file also in a denied folder, a write to a read-only folder, the network with none granted, a private
address with it granted, the person's screen -- shows it refused beside the same access inside allowed.

### A Developer Can Start

*A local model at work in your program in ten lines, in C++ or Python, from one install, and one call from there to the
kernel.*

The application developer is a new reader, and the toolchain is their barrier first. A reader will fight a toolchain
to read something remarkable; an application developer will not. The toolchain itself does not change -- C++23 module
interfaces are not portable between compilers, so there is no prebuilt C++ distribution -- and source consumption
through CMake is the answer. The C++ QuickStart becomes a program built against Mila with `FetchContent` that creates
an agent, calls a tool and prints the plan. Each QuickStart keeps a path that builds a network from components, because
a new entry point at the top draws every sample toward itself and the component path would erode by neglect without
anyone deciding it should.

**Python is one install.** `mila-llm` carries the agents API as `mila.Agent`, with the same contract and the same plan,
and a Python function can be a tool. The binding's per-family sessions stay beside it in this release, since the
inference server runs on them. The binding gains no component-level surface.

**A long session continues.** A conversation that grows toward the depth where its model stops being reliable is
compacted rather than ended: reasoning from earlier turns is dropped under that pressure, and the history is summarized
into a fresh context that keeps its instructions verbatim. The depth comes from
measurement of that model and format (`ContextProfile.md`), not a guess. Compaction is mechanism; what a summary keeps
is a policy the application may replace.

**The public message moves in this release and not before.** The README and the website describe Mila by the trait --
a library for developers to harness intelligence -- once the release that makes it true is tagged. Whether a C ABI is
needed at the agent boundary is decided during this release against the QuickStart experience; one would ship no
earlier than the next.

**Success criteria:** a tool session run past its model's measured reliable depth continues after compaction with its
system instructions still obeyed, measured by the instruction-retention arm; the reliable depth compaction uses is the
one each model's context profile reports; a Python program that installs `mila-llm` alone creates a `mila.Agent`,
registers a Python function as a tool and runs it under the same permissions, with the same plan a C++ program reads,
and the binding gains no component-level surface; both QuickStarts run from a clean machine with only the documented
prerequisites, with no initialization call, and each keeps a component-built path; and no public surface describes
Mila as a reference implementation first, or uses "adaptor".

### Upgrades Keep Your Models

*A model installed under v0.20 keeps loading under v0.21, and an older Mila can still install a model
it can run.*

v0.20 published the Gemma 4 12B with its feed-forward tensors named as this tree no longer names them
and its tied table in FP8, which the planner now refuses: an upgrade would stop the model a user
installed. And a pull refuses outright when the newest build of a model needs a newer Mila, even when
an earlier build in the same repository is one this Mila reads. The promise in `ModelDistribution.md`
*Compatibility* is made in this release, so both are kept here. The 12B itself is replaced: Google's
quantization-aware weights in Q4_0 are a different model from the FP4 package v0.20 published, so a user
who has the old one is told which model replaces it, and the old one keeps working.

**Success criteria:** every model v0.20 published loads on v0.21 as installed and answers a factual
prompt; a pull whose newest build needs a later Mila installs the newest one this Mila can read; and a
model another replaces names its successor wherever the store and Chat show it.

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

### Mila on Apple Silicon

*Mila runs your models on a Mac, through Metal, from the same packages -- experimental in this release.*

Mila was CUDA and the CPU until November 2026, when a development Mac arrived and the Metal backend began
(`Specifications/Metal/MetalBackend.md`). It is the first test of the claim the type axes were built on: a second GPU
vendor as a new partition of operations, with the components unchanged. On the M1 Pro Llama 3.2 3B already matches
HuggingFace in FP16, the M1's precision, and decodes at 0.83 of llama.cpp after 4096 tokens. This theme carries that to
the models this release publishes and to the things a person installs, all of them marked experimental: Mila Desktop
as a Mac app, signed and notarized, Mila for Python as a wheel for Apple silicon, and Chat and the model store built from
source as on every other platform (Todd, 2026-11-05). Published packages do not change; Metal reads the ones CUDA
reads. A 16 GB Mac holds less than a 16 GB card, since the GPU shares that memory with the Mac, so not every published
model fits one. Mila Desktop on a Mac runs without tools until the grant's checks are tested there, as *The Desktop
Agent* already says of every platform but Windows.

**Experimental, and the first to drain** (Todd, 2026-11-05): what has not landed by the date returns to `Vnext.md`
before anything else does.

**Success criteria:** on an M1 Pro with 16 GB and macOS 27, every model this release publishes that the planner admits
there passes its family's HuggingFace gate on Metal, in its published format; Mila Desktop installs on that Mac from a
signed, notarized download and runs one of those models in a conversation; `pip install mila-llm` on that Mac opens a
session on one and generates; Chat and `mila-store` build there from source with the documented toolchain and run one;
the rates on that Mac are shown beside llama.cpp's on the same Mac, by the same method; and every surface that offers
a Mac artifact says it is experimental.

---

## Future

Uncommitted work — no release, no date. An item **promotes** into the Current release, acquiring its
own version, date, and tag, when it is scheduled.

### v0.22.0 — A Deployment Chosen for Its Use

*Next, and not yet committed: it is promoted when v0.21.0 ships, and its items wait in
[`Vnext.md`](Issues/Vnext.md) under these themes. The design of record is what remains of
[`Direction.md`](Specifications/Direction.md) section 5; v0.21.0 ships the rest.*

**The release makes one claim.** A program says what it wants a model for, and the plan spends the card on that and
nothing else. It reads every model format v0.21.0 publishes, and it breaks the API where the binding's per-family
sessions go.

#### Deployment Planning

*A deployment is a model and the features its use case needs.*

A model's package declares what it can do -- modalities, a draft model, the contexts and cache formats it supports --
and a request selects from that, each feature fixed or left to the planner like any other value. The planner prices
the selection exactly and refuses it in its own terms: an image path that costs the context the request fixed is
refused as that, with what does fit beside it, rather than surfacing as an allocation failure. What is not selected
is not built and costs nothing. The weight format remains the caller's, as it always was. Splitting a model across
devices is `Deployment.md` Phase 5 and a later release.

**Success criteria:** a request selects every feature a package declares -- modality, draft model, context, cache
format -- and the plan prices each one exactly, by the same build-equals-plan gate as the rest; an unaffordable
selection is refused naming the feature and what fits without it; and an unselected feature allocates nothing.

#### Use Cases

*A named selection of features over the same request.*

`Mila::Agents` deploys a model for a use case -- a coding agent, a vision assistant, a long-document reader -- and its
defaults come from measurement: what a context is worth from the model's profile, what a draft model buys from its
measured speedup. A preset is a convenience over the request, never a second path to a deployment, and the plan it
produced is as readable as any other. A network a developer composed from components meets the agent here too
(`ModelHandle.md` 10.4).

**Success criteria:** an agent created for a named use case runs with exactly that use case's features resident, its
plan showing each one's bytes, and a program can change any of them before creation.

#### Applications

*The inference server consumes `Mila::Agents`, in the same position as any developer's program.*

The server's family enum and the binding's per-family session classes are the last of the architecture-to-type
erasures. When they are gone, what makes the server distinct is its own concern and nothing more: the HTTP wire shapes
and per-request statelessness. It keeps its second job as the conformance oracle.

**Success criteria:** the server contains no model-specific code and its family enum is gone; the binding's per-family
session classes are gone; a new architecture is added in one place, and no dispatch site outside the handle carries a
per-family branch; and the foreign-harness tool flows established in v0.20 -- Codex CLI and Claude Code CLI over the
OpenAI and Anthropic wire shapes -- still pass unchanged.

#### The Dense Qwen Members

*The first model added through the handle, which is the test of the claim that a new model is added in one place.*

### After v0.22.0

- **Intelligence that acts — `Direction.md` section 6.** An agent finishes a multi-step task in your
  process without re-reading its own history, and every step it took can be audited: the autonomy
  policy on `Mila::Agents` (Chat, already on it, runs under the policy "escalate before every tool"),
  splitting a model across devices (`Deployment.md` Phase 5, `LayerSplit.md`), a session saved and
  restored with its KV cache and no prefill, and two agents planned jointly on one machine. The
  measurement that decides it is `MilaProductFamily.md`'s two-loop comparison — the same model and
  tools driven in process and through the inference server by a foreign harness.
- **Muse Glimmer 30B — the named model target, and not the next release.** Meta's Apache 2.0,
  ungated 30B, chosen for *why it exists* rather than for what it resembles: it is tuned for tool
  use, long tasks, and failure recovery, which is the model an on-device agentic loop actually needs.
  It arrives after v0.22.0, which is `Direction.md` section 5 and adds no new family, because this is
  a second architecture. The
  [product definition](Specifications/MilaProductFamily.md) already reserves the **Agentic
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
  whether it holds across a second GPU vendor. Metal is now in v0.21.0, experimental (*Mila on Apple
  Silicon*, above, Todd, 2026-11-05); ROCm stays gated on hardware access (see SPONSORING.md).
  Success bar: an existing validated model path reproduces its token-for-token reference result on the
  new backend, with the component sources unchanged.
- **Platform portability** — `aarch64` as a build and correctness target (Mila is x86-64 Windows and
  Linux today), broadening the Linux compiler matrix, and a third compute-capability gate alongside
  sm_89 and sm_120. Grace-Blackwell-class hardware also puts a coherent unified-memory model in front
  of memory resources and the weight loader, both of which assume discrete device VRAM with explicit
  host-to-device staging — a design question, not only a port.
- **Model loading** — the load-time FP4 sidecar cache and concurrent read I/O.
