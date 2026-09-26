# Mila — Roadmap

Where Mila is going — the durable narrative of each release and what it means.

- **Open tasks** -> [BACKLOG.md](BACKLOG.md) · **Completed work** -> the
  [release notes](https://github.com/ToddThomson/Mila/releases)
- **How versions, branches, and releases work** -> [RELEASING.md](RELEASING.md)
- **Design rationale** -> `Mila/Specifications/`

The roadmap shows the **release in flight**, plus a **Future** tail. A release is reached through the
**themed workstreams** below; their tasks live in [BACKLOG.md](BACKLOG.md).

---

## v0.21.0 — Intelligence Your Program Owns

**Release Date:** _Target — 2026-12-15_

**The release makes one claim, and it is a pair.** A local model is at work inside your program in
ten lines of C++, **and** one call takes you from there to the kernel. The first half without the
second is a wrapper; the second without the first is v0.20. The design of record is
[`Direction.md`](Mila/Specifications/Direction.md) section 5.

**It is proven on families that are finished.** Qwen 3.8 and Gemma 4 shipped in v0.20 with parts
left over — a 2.82-bit build nobody can fetch, a KV cache left uncompressed on a card the weights
already fill, a checkpoint whose image weights the loader drops. They complete in this release, and
they are what the new entry point is demonstrated on. An entry point that has only ever met the
models it was written against is an abstraction nobody has stressed.

**One release rather than three.** The model handle's entry point is the deployment planner, and
`Mila::AI` is created over the handle. Built in separate releases, Chat, the inference server and the
Python binding would each move twice — onto the handle, then onto `Mila::AI` — and Chat's automatic
context would move twice with them. Together, each moves once.

**The date is fixed and the scope drains.** The release opened against 2026-10-31 as the handle and
the two families; on 2026-09-23, before any of that work started, it was widened to take in
`Direction.md` section 5 and re-dated. That was a re-plan at the start of the cycle, and it is the
only one: the date does not move again. What has not landed by 2026-12-15 returns to [`Vnext.md`](Mila/Issues/Vnext.md) and the
release ships with what it has, its claim stated as narrowed. **Gemma 4 image input drains first** —
it is the largest item whose size is not yet known, and the release's proof does not rest on it.

**The arc runs in five stages, in dependency order.** The stages are a sequence, not a partition —
an item is workable whenever its own blockers are gone.

1. **The contracts.** Pricing a deployment stops reading the device it is bound to, which changes no
   behaviour; the manifest gains the capability fields the handle will read; and the two questions
   that size Gemma's image path are answered by reading, before anything is built.
2. **The planner and the handle.** The build executes the prefill chunk it is given, `planDeployment`
   decides on one device, and the handle's factory takes a deployment request rather than a device.
3. **`Mila::AI`.** The object, and the in-process tool loop beneath it with the token-level splice.
4. **The applications.** Chat, the inference server and the binding rebuilt on `Mila::AI`, the
   QuickStarts rewritten, and the public message moved.
5. **The families finished.** Qwen 3.8, then Gemma 4, ending in modality.

Pre-1.0 still holds: breaking changes are acceptable, and this release carries several — the
per-family session classes, the `Mila/Adaptors/` directory, and Chat refusing an explicit context
length that does not fit where today it warns and tries.

### Model Handle

*One place that turns a model's name into the type that loads it, reading what the model declares
rather than inferring it from the family.*

It starts at the manifest. A model's capabilities — reasoning channel, context limits, and now
modality — are declared in its own record, which is additive rather than breaking because the
manifest already tolerates unknown fields and `instruct` proves the pattern. Today
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

### Deployment Planning

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

**Success criteria:** `Deployment.md` gates G1 to G4 and negatives N1 to N4, each forced to fail
once; Chat's automatic choices are reproduced by the planner on the measured models before Chat's own
code is deleted; and a Python session and an inference server request each run with an automatic
context length.

### Mila::AI

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
functions. Qwen cannot rewind its recurrent state, so it refuses prefix reuse; the loop reads that
from the handle as a property of the model rather than discovering it as a failed retry.

The autonomy policy is not in this release. Chat keeps its human approval gate as an application
concern.

**Success criteria:** one ten-line program runs Gemma, Llama and Qwen by changing only the name,
streaming and calling a tool; `plan()` and `model()` reach the objects that actually ran; a sample
creates an `AI` over a network it composed from components itself; and across a multi-turn tool
session, prefill tokens per turn equal the tokens the turn added, measured, for every family that
permits prefix reuse — for Qwen the refusal is reported, not discovered.

### Applications

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

### A Developer Can Start

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

### Qwen 3.8 Complete

*Everything the family was specced to be, including the parts v0.20 shipped around.*

v0.20 promoted Qwen 3.8 out of a research track and into the release, and the promotion was earned —
but it shipped the model rather than the family. Three things were left: the 2.82-bit build is
finished, validated and **unpublished**, so the claim the release was written around had to be
narrowed to the FP4 build at 16 GB; the KV cache is uncompressed on a card the weights already fill,
which is what bounds the context the model is actually sold at; and the dense members were never
built, although they reuse Llama blocks rather than the DeltaNet chassis and are cheap. A dense
member arriving after the handle is also the first test of the claim that a new model is added in
one place.

The user-visible one is smaller and worse: a 27B answers in a single block after a long silence,
because the per-token router was written against Gemma's four control tokens and Qwen's
`<think>`/`</think>` pair was never wired to it. It is the longest wait in Mila with nothing on
screen.

**Publishing the 2.82-bit build is not release work** — weights publish on their own schedule and a
release never re-publishes them — but the claim below depends on it, which is why it is named here.

**Success criteria:** a 27B model runs on a 12 GB card from a package a stranger can fetch;
FP8 KV cache compression measured against BF16 at the context the model is claimed for, by the same
protocol the weight allocation used, priced exactly by the footprint the planner reads, and the freed
margin spent deliberately rather than absorbed; the quality gate re-run at whatever context the
release ends up advertising rather than stopping at 16K; a dense member decodes token-for-token
against HuggingFace at BF16 and FP8; and Qwen streams its reasoning and its answer as separate
channels.

### Gemma 4 Complete

*The family Mila validates most closely, finished — including the half of the checkpoint the loader
currently discards.*

Gemma 4 is the family with the most Mila behind it and the most left on the table. The 26B-A4B
mixture of experts **already works** — router, expert bank, FP4 expert bank and streaming converter,
gated against HuggingFace at BF16 and FP4 — and appears in no capability row because no published
package uses it. Two protocol defects survive into multi-step tool use: a model working through a
sequence loses the reasoning that led to each step, and a malformed call parses as a call with no
arguments rather than as prose, which a client that executes `{}` would act on.

**Modality is the largest single item in the release.** The 12B is encoder-free — no vision tower —
so images enter through `vision_embedder` and an embedding projection directly into the decoder at
280 soft tokens, and audio through `embed_audio`. The converter skips both today. That makes this
materially smaller than the vision-language work in **Future** below, and it is also the thing that
de-risks it: patch embedding, soft-token placement, image templates, the application surface on both
wire protocols, manifest modality and footprint accounting all carry over, leaving a tower as the
only genuinely new piece. It also runs on a 12 GB card, so unlike that work it is not gated on
hardware.

**Its size is not yet known, and that is stated rather than estimated.** Two questions decide it:
whether image soft tokens must attend bidirectionally within a prefill, when every attention path in
Mila is causal, and the position scheme for image spans. Both are answered in the first stage of the
release, before anything is built, and the answers decide whether modality stays in it.

Two further questions are answered by a measurement rather than committed on intent, and each carries
its own stop condition: Google's quantization-aware 4-bit build is only worth loading if it beats
Mila's FP4, and its published drafter is only worth a speculative loop if a K-token verify costs
meaningfully less than K decodes on a bandwidth-bound FP4 path. The measurements are this release;
the implementations they would justify are not.

**Finished also means level with the other families** (`ModelFamilyParity.md`). Gemma is the
furthest along of the three, and its gaps are few but one of them is sharp: it cannot score a text,
so its quality above 131072 tokens has never been measured, while the planner may now choose up to
the 262144 its weights declare. And Chat still renders Gemma's prompt with its own copy of a
template the library already has, so the two can drift.

**Success criteria:** Gemma 4 12B accepts an image and answers about it, with the embedder gated
against HuggingFace and then token-for-token on a mixed prompt; modality is declared in the manifest
and read by the handle rather than by a family test; the 26B-A4B is fetchable and named in a
capability row; reasoning survives across tool calls within a turn and a malformed call is refused
rather than executed as empty; each measurement-gated question has a recorded result, including
the result "not worth doing"; Gemma scores a text at the head width a request asks for, and its
quality is measured at every context length the planner can choose for it; and Chat renders
Gemma's prompt with the library's template, not its own.

---

## Future

Uncommitted work — no release, no date. An item **promotes** into the Current release, acquiring its
own version, date, and tag, when it is scheduled.

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
  It arrives after v0.21.0, which is `Direction.md` section 5 and adds no new family, because this is
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
  **Sequencing:** v0.21.0 is `Direction.md` section 5; Muse Glimmer follows it. Training (advanced),
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
