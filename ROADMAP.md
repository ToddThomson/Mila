# Mila — Roadmap

Where Mila is going — the durable narrative of each release and what it means.

- **Open tasks** -> [BACKLOG.md](BACKLOG.md) · **Completed work** -> [CHANGELOG.md](CHANGELOG.md)
- **How versions, branches, and releases work** -> [RELEASING.md](RELEASING.md)
- **Design rationale** -> `Mila/Specifications/`

The roadmap shows the **release in flight**, plus a **Future** tail. A release is reached through the
**themed workstreams** below; their tasks live in [BACKLOG.md](BACKLOG.md).

---

## v0.20.0 — First Production Release

**Release Date:** _Target — H2 2026_

Mila is a craft-mastery project — understanding LLMs at the metal — not a llama.cpp/vLLM competitor.
For a project like this, "first release" means **complete and beautiful**, not a minimal slice. v0.20
delivers everything Mila has implemented and validated, as one coherent, tested, documented package:
**Gemma 4, Llama 3.x, and GPT-2** inference at FP32 / BF16 / FP8 / FP4, with tool calling; and
**training for FP32 GPT-2 / MLP**. The two halves have deliberately different reach: inference spans
every model and precision Mila supports, training covers the GPT-2 lineage at FP32. Reduced-precision
and GQA training are a later release — see **Future**.

This scope is a deliberate reunion of two bodies of work. The last year built the inference path
(Llama, quantization, the `OperationTraits` dispatch, the chat harness). The year before built a
fully **test-driven, Doxygen-documented** GPT-2 + training foundation — the MNIST and Bard samples,
the optimizer, the loss and backward kernels — which then fell behind the inference-era API churn and
was parked. v0.20 recovers that foundation to current-API quality rather than shipping inference
alone: resurrection, not invention.

v0.20 ships under a **locked product definition**
([MilaProductFamily.md](Mila/Specifications/MilaProductFamily.md)): Mila is an inference runtime
library plus adaptors distinguished by who closes the generation loop — the Chat harness (human gate)
and MIS (wire) ship with this release; the **Agentic adaptor is explicitly post-release**. The release
bar is that the definition's claims are demonstrable: clone, build, run Gemma 4 12B FP4 on a 12 GB
card, drive it from a foreign harness through MIS, and read the whole path from prompt to kernel with
no hidden engine.

Pre-1.0, "production" means validated and polished, not API-frozen — breaking changes remain
acceptable. The release is reached through the themed workstreams below, in dependency order: the test
suite is revived to become the correctness oracle; training is resurrected against green tests;
documentation describes the stabilized surface; hardening validates and packages it for the public;
and the MIS adaptor proves the product definition end-to-end. At `beta.1` **the library is frozen** —
`Mila/Src` gains no new capability, only hardening. **Model Distribution is the one deliberate
carve-in**, added during `beta.2` because a release nobody can get a model for is not an onboarding
story, and because it was an alpha omission rather than a new idea.

The freeze is drawn around the library, not the release. Everything that *consumes* Mila — the Chat
and MIS adaptors, the Python binding, the samples and the tools — is polish and hardening by nature:
it adds no capability to the runtime, it exposes and demonstrates what the runtime already has. Those
surfaces are expected to improve up to the release, and a gap between what Mila can do and what its
own adaptors reach is a defect in the demonstration rather than a feature request.

### Models

*The model families Mila delivers, each proven token-for-token against HuggingFace at its target
precision.*

- **Gemma 4 12B** — the flagship, and the default chat target at FP4, fitting a 12 GB consumer card.
  Tool calling validated; the 26B-A4B MoE follow-on stays Future.
- **Llama 3.1 8B, 3.2 3B, 3.2 1B** — the primary validated inference lineage; FP4 default with FP8 and
  BF16 alternatives, tool calling validated.
- **GPT-2** — the foundation model behind the MNIST and Bard training samples; FP32 / BF16, inference
  and train-from-scratch.

Which of these a given machine can actually run is a question the library answers rather than one the
user discovers by waiting for an out-of-memory error. A model reports what a build would allocate —
weights, KV cache, activation workspace — for a chosen context length, without allocating any of it,
without the weights present, and therefore for hardware the user does not yet own. The estimate comes
from the same components that do the allocating, so it cannot drift into fiction.

**Success criteria:** each family decodes token-for-token against HuggingFace at its target precision,
captured as CI-guarded regression tests; tool calling validated on Gemma 4 and Llama 3.2 3B / 3.1 8B;
a model's reported footprint matches what it actually allocates, held by test.

### Test Suite Revival

*Re-green the authored test suite to the current API and gate it in CI — the anti-rot ratchet and the
correctness oracle for everything after it.*

The first year of Mila was test-driven; the authored suite was largely commented out during the
inference-era refactors, leaving only ~24 of ~70 files active. This is recovery: the test *logic* is
authored, and the work re-aligns it to the post-refactor API. Two new slices remain — the inference
features built during the test drought (quantization, the Llama path) need coverage the old suite
never had, and the authored suite was **forward-only**, so every `backward()` the training samples
drive has zero coverage. A finite-difference gradient-check archetype is the precondition for Training
Revival.

**Success criteria:** the authored component / tensor / tokenizer suites re-enabled and green against
the current API; the redundant op-layer mirror tests retired (backend ops tested through the public
component, the sole exception being the unreachable weight-quantization white-box); new coverage for
the quantization and Llama inference paths; a per-component gradient-check archetype covering the
training backward path (MNIST spine first, Bard transformer stack second); the suite gated in CI so a
future API churn fails loudly instead of silently rotting coverage.

### Training Revival

*Resurrect the validated GPT-2 / MLP training path — MNIST and Bard — to current-API quality, proven
by its own revived tests. Scope is **FP32 GPT-2 / MLP only**: Llama 3.1/3.2 training, GQA training,
and reduced-precision (BF16) training all stay Future.*

MNIST (MLP) and Bard (GPT-2 generation) were complete, working training samples that are now being
revived. Reviving them reactivates the half of the library inference never exercises: the AdamW
optimizer, the loss and backward kernels, gradient flow, and train-from-scratch parameter
initialization. The revived **primitive/component tests** are the correctness oracle — the samples are
usage demos and the bug-discovery mechanism, not the test target. The work is sequenced **MNIST first,
then Bard**: MNIST is a pure MLP that exercises the full training spine on the smallest possible graph;
Bard then stacks the `GptTransformer`, the BPE/char tokenizers, and the sequence loader on an
already-proven spine.

**The precision boundary is FP32, and it is drawn deliberately rather than by omission.** Reduced-
precision training touches machinery FP32 never does — FP32 master parameters, stochastic-rounding
writeback, narrowing initializers — and that machinery was written but had never once executed: as of
`0.20.0-beta.2+16` the BF16 path could not compile, could not link, and would have trained from zero,
on top of an initializer that overran its own buffer. Fixing those was worth doing, and the code and
its tests stay in the tree, but a path whose first successful step happened during hardening is not a
path this release should claim. FP32 is also the better **reference** implementation, which is the
point of the project: a reader learning how training works should not first have to understand why
there are two copies of every weight.

**Success criteria:** the training-path **primitive suite** is the green/red oracle — the
gradient-check archetype, the AdamW step-convergence test, the concrete data-loader contract tests,
and init-at-precision — with a small **sample-independent** training-loop integration test as
composition/wiring insurance; train-from-scratch validated **at FP32**; all training-path tests
CI-gated. The MNIST and Bard samples are re-enabled and **run** against the current API (MNIST trains
to target accuracy, Bard generates coherent text). **Explicitly not in scope:** BF16 or FP8 training,
GQA training (`CudaGqaOp::backward` throws by design), and Llama fine-tuning.

### API Documentation

*Reconcile the Doxygen surface to the post-refactor reality and publish it — documentation held to
the same standard as the code.*

Doxygen-equal-to-features was a first-year discipline; the inference churn left the prose describing a
retired world, `@file` tags drifted from filenames, and `@param`/`@tparam` names no longer match
signatures. This workstream restores documentation accuracy, narrows the published surface to the
public `import Mila;` API, and publishes via a GitHub Action rather than committing generated docs to
the tree. It reconciles *drift* rather than authoring anew: the Doxygen already exists pervasively, and
Doxygen's own `WARN_*` output is the shrinking worklist.

**Success criteria:** `@file`/`@param`/`@tparam` drift cleared; file-level and symbol Doxygen reflects
the `OperationTraits` world and the spelled-out naming style; the published docs scope matches the
public API surface; the docs job renders C++23 module units faithfully and publishes from `master`;
Doxygen's own warnings gated as errors so doc drift fails the build.

### Production Hardening

*Validate, package, and distribute for external contributors. No new library capability.*

The convergence workstream: prove the primary Llama and Gemma targets against the HuggingFace oracle as
permanent regression tests, package Mila as a consumable source distribution (FetchContent is the
one supported path), stand up the Linux/clang portability gates and the
reproducible container build, and land the contributor-facing surface (coding standards, onboarding, a
guided reading path through one token's journey). Python consumption is the other half of the same
question and a different kind of artifact: a C++ consumer builds Mila from source, while `pip install
mila-llm` hands over a compiled extension that has to carry its own CUDA and work on a machine that
has none. Both platforms ship a wheel. Engineering detail lives in [BACKLOG.md](BACKLOG.md)
under this bucket.

**Success criteria:** an external consumer can build against Mila via FetchContent; `pip install
mila-llm` gives a working runtime on Windows and Linux with no CUDA Toolkit installed; the Linux/clang
build is a first-class, CI-compiled + WSL-tested platform, with
a reproducible container build; contributor onboarding (`CONTRIBUTING.md`, `getting-started.md`, a
guided reading path) complete; the public export surface frozen at the narrowest defensible umbrella; a
missing dispatch specialization reads as a sentence, not a constraint cascade. GPU-first: the CUDA
backend is the validated inference path; full CPU op parity is not a gate.

### Model Distribution

*One manifest describes every model, whatever its origin; a published model is one command away, and
the store that holds it can be inspected and emptied.*

Until now, using a Mila model meant already having the file, and the only way to get one was a
converter run needing PyTorch and 23.8 GB of source weights. That is the right workflow for adding a
model family and the wrong one for using a model Mila already publishes. The second problem is that a
path says nothing: every consumer — the chat catalog, the inference server, a user with a directory of
files — rebuilds by hand the knowledge of what a file is, what it needs, and where it came from.

This workstream makes a model a described thing with a name. **One manifest describes every model**,
whether it was fetched from a hub or built on the machine that loads it, so a model converted from a
gated family is as first-class as one Mila publishes. Retrieval, listing, removal and publishing then
become operations on described things rather than conventions over filenames. HuggingFace is the first
concrete hub behind an abstracted interface, and the `mila-llm` organization is the namespace Mila
publishes into; Mila runs no registry of its own and never uploads from the library.

Two boundaries define the design. **Loading never downloads** — a model is pulled deliberately, with
progress and a failure mode, and loaded from the local store afterward, so a multi-gigabyte transfer
can never begin inside a chat prompt or in response to an inference request. And **the flat `.bin`
container stops being a distributed form**: every distributed model is a safetensors artifact with a
manifest, which also retires the model aliases whose meaning nobody outside the codebase could
decode. Engineering detail lives in [BACKLOG.md](BACKLOG.md) under this bucket; the
design is [ModelDistribution.md](Mila/Specifications/ModelDistribution.md).

**Success criteria:** a clean machine pulls and runs Gemma 4 12B FP4 from `mila-llm` through named
commands, with no manual download and no converter; a model built locally from a family Mila cannot
republish is listed, loaded and described exactly like a fetched one; the store reports what is
installed and what it costs, and removing one variant does not damage another that shares its
tokenizer; Chat and the inference server share one store as separate processes; no catalogue entry
names a `.bin`; and a build without the hub still lists, locates and removes.

### Product Family — Adaptor Validation

*Prove the locked product definition's central claim: the whole path is demonstrable end-to-end.*

The [MilaProductFamily.md](Mila/Specifications/MilaProductFamily.md) definition ships two adaptors with
v0.20 — Chat (human gate) and MIS (Python wire) — distinguished by who closes the generation loop.
Both are consumers of a frozen library, so both stay open to change: where an adaptor cannot reach
something Mila already does, that gap is the work. The Agentic adaptor stays explicitly post-release.

**Success criteria:** a foreign harness (Codex CLI and Claude Code CLI over the OpenAI/Anthropic wire
shapes) drives Gemma 4 12B FP4 through MIS across plain-chat, single-tool, and tool-result-resume flows
with no leaked control tokens; the C++/Python grammar duplication is resolved by an explicit decision
(single-sourced via pybind, or pinned by a cross-language parity test).

---

## Future

Uncommitted work — no release, no date. An item **promotes** into the Current release, acquiring its
own version, date, and tag, when it is scheduled.

- **One model handle, before the family grows again.** The first work after the v0.20 tag, and a
  precondition for every model entry below. Naming a model is a runtime act; loading one is a
  compile-time type, and something has to bridge the two — today that bridge is written three times
  in two languages: Chat's `ModelVariant`, the binding's per-family session classes, and the
  inference server's own family enum. Nothing keeps them in step, and the drift is already visible:
  GPT-2 runs in Chat and is refused by the server, not by decision but because the second bridge was
  never written for it. A gap that reads as a policy.
  What makes this urgent is not the number of models. Under Mila's selection rule — the leading crest
  of open models that suit an agentic workflow on hardware you already own, not every model in
  existence — that list grows slowly and deliberately. The pressure is that each of those bridges
  currently assumes every model does the same things. That assumption holds exactly until a model
  arrives that does not, and the next two candidates both break it: one carries a vision tower, and
  both reason in a channel only Gemma has today. Then every dispatch site in all three places grows a
  per-family branch, and the cost of a new architecture stops being one chassis and becomes an edit
  to every consumer of one.
  So it lands first, in the runtime-adjacent agent core that Chat and the future Agentic adaptor
  share, reading the model's declared capabilities from its manifest rather than inferring them from
  its family. It leaves the thesis intact: the erasure is one call when a session opens, not one per
  layer or per token, and everything inside the forward pass stays exactly as explicit as it is now.
  Success bar: a new architecture is added in one place; the inference server serves every
  architecture the chat harness does; and no dispatch site carries a per-family branch.
- **Muse Glimmer 30B — the named next target.** Meta's Apache 2.0, ungated 30B, chosen for *why it
  exists* rather than for what it resembles: it is tuned for tool use, long tasks, and failure
  recovery, which is the model an on-device agentic loop actually needs. The
  [product definition](Mila/Specifications/MilaProductFamily.md) already reserves the **Agentic
  adaptor** — the loop closing on itself, on-device — as the post-release member of the family, and
  this is the model that makes it real rather than aspirational.
  Its text tower is close enough to Gemma 4 that the chassis carries over: a repeating
  local/local/local/global attention pattern, final logit softcapping, GQA, RMSNorm with a post-norm,
  a SiLU-gated FFN, a bounded sliding-window ring, and per-layer RoPE. It is dense, so MoE is not a
  prerequisite. Three details are new and each is silent if assumed away — a non-standard QK scale, an
  output multiplier, and RoPE disabled on the global layers rather than merely retuned.
  **The real work is that it is a vision-language model.** A fifty-layer ViT with window attention, 2D
  position embeddings and its own RoPE feeds a projector into the text model, and Mila is text-only:
  no patch embedding, no vision tower, no projector, no image-token path. That is a second
  architecture, and it is what makes this a tentpole rather than a chassis extension.
  **The binding constraint is hardware, not code.** Around 31B parameters is roughly 16 GB at FP4
  before any KV cache, against 12 GB on the card every current Mila claim is validated on. Mila's bar
  is token-for-token agreement with the HuggingFace reference, and that cannot be established on
  hardware which cannot load the model — so this target and the compute ask in
  [SPONSORING.md](SPONSORING.md) are one decision, not two.
  Success bar: greedy text decode matches the reference token-for-token; image-conditioned generation
  validated against the same oracle; tool calling driven end-to-end through MIS; and the Agentic
  adaptor closing a multi-step task on-device.
- **Qwen 3** — Mila's third architecture family: a Qwen 3 dense decoder with thinking mode,
  model-agnostic tool calling, and FP8 KV cache compression, validated on Qwen 3 8B Instruct at BF16
  and FP8. Reuses the Llama blocks (RMSNorm, SwiGLU, GQA, RoPE); the new work is the Chat layer (ChatML
  template, `ToolCallParser`, thinking-mode suppression) and FP8 KV cache (`PerChannelKvFp8<>`).
  Success bar: greedy decode at BF16 and FP8 each match HuggingFace token-for-token; tool calling
  validated end-to-end; thinking-mode suppression confirmed; FP8 KV cache quality acceptable vs. BF16.
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
  **Sequencing:** the slot after v0.20 goes to Muse Glimmer, above. Training (advanced), Qwen 3 and
  MoE all follow it rather than compete for it — MoE in particular is no longer a prerequisite for
  anything on the critical path, since the Muse Glimmer decoder is dense.
- **Architecture** — Mixture-of-Experts components (the `GatedMLP` reusable gated FFN, the grouped
  `MoeOp`, `Router` + `MixtureOfExperts`; foundation specified in `Specifications/FfnAndMoE.md`). The
  Gemma 4 dense chassis is the precursor to the 26B-A4B MoE model, which reuses the chassis and swaps
  only the FFN block. Also: speculative decoding, additional attention variants.
- **Performance** — Gemma 4 prefill/decode competitiveness levers (the fused W4A16 prefill GEMM, the
  flash-attention global prefill kernel, the FP4 decode-matvec bandwidth campaign), tensor parallelism,
  and deterministic gradient accumulation.
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
