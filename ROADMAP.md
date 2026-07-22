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
**Gemma 4, Llama 3.x, and GPT-2**; **inference and training**; FP32 / BF16 / FP8 / FP4; tool calling.

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
and the MIS adaptor proves the product definition end-to-end. At `beta.1` the feature set is
**frozen** — what remains is hardening, not new capability.

### Models

*The model families Mila delivers, each proven token-for-token against HuggingFace at its target
precision.*

- **Gemma 4 12B** — the flagship, and the default chat target at FP4, fitting a 12 GB consumer card.
  Tool calling validated; the 26B-A4B MoE follow-on stays Future.
- **Llama 3.1 8B, 3.2 3B, 3.2 1B** — the primary validated inference lineage; FP4 default with FP8 and
  BF16 alternatives, tool calling validated.
- **GPT-2** — the foundation model behind the MNIST and Bard training samples; FP32 / BF16, inference
  and train-from-scratch.

**Success criteria:** each family decodes token-for-token against HuggingFace at its target precision,
captured as CI-guarded regression tests; tool calling validated on Gemma 4 and Llama 3.2 3B / 3.1 8B.

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
by its own revived tests. Scope is GPT-2 / MLP only; Llama 3.1/3.2 training stays Future.*

MNIST (MLP) and Bard (GPT-2 generation) were complete, working training samples that are now being
revived. Reviving them reactivates the half of the library inference never exercises: the AdamW
optimizer, the loss and backward kernels, gradient flow, and train-from-scratch parameter
initialization. The revived **primitive/component tests** are the correctness oracle — the samples are
usage demos and the bug-discovery mechanism, not the test target. The work is sequenced **MNIST first,
then Bard**: MNIST is a pure MLP that exercises the full training spine on the smallest possible graph;
Bard then stacks the `GptTransformer`, the BPE/char tokenizers, and the sequence loader on an
already-proven spine.

**Success criteria:** the training-path **primitive suite** is the green/red oracle — the
gradient-check archetype, the AdamW step-convergence test, the concrete data-loader contract tests,
and init-at-precision — with a small **sample-independent** training-loop integration test as
composition/wiring insurance; train-from-scratch validated at the precisions the samples use; all
training-path tests CI-gated. The MNIST and Bard samples are re-enabled and **run** against the
current API (MNIST trains to target accuracy, Bard generates coherent text).

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

*Validate, package, and distribute for external contributors. No new features beyond the frozen set.*

The convergence workstream: prove the primary Llama and Gemma targets against the HuggingFace oracle as
permanent regression tests, package Mila as a consumable source distribution (FetchContent is the
supported path; `find_package` is parked), stand up the Linux/clang portability gates and the
reproducible container build, and land the contributor-facing surface (coding standards, onboarding, a
guided reading path through one token's journey). Engineering detail lives in [BACKLOG.md](BACKLOG.md)
under this bucket.

**Success criteria:** an external consumer can build against Mila via FetchContent; the Linux/clang
build is a first-class, CI-compiled + WSL-tested platform, with
a reproducible container build; contributor onboarding (`CONTRIBUTING.md`, `getting-started.md`, a
guided reading path) complete; the public export surface frozen at the narrowest defensible umbrella; a
missing dispatch specialization reads as a sentence, not a constraint cascade. GPU-first: the CUDA
backend is the validated inference path; full CPU op parity is not a gate.

### Product Family — Adaptor Validation

*Prove the locked product definition's central claim: the whole path is demonstrable end-to-end.*

The [MilaProductFamily.md](Mila/Specifications/MilaProductFamily.md) definition ships two adaptors with
v0.20 — Chat (human gate) and MIS (Python wire) — distinguished by who closes the generation loop. The
Chat surface is validated; net-new Chat feature work is **deferred to Future** under the feature freeze,
so this workstream closes **MIS**. The Agentic adaptor stays explicitly post-release.

**Success criteria:** a foreign harness (Codex CLI and Claude Code CLI over the OpenAI/Anthropic wire
shapes) drives Gemma 4 12B FP4 through MIS across plain-chat, single-tool, and tool-result-resume flows
with no leaked control tokens; the C++/Python grammar duplication is resolved by an explicit decision
(single-sourced via pybind, or pinned by a cross-language parity test).

---

## Future

Uncommitted work — no release, no date. An item **promotes** into the Current release, acquiring its
own version, date, and tag, when it is scheduled.

- **Qwen 3** — Mila's third architecture family: a Qwen 3 dense decoder with thinking mode,
  model-agnostic tool calling, and FP8 KV cache compression, validated on Qwen 3 8B Instruct at BF16
  and FP8. Reuses the Llama blocks (RMSNorm, SwiGLU, GQA, RoPE); the new work is the Chat layer (ChatML
  template, `ToolCallParser`, thinking-mode suppression) and FP8 KV cache (`PerChannelKvFp8<>`).
  Success bar: greedy decode at BF16 and FP8 each match HuggingFace token-for-token; tool calling
  validated end-to-end; thinking-mode suppression confirmed; FP8 KV cache quality acceptable vs. BF16.
- **v0.20 feature-frozen tails** — the Generation API surface tail (SamplerConfig rename, Llama/Gpt
  seedable sampling, eager sampler, accessor propagation), the Sample-API device-sampler migration for
  Llama/Gpt, the unspecced **Chat** feature milestone, a second module-compiler oracle (GCC 16) with a
  broadened Linux compiler matrix, and the ungated GPT-2 zero-auth quick-start (a first-run HTTPS
  weights fetch).
- **Ministral** — Ministral transformer with Sliding Window Attention; 3B Instruct (BF16) and 8B
  Instruct (FP8). Builds on the Llama foundation and the Qwen 3 tool-calling pipeline, reusing the SWA
  mask + bounded-KV ring cache from Gemma 4.
- **Training (advanced)** — a full LLaMA fine-tuning pipeline, loss-function GPU migration, gradient
  checkpointing, and checkpoint save/restore.
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
