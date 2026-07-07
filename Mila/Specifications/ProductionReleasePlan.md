# v0.20 Release Runway

The session-by-session traversal order from `0.20.0-alpha.6` to the v0.20 first production
release, locked 2026-07-07 alongside the product definition
([MilaProductFamily.md](Mila/Specifications/MilaProductFamily.md)).

**What this file is:** the *order of work*, one coherent session per entry. It adds no new tasks —
every session maps onto existing [ROADMAP.md](ROADMAP.md) milestone checkboxes and
[BACKLOG.md](BACKLOG.md) items, which remain the single source of truth for task status. Tick a
session here when it completes; delete this file when v0.20 ships. It is deliberately
self-expiring so it cannot drift into a fourth bookkeeping surface.

**Scope guard:** v0.20 = runtime + Chat + MIS with the Definition's claims demonstrable. The
Agentic adaptor, native-agent-core extraction, and delivered token-level splice are post-release
(see the spec's Release Boundary). If a session starts pulling Agentic work in, stop it.

Sessions average out — some are half-days, others may split in two. Ordering constraints are
noted; within a phase, adjacent sessions can sometimes swap. Every session ends at a VS2026
build/validate hand-off.

---

## Phase 1 — Close the correctness oracle (Test Suite Revival tail)

*Must come first: Phase 2 renames and migrations against dead tests is the exact rot pattern the
revival exists to end.*

- [ ] **Session 1 — Inference-drought backfill A: quantization + dispatch.**
  The one test slice the old suite never had, protecting the newest code. Load-time quantization
  coverage (`PerChannelFp8` / `PerGroupFp4`, the decode matvec kernels — the `CudaLinearOp`
  white-box is the sole legitimate op-layer test) plus `OperationTraits` dispatch coverage.
  *Exit:* quantization + dispatch tests authored and green locally.

- [ ] **Session 2 — Inference-drought backfill B: Llama path + Tensors tail.**
  Component coverage for RmsNorm / SwiGLU / GQA / RoPE and `LlamaModel::fromPretrained`; the
  remaining `Tensors/` tree items (`TensorOps.Transfer` device-split, `Structural` backfill,
  `TensorBuffer`, `TensorDataType*` maps, `Partitioning`, `Serialization`).
  *Exit:* Llama inference path has the coverage the drought skipped.

- [ ] **Session 3 — Training-test tail.**
  `MnistDataLoader` contract test; `AdamW.Cuda` companion (+ the deferred AdamW instrumentation
  strip-vs-gate call); optimizer step-convergence test (known convex objective in N steps);
  TrainingMode / RuntimeMode behavior coverage; CUDA `fill_normal` / `fill_uniform` BF16 fix with
  the init-at-precision `TYPED_TEST`; the Bard-spine training-loop integration analogue.
  *Exit:* the training-path primitive suite is complete per the Training Revival success criteria.

- [ ] **Session 4 — Verified green + the ratchet.**
  Full-suite green pass in one shot (CPU-only `MILA_ENABLE_CUDA=OFF` and the CUDA build — user
  runs, we triage reds); then wire the suite into CI as the anti-rot gate, and confirm the real
  docs-CI run (pinned-Doxygen download + Pages publish).
  *Exit:* the CI ratchet is live. Everything after this session is protected.

## Phase 2 — API close-outs

- [ ] **Session 5 — Sampler unification.**
  Migrate `LlamaModel` + `GptModel` onto the base `sampleNext()` (delete their host
  `sampleToken`s), wire `seedSampler` for Llama/Gpt reproducibility, eager sampler construction.
  Closes the LanguageNetwork milestone and two Generation API items in one stroke.
  *Exit:* one sampler path, three models, seedable everywhere.

- [ ] **Session 6 — Generation API tail + Optimizer dispatch.**
  `SamplingConfig` -> `SamplerConfig` rename (the deferred highest-risk cross-module rename — now
  safe behind Phase 1's green suite); `contextLength()` hoist to `LanguageModel` (mode-aware);
  `getNetworkConfig()` / `getModelConfig()` propagation to Llama/Gpt;
  `GemmaTransformer::getConfig()` hygiene + the `int64_t`-vs-`dim_t` settle; Optimizer dispatch
  onto `OperationTraits`.
  *Exit:* Generation API milestone closed.

## Phase 3 — Product family (grammar-in-runtime)

*Can interleave after Phase 1; kept after Phase 2 so the grammar module lands on a settled API.*

**Reorder (2026-07-07):** Session 7 was pulled ahead of Phase 1 as a continuation of the recent
Gemma tool-calling / MIS grammar work while context was hot. Sound because Session 7 authors its
own tests and the grammar surface is API-independent of Phase 2's renames; the one hard constraint
(Phase 2 after Phase 1) is preserved. Session 8's cross-language parity-into-CI wiring still waits
for Phase 1's ratchet (Session 4).

- [x] **Session 7 — Canonical grammar module.** (LANDED 2026-07-07, build + Chat validated.)
  Runtime module `Dnn.Components.GemmaProtocol`
  (`Src/Dnn/Components/Transformers/Gemma/Gemma.Protocol.ixx`) — control-token constants +
  `parseToolCall` / `formatToolCall` / `formatToolResponse`, seeded from the union of the two
  prior implementations. Folded in the Python side's spec-verified behaviors: `<|"|>` string
  delimiter (parse + render), integer-preserving coercion, tool-response output-field distillation
  with failed-tool error surfacing. Chat consumes it via `import Mila`; `Chat.GemmaToolCallParser`
  retired in place; own test `Gemma.Protocol.cpp`. (Turn/channel parse + control-token stripping
  stay in Chat for now — not drifted — as a follow-up to single-source the constants.)
  *Exit met:* one tool-grammar implementation; Chat renders the trained string delimiter.

- [ ] **Session 8 — Grammar scope call + MIS close-out.**
  Decide at the keyboard: pybind the grammar so `gemma_protocol.py` consumes the same source, OR
  (if not bounded) keep MIS on Python pinned by a cross-language parity test over a shared fixture
  corpus. Then the bounded MIS items: `top_p` wired through to the sampler, server README rewrite,
  channel-parser polish as time allows.
  *Exit:* the two adaptors cannot silently diverge again.

## Phase 4 — Training-only deferred work

- [ ] **Session 9 — Loss path + Dropout.** (May split in two.)
  Revive CrossEntropy / SoftmaxCrossEntropy (the dispatch struct started in alpha.6+68, not wired);
  re-author `Dropout` from `Dev/Components/Regularization/` (`CpuDropoutOp` / `CudaDropoutOp` +
  `OperationTraits` rows + the two-axis component rewrite).
  *Exit:* the training path no longer computes loss host-side; mask/backward covered.

- [ ] **Session 10 — Retirement sweep.**
  ProgressReporter mechanism (BPE training progress migrates here); final legacy-dispatch
  retirement (`OperationRegistry` + the arity bases — unblocked by Session 9); remaining marker
  buckets (C `dim_t` canonicalization, D correctness items, H org/docs).
  *Exit:* Consolidation milestone fully closed; no legacy dispatch left in the build.

## Phase 5 — Production Hardening + release

- [ ] **Session 11 — Validation sweep.**
  Llama 3.2 1B FP32, 3.2 3B BF16, 3.1 8B FP8 against the HuggingFace oracle; tool calling
  validated on Llama 3.2 3B and 3.1 8B Instruct.
  *Exit:* every shipped model/precision row in the README table re-verified on the release tree.

- [ ] **Session 12 — Comprehensibility.**
  The guided reading path — one token's journey (embed -> attend -> sample -> decode) traced
  through the actual source, readable by a strong C++ developer unaided; plus the Tier-3
  semantic-prose stragglers not already fixed in passing during Phase 1.
  *Exit:* the comprehensibility deliverable exists and reads true against the code.

- [ ] **Session 13 — Packaging.**
  `find_package(Mila)` validated by an external consumer build; the published slim Docker runtime
  image; the ungated GPT-2 zero-auth quick-start path.
  *Exit:* a stranger can consume Mila three ways without help.

- [ ] **Session 14 — Release.**
  `CONTRIBUTING.md` + `getting-started.md` polish; `good first issue` labels; the consolidated
  CHANGELOG entry for the closed milestones; ROADMAP stage sections deleted per convention;
  Version flip; tag.
  *Exit:* v0.20 shipped.
