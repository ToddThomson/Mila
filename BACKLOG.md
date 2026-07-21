# Mila — Backlog

The working task list — open engineering tasks for the release in flight. Narrative and success
criteria live in [ROADMAP.md](ROADMAP.md); shipped work in [CHANGELOG.md](CHANGELOG.md); design
rationale under `Mila/Specifications/`.

Each `###` bucket is a v0.20 theme, its name matching the ROADMAP section (the join). Status: `[ ]`
open · `[~]` in progress · `[x]` done (kept for the per-bucket gauge until the production release
prunes it). Tags: **[gate]** blocks the release · **[deferred]** parked · **[contributor]**
good-first-issue.

---

## Current release (v0.20.0)

### Models

- [~] Llama HF-parity regression test — add a `LlamaModel` parity test (Gemma has
  `GemmaModel.Parity.Cuda.cpp`, Llama has none); validate + record 3.1 8B FP8. Folds into Test Suite
  Revival's Llama-path backfill.
- [ ] Tool calling validated on Llama 3.2 3B and 3.1 8B Instruct.
- [ ] Triage `Llama.Block.ixx:132` view-aliasing — the Q/K/V splits of `qkv_out` may not be
  contiguous; confirm live-vs-benign and fix if live before claiming Llama HF validation.
- [ ] GQA standalone-`forward()` stub — component-level Gemma/Llama attention has no independent
  correctness oracle. Precondition for retiring the legacy GQA path. See `Specifications/GqaMemory.md`.
- [ ] **[contributor]** Llama 3.2 1B/3B weight tying — the aliasing plumbing shipped; add
  `tie_word_embeddings_` + post-load aliasing + `getMemoryStats` correction to `LlamaTransformer`.
  See `Specifications/WeightTying.md` §6.

### Test Suite Revival

- [~] Re-green the authored component / tensor / tokenizer suites to the current API — concrete
  component-class set re-enabled and build-green; only `SoftmaxCrossEntropy` (loss) parked for the
  loss-on-device work. 3 backward-numeric cases `GTEST_SKIP`'d pending filed bugs (CUDA Softmax
  backward stub, BF16 Swiglu backward dtype, GptBlock composed gradient).
- [~] Core `Tensor.ixx` coverage to the value-type archetype — remaining: `TensorOps.Transfer`
  device-split, `Structural`(`split`) backfill, and the wider `Tensors/` tree (`TensorBuffer`,
  `TensorDataType*` maps, `Partitioning`, `Serialization`). See `Specifications/Testing.Tensors.md`.
- [ ] Backfill inference-drought coverage — load-time quantization (`PerChannelFp8`/`PerGroupFp4`,
  decode matvec kernels), `OperationTraits` dispatch, the Llama path. The `CudaLinearOp` quantization
  white-box is the sole legitimate op-layer test (unreachable through the public component).
- [~] Re-green in sample-revival order — MNIST spine mostly landed; remaining: the `Core/Network.cpp`
  delta, GPU companions (`Network.Cuda`/`AdamW.Cuda`), then the Bard GPT-2 stack tail.
- [ ] Retire the redundant op-layer mirror tests — out of the CMake build; files kept on disk pending
  an explicit delete.
- [x] Gradient-check archetype (finite-difference numeric backward) — shared `Common/GradientCheck.h`
  fanned out across the training spine; MHA backward exonerated. Validated VS2026 2026-07-02.
- [x] Verify the full suite green in one pass (CPU-only `MILA_ENABLE_CUDA=OFF` + the CUDA build).
- [x] **[gate]** Wire the suite into CI as the anti-rot ratchet — `cpu-only-tests` job runs the CPU
  suite on every push/PR; GitHub Actions green at 0.20.0-alpha.6+116.

### Training Revival

- [x] Revive the MNIST (MLP) sample + validate — trains FP32 to ~97.9% test accuracy; spine tests green.
- [x] Revive the Bard (GPT-2) sample + validate — trains to coherent Shakespeare; fixed 3 latent CUDA
  training-backward bugs.
- [~] Data-loader contract tests — `TokenSequenceLoader` done; remaining: the `MnistDataLoader`
  contract test (normalization, one-hot targets, shuffle-on-reset, IDX magic-number).
- [~] Re-enable the AdamW path — `AdamW.Cpu.cpp` active with a convergence case; remaining:
  `AdamW.Cuda.cpp` companion + strip-vs-gate the `CudaAdamW.cu` / `CudaAdamWOptimizer.ixx:270` debug
  `printf`s in the same pass.
- [~] **[net-new]** Training-loop integration test (sample-independent) — MNIST spine covered by
  `Network.Cpu.cpp`; remaining: a GPT-2-stack analogue for the Bard spine.
- [ ] **[net-new]** Optimizer step-convergence test — minimizes a known convex objective in N steps
  (proves update direction + bias-correction, not just that `step()` runs).
- [ ] **[net-new]** TrainingMode / RuntimeMode behavior coverage — assert build/runtime-mode
  transitions allocate/skip gradient buffers correctly (regression guard for the lifecycle fix).
- [ ] Fix the CUDA `fill_normal` / `fill_uniform` FP32-only gap (corrupts BF16 train-from-scratch
  init) — pair with a BF16 init-at-precision `TYPED_TEST` that turns the silent corruption red.
- [ ] **[decoupled]** Revive the loss + backward path (CrossEntropy / SoftmaxCrossEntropy) — both
  samples compute loss host-side, so off the critical path to a converging sample.
- [ ] **[net-new, training-only]** Revive the `Dropout` component from `Dev/Components/Regularization/`.
- [ ] ProgressReporter — an injected per-operation progress facility for long-lived ops (BPE vocab
  training, `PretrainedReader` load, load-time quantization).
- [ ] Validation — training path proven by the primitive suite (gradient-checks, optimizer
  step-convergence, loader contracts, init-at-precision, the integration test), CI-gated; the samples
  run as demos.

### API Documentation

- [x] Narrow the published surface to the `import Mila;` API (EXTRACT flip + INPUT scoping).
- [x] Oracle — Doxygen's own `WARN_*` output wired as the shrinking worklist.
- [x] Tier 0 (non-ASCII / mojibake), Tier 1 (`@file` rename drift), Tier 2 (`@param`/`@tparam` name
  mismatches) all cleared to zero.
- [x] Ratchet — `WARN_AS_ERROR` set; doc drift fails the docs build.
- [x] Docs-site CI decoupled — canonical `Mila/Docs/Doxyfile`, `docs.yml` runs Doxygen 1.17 directly.
- [ ] Tier 3 — semantic staleness (retired-world prose); **folded into Test Suite Revival** — fix a
  file's prose while it is open for re-greening.
- [ ] Confirm the docs-CI run green on GitHub Actions (Pages publish + pinned-Doxygen download).

### Production Hardening

- [x] External consumer builds against Mila via **FetchContent** (gate met); `find_package` PARKED
  (retired in place, `MILA_INSTALL` OFF by default).
- [x] Freeze the narrowest defensible export surface — RESOLVED: the umbrella is as narrow as C++23
  modules allow (a type in a public template's interface must be visible, not merely reachable, at
  instantiation). A `Mila.ixx` header contract records the rule.
- [~] Contributor onboarding — `CONTRIBUTING.md` + `getting-started.md` DONE; remaining (GitHub-side):
  `good first issue` labels and the `dev -> master` default-branch flip.
- [~] Linux/clang first-class platform — WSL green, CI compiles under clang-21, container builds +
  runs Gemma 4 FP4. GCC 16 second oracle + broadened compiler matrix -> Future.
- [~] Reproducible container build — validated (clang-21 + gcc-15 host, CUDA 13.3); remaining: build
  against the bind-mounted tree, and CI building `FROM` the image rather than apt-installing.
- [~] Dispatch error UX — a missing `(Op, Device, Precision)` reads as one line, not a cascade. Core
  landed (declaration-only primary + `OperationSupported<...>` predicate); optional named kernel
  concepts + `OperationDispatch.md` §12 reconcile remain.
- [ ] Add the Samples build to CI (only tests build today).
- [ ] Guided reading path — one token's journey (embed -> attend -> sample -> decode) through the real
  source, readable by a strong C++ dev unaided.
- [ ] Backfill the README **Gemma 4 flagship** perf numbers from a profile run — prefill-vs-llama.cpp
  and FP4 decode tok/s. Placeholders "(pending a profile run)" + `<!-- BACKFILL -->` markers are in
  place; grep them. Beta.1-readiness (must be exact before the label).
- [ ] Published Docker runtime image — slim multi-stage GPU runtime, release-tagged, weights never baked in.
- [ ] Module import hygiene — Phase 0 exact-dup dedup, Phase 1 candidate report, Phase 2
  compiler-verified removal (Clang/GCC, not MSVC); plus domain-qualify generic single-segment module
  names (`Core`/`Utils`/`Components`/`Profiling` -> `Dnn.*`).
- [ ] Marker-debt triage — the ~94 remaining `REVIEW:` markers (56 files); the correctness "bucket D"
  items each need eyes-on.
- [ ] Broaden CI compiler coverage toward the supported matrix (adds MSVC + GCC 16 to clang-21).
- [ ] Stage model weights off the Windows bind mount for the container (native disk speed).
- [ ] **[contributor]** Llama-lineage CPU ops (`RmsNormOp`, `SwigluOp`, `RopeOp`, `TokenEmbeddingOp`,
  `CrossEntropyOp`) in `OperationTraits.Cpu.ixx` — demand-driven; absence is zero-cost on the GPU path
  (full CPU parity is not a gate).
- [ ] **[deferred, measure first]** Remove FP16 (superseded by BF16) — woven through live code
  (`CudaDataTypeMap<half>`, `CudaLinearOp` half branches, `*_fp16` GQA/MHA/LPE stubs); trace
  live-vs-dead before removal.

### Product Family — Adaptor Validation

- [~] MIS Gemma 4 tool-calling validated end-to-end — Codex + Claude Code CLI round-trips live; the
  native grammar reconciled to Google's canonical chat template (nine divergences fixed), pinned by an
  oracle. Remaining: N sequential distinct tool calls in one turn, channel-content parser polish,
  Codex-CLI re-validation on the reconciled grammar.
- [~] Grammar-in-runtime execution-time scope call — C++ and Python grammars held together by a
  cross-language parity test; MIS prompt pinned to Google's vendored template. Open for sign-off:
  whether to single-source via pybind or close on the parity test.
- [ ] In-turn thoughts dropped between tool calls — Google's multi-turn rule (strip prior-turn
  thoughts, keep the current turn's).
- [ ] MIS `top_p` dropped before the sampler — the pybind path does not forward it.
- [ ] Refine: buffer Gemma Anthropic streaming only when tools are present.
- [ ] Neutral binding output location — `Bindings/CMakeLists.txt:49` still copies `mila.pyd` into a
  non-neutral path.

---

## Future

Uncommitted / next-cycle work. Coarse by design — detailed tasking happens only when an item promotes
to the current release.

- **Qwen 3** (presumptive next release) — the dense decoder, thinking-mode suppression, model-agnostic
  tool calling, and FP8 KV cache (`PerChannelKvFp8<>`); the `OperationTraits<GqaOp, Cuda, BF16,
  PerChannelKvFp8<>>` specialization lands here.
- **v0.20 feature-frozen tails** — the Generation API surface tail (`SamplerConfig` rename, Llama/Gpt
  seedable sampling, eager sampler, config-accessor propagation, `contextLength()` hoist), the
  Sample-API device-sampler migration for Llama/Gpt, the Optimizer-dispatch migration onto
  `OperationTraits`, and the unspecced **Chat** feature milestone.
- **Ministral** — SWA transformer; reuses the Llama foundation, Qwen 3 tool-calling, and the Gemma 4
  SWA mask + bounded-KV ring.
- **Architecture / MoE** — generalize `GatedMLP`'s gate (GeGLU/ReGLU) + the CPU `SwigluOp`; grouped
  `MoeOp` + `Router` + `MixtureOfExperts`; `LlamaBlock` delegating to `GatedMLP`. See
  `Specifications/FfnAndMoE.md`.
- **Training (advanced)** — Llama fine-tuning, loss-function GPU migration, gradient checkpointing,
  checkpoint save/restore, GQA training (the dormant expanded-layout substrate).
- **Performance** — Gemma 4 competitiveness levers (fused W4A16 prefill GEMM, flash-attention global
  prefill kernel, FP4 decode-matvec bandwidth), tensor parallelism, deterministic gradient
  accumulation. See `Specifications/GqaMemory.md`, `W4A16` design notes.
- **Native low-precision compute (Blackwell+)** — microscaling data path, finer per-arch gating
  (sm_120, CUTLASS 4.x), "compute precision as a first-class axis".
- **Model loading** — load-time FP4 sidecar cache; concurrent / async read I/O for real queue depth.
- **Ungated GPT-2 zero-auth quick-start** — first-run HTTPS weights fetch (a runtime addition the
  freeze excludes). Freeze-compatible descope: host the pre-converted blob + a one-line download.
- **`ComponentType` vitality** — does `getType()` earn its keep, or retire the unused converter surface?
- **Python sample** — surface the `mila` binding as a standalone Python sample for the
  Python-majority audience. The binding already exists, so this is sample/doc work (no runtime
  feature) — a candidate beta.2 sprint rather than a hard v0.20 gate.
