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
- [ ] RoPE scaling disabled on the Llama load path — `Llama.ixx:703` has
  `.withRoPEScalingFactor( metadata.rope_scaling )` commented out with the reason recorded as unclear.
  Llama 3.1 8B's extended context depends on it; resolve before writing the 8B parity test, not after.
- [ ] `GptModel.ixx:330` hardcodes `eos_token_ = 50256` — should come from tokenizer metadata.
- [ ] GQA standalone-`forward()` stub — component-level Gemma/Llama attention has no independent
  correctness oracle. Precondition for retiring the legacy GQA path. See `Specifications/GqaMemory.md`.
  `GroupedQueryAttention.ixx:177` is the dead branch this task decides retire-vs-wire on: it returns an
  un-computed `output_view_` and is unreached in the validated path (Llama/Gemma drive
  `prefill()`/`decode()` directly).
- [ ] GQA `forward()` fallback is stale — `GroupedQueryAttention.ixx:299` records the non-KV-cache
  fallback as needing a correctness review, with the shape derivation commented out beneath it.
- [ ] `CudaMhaOp.ixx:433` initializes `active_max_seq_len_ = T_` with the reason unrecorded — confirm
  against the two-phase KV-cache contract (prefill full sequence, decode `outer_size == 1`).
- [ ] **[contributor]** Llama 3.2 1B/3B weight tying — the aliasing plumbing shipped; add
  `tie_word_embeddings_` + post-load aliasing + `getMemoryStats` correction to `LlamaTransformer`.
  See `Specifications/WeightTying.md` §6.

### Test Suite Revival

- [~] Re-green the authored component / tensor / tokenizer suites to the current API — concrete
  component-class set re-enabled and build-green; only `SoftmaxCrossEntropy` (loss) parked for the
  loss-on-device work. 3 backward-numeric cases `GTEST_SKIP`'d pending filed bugs (CUDA Softmax
  backward stub, BF16 Swiglu backward dtype, GptBlock composed gradient). The Softmax stub is not
  missing code — `CudaSoftmaxOp.ixx:73` deliberately throws `"needs review"` with the real
  `cuda_softmax_backward<float>` call commented out; the FP16 twin at `:103` is the same.
- [~] Core `Tensor.ixx` coverage to the value-type archetype — remaining: `TensorOps.Transfer`
  device-split, `Structural`(`split`) backfill, and the wider `Tensors/` tree (`TensorBuffer`,
  `TensorDataType*` maps, `Partitioning`, `Serialization`). See `Specifications/Testing.Tensors.md`.
  Eight `REVIEW:` markers name the specific contracts to pin: Copy as a no-op on empty tensors and on
  scalars (`TensorOps.Transfer.ixx:92`); context/device compatibility and the device-ID logic on the
  CUDA transfer path (`CudaTensorOps.Transfer.ixx:132,140,276`); sub-byte/packed FP4 sizing
  (`Tensor.ixx:267`, `TensorBuffer.ixx:78`); the size helper duplicated from `TensorBuffer`
  (`Tensor.ixx:83`); and the moved-from state (`Tensor.ixx:479`).
- [ ] Backfill inference-drought coverage — load-time quantization (`PerChannelFp8`/`PerGroupFp4`,
  decode matvec kernels), `OperationTraits` dispatch, the Llama path. The `CudaLinearOp` quantization
  white-box is the sole legitimate op-layer test (unreachable through the public component).
- [~] Re-green in sample-revival order — MNIST spine mostly landed; remaining: the `Core/Network.cpp`
  delta, GPU companions (`Network.Cuda`/`AdamW.Cuda`), then the Bard GPT-2 stack tail.
- [ ] Retire the redundant op-layer mirror tests — out of the CMake build; files kept on disk pending
  an explicit delete.
- [ ] Backward-path kernels disabled or unverified behind `REVIEW:` markers — `CudaSoftmaxOp.ixx:73`
  and `:103` throw `"needs review"` with the real calls commented out; `Gelu.Fp32.cu:65` records that
  the shipped backward is not the numerically stable `sech^2` form. Gradient-check these before the
  suite can claim backward coverage.
- [ ] `CudaResidualOp.ixx:116-117` — `input_A` / `input_B` are marked unused in the backward
  signature; either the contract is wrong or the parameters are dead.
- [ ] **Known-red CUDA tests (5) — beta-phase cleanup.** Surfaced by `x64-validate` ctest at the
  beta.1 cut (1417/1418 pass); all CUDA-path, so invisible to the CPU-only CI ratchet. Accepted
  non-blocking for beta.1; triage in the beta ladder (inference-path first):
  - `LinearCudaQuantizedTests.Forward_Fp4PrefillMatchesDecodeAcrossTokenMagnitudes` — FP4 prefill-vs-
    decode parity across token magnitudes; **inference / flagship FP4 path — triage first** (Gemma FP4
    validated token-for-token vs HF end-to-end, so likely a strict cross-path tolerance, not a live break).
  - `BpeTokenizerGemma.Encode_StartOfTurn_IsSingleAtomicToken` — **ROOT CAUSE FOUND** (marker at
    `Tests/Data/Tokenizers/Bpe/BpeTokenizer.Gemma.cpp:163`): the test is wrong, not the code. The Gemma 4
    tokenizer binary has no `<start_of_turn>` token — it uses `<|turn>` / `<turn|>`. Fix the assertion.
  - `RopeCudaTests.Backward_InverseRotationRecoversInput<Fp32>` — RoPE backward inverse-rotation recovery;
    backward/training path (inference uses forward only).
  - `LinearCudaTests.Backward_MatchesReferenceGradients<Bf16>` — BF16 Linear backward gradient match;
    likely backward-path tolerance/precision.
  - `DeviceRegistryTest.ThreadSafeDeviceOperations` — device-registry concurrency; confirm reproducible
    (suspect flaky) before chasing a real thread-safety gap.
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
  contract test (normalization, one-hot targets, shuffle-on-reset, IDX magic-number). Pin the TokenId
  signedness contract while there — `TokenSequenceLoader.ixx:44` records ids as semantically unsigned
  but stored `int32_t` to suit the CUDA encoder kernels.
- [~] Re-enable the AdamW path — `AdamW.Cpu.cpp` active with a convergence case; remaining:
  `AdamW.Cuda.cpp` companion + strip-vs-gate the `CudaAdamW.cu` / `CudaAdamWOptimizer.ixx:270` debug
  `printf`s in the same pass.
- [ ] Mixed-precision AdamW master parameters are zeroed, not copied — `CudaAdamWOptimizer.ixx:178`
  calls `zero( *master_param )` with the marker "For now, initialize to zero", so a mixed-precision run
  starts from zeroed masters instead of the current parameter values. Pair with the outdated
  precision-check / master-parameter logic flagged at `:169`.
- [~] **[net-new]** Training-loop integration test (sample-independent) — MNIST spine covered by
  `Network.Cpu.cpp`; remaining: a GPT-2-stack analogue for the Bard spine.
- [ ] **[net-new]** Optimizer step-convergence test — minimizes a known convex objective in N steps
  (proves update direction + bias-correction, not just that `step()` runs).
- [ ] **[net-new]** TrainingMode / RuntimeMode behavior coverage — assert build/runtime-mode
  transitions allocate/skip gradient buffers correctly (regression guard for the lifecycle fix). Three
  `REVIEW:` markers are the invariant to assert, each guarding a state the author believes unreachable:
  `TokenEmbedding.ixx:221` and `Lpe.ixx:187` ("if built and in training mode these buffers should
  always be initialized -- if not, it's a bug"), and `Lpe.ixx:495` ("must already be built").
- [ ] Fix the CUDA `fill_normal` / `fill_uniform` FP32-only gap (corrupts BF16 train-from-scratch
  init) — pair with a BF16 init-at-precision `TYPED_TEST` that turns the silent corruption red.
- [ ] **[decoupled]** Revive the loss + backward path (CrossEntropy / SoftmaxCrossEntropy) — both
  samples compute loss host-side, so off the critical path to a converging sample.
- [ ] **[net-new, training-only]** Revive the `Dropout` component from `Dev/Components/Regularization/`.
- [ ] ProgressReporter — an injected per-operation progress facility for long-lived ops (BPE vocab
  training, `PretrainedReader` load, load-time quantization). `BpeVocabulary.ixx:624` is the concrete
  call site: an inline every-100-merges elapsed-time print asking to become an async progress callback.
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
- [~] Confirm the docs-CI run green on GitHub Actions (Pages publish + pinned-Doxygen download).
  Blocker cleared: five relative markdown links added to `README.md` in `5503b59b` failed the
  `WARN_AS_ERROR` ratchet — Doxygen rewrites `[text](target)` into `\ref target` for any non-`http`
  target, and none of those `.md` files are in the Doxyfile `INPUT`. The beta.1 publish
  (run 29861454158) died there, so the live site is the 2026-06-09 Doxygen output and
  `/Mila/blog/` + `/Mila/api/` — both advertised in `README.md:339-342` — are 404.
- [~] Docs publish gate. RESOLVED (2026-07-24) that the site could only publish at a release:
  `docs.yml` now publishes from `dev` (path-filtered to `Web/**`, `Mila/Docs/Doxyfile`, the
  workflow) behind a structural + JSON-LD validation gate, and `web.yml` validates site changes on
  pull requests. STILL OPEN: a Doxygen doc-drift break from a `Src/**` or `README.md` change is not
  caught on the `dev` commit that causes it -- those paths deliberately do not trigger `docs.yml`
  (no auto-republish on every source commit). Add a non-deploying Doxygen check to
  `build-pipeline.yml` (no CUDA, no CMake) so it fails on the `dev` commit that causes it.
- [x] `Web/public/` and `.hugo_build.lock` are Hugo's generated output, committed to git (24
  files, including stale `public/writing/` paths from the rename to `blog`). CI builds to
  `build/site` and never reads them. Gitignore and untrack.

### Production Hardening

- [x] **`/W4` warning sweep, 252 -> 72 — and it surfaced a real defect.** `ModelArchive::close()`
  discarded `ZipSerializer::close()`'s `[[nodiscard]] bool` and wrapped it in a `try`/`catch` that
  could never fire (close reports failure by return, not by throwing), so a failed archive finalize
  — short write, I/O error on flush — set `closed_ = true` and reported success. **A truncated
  checkpoint was being reported as saved.** Fixed, along with the same swallow in `addMetadata()`
  and `ZipSerializer::open()` reopening over an archive that failed to close. Also cleared: 30 of 31
  C4702 (all one idiom — `if constexpr (cond) { throw; }` with the body trailing it rather than in
  an `else`, so the tail was unreachable instead of discarded), 4 C4189 dead locals, 7 C4834, 134
  C4100 in the `*.Dispatch.ixx` files, and ~160 C4018 test-loop counters left by the `dim_t` work.
  **C4100 turned out to be a stub census, not a style problem: 132 of 134 dispatch parameters were
  unreferenced because the function is an unimplemented or throw-only stub.** Unnamed parameters
  state that at the signature; the two genuine cases are `CudaLinearOp`'s `scales`, present for API
  parity with the quantized specializations.
- [ ] Finish the C4100 sweep in the op layer (~44 remaining, `Cuda*Op.ixx` / `Cpu*Op.ixx`). **Do it
  from the build's own file+line list, one site at a time — not with a regex.** A pattern matching
  `<tokens> <name> ( ... ) {` also matches `if ( status != cudaSuccess ) {`, so it comments out
  tokens inside conditions and mangles default arguments (`= nullptr` -> `= /*nullptr*/`); that
  attempt corrupted 112 sites across 43 files and was reverted. Leave `input_A`/`input_B` in
  `CudaResidualOp` named — their contract is an open question tracked above, not cosmetic debt.
- [ ] `GroupedQueryAttention.ixx:216` C4702 left deliberately: the `return` is unreachable because GQA
  backward is an unimplemented stub. The warning is honest reporting of a known-aspirational path and
  should clear itself when the GQA training path is built, not be suppressed.

- [x] External consumer builds against Mila via **FetchContent** (gate met); `find_package` PARKED
  (retired in place, `MILA_INSTALL` OFF by default).
- [x] Freeze the narrowest defensible export surface — RESOLVED: the umbrella is as narrow as C++23
  modules allow (a type in a public template's interface must be visible, not merely reachable, at
  instantiation). A `Mila.ixx` header contract records the rule.
- [x] Contributor onboarding — `CONTRIBUTING.md`, `getting-started.md`, `CODE_OF_CONDUCT.md`, and the
  rest of the GitHub Community Standards checklist in place; two `good first issue` issues opened. (The
  `dev -> master` default-branch flip was a stale note — `master` is and always was the default branch.)
- [~] Linux/clang first-class platform — WSL green, CI compiles under clang-21, container builds +
  runs Gemma 4 FP4. GCC 16 second oracle + broadened compiler matrix -> Future.
- [~] Reproducible container build — validated (clang-21 + gcc-15 host, CUDA 13.3); remaining: build
  against the bind-mounted tree, and CI building `FROM` the image rather than apt-installing.
- [~] Dispatch error UX — a missing `(Op, Device, Precision)` reads as one line, not a cascade. Core
  landed (declaration-only primary + `OperationSupported<...>` predicate); optional named kernel
  concepts + `OperationDispatch.md` §12 reconcile remain.
- [ ] Add the Samples build to CI (only tests build today).
- [ ] **`IExecutionContext` is exported but unreachable in practice.** `Mila.ixx` re-exports
  `Compute.IExecutionContext` and `Compute.ExecutionContextFactory` as public API, but no model
  factory accepts one — `GemmaModel/LlamaModel/GptModel::fromPretrained` take a `DeviceId`
  (`GemmaModel.ixx:119`) — and `Component` holds a *non-owning* pointer documented as "owned by the
  parent" (`Component.ixx:47`), so ownership parents up the component tree. Chat, the reference
  adaptor, never names either symbol. Decide: either a consumer genuinely can own a context (a
  `fromPretrained` overload taking `IExecutionContext*`, letting an application share one stream
  across models) or the two symbols should not be in the public umbrella. Surfaced 2026-07-25 while
  fact-checking a website claim that the application owns the execution context — it does not.
- [ ] `Mila/Samples/QuickStart/main.cpp:23` prints "framework initialized via find_package(Mila)" --
  wrong twice over, in the one sample whose job is to demonstrate consumption. Mila is a library, not
  a framework (MilaProductFamily.md), and `find_package` is PARKED with FetchContent as the supported
  path (this bucket, above). One-line copy fix.
- [ ] Guided reading path — one token's journey (embed -> attend -> sample -> decode) through the real
  source, readable by a strong C++ dev unaided.
- [x] Backfill the README **Gemma 4 flagship** perf numbers — prefill 1.14x behind llama.cpp, FP4
  decode 49 tok/s @32K (1.03x gap), from the published Discussion #17 measurements (RTX 4070, 12 GB).
- [ ] Published Docker runtime image — slim multi-stage GPU runtime, release-tagged, weights never baked in.
- [ ] Module import hygiene — Phase 0 exact-dup dedup, Phase 1 candidate report, Phase 2
  compiler-verified removal (Clang/GCC, not MSVC); plus domain-qualify generic single-segment module
  names (`Core`/`Utils`/`Components`/`Profiling` -> `Dnn.*`).
- [x] Marker-debt classify pass — all 89 `REVIEW:` markers (48 files) assigned to a class 2026-07-21
  and recorded in the task that owns each: Models 6, Test Suite Revival 14, Training Revival 7,
  Production Hardening 30, and 32 to the new **API Coherence** entry under Future. The markers
  themselves stay in source until their owning task resolves them.
- [ ] Delete the 16 `REVIEW:` markers whose disposition is already recorded — no analysis left, only
  removal: the 12 in `CudaGqa.Dispatch.ixx` answered by that file's own banner at `:36` ("retire in
  place as dormant training substrate"), plus `CudaOps.h:30` (declarations no longer needed),
  `Linear.cuh:83` (commented-out FP16 reductions), `Component.ixx:299` (commented-out accessor judged
  to add no value), and `CudaDeviceMemoryResource.ixx:139` (scoped to milestone Alpha.6, two stages
  stale).
- [x] Canonicalize `dim_t` for tensor-axis dimensions. Rule: **`dim_t` is the type of any value that
  describes a tensor axis — its extent, a position within it, or a count of its elements — at every
  API, config, component, and operation-interface boundary. Narrowing to `int` happens exactly once
  per call path, at the kernel launch site, through `narrowToKernelIndex()`. Kernel internals stay
  `int`; `size_t` never describes a dimension.** Landed: the three straggler configs (`Rope`, `Lpe`,
  `TokenEmbedding`) moved off `size_t`; the KV/positional interfaces (`IKvCacheLifecycle`,
  `IKvInference`, `IPackedKvInference`, `IPositionalDecode`, `IPositionalPairedOp`), the public
  `LanguageNetwork::decode`/`prefillFrom`/`rewindKvCache`, and the `LanguageModel`
  `maxSequenceLength`/`vocabSize` virtuals all widened; `xavier()` took `dim_t`; all six `REVIEW:`
  markers removed. `narrowToKernelIndex()` (`Tensor.Types.ixx`) is the single checked narrowing point
  — the margin is not theoretical, a Gemma 4 12B embedding table is ~1.0e9 elements against an
  `INT_MAX` of 2.1e9. Token ids are deliberately **out of scope** (a value, not an extent) —
  `TokenSequenceLoader.ixx:44` stays open under its own concern.
- [x] `Tensor::size()` returns `dim_t` — the last type in the dimension mix is gone. `ITensor::size()`,
  `Tensor::size()`, `size_`, `view_offset_` and both `view(shape, offset)` overloads moved over, along
  with `Component::parameterCount()` and all 15 overrides (a parameter count is an element count).
  `computeSize()` already returned `int64_t`, so this removed a silent per-construction conversion.
  **The `size_t` boundary, stated so it stays stable: `size_t` begins where element counts become
  bytes, or cross into a CUDA/std API. Mila-owned helpers that only forward an element count keep
  `dim_t`.** So `TensorBuffer` stays `size_t` throughout (allocation layer; its overflow guards depend
  on unsigned semantics), the `TensorOps` transfer/fill/math helpers carry `dim_t` and convert at the
  `cudaMemcpy` / `launch_*_kernel` edge, and `CudaTensorOps.Random` stays `size_t` because every
  consumer in it is curand or `cudaMalloc`. Two real defects fell out, not just type churn:
  `CudaLinearOp` narrowed the total element count to 32 bits **before** dividing by
  `cached_in_features_` (now divides in `dim_t`, narrows the quotient via `narrowToKernelIndex`), and
  the four `output_->size() < needed` capacity guards were comparing `size_t` against `int64_t`.
  Also swept 38 now-redundant `static_cast<dim_t>`/`<int64_t>` wrappers off config getters.
  **LESSON, and the reason this needed a rebuild rather than grep: changing a base virtual's return
  type silently un-overrides every stale override, leaving the class abstract — the error surfaces far
  away as C2672 at each `make_shared` site, not at the declaration.** Three test mocks
  (`HarnessComponent`, `MockChild`, `TestComponent`) were the entire blast radius; only four classes
  outside `Mila/Src` derive from `Component`/`ITensor` at all.
- [ ] Broaden CI compiler coverage toward the supported matrix (adds MSVC + GCC 16 to clang-21).
- [ ] Stage model weights off the Windows bind mount for the container (native disk speed).
- [ ] **[contributor]** Llama-lineage CPU ops (`RmsNormOp`, `SwigluOp`, `RopeOp`, `TokenEmbeddingOp`,
  `CrossEntropyOp`) in `OperationTraits.Cpu.ixx` — demand-driven; absence is zero-cost on the GPU path
  (full CPU parity is not a gate).
- [ ] **[deferred, measure first]** Remove FP16 (superseded by BF16) — woven through live code
  (`CudaDataTypeMap<half>`, `CudaLinearOp` half branches, `*_fp16` GQA/MHA/LPE stubs); trace
  live-vs-dead before removal. The trace is largely written: 8 `REVIEW:` markers already scope it —
  `CudaMhaOp.Dispatch.ixx:126,173`, `CudaLpeOp.Dispatch.ixx:18,105,152,173`, `CudaSoftmaxOp.ixx:79`,
  `CudaLinearOp.ixx:1068` (the last reading "we need only support bf16 for CUDA").

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
- **API Coherence** — the pre-1.0 consistency pass, and the precursor to any API-stability promise
  (RELEASING makes 1.0.0 a separate deliberate decision). 32 `REVIEW:` markers scope it, in four
  groups. *Construction:* factory design for tokenizers (`Tokenizer.ixx:45`), the half-baked
  `ComponentFactory` (`:30`), `GptTransformer::fromPretrained` vs `GptModel::fromPretrained`
  (`:123`, `:135`), ambiguous `LayerNormConfig` constructors (`:77`), `setParameters()` wanting
  weight+bias where only weight exists (`TokenEmbedding.ixx:421`, `Softmax.ixx:372`). *Naming:*
  `MemoryStats` `device_*`/`host_*` reading as the wrong axis (`:35`), `GptConfig::toString` bypassing
  getters (`:206`), `Rope.Config.ixx:120` max-sequence-length semantics. *Vitality — does this surface
  earn its keep:* `Tensor::getUId` (`:110`, `:588` — used only in tests), `CpuDevice.ixx:75`,
  `CompositeComponent.ixx:663` (no-op hook), `Network.ixx:335`, `Component.ixx:552`,
  `CudaMhaOp.ixx:758`, `Component.MemoryStats.ixx:122`. *Visibility:* `GroupedQueryAttention.ixx:73`
  and `MultiHeadAttention.ixx:66` agree that `initializeKVCache()` / `resetKVCache()` should become
  private behind a `friend class TransformerBase<>` once that common base exists — so this group is
  gated on the `TransformerBase<>` decision, not independent of it. *Placement / boilerplate:* where
  validation belongs (`Lpe.ixx:143`, `CudaGeluOp.ixx:89` wanting a shared helper, `CudaDevice.ixx:253`
  and `CudaHelpers.ixx:46` on redundant defensive checks), context casting repeated per-op
  (`CudaGeluOp.ixx:140`), allocation flags (`CudaPinnedMemoryResource.ixx:94`), dispatcher
  pass-through (`CudaRopeOp.Dispatch.ixx:128`), module grouping
  (`TensorOps.ixx:9`, `Tensor.Partitioning.ixx:12`), and two performance notes (`GemmaModel.ixx:512`
  double-copy on the token-id path, `LayerNorm.Fp32.cu:11` templating on training mode). Sibling of the
  `ComponentType` vitality question below; `GptModel.ixx:205` (hoist `onGenerating()` into the base)
  belongs with the Generation API tail above.
- **Architecture / MoE** — the presumptive post-v0.20 tentpole. Generalize `GatedMLP`'s gate
  (GeGLU/ReGLU) + the CPU `SwigluOp`; grouped `MoeOp` + `Router` + `MixtureOfExperts`; `LlamaBlock`
  delegating to `GatedMLP`. See `Specifications/FfnAndMoE.md`. Not a must for any single model, but the
  highest-leverage single investment: the niche Mila crests (best open model on a 16GB home card) has
  moved to sparse, and one router chassis unlocks three crests — the in-house Gemma 26B-A4B (control
  the reference, prove the machinery here first), **Qwen3-30B-A3B** (pure MoE, standard formats — the
  clean second test), and **gpt-oss-20b** (the first external crest; stacks the most distinct craft —
  MXFP4-native ingest, harmony channels mapping onto the Gemma channel streaming, attention sinks in
  the flash path). Chassis fit is ~70% there today (heterogeneous layers, sliding+full attention, GQA,
  RoPE axis, FP4, channel streaming); genuinely new is MoE dispatch + MXFP4-native weight ingest
  (`PerGroupMxFp4<32>`, E8M0 scales — checkpoint ships in fp4, so a load path that ingests nibbles
  directly, not the BF16->quantize-at-load assumption).
- **Training (advanced)** — Llama fine-tuning, loss-function GPU migration, gradient checkpointing,
  checkpoint save/restore, GQA training (the dormant expanded-layout substrate).
- **Performance** — Gemma 4 competitiveness levers (fused W4A16 prefill GEMM, flash-attention global
  prefill kernel, FP4 decode-matvec bandwidth), tensor parallelism, deterministic gradient
  accumulation. See `Specifications/GqaMemory.md`, `W4A16` design notes.
- **Native low-precision compute (Blackwell+)** — microscaling data path, finer per-arch gating
  (sm_120, CUTLASS 4.x), "compute precision as a first-class axis".
- **Compute backends beyond CUDA** — ROCm and Metal. `DeviceType::Rocm` / `::Metal` are reserved with
  `// FUTURE:` comments (`Mila/Src/Dnn/Compute/DeviceType.ixx:23`) and nothing else exists; `Device.ixx`
  docstrings already reference them. Per backend: memory resource, execution context, device layer, an
  `OperationTraits` partition, and the kernels. The component sources should not change — that is the
  claim under test. Hardware-gated (SPONSORING.md); publicly advertised there and in Discussion #7, so
  keep this entry honest about "reserved, not implemented".
- **Platform portability — aarch64 + coherent memory** — Mila has never been built on ARM (x86-64
  Windows/Linux only), so an aarch64 build is an unknown-size portability sweep of the same class as the
  Clang/GCC cross-compiler fixes. Carries three sub-threads: (a) a third arch gate beyond sm_89/sm_120;
  (b) container/published-image validation on an ARM Linux reference platform; (c) the coherent
  unified-memory question — memory resources and the mmap + pinned double-buffer loader assume discrete
  VRAM with explicit H2D staging, and a single-pool device has nothing to copy into. Scope (c) before
  assuming it is small: nobody has audited how deep the discrete-VRAM assumption runs.
- **Model loading** — load-time FP4 sidecar cache; concurrent / async read I/O for real queue depth.
- **Ungated GPT-2 zero-auth quick-start** — first-run HTTPS weights fetch (a runtime addition the
  freeze excludes). Freeze-compatible descope: host the pre-converted blob + a one-line download.
- **`ComponentType` vitality** — does `getType()` earn its keep, or retire the unused converter surface?
- **Python sample** — surface the `mila` binding as a standalone Python sample for the
  Python-majority audience. The binding already exists, so this is sample/doc work (no runtime
  feature) — a candidate beta.2 sprint rather than a hard v0.20 gate.
- **Discoverability (internal — not a README theme)** — site LIVE 2026-07-23 at **`mila.toddt.me`**
  (Cloudflare registrar + DNS, GitHub-issued cert, HTTPS enforced; the old
  `toddthomson.github.io/Mila` URL 301s to it). Landing page and writeups at the root, Doxygen
  demoted to `/api/`, one workflow and one artifact (Pages Source is "GitHub Actions", so the artifact
  *is* the whole site). The custom domain was taken immediately rather than deferred: a move resets
  accrued search signal, and the site was one day old with none — the cheapest moment it will ever be.
  Measured before the move: `Mila DNN` ranks #1 (the repo's `Src/Dnn/` tree and `Dnn.Components.*`
  module names are a large structural corpus), `Mila LLM` is unranked past page 4 (prose-only, and
  reframed only on 2026-07-20). Expect a trough: the `/api/` pages that carried the DNN corpus are now
  `noindex`. Open, in rough priority order:
  (a) **Verify `mila.toddt.me` in Google Search Console and submit the sitemap** — the highest-leverage
  remaining action. A new domain with no inbound links can sit undiscovered for weeks, and GSC is the
  only source of truth for which queries actually surface. Bing Webmaster Tools likewise.
  (b) **Duplicate content splits the writeups.** Every post carries a `discussion:` link and the same
  text lives on `github.com/.../discussions/N` — older, indexed, and on a far stronger domain. Google
  picks one; it will not pick us. Fix is editorial: trim each Discussion to a teaser plus a link to the
  canonical post on the site. Consolidates signal onto the domain we now own. Tooling landed
  (2026-07-24): a companion-thread template, `Tools/Blog/new_post_discussion.py` (opens the Discussion
  and writes its URL back into the post front matter), and a `Web/archetypes/blog.md` scaffold. In
  progress: #6 trimmed to a banner; #5 (CharLM) was an outlier — reworked into a new origin post
  (`/blog/charlm/`) rather than trimmed, its effusive AI-chat transcript to be shed from the thread.
  (c) **Revisit the `/api/` `noindex` once the authored pages have traction** — a sequencing call, not
  a permanent one. The original justification (a `robots.txt` cannot reach a subpath of a domain we do
  not control) died with the move to `mila.toddt.me`; the reason that survives is ratio. The build is
  1010 pages (487 class, 256 struct, 117 `dir_`, 51 member-index) against 16 authored ones — 98%
  templated output on a domain with no accrued authority, which is the thin-content pattern judged in
  aggregate. Established sites index API docs fine (cppreference, Boost); new ones should not lead with
  them. Note the trade is smaller than it looks: the DNN corpus ranks `Mila DNN` #1, but that query has
  no volume, and 743 pages of `Src/Dnn/` + `Dnn.Components.*` reinforce the *old* positioning while we
  are repositioning to LLM. Current marking is `noindex,follow`, so crawl paths stay open. When GSC
  shows the authored pages indexing, open **class and struct pages only** — never the `dir_` or
  member-index pages, which are pure navigation.
  (d) **Brand mark + share card — MARK DELIVERED 2026-07-26; share card still open.** The mark
  ships as `Web/static/mila-mark.svg`: the Achilles crest as an **a** (`#0a40c2`) beside a teal
  parallelogram as an **i** (`#0f9aa8`), reading "ai" with an M for Mila as the second reading.
  Landed in `baseof.html` as a `.lockup` (mark + wordmark as one object, scaled by `font-size`
  alone, `align-items: baseline` against a viewBox trimmed so the SVG's bottom edge IS the mark's
  baseline), in the hero above the h1, and as a full favicon set (`favicon.ico` 16/32/48,
  `icon.png`, opaque `apple-touch-icon.png`). Teal is now a second token: `--accent` carries
  structure, `--accent-2` everything clickable. CAVEAT, recorded deliberately: the a is a
  least-squares **trace of the 64px raster** (95.3% IoU), and that raster is clipped by its own
  canvas on the left edge and the right foot, so those edges are reconstructed rather than
  recovered — invisible on screen, not invisible in print or at banner size. Full colour and
  geometry record, and the directions rejected, in the session artefacts.
  STILL OPEN from this item: **`og:image` (1200x630) and flipping `twitter:card` to
  `summary_large_image`** — the share card was never cut; and the light-theme UI teal is a
  darkened `#0d818c` (4.64:1 for link text) which is deltaE 9.9 from the mark's own `#0f9aa8`,
  accepted 2026-07-26 as logo-ink-vs-interface-ink rather than resolved.
  Original framing, for the record: SUPERSEDES the 2026-07-23 decision
  ("the Achilles mark with the dot removed, no redesign"), reversed 2026-07-25 on two grounds: the
  original **vector source was never found** — the old business assets were purged, and everything
  in-repo is 64x64 raster (`Web/static/achilles.png`, `icon.png`), too soft for high-DPI and unable to
  make a 1200x630 `og:image` — and a mark that reads as a retired company's initial **A** undercuts a
  site whose whole ask is trust in the code. Design direction: keep the **"AI crest"** feel the current
  A suggests; the mark must read as Mila, not as a letter borrowed from elsewhere. Deliver as
  **vector**; do **not** ship a raster trace of the old mark. On landing: Mila-owned filename, update
  `<link rel=icon>` and the header `<img>` in `baseof.html`, restate the CSS comment that currently
  credits the accent colour to "the Achilles Software mark" (`--accent` `#0a40c2` is sampled from it —
  decide whether the palette follows the new mark), then add `og:image` and flip `twitter:card` to
  `summary_large_image`.
  Also found (2026-07-25): **fenced code blocks ignore the reader's theme.** `Web/hugo.toml` sets
  `markup.highlight.noClasses = true`, so Hugo emits per-token inline styles plus a hardcoded
  `background-color:#0d1117` -- every highlighted block renders dark on the light theme. The
  consequence is that the `.chroma .k/.c/.s/...` rules in `baseof.html` (which *are* theme-aware, via
  the `--k`/`--c`/`--s` custom properties) are dead code and have never applied. Flipping `noClasses`
  to `false` activates the CSS already written and makes code theme-correct sitewide; it changes the
  appearance of every post, so it is a taste call, not a drive-by. The landing-page snippet added
  2026-07-25 sidesteps this with an unlanguaged fence (no chroma, inherits `--code-bg`).
  Also found (2026-07-24): the Discussion->Hugo migration flattened structure in at least one post --
  emoji section-markers became plain lines and single-newline staccato collapsed into run-on
  paragraphs (GitHub hard-wraps single newlines; Hugo does not). Fixed in
  `lobotomized-attention-head-bug`; sweep the other eight for the same before promoting the site.
  Also in scope, independent of the site: retitle the Show-and-tell writeups so
  the technical subject leads (and fix the stray leading `#` rendering literally in #15 and #17), and
  rework the README's *second* paragraph to carry searchable vocabulary — the lead sentence stays
  exactly as it is, since the GitHub About line now matches it verbatim. Everything below is retained
  as rationale. Mila is effectively unfindable by search.
  A GitHub repo ranks on Google almost entirely through its README, and the current opening is brand
  copy ("at the metal", "explicit neural-network components") rather than anything a person types.
  The lead sentence stays as it is; the work is to make the *second* paragraph carry the vocabulary
  people actually search — running Gemma 4 locally, FP4 quantization on CUDA, LLM inference inside a
  12 GB card, a C++ alternative to llama.cpp — and to give section headings query-shaped names
  (`## Model Families` indexes against nothing). Secondary: the Discussion write-ups are the
  best-ranking assets here and their titles bury the technical subject (#15 and #17 also render a
  stray leading `#`). Ceiling is modest and worth stating up front — "Mila" is a contested term
  (Mila, the Quebec AI Institute, owns it outright), so long-tail
  technical queries are winnable and the brand word is not, and the backlink side runs into the
  no-social-media position. Marketing/positioning work: it never becomes a ROADMAP theme or a
  README-visible class.
