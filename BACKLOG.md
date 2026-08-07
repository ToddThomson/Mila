# Mila — Backlog

The open task list for the release in flight. Narrative and success criteria live in
[ROADMAP.md](ROADMAP.md); design rationale under `Mila/Specifications/`. **Completed work lives in
the git history** — the commit that landed it is the record.

Each `###` bucket is a v0.20 theme, its name matching the ROADMAP section (the only join).

**House rules.** An item is **three lines**: what, why it matters, `file:line`. Five if genuinely
complex. **Status lives in the checkbox** — `[ ]` open, `[~]` in progress — and never in the prose;
no dates, no "GREEN", no findings. **Done means deleted**, in the same commit as the work. Findings
worth reusing go to the owning spec or to memory, not here. Tags: **[gate]** blocks the release ·
**[deferred]** parked · **[contributor]** good-first-issue · **[crash]** reproduces as a crash ·
**[net-new]** authored from scratch, not revived · **[decoupled]** off the critical path.

**The size gate is lines per item, not total lines** — the failure mode is narrative, not item count.
Divide the lines in `## Current release` by the number of items in it; past **four** it has stopped
being a task list and needs a prune.

---

## Current release (v0.20.0)

### Models

- [ ] **KNOWN LIMITATION — the Llama chassis never received Gemma's memory gates.** The embedding and
  `lm_head` ignore the weight-quantization policy and are untied, so Llama 3.1 8B FP4 costs *more*
  than Gemma 4 12B FP4 (9.73 vs 8.65 GiB at 8192, widening to 12.08 vs 8.83 at 32768). Three fixes,
  each mirroring Gemma: pass the policy to `TokenEmbedding` (`Llama.ixx:117`), pass it to `lm_head`
  (`:119`, a deliberate quality call either way), and implement tying when `tie_word_embeddings` is
  set (matters most for the 1B/3B, tied upstream). Llama's `preatt`/`att` also span the full context
  where Gemma's collapse to the ring width — a separate defect, dominant at long context.
- [ ] **Gate B has no unquantized case.** Both footprint suites test FP4 only, so `NoWeightQuant` —
  the path a store name without an `-fp4`/`-fp8` suffix takes — has never been checked against
  `cudaMemGetInfo`. Add `llama-3.2-3b-it` at BF16: ~6.3 GiB, fits the 12 GB card, no spill.
- [~] **Attribute the Gate B residual.** Scratch is measured and is *not* the answer (~230 MiB on both
  models, essentially model-independent), leaving 1.015 GiB unattributed on Gemma and 0.449 on Llama.
  Next and cheap: per-allocation rounding — read `MemoryAllocationStats::allocationCount` (import
  `Compute.MemoryResourceTracker` directly; `Mila.ixx:95` comments the re-export out) and divide.
  Measurement noise floor is ~50-70 MiB, so nothing under ~0.1 GiB is signal.
- [ ] **Leaf-level Gate A for `Rope` is still unwritten**, and must not be a naive predict-vs-build
  equality: `RopeCacheRegistry` keys on (theta, max_seq_len, head_dim) and only the first owner
  allocates, so the assertion is registry-order dependent. Transformer-level dedup is in place.
- [ ] **GPT-2 has no `getRequiredMemory`**, so `gpt2-small` gets no pre-flight and Chat says nothing.
  Its footprint is the simplest of the three (no quantization policy, no ring, learned positional
  embeddings sized exactly `context_length`). Also gives the `generate()` crash below a budget to
  check against.
- [ ] **A pre-flight that cannot answer says nothing at all.** `Chat::predictFootprint` (`Chat.ixx`)
  catches every exception and returns `nullopt`; `reportFootprintBeforeLoad` then prints nothing, so
  an unreadable artifact header shows as silence followed by a confusing failure at load. One line at
  `verbose` naming the reason. See [[feedback_absent_output_is_evidence]].
- [ ] **`defaultContextFor()` is a compiled-in guess at the question the footprint API now answers.**
  `Chat.ModelCatalog.ixx` hard-codes 512 for Gemma, 1024 for GPT-2, 4096 otherwise, while
  `suggestFittingContext()` derives the answer in milliseconds and no VRAM. Keep the constant only as
  the no-CUDA fallback.
- [ ] **The fitting-context suggestion cannot be acted on from inside the session.** It advises editing
  the chat config JSON and restarting, immediately after measuring the answer. A `/context <n>` that
  reloads would close it; `switchModel` already proves release-then-reload works.
- [ ] **`/models` prices every row at the resident model's context, not at each row's own.** The
  header names the basis, but the same command answered 8192 before Gemma loaded and 512 after,
  moving `llama-3.1-8b-it` from 21.69 to 20.04 GB. A row's cost then depends on what happens to be
  loaded rather than on what loading *that* row would cost, which is the question the table exists
  to answer. Either price each row at its own default context, or fix the basis and say so.
- [ ] **Time a `/models` footprint probe.** Each row costs an artifact-header read plus a constructed
  graph, now up to three per row, and the number has never been taken. A dozen models on a 48-layer
  architecture is 36 constructions per keystroke; if it bites, put the column set behind a flag.
- [ ] **A per-row disk figure, if one ever returns, should be reclaimable bytes** — the blobs that model
  alone references. That is what deciding-what-to-delete wants, and prune's mark-and-sweep already
  computes the refcount; it is simply not exposed as a per-model query.
- [~] **Llama HF-parity regression test** — Gemma has `GemmaModel.Parity.Cuda.cpp`, Llama has none.
  Validate and record 3.1 8B FP8. Folds into Test Suite Revival's Llama-path backfill.
- [ ] **RoPE scaling is disabled on the Llama load path** — `Llama.ixx:703` has
  `.withRoPEScalingFactor( metadata.rope_scaling )` commented out for a reason recorded as unclear.
  3.1 8B's extended context depends on it; resolve *before* writing the 8B parity test.
- [ ] **Triage `Llama.Block.ixx:132` view-aliasing** — the Q/K/V splits of `qkv_out` may not be
  contiguous. Confirm live-vs-benign and fix if live, before claiming Llama HF validation.
- [ ] Tool calling validated on Llama 3.2 3B and 3.1 8B Instruct.
- [ ] **GQA standalone-`forward()` stub** — component-level Gemma/Llama attention has no independent
  correctness oracle, and `GroupedQueryAttention.ixx:177` returns an un-computed `output_view_` on an
  unreached branch. Precondition for retiring the legacy GQA path; clears the C4702 below with it.
  See `Specifications/GqaMemory.md`.
- [ ] **GQA `forward()` fallback is stale** — `GroupedQueryAttention.ixx:299` records the non-KV-cache
  fallback as needing a correctness review, with the shape derivation commented out beneath it.
- [ ] `CudaMhaOp.ixx:433` initializes `active_max_seq_len_ = T_` with the reason unrecorded — confirm
  against the two-phase KV-cache contract (prefill full sequence, decode `outer_size == 1`).
- [ ] `GptModel.ixx:386` hardcodes `eos_token_ = 50256` — should come from tokenizer metadata.
- [~] **[gate, crash] Verify the `generate()` context-overflow guard.** Gemma's bound check is now
  ported to `GptModel.ixx` and `LlamaModel.ixx` — both return `GenerateStatus::ContextOverflow`
  instead of decoding past `context_length_`. Unbuilt and unrun. Reproduce with `gpt2-small` (it is
  in the store) at a 1024 context, generating past ~1005 tokens with Chat's clamp bypassed — the
  number the original crash was observed at.
- [ ] **[contributor]** Llama 3.2 1B/3B weight tying — the aliasing plumbing shipped; add
  `tie_word_embeddings_` + post-load aliasing + `getMemoryStats` correction to `LlamaTransformer`.
  See `Specifications/WeightTying.md` §6.

### Test Suite Revival

- [~] **Re-green the authored component / tensor / tokenizer suites to the current API.** Concrete
  component classes are re-enabled and build-green; `SoftmaxCrossEntropy` is parked for the
  loss-on-device work, and 3 backward-numeric cases are `GTEST_SKIP`'d pending the filed bugs below.
- [~] **Core `Tensor.ixx` coverage to the value-type archetype.** Remaining: the `TensorOps.Transfer`
  device split, and the wider `Tensors/` tree (`TensorBuffer`, `TensorDataType*` maps, `Partitioning`,
  `Serialization`). Eight `REVIEW:` markers name the exact contracts to pin — see
  `Specifications/Testing.Tensors.md`.
- [~] **Backfill inference-drought coverage.** `OperationTraits` dispatch is done; remaining are the
  load-time quantization white-box (`PerChannelFp8`/`PerGroupFp4`, the decode matvec kernels — the one
  legitimate op-layer test, unreachable through the public component) and the Llama path.
- [~] **Re-green in sample-revival order** — MNIST spine mostly landed. Remaining: the
  `Core/Network.cpp` delta, the GPU companions (`Network.Cuda`/`AdamW.Cuda`), then the Bard GPT-2 tail.
- [ ] **Backward-path kernels disabled or unverified.** `CudaSoftmaxOp.ixx:73` and `:103` throw
  `"needs review"` with the real calls commented out; `Gelu.Fp32.cu:65` records that the shipped
  backward is not the numerically stable `sech^2` form. Gradient-check these before the suite claims
  backward coverage — and sweep the *unmarked* backward kernels too, per-precision twin by twin: the
  RoPE FP32 backward was arithmetically wrong while its BF16 sibling was correct, in a file carrying
  no marker at all.
- [ ] **`ResidualConfig` advertises a scaling factor that no backward implements and the two devices
  disagree about in forward.** CUDA forward honours it, CUDA backward takes no scale, and the CPU op
  ignores it entirely; the only guard is a **debug-only** assert at `CudaResidualOp.ixx:106`, so
  release builds train silently wrong. Cheapest correct fix, freeze-compatible because it removes an
  unimplemented knob: have `validate()` reject `scaling_factor != 1.0f` (`ResidualConfig.ixx:97`).

### Training Revival

- [~] **Data-loader contract tests** — `TokenSequenceLoader` done; remaining is the `MnistDataLoader`
  contract (normalization, one-hot targets, shuffle-on-reset, IDX magic number). Pin the TokenId
  signedness contract while there — `TokenSequenceLoader.ixx:44`.
- [~] **Re-enable the AdamW path** — `AdamW.Cpu.cpp` is active with a convergence case. Remaining: the
  `AdamW.Cuda.cpp` companion, plus strip-vs-gate the debug `printf`s in `CudaAdamW.cu` and
  `CudaAdamWOptimizer.ixx:270` in the same pass.
- [~] **[net-new]** Training-loop integration test (sample-independent) — MNIST spine is covered by
  `Network.Cpu.cpp`; remaining is a GPT-2-stack analogue for the Bard spine.
- [ ] **[net-new]** Optimizer step-convergence test — minimize a known convex objective in N steps, so
  the update direction and bias correction are proven rather than just that `step()` runs.
- [ ] **[net-new]** TrainingMode / RuntimeMode coverage — assert that mode transitions allocate and skip
  gradient buffers correctly. Three `REVIEW:` markers are the invariant to assert, each guarding a
  state believed unreachable: `TokenEmbedding.ixx:221`, `Lpe.ixx:187`, `Lpe.ixx:495`.
- [ ] **[decoupled]** Revive the loss + backward path (CrossEntropy / SoftmaxCrossEntropy) — both
  samples compute loss host-side, so this is off the critical path to a converging sample.
- [ ] **[net-new, training-only]** Revive the `Dropout` component.
- [ ] **Validation** — the **FP32** training path proven by the primitive suite (gradient checks,
  step-convergence, loader contracts, init-at-precision, the integration test), CI-gated; samples run
  as demos. BF16 and GQA training move to the Training (advanced) release.

### API Documentation

- [ ] **Tier 3 — semantic staleness** (retired-world prose). Folded into Test Suite Revival: fix a
  file's prose while it is already open for re-greening.
- [~] **Confirm the docs-CI run green on GitHub Actions** (Pages publish + pinned-Doxygen download).
  The beta.1 publish died on the `WARN_AS_ERROR` ratchet, so the live site is the 2026-06-09 output
  and `/Mila/blog/` + `/Mila/api/` — both advertised at `README.md:339-342` — are 404.
- [~] **Docs publish gate.** `docs.yml` publishes from `dev` behind a validation gate, but a Doxygen
  doc-drift break from a `Src/**` or `README.md` change is not caught on the commit that causes it —
  those paths deliberately do not trigger it. Add a non-deploying Doxygen check to
  `build-pipeline.yml` (no CUDA, no CMake).
- [ ] **`Mila/Bindings/README.md:7` says the binding "knows nothing about HTTP, chat, or any wire
  protocol"** — true before the store reached Python, wrong now: `Mila_py.cpp` exports `HttpResponse`,
  `HubModel`, `ModelStore` and an `HttpFetchDelegate`. It is the one paragraph explaining why the
  binding sits beside `Src` rather than under `Adaptors/`, so the stale sentence undermines the
  boundary it exists to draw.

### Production Hardening

- [ ] **One concept, two names: the compute-precision template parameter is `TPrecision` in most of the
  tree and `TComputePrecision` in nine files.** Same axis, split along no principle — `Linear` and
  `GroupedQueryAttention` differ from their own siblings — and it cost two compile errors in one
  session. Rename to the majority spelling: 126 occurrences, 9 files, 0 in `Mila/Tests`. Not a blind
  sweep — `GroupedQueryAttention.ixx` and `CudaRopeOp.ixx` use *both* and need hand work. Related:
  CLAUDE.md mandates `TWeightQuantization` and the code says `TWeightQuant`.
- [ ] **`getStorageSize` is implemented three times** — `Mila::Dnn::detail::getStorageSize`
  (`Tensor.ixx:81`, carrying a `REVIEW:` that already asks why), `Detail::getStorageSize`
  (`TensorBuffer.ixx:221`), and `Mila::Dnn::storageBytes` (`Component.MemoryStats.ixx`). The two
  namespaces differ only in case. Blocker: `Tensor.ixx` cannot import `Dnn.Component` without a cycle.
- [ ] **Isolate third-party warnings structurally** with `/external:I` + `/external:W0` (`-isystem` for
  Clang/GCC). The real target is warnings from third-party header text pulled into Mila's own TUs, not
  their sources (`/W4` at `Mila/CMakeLists.txt:87` is `PRIVATE` and never reached them). Precondition
  for any warnings-as-errors gate. Budget for two frictions: those headers enter through module global
  module fragments, and `/external:` does nothing for nvcc diagnostics.
- [ ] **`save_` is public on `Component` and protected on `CompositeComponent`.** Legal, but the
  accessibility of one virtual then depends on the static type you hold, and a caller holding a
  concrete composite cannot invoke it (C2248, worked around with an `exposeSave()` forwarder in
  `Tests/Dnn/Core/CompositeComponent.cpp`). The trailing underscore suggests non-public is the intent,
  making `Component.ixx:407` the declaration that is wrong.
- [~] **GPT-2 and Llama 3 pre-tokenization silently runs the ASCII fallback on every MSVC build.** Both
  canonical patterns use `\p{L}`/`\p{N}` (`BpePreTokenizationMode.ixx:33`, `:57`), MSVC's `std::regex`
  does not implement them, so `BpeTokenizer.ixx:344` throws on **every** construction and takes the
  approximation branch. The warning is now emitted once per process, which makes it visible but not
  correct. No parity test catches it — ASCII tokenizes identically either way. The real fix is a
  dependency decision: a real regex engine (PCRE2/RE2) or hand-rolled Unicode class matching.
- [ ] **`CudaManagedMemoryResource.ixx:85` builds a detailed error message then throws a bare
  `std::bad_alloc`**, discarding it; `CudaPinnedMemoryResource.ixx:101` throws with no message at all.
  `CudaDeviceMemoryResource` gets this right — align both on `CudaBadAlloc` so an OOM says which
  device, which size, which resource.
- [ ] **`GroupedQueryAttention.ixx:216` C4702 is left deliberately** — the one warning in the tree. It
  self-clears when the GQA training path is built, where a suppression would have to be remembered.
  **Note for the warnings-as-errors decision:** a blanket `/WX` would force it silent; escalating only
  the defect-class codes leaves it visible.
- [ ] **`IExecutionContext` is exported but unreachable in practice.** `Mila.ixx` re-exports it and
  `ExecutionContextFactory` as public API, but no model factory accepts one (`GemmaModel.ixx:119`
  takes a `DeviceId`) and `Component` holds a non-owning pointer owned by its parent. Decide: a
  `fromPretrained` overload taking `IExecutionContext*`, or drop both from the umbrella.
- [ ] **If C1128 recurs** on `MilaTests`, `ProfileModel` or `ExportArtifact`, switch from the per-target
  `/bigobj` on `ChatApp` to one project-wide `add_compile_options`. **Todd's call** — it touches every
  target's flags, so it was deliberately not taken unilaterally.
- [~] **[gate] Confirm both Linux CI jobs go green on the next push to `dev`.** The two causes are
  fixed — `libssl-dev` for curl's `find_package(OpenSSL)` on the CUDA image, and the CPU-only break the
  configure failure masked (`Component.ixx` imported `Compute.CudaPinnedMemoryResource` unguarded; the
  staging resource is now `DeviceTypeTraits<TDeviceType>::host_staging_memory_resource`). A whole-graph
  scan finds no remaining unguarded CUDA import, but that proves the import graph, not the compile.
- [~] **`MILA_ENABLE_LIBCURL=OFF` now rides the CPU-only CI job** — the Linux wheel's configuration,
  which had no Linux coverage anywhere (the only OFF preset is a *Windows* Debug build,
  `CMakePresets.json:147`). Green on that job is what closes this; it has never run.
- [ ] **`CMakeLists.txt:266` pins curl at 8.11.1 under a `REVIEW:` marker naming 8.21 as current.** A
  vendored TLS-adjacent dependency in a published binary is the one pin where staleness has a security
  cost. Decide the bump or record why 8.11.1 stands.
- [ ] **[gate] PyPI advertises Linux and ships only `win_amd64`.** `pyproject.toml:37` carries
  `Operating System :: POSIX :: Linux` and the sole published file is
  `mila_llm-0.20.0b2.dev20-cp313-cp313-win_amd64.whl` — no Linux wheel, no sdist, so `pip install` on
  Linux fails with nothing to fall back to. Release metadata is immutable, so the live page stays wrong
  until the next release carries the Linux wheel below.
- [~] **[gate] The WINDOWS wheel still has no clean-room run.** Linux is done — `python:3.13-slim`
  under Docker with `--gpus all` is a genuine clean room (the driver is injected, the Toolkit is not),
  and all six CUDA runtime libraries resolved from site-packages. Windows cannot be tested locally:
  Windows 11 Home has no `Containers` or Hyper-V feature at all (measured, not assumed). Sequenced at
  the beta.2 release — `workflow_dispatch` needs `wheel-cleanroom.yml` on `master` first.
- [ ] **The `>=3.13,<3.14` wheel pin is an accident, not a floor** — `Docker/build-mis.sh:60` says it
  "exists to match the committed cp313 Windows binding". The binding uses no CPython API directly
  (pure pybind11, floor 3.8) and `__init__.py` needs only 3.9, so the real floor is **3.9**. cp312 is
  ~28-31% of PyPI downloads against 3.13's ~13% and 3.14's ~4-6%. Widening means a range plus a second
  interpreter in `pyproject.toml:14`, `CMakePresets.json:180`, `Dockerfile.wheel:48`,
  `build-wheel-windows.ps1:21` — nothing has yet compiled against 3.12 or 3.14. Todd's call, pending.
- [ ] **`Mila/Tools` has no off switch** — gated on `PROJECT_IS_TOP_LEVEL` alone
  (`Mila/CMakeLists.txt:1081`), so the wheel configure builds `tokenize` and `ExportArtifact`, neither
  of which can go in a wheel. Every other subdirectory has a `MILA_ENABLE_*`; this one costs build time
  on an artifact that discards it.
- [ ] **ProgressReporter** — an injected per-operation progress facility for long-lived ops (BPE vocab
  training, `PretrainedReader` load, load-time quantization). `BpeVocabulary.ixx:624` is the concrete
  call site: an every-100-merges elapsed-time print asking to become an async callback.
- [ ] `Version::getMajor()`/`getMinor()`/`getPatch()` are non-const (`Src/Version.ixx`), so the
  version-skew comparison needs a mutable copy.
- [~] **Linux/clang as a first-class platform** — WSL green, CI compiles under clang-21, the container
  builds and runs Gemma 4 FP4. The GCC 16 second oracle and the broadened matrix move to Future.
- [~] **Reproducible container build** — validated on clang-21 + gcc-15 host, CUDA 13.3. Remaining:
  build against the bind-mounted tree, and have CI build `FROM` the image rather than apt-installing.
- [~] **Dispatch error UX** — a missing `(Op, Device, Precision)` reads as one line, not a cascade.
  Core landed; the optional named kernel concepts and the `OperationDispatch.md` §12 reconcile remain.
- [ ] **Five files still hand-roll the staging memory resource `DeviceTypeTraits` now carries.** Each
  writes `#ifdef MILA_HAS_CUDA` plus a `conditional_t` (or a guarded `if constexpr`) that is exactly
  `host_staging_memory_resource`: `Gemma.Block.ixx:820`, `Gemma.ixx:527`, `Llama.ixx:484`,
  `GptTransformer.ixx:615`, `GemmaModel.ixx:110` (and `LlamaModel.ixx`). Converting them removes six
  preprocessor blocks from module purviews — see [[feedback_no_ifdef_in_modules]].
- [ ] **Module import hygiene** — Phase 0 exact-duplicate dedup, Phase 1 candidate report, Phase 2
  compiler-verified removal (Clang/GCC, not MSVC), plus domain-qualifying the generic single-segment
  module names (`Core`/`Utils`/`Components`/`Profiling` -> `Dnn.*`).
- [ ] **Delete the 16 `REVIEW:` markers whose disposition is already recorded** — no analysis left, only
  removal: the 12 in `CudaGqa.Dispatch.ixx` answered by that file's own banner at `:36`, plus
  `CudaOps.h:30`, `Linear.cuh:83`, `Component.ixx:299`, `CudaDeviceMemoryResource.ixx:139`.
- [ ] **`Mila/Samples/QuickStart/main.cpp:23` prints "framework initialized via find_package(Mila)"** —
  wrong twice over, in the one sample whose job is to demonstrate consumption. Mila is a library, and
  `find_package` is parked with FetchContent as the supported path. One-line copy fix.
- [ ] **Guided reading path** — one token's journey (embed -> attend -> sample -> decode) through the
  real source, readable by a strong C++ dev unaided.
- [ ] Add the Samples build to CI (only tests build today).
- [ ] Published Docker runtime image — slim multi-stage GPU runtime, release-tagged, weights never baked in.
- [ ] Broaden CI compiler coverage toward the supported matrix (adds MSVC + GCC 16 to clang-21).
- [ ] Stage model weights off the Windows bind mount for the container (native disk speed).
- [ ] **[contributor]** Llama-lineage CPU ops (`RmsNormOp`, `SwigluOp`, `RopeOp`, `TokenEmbeddingOp`,
  `CrossEntropyOp`) in `OperationTraits.Cpu.ixx` — demand-driven; absence is zero-cost on the GPU path.
- [ ] **[deferred, measure first]** Remove FP16 (superseded by BF16) — woven through live code; trace
  live-vs-dead first, and 8 `REVIEW:` markers already scope it. Note the odd row it collides with:
  **CUDA `LayerNormOp` is registered at FP32 and FP16 and *not* BF16**, so deleting the FP16 row leaves
  CUDA LayerNorm FP32-only. Pinned by a `static_assert` so this work must confront it.

### Model Distribution

- [ ] **`Mila/Samples/Python` has no sample that pulls, and its README describes a retired world.**
  The binding exposes seven store methods; the samples drive three. The README says "Two samples"
  (there are three — `store.py` is unlisted), says a wheel is "post-v0.20 work" at line 96 while line
  20 tells you to `pip install mila-llm`, claims "no weight download", and omits every distribution
  type from its binding table.
- [ ] **There is still no headless pull.** Chat now opens on an empty store, so a clean machine can
  reach `/install`, but the only thing in the product that pulls is an interactive command — which is
  why the cold download can only be exercised by hand. A `pull` verb on the tool would make the gate
  below testable without a human at a prompt, and it is the one store verb `ExportArtifact` lacks.
- [ ] **`/models --online` still cannot answer "will it run here".** Download size now comes from each
  manifest, but the fit question — the one `/models` answers for installed rows, `!` marker and all —
  needs a real footprint. Take it from a `Range` read of the safetensors header (8-byte length then
  JSON, both at the file's start) so the online row uses the *same* code as the installed row and one
  number means one thing; an estimate in that column would quietly cost the table its credibility.
  The transport is proven; the blocker is that the footprint path takes a path, not a byte range —
  a `Mila/Src` change, which is what makes this the one online-listing item that is not adaptor work.
- [ ] **`/models --online` costs one GET per listed model.** Invisible at one model, N+1 requests at N.
  Only worth revisiting if the published set grows; noted so the cause is known when it does.
- [ ] **Project distribution into the Python binding — steps 2b-4.** Decided (option C): one `pull`,
  two transports; Python supplies bytes, not procedure. Step 2b is `from_store( name, context_length,
  device_index )` on both sessions plus `BpeTokenizer.from_store()`, which kills the path-pairing and
  MIS's family branch and needs no transport; then `mila.store` / `mila.hub`; then MIS onto it,
  retiring `MILA_MODEL_PATH`/`MILA_TOKENIZER_PATH`. Watch: release the GIL inside the sink or the
  transfer serializes, and `py::bytes` copies where a `py::buffer` does not — at 6.35 GB that matters.
- [ ] **`NOTICE.md:33` omits curl, and may no longer need to.** The note treats notice-carrying as open
  for "a binary distribution that links them" — but **both** wheel presets are now
  `MILA_ENABLE_LIBCURL=OFF`, so a wheel built today contains no curl at all. Establish whether the
  *published* artifact predates that change before writing anything: the answer decides whether this is
  an obligation or a non-issue. The same note points at a bucket that no longer exists; fix that either way.
- [ ] **The published wheel still teaches the retired form.** Its README and `__init__.py` docstring
  instruct users to pair `gemma4_12b_it_bf16.bin` with `gemma_tokenizer.bin`, and
  `LlamaModel.from_pretrained` still takes a `quantize_fp8` **boolean** — FP8-only, no FP4 — where the
  store carries quantization in the name. Fix with step 2b, not before.
- [ ] **`mila/__init__.py` is copied by a `POST_BUILD` step of a target it is not a source of.**
  `Mila/Bindings/CMakeLists.txt:63,83,101` stage it with `copy_if_different` off
  `add_custom_command(TARGET MilaPy POST_BUILD)`, which runs only when `MilaPy` relinks — so editing
  only `__init__.py` leaves every staged copy stale and the sample fails with a missing attribute.
  Use `add_custom_command(OUTPUT ...)` with `DEPENDS` on the source.
- [ ] **[gate] Migrate the remaining Llama and GPT-2 rows into the store, then delete the models-directory
  fallback branch.** The catalogue is gone and Gemma plus Llama 3.2 3B load from the store; seven rows
  still resolve through the `REVIEW:`-marked loose-file path. `.bin` leaves the catalogued set with the
  branch, **not the reader** — `PretrainedReader.ixx:229` sniffs the magic and would strand every
  `.bin` on disk. *Done when:* a clean machine pulls and runs Gemma 4 through named commands.
- [ ] **The licensing story is per-family and must not be generalized.** Gemma 4 is Apache 2.0 (public,
  ungated); Gemma 3 and earlier carry the Gemma Terms of Use; **Llama 3.1/3.2 may be republished, but
  attributed** — ship the agreement, display "Built with Llama" and Meta's notice, pass along the AUP,
  and begin the model name with "Llama" (`llama-3.1-8b-it-fp4` already does). Gating is a *policy*
  choice, not a licence condition. See [[project_gemma4_apache2_license]].
- [ ] **The `mila-llm` organization has no organization card** — it is the landing page for anyone
  following a coordinate, and it is currently HuggingFace's placeholder. Needs: what a Mila artifact
  is, that it is loadable only by Mila and deliberately not NVFP4/MXFP4, the coordinate form, and the
  link to mila.toddt.me. See [[project_positioning_reference_impl]] — never lead with throughput.
- [ ] **`ExportArtifact` names one of its nine modes, and its verbs wear option syntax.** Rename the
  binary to `modelmgr` and convert the modes to subcommands (`export`, `transcode`, `package`,
  `validate`, `install`, `rename`, `compare`, `fingerprint`, `fetch`). `--package` is today both a mode
  and an option of export mode, a collision the code has to comment on at `ExportArtifact.cpp:212`.
  Sequence it **after** the gate chain — `--fetch` is load-bearing until the cold download is green.
- [ ] **Packaging then installing hashes every file twice** — `buildPackage` hashes to derive the
  manifest digests and `install` hashes again to verify adoption (~50 s of the ~60 s Llama 3B
  migration, ~2 minutes on the 8B). Neither check is wrong alone, so the fix is a combined verb.
  `publish_model.py` has the same defect for its own reason.
- [ ] **`prune()` is destructive on a store that predates records.** Every pre-record blob is by
  definition unreferenced, so a first sweep on an upgraded store reclaims all of it — 6.33 GB in the
  case actually observed. Blobs-with-zero-records is a recognizable state and should be reported
  rather than silently swept.
- [ ] **`isAbandoned()`'s 24-hour lock reclamation is untested** — it needs a file with a backdated
  write time. Make the threshold a constructor parameter so a test can set it to zero; that is a better
  shape than backdating with `last_write_time()`.
- [ ] **The `fopen` -> `<fstream>` conversion is still available in three modules** — `SafeTensors.ixx`
  and `TokenSequenceLoader.ixx` are straight swaps and are now the library's only source of C4996.
  **`PretrainedReader.ixx` is not**: it deliberately uses positioned `ReadFile`/`pread` alongside the
  mapping because faulting a large model through the mapped view throttles below disk bandwidth — that
  one needs the exemption. Clearing the first two unblocks the warnings-as-errors ratchet.

### Product Family — Adaptor Validation

- [ ] **`ToolCallParser::parse` routes ANY response containing `[` into the tool-call parser** —
  `Chat.ToolCallParser.ixx:63` uses `response.find( '[' )` where the class's own doc comment at `:35`
  says "Leading `[`" and the nested `parseTagged` path at `:109` tests it correctly. Found on an
  ordinary Llama 3B turn. It degrades gracefully today, but any prose with a bracket (markdown links,
  `[1]` footnotes, an array literal) enters the path, and a parse that ever *succeeds* on prose would
  swallow the answer and emit a phantom tool call.
- [ ] **The logger writes over the spinner** — `Logging` writes to the console independently of
  `ConsoleRenderer`, which owns that line, so a model switch renders the warning on top of the spinner.
  Cosmetic, but it is the first thing a user sees on every switch that logs.
- [~] **MIS Gemma 4 tool-calling validated end-to-end** — Codex and Claude Code CLI round-trips are
  live and the native grammar is reconciled to Google's canonical template, pinned by an oracle.
  Remaining: N sequential distinct tool calls in one turn, channel-content parser polish, and
  Codex-CLI re-validation on the reconciled grammar.
- [~] **Grammar-in-runtime execution-time scope call** — the C++ and Python grammars are held together
  by a cross-language parity test. Open for sign-off: single-source via pybind, or close on the test.
- [ ] **In-turn thoughts dropped between tool calls** — Google's multi-turn rule is to strip
  prior-turn thoughts and keep the current turn's.
- [ ] Buffer Gemma Anthropic streaming only when tools are present.
- [ ] **Chat reports "Thinking: balanced" for models that have no thinking mode.** `show_thinking` is a
  session-config flag, but only Gemma routes a reasoning channel — the welcome banner and `/model`
  show an effort level for Llama and GPT-2 regardless, reading as a capability they lack. The banner
  prints it beside `Model: none` too, which is an effort level for a model that does not exist.
- [ ] **`main.cpp` re-checks what the store already guarantees** — after `resolveModel` succeeds it
  tests `exists()` on both paths, but `locate()` refuses an incomplete record. Harmless duplication,
  except `/model` has no equivalent check; if the guarantee is doubted, the check belongs in the store.

---

## Future

Next-cycle work. Coarse by design — detailed tasking happens only when an item promotes into a release.

- **Qwen 3** (presumptive next release) — the dense decoder, thinking-mode suppression, model-agnostic
  tool calling, and FP8 KV cache; the `OperationTraits<GqaOp, Cuda, BF16, PerChannelKvFp8<>>`
  specialization lands here.
- **Architecture / MoE** — the presumptive post-v0.20 tentpole; one router chassis unlocks Gemma
  26B-A4B, Qwen3-30B-A3B and gpt-oss-20b. See [[project_moe_tentpole_direction]].
- **Gemma 4 MTP** — the self-speculative drafter, sequenced ahead of MoE.
- **Ministral** — SWA transformer; reuses the Llama foundation, Qwen 3 tool calling, and the Gemma 4
  SWA mask + bounded-KV ring.
- **v0.20 library-frozen tails** — the Generation API surface tail (`SamplerConfig` rename, Llama/Gpt
  seedable sampling, eager sampler, config-accessor propagation, `contextLength()` hoist), the
  Sample-API device-sampler migration for Llama/Gpt, and the Optimizer-dispatch migration onto
  `OperationTraits`. All `Mila/Src`, which is why they wait. Adaptor work does not.
- **Model serialization** — the remaining checkpoint round-trip and distribution-artifact phases.
  Design, defect analysis and the phase plan are in `Specifications/ModelSerialization.md`.
- **API Coherence** — the pre-1.0 consistency pass, and the precursor to any API-stability promise.
- **Warnings-as-errors ratchet.** Constraints worth keeping: it requires the `/external:W0` isolation
  first; enforce in **CI only**, never locally; ratchet on the count *not increasing* before demanding
  zero; **MSVC first**, since `/WX` across three compilers means the union of three opinions must be
  zero; and dormant-but-retained code warns by nature — suppress per-file in CMake pointing at the
  owning task, never with `#pragma warning` in module code. Land it **after** v0.20 ships.
- **Parallel range downloads for model retrieval — MEASURED, and closed.** One connection pulled 6.33
  GB in 10-15 minutes on a 100 Mbps line, against a 9.1-minute theoretical floor: the single stream
  already saturates the link, so there is no headroom for concurrency to recover. The earlier LM Studio
  comparison (~2 hours, same connection) was measuring that client, not the ceiling. Do not implement.
- **Training (advanced)** — Llama fine-tuning, loss-function GPU migration, gradient checkpointing,
  and BF16/GQA training.
- **Performance** — the Gemma 4 competitiveness levers: the fused W4A16 prefill GEMM and
  flash-attention on the global layers. See [[project_w4a16_prefill_gemm]].
- **Native low-precision compute (Blackwell+)** — the microscaling data path and finer per-arch gating.
- **Compute backends beyond CUDA** — ROCm and Metal; `DeviceType::Rocm` / `::Metal` are reserved and
  unimplemented.
- **Platform portability — aarch64 + coherent memory.** Mila has never been built on ARM.
- **Model loading** — a load-time FP4 sidecar cache, and concurrent/async read I/O for real queue depth.
- **Ungated GPT-2 zero-auth quick-start** — a first-run HTTPS weights fetch.
- **`ComponentType` vitality** — does `getType()` earn its keep, or does the unused converter surface retire?
- **Discoverability** (internal, not a README theme) — the site is live at `mila.toddt.me`.
