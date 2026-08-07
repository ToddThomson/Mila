# Mila — Backlog

The open task list for the release in flight. Narrative and success criteria live in
[ROADMAP.md](ROADMAP.md); design rationale under `Mila/Specifications/`. **Completed work lives in
the git history** — the commit that landed it is the record.

Each `###` bucket is a v0.20 theme, its name matching the ROADMAP section (the only join).

**House rules.** An item is **three lines**: what, why it matters, `file:line`. Five if genuinely
complex. **Status lives in the checkbox** — `[ ]` open, `[~]` in progress — and never in the prose;
no dates, no "GREEN", no findings. **Done means deleted**, in the same commit as the work. Findings
worth reusing go to the owning spec or to memory, not here. Tags: **[gate]** blocks the release ·
**[deferred]** parked · **[contributor]** good-first-issue.

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
  the no-CUDA fallback. **Todd's call** — it touches the freeze boundary.
- [ ] **The fitting-context suggestion cannot be acted on from inside the session.** It advises editing
  the chat config JSON and restarting, immediately after measuring the answer. A `/context <n>` that
  reloads would close it; `switchModel` already proves release-then-reload works.
- [ ] **Time a `/models` footprint probe.** Each row costs an artifact-header read plus a constructed
  graph, now up to three per row, and the number has never been taken. A dozen models on a 48-layer
  architecture is 36 constructions per keystroke; if it bites, put the column set behind a flag.
- [ ] **A per-row disk figure, if one ever returns, should be reclaimable bytes** — the blobs that model
  alone references. That is what deciding-what-to-delete wants, and prune's mark-and-sweep already
  computes the refcount; it is simply not exposed as a per-model query.
- [ ] **`getStorageSize` is implemented three times** — `Mila::Dnn::detail::getStorageSize`
  (`Tensor.ixx:81`, carrying a `REVIEW:` that already asks why), `Detail::getStorageSize`
  (`TensorBuffer.ixx:221`), and `Mila::Dnn::storageBytes` (`Component.MemoryStats.ixx`). The two
  namespaces differ only in case. Blocker: `Tensor.ixx` cannot import `Dnn.Component` without a cycle.
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
- [ ] `GptModel.ixx:330` hardcodes `eos_token_ = 50256` — should come from tokenizer metadata.
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
- [ ] **ProgressReporter** — an injected per-operation progress facility for long-lived ops (BPE vocab
  training, `PretrainedReader` load, load-time quantization). `BpeVocabulary.ixx:624` is the concrete
  call site: an every-100-merges elapsed-time print asking to become an async callback.
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
- [ ] **Isolate third-party warnings structurally** with `/external:I` + `/external:W0` (`-isystem` for
  Clang/GCC). The real target is warnings from third-party header text pulled into Mila's own TUs, not
  their sources (`/W4` at `Mila/CMakeLists.txt:87` is `PRIVATE` and never reached them). Precondition
  for any warnings-as-errors gate. Budget for two frictions: those headers enter through module global
  module fragments, and `/external:` does nothing for nvcc diagnostics.
- [ ] **`save_` is public on `Component` and protected on `CompositeComponent`.** Legal, but the
  accessibility of one virtual then depends on the static type you hold, and a caller holding a
  concrete composite cannot invoke it (C2248, worked around with an `exposeSave()` forwarder in
  `Tests/Dnn/Core/CompositeComponent.cpp`). The trailing underscore suggests non-public is the intent,
  making `Component.ixx:163` the declaration that is wrong.
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
- [ ] **Linux CI cannot configure since libcurl landed** — curl's `find_package(OpenSSL)` fails on the
  CUDA devel image, which installs no `libssl-dev` (`build-pipeline.yml:63`, `:145`). Both jobs die at
  configure, so the clang portability gate and the CPU-only ratchet compile nothing. Underneath it sits
  an older break the configure failure now masks: the CPU-only job failed on `Component.ixx:34`
  importing `Compute.CudaPinnedMemoryResource` in a `MILA_ENABLE_CUDA=OFF` build.
- [ ] **`MILA_ENABLE_LIBCURL=OFF` has no Linux coverage, and it is the Linux wheel's configuration.**
  The only OFF preset is `x64-debug-no-libcurl`, a *Windows Debug* build whose own description names it
  "the Linux wheel's configuration" (`CMakePresets.json:147`); every `linux-clang-*` preset leaves it
  ON. Turning it OFF on the CPU-only CI job gives the wheel's configuration a gate for free.
- [ ] **`CMakeLists.txt:254` pins curl at 8.11.1 under a `REVIEW:` marker naming 8.21 as current.** A
  vendored TLS-adjacent dependency in a published binary is the one pin where staleness has a security
  cost. Decide the bump or record why 8.11.1 stands.
- [~] **Linux/clang as a first-class platform** — WSL green, CI compiles under clang-21, the container
  builds and runs Gemma 4 FP4. The GCC 16 second oracle and the broadened matrix move to Future.
- [~] **Reproducible container build** — validated on clang-21 + gcc-15 host, CUDA 13.3. Remaining:
  build against the bind-mounted tree, and have CI build `FROM` the image rather than apt-installing.
- [~] **Dispatch error UX** — a missing `(Op, Device, Precision)` reads as one line, not a cascade.
  Core landed; the optional named kernel concepts and the `OperationDispatch.md` §12 reconcile remain.
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

- [~] **Nothing has touched a real server since the HTTP relayering — this is the priority.** Redirect
  handling, header construction and status mapping were all rewritten and no test reaches the network.
  `ExportArtifact --fetch <hf-lfs-url>` is the cheapest live exercise of exactly that path (it crosses
  the CDN redirect and reports status, final URL and digest); then `/models --online`, then
  `/install`, then re-run `Mila/Samples/Python/store.py`. Design: `Specifications/HttpClient.md`.
- [ ] **[gate] The cold download has never succeeded end-to-end.** The cache was seeded by hand from
  verified `--fetch` copies. One real Chat download failed its digest check while the same client
  fetched the exact digest through `--fetch`; leading explanation is a corrupt transfer the integrity
  check caught, but it is unproven and possibly intermittent. A mismatch now keeps the file as
  `.rejected` and reports the byte count: exactly 6799927760 bytes with a wrong digest means altered
  in flight, any other count means a length bug.
- [ ] **[gate] The published `mila-llm/gemma-4-12b-it` repository is incompatible with this build.** Its
  `mila.json` is the old `variants:{}` schema the new parser refuses, and its name lacks the `-fp4`
  suffix the flat scheme requires — nothing can pull it. Rename and rewrite the manifest **before**
  anyone else pulls; the model card needs the same pass (it still shows `makeHuggingFaceRemoteAccess()`
  and a `/pull` line with a coordinate that no longer parses).
- [ ] **Project distribution into the Python binding — steps 2b-4.** Decided (option C): one `pull`,
  two transports; Python supplies bytes, not procedure. Step 2b is `from_store( name, context_length,
  device_index )` on both sessions plus `BpeTokenizer.from_store()`, which kills the path-pairing and
  MIS's family branch and needs no transport; then `mila.store` / `mila.hub`; then MIS onto it,
  retiring `MILA_MODEL_PATH`/`MILA_TOKENIZER_PATH`. Watch: release the GIL inside the sink or the
  transfer serializes, and `py::bytes` copies where a `py::buffer` does not — at 6.35 GB that matters.
- [ ] **`transport=None` means two different things depending on the platform.** The Windows wheel is
  built with libcurl and pulls; the Linux wheel cannot link it (curl's system OpenSSL is not on the
  manylinux whitelist), and nothing in `Package/src/mila/__init__.py` supplies a replacement — so the
  same call returns `NullHttpTransport`'s refusal, quoting a CMake flag a pip user cannot act on. Ship
  a stdlib transport in the package (`HttpClient` already drives redirects and `Range`), or say so.
- [ ] **curl is missing from `NOTICE.md:33`** while the published Windows wheel statically links it into
  `_mila.pyd`. The note below the table treats notice-carrying as an open question for "a binary
  distribution that links them" — that binary now exists on PyPI, so it is an obligation. The same note
  points at a *Project Hygiene & Contributor Readiness* bucket that no longer exists; fix both.
- [ ] **The published wheel still teaches the retired form.** Its README and `__init__.py` docstring
  instruct users to pair `gemma4_12b_it_bf16.bin` with `gemma_tokenizer.bin`, and
  `LlamaModel.from_pretrained` still takes a `quantize_fp8` **boolean** — FP8-only, no FP4 — where the
  store carries quantization in the name. Fix with step 2b, not before.
- [ ] **`mila/__init__.py` is copied by a `POST_BUILD` step of a target it is not a source of.**
  `Mila/Bindings/CMakeLists.txt:63,83,101` stage it with `copy_if_different` off
  `add_custom_command(TARGET MilaPy POST_BUILD)`, which runs only when `MilaPy` relinks — so editing
  only `__init__.py` leaves every staged copy stale and the sample fails with a missing attribute.
  Use `add_custom_command(OUTPUT ...)` with `DEPENDS` on the source.
- [ ] **Migrate the remaining Llama and GPT-2 rows into the store, then delete the models-directory
  fallback branch.** The catalogue is gone and Gemma plus Llama 3.2 3B load from the store; seven rows
  still resolve through the `REVIEW:`-marked loose-file path. `.bin` leaves the catalogued set with the
  branch, **not the reader** — `PretrainedReader.ixx:229` sniffs the magic and would strand every
  `.bin` on disk. *Done when:* a clean machine pulls and runs Gemma 4 through named commands.
- [ ] **[crash] `generate()` walks off the end of the context instead of stopping.** Chat clamps the
  budget, but that is a consumer working around a library defect. GPT-2 is where it shows because its
  positional embeddings are learned — exactly `context_length` of them — so position 1024 is an
  out-of-bounds lookup. `GptModel.ixx:307`'s own default has the same flaw:
  `max_new_tokens.value_or( context_length_ )` never subtracts the prompt. `GenerateStatus::ContextLimit`
  already exists and simply does not fire first. **Capture the crash output before fixing** — an
  out-of-bounds LPE read and a KV-cache overrun look identical from outside.
- [ ] **A pre-quantized FP4 artifact loads but generates garbage** (endless thinking tokens). Reloading
  the known-good `.bin` through the same `switchModel` path is coherent, so the switch machinery is
  innocent and the artifact load is at fault. Note a SHA-256-identical re-export does **not** prove the
  load correct — it proves data fidelity, not model correctness. Two forward-output CUDA tests are in
  place at unit scale; the 12B FP8 A/B is not available (~12 GB on a 12 GB card).
- [ ] **The licensing story is per-family and must not be generalized.** Gemma 4 is Apache 2.0 (public,
  ungated); Gemma 3 and earlier carry the Gemma Terms of Use; **Llama 3.1/3.2 may be republished, but
  attributed** — ship the agreement, display "Built with Llama" and Meta's notice, pass along the AUP,
  and begin the model name with "Llama" (`llama-3.1-8b-it-fp4` already does). Gating is a *policy*
  choice, not a licence condition. See [[project_gemma4_apache2_license]].
- [ ] **The `mila-llm` organization has no organization card** — it is the landing page for anyone
  following a coordinate, and it is currently HuggingFace's placeholder. Needs: what a Mila artifact
  is, that it is loadable only by Mila and deliberately not NVFP4/MXFP4, the coordinate form, and the
  link to mila.toddt.me. See [[project_positioning_reference_impl]] — never lead with throughput.
- [ ] **Packaging then installing hashes every file twice** — `buildPackage` hashes to derive the
  manifest digests and `install` hashes again to verify adoption (~50 s of the ~60 s Llama 3B
  migration, ~2 minutes on the 8B). Neither check is wrong alone, so the fix is a combined verb.
  `publish_model.py` has the same defect for its own reason.
- [ ] **Progress reporting is unthrottled, and is now worse than when it was filed.** `pullModel`'s
  progress lambda in `Chat.ModelCatalog.ixx` redraws on *every* chunk with no percentage test at all —
  at 6.33 GB that is far past the hundreds of redraws originally reported. Throttle on the printed
  percentage changing.
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
- [ ] **Chat reports "Thinking: balanced" for models that have no thinking mode.** `show_thinking` is a
  session-config flag, but only Gemma routes a reasoning channel — the welcome banner and `/model`
  show an effort level for Llama and GPT-2 regardless, reading as a capability they lack.
- [ ] **`main.cpp` re-checks what the store already guarantees** — after `resolveModel` succeeds it
  tests `exists()` on both paths, but `locate()` refuses an incomplete record. Harmless duplication,
  except `/model` has no equivalent check; if the guarantee is doubted, the check belongs in the store.
- [ ] `Version::getMajor()`/`getMinor()`/`getPatch()` are non-const (`Src/Version.ixx`), so the
  version-skew comparison needs a mutable copy.

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
- [ ] **[gate] PyPI advertises Linux and ships only `win_amd64`.** `pyproject.toml:37` carries
  `Operating System :: POSIX :: Linux` and the sole published file is
  `mila_llm-0.20.0b2.dev20-cp313-cp313-win_amd64.whl` — no Linux wheel, no sdist, so `pip install` on
  Linux fails with nothing to fall back to. Release metadata is immutable, so the live page stays wrong
  until the next release carries the Linux wheel below.
- [ ] **The Windows wheel has no preset-driven equivalent of the Linux one.** `x64-wheel` configures
  correctly, but `pip wheel` over the staged package is still a manual step with no script, so the two
  platforms' wheels are produced by different procedures. `Docker/build-wheel.sh` is the shape to
  mirror — including emptying the output directory, which is what keeps a stale version from riding
  along in the publish glob.
- [ ] **[gate] The Windows wheel has never been tested without a CUDA Toolkit installed.** The Linux
  wheel was missing `nvidia-cuda-runtime` for three environments before a CUDA-free image caught it —
  every earlier test passed because the host had a toolkit. Windows links cudart statically so it
  *should* not have the same hole, but that is the reasoning that hid it before. Needs a Windows
  machine or container with no toolkit; `_toolkit_directories()` reads a fixed path that cannot be
  hidden by unsetting an environment variable.
- [ ] **Neither wheel has run a model.** Both import and reach the store on a CUDA-free host, but
  nothing has called `initialize()` against a real GPU from an installed wheel, let alone loaded
  weights. WSL Ubuntu has GPU passthrough via the Windows driver and can install the
  `manylinux_2_38` wheel (glibc 2.43 clears the 2.38 floor), so it is the environment for this.
- [ ] **`Mila/Tools` has no off switch** — gated on `PROJECT_IS_TOP_LEVEL` alone
  (`Mila/CMakeLists.txt:1081`), so the wheel configure builds `tokenize` and `ExportArtifact`, neither
  of which can go in a wheel. Every other subdirectory has a `MILA_ENABLE_*`; this one costs build time
  on an artifact that discards it.

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
- **v0.20 feature-frozen tails** — the Generation API surface tail (`SamplerConfig` rename, Llama/Gpt
  seedable sampling, eager sampler, config-accessor propagation, `contextLength()` hoist), the
  Sample-API device-sampler migration for Llama/Gpt, the Optimizer-dispatch migration onto
  `OperationTraits`, and the unspecced **Chat** feature milestone.
- **Model serialization** — the remaining checkpoint round-trip and distribution-artifact phases.
  Design, defect analysis and the phase plan are in `Specifications/ModelSerialization.md`.
- **API Coherence** — the pre-1.0 consistency pass, and the precursor to any API-stability promise.
- **Warnings-as-errors ratchet.** Constraints worth keeping: it requires the `/external:W0` isolation
  first; enforce in **CI only**, never locally; ratchet on the count *not increasing* before demanding
  zero; **MSVC first**, since `/WX` across three compilers means the union of three opinions must be
  zero; and dormant-but-retained code warns by nature — suppress per-file in CMake pointing at the
  owning task, never with `#pragma warning` in module code. Land it **after** v0.20 ships.
- **Parallel range downloads for model retrieval** — N concurrent `Range` requests into one staging
  file; content addressing means correctness rests on the final digest, not arrival order. **Measure
  first, and the evidence is against it:** LM Studio took ~2 hours for a comparable Gemma 4 12B on the
  same connection, which suggests the ceiling is HuggingFace's edge rather than per-connection TCP. If
  the two clients land within a factor of each other, close this rather than implement it.
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
