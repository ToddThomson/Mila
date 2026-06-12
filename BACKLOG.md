# Mila — Backlog

Open engineering tasks not yet completed. This is the working task list.

- **Milestone vision and success criteria** live in [ROADMAP.md](ROADMAP.md).
- **Completed, validated work** lives in [CHANGELOG.md](CHANGELOG.md).
- **Design rationale** lives under `Mila/Specifications/`.

As contributors arrive (the `0.20` production cycle), the contributor-facing items here are
promoted to `good first issue` discovery Issues (a GitHub mechanism, distinct from inbound
user issues); the rest stay in this file. Until then it is the single flat backlog for a solo
maintainer.

Legend: **[gate]** blocks the named milestone · **[deferred]** parked, revisit on the
stated trigger · **[contributor]** good-first-contribution / demand-driven, not a release gate.

---

## Alpha.6 — Consolidation (closing the alpha line)

Alpha.5's success criteria are met (greedy decode at FP8 with no catastrophic divergence on
Llama 3.2 3B and 3.1 8B, 8B within the 12 GB budget). Alpha.6 closes the alpha line:
feature-freeze (no new features) and burn down the debt so the public release earns the beta
label. The FIXME/TODO burndown + debug-strip work is itemized under
Project Hygiene below; the migration/cleanup tasks specific to closing alpha are here.

- [ ] **[gate]** CPU Linear traits — `OperationTraits<LinearOp, Cpu, FP32, NoWeightQuant>` specialization; retires the last `CpuLinearOpTypeMap` dependency
- [ ] `Dropout` component — still on `OperationRegistry::createUnaryOperation` and excluded from the build (`#`-commented in `Mila/CMakeLists.txt`); migrate to a `DropoutOp` `OperationTraits` specialization before re-enabling
- [ ] Physically delete the retired registry / registrar / typemap / arity-base source files now retained on disk for reference (`OperationRegistry`, `OperationRegistryHelpers`, `OperationsRegistrar`, `OperationRegistrarHelpers`, `UnaryOperation`/`BinaryOperation`/`PairedOperation`, `LinearOpTypeMap`, `GqaOpTypeMap`, `CpuLinearOpTypeMap`, `CudaGqaOpTypeMap`, `CudaLinearOpTypeMap`, and the deprecated `FusedComponent`)

- [ ] **Couple parameter initialization to runtime mode** — `BuildContext::initialize_parameters` defaults `true` independently of `RuntimeMode`, so an inference-mode build can silently run (then immediately discard) full parameter initialization. `GptModel::fromPretrained` hit exactly this; fixed for now by passing `false` explicitly (matching `LlamaModel`), but that relies on every load path remembering the third argument. Structural fix: derive the `initialize_parameters` default from `RuntimeMode` (`Inference` => no init unless explicitly requested) so no load path can regress by omitting the flag. The per-device init wiring (TensorOps `zero`/`fill`/`xavier`/`fill_normal`) is now live and gated per component via `shouldInitializeParameters()`; this default-coupling remains as hardening so a future load path cannot silently re-run (and discard) init

Deferred / not alpha-close gates:

- [ ] **[deferred, milestone TBD]** Token sampling (temperature / top-k / top-p) — `OperationTraits<SamplingOp, Cuda, FP32>` and `<…, BF16>` specializations; `TokenSampler` component + `CudaSamplingOp` per `Specifications/TokenSampling.md`. **Pushed out of Alpha.6** (feature freeze — no new features); milestone undecided, to be assigned later. Not a 0.20 gate — greedy decode is already validated, so this is additive
- [ ] **[deferred, training-only]** AdamW debug instrumentation — the per-value `isfinite`/limit `printf` anomaly guards in `CudaAdamW.cu` (6 sites) plus the leftover `printf` in `CudaAdamWOptimizer.ixx:270` are training bring-up scaffolding. Left untouched by the Alpha.6 inference-consolidation debug strip because the AdamW path is training-only: off the validated inference path, exercised solely by the parked MNIST/Bard samples, and untested (`AdamW.Cuda.cpp`/`AdamW.Cpu.cpp` disabled in `Tests/CMakeLists.txt`). When Training is picked up, decide strip-vs-gate (the `KERNEL_ASSERT` invariant checks are already `NDEBUG`-gated and zero-cost in release; the `printf`s are not) and re-enable the optimizer tests in the same pass
- [ ] **[deferred, training-only]** CUDA `fill_normal`/`fill_uniform` are FP32-only — they cast the raw buffer to `float*` and `curandGenerate` into it, so BF16/FP16 reduced-precision **train-from-scratch on CUDA** corrupts weight/embedding init. Reachable now that `xavier`/`normal` init is wired (`TokenEmbedding` wte is BF16 on the Llama path). Harmless for inference (init gated off) and for CPU (the `CpuTensorOps.Random` added this cycle converts element-wise). Fix: generate into a temp float buffer + a convert pass — the CUDA dtype counterpart to the CPU Random backend
- [ ] **[deferred, needs recall + live-vs-dead analysis]** Remove FP16 — superseded by BF16. FP16 was implemented first; once BF16 landed there is no reason to carry both for LLM inference (BF16's wider exponent range is strictly preferable, no loss scaling). Scaffolding is woven through *live* code: `CudaDataTypeMap<half>`, the `half`/`CUDA_R_16F`/`CUBLAS_COMPUTE_32F_FAST_16F` branches in `CudaLinearOp`, `half` throw-stubs in `CudaLinearOp.Plans`, and the commented `*_fp16` backward/permute stubs across the GQA/MHA/LPE/Softmax dispatch (these are the marker-triage "bucket B"). Trace live-vs-dead `half` paths before removal — not a mechanical delete
- [ ] **[deferred → 0.21 Qwen 3 cycle]** `OperationTraits<GroupedQueryAttentionOp, Cuda, BF16, PerChannelKvFp8<>>` specialization — pending `CudaGqaOp` FP8 KV cache support
- [ ] **[contributor]** Llama-lineage CPU ops (`RmsNormOp`, `SwigluOp`, `RopeOp`, `TokenEmbeddingOp`, `CrossEntropyOp`) in `OperationTraits.Cpu.ixx` — demand-driven; compile-time dispatch makes their absence zero-cost on the GPU path and a localized compile error if a `<Cpu, …>` Llama is instantiated
- [ ] **[deferred, measure first]** Phase 6b H2D pipelining — a dedicated load stream + CUDA events threaded through `loadParameter`/`quantize` would overlap H2D-with-H2D, but ~16 GB over PCIe 4.0 (~2.3s) is the floor, so pursue only if a profile shows the load is sync-bound rather than disk-bound

---

## Alpha.7 — Test Suite Revival

Recover the authored test suite, not write one. The first Mila year was test-driven; ~70 test
files exist under `Mila/Tests/**` but only ~24 are active — the rest were `#`-commented during the
inference-era refactors (`Tests/CMakeLists.txt:107` names the cause: "too many tests to refactor for
Component lifecycle changes"). The work is three buckets: (1) re-green what exists, (2) translate it
to the post-refactor API, (3) backfill what the old suite never covered. The non-negotiable
deliverable is the **CI ratchet** — the suite rotted because nothing gated it, so revival without a
gate just reschedules the next rot. Gated behind the Alpha.6 `CompositeComponent`/`setTraining`
lifecycle fix, which is what currently forces the component tests off.

- [ ] **[gate]** Re-green the authored component / operation / tensor / tokenizer suites against the current API — re-enable the `#`-commented files in `Tests/CMakeLists.txt`. Bucket 1 (uncomment + fix trivially broken) + bucket 2 (translate to the post-refactor surface: `OperationTraits` dispatch, the `Operation` base-class collapse, the precision axes, the Alpha.6 lifecycle fix)
- [ ] Backfill coverage for the inference-drought features the old suite never had — load-time quantization (`PerChannelFp8`/`PerGroupFp4`, the decode matvec kernels), `OperationTraits` dispatch, the Llama path (RmsNorm/SwiGLU/GQA/RoPE components, `LlamaModel::fromPretrained`). Genuinely new, not recovery
- [ ] **[gate]** Wire the suite into CI as the anti-rot ratchet — build on the `MILA_ENABLE_CUDA=OFF` CPU-only gate so a future API churn fails the build instead of silently re-commenting coverage. This is the deliverable that keeps the revival alive
- [ ] Do not revive tests for code being deleted — retire the disabled `UnaryOperation`/`BinaryOperation` tests alongside the base-class removal (Alpha.6); same for the registry/typemap tests
- [ ] Calibration is the MNIST-plus-tests spike under Alpha.8 — Training Revival — it measures the per-file bucket-2 translation cost on a representative slice before the suite-wide estimate is trusted

---

## Alpha.8 — Training Revival

Recover the validated GPT-2 / MLP training path. MNIST (MLP) and Bard (GPT-2 generation) were
complete, working samples parked behind an explicit `FIXME: Re-enable after alpha.5 completed`
trigger (`Mila/Samples/CMakeLists.txt:3-4`) that has now fired. Reviving them reactivates the half
of the library inference never exercises — the AdamW optimizer, the loss and backward kernels,
gradient flow, train-from-scratch init. The revived tests are the oracle: a sample "converges" only
when its test says so. Several deferred Alpha.6 items (AdamW debug instrumentation, the CUDA
`fill_normal` FP32-only gap) fold into this milestone. **Scope is GPT-2 / MLP training only** —
Llama 3.1/3.2 fine-tuning is explicitly out of this release, remaining a Future Direction.

- [ ] **(lead — timeboxed spike)** Revive **MNIST + its tests** against the current API — MNIST is the MLP (simpler than Bard's GPT-2/BPE/transformer surface), so it is the cheapest representative slice. Re-enable the sample (`Mila/Samples/CMakeLists.txt:3`) and its tests; pass/fail = builds, runs, trains to target accuracy, tests green. Measures all three revival buckets at once and sets the milestone dates on evidence rather than the day-or-3 estimate. **Do this first**
- [ ] Re-enable MNIST + Bard in the build — flip both `FIXME: Re-enable after alpha.5 completed` triggers (`Mila/Samples/CMakeLists.txt:3-4`) and add the Samples build to CI (pairs with the Project Hygiene "Samples build to CI" item)
- [ ] Re-enable + re-align the AdamW path — `AdamW.Cuda.cpp` / `AdamW.Cpu.cpp` (disabled in `Tests/CMakeLists.txt:190-191`); resolve the deferred AdamW debug instrumentation (strip-vs-gate the `CudaAdamW.cu` printf guards + `CudaAdamWOptimizer.ixx:270`) in the same pass
- [ ] Fix the CUDA `fill_normal`/`fill_uniform` FP32-only gap (the deferred Alpha.6 training-only item) — it corrupts BF16 train-from-scratch init; the CUDA dtype counterpart to the `CpuTensorOps.Random` backend
- [ ] Revive the loss + backward path — CrossEntropy / SoftmaxCrossEntropy components and tests (`Mila/Tests/Dnn/Components/Losses/*` exist, commented) and the backward-pass stubs (Alpha.6 bucket D)
- [ ] **ProgressReporter mechanism** — design the cross-cutting progress facility for long-lived ops (the `BpeVocabulary` training `\r` progress at `:600`/`:613`, plus `PretrainedReader` load and load-time quantization are candidates). Injected per-operation (on the op's config, **not** a global facade — progress is scoped to one call, unlike the process-wide logger), null default, library owns throttling, cancellation first-class (`bool` return or `std::stop_token`), documented threading contract. Mirrors the Logging subsystem's *shape* but is a separate concern (progress = transient/overwrite-in-place; logging = append-only events). The Alpha.6 debug strip leaves the `BpeVocabulary` training progress in place as living training-path code — it migrates here, it is not deleted
- [ ] Validation — MNIST trains to its target accuracy; Bard generates coherent text; train-from-scratch validated at the precisions the samples use; the AdamW / loss / training-path tests green and CI-gated

---

## 0.20 (first production release) — Packaging

A downstream app consuming Mila via `find_package(Mila)` currently fails to build. C++23
module interface units cannot ship as portable BMIs, so the consumer's toolchain
recompiles the installed `.ixx` units, and each pulls its kernel header via a file-relative
quoted include that resolves against the wrong tree on install. The real defect is how the
`Mila` target is composed: kernel `.cuh`/`.h` headers are raw `add_library` sources (no base
dir, no install rule, no usage requirement); CUDA sources are added unconditionally though
`enable_language(CUDA)` is gated on `MILA_HAS_CUDA`; and three categories (`.cu` link-only
instantiations, `.cuh`/`.h` shippable declarations, `.ixx` interface units) are flattened
into one list.

- [ ] **[gate]** Single coherent restructuring (not a destination patch): model headers as `FILE_SET HEADERS TYPE HEADERS BASE_DIRS Src`; migrate file-relative quoted includes to angled includes anchored at one `Src` root (vendored `Deps` gets its own root so nlohmann becomes `<nlohmann/json.hpp>`); set `BASE_DIRS` on the `CXX_MODULES` file sets to the same `Src` root; move all CUDA `.cu`/`.cuh` sources under `if(MILA_HAS_CUDA)` via `target_sources`; replace the `install(DIRECTORY …)` glob with `install(TARGETS Mila … FILE_SET HEADERS)`. The include root must be on Mila's own build path (current root at `Mila/CMakeLists.txt:128` is INTERFACE-only — make it PUBLIC or add a PRIVATE entry or the in-tree build breaks once includes are anchored); install the generated `Version.h` and `Deps/` alongside the modules. Validate with a throwaway `find_package(Mila)` + `import Mila;` consumer wired into CI (Mila's own CI stays green and will not catch packaging regressions on its own)
- [ ] Suggested sequencing: convert one CUDA op to angled includes and get the in-tree build green first (proves the root/`-I` model), then bulk-convert backend-by-backend (the compiler flags every missed header), then do the install-side CMake and the consumer test last
- [ ] **[deferred, later in Beta]** Decide whether the kernel `.cuh` *declarations* belong in the public install surface at all — kernels are explicitly instantiated per precision in `.cu` files compiled into the archive, so consumers link the symbols and only need declarations to call the launch wrappers; the shippable surface may be reducible. Separate architectural decision, out of scope for the packaging fix above

---

## 0.20 (first production release) — Module Hygiene (includes/imports + Doxygen)

Over alpha the module surface accumulated unneeded `#include`s/`import`s and drifted
Doxygen. Both are large, mechanical, low-risk-per-edit, high-volume diffs, deferred until
a cross-compiler build existed (a hard prerequisite for the include work). The WSL Clang
oracle now exists (Clang 21 + CUDA 13.3 + gcc-15 host); GCC 16 and the dev-container build
remain to be validated. Surface: 287 `.ixx` units, ~1,810 `import` lines, ~1,419 `#include`
lines, ~1,950 `@brief` / ~1,100 `@param` / ~257 `@tparam` / ~218 `@file` tags across 258 files.

There is no reliable off-the-shelf tool for C++23 module `import` cleanup (IWYU and clangd
do not understand the module graph), so the compiler is the only ground truth. MSVC
transitive resolution is the trap: a line can be removed and MSVC still compiles because the
symbol arrives transitively — so "still builds on MSVC" does not prove the line was unused.
The honest oracle is a **Clang or GCC** build. The cruft is real and visible — even
`Linear.ixx`, the dispatch reference, imports `Dnn.TensorOps` twice.

Includes/imports:

- [ ] Phase 0 — exact-duplicate `import`/`#include` dedup within each file; pure text analysis, scriptable across all 287 units, zero compile cost and zero risk
- [ ] Phase 1 — candidate report (no edits): heuristic scan flagging imports/includes whose symbols never appear in the file body; over-reports by design, so it is a worklist to size the job, not a verdict
- [ ] Phase 2 — compiler-verified removal, leaf modules first: scripted remove → rebuild → revert-on-failure, batched per file with binary-search on failures, verified against Clang/GCC rather than MSVC so visible cruft is not traded for invisible transitive coupling

Doxygen staleness (these tiers, plus the docs-site CI items under "Release Assets & CI" below, are the engineering detail of the **Alpha.9 — API Documentation** milestone in [ROADMAP.md](ROADMAP.md)):

- [ ] Tier 1 — `@file` rename drift: 34 files whose `@file` tag does not match the filename (e.g. `RocmDevice.ixx` tagged `VulkanDevice.ixx`, `CudaMhaOp.ixx` tagged `CudaAttentionOp.ixx`, `Lpe.ixx` tagged `Gpt2Encoder.ixx`). The correct value is `basename` — fully scriptable, no judgment
- [ ] Tier 2 — `@param`/`@tparam` name mismatches: documented names no longer in the signature. Mechanical and high-confidence, but signatures span lines, so emit a candidate list for review before batch-fixing
- [ ] Tier 3 — semantic staleness (per-subsystem judgment): `@brief`/descriptions describing the retired world (components "registering with `OperationRegistry`", "deriving from `UnaryOperation`/`BinaryOperation`", string-keyed dispatch), naming drift (`TWeightQuant` in prose vs. the spelled-out style), file-level `@brief`s exceeding the 1-3 sentence rule. One settled subsystem at a time; leave subsystems mid-refactor alone

---

## 0.20 (first production release) — Public API Surface (narrowing the `Mila` umbrella)

The supported public entry point is a single `import Mila;`. Internal module names
(`Dnn.*`, `Compute.*`) are an implementation detail (also why they are intentionally not
`Mila.`-prefixed — the `Mila::` namespace already scopes symbols). Tests/samples import
submodules directly (14 today) and are not bound by the public contract. The mechanism is
correct; the open work is *scope*. At freeze the failure modes are asymmetric: too narrow is
widened later by adding exports (non-breaking); too broad can only be fixed by removing
exports (breaking). Beta should freeze the **narrowest defensible** surface. Today `Mila.ixx`
re-exports essentially the whole tree, locking in (1) every consumer recompiling the full
transitive closure into BMIs, and (2) every re-exported symbol as a frozen promise.

- [ ] Define an explicit public allowlist for `Mila.ixx` — the inference surface (models, components, tensors, execution context, `initialize`/`shutdown`, tokenizers); treat the export list as the literal API spec
- [ ] Demote non-public modules to unexported internal (still directly importable by tests/samples): `OperationRegistry`/`OperationRegistryHelpers`/`OperationsRegistrar`, `UnaryOperation`/`BinaryOperation` (both slated for removal), `Dnn.TensorBuffer` ("remove after testing"), the per-device operation modules
- [ ] Stop re-exporting the vendored `nlohmann` module/namespace through the public surface — it hands a breaking change to a third party's release schedule; the Chat sample's direct `import nlohmann.json` is a sample-layer concern
- [ ] Domain-qualify generic single-segment module names that are global-collision magnets on co-link — `Core`, `Utils`, `Components`, `Profiling` (e.g. `Dnn.Core`, `Dnn.Utils`); targeted handful of renames, independent of the no-`Mila.`-prefix rule
- [ ] **[deferred, non-breaking]** If training becomes a first-class public concern, add a separate `Mila.Training` umbrella rather than widening `Mila` — the additive direction keeps the inference surface tight

---

## 0.20 (first production release) — Release Assets & CI

Mila is source-distributed (clone to contribute; `find_package(Mila)` from a source install
to consume), so most release-asset machinery is unnecessary — GitHub auto-generates source
archives per tag, so **tagging `master` is the release**. Release flow is a `dev` → `master`
PR; CI validates on the PR; docs publish only from `master`. During alpha the **default
branch is `dev`**; at beta **switch the default to `master`** (README/roadmap links are
branch-agnostic, so no content change needed).

Documentation site (the genuinely GitHub-bound deliverable):

- [ ] Docs generated by a GitHub Action, never committed to the source tree (Doxygen output for 287 modules with call graphs is thousands of files + binary graphs; committing per release poisons the repo history)
- [ ] Fully decouple the docs job from the build — PARTIALLY DONE (docs is its own workflow running `cmake --build --target docs`, compiling nothing). Remaining: the job still runs CMake configure inside the CUDA container because the `docs` target is CMake-registered and the root configure requires the CUDA toolkit. Full decoupling = driving Doxygen without a CUDA-dependent configure (standalone Doxyfile, or a CUDA-free docs-only configure path)
- [ ] Narrow what docs expose to match the public API surface — current config sets `EXTRACT_ALL`/`EXTRACT_PRIVATE`/`EXTRACT_STATIC` recursively over all of `Mila/Src`; published docs should show the `import Mila;` public surface, not every private member of 287 modules (pairs with Public API Surface)
- [ ] Verify Doxygen renders C++23 module units faithfully — module support is young; `export module`/partitions/`import` may misrepresent structure. Depends on the Doxygen staleness pass so generated docs are not loud with `WARN_NO_PARAMDOC`
- [ ] Bump docs Doxygen version — `docs.yml` installs via apt (Ubuntu 26.04 pins **1.15**; latest **1.17**); module rendering fidelity improves across releases, so install 1.17 from the upstream tarball. Pairs with module-rendering verification

CI correctness:

- [ ] CI/CD pipeline efficiency pass (**[deferred, optimization not a trust gate]**) — measured master run: Build ~18.5 min + packaging gates ~25 min = ~44 min; dev pushes ~20 min. Dominant cost is **C++23 module compilation, not `.cu` kernels** — so the CUDA-only ccache launcher misses the bottleneck. The module tree compiles up to **three times per master run** (Build, `find_package` gate, FetchContent gate). Levers ranked: (1) module-aware caching — spike whether clang-21 + ccache 4.x can reliably cache module compiles (BMIs are compiler/path-sensitive) — highest value, hardest, only one hitting the bottleneck; (2) `-O0`/Debug in CI (partial win — BMI generation is front-end work `-O0` does not reduce); (3) move FetchContent full-rebuild to tag-only if cadence rises
- [ ] Broaden compiler coverage toward the supported matrix — CI builds only **clang-21**; the primary dev compiler (MSVC 2026) and the working GCC 16 path are untested, so the compiler that previously broke the build (VS 2026 pre-18.6.2 module regression) is the one CI cannot catch. A multi-compiler CI is also the cross-compiler oracle the include/import hygiene pass needs

Docker image publish is optional and only if the runtime image stays a beta deliverable —
a release-tagged GHCR push is a natural CI-on-tag job but equally a local `docker build &&
docker push`; automation-of-convenience, not a gate.

---

## 0.20 (first production release) — Project Hygiene & Contributor Readiness

A beta is a trust signal; these items are about the project not contradicting itself or
wasting a newcomer's first hour.

- [ ] FIXME/TODO debt triage (IN PROGRESS) — the source carried ~71 `FIXME` + ~69 `REVIEW` + ~25 `TODO`; `FIXME` reads as "known broken", and several were commented-out core paths. DONE + validated: **bucket A** (the bypassed weight initializers / commented `xavier`/`normal` calls — full parameter-init subsystem restored, see CHANGELOG) and **bucket C** (CUDA `setCurrentDevice`). Remaining: FP16 stubs (bucket B — deferred, see "Remove FP16" above), backward-pass stubs (D), the training-lifecycle `isTraining()` demotes (E — tied to the `CompositeComponent` setTraining/build bug), and the design `REVIEW` set (G). Fix the real ones; demote the rest to neutral notes + tracked tasks **here in BACKLOG** (not GitHub issues — those are requester-authored), do not ship literal `FIXME` in public source. Distinct from the "debug instrumentation gated/removed" gate (the `std::cout` 12 files / `std::cerr` 5 / `printf` 6 usage)
- [ ] Debug instrumentation fully gated or removed — substantially done by the Alpha.6 debug-instrumentation strip (kernel `printf`/anomaly guards removed; the BPE tokenizer warning + vocab-load info routed to `Logging::Logger`, the encode timer and progress prints deleted). Training-path instrumentation is intentionally NOT stripped — it is deferred to its owning milestone (the AdamW debug item above; the `BpeVocabulary` training progress -> ProgressReporter under Training Revival)
- [ ] Test coverage of core components — now owned by the **Alpha.7 — Test Suite Revival** milestone above (re-green the ~70 authored test files, the CI ratchet, and the inference-drought backfill). No longer a loose Beta line
- [ ] Add the Samples build to CI (currently only tests build) so a contributor's first sample build is not the thing that breaks
- [ ] `good first issue` labels on GitHub (Beta requirement) — the exact label is `good first issue` (spaces, lowercase; hyphens break GitHub's `/contribute` + aggregator discovery). These are maintainer-authored discovery Issues promoted from this backlog (a GitHub *mechanism*, distinct from inbound user issues), each well-scoped with acceptance criteria + file paths. Mint when courting contributors (~default-branch switch); pairs with the community-health files already landed and the `CONTRIBUTING.md` gate
- [ ] `CONTRIBUTING.md` coding-standards section + `getting-started.md` onboarding guide (user-first, contributor superset) (Beta requirements)
- [ ] Ungated GPT-2 quick-start path for zero-auth first run (Beta requirement) — pre-converted permissively-licensed weights hosted on Hugging Face, fetched on first run via `resolve/` URLs over HTTPS (no Python/venv/auth); gated weights (Llama) stay a user-supplied offline conversion step
- [ ] Published Docker runtime image — slim multi-stage GPU runtime (built in CUDA `-devel`, artifacts copied into `-runtime`), release-tagged; gated weights never baked in (Beta requirement, see Distribution in ROADMAP)
