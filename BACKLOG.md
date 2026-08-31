# Mila — Backlog

**Work committed to the release in flight, and nothing else.** Narrative and success criteria live
in [ROADMAP.md](ROADMAP.md); design rationale under `Mila/Specifications/`. **Completed work lives
in the git history** — the commit that landed it is the record.

Everything upstream of that commitment lives in [`Mila/Issues/`](Mila/Issues/README.md): findings
land in `Untriaged.md` untriaged, and triage promotes them here, parks them in `Future.md`, or deletes
them. **Nothing is written straight into this file** — an item arrives here only by being judged
release work, which is what membership in it claims.

**The admission test.** Point at an item and name the ROADMAP success criterion that fails if it
never ships. If you cannot name one, it belongs in `Mila/Issues/`, not here. The test has to be
external to the moment, because in the moment every item feels worth doing.

Each `###` bucket is a v0.20 theme, its name matching the ROADMAP section (the only join).

**House rules.** An item is **three lines**: what, why it matters, `file:line`. Five if genuinely
complex. **Status lives in the checkbox** — `[ ]` open, `[~]` in progress, `[x]` done — and never in
the prose; no dates, no "GREEN", no findings. **Done means deleted**, in the same commit as the
work: `[x]` is a working-tree marker so finished items are visible while the change is reviewed, and
the commit that lands the work removes them. **No `[x]` is ever committed.** Findings
worth reusing go to the owning spec or to memory, not here. Tags: **[gate]** blocks the release ·
**[crash]** reproduces as a crash · **[net-new]** authored from scratch, not revived ·
**[decoupled]** off the critical path. Disposition is a file in `Mila/Issues/`, not a tag: parked is
`Future.md`, good-first-issue is `Contributor.md`.

**The size gate is lines per item, not total lines** — the failure mode is narrative, not item count.
Divide the lines in `## Current release` by the number of items in it; past **four** it has stopped
being a task list and needs a prune.

---

## Current release (v0.20.0)

### Models

- [ ] **Gate B has no unquantized case.** Both footprint suites test FP4 only, so `NoWeightQuant` —
  the path a store name without an `-fp4`/`-fp8` suffix takes — has never been checked against
  `cudaMemGetInfo`. Add `llama-3.2-3b-it` at BF16: ~6.3 GiB, fits the 12 GB card, no spill.
- [~] **Attribute the Gate B residual.** Scratch is measured and is not the answer, leaving ~1.0 GiB
  unattributed on Gemma and ~0.45 on Llama; the Qwen sighting was the un-pooled per-layer transients,
  a different and larger defect, and its numbers must not be folded in. Next and cheap: per-allocation
  rounding — read `MemoryAllocationStats::allocationCount` (import `Compute.MemoryResourceTracker`
  directly; `Mila.ixx:95` comments the re-export out) and divide. Nothing under ~0.1 GiB is signal.
- [ ] **`cudaMemGetInfo` cannot see WDDM's shared allocation, so every Windows VRAM measurement
  understates.** Anything deciding whether a model fits needs the per-process counters instead
  (`Get-Counter "\GPU Process Memory(pid_N*)\Dedicated Usage"` and `\Shared Usage`), which is what
  Task Manager reads. Note it in `MemoryFootprint.md`, whose premise is answering "does this fit".
  [[project_wddm_spill_mechanism]]
- [ ] **DECISION OWED — `BuildContext::withInstalledOutput` is an unenforced promise**
  (`Component.BuildContext.ixx:208`). The pooling predicate is authored three times in `Qwen.ixx`
  (`:382`, `:602`, `:626`) and the ~6.5 GiB DeltaNet understatement was one site existing while
  another did not; the workspace factories fuse describing the slot set with allocating it. Proposed
  split — bind unallocated pre-build, materialize in `build()` — in `MemoryFootprint.md` §4.5.
- [ ] **Gemma owes a block-level Gate A case**, per the per-block-kind rule in `MemoryFootprint.md`
  §4.5. `Gemma.Block.Cuda.cpp` calls `getRequiredMemory` nowhere and the local and global kinds share
  one max-geometry workspace. Blocked on an exported `makeGemmaBlockWorkspace` — Gemma builds its
  workspace inside private `GemmaTransformer::allocateBlockWorkspace` (`Gemma.ixx:1110`), so no test
  can construct one.
- [ ] **Leaf-level Gate A for `Rope` is still unwritten**, and must not be a naive predict-vs-build
  equality: `RopeCacheRegistry` keys on (theta, max_seq_len, head_dim) and only the first owner
  allocates, so the assertion is registry-order dependent. Transformer-level dedup is in place.
- [ ] **GPT-2 has no `getRequiredMemory`**, so `gpt2-small` gets no pre-flight and Chat says nothing.
  Its footprint is the simplest of the three — no quantization policy, no ring, learned positional
  embeddings sized exactly `context_length` — and it gives the `generate()` crash a budget. Nine
  components still throw the base's by-design error (Gelu, MultiHeadAttention, Lpe, GatedMLP, MLP,
  SoftmaxCrossEntropy, LayerNorm, Softmax, GptBlock); the contract lands family by family
  (`Core/Component.ixx:615`) and GPT-2 is the family outstanding.
- [~] **No Llama parity test exists, and the README's own wording admits it.** Gemma has
  `GemmaModel.Parity.Cuda.cpp` and Qwen's needs the 27B weights, so the cheapest model that fits both
  cards cannot be checked against a reference — while `README.md:162-165` says "validated against
  HuggingFace" for BF16/FP32 and only "coherent generation" for FP4, which is the precision every
  published model actually runs. Validate and record 3.1 8B FP8, then the FP4 claim.
- [ ] **RoPE scaling is disabled on the Llama load path** — `Llama.ixx:703` has
  `.withRoPEScalingFactor( metadata.rope_scaling )` commented out for a reason recorded as unclear.
  3.1 8B's extended context depends on it; resolve *before* writing the 8B parity test.
- [ ] **Triage `Llama.Block.ixx:132` view-aliasing** — the Q/K/V splits of `qkv_out` may not be
  contiguous. Confirm live-vs-benign and fix if live, before claiming Llama HF validation.
- [ ] Tool calling validated on Llama 3.2 3B and 3.1 8B Instruct.
- [ ] **`gemma_greedy_parity.py` diffs an FP4 Mila against a BF16 HuggingFace reference and does not
  say so.** `Mila/Tools/Converters/Gemma/gemma_4_BF16/gemma_greedy_parity.py:70` loads through the
  binding's FP4 default, so any divergence it reports mixes quantization error with a real defect.
  `from_pretrained` now takes `quantization=`, so the honest comparison is one argument away — on a
  card that can hold a BF16 12B. State which it ran either way.

### Observability

- [ ] **`matchesPath`'s `*` crosses the path separator, and its own doc says it does not.**
  `CompositeComponent.ixx:405-406` teaches `"qwen.blk_*"` as "every block, but not their children"
  and offers `"qwen.blk_*.*"` for the children; both are false — `*` matches dots, so the two
  patterns select the same set. Measured: `"*.tf_layer_*"` selected 816 components on Gemma 4 12B,
  which has 48 layers. Decide whether `*` should stop at a dot or the doc should describe what it
  does; `Observability.md` §11's path-matching bullet carries the same implication.

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
- [~] **Test coverage behind the samples.** Both MNIST and Bard run; what is missing is the suite
  under them — the `Core/Network.cpp` delta and the GPU companions (`Network.Cuda`/`AdamW.Cuda`).
  Sample-independent, so a green sample is not evidence the primitives are pinned.
- [ ] **`LlamaModel`'s context-overflow guard has no test.** `LlamaModel.ixx:336` carries the bound
  GPT-2's three cases pin, and nothing exercises it. Llama's overrun is the quiet one — it walks the
  KV cache rather than crashing — so absence of a report is not evidence. Template:
  `Tests/Dnn/Models/GptModel.Cuda.cpp`, a weightless checkpoint at a small deployment context.
- [ ] **Backward-path kernels disabled or unverified.** `CudaSoftmaxOp.ixx:73` and `:103` throw
  `"needs review"` with the real calls commented out; `Gelu.Fp32.cu:65` records that the shipped
  backward is not the numerically stable `sech^2` form. Gradient-check these before the suite claims
  backward coverage — and sweep the *unmarked* backward kernels per-precision twin by twin: the RoPE
  FP32 backward was wrong while its BF16 sibling was correct, in a file carrying no marker at all.

### Training Revival

- [~] **Data-loader contract tests** — `TokenSequenceLoader` done; remaining is the `MnistDataLoader`
  contract (normalization, one-hot targets, shuffle-on-reset, IDX magic number). Pin the TokenId
  signedness contract while there — `TokenSequenceLoader.ixx:44`.
- [~] **Re-enable the AdamW path** — `AdamW.Cpu.cpp` is active with a convergence case. Remaining: the
  `AdamW.Cuda.cpp` companion, plus strip-vs-gate the debug `printf`s in `CudaAdamW.cu` and
  `CudaAdamWOptimizer.ixx:270` in the same pass.
- [~] **[net-new]** Training-loop integration test (sample-independent) — the MNIST spine is covered
  by `Network.Cpu.cpp`; remaining is a GPT-2-stack analogue for the Bard spine.
- [ ] **[net-new]** Optimizer step-convergence test — minimize a known convex objective in N steps, so
  the update direction and bias correction are proven rather than just that `step()` runs.
- [ ] **[net-new]** TrainingMode / RuntimeMode coverage — assert that mode transitions allocate and skip
  gradient buffers correctly. Three `REVIEW:` markers are the invariant to assert, each guarding a
  state believed unreachable: `TokenEmbedding.ixx:221`, `Lpe.ixx:187`, `Lpe.ixx:495`.
- [ ] **`ResidualConfig` advertises a scaling factor that no backward implements and the two devices
  disagree about in forward.** CUDA forward honours it, CUDA backward takes no scale, and the CPU op
  ignores it entirely; the only guard is a debug-only assert at `CudaResidualOp.ixx:106`, so release
  builds train silently wrong. Cheapest correct fix, freeze-compatible because it removes an
  unimplemented knob: have `validate()` reject `scaling_factor != 1.0f` (`ResidualConfig.ixx:97`).
- [ ] **Validation** — the **FP32** training path proven by the primitive suite (gradient checks,
  step-convergence, loader contracts, init-at-precision, the integration test), CI-gated; samples run
  as demos. BF16 and GQA training move to the Training (advanced) release.

### API Documentation

- [ ] **Tier 3 — semantic staleness** (retired-world prose). Folded into Test Suite Revival: fix a
  file's prose while it is already open for re-greening.
- [ ] **`Component` documents a compute contract it does not declare.** `Component.ixx:132-133` and
  `:728` teach "`forward()` requires `build()`" and "`backward()` requires `isTrainingMode()`", but
  neither the base, `CompositeComponent` nor `Network` declares those methods. Correct the prose to
  name the concrete methods it means.
- [ ] **One concept, two names: the compute-precision template parameter is `TPrecision` in most of
  the tree and `TComputePrecision` in nine files** (122 occurrences). Same axis, split along no
  principle — `Linear` and `GroupedQueryAttention` differ from their own siblings — and it has cost
  compile errors. Not a blind sweep: `GroupedQueryAttention.ixx` and `CudaRopeOp.ixx` use both. The
  larger half is `TWeightQuant` where CLAUDE.md mandates `TWeightQuantization` — 97 occurrences over
  12 files, three of them specs. Same files, so one pass.
- [ ] **Nothing checks Doxygen when doc drift is introduced.** A break from a `Src/**` or `README.md`
  change is caught only by `publish-site.yml`, which is now manual — so nothing exercises Doxygen
  between publishes at all, and seventy-five errors once accumulated unseen and then blocked the site.
  Add a non-deploying Doxygen check to `build-pipeline.yml` (no CUDA, no CMake).

### Production Hardening

- [ ] **`import Mila;` breaks the standard library in the consumer's translation unit.** Three
  failures in a real FetchContent consumer, absent without it: stream **input** fails on an undefined
  `basic_istream::sentry`; instantiating a model needs `<sstream>` **before** the import, since
  virtual `Component::toString()` compiles in via the vtable; import-before-includes is fatal (C1116).
  `Samples/QuickStart/Cpp/main.cpp` carries two workarounds. [[project_import_mila_breaks_std]]
- [ ] **The Python binding discards `GenerateStatus`, so the two quick starts cannot reach parity.**
  All three sessions in `Mila_py.Wrappers.cpp` (`:553`, `:657`) do `(void)impl_->model->generate(...)`,
  so a Python caller cannot tell EOS from the `max_new_tokens` cap from context overflow from a
  cancellation. The C++ quick start prints `[stop]`; the Python one prints nothing, and that gap is
  visible to anyone reading the website's two first tabs side by side.
- [ ] **`mila/__init__.py` is copied by a `POST_BUILD` step of a target it is not a source of.**
  `Mila/Bindings/CMakeLists.txt:95` stages it with `copy_if_different` off
  `add_custom_command(TARGET MilaPy POST_BUILD)`, which runs only when `MilaPy` relinks — so editing
  only `__init__.py` leaves `<build dir>/python/mila/` stale and a sample fails with a missing
  attribute. Use `add_custom_command(OUTPUT ...)` with `DEPENDS` on the source.
- [ ] **The 16K perplexity gate needs a re-run before any 32K claim.** From 8K to 16K the oracle
  improves 7.2% and the plan only 3.4%, so the quantized arm captures about half the benefit of extra
  context — the compounding signature the recurrent layers make plausible. Table and caveats in
  `Qwen3.8.md` §8 item 9; `DISABLED_QualityGateAcrossContextLengths` is the harness.
- [ ] **The head's two paths do not agree to the last digit, so a perplexity comparison must fix the
  width.** Same weights and corpus, width 1 (decode matvec) and width 64 (W4A8-FP8 GEMM) differ in the
  third decimal. Small, but head width is part of the measurement protocol rather than a free
  performance knob — both arms of a quantization comparison must use the same one. **Likely already
  recorded** at `Qwen3.8.md:509` and `:546`; verify, and if so this item is the duplicate.
- [ ] **`Qwen3.8.md` §8 gates the 16 GiB oracle on token-for-token cross-arch agreement, which cannot
  pass at any precision.** BF16, FP8 and FP4 all fork between Ada and Blackwell, at a token index set
  by the prompt rather than the precision, while each card is deterministic run-to-run — FP
  non-associativity, not a defect. Restate the gate as teacher-forced; perplexity never samples.
- [~] **GPT-2 and Llama 3 pre-tokenization silently runs the ASCII fallback on every build and
  platform**, the published Linux container included: `\p{L}`/`\p{N}`
  (`BpePreTokenizationMode.ixx:33`, `:57`) compile in no standard `std::regex`, so
  `BpeTokenizer.ixx:344` throws and approximates, and no parity test catches it. Settle together —
  whether the fixtures are English-only (if so the site's parity claim is untested), and PCRE2/RE2
  against a hand-written Unicode scanner.
- [ ] **DECISION OWED — no model factory accepts an `IExecutionContext`.** The earlier premise that
  it is exported but unreachable is false: it is the parameter type of the public `TensorOps`
  transfer functions (`TensorOps.Transfer.ixx:90`) and of `Component::setExecutionContext`
  (`Component.ixx:896`), and `Samples/MNIST/Src/MnistClassifier.ixx:84` builds a network on one from
  the factory. What remains is that `fromPretrained` takes a `DeviceId` (`GemmaModel.ixx:130`), so
  two models cannot share a stream — which `IExecutionContext.ixx:66-74` documents as deliberate,
  because an overload would make the activation observer a cross-model leak. Confirm or change it.
- [ ] **`CMakeLists.txt:266` pins curl at 8.11.1 under a `REVIEW:` marker naming 8.21 as current.** A
  vendored TLS-adjacent dependency in a published binary is the one pin where staleness has a security
  cost. Decide the bump or record why 8.11.1 stands.

#### Release mechanics

- [ ] **[gate] The wheel matrix has never had a clean-room run on Windows, and PyPI advertises a Linux
  wheel that does not exist.** `pyproject.toml:37` carries `POSIX :: Linux` while the sole published
  file is `win_amd64`, and release metadata is immutable. Linux is clean-room proven under
  `python:3.13-slim`; Windows cannot be tested locally (Windows 11 Home has no Containers or Hyper-V).
  Both resolve only through a release cycle, and the matrix needs `wheel-cleanroom.yml` on `master`.
- [ ] **The published wheel still stops before Blackwell.** The library default carries `120`
  (`Mila/CMakeLists.txt:24`) and the runtime image always did, but the `x64-wheel` and `linux-wheel`
  presets pin `75;80;86;89;90` (`CMakePresets.json:183-184`, `:214-215`), so a `mila-llm` install on
  an RTX 50-series card JITs from sm_90 PTX at first launch. Adding `120` costs one more CUDA compile
  per wheel; the alternative is saying so on the PyPI page.
- [ ] **Publish `mila-llm-server` to PyPI.** The restructure is done and the version derives from
  `Version.txt`, so what remains is the release step: RELEASING covers the four CUDA wheels and says
  nothing about MIS. One `py3-none-any` file from `python -m build`, beside the wheel upload.
- [ ] Add the Samples build to CI (only tests build today).

#### Container

- [~] **Publish the Docker runtime image.** The image builds and all three entrypoint verbs are
  verified in a container — `install` pulled into a fresh volume, `chat` listed that store, `serve`
  bound 6452 and answered a real `/v1/chat/completions` with a model loaded from a read-only mount of
  the host store. No publish build has ever been made: verification used single-arch `89`, where a
  published image needs `89;90;120` and `MILA_CLEAN_BUILD=1` (`--no-cache` leaves BuildKit cache
  mounts intact, which has already produced two silently wrong images in one day). Take the site's
  devel cost figures from that build — `docker manifest inspect` and `docker images`.
- [ ] **Every container build path defaults to an arch a published image cannot use.**
  `Docker/build-chat.sh:25` defaults `MILA_CUDA_ARCH=native` and passes it to both
  `CMAKE_CUDA_ARCHITECTURES` and `MILA_LIBRARY_CUDA_ARCHITECTURES`, so the image carries kernels only
  for the GPU that built it — and `native` does not resolve on the GPU-less builder a publish runs on.
  The publish pipeline must set the portable list explicitly.
- [ ] **Decide the container tag scheme, including whether a pre-release gets `latest`.** RELEASING
  covers dropping `+build` (OCI forbids `+`) and nothing else. `latest` is what a bare
  `docker run toddthomson/mila-llm` resolves to, so pointing it at a beta makes the beta the default
  for everyone who does not read the tag list. Repository name is decided: **`toddthomson/mila-llm`**.
- [ ] **Docker Hub Overview page is an authored surface, so give it a source in the repo.** It is what
  search shows and it carries the container-distribution message; hand-editing it in the browser is
  how the HF org card came to need a rewrite. [[project_four_channel_roles]]
- [~] **The runtime image ships a binding that cannot import, and the gate says it is fine.**
  `site-packages/mila/` holds only `__init__.py`, so `install` and `serve` both die on
  `ImportError: No module named 'mila._mila'` — the extension reaches the image only as a
  `POST_BUILD` side-effect into the source tree, which a cache-warm compile never re-runs. Install
  from `/build/python/mila`, where the build actually writes it.
- [ ] **The `ldd` gate passes when the file it checks is absent.** `Dockerfile.runtime`'s runtime
  stage greps for `"not found"`, but an unmatched glob makes the shell hand `ldd` a literal pattern
  and it answers `"No such file or directory"` — so the gate printed "Shared library check passed"
  over a missing extension. Assert the file exists first, then check its NEEDED entries.
- [ ] **`Docker/build-mis.sh:76` looks broken on the current image.** It runs
  `pip install --no-deps -e Mila/Bindings/Package` under the container's Python 3.14, and `mila-llm`'s
  `requires-python` is `>=3.12,<3.14`; `--no-deps` does not suppress that check. The script's own
  comment shows the ceiling was handled for the server deps and missed for the package. Verify in a
  container, then add `--ignore-requires-python` as the runtime image now does.
- [~] **Reproducible container build** — validated on clang-21 + gcc-15 host, CUDA 13.3. Remaining:
  build against the bind-mounted tree, and have CI build `FROM` the image rather than apt-installing.
- [~] **Linux/clang as a first-class platform** — WSL green, CI compiles under clang-21, the container
  builds and runs Gemma 4 FP4. The GCC 16 second oracle and the broadened matrix move to Future.

#### Library hygiene

- [~] **Dispatch error UX** — a missing `(Op, Device, Precision)` reads as one line, not a cascade.
  Core landed; the optional named kernel concepts and the `OperationDispatch.md` §12 reconcile remain.
- [ ] **Guided reading path** — one token's journey (embed -> attend -> sample -> decode) through the
  real source, readable by a strong C++ dev unaided.

### Model Distribution

- [ ] **The published model cards still say `/install`.** The sources are correct; the live copies on
  huggingface.co only change on a re-publish, and they are what a new user reads before they have Mila
  at all. Fold the card refresh into the next publish.
- [ ] **`--instruct` is undocumented in `--package` mode, and its absence is silent.** The flag is
  parsed (`ExportArtifact.cpp:142`) but missing from the package-mode option list (`:42-56`), so
  omitting it writes `instruct: false` into the manifest with no warning — changing the prompt
  template every consumer applies. Document it, and consider refusing an instruct-named model that
  declares otherwise.
- [~] **Sweep the remaining "artifact" prose to model/weights.** The ten model cards, Chat, the pybind
  layer and MIS are converted. Still open: the QuickStart Python samples and the maintainer docs
  (`Publishing/README.md`, `Tools/README.md`, `getting-started.md`, `Data/Models/README.md`,
  `Tools/Quantization/README.md`); `Mila/Src` prose is 117 occurrences over 21 files, the low-priority
  tail. Must NOT change: `tool_bridge.py:84`/`:455`. [[project_artifact_vocabulary_rule]]
- [ ] **`GB` is printed for a GiB division across the whole toolchain.** Six sites in
  `ExportArtifact.ixx` (`:402`, `:470`, `:603`, `:682`, `:800`, `:1108`) and `formatBytes` in
  `Cli.ixx:64`, which is what `mila models` shows a user. Consistently 7% off; one shared helper.
- [ ] **Only Gemma refuses a pre-quantized model whose policy is not the one it compiled.**
  `GemmaModel.ixx:640` (and `:704` for the footprint sibling) compares `reader.getWeightQuantization()`
  against the requested policy; `LlamaModel::fromPretrainedImpl` and `GptModel` never read it. The
  storage dtype cannot substitute — FP4 at group 128 and 64 are both U8 — so a mismatch reinterprets
  the nibble layout and runs wrong. `ExportArtifact` emits Llama weights, so the hole is reachable.
- [ ] **`ModelSerialization.md` Phase 7 describes work that shipped.** The distribution path exists end
  to end — `savePretrained` (`LanguageModel.ixx:116`), the `mila_quantization` metadata key, the
  reader, the policy check, `Linear`'s pre-packed load branch, and `Tools/ExportArtifact` driving it.
  The phase text still calls it unwritten and the freeze-boundary table still lists it out of bounds.
- [ ] **A mistyped model name is reported as an authentication failure, and only to users without a
  token.** `HuggingFaceHub.ixx:283` maps every 401 to "no valid HuggingFace token", and HuggingFace
  hides repository existence from strangers — so an authenticated caller gets 404 and the right
  message while a new user is sent to obtain a token they never need. Invisible to anyone who has run
  `huggingface-cli login`; a typo is the likeliest failure on the evaluation path. When no token was
  sent and the owner is `mila-llm`, lead with the name being wrong.
- [ ] **No C++ tool has a `pull` verb**, so the cold download cannot be exercised from a C++-only
  machine without a human at the `/install` prompt. Python is covered — `ModelStore.pull` is bound
  (`Mila_py.cpp:309`) and is what pulled 6.33 GB in the Linux clean room — so this is a gap in the
  tool. It lands on `mila` with the other store verbs, and is not `ExportArtifact --fetch`.
- [ ] **`gpt2-small` installs and then cannot be used from Chat**, so it is the wrong first model for
  a quick start: the walkthrough ends in a 623 MB download and no conversation. Chat refuses base
  models by design, and `/models` now says so in the row — but that is *after* the download. Either
  the getting-started paths name an instruct model, or `/install` says so before the transfer.
- [ ] **`gpt2-small`'s installed record predates `kLicenseRole`.** The store copy declares weights and
  tokenizer only, so the hub repo carries LICENSE and the local disk does not — the exact split the
  legal-files change exists to close. Reinstall from `Data/Models/Packages/gpt2-small`; both blobs are
  already adopted, so it costs one small file.
- [ ] **`gemma-4-12b-it-fp4` now has two manifests.** The package directory carries the current one;
  `ModelCards/gemma-4-12b-it-fp4/mila.json` is the pre-package copy and no longer matches. One of them
  has to go, and the card directory's `publish.json` flow goes with it.
- [ ] **The licensing story is per-family and must not be generalized.** Gemma 4 is Apache 2.0 (public,
  ungated); Gemma 3 and earlier carry the Gemma Terms of Use; Llama 3.1/3.2 may be republished but
  attributed — ship the agreement, display "Built with Llama" and Meta's notice, pass along the AUP,
  and begin the model name with "Llama". Gating is a policy choice, not a licence condition.
  [[project_gemma4_apache2_license]]
- [ ] **`NOTICE.md:33` omits curl, and may no longer need to.** The note treats notice-carrying as open
  for "a binary distribution that links them", but both wheel presets are now
  `MILA_ENABLE_LIBCURL=OFF`, so a wheel built today contains no curl at all. Establish whether the
  published artifact predates that change; the answer decides whether this is an obligation or a
  non-issue. The same note points at a bucket that no longer exists — fix that either way.
- [ ] **The README implies FP8 and BF16 are reachable, and after an FP4-only publishing decision they
  are not.** `applyRequestedQuantization` refuses to reload pre-quantized weights as anything else, so
  every published model is FP4-at-runtime and the FP8 rows at `README.md:163,165` are converter-only
  capabilities. Say so, or the table promises a deployment nobody can reach.
- [ ] **`prune()` is destructive on a store that predates records.** Every pre-record blob is by
  definition unreferenced, so a first sweep on an upgraded store reclaims all of it — 6.33 GB in the
  case observed. Blobs-with-zero-records is a recognizable state and should be reported rather than
  silently swept.
- [ ] **Two Validated Capabilities rows are withheld pending evidence, and will be forgotten
  otherwise.** `pip install mila-llm` goes in once the Windows clean-room gate is green and the wheels
  are on PyPI; the footprint pre-flight goes in once GPT-2 has `getRequiredMemory` and Gate B has
  covered `NoWeightQuant` — until then it can only be claimed for Gemma 4 and Llama.

### Product Family — Adaptor Validation

- [ ] **`gemma_protocol.py` is retired in place and nothing imports it — delete it when ready.**
  Its 856 lines are superseded by `Gemma.Protocol.ixx` plus `gemma_bridge.py`, and it carries a
  header saying so. Kept on disk per the retire-don't-delete rule; removing it is a VS deletion.
- [ ] **In-turn thoughts dropped between tool calls** — Google's multi-turn rule is to strip
  prior-turn thoughts and keep the current turn's.
- [~] **MIS Gemma 4 tool-calling validated end-to-end** — Codex and Claude Code CLI round-trips are
  live and the native grammar is reconciled to Google's canonical template, pinned by an oracle.
  Remaining: N sequential distinct tool calls in one turn, channel-content parser polish, and
  Codex-CLI re-validation on the reconciled grammar.
- [ ] **Qwen streams nothing — the harness routes tokens by Gemma's four control ids.**
  `FamilyTraits::streaming_capable` is false for Qwen (`Chat.FamilyTraits.ixx`), so a 27B answers in
  one buffered block after a long silence. Qwen has one marker pair, `<think>`/`</think>`, which is
  enough to route reasoning from answer; the per-token router just has not been written for it.
- [ ] **Prompt-prefix reuse is unavailable on any model with DeltaNet layers, and the refusal is
  silent.** `QwenDeltaNetBlock::rewindKvCache` always returns false — correctly, since a recurrent
  state is a lossy summary — and `QwenTransformer::rewindKvCache` ANDs that into a whole-stack
  refusal. MIS must report it as a model property and plan around it, not retry; Chat is exempt. The
  block mechanism exists (`snapshotState`/`restoreState`); a whole-model policy does not.
- [ ] **`mila serve <args>` is broken on Windows and cannot report the server's exit code.**
  `runProgram` (`Cli.ixx:100`) hands a concatenated string to `std::system`, so cmd.exe strips the
  outer quotes of the whole command line and no argument survives; the code returned is the shell's.
  Launch with an argument vector (`CreateProcessW` / `posix_spawn`) behind a CMake-selected module
  partition, since module code carries no `#ifdef`.
- [~] **Rework Chat configuration to layered resolution** — design and phasing in
  `Mila/Specifications/ChatConfiguration.md`. Phases 1-5 have landed. Remaining: phase 7, the two
  `ModelRecord` fields, which touches Model Distribution.
