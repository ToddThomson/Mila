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

- [ ] **`ExecutionContextFactory.ixx:30-33` uses `#ifdef MILA_HAS_CUDA` inside the module purview.**
  The guard sits in the exported `createExecutionContext` body rather than the global module fragment;
  the CUDA arm belongs in a partition or a CMake-selected unit. [[feedback_no_ifdef_in_modules]]
- [ ] **`Component` documents a compute contract it does not declare.** `Component.ixx:132-133` and
  `:728` teach "`forward()` requires `build()`" and "`backward()` requires `isTrainingMode()`", but
  neither the base, `CompositeComponent` nor `Network` declares those methods. Correct the prose to
  name the concrete methods it means.
- [ ] **`Component` carries training's bookkeeping without training's act.** `zeroGradients()` (`:362`)
  and `getGradients()` (`:718`) are on the base and `backward` is not, so `Linear::getGradients()`
  returning empty (`Linear.ixx:524`) leaves "inference-only" and "nothing accumulated yet"
  indistinguishable. `TransformerApiReadiness.md` item 8 argues this at network level only.
- [ ] **`GemmaConfig::getRotaryDimForLayer()` is dead library code.** Its only callers are two
  assertions in `Gemma.Config.cpp:183,198`; the live path reaches the same value through
  `rotaryDim()` -> `getGlobalRotaryDim()` (`Gemma.Block.ixx:184`). Two names for one concept, the dead
  one reading as the live one. Delete it; point the test at the live accessor. `Gemma.Config.ixx:536`
- [ ] **The FP32 materializing softmax kernels still store and reload the score row.**
  `Gqa.Prefill.Fp32.cu:67`/`:73` and the FP32 common softmax park unnormalized exponentials in
  `att_row` and reload them to normalize — a wasted round trip over the widest prefill transient. The
  BF16 kernels recompute `expf` on the store pass; in FP32 that buys no accuracy, so measure first.
- [ ] **Nothing stops a new BF16 softmax kernel reacquiring the store/reload.** The four materializing
  sites were found by grep rather than by a failing test, and one was written three months after the
  decode path was fixed. Masking differs per site (causal, causal+padded, causal+window, ring-slot) so
  a shared helper buys little; a test pinning single-narrowing against an FP32 oracle would.
- [ ] **A parity script cites a debug flag that no longer exists.** `kGemmaDumpActivations` is gone
  from `Mila/Src`, but `Gemma/gemma_4_BF16/hf_gemma_activation_dump.py:4` still tells the reader to
  diff its output against it. The replacement is `LanguageNetwork::setStageProbe`
  (`LanguageNetwork.ixx:101`); `GemmaModel::fingerprintPrefill` (`GemmaModel.ixx:315`) is not the
  substitute — it localizes a NaN rather than comparing per-layer activations.
- [ ] **A consumer cannot instantiate a CUDA component without importing a non-public module.**
  `Mila.ixx` exports `Compute.IExecutionContext` but not `Compute.ExecutionContext`, and
  `CudaGqaOp::build` (`CudaGqaOp.ixx:260`) needs the latter COMPLETE — so consumer-side instantiation
  fails with "use of undefined type" until it imports that module directly. Decide whether the
  umbrella exports it. [[project_cuda_component_needs_execution_context]]
- [ ] **No build or CI step runs `compute-sanitizer`, and nothing else finds this class of defect.**
  `Mila/Src` carries zero error checks across its 110 kernel launch sites — defensible, since an
  in-kernel fault is async, but it makes an out-of-bounds access that changes no output invisible; the
  W4A8 staging defect survived the full passing suite and one sanitizer run named it. Add a pass over
  a targeted CUDA subset (`compute-sanitizer --tool memcheck`, roughly 10x slowdown).
- [ ] **The Llama chassis never received Gemma's memory gates, so 8B FP4 costs more than 12B FP4.**
  The embedding and `lm_head` ignore the weight-quantization policy and are untied. Three fixes, each
  mirroring Gemma: pass the policy to `TokenEmbedding` (`Llama.ixx:117`) and `lm_head` (`:119`), and
  implement tying when `tie_word_embeddings` is set. Llama's `preatt`/`att` also span the full context
  where Gemma's ring — separate, and dominant at long context. [[project_llama_chassis_memory_gates]]
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
  embeddings sized exactly `context_length` — and it gives the `generate()` crash a budget.
- [ ] **A pre-flight that cannot answer says nothing at all.** `Chat::predictFootprint` (`Chat.ixx`)
  catches every exception and returns `nullopt`, so an unreadable header shows as silence followed by
  a confusing failure at load. One line at `verbose`. [[feedback_absent_output_is_evidence]]
- [ ] **`FamilyTraits::default_context` is a compiled-in guess at the question the footprint API now
  answers.** `Chat.FamilyTraits.ixx:61` hard-codes 512 for Gemma, 4096 for Llama, 1024 for GPT-2,
  while `resolveAutomaticContext()` derives the answer in milliseconds and no VRAM. Keep the constant
  only as the no-CUDA fallback, already the role `main.cpp:863` passes it in.
- [ ] **The published model cards still say `/install`.** The sources are correct; the live copies on
  huggingface.co only change on a re-publish, and they are what a new user reads before they have Mila
  at all. Fold the card refresh into the next publish.
- [ ] **`/context`, `/set`, `/thinking` and the `/model` subcommands have no tests.** They are the
  first Chat commands that rebuild a model, refuse an input on derived arithmetic, or resolve a name
  case-insensitively. Cover the context floor, the ladder's fit search, and `resolveStoredName` — that
  last matters most, store lookup being a path lookup and so case-sensitive on Linux, not on Windows.
- [ ] **An `unknown` GPU FIT verdict prints no reason anywhere.** `verdictFor` distinguishes
  measured-and-too-big (`no`) from could-not-predict (`unknown`), but the reason is discarded. It
  belongs at `/verbose all`, matching `reportFootprintBeforeLoad`; the listing does not currently
  receive the detail level.
- [ ] **`/models` measures a per-model context and then throws it away.** `LadderFit::context_length`
  holds the largest fitting rung and the column does not print it, because the ladder tests memory
  alone — its top rung claimed `128K` for Gemma where the session runs 56320. A `CONTEXT` column needs
  the chunk test on the ladder (`FootprintPrediction::prefill.isBudgetConstrained()`) and finer rungs;
  at 1-2 ms per probe a six-rung ladder is ~12 ms per row. Only if users pick models by context.
- [ ] **`temperature`/`top_k`/`top_p` still have no command-line flags.** `/set` reaches them in a
  session and `session.json` at startup, so a `-p` one-shot cannot vary them at all — the invocation
  most likely to want a fixed temperature. `main.cpp:935` reads all three from settings already, so
  this is three flag producers, not a design.
- [ ] **A per-row disk figure, if one ever returns, should be reclaimable bytes** — the blobs that
  model alone references. That is what deciding-what-to-delete wants, and prune's mark-and-sweep
  already computes the refcount; it is simply not exposed as a per-model query.
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
- [ ] **GQA standalone-`forward()` stub** — component-level Gemma/Llama attention has no independent
  correctness oracle, and `GroupedQueryAttention.ixx:177` returns an un-computed `output_view_` on an
  unreached branch. Precondition for retiring the legacy GQA path; clears the C4702 below with it.
  See `Specifications/GqaMemory.md`.
- [ ] **GQA `forward()` fallback is stale** — `GroupedQueryAttention.ixx:299` records the non-KV-cache
  fallback as needing a correctness review, with the shape derivation commented out beneath it.
- [ ] `CudaMhaOp.ixx:433` initializes `active_max_seq_len_ = T_` with the reason unrecorded — confirm
  against the two-phase KV-cache contract (prefill full sequence, decode `outer_size == 1`).
- [ ] `GptModel.ixx:386` hardcodes `eos_token_ = 50256` — should come from tokenizer metadata.
- [ ] **`LlamaModel`'s context-overflow guard has no test.** `LlamaModel.ixx:336` carries the bound
  GPT-2's three cases pin, and nothing exercises it. Llama's overrun is the quiet one — it walks the
  KV cache rather than crashing — so absence of a report is not evidence. Template:
  `Tests/Dnn/Models/GptModel.Cuda.cpp`, a weightless checkpoint at a small deployment context.
- [ ] **[contributor]** Llama 3.2 1B/3B weight tying — the aliasing plumbing shipped; add
  `tie_word_embeddings_` + post-load aliasing + `getMemoryStats` correction to `LlamaTransformer`.
  See `Specifications/WeightTying.md` §6.
- [ ] **`cuda_fp4a16_gemm` and `cuda_fp4_dequantize_to_bf16` fall through to a SILENT no-op on an
  unsupported group size.** Both switch on `group_size` with `default: break` (`CudaW4A16Gemm.cu:398`,
  `:428`), so a size outside {64, 128} launches nothing and leaves the staging buffer holding the
  previous strip — wrong logits, no error. `CodebookDequantize.cu` now throws; make these match. Only
  reachable by adding a `PerGroupFp4<N>` policy, which is why it has never fired.
- [ ] **`AttentionOutputGate` now has two callers and one of them is not attention.**
  `QwenDeltaNetBlock` uses it for the mixer's output gate. The component is mechanically generic
  (`out = TGate(gate) * value`); the name is not. Rename, or accept the mismatch deliberately.
  `Components/Attention/OutputGate/`
- [ ] **The MTP head cannot be gated against HuggingFace at all.** transformers 5.12.1 declares
  `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` and implements no MTP class, so the parity
  harness has nothing to compare against and the wiring is read from tensor shapes and family
  convention. The converter skips the tensors today.
- [ ] **`getRequiredMemory` is unimplemented on nine components, and the base throws by design.**
  Gelu, MultiHeadAttention, Lpe, GatedMLP, MLP, SoftmaxCrossEntropy, LayerNorm, Softmax and GptBlock —
  so `GptModel::getRequiredMemory` throws the way Qwen's did. The contract lands family by family
  (`Core/Component.ixx:615`); GPT-2 is the family still outstanding.
- [ ] **The Llama converter writes a metadata key the reader never parses.** It emits `norm_eps`;
  `parseMetadataJSON` extracts `norm_epsilon` (Gemma and the packer both emit that). Harmless only
  because `LlamaModel::configFromMetadata` never reads the epsilon.
  `Tools/Converters/Llama/convert_weights.py:188`

### Observability

- [ ] **`setStageProbe` is undesigned public API that observability supersedes.**
  `LanguageModelNetwork.ixx:143`'s default accepts a probe and never fires it, so on Llama and Gpt
  "not instrumented" and "clean" are indistinguishable — a false negative in a NaN detector. Its one
  consumer reaches it through `if constexpr ( requires { ... } )`, so a signature change silently stops
  the probing. `TransformerApiReadiness.md` item 6
- [ ] **"One context, one model tree" is an accident, and observability makes it load-bearing.** Two
  models cannot share an `IExecutionContext` today — the factory always allocates
  (`ExecutionContextFactory.ixx:23`) and `Component::setExecutionContext` is protected and throws when
  already set (`Component.ixx:765`, `:779`). Nothing forbids a future overload accepting one, which
  would silently become a cross-model observation leak. State the contract. `Observability.md` §6.3
- [ ] **A value-reading sink has to name the model's compute precision.** The sink gets
  `const ITensor&`, whose `rawData()` is type-erased, so anything wanting numbers does a
  `dynamic_cast` to `Tensor<TPrecision, MR>` then its own `toHost`; both consumers now do this.
  Whether observation should offer a typed convenience is open — `Observability.md` §11.2, v0.21.

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
- [ ] **A test discards a `[[nodiscard]] GenerateStatus`, and warns on every build.**
  `QwenModel.Load.Cuda.cpp:189` calls `model->generate(...)` for its side effects inside a lambda,
  producing C4834. The status is the only channel reporting why generation stopped, so a test
  ignoring it cannot tell a completed run from an aborted one. Assert it instead of casting it away.
- [ ] **Backward-path kernels disabled or unverified.** `CudaSoftmaxOp.ixx:73` and `:103` throw
  `"needs review"` with the real calls commented out; `Gelu.Fp32.cu:65` records that the shipped
  backward is not the numerically stable `sech^2` form. Gradient-check these before the suite claims
  backward coverage — and sweep the *unmarked* backward kernels per-precision twin by twin: the RoPE
  FP32 backward was wrong while its BF16 sibling was correct, in a file carrying no marker at all.
- [ ] **[crash] Bring the GPT-2 CPU path up to current standards — its own session.** The CPU
  operation layer treats build-time extents as runtime extents, so CPU inference has never run a
  prompt shorter than its built context. Remaining after the encoder and LayerNorm fixes:
  `CpuLinearOp:259,264`, `CpuSoftmaxOp`, `CpuSoftmaxCrossEntropyOp`, `CpuAttentionOp` — the last not
  mechanical, since `B_`/`T_` size its `{B,NH,T,T}` buffer at `:269`. Pattern: `CudaLpeOp:192-196`.
- [ ] **`ResidualConfig` advertises a scaling factor that no backward implements and the two devices
  disagree about in forward.** CUDA forward honours it, CUDA backward takes no scale, and the CPU op
  ignores it entirely; the only guard is a debug-only assert at `CudaResidualOp.ixx:106`, so release
  builds train silently wrong. Cheapest correct fix, freeze-compatible because it removes an
  unimplemented knob: have `validate()` reject `scaling_factor != 1.0f` (`ResidualConfig.ixx:97`).

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
- [ ] **[decoupled]** Revive the loss + backward path (CrossEntropy / SoftmaxCrossEntropy) — both
  samples compute loss host-side, so this is off the critical path to a converging sample.
- [ ] **[net-new, training-only]** Revive the `Dropout` component.
- [ ] **Validation** — the **FP32** training path proven by the primitive suite (gradient checks,
  step-convergence, loader contracts, init-at-precision, the integration test), CI-gated; samples run
  as demos. BF16 and GQA training move to the Training (advanced) release.

### API Documentation

- [ ] **Tier 3 — semantic staleness** (retired-world prose). Folded into Test Suite Revival: fix a
  file's prose while it is already open for re-greening.
- [ ] **Nothing checks Doxygen when doc drift is introduced.** A break from a `Src/**` or `README.md`
  change is caught only by `publish-site.yml`, which is now manual — so nothing exercises Doxygen
  between publishes at all, and seventy-five errors once accumulated unseen and then blocked the site.
  Add a non-deploying Doxygen check to `build-pipeline.yml` (no CUDA, no CMake).

### Production Hardening

- [ ] **The Phase 5 prompt set is the weak link in the quality story.** Six prompts is enough to show
  trajectory cost and fork index are decoupled, not enough to put a threshold on the mean the way 1.25
  sits on the perplexity ratio, and the classes that matter for this model's use are
  underrepresented — one code prompt, no tool call, no multi-turn, no long-context prompt. Widening is
  two loads and ~18 s per six prompts. `DISABLED_DivergenceAgainstTheOracle`
- [ ] **The 16K perplexity gate needs a re-run before any 32K claim.** From 8K to 16K the oracle
  improves 7.2% and the plan only 3.4%, so the quantized arm captures about half the benefit of extra
  context — the compounding signature the recurrent layers make plausible. Table and caveats in
  `Qwen3.8.md` §8 item 9; `DISABLED_QualityGateAcrossContextLengths` is the harness.
- [ ] **CUDA's device 0 is not nvidia-smi's, and the default `DeviceId{ Cuda, 0 }` picks the 12 GiB
  card.** Any load sized for the 16 GiB card aborts by default with no diagnostic in about two
  seconds, so the failure looks like a model defect rather than a device choice. Every measurement
  landing on a specific card pins `CUDA_VISIBLE_DEVICES` to its UUID. Worth a note wherever
  `fromPretrained`'s default device is documented. [[project_cuda_index_is_not_nvidia_smi_index]]
- [ ] **The head's two paths do not agree to the last digit, so a perplexity comparison must fix the
  width.** Same weights and corpus, width 1 (decode matvec) and width 64 (W4A8-FP8 GEMM) differ in the
  third decimal. Small, but head width is part of the measurement protocol rather than a free
  performance knob — both arms of a quantization comparison must use the same one.
- [ ] **Scoring speed is a prefill problem, so do not build the device-side reduction.** Measured: the
  model forward is 68% of scoring cost and the transfer is negligible, capping a perfect device-side
  reduction at 1.45x by Amdahl. If scoring speed is ever wanted, parallelise the host `exp` loop
  across cores — the rows are independent, no kernel, no numerics risk. `Qwen3.8.md` §8
- [ ] **Only Qwen can be scored.** Gemma, Llama and GPT-2 build their heads at T=1 with no width
  parameter (`Gemma.ixx:279`, `Llama.ixx:239`, `GptTransformer.ixx:365`), so `scoreTokens` throws the
  base's `logic_error`. The gate wants Gemma and Llama too, and each needs the same two pieces: a head
  width in its config, and the window loop.
- [ ] **`SoftmaxCrossEntropy` has never been compiled, and its CUDA dispatch promises a BF16 it cannot
  deliver.** Its only test is commented out (`Mila/Tests/CMakeLists.txt:276`) and written against a
  3-parameter component that now takes 2; `OperationTraits.Cuda.ixx:422` maps
  `<CrossEntropyOp, Cuda, BF16>` to a type with no `__nv_bfloat16` specialization, the `half` one a
  silent stub (`CudaSoftmaxCrossEntropyOp.ixx:99`). The vocab >= 1024 block kernel has never run; its
  `__shfl_sync` gives warps 1+ `-INFINITY`.
- [ ] **`Qwen3.8.md` §8 gates the 16 GiB oracle on token-for-token cross-arch agreement, which cannot
  pass at any precision.** BF16, FP8 and FP4 all fork between Ada and Blackwell, at a token index set
  by the prompt rather than the precision, while each card is deterministic run-to-run — FP
  non-associativity, not a defect. Restate the gate as teacher-forced; perplexity never samples.
- [ ] **Every FP4/FP8 perf number in the tree is Ada-at-x16, and the 4070 now sits on a chipset Gen4
  x4 link.** No baseline was captured before the move, so the published figures carry an unmeasured
  link change. Re-measure before any of them is quoted again.
- [ ] **`PerGroupFp4` carries FP32 scales, and the offline tooling simulates FP16 ones.**
  `Policies.ixx:112` sets `kScaleDtype = FP32`; `formats.py`'s `fake_fp4_e2m1` rounds the scale
  through FP16, because `Qwen3.8.md` §5 budgets FP4 at 4.125 bits. The two disagree by 0.125 bits per
  weight, so the packer's simulated damage is not the damage Mila's quantizer inflicts. Moving the
  runtime to FP16 scales is the fix §5 names, and a free win for Gemma and Llama independently.
- [ ] **The FP4 and FP8 wire formats are defined only by a CUDA kernel.**
  `cuda_quantize_fp4_per_group` is the sole place the nibble order and `/6.0f` scale convention are
  written down, so every published model's byte layout is unreadable from CPU-only CI and unstatable
  in the spec. Copy `CodebookPacking.ixx` — normative layout, host codec, `--emit-fixture` holding
  kernel and Python packer to it both ways; that fixture is also the only reason `packing.py`'s
  `quantize_fp4_e2m1`/`dequantize_fp4_e2m1` are retained. Add `Fp4Packing.ixx`/`Fp8Packing.ixx`.
- [ ] **One concept, two names: the compute-precision template parameter is `TPrecision` in most of
  the tree and `TComputePrecision` in nine files.** Same axis, split along no principle — `Linear` and
  `GroupedQueryAttention` differ from their own siblings — and it has cost compile errors. Rename to
  the majority spelling: 126 occurrences, 9 files, 0 in `Mila/Tests`; not a blind sweep, since
  `GroupedQueryAttention.ixx` and `CudaRopeOp.ixx` use both. CLAUDE.md also mandates
  `TWeightQuantization` where the code says `TWeightQuant`.
- [ ] **`getStorageSize` is implemented three times** — `Mila::Dnn::detail::getStorageSize`
  (`Tensor.ixx:81`, carrying a `REVIEW:` that already asks why), `Detail::getStorageSize`
  (`TensorBuffer.ixx:221`), and `Mila::Dnn::storageBytes` (`Component.MemoryStats.ixx`). The two
  namespaces differ only in case. Blocker: `Tensor.ixx` cannot import `Dnn.Component` without a cycle.
- [ ] **Isolate third-party warnings structurally** with `/external:I` + `/external:W0` (`-isystem`
  for Clang/GCC). The target is warnings from third-party header text pulled into Mila's own TUs, not
  their sources (`/W4` at `Mila/CMakeLists.txt:87` is `PRIVATE` and never reached them). Precondition
  for any warnings-as-errors gate. Two frictions: those headers enter through module global module
  fragments, and `/external:` does nothing for nvcc diagnostics.
- [ ] **`save_` is public on `Component` and protected on `CompositeComponent`.** Legal, but the
  accessibility of one virtual then depends on the static type you hold, and a caller holding a
  concrete composite cannot invoke it (C2248, worked around with an `exposeSave()` forwarder in
  `Tests/Dnn/Core/CompositeComponent.cpp`). The trailing underscore suggests non-public is intended,
  making `Component.ixx:407` the wrong declaration.
- [~] **GPT-2 and Llama 3 pre-tokenization silently runs the ASCII fallback on every build and
  platform**, the published Linux container included: `\p{L}`/`\p{N}`
  (`BpePreTokenizationMode.ixx:33`, `:57`) compile in no standard `std::regex`, so
  `BpeTokenizer.ixx:344` throws and approximates, and no parity test catches it. Settle together —
  whether the fixtures are English-only (if so the site's parity claim is untested), and PCRE2/RE2
  against a hand-written Unicode scanner.
- [ ] **The BPE ASCII-fallback warning fires for every Llama and GPT-2 session, including the ones it
  does not apply to.** `BpeTokenizer.ixx:378` warns at construction under `std::call_once`, so an
  evaluator typing English is told about a path they never take, as the first thing after the welcome
  box. The claim is true and should not be softened; it should be timely — warn on first non-ASCII
  input. Touches `Mila/Src` and needs agreement; cheaper interim is one console line, detail in docs.
- [ ] **`CudaManagedMemoryResource.ixx:85` builds a detailed error message then throws a bare
  `std::bad_alloc`**, discarding it; `CudaPinnedMemoryResource.ixx:101` throws with no message at all.
  `CudaDeviceMemoryResource` gets this right — align both on `CudaBadAlloc` so an OOM says which
  device, which size, which resource.
- [ ] **`GroupedQueryAttention.ixx:216` C4702 is left deliberately** — one of the two warnings in the
  tree. It self-clears when the GQA training path is built, where a suppression would have to be
  remembered. For the warnings-as-errors decision: a blanket `/WX` forces it silent; escalating only
  the defect-class codes leaves it visible.
- [ ] **`Chat.Footprint.ixx:23` defines a variable in the global module fragment** — C5202, the tree's
  second warning. A GMF admits only preprocessor directives and the two `inline constexpr bool`
  definitions are not that; the `#ifdef` cannot simply move to the body either, so a CMake-supplied
  definition is the shape that fits. [[feedback_no_ifdef_in_modules]]
- [ ] **`IExecutionContext` is exported but unreachable in practice.** `Mila.ixx` re-exports it and
  `ExecutionContextFactory` as public API, but no model factory accepts one (`GemmaModel.ixx:119`
  takes a `DeviceId`) and `Component` holds a non-owning pointer owned by its parent. Decide: a
  `fromPretrained` overload taking `IExecutionContext*`, or drop both from the umbrella.
- [ ] **If C1128 recurs** on `MilaTests`, `ProfileModel` or `ExportArtifact`, switch from the
  per-target `/bigobj` on `ChatApp` to one project-wide `add_compile_options`. **Todd's call** — it
  touches every target's flags.
- [ ] **The README's six CI badges are decorative fiction.** `README.md:18-21` builds a Branch x
  (Build/Test/Docs) table by passing `job=build`/`test`/`docs` to the badge endpoint, which has no
  such parameter — all six fetch identically. `build-pipeline.yml`'s real jobs are `compile-and-gate`
  and `cpu-only-tests`, with no docs job at all. Two honest badges beat six that cannot fail apart.
- [ ] **Three different GCC floors are stated in the tree, and only one is measured.** `README.md:250`
  says GCC 16, `README.md:267` implies 15.3 works, and `CLAUDE.md` says GCC 15.3+. What was validated
  is 16 works / 15.2 fails; 15.3 has never been built. State the measured floor in one place.
- [ ] **Both onboarding docs state the build-option defaults backwards.** `MILA_ENABLE_TESTING` is
  `${PROJECT_IS_TOP_LEVEL}` and `MILA_ENABLE_DOCS` is `ON` (`CMakeLists.txt:67,77`), but
  `README.md:282,296` and `getting-started.md:79,212` call both `OFF`. `getting-started.md:221`
  compounds it with `find_package(Doxygen REQUIRED)` where the call must stay unqualified
  (`Mila/Docs/CMakeLists.txt:12`), and sells Graphviz for call graphs the Doxyfile disables.
- [ ] **A FetchContent consumer inherits Mila's `docs` target, aimed at the consumer's own source
  tree.** `MILA_ENABLE_DOCS` defaults `ON` and is not gated on `PROJECT_IS_TOP_LEVEL`, so a downstream
  configure offers a target whose `WORKING_DIRECTORY` and output path are `${CMAKE_SOURCE_DIR}`
  (`Mila/Docs/CMakeLists.txt:21-24`) — the consumer's root in a subproject build. Same class as the
  `tokenize` item; copy `MILA_ENABLE_SAMPLES`.
- [ ] **The preset list names four presets that do not exist and omits nine that do.**
  `getting-started.md:207` offers `x86-debug`, `x86-release`, `linux-debug`, `macos-debug` — none are
  in `CMakePresets.json`, where the Linux entries are `linux-clang-{debug,release}` and
  `linux-clang-cpu-{debug,release}`; `CLAUDE.md` repeats the `x86-*` pair. A reader following either
  gets "no such preset" on their first configure.
- [ ] **`getting-started.md:229` pins the dev container a release behind** — CUDA 13.0 / Clang 19 /
  CMake 4.x against the actual CUDA 13.3 / clang-21 / gcc-15 / CMake 4.2.3
  (`Docker/Dockerfile:18,48,50,61`). `README.md:321` and `Docker/README.md:13` both have it right.
- [ ] **`CLAUDE.md` documents the retired Chat alias set** — `gpt2`, `llama-1b`, `llama-3b`,
  `llama-8b` plus the `llama31`/`llama32` filename-prefix rule. There is no catalogue and no filename
  construction; `/model` takes a store name. Agent-facing rather than user-facing, which is why it
  rotted unnoticed, but it actively misdirects work.
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
- [ ] **Add Python 3.14 once 3.12 is proven.** It is the interpreter Ubuntu 26.04 ships and therefore
  the dev container's `python3`, which is why `Docker/build-mis.sh` still restates MIS's dependency
  list and installs MIS with `--ignore-requires-python`. Only a 3.14 wheel retires that duplication.
  Needs `uv python install 3.14` on Windows and one deadsnakes line in `Dockerfile.wheel`.
- [ ] **Publish `mila-llm-server` to PyPI.** The restructure is done and the version derives from
  `Version.txt`, so what remains is the release step: RELEASING covers the four CUDA wheels and says
  nothing about MIS. One `py3-none-any` file from `python -m build`, beside the wheel upload.
- [ ] **CI jobs have no `timeout-minutes`, so a hang costs six hours.** It has recurred in two
  different jobs on one day — once stalling in `Run CPU test suite` at 75+ min against a 14m29s
  baseline, once in `Build` on the pybind11 wrapper TU that compiled in 3m43s on the identical tree in
  a parallel run. A re-run is the only remedy available, and against a normal ~45-minute round trip a
  bound near 60 turns a repeat into a legible failure. `.github/workflows/build-pipeline.yml`
- [ ] **Split the packaging gate into its own job that configures but does not build.** The gate does
  not consume the parent build at all — it passes `MILA_SOURCE_DIR=${CMAKE_SOURCE_DIR}` and compiles
  Mila from scratch under `_deps/mila-build/`, needing only that CMake has configured. Today the
  release PR builds Mila for ~45 min and the gate builds it again, in series; as two parallel jobs the
  critical path is the compile alone. `Mila/Tests/Packaging/CMakeLists.txt:43`
- [ ] **A `dev` push and an open PR for the same SHA run the whole pipeline twice, and the redundant
  one blocks the merge.** The PR run is a strict superset — same tree, plus the packaging gates
  `build-pipeline.yml:114` skips on a `dev` push — but both report the same check names on the same
  SHA. Suppress the push run, and comment exactly when each job runs: an `if:` in this same file once
  hid a broken packaging gate for 32 commits, and a first pass proposed suppressing the wrong run.
- [ ] **`actions/setup-python@v5` still declares Node 20, which GitHub has deprecated** — it warns on
  every clean-room run. Every other action in the tree is on `@v5` and clean; bump it and re-check the
  rest — the deprecation applies by action version. `.github/workflows/wheel-cleanroom.yml`
- [ ] Add the Samples build to CI (only tests build today).
- [ ] Broaden CI compiler coverage toward the supported matrix (adds MSVC + GCC 16 to clang-21).

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
- [ ] **ONE image holding all of Mila, with two entry points** — not one per adaptor. Chat and MIS are
  two interfaces onto the same runtime: same `libMila`, same binding, same store on the same mount, so
  splitting them duplicates the library and makes the user choose an artifact before they know which
  interface they want. Chat is the default `CMD`; MIS is a second entry point. Accepted cost: a
  Chat-only user carries Python, the binding and FastAPI.
- [ ] **The real split is devel vs runtime, and only the runtime half is published.** Stage 1 is
  today's `nvidia/cuda:*-devel` base building everything; stage 2 is a `*-runtime` base carrying only
  the built binaries, the binding, MIS and the store tooling. That is where the size saving is, far
  more than any adaptor split. `Docker/Dockerfile` stays as the contributor build environment, never
  published.
- [ ] **Docker Hub Overview page is an authored surface, so give it a source in the repo.** It is what
  search shows and it carries the container-distribution message; hand-editing it in the browser is
  how the HF org card came to need a rewrite. [[project_four_channel_roles]]
- [ ] **Nothing cites `scripts/dockerhub/`.** Four files remain, in two channel groups; `RELEASING.md`
  and `wheel-cleanroom.yml` both reach into `pypi/`, but neither `README.md`, `getting-started.md` nor
  `Docker/README.md` names the image half. `build-runtime-image.sh` carries the published arch list,
  so it is undiscoverable knowledge until the publish script absorbs it.
- [~] **The runtime image ships a binding that cannot import, and the gate says it is fine.**
  `site-packages/mila/` holds only `__init__.py`, so `install` and `serve` both die on
  `ImportError: No module named 'mila._mila'` — the extension reaches the image only as a
  `POST_BUILD` side-effect into the source tree, which a cache-warm compile never re-runs. Install
  from `/build/python/mila`, where the build actually writes it.
- [ ] **The `ldd` gate passes when the file it checks is absent.** `Dockerfile.runtime`'s runtime
  stage greps for `"not found"`, but an unmatched glob makes the shell hand `ldd` a literal pattern
  and it answers `"No such file or directory"` — so the gate printed "Shared library check passed"
  over a missing extension. Assert the file exists first, then check its NEEDED entries.
- [ ] **The binding's staged extensions accumulate in the source tree and nothing prunes them.**
  MilaPy's POST_BUILD writes `_mila*.so`/`_mila*.pyd` into `Mila/Bindings/Package/src/mila/`, so a
  checkout collects one per interpreter and platform ever built, all untracked and all swept into a
  Docker build context until `.dockerignore` excluded them. Clean stale ones on build, or stage
  outside the source tree.
- [ ] **The wheel VERSION file is written into the source tree from any build.**
  `Mila/Bindings/CMakeLists.txt:65` runs `file(WRITE ...Package/VERSION)` at configure time,
  unguarded, so a FetchContent consumer writes into whatever tree Mila was fetched from — the same
  class as the POST_BUILD staging now behind `PROJECT_IS_TOP_LEVEL`. The two belong under one guard.
- [ ] **`Docker/build-mis.sh:76` looks broken on the current image.** It runs
  `pip install --no-deps -e Mila/Bindings/Package` under the container's Python 3.14, and `mila-llm`'s
  `requires-python` is `>=3.12,<3.14`; `--no-deps` does not suppress that check. The script's own
  comment shows the ceiling was handled for the server deps and missed for the package. Verify in a
  container, then add `--ignore-requires-python` as the runtime image now does.
- [ ] **The devel image's `mila-chat` wrapper shares its name with the binary it wraps, and its
  `cd /build` is redundant** — `executable_directory()` reads `/proc/self/exe`, confirmed by the
  runtime image running `chat` from `-w /` and `-w /tmp`. `Docker/Dockerfile:94` installs
  `run-chat.sh` as `/usr/local/bin/mila-chat`; the binary is `/build/mila-chat`. Drop the wrapper for
  a symlink or keep it for the not-built message, and drop the `cd` from `Docker/run-chat.sh:24`.
- [ ] **`Docker/README.md:69` credits ChatApp with a compiled-in `MODELS_DIR`.** It has none — the
  only `MODELS_DIR` in the tree is `Mila/Profiling/ProfileModel/CMakeLists.txt:22`. Chat resolves
  models through `MILA_CACHE_DIR` and its config through the working directory, which is why the
  published image can drop the bind mount. The claim reads as a hard dependency on `/mila`.
- [ ] Stage model weights off the Windows bind mount for the container (native disk speed).
- [~] **Reproducible container build** — validated on clang-21 + gcc-15 host, CUDA 13.3. Remaining:
  build against the bind-mounted tree, and have CI build `FROM` the image rather than apt-installing.
- [~] **Linux/clang as a first-class platform** — WSL green, CI compiles under clang-21, the container
  builds and runs Gemma 4 FP4. The GCC 16 second oracle and the broadened matrix move to Future.

#### Library hygiene

- [~] **Dispatch error UX** — a missing `(Op, Device, Precision)` reads as one line, not a cascade.
  Core landed; the optional named kernel concepts and the `OperationDispatch.md` §12 reconcile remain.
- [ ] **Five files still hand-roll the staging memory resource `DeviceTypeTraits` now carries.** Each
  writes `#ifdef MILA_HAS_CUDA` plus a `conditional_t` that is exactly `host_staging_memory_resource`:
  `Gemma.Block.ixx:820`, `Gemma.ixx:527`, `Llama.ixx:484`, `GptTransformer.ixx:615`,
  `GemmaModel.ixx:110` (and `LlamaModel.ixx`). Converting them removes six preprocessor blocks from
  module purviews. [[feedback_no_ifdef_in_modules]]
- [ ] **Module import hygiene** — Phase 0 exact-duplicate dedup, Phase 1 candidate report, Phase 2
  compiler-verified removal (Clang/GCC, not MSVC), plus domain-qualifying the generic single-segment
  module names (`Core`/`Utils`/`Components`/`Profiling` -> `Dnn.*`).
- [ ] **Delete the 16 `REVIEW:` markers whose disposition is already recorded** — no analysis left,
  only removal: the 12 in `CudaGqa.Dispatch.ixx` answered by that file's own banner at `:36`, plus
  `CudaOps.h:30`, `Linear.cuh:83`, `Component.ixx:299`, `CudaDeviceMemoryResource.ixx:139`.
- [ ] **The `fopen` -> `<fstream>` conversion is still available in three modules** —
  `SafeTensors.ixx` and `TokenSequenceLoader.ixx` are straight swaps and the library's only source of
  C4996. **`PretrainedReader.ixx` is not**: it deliberately uses positioned `ReadFile`/`pread`
  alongside the mapping, because faulting a large model through the mapped view throttles below disk
  bandwidth — that one needs the exemption. Clearing the first two unblocks the warnings ratchet.
- [ ] **ProgressReporter** — an injected per-operation progress facility for long-lived ops (BPE vocab
  training, `PretrainedReader` load, load-time quantization). `BpeVocabulary.ixx:624` is the concrete
  call site: an every-100-merges elapsed-time print asking to become an async callback.
- [ ] `Version::getMajor()`/`getMinor()`/`getPatch()` are non-const (`Src/Version.ixx`), so the
  version-skew comparison needs a mutable copy.
- [ ] **Guided reading path** — one token's journey (embed -> attend -> sample -> decode) through the
  real source, readable by a strong C++ dev unaided.
- [ ] **[contributor]** Llama-lineage CPU ops (`RmsNormOp`, `SwigluOp`, `RopeOp`, `TokenEmbeddingOp`,
  `CrossEntropyOp`) in `OperationTraits.Cpu.ixx` — demand-driven; absence is zero-cost on the GPU path.
- [ ] **[deferred, measure first]** Remove FP16 (superseded by BF16) — woven through live code; trace
  live-vs-dead first, and 8 `REVIEW:` markers already scope it. Note the odd row it collides with:
  CUDA `LayerNormOp` is registered at FP32 and FP16 and *not* BF16, so deleting the FP16 row leaves
  CUDA LayerNorm FP32-only. Pinned by a `static_assert`, so this work must confront it.

### Model Distribution

- [ ] **`ExportArtifact` gives the store verbs back to `mila`, and its own name is undecided.** Its
  install/rename/validate verbs duplicate the store tool, and Chat (`Chat.ModelCatalog.ixx:387`) and
  MIS (`model_worker.py:90`) point users at the wrong one; the nine modes should be subcommands, since
  `--package` is both a mode and an option of one (`ExportArtifact.cpp:212`). Seven touch no GPU, yet
  `Tools/CMakeLists.txt:10` gates the binary behind `MILA_ENABLE_CUDA`. **Name is Todd's call.**
- [ ] **`ExportArtifact`'s own naming drift, independent of the rename.** `--emit-manifest` is a
  synonym for `--package <dir>` differing only in its default directory (`ExportArtifact.cpp:394`);
  `ExportOptions`/`InstallRequest`/`PackageArtifactRequest` are three suffixes for one role; and
  `weightQuantizationVariantName` (`:103`) sits one character from `Src`'s `weightQuantizationName`
  while returning `cb2-3` where that returns `codebook` — a Qwen constant behind a generic name.
- [ ] **`--instruct` is undocumented in `--package` mode, and its absence is silent.** The flag is
  parsed (`ExportArtifact.cpp:142`) but missing from the package-mode option list (`:42-56`), so
  omitting it writes `instruct: false` into the manifest with no warning — changing the prompt
  template every consumer applies. Document it, and consider refusing an instruct-named model that
  declares otherwise.
- [ ] **`--fingerprint` is Gemma-only, so no other family has a load-parity probe.**
  `ExportArtifact.ixx:968` refuses anything without `fingerprintPrefill`, which means the two Qwen
  models cannot be diffed against their source the way a Gemma one can.
- [~] **Sweep the remaining "artifact" prose to model/weights.** The ten model cards, Chat, the pybind
  layer and MIS are converted. Still open: the QuickStart Python samples and the maintainer docs
  (`Publishing/README.md`, `Tools/README.md`, `getting-started.md`, `Data/Models/README.md`,
  `Tools/Quantization/README.md`); `Mila/Src` prose is 117 occurrences over 21 files, the low-priority
  tail. Must NOT change: `tool_bridge.py:84`/`:455`. [[project_artifact_vocabulary_rule]]
- [ ] **`ModelCards/TEMPLATE.md` does not exist.** `Publishing/README.md:40` says "the two Llama cards
  are the template", and template-by-example is what propagated one meaningless sentence into all six
  cards verbatim. The end-user prose rules are written; the template comes next, then the card
  rewrites. [[feedback_end_user_prose_boundary]]
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
- [ ] **`/models --online` answers SUPPORT but still cannot answer FIT.** How much context fits is
  unanswerable because `ModelManifest` (`ModelManifest.ixx:53`) carries no geometry. Two ways in,
  different owners: a `Range` read of the safetensors header so the online row runs the same
  `largestFittingContext` as the installed one (blocked on the footprint path taking a path, not a
  byte range), or geometry fields in the manifest — phase 7. Never an estimate.
- [ ] **`/models --online` costs one GET per listed model.** Invisible at one model, N+1 requests at
  N. Only worth revisiting if the published set grows; noted so the cause is known when it does.
- [ ] **Publish `Llama-3.2-1B-Instruct-fp4` as the evaluation model** — sequenced after the 3B path is
  proven. Roughly 0.7-0.9 GB against the 3B's 2.87, dropping the evaluation path's VRAM floor to about
  a gigabyte so an 8 GB card stops being excluded from "does it work". Convert, export, validate
  GENERATION rather than per-layer parity, publish with a card. Test against the tools-free system
  prompt first — a 1B is more prompt-sensitive than the 3B.
- [ ] **`gpt2-small` installs and then cannot be used from Chat**, so it is the wrong first model for
  a quick start: the walkthrough ends in a 623 MB download and no conversation. Chat refuses base
  models by design, and `/models` now says so in the row — but that is *after* the download. Either
  the getting-started paths name an instruct model, or `/install` says so before the transfer.
- [ ] **`gpt2-small`'s installed record predates `kLicenseRole`.** The store copy declares weights and
  tokenizer only, so the hub repo carries LICENSE and the local disk does not — the exact split the
  legal-files change exists to close. Reinstall from `Data/Models/Packages/gpt2-small`; both blobs are
  already adopted, so it costs one small file.
- [ ] **The org card defines a Mila model as "already quantized", which `gpt2-small` makes false.**
  The catalogue is now pre-quantized deployment models plus a reference model for reading and
  training. Say there that MIS does not serve GPT-2, so nobody files it as a bug. Card source is
  `.internal/Marketing/HuggingFaceOrgCard.md`.
- [ ] **`gemma-4-12b-it-fp4` now has two manifests.** The package directory carries the current one;
  `ModelCards/gemma-4-12b-it-fp4/mila.json` is the pre-package copy and no longer matches. One of them
  has to go, and the card directory's `publish.json` flow goes with it.
- [ ] **The "which licenses require a displayed attribution" rule is written twice in two languages** —
  `requiredAttributionFor` in `Chat.ModelCatalog.ixx` and `license_id.startswith("llama")` in
  `publish_model.py:209`. They agree today. A third family with a display duty is what separates them.
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
- [ ] **`gemma_greedy_parity.py` diffs an FP4 Mila against a BF16 HuggingFace reference and does not
  say so.** `Mila/Tools/Converters/Gemma/gemma_4_BF16/gemma_greedy_parity.py:70` loads through the
  binding's FP4 default, so any divergence it reports mixes quantization error with a real defect.
  `from_pretrained` now takes `quantization=`, so the honest comparison is one argument away — on a
  card that can hold a BF16 12B. State which it ran either way.
- [ ] **Packaging then installing hashes every file twice** — `buildPackage` hashes to derive the
  manifest digests and `install` hashes again to verify adoption (~50 s of the ~60 s Llama 3B
  migration, ~2 minutes on the 8B). Neither check is wrong alone, so the fix is a combined verb.
  `publish_model.py` has the same defect for its own reason.
- [ ] **`prune()` is destructive on a store that predates records.** Every pre-record blob is by
  definition unreferenced, so a first sweep on an upgraded store reclaims all of it — 6.33 GB in the
  case observed. Blobs-with-zero-records is a recognizable state and should be reported rather than
  silently swept.
- [ ] **`isAbandoned()`'s 24-hour lock reclamation is untested** — it needs a file with a backdated
  write time. Make the threshold a constructor parameter so a test can set it to zero; that is a
  better shape than backdating with `last_write_time()`.
- [ ] **A 15.09 GiB blob is orphaned in the local model store.** No record references the pre-export
  cb2-3 weights since the 11.05 GiB build replaced them, and the store has no garbage collector — the
  general gap, not this one file. A `mila` verb that lists unreferenced blobs and removes them on
  request is the shape.
- [ ] **`Mila/Tools` has no off switch** — gated on `PROJECT_IS_TOP_LEVEL` alone
  (`Mila/CMakeLists.txt:962`), so the wheel configure builds `tokenize` and `ExportArtifact`, neither
  of which can go in a wheel. Every other subdirectory has a `MILA_ENABLE_*`.
- [ ] **`tokenize` writes its executable into the consumer's source tree.**
  `Tools/Tokenize/CMakeLists.txt:13` sets `RUNTIME_OUTPUT_DIRECTORY` under `${CMAKE_SOURCE_DIR}`,
  which in a subproject build is the consumer's root. `PROJECT_SOURCE_DIR` is the fix, but the tool is
  run from `Data/Tools/<CONFIG>`, so moving it is a workflow change. Masked today by the
  `PROJECT_IS_TOP_LEVEL` gate above; it becomes live the moment Tools gets its `MILA_ENABLE_*` switch.
- [ ] **`mila/__init__.py` is copied by a `POST_BUILD` step of a target it is not a source of.**
  `Mila/Bindings/CMakeLists.txt:95` stages it with `copy_if_different` off
  `add_custom_command(TARGET MilaPy POST_BUILD)`, which runs only when `MilaPy` relinks — so editing
  only `__init__.py` leaves `<build dir>/python/mila/` stale and a sample fails with a missing
  attribute. Use `add_custom_command(OUTPUT ...)` with `DEPENDS` on the source.
- [ ] **Two Validated Capabilities rows are withheld pending evidence, and will be forgotten
  otherwise.** `pip install mila-llm` goes in once the Windows clean-room gate is green and the wheels
  are on PyPI; the footprint pre-flight goes in once GPT-2 has `getRequiredMemory` and Gate B has
  covered `NoWeightQuant` — until then it can only be claimed for Gemma 4 and Llama.

#### Website

- [~] **Reconcile `Web/content/start.md` with the Get Started band on the home page.** The four tabs
  landed as `#qs` in `Web/layouts/index.html`, but the nav and the home-page "Get started" box still
  point at `/start/` with older clone-and-build content, so the site has two getting-started surfaces.
  That page's §3 is retired in every sentence — conversion as the path, "no separate quantized
  checkpoint to manage", and "Llama and Gemma are gated", now backwards. This reconcile owns §3.
- [ ] **The home page hardcodes `0.20.0-beta.3` in three places, so the site and the release tag must
  ship naming the same version.** Two image tags in the Docker panel and the Evaluating band, plus the
  FetchContent pin; every later release breaks those commands until the copy is updated with it. The
  C++ tab compounds it by pinning `v0.20.0-beta.2` while its sample output reads `0.20.0-beta.3`.
  `Web/layouts/index.html` — `#p-docker`, `#evaluate`, `#p-cpp` steps 1 and 3.
- [ ] **The Evaluating band's commands leave a stopped container behind on every run.** No `--rm`, so
  a QA afternoon accumulated four and `docker image rm` then failed with a conflict the user has no
  context for. Nothing is lost — the model lives in the named volume. **The devel tab must NOT get
  `--rm`**: that image is a configured environment where the reader edits `~/myapp`, and its gap is
  the opposite — nothing says how to re-enter it. `Web/layouts/index.html`, `#evaluate`
- [ ] **`Web/content/docs.md:28` states "quantization has no checkpoint format."** True when written,
  false now — every published model is a quantized checkpoint. The surrounding point (the type chooses
  the reduced-precision path) still stands and should survive the correction.
- [ ] **The site links GitHub and nothing else.** No HuggingFace, no PyPI, so the primary marketing
  site does not point at the model store or the package. [[project_four_channel_roles]]
- [~] **Mila is a library, never a "runtime."** The noun names an engine you hand a model to, so it
  argues with "no hidden execution engine" in the same breath. Three user-facing sites remain:
  `Web/content/docs.md:38`, `blog/implementing-gemma-4.md:4`, `blog/gemma-4-docker-openai-api.md:4`.
  Not a sweep — "at runtime", "runtime dispatch" and the two places naming what Mila is *not* are
  correct. Whether `Mila/Src`'s own design name changes is a separate open call.
- [ ] **A blog post ships with no `discussion:` line** —
  `Web/content/blog/longer-context-fixed-the-crash.md`.
- [ ] **Two orphaned brand assets still carry the old Achilles mark.** `icon.png` at the repo root and
  `Web/static/achilles.png`, neither referenced by any page, template, README or the Doxyfile, which
  sets no `PROJECT_LOGO` at all. Delete rather than replace: `Brand/generate.py` emits the current
  mark into `Web/static/` only, so a root copy would be a second source to drift.

### Product Family — Adaptor Validation

- [ ] **`gemma_protocol.py` is retired in place and nothing imports it — delete it when ready.**
  Its 856 lines are superseded by `Gemma.Protocol.ixx` plus `gemma_bridge.py`, and it carries a
  header saying so. Kept on disk per the retire-don't-delete rule; removing it is a VS deletion.
- [ ] **In-turn thoughts dropped between tool calls** — Google's multi-turn rule is to strip
  prior-turn thoughts and keep the current turn's.
- [ ] Buffer Gemma Anthropic streaming only when tools are present.
- [~] **MIS Gemma 4 tool-calling validated end-to-end** — Codex and Claude Code CLI round-trips are
  live and the native grammar is reconciled to Google's canonical template, pinned by an oracle.
  Remaining: N sequential distinct tool calls in one turn, channel-content parser polish, and
  Codex-CLI re-validation on the reconciled grammar.
- [ ] **Qwen streams nothing — the harness routes tokens by Gemma's four control ids.**
  `FamilyTraits::streaming_capable` is false for Qwen (`Chat.FamilyTraits.ixx`), so a 27B answers in
  one buffered block after a long silence. Qwen has one marker pair, `<think>`/`</think>`, which is
  enough to route reasoning from answer; the per-token router just has not been written for it.
- [ ] **Qwen tool results are not merged into one user turn.** The checkpoint's template folds
  consecutive `tool` messages into a single `<|im_start|>user` turn holding several `<tool_response>`
  spans; `Qwen.Protocol.ixx` emits one turn each. Unreachable today — the harness dispatches one call
  per round — and it becomes wrong the moment parallel calls land.
- [ ] **Prompt-prefix reuse is unavailable on any model with DeltaNet layers, and the refusal is
  silent.** `QwenDeltaNetBlock::rewindKvCache` always returns false — correctly, since a recurrent
  state is a lossy summary — and `QwenTransformer::rewindKvCache` ANDs that into a whole-stack
  refusal. MIS must report it as a model property and plan around it, not retry; Chat is exempt. The
  block mechanism exists (`snapshotState`/`restoreState`); a whole-model policy does not.
- [ ] **Model capabilities belong in the manifest, not in a family switch.** `thinking_capable` and
  `streaming_capable` are both `family == Gemma` (`Chat.FamilyTraits.ixx`), and
  `default_context`/`max_context` are per-family constants beside them, so two models of one family
  cannot differ and a non-Gemma reasoning model reads as having no channel. `instruct` is already
  record-declared and proves the pattern; the manifest tolerates unknown fields, so this is additive.
  Do it before the next chassis threads a second switch.
- [ ] **`ToolCallParser::parse` routes ANY response containing `[` into the tool-call parser** —
  `Chat.ToolCallParser.ixx:63` uses `response.find( '[' )` where the class's own doc comment at `:35`
  says "Leading `[`" and the nested `parseTagged` path at `:109` tests it correctly. It degrades
  gracefully today, but any prose with a bracket enters the path, and a parse that ever *succeeds* on
  prose would swallow the answer and emit a phantom tool call.
- [ ] **`ModelSize` is dead.** Declared in `Chat.Config.ixx` with four values and read nowhere — the
  model's identity is its store name, which is what replaced it. Left in place it invites the next
  family to add a fifth value that nothing will ever read.
- [ ] **A session cannot move cards without restarting.** `--device N` and the `device` key choose the
  card at startup and every device question follows it, but there is no `/device` command — `/set` is
  sampling knobs only, by its own contract. `/context` shows the shape a reload-on-change command
  takes (`Chat.ixx`).
- [ ] **Library log output collides with Chat's spinner, and the first thing an evaluator reads is a
  corrupted line.** The spinner is mid-line when the BPE warning fires, so the warning is spliced into
  the loading message. `main.cpp:927` installs a stock `ConsoleSink`; since it derives from `Logger`
  and `Mila::initialize()` takes a sink, Chat can supply a spinner-aware one — adaptor-side, no
  library change. That sink should also drop `file:line:function` from user-facing output.
- [ ] **`printThinking` still takes the plain-text path.** The answer block paints style spans; the
  reasoning block does not, so a heading or bold label inside a thought renders unstyled
  (`Chat.Renderer.ixx:176`). Harmless today, but it is the one renderer entry point that ignores
  attributes — which is how a second convention starts.
- [ ] **Wrapped list items do not hang-indent.** A continuation line starts at the bullet's own indent
  rather than under the item text, so a wrapped item reads as a new paragraph. `wordWrap` preserves a
  line's leading indent but has no notion of a continuation indent (`Chat.RichText.ixx:99`).
- [ ] **`Chat.StreamingDisplay` has no tests.** `RichText` now has 18 (`Mila/Tests/Adaptors/Chat/`),
  but `holdPoint` and the chunk-boundary behaviour that produced the nested-bullet defect are
  unpinned. Harder than RichText: the module imports `Chat.Renderer` and `Chat.Config`, so it needs
  either a seam or those modules in the test target.
- [ ] **Chat's `context_length` needs an `auto`, and the interim clamp is a placeholder.** One session
  config serves every model a session loads, so the number is either too small for a 12B or fatal for
  GPT-2's 1024-row learned positions; today it is clamped by `maxContextFor` (`Chat.ModelCatalog.ixx`),
  a per-family constant honest only for GPT-2. The answer is the largest context that fits the card,
  which `getRequiredMemory(BuildContext)` computes — open: the headroom fraction, and the no-fit case.
- [ ] **Decide where a user's Chat config lives — a container user has nowhere to put settings.**
  `session.json` ships inside the image layer, so changing `temperature` means mounting a file over
  it, and `--config` assumes a file you can already write. Related: `chat-state.json` sits in the
  store root, which `resolveStoreRoot()` puts in a *cache* directory on Linux. Two shapes weighed
  (beside the store, or `MILA_CONFIG_DIR`); settle once — `context_length: auto` wants the same home.
- [ ] **A non-interactive `chat` must name its model, and only interactive surfaces are exempt.**
  Inferring one from a single-model store changes the command's meaning once a second is installed,
  and persisting the choice would put it in `chat-state.json`, which lives in a cache directory — so
  the quick start would work, then fail after an eviction. Site copy is fixed; the sweep of every
  surface showing a scripted `install` then `chat` remains. `CMD ["chat"]` is fine — interactive.
- [ ] **The download bar restarts per file and never says which file.** A model is a manifest, a
  tokenizer and the weights, so the user watches 0-100% twice with nothing distinguishing the runs.
  `ProgressCallback` is `(received, total)` only (`HttpClient.ixx:63`), so the CLI cannot label what
  it is drawing; adding the file name, or an (index, count) pair, is a library signature change and
  needs agreement. The sub-megabyte manifest is already suppressed in `Mila/Tools/Cli/Cli.ixx`.
- [ ] **`mila serve <args>` is broken on Windows and cannot report the server's exit code.**
  `runProgram` (`Cli.ixx:100`) hands a concatenated string to `std::system`, so cmd.exe strips the
  outer quotes of the whole command line and no argument survives; the code returned is the shell's.
  Launch with an argument vector (`CreateProcessW` / `posix_spawn`) behind a CMake-selected module
  partition, since module code carries no `#ifdef`.
- [ ] **`Chat.Json` is a byte-for-byte duplicate of the `nlohmann.json` module** —
  `Mila/Adaptors/Chat/Src/Json.ixx` versus `Mila/Src/Utils/json.ixx`, both including the same header
  from their global module fragment, and Chat imports one in `Chat.ixx` and the other in
  `Chat.ModelCatalog.ixx`. Drop `Json.ixx` from the target and import `nlohmann.json` everywhere.
- [ ] **`main.cpp` re-checks what the store already guarantees** — after `resolveModel` succeeds it
  tests `exists()` on both paths, but `locate()` refuses an incomplete record. Harmless duplication,
  except `/model` has no equivalent check; if the guarantee is doubted, the check belongs in the store.
- [~] **Rework Chat configuration to layered resolution** — design and phasing in
  `Mila/Specifications/ChatConfiguration.md`. Phases 1-5 have landed. Remaining: phase 7, the two
  `ModelRecord` fields, which touches Model Distribution.
- [ ] **`import Mila;` breaks the standard library in the consumer's translation unit.** Three
  failures in a real FetchContent consumer, absent without it: stream **input** fails on an undefined
  `basic_istream::sentry`; instantiating a model needs `<sstream>` **before** the import, since
  virtual `Component::toString()` compiles in via the vtable; import-before-includes is fatal (C1116).
  `Samples/QuickStart/Cpp/main.cpp` carries two workarounds. [[project_import_mila_breaks_std]]
- [ ] **Make `packaging_fetchcontent_consumer` instantiate a model and read input.** Its fixture is a
  version print, which is why the defect above sat undetected — the gate proves Mila *links*, not that
  its module is *usable*. It needs no GPU and no model to catch all three failures: they are
  compile-time. Cheapest possible guard for the entire C++ consumer story.
- [ ] **The Python binding discards `GenerateStatus`, so the two quick starts cannot reach parity.**
  All three sessions in `Mila_py.Wrappers.cpp` (`:553`, `:657`) do `(void)impl_->model->generate(...)`,
  so a Python caller cannot tell EOS from the `max_new_tokens` cap from context overflow from a
  cancellation. The C++ quick start prints `[stop]`; the Python one prints nothing, and that gap is
  visible to anyone reading the website's two first tabs side by side.
- [ ] **Neither Chat nor the quick starts have a test model, so neither has a test path.**
  `gpt2-small` loaded in seconds and surfaced both the `context_length` crash and the thinking-row
  defect; every remaining model is multi-gigabyte, and Chat now refuses base models. A single-shot
  sample — prompt in, tokens out, exit code — is CI-shaped given a model in the store, and
  `packaging_fetchcontent_consumer` proves only that its own fixture compiles. Both need one fixture
  that requires no download.
- [ ] **A full `mila-chat` QA pass is owed** — an uninstalled name through `/model <name>` (the only
  hub-fetch path), `resolveStoredName`'s ambiguity refusal, `/context` below the derived floor,
  `/set` bounds rejection, and the `unknown` GPU FIT verdict. Watch for: `/model <name>` no longer
  loads, and the break is silent.
- [ ] **Decide whether a Python completion sample needs a `GptSession` before it can exist.**
  `Samples/QuickStart/Python/generate.py` already shows completion via `--raw`, so the only gap is
  GPT-2 itself — `LlamaModel`, `GemmaModel` and `QwenModel` are the sessions the binding exposes,
  which is also why MIS refuses the architecture. A binding decision, not a sample one.

---

## Future

Next-cycle work. Coarse by design — detailed tasking happens only when an item promotes into a release.

- **[gate] One typed model handle + factory, before ANY next chassis** — the architecture-to-concrete
  erasure exists three times in two languages (Chat's `ModelVariant`, the binding's `*Session`
  classes, MIS's `ModelFamily`), which is why GPT-2 is missing from MIS. Lands in the runtime-adjacent
  native agent core; sequencing in `MilaProductFamily.md` Open Decision 2. **After the v0.20 tag,
  before the chassis expansion below.**
- **The library should own architectural identity** — the set of architectures is the set of model
  classes `Mila/Src` implements, held today as a compile-time type and an unvalidated manifest string
  with nothing connecting them, so each consumer writes its own bridge (`familyFromArchitecture` in
  `Chat.ModelCatalog.ixx:159`, `architecture == "gemma"` at `Mila_py.Wrappers.cpp:413`). Home is
  `Distribution`, beside the manifest reader, not `Dnn`. The library owns the identity only; traits
  merely keyed on it stay with the consumer they describe.
  [[project_architecture_identity_ownership]]
- **Qwen 3** — the dense decoder, thinking-mode suppression, model-agnostic tool calling, and FP8 KV
  cache; the `OperationTraits<GqaOp, Cuda, BF16, PerChannelKvFp8<>>` specialization lands here.
- **Qwen 3.8 FP4 — FP8 the embedding table so the model is wholly device-resident.**
  `QwenOraclePrecisionPlan::EmbeddingTable` is `NoWeightQuant` and host-resident, so every decode step
  gathers a row over PCIe; at `PerChannelFp8<>` the 1.271 B table halves to ~1.19 GiB and sits beside
  the 12.31 GiB FP4 body inside a 15.93 GiB card. Gemma 4's D4 Design B is the precedent and
  `TokenEmbedding` already accepts a per-channel table policy. `Qwen.PrecisionPlan.ixx:153`
- **Architecture / MoE** — the presumptive post-v0.20 tentpole; one router chassis unlocks Gemma
  26B-A4B, Qwen3-30B-A3B and gpt-oss-20b. [[project_moe_tentpole_direction]]
- **Gemma 4 MTP** — the self-speculative drafter, sequenced ahead of MoE.
- **Ministral** — SWA transformer; reuses the Llama foundation, Qwen 3 tool calling, and the Gemma 4
  SWA mask + bounded-KV ring.
- **v0.20 library-frozen tails** — the Generation API surface tail (`SamplerConfig` rename, Llama/Gpt
  seedable sampling, eager sampler, config-accessor propagation, `contextLength()` hoist), the
  Sample-API device-sampler migration for Llama/Gpt, and the Optimizer-dispatch migration onto
  `OperationTraits`. All `Mila/Src`, which is why they wait. Adaptor work does not.
- **Model serialization** — the remaining checkpoint round-trip and distribution phases. Design and
  phase plan in `Specifications/ModelSerialization.md`.
- **Retire quantize-on-load — one load shape for every policy.** `Linear::loadParameter` refuses a
  compute-precision blob, uploads packed bytes, binds, derives; the dtype sniff at `Linear.ixx:601`
  and `CudaLinearOp::quantize()` go, and FP8/FP4 fitting joins the sub-4-bit fitter in
  `Tools/Quantization` — one producer for every format, and model production stops needing a GPU. The
  codebook path is already this shape (`:574`). Depends on the FP4/FP8 codecs, and takes Chat's
  load-time quantization keyword with it. An API change to `Mila/Src`, which is why it waits.
  [[project_quantization_offline]]
- **Python binding — numeric access, not component access.** Add a session-level `forward()` returning
  logits, plus final hidden states, to `LlamaSession`/`GemmaSession`; from Python a parity run can
  compare token ids and nothing else today. Component, tensor and training bindings are ruled out:
  `TDeviceType x TPrecision x TWeightQuantization` is erased only at the session PIMPL.
  `Mila_py.Wrappers.ixx:362`
- **API Coherence** — the pre-1.0 consistency pass, and the precursor to any API-stability promise.
  First named item: **`loadModel`/`saveModel` and `loadCheckpoint`/`saveCheckpoint`** — verb plus what
  you get, both directions. "Pretrained" is relative to a fine-tuning stage Mila does not have and is
  doubly wrong on the write side; "artifact" is build vocabulary for a file that is simply a model;
  `from` names the *source* form, so `fromCheckpoint` earns it and `fromModel` cannot. Document the
  distinction: a checkpoint carries epoch and loss as one of a series, a model is terminal. One
  wrinkle: `Network::load( archive, mode )` restores into an existing graph. **The methods are the
  small half** — `kArtifactMinimumMilaVersion`, `ModelDistribution.md`, both model cards,
  `from_pretrained`, MIS and the samples all speak the old vocabulary. Sequence with the
  `ExportArtifact` rename and the binding's `quantize_fp8` fix.
- **Warnings-as-errors ratchet.** Requires the `/external:W0` isolation first; enforce in **CI only**,
  never locally; ratchet on the count *not increasing* before demanding zero; **MSVC first**, since
  `/WX` across three compilers means the union of three opinions must be zero; and dormant-but-retained
  code warns by nature — suppress per-file in CMake pointing at the owning task, never with
  `#pragma warning` in module code. Land it **after** v0.20 ships.
- **Training (advanced)** — Llama fine-tuning, loss-function GPU migration, gradient checkpointing,
  and BF16/GQA training.
- **Performance** — Gemma 4's levers (the fused W4A16 prefill GEMM, flash-attention on the global
  layers) and the codebook path's own, each a measured gap with its numbers in `Qwen3.8.md` §8: the
  decode GEMVs' bandwidth shortfall against FP4 (amortize the unpack across output rows, or bucket
  activations by code); staging the sub-4-bit prefill to FP8 so it reaches the sm89 tensor GEMM
  instead of a BF16 one, gated on e4m3's 3 mantissa bits over 2.82-bit codes; the per-chunk staging
  dequantize, unmeasured across the rung ladder (`Qwen.ixx:110`); tensor cores for the DeltaNet
  chunked kernel, worth ~13% of prefill; and whether Gemma's ring softmax is reachable from Qwen's 16
  full-attention layers. [[project_w4a16_prefill_gemm]]
- **Whole-model prefix caching for Qwen** — the 48 DeltaNet layers need the snapshot/restore copy and
  the 16 attention layers the positional rewind, and nothing combines them. Deferred as a policy
  question: how many prefixes to hold in host RAM, and eviction. `Qwen3.8.md` §8, `PromptCaching.md`
- **Native low-precision compute (Blackwell+)** — the microscaling data path and finer per-arch gating.
- **Compute backends beyond CUDA** — ROCm and Metal; `DeviceType::Rocm` / `::Metal` are reserved and
  unimplemented.
- **Platform portability — aarch64 + coherent memory.** Mila has never been built on ARM.
- **Model loading** — a load-time FP4 sidecar cache, and concurrent/async read I/O for real queue depth.
- **Ungated GPT-2 zero-auth quick-start** — a first-run HTTPS weights fetch.
- **`ComponentType` vitality** — does `getType()` earn its keep, or does the unused converter surface
  retire?
- **Discoverability** (internal, not a README theme) — the site is live at `mila.toddt.me`.
