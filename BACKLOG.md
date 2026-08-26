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

- [ ] **Observability — carved into v0.20.** A consumer holding a `LanguageModel` cannot reach a
  composition tree that is otherwise fully public, so every activation investigation in this repo
  was built as scaffolding outside `Mila/Src`. Design of record is `Specifications/Observability.md`;
  it supersedes the requirements this item used to carry and names its own boundary.
  The cost question is **settled**: an unattached publication check is not measurable on
  `Gelu::forward` (spec section 7, measured 2026-08-25), so the runtime design stands and no
  compile-time gate is needed. Still owed once instrumentation is real: the section 10 criterion
  of no movement in decode tok/s on the 27B at 16K.
- [ ] **`setStageProbe` is undesigned public API and becomes a consumer of the above.**
  `LanguageModelNetwork.ixx:143`'s default accepts a probe and never fires it, so on Llama and Gpt
  "not instrumented" and "clean" are indistinguishable — a false negative in a NaN detector. Its one
  consumer reaches it through `if constexpr ( requires { ... } )`, so a signature change silently
  stops the probing. `TransformerApiReadiness.md` item 6.
- [ ] **`ExecutionContextFactory.ixx:30-33` uses `#ifdef MILA_HAS_CUDA` inside the module purview.**
  The guard sits in the exported `createExecutionContext` body, not in the global module fragment,
  which is the preprocessor-in-module-code pattern the tree rules out; the CUDA arm belongs in a
  partition or a CMake-selected unit. Found while confirming the factory allocates a fresh context
  per call.
- [ ] **"One context, one model tree" is an accident, and observability would make it load-bearing.**
  Two models cannot share an `IExecutionContext` today — the factory always allocates
  (`ExecutionContextFactory.ixx:23`), every transformer mints its own, and
  `Component::setExecutionContext` is protected and throws when already set (`Component.ixx:765`,
  `:779`). Nothing forbids a future overload accepting a context, which is a reasonable
  optimization and would silently become a cross-model observation leak. State the contract on
  `setExecutionContext`. `Specifications/Observability.md` 6.3.
- [ ] **`Component` documents a compute contract it does not declare.** `Component.ixx:132-133`
  teaches the lifecycle as "`forward()` requires `build()` to have completed" and "`backward()`
  requires `isTrainingMode() == true`", and `:728` names both again, but the base declares neither
  — nor does `CompositeComponent` or `Network`. A reader is given rules anchored to methods that
  are not on the type. Correct the documentation to describe the concrete methods it means.
- [ ] **`Component` carries training's bookkeeping without training's act.** `zeroGradients()`
  (`:362`) and `getGradients()` (`:718`) are on the base; `backward` is not. It fails the way
  `setStageProbe` does: `Linear::getGradients()` returns empty when gradients were never allocated
  (`Linear.ixx:524`), so "inference-only" and "training, nothing accumulated yet" are
  indistinguishable to a caller. `TransformerApiReadiness.md` item 8 made this argument at network
  level only; the same asymmetry sits one level down.

- [ ] **`GemmaConfig::getRotaryDimForLayer()` is dead library code.** Its only callers are two
  assertions in `Gemma.Config.cpp:183,198`; the block reaches the same value through
  `rotaryDim()` -> `getGlobalRotaryDim()` (`Gemma.Block.ixx:184`). Two names for one concept,
  and the dead one reads as the live one — it was cited as evidence about the RoPE path during
  the 2026-08-19 investigation and had to be corrected. Delete it and let the test use the live
  accessor. `Gemma.Config.ixx:536`.
- [ ] **The FP32 materializing softmax kernels still store and reload the score row.**
  `Gqa.Prefill.Fp32.cu:67`/`:73` and the FP32 common softmax park the unnormalized exponentials
  in `att_row` and reload them to normalize. In FP32 the round trip is lossless, so this is
  purely wasted traffic — one write pass and one read pass over the widest transient in the
  prefill pipeline. The BF16 kernels were converted to recompute `expf` on the store pass
  (2026-08-19); FP32 was deliberately left alone because recomputing there buys no accuracy, so
  the trade is one `expf` against a global round trip and is unmeasured. Measure before copying
  the change across.
- [ ] **Nothing stops a new BF16 softmax kernel from reacquiring the store/reload.**
  `70007c7b` wrote one three months after `2491d1b7` fixed the decode path, and the four
  materializing sites fixed on 2026-08-19 were found by grep rather than by a failing test. The
  masking differs per site (causal, causal+padded, causal+window, ring-slot), so a shared helper
  would be three lines of arithmetic behind four predicates and buy little; a test that pins
  single-narrowing against an FP32 oracle would.
- [ ] **A parity script cites a debug flag that no longer exists.**
  `kGemmaDumpActivations` was temporary and is gone from `Mila/Src`, but
  `Gemma/gemma_4_BF16/hf_gemma_activation_dump.py:4` still tells the reader to diff its output
  against what Mila prints. The replacement exists and the script should name it instead:
  `LanguageNetwork::setStageProbe` (`LanguageNetwork.ixx:101`), implemented by both Gemma and
  Qwen. Note the nearest consumer is not a parity dump — `GemmaModel::fingerprintPrefill`
  (`GemmaModel.ixx:315`) prints stages only up to the first NaN, so it localizes a NaN rather
  than comparing per-layer activations.
- [ ] **A consumer cannot instantiate a CUDA component without importing a non-public module.**
  `Mila.ixx` exports `Compute.IExecutionContext` but not `Compute.ExecutionContext`, and instantiating
  any CUDA block reaches `CudaGqaOp::build` (`CudaGqaOp.ixx:260`), which needs `ExecutionContext<Cuda>`
  COMPLETE rather than merely reachable — so the instantiation fails with "use of undefined type".
  Gemma escapes only because `GemmaModel` instantiates its block inside the library; `Qwen.Block.Cuda.cpp`
  is the first consumer-side instantiation in the tree and had to `import Compute.ExecutionContext`
  directly. Same rule as the `Mila.ixx:207` policy note — decide whether the umbrella exports it.
- [ ] **No build or CI step runs `compute-sanitizer`, and nothing else can find this class of defect.**
  Measured 2026-08-15: `Mila/Src` carries 29 `cudaCheckStatus`/`cudaCheckLastError` calls across 7
  files, all in allocation, transfer and setup paths — and **zero** across the 110 kernel launch
  sites in 63 `.cu` files. That is a defensible design (an in-kernel fault is async, so catching it
  near its cause costs a synchronize per launch), but it means an out-of-bounds access that does not
  change any output is invisible: the W4A8 staging defect survived 32 days and 1606 passing tests,
  and one sanitizer run named it. Add a sanitizer pass over a small CUDA test subset —
  `CUDA/v13.3/compute-sanitizer/compute-sanitizer.exe --tool memcheck`, roughly 10x slowdown, so a
  targeted filter rather than the whole suite.
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
- [~] **Attribute the Gate B residual.** Scratch is measured and is *not* the answer (~230 MiB on
  both models, essentially model-independent), leaving 1.015 GiB unattributed on Gemma and 0.449 on
  Llama. The Qwen sighting that looked like a third instance turned out to be the un-pooled
  per-layer transients above, which is a different and much larger defect — so this item is back to
  Gemma and Llama alone, and the Qwen numbers should not be folded into it. Next and cheap:
  per-allocation rounding — read `MemoryAllocationStats::allocationCount` (import
  `Compute.MemoryResourceTracker` directly; `Mila.ixx:95` comments the re-export out) and divide.
  Noise floor ~50-70 MiB, so nothing under ~0.1 GiB is signal.
- [ ] **`cudaMemGetInfo` cannot see WDDM's shared allocation, so every VRAM measurement on Windows
  understates.** It reported "10.85 GiB consumed, 0 free" for a process that had committed 11.21 GB
  dedicated **plus 8.53 GB shared**. Anything that decides whether a model fits needs the
  per-process counters instead: `Get-Counter "\GPU Process Memory(pid_N*)\Dedicated Usage"` and
  `\Shared Usage`, which is what Task Manager reads. Worth a note in `MemoryFootprint.md`, since the
  footprint work's whole premise is answering "does this fit" before loading.
- [ ] **DECISION OWED — `BuildContext::withInstalledOutput` is an unenforced promise
  (`Component.BuildContext.ixx:208`).** The pooling predicate is authored three times in `Qwen.ixx`
  (`:382`, `:602`, `:626`) and the ~6.5 GiB DeltaNet understatement was one site existing while
  another did not. Root cause: the workspace factories fuse describing the slot set with allocating
  it, forcing prediction ahead of installation. Proposed split — bind unallocated pre-build,
  materialize in `build()` — written up in `MemoryFootprint.md` s4.5; direction not settled.
- [ ] **Gemma owes a block-level Gate A case, per the per-block-kind rule in `MemoryFootprint.md`
  s4.5.** `Gemma.Block.Cuda.cpp` calls `getRequiredMemory` nowhere; the local and global kinds share
  one max-geometry workspace and neither has a predict-vs-build case. Qwen has both. Blocked on an
  exported `makeGemmaBlockWorkspace` — Gemma builds its workspace inside the private
  `GemmaTransformer::allocateBlockWorkspace` (`Gemma.ixx:1110`), so no test can construct one.
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
- [ ] **`FamilyTraits::default_context` is a compiled-in guess at the question the footprint API now
  answers.** `Chat.FamilyTraits.ixx:61` hard-codes 512 for Gemma, 4096 for Llama, 1024 for GPT-2,
  while `resolveAutomaticContext()` derives the answer in milliseconds and no VRAM. Keep the constant
  only as the no-CUDA fallback, which is already the role `main.cpp:863` passes it in.
- [ ] **The published model cards still say `/install`.** The sources are correct as of 2026-08-17,
  but the live copies on huggingface.co only change on a re-publish, and they are what a new user
  reads before they have Mila at all. Fold the card refresh into the next publish.
- [ ] **`/context`, `/set`, `/thinking` and the `/model` subcommands have no tests.** They landed
  2026-08-17 and are the first Chat commands that rebuild a model, refuse an input on derived
  arithmetic, or resolve a name case-insensitively. Cover the context floor (transcript + reasoning
  + headroom), the ladder's fit search, and `resolveStoredName` — all pure functions of inputs a
  test can supply. The last one matters most: it exists because store lookup is a filesystem path
  lookup and therefore case-sensitive on Linux and not on Windows, so it is exactly the kind of
  thing that passes on a dev box and fails in the container.
- [ ] **An `unknown` GPU FIT verdict prints no reason anywhere.** `verdictFor` distinguishes
  measured-and-too-big (`no`) from could-not-predict (`unknown`), but the reason is now discarded —
  the table prints one note and nothing else, by design. `/verbose all` is where it belongs, matching
  `reportFootprintBeforeLoad`, which already names a missing prediction only at that level; the
  listing does not currently receive the detail level.
- [ ] **`/models` measures a per-model context and then throws it away.** `LadderFit::context_length`
  holds the largest fitting rung, and the column deliberately does not print it: the ladder tests
  memory alone, so its top rung claimed `128K` for Gemma where the session actually runs 56320 —
  `resolveAutomaticContext` also requires an unconstrained prefill chunk. A `CONTEXT` column needs
  the chunk test on the ladder (`FootprintPrediction::prefill.isBudgetConstrained()`, already
  returned) and rungs fine enough that Gemma does not read as 32K when the answer is 56320. Worth it
  only if picking between models by context length turns out to be a thing users do.
- [ ] **`temperature`/`top_k`/`top_p` still have no command-line flags.** `/set` reaches them in a
  session and `session.json` at startup, so a `-p` one-shot cannot vary them at all -- which is the
  invocation most likely to want a fixed temperature. `main.cpp:935` reads all three from settings
  already, so this is three `--flag` producers, not a design.
- [ ] **A footprint probe costs 1-2 ms, so per-row context resolution is affordable.** Measured
  2026-08-17 from `*FootprintCudaTests.GetDeploymentFootprint_*` (3 predictions per test, warm):
  Llama 32-layer 2-4 ms, Gemma 48-layer 5-7 ms, plus a one-off ~40 ms cold header read per
  checkpoint. Today's whole six-row table is ~20 ms warm. A full `resolveAutomaticContext` scan is
  1024-steps from the ceiling, so ~75-120 probes and 150-250 ms per row — affordable but not free;
  a coarse ladder (4K/8K/16K/32K/64K/128K, 6 probes) buys a per-row context for ~12 ms and is sound
  where bisection is not (`Chat.Footprint.ixx:347` — the curve is not monotonic).
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
- [ ] **`LlamaModel`'s context-overflow guard has no test.** GPT-2's is pinned by three cases in
  `Tests/Dnn/Models/GptModel.Cuda.cpp`; `LlamaModel.ixx:336` carries the identical bound and nothing
  exercises it. Llama's overrun is the quiet one — it walks the KV cache rather than crashing — so
  absence of a report is not evidence. The GPT-2 fixture is the template: a weightless checkpoint at
  a small deployment context reaches the boundary in single digits.
- [ ] **[contributor]** Llama 3.2 1B/3B weight tying — the aliasing plumbing shipped; add
  `tie_word_embeddings_` + post-load aliasing + `getMemoryStats` correction to `LlamaTransformer`.
  See `Specifications/WeightTying.md` §6.
- [ ] **The codebook decode GEMVs are 1.4x off the Section 5 ceiling.** 33.7 tok/s measured on
  the packed 27B, fully resident, against a 47 tok/s bytes-per-token ceiling — 314 GB/s of the
  4070's 504. The three GEMVs were 95% of decode time before the residency fix (cb4 45%, cb8
  27%, FP4 23%) and the attribution is owed a re-run at the new rate. Phase 1 already found
  these kernels instruction-bound and named the fix — amortizing the unpack across several
  output rows per thread, or bucketing activations by code. `DISABLED_DecodeRate` in
  `QwenModel.Load.Cuda.cpp` is the harness.
- [ ] **DeltaNet prefill runs the recurrence sequentially — O(T) in sequence steps.** The chunked
  UT-transform formulation is what makes long-prompt prefill affordable on the 48 DeltaNet layers,
  and without it the 27B is not shippable at prefill. The recurrent kernel is the oracle it must be
  validated against, bitwise where fp32 allows. `Cuda/Operations/DeltaNet/Kernels/GatedDeltaRule.cu`.
- [ ] **Prompt-prefix reuse is silently unavailable on any model with DeltaNet layers.**
  `QwenDeltaNetBlock::rewindKvCache` always returns false — correctly, since a recurrent state is a
  lossy summary that cannot be rolled back — and `QwenTransformer::rewindKvCache` ANDs that into a
  whole-stack refusal. Chat and MIS need to report this as a model property, not retry.
- [ ] **`AttentionOutputGate` now has two callers and one of them is not attention.**
  `QwenDeltaNetBlock` uses it for the mixer's output gate. The component is mechanically generic
  (`out = TGate(gate) * value`); the name is not. Rename to something generic, or accept the
  mismatch deliberately. `Components/Attention/OutputGate/AttentionOutputGate.ixx`.
- [ ] **The MTP head cannot be gated against HuggingFace at all.** transformers 5.12.1 declares
  `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` and implements no MTP class, so the Phase 4
  parity harness has nothing to compare it to. Its wiring is read from tensor shapes and family
  convention rather than from a reference. The converter skips the tensors today.
- [ ] **`getRequiredMemory` is unimplemented on nine components, and the base throws by design.**
  Gelu, MultiHeadAttention, Lpe, GatedMLP, MLP, SoftmaxCrossEntropy, LayerNorm, Softmax and
  GptBlock — so `GptModel::getRequiredMemory` throws the way Qwen's did until `Activation` was
  converted. The contract lands family by family (`Core/Component.ixx:615`); GPT-2 is the family
  still outstanding.
- [ ] **The Llama converter writes a metadata key the reader never parses.** It emits `norm_eps`;
  `parseMetadataJSON` extracts `norm_epsilon` (Gemma and the packer both emit that). Harmless today
  only because `LlamaModel::configFromMetadata` does not read the epsilon at all — it takes
  `LlamaConfig`'s default. `Tools/Converters/Llama/convert_weights.py:188`.

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
  `QwenModel.Load.Cuda.cpp:189` calls `model->generate(...)` for its side effects inside a
  lambda, producing C4834. The status is the only channel that reports why generation stopped,
  so a test that ignores it cannot tell a completed run from an aborted one. Assert it instead
  of casting it away. Note this is a third warning in the tree, in `Mila/Tests` rather than
  `Mila/Src`, where the two Production Hardening warning items live.
- [ ] **Backward-path kernels disabled or unverified.** `CudaSoftmaxOp.ixx:73` and `:103` throw
  `"needs review"` with the real calls commented out; `Gelu.Fp32.cu:65` records that the shipped
  backward is not the numerically stable `sech^2` form. Gradient-check these before the suite claims
  backward coverage — and sweep the *unmarked* backward kernels too, per-precision twin by twin: the
  RoPE FP32 backward was arithmetically wrong while its BF16 sibling was correct, in a file carrying
  no marker at all.
- [ ] **[crash] Bring the GPT-2 CPU path up to current standards — its own session.** The CPU
  operation layer treats build-time extents as runtime extents, so CPU inference has never run a
  prompt shorter than its built context, which is every real generation. Components pass a max-sized
  buffer and view the result down (`Lpe.ixx:154`, `LayerNorm.ixx:132-146`); the ops loop to what
  `build()` cached. Remaining after the encoder and LayerNorm fixes: `CpuLinearOp:259,264`,
  `CpuSoftmaxOp`, `CpuSoftmaxCrossEntropyOp`, and `CpuAttentionOp` — the last is not mechanical, since
  `B_`/`T_` size its `{B,NH,T,T}` score buffer at `:269` as well as every loop. `CudaLpeOp:192-196` is
  the reference pattern, and note that `CpuEncoderOp::build` calls its own shape validator before
  setting those members, so a bound cannot live in the shared one. The CPU reference code is why this
  matters; GPT-2 itself ranks below Llama and Gemma.
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
- [~] **[net-new]** Training-loop integration test (sample-independent) — the MNIST spine is covered
  by `Network.Cpu.cpp`; remaining is a GPT-2-stack analogue for the Bard spine. Bard itself runs;
  this is the test that would catch a regression in it.
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
  between publishes at all. Seventy-five errors accumulated unseen from 2026-07-27 to 08-12 and then
  blocked the site. Add a non-deploying Doxygen check to `build-pipeline.yml` (no CUDA, no CMake).
- [ ] **The CPM gate silently validates the PREVIOUS release, and reports success.**
  `MILA_CPM_GIT_TAG` is set with `CACHE STRING`, which never overwrites an existing entry, so a build
  directory reused across releases keeps the tag it was first configured with. At beta.2 the gate ran in
  a July build dir, tested `v0.20.0-beta.1`, and passed in 134s off a warm cache — proving nothing about
  the release it was run for. The failure is silent in both directions: nothing prints the tag under
  test, and a cached pass looks identical to a real one. Fix: derive the tag with `FORCE`, or key the
  gate's work directory on the tag, or `message(STATUS ...)` the tag at configure and print it in the
  test output so a stale run is visible. RELEASING's "stale CPM cache" note only mentions deleting
  `cpm-cache` for *misbehaviour* — it does not warn that a passing run may be the wrong tag; step 6
  should say to pass `-DMILA_CPM_GIT_TAG=` explicitly, not rely on the default.
  `Mila/Tests/Packaging/CMakeLists.txt:81`
- [ ] **Wheel validation is ordered backwards: move it before the PR, onto `dev`.** RELEASING puts
  publishing at step 7, after the tag, justified by the wheel version needing `Version.txt` stripped of
  `+build` — but that happens at **step 1, the release-prep commit on `dev`**, so from that commit
  onward `dev` already produces the release wheels. The tag contributes nothing to a wheel (verified at
  beta.2: the merge commit's tree is byte-identical to `dev`'s head tree). Only the **CPM gate** is
  genuinely post-tag, because it clones from GitHub at the tag. New order: 1, then build + TestPyPI +
  clean room, then PR, merge, tag, CPM gate, PyPI. At beta.2 we tagged, published a Release and a
  Discussion, and only then found the wheels could not be validated at all — under this order that
  lands while nothing is immutable.
- [ ] **Rule to add with it: TestPyPI takes only `.devN` snapshots; the plain release version goes only
  to PyPI.** A filename is burned permanently on first upload, so validating at `0.20.0b2` leaves no
  second attempt if a fix is needed — and a stray `0.20.0b2` upload on 2026-07-28 poisoned that
  release's `Requires-Python` at `>=3.13` (PyPI fixes it at the release level from the first file and
  never updates it), which broke both 3.12 legs of the beta.2 clean room three weeks later and could
  not be repaired: delete does not free filenames, and yank changes neither metadata nor an exact pin.
  The rule falls out for free — the last `dev` commit before the prep commit already produces a unique
  `0.20.0b2.devN`, the same binaries under a disposable version. `RELEASING.md:247`
- [ ] **RELEASING step 8 recommends `--generate-notes`, which cannot work here.** GitHub builds those
  notes from the pull requests merged between two tags; Mila lands all work as direct commits on `dev`
  and opens exactly one PR per release, so beta.2 would have produced a one-line release for 48 commits
  of work. The substance lives only in the commit messages. State instead that the body is authored from
  the commit range and passed with `--notes-file`, matching what beta.1 actually did (its release body
  is hand-written; only the trailing `Full Changelog` footer is generated). `--prerelease` stays — it is
  load-bearing for the Latest-release badge. Same source feeds both destinations: the commit range
  produces one authored summary, which becomes the GitHub Release body at any tag and distils into the
  CHANGELOG entry at a production release. `RELEASING.md:229`

### Production Hardening

- [ ] **Phase 5's perplexity gate PASSES — ratio 1.139 at 16K against a pre-registered 1.25.**
  `DISABLED_QualityGateAcrossContextLengths`, both arms at 4096/8192/16384 over the same ~31,650
  positions of wikitext-2 test, one card pinned by UUID. The ratio is flat across 1024-16384
  (1.09-1.14), so the gap does not widen with context. Full table and the two caveats in
  `Qwen3.8.md` section 8 item 9. **Watch item:** from 8K to 16K the oracle improves 7.2% and the
  plan only 3.4% — the quantized arm captures about half the benefit of extra context, which is the
  compounding signature the recurrent layers make plausible. Re-run before any 32K claim.
- [ ] **A value-reading observation sink has to name the model's compute precision.** The sink gets
  `const ITensor&`, whose `rawData()` is type-erased, so anything wanting numbers rather than shapes
  does a `dynamic_cast` to `Tensor<TPrecision, MR>` then its own `toHost`. Both consumers now do
  this. It works; whether observation should offer a typed convenience is open, and `Observability.md`
  11.2 parks it for v0.21.
- [ ] **All three Phase 5 quality criteria are measured and consistent; the prompt set is now the
  weak link.** Six prompts is enough to show trajectory cost and fork index are decoupled, not
  enough to put a threshold on the mean (0.1129 nats/token) the way 1.25 sits on the perplexity
  ratio. Widening it is cheap — two loads and ~18 s per six prompts — and the classes that
  matter for this model's use are underrepresented: one code prompt, no tool call, no multi-turn,
  no long-context prompt. `DISABLED_DivergenceAgainstTheOracle`.
- [ ] **CUDA's device 0 is not nvidia-smi's, and the default `DeviceId{ Cuda, 0 }` picks the 12 GiB
  card.** nvidia-smi reports the 5060 Ti at index 0; CUDA orders the 4070 first, so any load sized
  for the 16 GiB card aborts by default — the FP4 oracle's 12.31 GiB did, with no diagnostic, in
  about two seconds. Every measurement that must land on a specific card has to pin
  `CUDA_VISIBLE_DEVICES` to its UUID. Worth a note wherever `fromPretrained`'s default device is
  documented, since the failure looks like a model defect rather than a device choice.
- [ ] **The head's two paths do not agree to the last digit, so a perplexity comparison must fix the
  width.** Same artifact and corpus: width 1 (decode matvec) gives 7.513, width 64 (W4A8-FP8 GEMM)
  gives 7.515. Small, but head width is part of the measurement protocol rather than a free
  performance knob — both arms of a quantization comparison must use the same one.
- [ ] **Scoring speed is a PREFILL problem, not a reduction problem — measured, so do not build the
  device-side reduction.** `DISABLED_ScoringCostBreakdown` on the packed 27B: model forward 23.2 s,
  scoring 34.2 s, so the forward is **68%** and everything scoring adds is 32%. A perfect device-side
  reduction is therefore capped at **1.45x** by Amdahl, permanently. The transfer is not a factor
  either (3.7 GB, ~0.3 s) — the host `exp` loop is essentially the whole 11 s overhead, matching its
  9.2 s prediction. **If scoring speed is ever wanted, parallelise the host loop across cores** —
  the rows are independent, it is a few lines, and it captures most of the same 11 s with no kernel,
  no dead-code revival and no numerics risk. The real lever is the chunked UT-transform prefill
  kernel already filed below: it owns the 68%.
- [ ] **Only Qwen can be scored.** Gemma, Llama and GPT-2 build their heads at T=1 with no width
  parameter (`Gemma.ixx:279`, `Llama.ixx:239`, `GptTransformer.ixx:365`), so `scoreTokens` throws the
  base's `logic_error`. The gate wants Gemma and Llama too, and each needs the same two pieces: a head
  width in its config, and the window loop.
- [ ] **`SoftmaxCrossEntropy` has never been compiled, and its CUDA dispatch promises a BF16 it
  cannot deliver.** Its only test is commented out of the build (`Mila/Tests/CMakeLists.txt:276`) and
  is written against a 3-parameter component that now takes 2, so nothing instantiates the template.
  `OperationTraits.Cuda.ixx:422` maps `<CrossEntropyOp, Cuda, BF16>` to a type whose
  `cuda_softmax_crossentropy_impl<__nv_bfloat16>` has no specialization — a hard compile error the
  moment anything uses it — while the `half` specialization is a silent empty stub
  (`CudaSoftmaxCrossEntropyOp.ixx:99`). Two more things to check when reviving: the block kernel
  (vocab >= 1024) has never run, and its final `__shfl_sync` broadcasts each warp's own lane 0 while
  only warp 0 holds the reduction, which on inspection gives warps 1+ `-INFINITY` for the max; and
  the op writes losses in the logits' dtype, which is the wrong precision for accumulating them.
- [ ] **`Qwen3.8.md` section 8 gates the 16 GiB oracle on token-for-token cross-arch agreement, which
  cannot pass at any precision.** Measured on Llama 3.2 3B greedy, Ada vs Blackwell: BF16, FP8 and FP4
  all fork, at a token index set by the prompt rather than the precision. Each card is deterministic
  run-to-run — this is FP non-associativity between two architectures, not a defect. Restate the gate
  as teacher-forced (perplexity never samples, so it never hits this).
- [ ] **No Llama parity test exists** — only Gemma and Qwen have one, and Qwen's needs the 27B
  artifacts. So the cheapest model that fits both cards cannot be checked against an FP32 reference,
  which is what would prove neither architecture sits systematically further from ground truth.
  `Mila/Tests/Dnn/Models/` has the two existing parity tests to copy.
- [ ] **Every FP4/FP8 perf number in the tree is Ada-at-x16, and the 4070 now sits on a chipset Gen4
  x4 link.** No baseline was captured before the move, so the published figures carry an unmeasured
  link change. Re-measure before any of them is quoted again.
- [ ] **`PerGroupFp4` carries FP32 scales, and the offline tooling simulates FP16 ones.**
  `Policies.ixx:112` sets `kScaleDtype = FP32`; `formats.py`'s `fake_fp4_e2m1` rounds the
  scale through FP16, because Qwen3.8.md section 5 budgets FP4 at 4.125 bits — 4 + 16/128 —
  and prices the whole allocation on it. So the two disagree by 0.125 bits per weight and,
  more importantly, the packer's simulated FP4 damage is not quite the damage Mila's load-time
  quantizer inflicts. Moving the runtime to FP16 scales is the fix Section 5 already names,
  and it is a free win for Gemma and Llama independently of Qwen. Until then no FP4 number
  from the offline tools is exactly a Mila number.
- [ ] **The FP4 and FP8 wire formats are defined only by a CUDA kernel.** `cuda_quantize_fp4_per_group`
  is the sole place the nibble order and the `/6.0f` scale convention are written down, so the byte
  layout of every published artifact is unreadable from CPU-only CI and unstatable in the spec.
  `CodebookPacking.ixx` is the shape to copy — normative layout plus a host codec, with a generated
  fixture holding the kernel and the Python packer to it in both directions. Add `Fp4Packing.ixx` and
  `Fp8Packing.ixx` beside it. Hardening a shipped format, not a new capability; see `Quantization.md`,
  *Fitting is offline, encoding is a codec*.
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
- [ ] **`GroupedQueryAttention.ixx:216` C4702 is left deliberately** — one of the two warnings in the
  tree. It self-clears when the GQA training path is built, where a suppression would have to be
  remembered.
  **Note for the warnings-as-errors decision:** a blanket `/WX` would force it silent; escalating only
  the defect-class codes leaves it visible.
- [ ] **`Chat.Footprint.ixx:23` defines a variable in the global module fragment** — C5202, the
  tree's second warning. The comment says the preprocessor test is "confined to the fragment so the
  module body stays free of it", but a GMF admits only preprocessor directives and the two
  `inline constexpr bool` definitions are not that. The `#ifdef` cannot simply move to the body
  either — see [[feedback_no_ifdef_in_modules]]; a CMake-supplied definition is the shape that fits.
- [ ] **`IExecutionContext` is exported but unreachable in practice.** `Mila.ixx` re-exports it and
  `ExecutionContextFactory` as public API, but no model factory accepts one (`GemmaModel.ixx:119`
  takes a `DeviceId`) and `Component` holds a non-owning pointer owned by its parent. Decide: a
  `fromPretrained` overload taking `IExecutionContext*`, or drop both from the umbrella.
- [ ] **If C1128 recurs** on `MilaTests`, `ProfileModel` or `ExportArtifact`, switch from the per-target
  `/bigobj` on `ChatApp` to one project-wide `add_compile_options`. **Todd's call** — it touches every
  target's flags, so it was deliberately not taken unilaterally.
- [ ] **The README's six CI badges are decorative fiction.** `README.md:18-21` builds a Branch x
  (Build/Test/Docs) table by passing `job=build`/`test`/`docs` to the badge endpoint, which has no
  such parameter — all six fetch identically (verified: `job=build` and `job=test` both return
  `Mila CI - passing`). `build-pipeline.yml`'s real jobs are `compile-and-gate` and `cpu-only-tests`,
  and it has no docs job at all — docs are `publish-site.yml`. Two honest badges beat six that cannot fail
  independently.
- [ ] **Three different GCC floors are stated in the tree, and only one is measured.** `README.md:250`
  says GCC 16, `README.md:267` says "GCC 15.2 and earlier cannot" (implying 15.3 works), and
  `CLAUDE.md` says GCC 15.3+. What was actually validated is 16 works / 15.2 fails; 15.3 has never
  been built. State the measured floor in one place and stop implying the untested one.
- [ ] **Both onboarding docs state the build-option defaults backwards.** `MILA_ENABLE_TESTING` is
  `${PROJECT_IS_TOP_LEVEL}` and `MILA_ENABLE_DOCS` is `ON` (`CMakeLists.txt:67,77`), but
  `README.md:282,296` and `getting-started.md:79,212` call both `OFF` — so "omit the flag for a
  library-only build" changes nothing on a fresh clone. `getting-started.md:221` compounds it with
  `find_package(Doxygen REQUIRED)`; the call is unqualified and `Mila/Docs/CMakeLists.txt:12` explains
  at length why it must stay that way. Graphviz is sold for call graphs the Doxyfile disables.
- [ ] **A FetchContent consumer inherits Mila's `docs` target, aimed at the consumer's own source
  tree.** `MILA_ENABLE_DOCS` defaults `ON` and is not gated on `PROJECT_IS_TOP_LEVEL`, so a
  downstream configure prints "Doxygen documentation target 'docs' created" and offers a target
  whose `WORKING_DIRECTORY` and output path are `${CMAKE_SOURCE_DIR}`
  (`Mila/Docs/CMakeLists.txt:21-24`) — in a subproject build that is the *consumer's* root, so
  building it would write `<their-repo>/build/docs` and run Mila's Doxyfile against their tree.
  Observed on a real standalone consumer configure 2026-08-13, not inferred. Same
  `CMAKE_SOURCE_DIR`-in-a-subproject class as the `tokenize` item; `MILA_ENABLE_SAMPLES` gets this
  right and is the pattern to copy.
- [ ] **The preset list names four presets that do not exist and omits nine that do.**
  `getting-started.md:207` offers `x86-debug`, `x86-release`, `linux-debug`, `macos-debug` — none are
  in `CMakePresets.json`, and the real Linux entries are `linux-clang-{debug,release}` and
  `linux-clang-cpu-{debug,release}`. `CLAUDE.md` repeats the `x86-*` pair. A reader following either
  file gets "no such preset" on their first configure.
- [ ] **`getting-started.md:229` pins the dev container a release behind** — CUDA 13.0 / Clang 19 /
  CMake 4.x against the actual CUDA 13.3 / clang-21 / gcc-15 / CMake 4.2.3
  (`Docker/Dockerfile:18,48,50,61`). `README.md:321` and `Docker/README.md:13` both have it right, so
  this is the one file out of step.
- [ ] **`CLAUDE.md` documents the retired Chat alias set** — "Model aliases: `gpt2`, `llama-1b`,
  `llama-3b`, `llama-8b`" plus the `llama31`/`llama32` filename-prefix rule. There is no catalogue and
  no filename construction; `/model` takes a store name. It is agent-facing rather than user-facing,
  which is why it rotted unnoticed, but it actively misdirects work.
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
- [~] **Clean-room the four wheels.** All four are built at one version, each carrying only its own
  extension and no vendored CUDA, and the Windows cp312 wheel installs into a clean 3.12 venv and
  generates. What no developer machine can answer is whether they load with no CUDA Toolkit present —
  that is the `Wheel clean room` workflow, now a 4-leg matrix, and it has never run on any wheel.
  Dispatching it needs `wheel-cleanroom.yml` on the default branch.
- [ ] **Add 3.14 once 3.12 is proven.** It is the interpreter Ubuntu 26.04 ships, so it is what the
  dev container's `python3` is — which is why `Docker/build-mis.sh` still restates MIS's dependency
  list and installs MIS itself with `--ignore-requires-python`. Widening to 3.12+3.13 did not retire
  that duplication; only a 3.14 wheel will. Needs `uv python install 3.14` on Windows and one
  deadsnakes line in `Dockerfile.wheel`.
- [ ] **`Mila/Tools` has no off switch** — gated on `PROJECT_IS_TOP_LEVEL` alone
  (`Mila/CMakeLists.txt:962`), so the wheel configure builds `tokenize` and `ExportArtifact`, neither
  of which can go in a wheel. Every other subdirectory has a `MILA_ENABLE_*`; this one costs build time
  on an artifact that discards it.
- [ ] **`tokenize` writes its executable into the consumer's source tree.**
  `Tools/Tokenize/CMakeLists.txt:13` sets `RUNTIME_OUTPUT_DIRECTORY` under `${CMAKE_SOURCE_DIR}`,
  which in a subproject build is the *consumer's* root, not Mila's. `PROJECT_SOURCE_DIR` is the fix,
  but the tool is run from `Data/Tools/<CONFIG>`, so moving it is a workflow change, not a rename.
  Masked today by the `PROJECT_IS_TOP_LEVEL` gate above — it becomes live the moment Tools gets its
  `MILA_ENABLE_*` switch. `CMAKE_SOURCE_DIR` vs `PROJECT_SOURCE_DIR` is the exact class
  `packaging_fetchcontent_consumer` exists to catch, and it is getting past it.
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
- [ ] **Guided reading path** — one token's journey (embed -> attend -> sample -> decode) through the
  real source, readable by a strong C++ dev unaided.
- [ ] **The wheel VERSION file is still written into the source tree from any build.**
  `Mila/Bindings/CMakeLists.txt:65` runs `file(WRITE ...Package/VERSION)` at configure time,
  unguarded, so a FetchContent consumer writes into whatever tree Mila was fetched from — the same
  class as the POST_BUILD staging now behind `PROJECT_IS_TOP_LEVEL`. Harmless today (same content,
  gitignored) and left alone mid-release, but the two belong under one guard.
- [ ] **CI jobs have no `timeout-minutes`, so a hang costs six hours.** Both jobs at `beta.2+47` hit
  GitHub's 6h ceiling, the CPU one stalled in `Run CPU test suite` after a clean configure and build.
  **It HAS now recurred, twice in one day, in two different jobs** — so it is runner-level, not one
  flaky step. (a) PR run 31512756287 job 93850377071 stalled in `Run CPU test suite`, 75+ min against a
  14m29s baseline; cancelled and re-run, then passed in 15m6s. (b) Push run 31512752392 job 93850364582
  stalled in `Build` on target `[993/1135] Mila_py.Wrappers.cpp.o` — the pybind11 wrapper TU, which
  compiled in **3m43s** on the identical tree in the parallel PR run — for over an hour. A re-run is the
  only remedy available, and without a bound each stall costs six hours of runner time.
  Against a normal ~45-minute round trip, a bound near 60 turns a repeat into
  a legible failure instead of a day of runner time. `.github/workflows/build-pipeline.yml`
- [ ] **Split the packaging gate into its own job that configures but does not build.** VERIFIED: the
  gate does not consume the parent build at all — `add_test` passes `MILA_SOURCE_DIR=${CMAKE_SOURCE_DIR}`
  (the source tree) and `drive_fetchcontent.cmake` configures and builds a consumer in its own
  `WORK_DIR`, compiling Mila from scratch under `_deps/mila-build/`. It needs only that CMake has
  *configured*, so the test is registered. Today the release PR built Mila for ~45 min and then the gate
  built it again: 45 + 20 in series. As two parallel jobs the critical path is the compile alone, with
  the gate finishing inside it — same coverage, same total compute, ~20 min off every release PR, and no
  assumption that the merge tree equals the head tree. `Mila/Tests/Packaging/CMakeLists.txt:43`
- [ ] **A `dev` push and an open PR for the same SHA run the whole pipeline twice, and the redundant
  one blocks the merge.** The PR run is a strict superset — same tree (merge and head tree hashes
  verified identical), plus the packaging gates that `build-pipeline.yml:114` skips on a `dev` push. Both
  report the same check names on the same SHA, so the pending push run gates the merge while adding no
  information. Suppress the push run when a PR for that SHA is open. Write it carefully and comment
  exactly when each job runs: an `if:` in this same file hid a broken packaging gate for 32 commits, and
  a first pass at this idea proposed suppressing the PR run — the wrong one, since it is the only run
  that gates packaging.
- [ ] **Two orphaned brand assets still carry the old Achilles mark.** `icon.png` at the repo root
  (tracked, dated 2025-01-09) and `Web/static/achilles.png` (retired from the templates at
  `beta.2+10`, file left behind). Neither is referenced by any page, template, README or the
  Doxyfile, which sets no `PROJECT_LOGO` at all. Delete rather than replace: `Brand/generate.py`
  emits the current mark into `Web/static/` only, so a root copy would be a second source to drift.
- [ ] **`actions/setup-python@v5` still declares Node 20, which GitHub has deprecated** — it warns on
  every clean-room run (the runner already substitutes Node 24, so this is a notice, not a break).
  Every other action in the tree is already on `@v5` and clean; this is the only straggler. Bump it and
  re-check the rest at the same time, since the deprecation applies by action version, not by workflow.
  `.github/workflows/wheel-cleanroom.yml`
- [ ] Add the Samples build to CI (only tests build today).
- [~] **Publish the Docker runtime image.** The image itself is now PROVEN, 2026-08-13: the committed
  `Dockerfile.runtime` builds (first time ever), and all three entrypoint verbs were verified in a
  container — `install gpt2-small` pulled 624 MB into a fresh volume; `chat` started and listed that
  store; `serve` bound 6452 and answered a real `/v1/chat/completions` request with
  `Llama-3.2-3B-Instruct-fp4` loaded from a read-only mount of the host store (`finish_reason: stop`).
  The never-run `ENV MILA_PORT=6452` fix is correct — `config.py` sets `env_prefix="MILA_"`, and no
  `.env` ships in the image, so `settings.port` resolves to 6452. What remains is the publish itself.
- [ ] **ONE image holding all of Mila, with two entry points** — not one per adaptor. Chat and MIS are
  two interfaces onto the same runtime: same `libMila`, same binding, same store on the same mount, so
  splitting them would duplicate the library and make the user choose an artifact before they know which
  interface they want. Chat is the default `CMD`; MIS is a second entry point (`... mila-llm serve`,
  publishing its port). `build-all.sh` already builds exactly this set and its own comment calls the
  binding "part of the full product" — the chat/mis/all scripts are three CMake configurations of one
  image, not three images. Accepted cost: a Chat-only user carries Python, the binding and FastAPI,
  which is a couple of hundred MB against a multi-GB CUDA base.
- [ ] **The real split is devel vs runtime, and only the runtime half is published.** Stage 1 is
  today's `nvidia/cuda:*-devel` base building everything; stage 2 is a `*-runtime` base carrying only
  the built binaries, the binding, MIS and the store tooling — no compilers, toolkit or headers. That is
  where the size saving is, far more than any adaptor split. `Docker/Dockerfile` stays as it is: the
  contributor build environment, built from the repo and never published.
- [ ] **Publish the image to Docker Hub**, the one container channel. The image tag drops the `+build`
  metadata, which OCI forbids — already noted in RELEASING's versioning section. Name decided:
  **`toddthomson/mila-llm`** — a Docker
  namespace admits only lowercase letters and digits, so `mila-llm` can only be the repository half,
  which is where the name shared with HF and PyPI lands.
- [ ] **Every container build path defaults to an arch a published image cannot use.**
  `Docker/build-chat.sh:25` defaults `MILA_CUDA_ARCH=native` and passes it to both
  `CMAKE_CUDA_ARCHITECTURES` and `MILA_LIBRARY_CUDA_ARCHITECTURES`, so the image carries kernels only
  for the GPU that built it — and `native` does not resolve on the GPU-less builder a publish job runs
  on. The publish pipeline must set the portable list explicitly; the failure it otherwise ships is
  already in `Docker/README.md`'s troubleshooting section.
- [ ] **The published wheel still stops before Blackwell.** The library default now carries `120`
  (`Mila/CMakeLists.txt:24`) and the runtime image always did, but the `x64-wheel` and `linux-wheel`
  presets pin their own `75;80;86;89;90` (`CMakePresets.json:183-184`, `:214-215`), so a `mila-llm`
  install on an RTX 50-series card JITs from sm_90 PTX at first launch. Adding `120` costs one more
  full CUDA compile per wheel and a larger artifact; the alternative is saying so on the PyPI page.
- [ ] **`Docker/build-mis.sh:76` looks broken on the current image, by the defect that just failed
  the runtime build.** It runs `pip install --no-deps -e Mila/Bindings/Package` under the container's
  Python 3.14, and `mila-llm`'s `requires-python` is `>=3.12,<3.14`; `--no-deps` does not suppress
  that check. The script's own comment shows the ceiling was handled for the server DEPS and missed
  for the package itself. Verify in a container, then add `--ignore-requires-python` as the runtime
  image now does.
- [~] **Rework Chat configuration to layered resolution** — design and phasing in
  `Mila/Specifications/ChatConfiguration.md`. Phases 1-5 have landed (image directory, validation and
  the exit path, the flag set, the family table and the layered merge with origins, the predictor's
  failure reason, `context_length: "auto"`, prompt resolution and the end of `Data/`, the config
  root). Remaining: phase 7, the two `ModelRecord` fields, which touches Model Distribution.
- [ ] **`Chat.Json` is a byte-for-byte duplicate of the `nlohmann.json` module** —
  `Mila/Adaptors/Chat/Src/Json.ixx` versus `Mila/Src/Utils/json.ixx`, both including the same header
  from their global module fragment. Chat imports one in `Chat.ixx` and the other in
  `Chat.ModelCatalog.ixx`. Drop `Json.ixx` from the target and import `nlohmann.json` everywhere.
- [ ] **`mila serve <args>` is broken on Windows and cannot report the server's exit code.**
  `runProgram` (`Cli.ixx:100`) hands a concatenated string to `std::system`, so cmd.exe strips the
  outer quotes of the whole command line — MEASURED 2026-08-15: `mila chat --help` died with
  `'D:\...\mila-chat.exe" "--help' is not recognized`. No argument survives, and the code returned is
  the shell's. Launch with an argument vector (`CreateProcessW` / `posix_spawn`) behind a
  CMake-selected module partition, since module code carries no `#ifdef`.
- [ ] **The devel image's `mila-chat` wrapper now shares its name with the binary it wraps**, and its
  `cd /build` is redundant since `executable_directory()` reads `/proc/self/exe`. `Docker/Dockerfile:94`
  installs `run-chat.sh` as `/usr/local/bin/mila-chat`; the built binary is `/build/mila-chat`. Verify in
  a container, then either drop the wrapper for a symlink or keep it purely for the not-built message.
- [ ] **Drop the redundant `cd` from `Docker/run-chat.sh:24`.** CONFIRMED redundant 2026-08-15: the
  runtime image ran `chat` correctly from `-w /tmp` and `-w /`, resolving both its config and a
  `--system-prompt` by name, so `/proc/self/exe` carries it. The wrapper is devel-only, so the
  confirmation is by the shared binary rather than a devel container run. `Mila/Tools/Cli` needs no
  change at all — it never changed directory; `main.cpp:35` reads `/proc/self/exe` and
  `current_path()` at `Cli.ixx:51` is an unreachable last-resort fallback on both shipped platforms.
- [ ] **Library log output collides with Chat's spinner, and the first thing an evaluator reads is a
  corrupted line.** Measured 2026-08-16 on the website's Evaluating path: the spinner is mid-line when
  the BPE warning fires, so the console shows `Loading Llama-3.2-3B-Instruct-fp415:49:42.425 [WARN ]
  BpeTokenizer.ixx:378:operator: ...`. `main.cpp:927` installs a stock `ConsoleSink` that knows nothing
  about the renderer. `ConsoleSink` derives from `Logger` and `Mila::initialize()` takes a sink, so
  Chat can supply a spinner-aware one that erases the line, writes, and lets the spinner redraw --
  adaptor-side, no library change. The same sink should drop `file:line:function` from user-facing
  output; that is developer detail in a product surface.
- [ ] **The BPE ASCII-fallback warning fires for every Llama and GPT-2 session, including the ones it
  does not apply to.** `BpeTokenizer.ixx:378` warns at construction under `std::call_once`, so an
  evaluator typing English is told non-ASCII "WILL tokenize differently from the HuggingFace
  reference" about a path they never take -- as the first thing on screen after the welcome box. The
  claim is true and should not be softened; it should be *timely*. Warn on first non-ASCII input
  instead of at construction, so it fires exactly when it is true. Touches `Mila/Src` and needs
  agreement. Cheaper interim: one line on the console, detail in the docs.
- [ ] **The Evaluating band's commands leave a stopped container behind on every run.** No `--rm`, so
  QA on 2026-08-16 accumulated four in an afternoon and `docker image rm` then failed with a
  conflict the user has no context for. Nothing is lost by adding it: the model lives in the named
  volume, which is why the volume exists. `Web/layouts/index.html`, the `#evaluate` steps.
  **The devel tab must NOT get `--rm`** -- that image is a configured environment where the reader
  edits `~/myapp`, and removing the container discards their work. That panel has the opposite gap:
  nothing tells them how to re-enter the container they already have.
- [ ] **Is the token-for-token parity claim tested with non-ASCII input?** Established 2026-08-16:
  the BPE ASCII fallback is active on EVERY platform, including the published Linux container under
  clang-21 -- `\p{L}`/`\p{N}` are ECMAScript 2018 and absent from the grammar C++ adopted, so no
  standard `std::regex` compiles the Unicode pattern. Llama 3 and GPT-2 therefore pre-tokenize
  non-ASCII differently from the HuggingFace reference *in the shipping image*, not merely on a
  Windows dev box. If the parity fixtures are English-only, the divergence is untested rather than
  bounded, and the site states parity without qualification. Establish which it is, then either
  qualify the claim or fix the pre-tokenizer (a hand-written scanner, not `std::regex`).
- [ ] **The download bar restarts per file and never says which file.** A model is a manifest, a
  tokenizer and the weights, so the user watches 0-100% twice with nothing distinguishing the runs --
  it reads as a restart. `ProgressCallback` is `(received, total)` only (`HttpClient.ixx:63`), so the
  CLI cannot label what it is drawing. Adding the file name, or a (index, count) pair, is a library
  signature change and needs agreement. The sub-megabyte manifest is already suppressed in
  `Mila/Tools/Cli/Cli.ixx`, which removes the worst of it.
- [ ] **A model name that does not exist is reported as an authentication failure.** Measured
  2026-08-16: `install NoSuchModel-xyz` exits 1 with "no valid HuggingFace token. Set HF_TOKEN, or
  run 'huggingface-cli login'." Every published Mila model is ungated and needs no token -- the site
  says so three times -- so this sends a user who merely typo'd a name to set up authentication they
  will never need. The store request 404s (or 401s, since HF hides existence on private repos) and
  the token branch is being taken for both. Reached from the C++ CLI, so it affects the devel image
  too; the runtime image's `install` now shares it. Distinguish "no such model" from "not
  authorised" before the first publish -- a typo is the likeliest failure on the evaluation path.
- [ ] **A non-interactive `chat` must name its model, and only the interactive surfaces are exempt.**
  Settled 2026-08-16: naming it is correct, not a workaround. Chat switches models at runtime, so
  inferring one from a store that happens to hold exactly one would change the command's meaning
  once a second is installed — and carrying the choice across two processes would persist it in
  `chat-state.json`, which `resolveStoreRoot()` puts in a *cache* directory, so the quick start
  would work then fail after an eviction. The site copy is fixed. What remains is the sweep: every
  surface that shows a scripted `install` followed by `chat` names the model, and bare `chat`
  keeps failing well ("No model is loaded. Name one with --model") rather than being made to guess.
  `Dockerfile.runtime:249`'s `CMD ["chat"]` is fine — that path is interactive.
- [~] **The runtime image ships a binding that cannot import, and the gate says it is fine.** Two
  defects, measured 2026-08-14 on a clean `--target runtime` build: `site-packages/mila/` holds only
  `__init__.py`, so `install` and `serve` both die on `ImportError: No module named 'mila._mila'`. The
  extension reaches the image only as a `POST_BUILD` side-effect into the source tree, which a cache-warm
  compile never re-runs, and `.dockerignore` (correctly) keeps `_mila*.so` out of the context. Install
  from `/build/python/mila`, where the build actually writes it, rather than from the copied source tree.
- [ ] **The `ldd` gate passes when the file it checks is absent.** `Dockerfile.runtime`'s runtime stage
  greps for `"not found"`, but an unmatched glob makes the shell hand `ldd` a literal pattern and it
  answers `"No such file or directory"` — so the gate printed "Shared library check passed" over a
  missing extension. A gate that reports success on an absent artifact is worse than no gate. Assert the
  file exists first, then check its NEEDED entries.
- [ ] **The binding's staged extensions accumulate in the source tree and nothing prunes them.**
  MilaPy's POST_BUILD writes `_mila*.so`/`_mila*.pyd` into `Mila/Bindings/Package/src/mila/`, so a
  checkout collects one per interpreter and platform ever built — four here, two of them Windows
  `.pyd` — all untracked, and all swept into a Docker build context until `.dockerignore` excluded
  them. Either clean stale ones on build or stage outside the source tree.
- [ ] **`Docker/README.md:69` credits ChatApp with a compiled-in `MODELS_DIR`.** It has none —
  the only `MODELS_DIR` in the tree is `Mila/Profiling/ProfileModel/CMakeLists.txt:22`. Chat resolves
  models through `MILA_CACHE_DIR` and its config through the working directory, which is why the
  published image can drop the bind mount at all. The claim reads as a hard dependency on `/mila`.
- [ ] **The devel tab's "roughly 12 GB" is an unmeasured estimate in the wrong unit.** The runtime
  figures are now measured -- 1.54 GB compressed across 25 layers from the registry manifest, 4.11 GB
  unpacked -- and the Evaluating band states both. Devel measured 7.34 GB compressed / 21.6 GB
  unpacked at single-arch 89, but the published image is `89;90;120` and both grow by an unknown
  amount, so there is no honest replacement number until a publish build exists. Take both figures
  from that build (`docker manifest inspect` for the download, `docker images` for the disk) and
  state them the same way. `Web/layouts/index.html`, the `#p-docker` cost cell.
- [ ] **A publish build of the runtime image has never been made.** Verification used a single-arch
  (`89`) build; a published image must be `89;90;120` and needs `MILA_CLEAN_BUILD=1`, since
  `--no-cache` leaves BuildKit cache mounts intact. **This is not theoretical — measured 2026-08-14:** a
  changed Chat config did not reach a rebuilt image because the copied tree in the cache mount was
  never re-copied, so the image silently shipped the previous tree's config and behaved to match. Stale
  cache-mount content has now produced two wrong images in one day (this and the missing binding
  extension), both silently. Open decisions before any push: whether a
  pre-release gets `latest` (a bare `docker run toddthomson/mila-llm` resolves to it), and the
  Docker Hub Overview page needing a source in the repo rather than browser edits.
- [ ] **Decide the container tag scheme, including whether a pre-release gets `latest`.** RELEASING
  covers dropping `+build` (OCI forbids `+`) and nothing else. `latest` is what a bare
  `docker run toddthomson/mila-llm` resolves to, so pointing it at a beta makes the beta the default
  for everyone who does not read the tag list.
- [ ] **Docker Hub Overview page is an authored surface, so give it a source in the repo.** It is what
  search shows and it carries the container-distribution message; hand-editing it in the browser is how
  the HF org card came to need a rewrite. See [[project_four_channel_roles]] — five channels, five jobs.
- [ ] Broaden CI compiler coverage toward the supported matrix (adds MSVC + GCC 16 to clang-21).
- [ ] Stage model weights off the Windows bind mount for the container (native disk speed).
- [ ] **[contributor]** Llama-lineage CPU ops (`RmsNormOp`, `SwigluOp`, `RopeOp`, `TokenEmbeddingOp`,
  `CrossEntropyOp`) in `OperationTraits.Cpu.ixx` — demand-driven; absence is zero-cost on the GPU path.
- [ ] **[deferred, measure first]** Remove FP16 (superseded by BF16) — woven through live code; trace
  live-vs-dead first, and 8 `REVIEW:` markers already scope it. Note the odd row it collides with:
  **CUDA `LayerNormOp` is registered at FP32 and FP16 and *not* BF16**, so deleting the FP16 row leaves
  CUDA LayerNorm FP32-only. Pinned by a `static_assert` so this work must confront it.

### Model Distribution

- [ ] **`ExportArtifact` becomes `mila-compress` and gives the store back to `mila`.** It grew
  install/rename/validate verbs that duplicate the store tool — `ExportArtifact --install` adopts a
  local package while `mila install` downloads a published one, the same word for two operations,
  and Chat (`Chat.ModelCatalog.ixx:387`) and MIS (`model_worker.py:90`) both point users at the
  wrong one. `mila-compress` keeps export, fingerprint, transcode, package. The name is in four
  user-facing strings and five docs, two of them Python, so nothing catches them at compile time.
- [ ] **Only Gemma refuses a pre-quantized artifact whose policy is not the one it compiled.**
  `GemmaModel.ixx:640` (and `:704` for the footprint sibling) compares `reader.getWeightQuantization()`
  against the requested policy; `LlamaModel::fromPretrainedImpl` and `GptModel` never read it. The
  storage dtype cannot substitute — FP4 at group 128 and group 64 are both U8 — so a mismatch
  reinterprets the nibble layout and produces a model that runs and is wrong. `ExportArtifact` already
  emits Llama artifacts, so the hole is reachable today.
- [ ] **`ModelSerialization.md` Phase 7 describes work that shipped.** The distribution artifact exists
  end to end — `savePretrained` (`LanguageModel.ixx:116`), the `mila_quantization` metadata key, the
  reader, the policy check, `Linear`'s pre-packed load branch, and `Tools/ExportArtifact` driving it.
  The phase text still calls it unwritten and the freeze-boundary table still lists it out of bounds.
- [ ] **A mistyped model name reports an authentication failure, but only to users without a token.**
  `HuggingFaceHub.ixx:283` maps every 401 to "no valid HuggingFace token. Set HF_TOKEN, or run
  'huggingface-cli login'." MEASURED both ways on 2026-08-14 with `gtp2-small`: an authenticated caller
  gets 404 and the correct "no such repository" message, an anonymous one gets 401 and is sent to obtain
  a token they do not need, which then fails identically. HuggingFace will not reveal repository
  existence to a stranger. The defect is therefore invisible to anyone who has ever run
  `huggingface-cli login` and hits only new users. When no token was sent and the owner is the public
  `mila-llm` org, lead with the name being wrong and point at `mila models --online`.
- [ ] **Publish `Llama-3.2-1B-Instruct-fp4` as the evaluation model** — sequenced AFTER the 3B path is
  proven, not before. Roughly 0.7-0.9 GB against the 3B's 2.87, and it drops the evaluation path's VRAM
  floor to about a gigabyte, so an 8 GB card stops being excluded from "does it work". Convert with
  `Tools/Converters/convert_llama_weights.py`, export the FP4 artifact, validate GENERATION rather than
  per-layer parity, publish with a card. Test it against the tools-free system prompt first: a 1B is more
  prompt-sensitive than the 3B, which refused plain questions until the default prompt changed.
- [ ] **`gpt2-small` installs and then cannot be used from Chat**, so it is the wrong first model for a
  quick start: the walkthrough ends in a 623 MB download and no conversation. Chat refuses base models by
  design ("a base model, and Chat is an instruct harness"). Since 2026-08-17 `/models` says so in the row
  ("not from Chat (base model)"), but that is *after* the download. Either the getting-started paths name
  an instruct model, or `/install` says so before the transfer rather than after it.
- [~] **Reconcile `Web/content/start.md` with the Get Started band now on the home page.** The four
  tabs (C++/Python/Docker/Clone) landed in `Web/layouts/index.html` as `#qs`, fragment-addressable
  per panel (`#p-docker`), so the channels have their deep-link target. What is unresolved is that
  the site now has two getting-started surfaces: the band, and `/start/`, which the nav and the
  home-page "Get started" box still point at with the older clone-and-build content.
  **Supersedes the §3 item below** — do not fix that section separately, the reconcile owns it.
- [ ] **`Web/content/start.md` §3 "Get model weights" is retired in every sentence** — conversion as
  the path, "there is no separate quantized checkpoint to manage" (mila-llm is exactly that), and
  "GPT-2 is the easiest first target: it is ungated... Llama and Gemma are gated and require auth",
  which is now backwards: all four published models are ungated and need no account, and `gpt2-small`
  is published like the rest. The page's front-matter description also sells "convert model
  weights". This is the getting-started path on the primary marketing site.
- [ ] **The home page hardcodes `0.20.0-beta.3` in three places, so the site and the release tag must
  ship naming the same version.** Two image tags in the Docker panel and the Evaluating band, plus the
  FetchContent pin. Every later release breaks those commands until the copy is updated with it.
  `Web/layouts/index.html` — the `#p-docker` panel and `#evaluate`.
- [ ] **The C++ tab pins `v0.20.0-beta.2` but its sample output reads `Mila 0.20.0-beta.3`.** Carried
  over verbatim from the approved mockup. One of the two is wrong to a reader who checks.
  `Web/layouts/index.html`, `#p-cpp` steps 1 and 3.
- [ ] **Nothing cites `scripts/dockerhub/`.** Four files remain, in two channel groups; `RELEASING.md`
  and `wheel-cleanroom.yml` both reach into `pypi/`, but neither `README.md`, `getting-started.md` nor
  `Docker/README.md` names the image half. `build-runtime-image.sh` is the one carrying the published
  arch list, so it is undiscoverable knowledge until the publish script absorbs it.
- [ ] **`Web/content/docs.md:28` states "quantization has no checkpoint format."** True when written,
  false now — every published model is a quantized checkpoint. The surrounding point (the type
  chooses the reduced-precision path) still stands and should survive the correction.
- [ ] **The site links GitHub and nothing else.** No HuggingFace, no PyPI, so the primary marketing
  site does not point at the model store or the package. See [[project_four_channel_roles]] —
  four channels, four jobs, and the site is the hub.
- [~] **Mila is a library, never a "runtime."** The noun names an engine you hand a model to, so it
  argues with "no hidden execution engine" in the same breath. Three user-facing sites remain:
  `Web/content/docs.md:38`, `Web/content/blog/implementing-gemma-4.md:4` and
  `Web/content/blog/gemma-4-docker-openai-api.md:4`. Not a sweep: "at runtime", "runtime dispatch"
  and the two places naming what Mila is *not* (`flashattention-prefill-kernel.md:98`,
  `docs.md:8`) are correct as written. Whether "the runtime" as the name for `Mila/Src` versus its
  adaptors also changes is a separate, open call — it would rename a design term across
  `MilaProductFamily.md`, `CLAUDE.md` and four READMEs.
- [ ] **Two Validated Capabilities rows are deliberately withheld pending evidence, and will be
  forgotten otherwise.** `pip install mila-llm` goes in once the Windows clean-room gate is green and
  beta.2's wheels are on PyPI; the footprint pre-flight goes in once GPT-2 has `getRequiredMemory` and
  Gate B has covered `NoWeightQuant` — until then it can only be claimed for Gemma 4 and Llama.
- [ ] **No C++ tool has a `pull` verb**, so the cold download cannot be exercised from a C++-only
  machine without a human at the `/install` prompt. Python is already covered:
  `ModelStore.pull(name, owner, transport=None)` is bound (`Mila_py.cpp:309`) and is what pulled
  6.33 GB in the Linux clean room, so this is a gap in the tool, not in the product. It lands on
  `mila` with the other store verbs, and is **not** `ExportArtifact --fetch`, which is an HTTP-client
  diagnostic that happens to write a file.
- [ ] **`/models --online` answers SUPPORT but still cannot answer FIT, and only fit needs the
  `Mila/Src` change.** Landed 2026-08-17: the column settles architecture, variant, `instruct` and
  the one certain memory negative (download alone exceeds the card — sound because the download is a
  *lower* bound on device weights, since FP4 unpacks and the unquantized tables ride along). What is
  still unanswerable is how much context fits, because `ModelManifest` (`ModelManifest.ixx:53`)
  carries **no geometry** — no layer count, no head dims — so there is nothing to predict from.
  Two ways in, and they are different owners: a `Range` read of the safetensors header (8-byte length
  then JSON, both at the file's start) so the online row runs the *same* `largestFittingContext` as
  the installed row, blocked on the footprint path taking a path rather than a byte range; or
  geometry fields in the manifest, which is Model Distribution's phase 7. Never an estimate in that
  column — a yes there that a load then contradicts costs the whole table its credibility.
- [ ] **`/models --online` costs one GET per listed model.** Invisible at one model, N+1 requests at N.
  Only worth revisiting if the published set grows; noted so the cause is known when it does.
- [ ] **`NOTICE.md:33` omits curl, and may no longer need to.** The note treats notice-carrying as open
  for "a binary distribution that links them" — but **both** wheel presets are now
  `MILA_ENABLE_LIBCURL=OFF`, so a wheel built today contains no curl at all. Establish whether the
  *published* artifact predates that change before writing anything: the answer decides whether this is
  an obligation or a non-issue. The same note points at a bucket that no longer exists; fix that either way.
- [ ] **`mila/__init__.py` is copied by a `POST_BUILD` step of a target it is not a source of.**
  `Mila/Bindings/CMakeLists.txt:95` stages it with `copy_if_different` off
  `add_custom_command(TARGET MilaPy POST_BUILD)`, which runs only when `MilaPy` relinks — so editing
  only `__init__.py` leaves `<build dir>/python/mila/` stale and a sample fails with a missing
  attribute. Use `add_custom_command(OUTPUT ...)` with `DEPENDS` on the source.
- [ ] **`gpt2-small` is published but its installed record predates `kLicenseRole`.** The store copy
  (installed 2026-08-03) declares weights and tokenizer only, so the hub repo carries LICENSE and the
  local disk does not — the exact split the legal-files change exists to close. Reinstall from
  `Data/Models/Packages/gpt2-small`; both blobs are already adopted, so it costs one small file.
- [ ] **`gpt2-small` is the first published artifact with no quantization suffix**, so it lands on the
  `NoWeightQuant` path that Gate B has never checked, and on GPT-2's missing `getRequiredMemory` —
  Chat's pre-flight says nothing for it. Both are open above, and both are now live rather than
  hypothetical.
- [ ] **The org card defines an artifact as "already quantized", which `gpt2-small` makes false.**
  The catalogue is now pre-quantized deployment artifacts *plus* a reference model for reading and
  training. Say there that MIS does not serve GPT-2, so nobody files it as a bug. Card source is
  `.internal/Marketing/HuggingFaceOrgCard.md`.
- [ ] **`--instruct` is undocumented in `--package` mode, and its absence is silent.** The flag is
  parsed (`ExportArtifact.cpp:142`) but missing from the package-mode option list in the usage text
  (`:42-56`), so omitting it writes `instruct: false` into the manifest with no warning — changing
  the prompt template every consumer applies. Caught only by diffing a rebuilt manifest against the
  previous one. Document it, and consider refusing an instruct-named model that declares otherwise.
- [ ] **`gemma-4-12b-it-fp4` now has two manifests.** The package directory carries the current one;
  `ModelCards/gemma-4-12b-it-fp4/mila.json` is the pre-package copy and no longer matches. One of
  them has to go, and the card directory's `publish.json` flow goes with it.
- [ ] **The "which licenses require a displayed attribution" rule is written twice in two languages** —
  `requiredAttributionFor` in `Chat.ModelCatalog.ixx` and `license_id.startswith("llama")` in
  `publish_model.py:209`. They agree today. A third family with a display duty is what separates them.
- [ ] **The README implies FP8 and BF16 are reachable, and after an FP4-only publishing decision they
  are not.** `applyRequestedQuantization` refuses to reload a pre-quantized artifact as anything
  else, so every published model is FP4-at-runtime and the FP8 rows at `README.md:163,165` are
  converter-only capabilities. Say so, or the table promises a deployment nobody can reach.
- [ ] **`gemma_greedy_parity.py` diffs an FP4 Mila against a BF16 HuggingFace reference and does not
  say so.** `Mila/Tools/Converters/Gemma/gemma_4_BF16/gemma_greedy_parity.py:70` loads through the
  binding's FP4 default, so any divergence it reports mixes quantization error with a real defect.
  `from_pretrained` now takes `quantization=`, so the honest comparison is one argument away — on a
  card that can hold a BF16 12B. State which it ran either way.
- [ ] **Llama FP4 is not parity-tested, and the README's own wording admits it.** The BF16 and FP32
  rows say "Validated against HuggingFace"; the FP4 rows say "coherent generation"
  (`README.md:162-165`). `GemmaModel.Parity.Cuda.cpp` is the only parity test in the tree. Publishing
  FP4 only makes this the claim the whole catalogue now rests on — see the Llama HF-parity item under
  Models.
- [ ] **The licensing story is per-family and must not be generalized.** Gemma 4 is Apache 2.0 (public,
  ungated); Gemma 3 and earlier carry the Gemma Terms of Use; **Llama 3.1/3.2 may be republished, but
  attributed** — ship the agreement, display "Built with Llama" and Meta's notice, pass along the AUP,
  and begin the model name with "Llama" (`Llama-3.1-8B-Instruct-fp4` does). Gating is a *policy*
  choice, not a licence condition. See [[project_gemma4_apache2_license]].
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

- [ ] **Chat cannot choose a GPU.** `ChatConfig` has no device field and the runtime's device string
  is always resolved as `CUDA:0`, so on the two-card rig the only way to reach the second GPU is
  `CUDA_VISIBLE_DEVICES=1` in the environment. The library already names devices `CUDA:N`
  (`CudaDevice.ixx:88`); the gap is that no adaptor setting reaches it.
- [ ] **`ToolCallParser::parse` routes ANY response containing `[` into the tool-call parser** —
  `Chat.ToolCallParser.ixx:63` uses `response.find( '[' )` where the class's own doc comment at `:35`
  says "Leading `[`" and the nested `parseTagged` path at `:109` tests it correctly. Found on an
  ordinary Llama 3B turn. It degrades gracefully today, but any prose with a bracket (markdown links,
  `[1]` footnotes, an array literal) enters the path, and a parse that ever *succeeds* on prose would
  swallow the answer and emit a phantom tool call.
- [ ] **The logger writes over the spinner** — `Logging` writes to the console independently of
  `ConsoleRenderer`, which owns that line, so a model switch renders the warning on top of the spinner.
  Cosmetic, but it is the first thing a user sees on every switch that logs.
- [ ] **`printThinking` still takes the plain-text path.** The answer block paints style spans; the
  reasoning block does not, so a heading or bold label inside a thought renders unstyled
  (`Chat.Renderer.ixx:176`). Harmless today — the whole block is dim by design — but it is now the
  one renderer entry point that ignores attributes, which is exactly how a second convention starts.
- [ ] **Wrapped list items do not hang-indent.** A continuation line starts at the bullet's own
  indent rather than under the item text (`• Protostar: ...` / `material inward.`), so a wrapped
  item reads as a new paragraph. `wordWrap` preserves a line's leading indent but has no notion of a
  continuation indent (`Chat.RichText.ixx:99`).
- [ ] **`Chat.StreamingDisplay` has no tests.** `RichText` now has 18 (`Mila/Tests/Adaptors/Chat/`),
  but `holdPoint` and the chunk-boundary behaviour that produced the nested-bullet defect are still
  unpinned. Harder than RichText: the module imports `Chat.Renderer` and `Chat.Config`, so it needs
  either a seam or those modules in the test target.
- [~] **MIS Gemma 4 tool-calling validated end-to-end** — Codex and Claude Code CLI round-trips are
  live and the native grammar is reconciled to Google's canonical template, pinned by an oracle.
  Remaining: N sequential distinct tool calls in one turn, channel-content parser polish, and
  Codex-CLI re-validation on the reconciled grammar.
- [~] **Grammar-in-runtime execution-time scope call** — the C++ and Python grammars are held together
  by a cross-language parity test. Open for sign-off: single-source via pybind, or close on the test.
- [ ] **In-turn thoughts dropped between tool calls** — Google's multi-turn rule is to strip
  prior-turn thoughts and keep the current turn's.
- [ ] Buffer Gemma Anthropic streaming only when tools are present.
- [ ] **Chat's `context_length` needs an `auto`, and the interim clamp is a placeholder.** One session
  config serves every model a session loads, so the number is either too small for a 12B or fatal for
  GPT-2's 1024-row learned positions; today it is clamped by `maxContextFor` (`Chat.ModelCatalog.ixx`),
  a per-family constant that is honest only for GPT-2. The real answer is the largest context that
  fits the card, which `getRequiredMemory(BuildContext)` can already compute — open questions are the
  headroom fraction and the behaviour when even the minimum does not fit.
- [ ] **Model capabilities belong in the manifest, not in a family switch — the second reasoning
  family breaks the current scheme.** `thinking_capable` and `streaming_capable` are both
  `family == Gemma` (`Chat.FamilyTraits.ixx`), and `default_context`/`max_context` are per-family
  constants in the same table, so two models of one family cannot differ and a non-Gemma reasoning
  model reads as
  having no channel. `instruct` is already record-declared and is the proof of the pattern; the
  manifest tolerates unknown fields, so adding `context_length` and a reasoning-channel declaration
  is additive, with `minimum_mila_version` as the lever when a model needs a newer build. Doing this
  BEFORE the next chassis is what stops it threading a second switch through every site.
- [ ] **`import Mila;` breaks the standard library in the consumer's translation unit.** Three
  failures, all in a real standalone FetchContent consumer, all absent without the import:
  (1) any C++ stream **input** fails — `std::getline(cin, string)` and `cin.getline(char*, n)` both
  die on "'_Ok' uses undefined class `basic_istream::sentry`"; (2) instantiating any model fails
  unless the consumer includes `<sstream>` **before** the import, because `Component::toString()` is
  virtual so `GemmaModel::toString()`'s body compiles into the consumer via the vtable and uses
  `std::ostringstream`; (3) putting `import Mila;` before the includes is fatal — C1116
  "unrecoverable error importing module 'Compute.CpuDevice'", with MSVC's own report-a-modules-bug
  note. Proven by compiling the identical `getline` call with and without the import: clean without.
  Nothing caught this because Chat and the tests build *inside* the tree, and
  `packaging_fetchcontent_consumer`'s fixture only calls `initialize()` and prints a version — it
  never instantiates a model or reads input. `Samples/QuickStart/Cpp/main.cpp` carries two named
  workarounds (`<sstream>`, and `std::fgets` instead of stream input) that should be deleted when
  this is fixed. Likely a GMF-reachability problem; MSVC 14.51.36231, CUDA 13.3.
- [ ] **Test whether `import std;` in the consumer clears the `import Mila;` std breakage.** Leading
  hypothesis for the item above: with no textual std headers there is no include/import duality to
  confuse ownership, and Microsoft's own C1116 page names mixing `import` and `#include` as a cause.
  Not yet run — `import std` is still experimental in CMake 4.0.1 and `CMAKE_CXX_MODULE_STD` must be
  set *before* `project()`, so it needs a fresh build directory and a full from-source Mila build
  (~15 min). Two caveats if it works: it raises the consumer floor to an experimental CMake feature,
  which is a lot to ask of a quickstart; and it only fixes the consumer side, since Mila's own
  modules `#include` in their GMFs — if that is the root cause the real fix is in `Mila/Src`.
  Searched 2026-08-13: no filed bug matches this signature, and MSVC emitted its own
  report-a-modules-bug note, so it may be unreported.
- [ ] **Make `packaging_fetchcontent_consumer` instantiate a model and read input.** Its fixture is
  a version print, which is why the defect above sat undetected — the gate proves Mila *links*, not
  that its module is *usable*. It needs no GPU and no model to catch all three failures: they are
  compile-time. Cheapest possible guard for the entire C++ consumer story.
- [ ] **The Python binding discards `GenerateStatus`, so the two quick starts cannot reach parity.**
  `Mila_py.Wrappers.cpp:657` (Gemma) and `:553` (Llama) do `(void)impl_->model->generate(...)`, so a
  Python caller cannot tell EOS from the `max_new_tokens` cap from context overflow from a
  cancellation — which `LanguageModel::generate`'s own docstring calls the one outcome a caller
  cannot reconstruct from the token stream. The C++ quick start prints `[stop]`; the Python one
  prints nothing, and that gap is visible to anyone reading them side by side as the website's two
  first tabs. Session-depth, so consistent with the settled binding scope.
- [ ] **Nothing stops the quick starts rotting again — a single-shot sample is testable and the old
  one never was.** Prompt in, tokens out, exit code is CI-shaped given a model in the store, which
  is the only real defence; `packaging_fetchcontent_consumer` proves its own fixture compiles, not
  this sample. Blocked on a fixture that needs no multi-gigabyte download — see the `gpt2-small`
  Chat-test-model item above, which is the same gap.
- [ ] **Decide whether a Python completion sample needs a `GptSession` before it can exist.**
  `Samples/QuickStart/Python/generate.py` already shows completion as a mode via `--raw`, so the only gap is
  GPT-2 itself — and the binding exposes just `LlamaModel` and `GemmaModel`. That is also why MIS
  refuses the architecture (`Server/model_worker.py:40`: "gpt2 has a record shape and no session"),
  making this a binding decision, not a sample one. Session-depth, so consistent with the settled
  binding scope; still net-new projection surface for a sample Python largely already has.
- [ ] **Chat lost its fast test model when base models were refused.** `gpt2-small` loaded in
  seconds and is what surfaced both the `context_length` crash and the thinking-row defect. Every
  remaining model is multi-gigabyte, so without a fixture that needs no download, Chat's test path
  is one nobody will run.
- [ ] **Decide where a user's Chat config lives — a container user has nowhere to put settings.**
  `session.json` ships inside the image layer, so changing `temperature` or `context_length` means
  mounting a file over it; `--config` assumes a file you can already write. Related: `chat-state.json`
  now sits in the store root, which `resolveStoreRoot()` resolves to a *cache* directory on Linux —
  survivable for a recoverable model name, wrong for real config. Two shapes weighed (beside the
  store, or `MILA_CONFIG_DIR` with both paths under one volume mount); settle it once, since the
  `context_length` `auto` setting will want the same home.
- [ ] **Publish `mila-llm-server` to PyPI.** The restructure is done and the version now derives from
  `Version.txt` like the binding's, so what remains is the release step: RELEASING covers the four
  CUDA wheels and says nothing about MIS. It is one `py3-none-any` file, built from the configured
  checkout with `python -m build`, and it goes beside the wheel upload in the same window.
- [ ] **`main.cpp` re-checks what the store already guarantees** — after `resolveModel` succeeds it
  tests `exists()` on both paths, but `locate()` refuses an incomplete record. Harmless duplication,
  except `/model` has no equivalent check; if the guarantee is doubted, the check belongs in the store.

---

## Future

Next-cycle work. Coarse by design — detailed tasking happens only when an item promotes into a release.

- **[gate] One typed model handle + factory, before ANY next chassis** — the architecture-to-concrete
  erasure exists three times in two languages (Chat's `ModelVariant`, the binding's `*Session`
  classes, MIS's `ModelFamily`), which is why GPT-2 is missing from MIS. Lands in the runtime-adjacent
  native agent core; sequencing and reasoning in `MilaProductFamily.md` Open Decision 2. **After the
  v0.20 tag, before the chassis expansion below** — the chassis is what multiplies the cost.
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
- **The library should own architectural identity** — the set of architectures is the set of model
  classes `Mila/Src` implements, and it is held today as a compile-time type and an unvalidated
  manifest string with nothing connecting them, so each consumer writes its own bridge
  (`familyFromArchitecture` in `Chat.ModelCatalog.ixx:159`, `architecture == "gemma"` at
  `Mila/Bindings/Mila_py.Wrappers.cpp:413`) and a fourth model class means editing all of them. Home
  is `Distribution`, beside the manifest reader, not `Dnn`. The library owns the identity only:
  traits merely keyed on it, such as Chat's `streaming_capable`, stay with the consumer they
  describe. Chat's `Chat.FamilyTraits.ixx` is acknowledged as the wrong owner and is fine until this
  lands.
- **Model serialization** — the remaining checkpoint round-trip and distribution-artifact phases.
  Design, defect analysis and the phase plan are in `Specifications/ModelSerialization.md`.
- **Retire quantize-on-load — one load shape for every policy.** `Linear::loadParameter` refuses a
  compute-precision blob, uploads packed bytes, binds, derives; the dtype sniff at `Linear.ixx:601`
  and `CudaLinearOp::quantize()` go, and FP8/FP4 fitting joins the sub-4-bit fitter in
  `Tools/Quantization` — one producer for every format, and artifact production stops needing a GPU.
  The codebook path is already this shape (`:574`). Depends on the FP4/FP8 codecs, and takes Chat's
  `/model <alias> fp8` load-time keyword and `quantization_applied_at_load` with it. An API change
  to `Mila/Src`, which is why it waits. Rationale in `Quantization.md`.
- **Python binding — numeric access, not component access.** Add a session-level `forward()` returning
  logits, plus final hidden states, to `LlamaSession`/`GemmaSession`; from Python a parity run can
  compare token ids and nothing else today. Component, tensor and training bindings are ruled out:
  `TDeviceType x TPrecision x TWeightQuantization` is erased only at the session PIMPL, so each
  component would multiply the wrapper. `Mila/Bindings/Mila_py.Wrappers.ixx:362`
- **API Coherence** — the pre-1.0 consistency pass, and the precursor to any API-stability promise.
  Its first named item: **`loadModel`/`saveModel` and `loadCheckpoint`/`saveCheckpoint` — verb plus
  what you get, both directions.** Two words go: "pretrained" is relative to a fine-tuning stage Mila
  does not have and is doubly wrong on the write side, where the bytes are ones Mila quantized;
  "artifact" is build-tooling vocabulary for a file that is simply a model. `from` goes with them --
  it names the *source* form, which only informs when the source differs from what you get, so
  `fromCheckpoint` earns it and `fromModel` cannot. `saveCheckpoint` already has the target shape and
  becomes the template rather than the exception. Distinction to document: a checkpoint carries epoch
  and train/val loss as one of a series (`Dev/Training` drew this); a model is terminal. One wrinkle:
  `Network::load( archive, mode )` restores into an existing graph, so a static `loadModel` uses the
  verb differently -- the suffix and the static call site are what separate them. **The methods are
  the small half:** `kArtifactMinimumMilaVersion`, `ModelDistribution.md`, both model cards, the
  binding's `from_pretrained`, MIS and the samples all speak the old vocabulary, and changing it
  piecemeal is how one concept ends up with three names. Sequence with the `ExportArtifact` ->
  `modelmgr` rename, which it makes more coherent, and the binding's `quantize_fp8` fix.
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
