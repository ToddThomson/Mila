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
  `prefill()`/`decode()` directly). The C4702 at `:216` below is the same root cause seen from the
  backward side — both clear together when this task lands.
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
- [x] **`Structural` (`split`) backfill — and it found a memory-safety defect.** Verified green 2026-07-28.
  `Tests/Dnn/Tensors/TensorOps/Structural.Cuda.cpp` covers both overloads, the null-context default
  stream, and every documented precondition throw. No `.Cpu.cpp` peer: `split` is CUDA-only (there is
  no `CpuTensorOps::split`), so the spec's "CPU file + `.Cuda.cpp` companion" pattern does not apply
  here — a CPU instantiation would not compile.
  **The defect: `StructuralOps::split` validated a flat `D % 4` for both precisions, but the two
  kernels move one 16-byte vector per thread — `float4` is 4 elements, `uint4` is 8 BF16.** A BF16
  slice width of 4 therefore passed validation, then ran the kernel's `D0/8` index arithmetic on a
  truncated quotient and stored eight elements into a four-element output row: an out-of-bounds write,
  not merely a wrong result. Unreachable from the shipped models (Gemma/Llama fused projections are all
  multiples of 8), which is why it survived. Fixed by keying the alignment off the element width
  (`kVectorElements = 16 / sizeof(T)`); both docstrings said "multiples of 4" and now state the rule.
  Three regression tests pin it.
- [~] Core `Tensor.ixx` coverage to the value-type archetype — remaining: `TensorOps.Transfer`
  device-split, and the wider `Tensors/` tree (`TensorBuffer`,
  `TensorDataType*` maps, `Partitioning`, `Serialization`). See `Specifications/Testing.Tensors.md`.
  Eight `REVIEW:` markers name the specific contracts to pin: Copy as a no-op on empty tensors and on
  scalars (`TensorOps.Transfer.ixx:92`); context/device compatibility and the device-ID logic on the
  CUDA transfer path (`CudaTensorOps.Transfer.ixx:132,140,276`); sub-byte/packed FP4 sizing
  (`Tensor.ixx:267`, `TensorBuffer.ixx:78`); the size helper duplicated from `TensorBuffer`
  (`Tensor.ixx:83`); and the moved-from state (`Tensor.ixx:479`).
- [~] Backfill inference-drought coverage — load-time quantization (`PerChannelFp8`/`PerGroupFp4`,
  decode matvec kernels), `OperationTraits` dispatch, the Llama path. The `CudaLinearOp` quantization
  white-box is the sole legitimate op-layer test (unreachable through the public component).
  **`OperationTraits` dispatch DONE 2026-07-28** — `Tests/Dnn/Compute/Operations/OperationTraits.cpp`
  (+ `.Cuda.cpp`). The compile-time seam every component resolves its op through had no direct test at
  all. Written as `static_assert`s, not runtime checks: the contract under test *is* the compile-time
  one, so a regression fails the build rather than a binary, and the CPU half rides the CPU-only CI
  ratchet. Deliberately never names a concrete op class — those are not re-exported through the
  umbrella, and pinning them would couple the test to internal naming instead of to the dispatch
  contract. What it pins: the `OperationSupported` predicate returning **false rather than
  hard-erroring** (the seam a multi-precision typed test needs, and which only holds while the primary
  stays undefined); every registered row on both devices; that the policy axis is part of the key
  (`LinearOp` under the default `void` policy must *not* resolve, a weight policy must not satisfy
  GQA's KV axis, an unregistered `PerGroupFp4<32>` group size must fail); and that distinct policies
  and precisions resolve to **distinct types**, which is what stops a policy being accepted and then
  quietly dispatching to the wrong kernel. Deliberate absences are pinned as contract too, so removing
  one is a visible edit: no BF16 anywhere on CPU, no Llama-lineage CPU ops, no CPU GQA, no
  `PerChannelKvFp8` (Qwen 3), no `PerGroupFp4` token embedding.
  **Surfaced by writing it — the odd row in the whole table: `LayerNormOp` CUDA is registered at FP32
  and FP16 and *not* BF16**, while BF16 is the primary target and FP16 is slated for removal. Deleting
  the FP16 row under "Remove FP16" leaves CUDA LayerNorm FP32-only. Pinned by a `static_assert` so that
  work has to confront it rather than discover it. Remaining in this item: the load-time quantization
  white-box and the Llama path.
- [~] Re-green in sample-revival order — MNIST spine mostly landed; remaining: the `Core/Network.cpp`
  delta, GPU companions (`Network.Cuda`/`AdamW.Cuda`), then the Bard GPT-2 stack tail.
- [x] **Retire the redundant op-layer mirror tests — closed 2026-07-28 as already done.** The item
  read as pending work ("files kept on disk pending an explicit delete") but there is nothing to
  delete: `Tests/Dnn/Compute/Operations/` holds only `OperationBase.cpp` and the two new
  `OperationTraits` contract files, there is no `Operations/Cuda/` directory, and
  `git log --all --diff-filter=AD` over `Mila/Tests/**` returns no op-layer test path ever tracked.
  The files went with their CMake entries; only the DELETE banners at `Tests/CMakeLists.txt:130`
  and `:207` remain, now describing entries no longer beneath them. The rationale they record is
  still worth keeping — backend ops are implementation detail, not in `import Mila;`, and their
  numerics are reachable through the component — so the banners stay as the standing rule against
  re-adding op-layer mirrors.
- [x] **Deleted the orphaned `Tests/Dnn/Models/Gpt2/DatasetReader.cpp`.** Found 2026-07-28 by an
  orphan scan (every `.cpp` on disk against every path in
  `Tests/CMakeLists.txt`, commented entries included; the only other hits were the two
  `Packaging/*_consumer/main.cpp`, which their own sub-CMakeLists build). It is in no CMake list, so
  it has never been compiled against the current API, and it could not be: it targets
  `Gpt2::DatasetReader`, which no longer exists anywhere in `Mila/Src` or `Mila/Samples`, and its
  `MockLogger` overrides a `log( const std::string&, int )` virtual that the current
  `Logging::Logger` does not have (it exposes `log_trace`/`log_debug`/`log_info` taking
  `std::string_view`). Last touched in `76e87955` (0.13.12-alpha.5). Superseded by
  `TokenSequenceLoader`, which already has live coverage in `Tests/Data/Loaders/`. **Not a Bard
  revival fragment** — the Bard tail needs the GPT-2 *training spine*, not this reader. Deleted
  rather than retired in place (explicit call): retire-in-place exists to keep superseded code
  readable, and here both the subject and its successor are already in the tree, so the file added
  nothing a reader could use. Its `DataWraparound` case is worth *not* reviving in any form — it
  slept 2000 ms for background threads and asserted on log message text.
- [ ] Backward-path kernels disabled or unverified behind `REVIEW:` markers — `CudaSoftmaxOp.ixx:73`
  and `:103` throw `"needs review"` with the real calls commented out; `Gelu.Fp32.cu:65` records that
  the shipped backward is not the numerically stable `sech^2` form. Gradient-check these before the
  suite can claim backward coverage. **Raised in priority 2026-07-28 by the RoPE FP32 finding above:
  that backward was not merely unverified, it was arithmetically wrong in the first component of every
  rotated pair, in a kernel carrying no `REVIEW:` marker at all and sitting beside a correct helper it
  never called.** The marked kernels are the known unknowns; the RoPE defect says the unmarked backward
  kernels need the same finite-difference sweep, not just the flagged ones. `Rope.Bf16.cu` being correct
  while `Rope.Fp32.cu` was not also means per-precision twins must be checked independently — a green
  BF16 case is not evidence for its FP32 sibling.
- [x] `CudaResidualOp.ixx:116-117` — `input_A` / `input_B` in the backward signature: **answered, the
  parameters are dead and permanently so.** `ConnectionType` has exactly one enumerator, `Addition`, so
  both partial derivatives of `A + B` are constant and the gradient depends on the output gradient alone.
  That is a property of a linear combination, not of the current implementation — only a nonlinear
  connection type (multiplication, where `dy/dA = B`) could make them live, and none exists. The
  signature keeps them for binary-op uniformity (`CpuResidualOp::backward` carries the same dead
  `input_b`); both unnamed with the reason stated at the site.
- [ ] **`ResidualConfig` advertises a scaling factor that no backward implements, and that the two
  devices disagree about in forward.** `withScalingFactor(float)` is public and `validate()` accepts any
  value `> 0` (`ResidualConfig.ixx:97`). CUDA forward honours it — `y = a + scale * b`
  (`Residual.Fp32.cu:34`) — but CUDA *backward* takes no scale parameter at all and the kernel writes
  `dA = dY; dB = dY`, so for any factor other than 1.0 the gradient w.r.t. `input_b` is wrong by exactly
  that factor. The only guard is `assert( scale_ == 1.0f )` at `CudaResidualOp.ixx:106`, which is
  **debug-only**: under `x64-release` / `x64-profile` / `x64-validate` it compiles out and a scaled
  residual trains silently wrong. `CpuResidualOp` ignores the factor entirely (no `scale` token in the
  file; forward is `Y[i] = A[i] + B[i]`), so the same config also gives different *forward* results per
  device. **Cheapest correct fix, and freeze-compatible because it removes an unimplemented knob rather
  than adding a feature: have `validate()` reject `scaling_factor != 1.0f` instead of only `<= 0`** —
  one fail-fast at config time on both devices, replacing a debug assert buried in a CUDA op. Implementing
  scale properly in both backwards plus the CPU forward is the alternative, and is more code on a training
  path that has never been validated end to end.
- [x] **Known-red CUDA tests (5) — CLOSED 2026-07-28, suite verified green in one pass (VS2026).**
  Surfaced by `x64-validate` ctest at the beta.1 cut (1417/1418 pass); all CUDA-path, so invisible to
  the CPU-only CI ratchet. Accepted non-blocking for beta.1, triaged and fixed in one session.
  **Two of the five were defects in the *test*, one was a defect in a *shipped kernel*, one was a
  stale budget, and one was an unimplemented kernel that threw — the recorded "suspect flaky" and
  "likely tolerance" priors were wrong in three of five cases. Worth carrying into the next triage:
  a red numerics test is not by itself evidence of a numerics problem.**
  - [x] `LinearCudaQuantizedTests.Forward_Fp4PrefillMatchesDecodeAcrossTokenMagnitudes` — **stale
    tolerance, not a live break.** A bit-faithful CPU model of both paths on this exact fixture (FP4
    group quantize -> FP4->FP8 upcast with the per-tensor `sB` -> per-token BF16->FP8 activation
    quantize -> FP32 accumulate -> BF16 epilogue, against the W4A16 decode matvec) puts the worst
    correct-path deviation at **0.061 * row_absmax** against the test's 5e-2 budget — red by 1.22x,
    with 90 of 4096 comparisons over. The 5e-2 was calibrated for `kUseFp8ActivationPrefill=false`
    (BF16 staging, worst 0.0073, a 7x margin) and never re-derived when W4A8-FP8 shipped ON in
    `8724aa68`; the model reproduces that 7x control margin exactly, which is what calibrates it.
    Weight-FP8 and activation-FP8 rounding contribute in roughly equal shares, so there is no single
    kernel lever. Budget raised to **1e-1 * row_absmax** (~1.6x headroom) after confirming the test
    still discriminates: per-tensor activation scaling, a stale/degenerate `sB`, and a swapped nibble
    packing order overshoot the new budget by 10x, 10x and 32x. Anchoring on the row's L1 reference
    mass instead of absmax was measured and rejected (same 2.6x row spread, less obvious quantity).
  - [x] `BpeTokenizerGemma.Encode_StartOfTurn_IsSingleAtomicToken` — the test was wrong, not the code;
    confirmed against `BpeVocabulary.ixx:1465`, which registers `<|turn>`/`<turn|>` and friends.
    Replaced by `Encode_ControlTokens_AreSingleAtomicTokens` (sweeps all eleven registered Gemma 4
    control tokens) plus `Encode_Gemma3TurnMarkers_AreNotInTheVocabulary`, which pins the Gemma 3 ->
    Gemma 4 protocol change the old test asserted backwards.
  - [x] `RopeCudaTests.Backward_InverseRotationRecoversInput<Fp32>` — **a real defect in a shipped
    kernel, not a test problem.** `Rope.Fp32.cu` defines a correct `rotate_pair<negate_sin>` helper and
    then never calls it: both `rope_rotate_kernel` and `rope_decode_kernel` open-code the rotation.
    `rope_rotate_kernel`'s backward branch patched only `r1` and left `r0 = x0*c - x1*s` — the *forward*
    formula — so the round trip returned `x0*cos(2t) - x1*sin(2t)` in the first component of every pair
    instead of `x0`. `rope_decode_kernel` is worse: templated on `negate_sin` and ignoring it entirely.
    The BF16 twin (`Rope.Bf16.cu`) writes both branches correctly, which is exactly why only `<Fp32>`
    was red — BF16's 5e-2 budget is not the reason, its kernel was simply right. Both FP32 kernels now
    call `rotate_pair<negate_sin>`. **RoPE backward has therefore never been correct in FP32**; the
    decode-kernel half is inference-reachable in principle but only ever instantiated with
    `negate_sin=false`, so no shipped inference path was affected.
  - [x] `DeviceRegistryTest.ThreadSafeDeviceOperations` — **the race was in the test harness, not the
    registry.** `std::vector<bool> results` is the packed-bit specialization, so twenty threads writing
    distinct indices were doing read-modify-write on the same words and losing each other's results —
    a textbook lost update, and the exact signature of the "fails intermittently" marker that sat on
    the test. Switched to `std::vector<char>`, where distinct elements are distinct memory locations.
    Note this means the test never actually exercised what it claims to; it should be watched for a
    while before the registry's own thread-safety is considered evidenced.
  - [x] `LinearCudaTests.Backward_MatchesReferenceGradients<Bf16>` — **not tolerance: the BF16 bias
    gradient was an unimplemented stub that threw.** `compute_bias_gradient`
    (`CudaLinearOp.Plans.ixx:150`) threw `std::logic_error( "Bias gradient for bfloat16 not yet
    implemented" )`; the test builds with bias, so `bias_grad_ != nullptr` and backward threw outright.
    FP32 passed only because `cuda_reduce_sum_batch_fp32` exists. The consequence was wider than the
    single red test: **BF16 `Linear` could not train with bias at all.** Implemented 2026-07-28 (Todd's
    scope call) as `cuda_reduce_sum_batch_bf16` in `Kernels/MatMul/CudaReduction.cu`, declared in
    `Linear.cuh` where the commented-out stub had sat. One block per 32-column tile, `blockDim (32,
    vstep)`, so consecutive lanes read consecutive columns and the strided row walk stays coalesced;
    a bounds guard placed around the accumulation loop only (never around the `__syncthreads()`) lets
    it handle any `out_features`, so it needs no fallback twin like the FP32 pair. **It accumulates in
    FP32 and converts once on the final store — a BF16 running sum would be wrong rather than merely
    imprecise, since with 8 mantissa bits a term more than 256x below the partial leaves the partial
    unchanged and a long batch silently stops accumulating.** Follows the `+=` contract the FP32
    kernels and `Linear::backward` already assume. FP16 deliberately still throws: it is scoped by the
    "Remove FP16" trace in Production Hardening, and the `REVIEW:` marker at that declaration now says
    so instead of asking whether the function is needed.
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
- [x] **Mixed-precision AdamW master parameters were zeroed, not copied — every BF16/FP16 model
  trained from zero.** `CudaAdamWOptimizer::addParameter` allocated an FP32 master per parameter and
  called `zero( *master_param )` under a "For now, initialize to zero" marker. The master is not a
  mirror: the kernel reads `old_param = master ? master[idx] : (float)params[idx]`
  (`CudaAdamW.cu:131`), applies the update to it, and writes the narrowed result back to the
  parameter. So on the **first** `step()` the initialized weights were discarded and every element
  was driven to `-lr * (0 + wd * 0) == 0`. Fixed by widening the live parameter into the master at
  registration via `launch_convert_copy_kernel<NativeType, float>` — the same primitive the transfer
  path uses, chosen over the typed `copy()` free function because `addParameter` receives an
  `ITensor*` and must not assume a memory resource to downcast to. The stale `REVIEW:` at `:169`
  ("this precision check and master parameter logic is outdated") is resolved and removed.
  **Oracle — net-new, and its absence is the whole story: the mixed-precision path had no test in
  any file, active or disabled.** `AdamW.Cuda.cpp` is 40 FP32 cases that never mention masters, BF16,
  or mixed precision, so even reviving it would not have caught this. New
  `Tests/Dnn/Optimizers/AdamW.MixedPrecision.Cuda.cpp` pins the contract with an oracle chosen so a
  correct optimizer is exactly a no-op: zero gradient plus zero weight decay must leave the parameter
  untouched, where the defect drives it to zero. Plus a decay-only case (pins the master as the value
  decay applies *to*) and a five-step stability case (the master is rewritten each step, so an
  initialization defect compounds).
- [x] **`AdamWOptimizer<Cuda, BF16>` could not be instantiated at all — the public optimizer wrapper
  was unusable for mixed precision.** Found 2026-07-28 while writing the test above, which failed to
  compile at `AdamW.ixx:48`. The wrapper selected its backend with
  `std::conditional_t<TDeviceType == Cuda, CudaAdamWOptimizer<TPrecision>, CpuAdamWOptimizer<TPrecision>>`,
  and `std::conditional_t` **names both branches**: selecting the CUDA implementation still required
  `CpuAdamWOptimizer<BF16>` to be a valid template-id. That template is constrained by
  `PrecisionSupportedOnDevice<TPrecision, Cpu>`, and `BF16::supported_on_cpu` is `false`, so the
  never-selected branch failed its own constraints (C7602/C2923). **The device-agnostic wrapper was
  therefore broken for exactly the configuration the master-parameter path exists to serve**, which is
  why the zeroed master above was never observed through the public API. Replaced with a
  `Detail::AdamWImplFor<TDeviceType, TPrecision>` trait specialized per device, so only the selected
  branch is named; a missing pair is a hard compile error, matching the `OperationTraits` convention.
  Also reduces the file from two `#ifdef MILA_HAS_CUDA` sites to one.
  **The general lesson, worth applying anywhere else this pattern appears: `std::conditional_t` is
  not a lazy select.** Both arms are instantiated as template-ids, so it is unsafe whenever the arms
  are constrained templates that do not both accept the same arguments. Grep for `conditional_t` over
  a constrained template before trusting it. (Swept `Mila/Src` 2026-07-28: ten other sites, all
  selecting between plain types or unconstrained templates. AdamW was the only instance.)
- [x] **`adamw_update` had no BF16 instantiation — `CudaAdamWOptimizer<BF16>` compiled and failed to
  link.** Third defect in the same path, found only once the two above were fixed and the test got as
  far as linking. `CudaAdamW.cu` explicitly instantiates `<float, float>`, `<__half, __half>` and
  `<__half, float>`, and no BF16 variant. The kernel body always supported it — the
  `stochastic_rounding( float, __nv_bfloat16*, ... )` overload exists for exactly this path — so this
  was a missing instantiation, not missing functionality. It never surfaced because nothing could
  reference the symbol: the device-agnostic wrapper could not be instantiated for BF16 either.
  Added `adamw_update<__nv_bfloat16, __nv_bfloat16>`. Deliberately did **not** add
  `init_from_master<__nv_bfloat16>` for symmetry with the FP32/FP16 blocks — `init_from_master` is
  not declared in `Kernels/CudaOptimizers.h` at all, so it is unreachable from outside the `.cu` and
  its existing instantiations are dead weight.
  **Taken together, the three defects are one finding: BF16 optimizer support was written end to end
  — master parameters, the stochastic-rounding writeback, the precision branch — and never once
  exercised. It could not compile, could not link, and would have trained from zero if it had.**
  All three verified green 2026-07-28 (build + full suite + Gemma 4 chat coherent). That run is the
  first time a BF16 CUDA optimizer step has executed in this codebase, so the three new cases in
  `AdamW.MixedPrecision.Cuda.cpp` are new information rather than a regression check.
- [~] **[net-new]** Training-loop integration test (sample-independent) — MNIST spine covered by
  `Network.Cpu.cpp`; remaining: a GPT-2-stack analogue for the Bard spine.
- [ ] **[net-new]** Optimizer step-convergence test — minimizes a known convex objective in N steps
  (proves update direction + bias-correction, not just that `step()` runs).
- [ ] **[net-new]** TrainingMode / RuntimeMode behavior coverage — assert build/runtime-mode
  transitions allocate/skip gradient buffers correctly (regression guard for the lifecycle fix). Three
  `REVIEW:` markers are the invariant to assert, each guarding a state the author believes unreachable:
  `TokenEmbedding.ixx:221` and `Lpe.ixx:187` ("if built and in training mode these buffers should
  always be initialized -- if not, it's a bug"), and `Lpe.ixx:495` ("must already be built").
- [x] **CUDA `fill_normal` / `fill_uniform` FP32-only gap fixed — it was a heap overrun, not just
  corrupt values.** Verified green 2026-07-28 (build + full suite + Gemma 4 chat coherent). Both functions are constrained to `is_float_type` (FP32/FP16/BF16/FP8) but
  unconditionally cast the destination to `float*` and generated `n` FP32 values into it. For a BF16
  tensor that is **4n bytes written into a 2n-byte buffer** — a full tensor's worth of overrun past
  the end — and the values that did land in range were FP32 bit patterns reinterpreted in pairs as
  BF16, which puts roughly half of them around 1e14. Reachable from `Linear<Cuda, BF16>` on a
  training build via `xavier`, and from `TokenEmbedding`/`Lpe` via `fill_normal`. `CpuTensorOps`
  was always correct (it converts per element through `static_cast<NativeType>`), so this was a
  CPU/CUDA divergence as well as a defect.
  **Fix:** non-FP32 tensors generate into an FP32 scratch buffer and narrow through a new
  `launch_convert_f32` (BF16 and FP16 overloads); FP32 still generates in place, and the odd-count
  Box-Muller padding path is preserved. `fill_uniform` scales and shifts at FP32 *before* narrowing,
  so only the final value rounds. FP8 and FP4 are now a **compile error** naming the type rather
  than a silent narrowing — train-from-scratch into a 4-bit weight is not a meaningful request, and
  the `is_float_type` constraint alone would have admitted it.
  **Oracle:** `Tests/Dnn/Tensors/TensorOps/Random.Cuda.cpp`, a FP32+BF16 `TYPED_TEST` sweep —
  distribution shape, uniform bounds, the Glorot bound, the odd-count path, empty-tensor no-ops, and
  a neighbouring-allocation check aimed at the overrun. Seeded via `RandomGenerator::setSeed` so a
  distribution failure repeats rather than flickers. This closes the **init-at-precision** success
  criterion ROADMAP names for Training Revival, and retires the `Testing.Tensors.md` note telling
  future readers not to write CUDA tests against this path.
- [ ] **[decoupled]** Revive the loss + backward path (CrossEntropy / SoftmaxCrossEntropy) — both
  samples compute loss host-side, so off the critical path to a converging sample.
- [ ] **[net-new, training-only]** Revive the `Dropout` component from `Dev/Components/Regularization/`.
- [ ] ProgressReporter — an injected per-operation progress facility for long-lived ops (BPE vocab
  training, `PretrainedReader` load, load-time quantization). `BpeVocabulary.ixx:624` is the concrete
  call site: an inline every-100-merges elapsed-time print asking to become an async progress callback.
- [ ] Validation — the **FP32** training path proven by the primitive suite (gradient-checks,
  optimizer step-convergence, loader contracts, init-at-precision, the integration test), CI-gated;
  the samples run as demos. Scope narrowed to FP32 GPT-2 / MLP 2026-07-28 (see ROADMAP): BF16 and GQA
  training move to the Training (advanced) release. The BF16 optimizer and initializer defects fixed
  in `e585be9d` stay fixed and stay tested — dormant and guarded, not reverted — they are simply not
  something v0.20 claims.

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

- [ ] Isolate third-party warnings structurally with `/external:I` + `/external:W0` (and `-isystem` for
  Clang/GCC). **Sequenced with the warnings-as-errors ratchet in `## Future`, not before it** — see the
  sizing below. `Mila/CMakeLists.txt:87` sets `/W4` as `target_compile_options(Mila PRIVATE ...)`, which
  reaches **only the `Mila` target**: miniz, nlohmann_json, cutlass, pybind11 and gtest are separate CPM/
  FetchContent targets built with their own flags, so their *sources* were never compiled at Mila's level.
  What `/external:` actually fixes is the other half — warnings emitted from inside third-party **header
  text pulled into Mila's own TUs** (miniz.h, nlohmann/json.hpp, cutlass, CUDA headers, std internals).
  Real, but narrower than this item used to claim. Two frictions to budget for: those headers enter
  through **module global-module-fragments**, and `/external:` behaviour across GMF/BMI generation is not
  well-trodden — validating it costs a full rebuild of a module-heavy CUDA tree; and `/external:` does
  nothing for **nvcc** diagnostics (the `#177-D` class), which come from the CUDA frontend. **Still the
  real precondition for any warnings-as-errors gate: without it, "warnings outside our control" is
  whatever your dependencies emit this month, and a CUDA toolkit bump breaks the build.** What it is
  *not*, once the count is small, is the only way to learn the first-party number — at n=12 that was one
  read of the Error List, and **the belief that the residue was third-party proved false**: the 4 C4267
  rows pointing into `<optional>` and `<xutility>` were a single first-party narrowing (below), and
  `/external:W0` would have **hidden** them rather than isolated them. Weigh that against the isolation:
  the std/miniz rows are where our own template arguments surface.

- [x] **`Network::save()` writes an archive missing most of the model's weights, and reports success.**
  Found 2026-07-29 while scoping the checkpoint API. `Component::save_()` is pure virtual and has 25
  overrides; **10 are empty** (`(void)archive; (void)mode;`). Six of those are correct —
  `Softmax.ixx:213`, `SoftmaxCrossEntropy.ixx:168`, `Rope.ixx:197`, `Residual.ixx:224`,
  `MultiHeadAttention.ixx:248` and `GroupedQueryAttention.ixx:406` all report `parameterCount() == 0`,
  so there is nothing to write. **Four own parameters and drop them silently:** `RmsNorm.ixx:169`
  (every norm weight in Gemma and Llama), `LayerNorm.ixx:210`, `TokenEmbedding.ixx:250` (the embedding
  table — the single largest tensor in the model) and `Lpe.ixx:274`. `Linear` is the only component
  that writes real tensor blobs, so a saved Gemma or Llama archive holds the projection weights and
  nothing else, with no error and no warning. Same class as the `ModelArchive::close()` defect below —
  a save that lies — and it survives because nothing in the tree calls `Network::save()` on a
  transformer.
  **Minimum honest fix, freeze-compatible: a component reporting `parameterCount() > 0` with no
  `save_` implementation must throw rather than no-op**, matching `GptModel.ixx:164`, where
  `fromCheckpoint()` already refuses instead of pretending. Implementing the four is the larger fix and
  belongs with the checkpoint API in `## Future`. Phase 0 of
  `Specifications/ModelSerialization.md`.

- [x] **`CompositeComponent::save_` collides every descendant into one archive scope — and this one
  makes the other two moot.** `Network::save()` pushes `components/<name>` per top-level child
  (`Network.ixx:510`), but the composite's own recursion at `CompositeComponent.ixx:783` calls
  `component->save_( archive, mode )` with **no `ScopedScope`**. Every descendant therefore writes into
  its parent's scope: in a 48-block transformer every `Linear` in every block writes
  `tensors/weight/data.bin` at the same path, each overwriting the last. Compounding it, the same
  function records `type` / `version` / `child_count` / `child_names` through `archive.addMetadata()`
  (`:764`-`:782`), which is the **archive-global** store — `ZipSerializer.ixx:426` writes the unscoped
  path `metadata/<key>`, bypassing `scopedPath()` — so every composite in the model overwrites the same
  four keys. `ModelArchive` supports the nesting (`scope_stack_` is a stack and `currentPrefix()` joins
  it); nothing pushes. **Fix this before implementing the four missing `save_` bodies above, or the
  result is a larger archive that is still wrong.** Phase 1 of
  `Specifications/ModelSerialization.md`.

- [x] **`Linear::save_` writes a truncated, mislabelled blob for any non-FP32 weight.**
  `Linear.ixx:306` takes `tmeta.dtype` and `:308` takes `tmeta.total_bytes` from the **device** tensor
  (`size() * elementSize()`), then the CUDA branch stages through a host
  `Tensor<dtype_t::FP32, CpuMemoryResource>` (`:317`) and hands `writeTensorBlob` that FP32 buffer with
  the device tensor's byte count (`:323`). For a BF16 weight that writes **half the staged buffer** and
  labels the result BF16 when the bytes are FP32 — every value wrong, not merely truncated. The bias
  block repeats it verbatim (`:330`, `:332`, `:341`, `:347`). The CPU branch is correct (no staging, no
  conversion), which is why the path reads as working. Quantized weights are not handled at all: a
  `PerGroupFp4<128>` weight is packed nibbles plus per-group scales, and neither the packing nor the
  scales have a representation here. Settle the staging dtype (mirror the device dtype rather than
  widen to FP32) alongside the quantized-artifact question in `## Future`. Phases 2-3 of
  `Specifications/ModelSerialization.md`, which also calls for hoisting the device-to-host staging out
  of `Linear` into one shared helper — the other four parameter-owning components need it, and copying
  the current branch would replicate this defect four more times.

- [x] **`Serialization.Tensor` is exported by nobody, so `Component::loadParameter` names a type its
  callers cannot see.** **Fixed 2026-07-29 while writing the Phase 4 round-trip test, which could not
  be written without it** — a test double cannot override `loadParameter` when its parameter type is
  unnameable, so this went from a filed observation to a blocker in one step. Added
  `export import Serialization.Tensor;` to `Mila.ixx`. **Verify by full rebuild, not by the test
  passing**: export-surface changes fail asymmetrically and per-compiler, so a green MSVC build is
  necessary and not sufficient — the clang leg is the real check. `Mila.ixx` exports six `Serialization.*` modules (`:276-281`) and **not**
  `Serialization.Tensor`, and no other module re-exports it. But `Component` *is* exported
  (`Mila.ixx:143`) and its public `loadParameter( const std::string&, const ITensorBlob& )`
  (`Component.ixx:509`) takes a type from that module — as do `TensorMetadata`, `TensorBlob<MR>`,
  `TensorBlobView`, `writeTensorBlob` and `readTensorBlob`. So the parameter-loading entry point is
  **reachable but not visible** to anyone consuming `import Mila;`: the signature resolves, the
  argument type cannot be named. Found 2026-07-29 writing the serialization tests, which wanted
  `readTensorBlob( archive, prefix )` to verify a saved blob and had to go through `ModelArchive`'s
  own `readMetadata`/`getFileSize`/`readBlobInto` instead. **This is the exact failure class already
  recorded for the quantization policies** — a type in a public interface must be visible, not merely
  reachable — including its nastiest property: it fails *asymmetrically and per-compiler*, so MSVC
  green is not evidence. Fix is one `export import Serialization.Tensor;` in `Mila.ixx`, but confirm
  against a full rebuild rather than a grep.

- [ ] **`save_` is public on `Component` and protected on `CompositeComponent`.** `Component.ixx`
  declares it in the `public:` section (`:163`-`586`); `CompositeComponent.ixx` overrides it at `:767`,
  inside `protected:` (`:651`). Legal, and calls through a `Component*` still work — which is why
  `Network::saveComponentGraph` never noticed — but it means the accessibility of one virtual depends
  on the static type you hold, and a caller holding a concrete composite cannot invoke it.
  Surfaced 2026-07-29 as C2248 in `Tests/Dnn/Core/CompositeComponent.cpp`, worked around with an
  `exposeSave()` forwarder. Pick one level and apply it in both places; the trailing-underscore
  convention suggests non-public is the intent, in which case `Component`'s declaration is the one
  that is wrong.

- [~] **Two exported types are named `MemoryStats`, and `import Mila;` makes both visible.** Fixed
  2026-07-29 (unbuilt): the allocator-level one is now `Mila::Dnn::Compute::MemoryAllocationStats`;
  `Mila::Dnn::MemoryStats` (the per-component figure) keeps the plain name, since it is the one on the
  public `Component::getMemoryStats()` contract. **Contained to two files** — the tracker itself and the
  parked `TensorBuffer.Tracking.cpp` — because `export import Compute.MemoryResourceTracker;` is
  *commented out* at `Mila.ixx:95`. The type was never in the documented public surface at all; MSVC was
  leaking it transitively through `TensorBuffer.ixx`, which is the same over-sharing already on record
  from the quantization-policy episode, and is why the clash appeared on MSVC and would not have on
  clang. The three call-site qualifications added as a workaround are reverted.
  `Mila::Dnn::MemoryStats` (`Core/Component.MemoryStats.ixx:33`, the per-component figure returned by
  `Component::getMemoryStats()`) and `Mila::Dnn::Compute::MemoryStats`
  (`Compute/MemoryResourceTracker.ixx:19`, the allocator-level figure). Both are exported from
  `Mila.ixx`, so any consumer with `using namespace Mila::Dnn;` **and**
  `using namespace Mila::Dnn::Compute;` — the pair every test file opens with — gets **C2872 on an
  unqualified `MemoryStats`**. Found 2026-07-29 when the serialization tests tripped it in
  `Tests/Dnn/Core/CompositeComponent.cpp`; worked around there by qualifying, which is a fix for the
  call site and not for the collision. Two same-named exported structs one namespace apart is an
  API-coherence defect, not a naming preference: the resolution is to rename one (the component-level
  one reads naturally as `ComponentMemoryStats`, the tracker one as `AllocatorMemoryStats`) rather
  than to keep qualifying at every consumer. Fold into the **API Coherence** pass in `## Future`.

- [x] **Every shipped composite bypassed the Phase 1 scoping fix, and the Phase 1 test could not see
  it.** Found 2026-07-29 when adding `Component::load_` turned a C2248 in `GptBlock.ixx:369`.
  `GptBlock`, `Llama.Block`, `Gemma.Block`, `MLP` and `GatedMLP` all register their children through
  `addComponent` **and** override `save_` with a hand-rolled walk that pushes **no scope** — so the
  scoping repair landed in `CompositeComponent::save_` never ran for any of them, and inside every
  transformer block all eight-to-sixteen children still wrote `tensors/weight/data.bin` at one path,
  each overwriting the last. **The Phase 1 test passed because its `TestComposite` does not override
  `save_`** — the fix was verified against a double that took the code path the real composites
  replace. Two further defects fell out of the same audit: `Llama.Block` listed **ten** children where
  eleven are registered, so one was never written at all; and `Gemma.Block` never wrote
  `layer_scalar`, the per-layer output scale it owns directly, so a Gemma archive silently lost the
  scales — a numerics change, not a missing extra.
  **Fix:** the four pure containers drop their overrides entirely and inherit the base traversal
  (safe: every hand-rolled member is resolved out of the child registry by name, so the base walks a
  superset). `Gemma.Block` keeps an override that calls the base and then writes its own
  `layer_scalar`, with a matching `load_` — it is the one composite in the tree that owns a parameter,
  and `CompositeComponent::load_` only recurses, so the own-parameter half has to be explicit.
  **The compile error was the useful part:** a hand-rolled `load_` on a class template is never
  instantiated until something calls it, so `GptBlock::load_` had been dead and unchecked. Making
  `load_` virtual put it in the vtable, forced instantiation, and the body failed immediately.
  **Generalizable: adding a virtual to a base can convert dormant same-signature methods in derived
  classes into live overrides — grep for the name across derived classes before adding one.**

- [~] **GPT-2 and Llama 3 pre-tokenization silently runs the ASCII fallback on every MSVC build.**
  **Interim fix 2026-07-29 (unbuilt): the warning is now emitted once per process via `std::call_once`,
  not once per tokenizer, and states the limitation as permanent rather than possible.** That makes the
  signal visible; it does not make the tokenization correct. The real fix is unchanged and still open —
  see below.
  Found 2026-07-29 by capturing first-chance exceptions across the whole suite under `cdb`. Both
  canonical patterns use Unicode classes — `BpePreTokenizationMode.ixx:33` and `:57` are
  `\p{L}`/`\p{N}` throughout — and **MSVC's `std::regex` ECMAScript mode does not implement `\p{...}`**,
  so `BpeTokenizer::initializePreTokenization` (`BpeTokenizer.ixx:344`) throws `regex_error` on
  **every** construction and takes the documented ASCII-approximation branch at `:346`. Measured: 52
  of the suite's 416 first-chance exceptions, one per tokenizer construction, 100% hit rate.
  **The fallback is deliberate and even logs a warning — that is what makes it worth filing.** The
  warning says "Non-ASCII text may tokenize differently from the HuggingFace reference", and on
  Windows that is not a *may*: it is every run, always. The signal is drowned by its own frequency,
  which is precisely how a permanent correctness limitation reads as routine noise. Gemma is
  unaffected (SentencePiece returns early at `:330`); **Llama 3 and GPT-2 are affected on the primary
  development and release platform.** No parity test would catch it — an ASCII prompt tokenizes
  identically under both patterns, so the divergence only appears with accented text, CJK or emoji.
  **Fix is a dependency decision, not a patch:** `std::regex` cannot express these patterns, so it
  needs a real engine (PCRE2, RE2, `boost::regex`) or hand-rolled Unicode class matching. Worth
  pricing against the reference-implementation positioning — "tokenizes like the reference" is a
  claim the project would want to hold. Interim, cheap, and honest: log the warning **once** per
  process rather than per construction so it is visible, and state the limitation in the tokenizer
  docs.

- [~] **`copy()` issues every device-to-device transfer twice.** Fixed 2026-07-29 (unbuilt): the second
  block is now `else if constexpr`. Safe by construction — the first condition
  (`!src_host && !dst_host`) strictly implies the second (`!src_host || !dst_host`), so chaining them
  changes behaviour for the device-to-device case only, which is exactly the one that was running
  twice. No test asserts the absence of a duplicate copy; a green suite proves no regression, and the
  correctness of the chain is by inspection. Noticed 2026-07-29 while tracing the
  dispatch for the serialization staging copy; unrelated to that work. `TensorOps.Transfer.ixx:100`
  handles the both-device-only case and calls `TensorOps<device>::copy`, then **falls through** to the
  `:112` block — which is a second `if constexpr`, not an `else if`. Its condition
  (`!src_host || !dst_host`) is also true for two device tensors, and its inner `!src_host` branch
  issues the identical copy again. Benign in result (a repeated copy is idempotent) but it doubles the
  cost of every D2D transfer and the `cudaMemcpyAsync` traffic that goes with it. Making the second
  block an `else if` is the whole fix; the third block's "Both are host-accessible" comment already
  reads as though the chain were exclusive.

- [x] **Serialization Phases 0-3 DONE and verified 2026-07-29 — including the Phase 2-3 criteria,
  closed by the round-trip coverage added with Phases 4-5.** `getParameterNames()` is now proven to
  match what `loadParameter` accepts for all five parameter-owning components: `Linear`, `LayerNorm`
  and `Lpe` through the GPT round trip, `RmsNorm` and `TokenEmbedding` through the Llama one, each
  asserted as one archive blob per parameter rather than restated as a list.
  *(status when first written, kept for the reasoning)* Build green, full suite green, and the new
  coverage below green — the first time the save path has ever executed. Before it, no test called
  `Network::save()` or any real `save_`; the only `save_` bodies the suite touched were stub overrides
  in test doubles, so green meant the templates instantiated and nothing more.
  **Remaining, deliberately not claimed:** `Specifications/ModelSerialization.md` sets Phase 2's bar at
  "for each of the five, `getParameterNames()` matches the set `loadParameter` accepts" and Phase 3's
  at "a saved Gemma archive contains one blob per parameter". Neither is asserted. The tests below
  prove the *machinery* — the guard, the layout, the per-tensor writer, the staging dtype — against
  test doubles; **the five real components' name-to-loader agreement (`Linear`, `RmsNorm`, `LayerNorm`,
  `TokenEmbedding`, `Lpe`) has no oracle**, and a mismatch there would surface only when the Phase 4
  load traversal tried to read a name save never wrote. Cheapest honest close is the Phase 4 round
  trip; a literal-name assertion per component is weaker but nearly free, and `RmsNorm.Cpu.cpp` /
  `TokenEmbedding.Cpu.cpp` are not in the active CMake list, so two of the five would need reviving
  first.
  Coverage added 2026-07-29, in place rather than in parallel files:
  **`Tests/Dnn/Core/Component.cpp`** — the guard on all three of its cases (parameters owned but
  unnamed throws; named passes; parameterless passes, which is what the six stateless empty `save_`
  bodies rely on), plus `saveParameterToArchive` on the host branch asserting dtype, shape, recorded
  byte count and the blob's actual size agree.
  **`Tests/Dnn/Core/CompositeComponent.cpp`** — the layout defect: sibling and grandchild blobs at
  distinct paths with a **count of `data.bin` entries** (3, not 1 overwritten three times); composite
  metadata under its own scope with nothing at the archive-global `metadata/<key>`; child order
  recorded as registered, using deliberately non-alphabetical names so a sorted or unordered container
  fails; and the guard rejecting a child that owns unnamed parameters. `MockChild` gained
  `getParameterNames()` and a real `save_` so the traversal has a faithful leaf to drive.
  **`Tests/Dnn/Core/Component.Cuda.cpp` (new)** — the staging branch, unreachable from CPU because it
  needs a device-only dtype. **The byte count alone does not discriminate here**: the old code staged
  through FP32 and wrote the BF16 count, producing a file of exactly the right size holding the first
  half of the FP32 data. So the test reads the contents — BF16 1.5 is six identical `0x3FC0` units,
  where the truncated FP32 buffer would alternate `0x0000` / `0x3FC0`. Wired into the
  `MILA_ENABLE_CUDA` block of `Tests/CMakeLists.txt`. Covers all
  three defects above. `Specifications/ModelSerialization.md` carries the design; what landed:
  **Phase 0** — `Component::requireSerializableParameters()` (virtual, public) throws when a component
  reports `parameterCount() > 0` and names none, called by both save traversals so the failure names the
  component instead of surfacing as a short archive; `CompositeComponent` overrides it to a no-op, since
  a composite sums its children's parameters but names none of its own and the recursion checks each
  child in turn. The six legitimately-parameterless `save_` bodies now take unnamed parameters and say
  why they are empty.
  **Phase 1** — `CompositeComponent::save_` pushes a `ScopedScope` per child, and its four metadata keys
  moved from `addMetadata()` (archive-global) to `writeMetadata( "meta.json" )` (scoped). It now iterates
  `child_components_` (vector, insertion order) rather than `child_component_map_` (unordered) — **the
  map made both the recorded child order and the write order vary between runs**, which would have made
  archives non-reproducible; the two containers hold the same children under the same names.
  **Phase 2** — `getParameterNames()` implemented on all five parameter-owning components, matching the
  vocabulary `loadParameter` already accepts (`weight`/`bias`, `wte`, `wte`+`wpe`).
  **Phase 3** — new protected `Component::saveParameterToArchive()`, the save counterpart to
  `loadParameterFromBlob`: it stages a device parameter through a host tensor of the **same** dtype and
  sizes the blob from `getStorageSize()`. The four missing `save_` bodies are written against it, and
  `Linear::save_` was rewritten onto it — the old body was deleted rather than retired, since keeping a
  dead copy of a truncating write helps nobody.
  **One design call worth flagging: `Linear` and `TokenEmbedding` now refuse outright on the quantized
  path** (`if constexpr ( kIsQuantized ) throw`). A quantized weight is packed storage plus a scale
  companion and the archive cannot express that pairing; quantization is applied on load for inference,
  and a checkpoint is written from the unquantized training path. Better an explicit refusal naming the
  dtype than a blob nothing can read back.
  **First build round, and it corrected the fix rather than the diagnosis (C7602 at `Component.ixx:866`):
  same-dtype staging cannot use `CpuMemoryResource`.** `isValidTensor` rejects a host-only memory
  resource for any `is_device_only` dtype (`TensorDataTypeTraits.ixx:331`), and **every reduced
  precision Mila trains or serves in is device-only** — BF16, FP16, FP8_E4M3/E5M2, FP4_E2M1/E3M0 all
  set `is_device_only = true`; only FP32 and the integer types are host-compatible. So
  `Tensor<BF16, CpuMemoryResource>` is not a valid template-id, which is **why the original code widened
  to FP32** — that part was not arbitrary, only the byte count paired with it was wrong. Fixed by
  staging through `CudaPinnedMemoryResource`, which is both host- and device-accessible and therefore
  satisfies the constraint while staying readable on the host (`Tensor.Constructors.Cuda.cpp:145`
  static_asserts `isValidTensor<FP8_E5M2, CudaPinnedMemoryResource>`). It is also the same staging
  memory the load direction already uses in `PretrainedReader`. Constructed with
  `parameter.getDeviceId()`, not `Device::Cpu()`.
  **Verified by reading, not building:** `TensorOps::copy` self-synchronizes on the D2H path
  (`CudaTensorOps.Transfer.ixx:244` sets `needs_sync = true` with the comment "Always sync D2H
  transfers"), so the staging copy completes before `writeTensorBlob` reads the host buffer — that was
  the one silent-corruption risk in the new helper. **Expect build rounds:** four files gained
  `import Serialization.Metadata;` (only `Linear.ixx` had it), and `Component.ixx` gained
  `import Compute.CpuMemoryResource;`. `copy()` resolves through ADL at instantiation exactly as the
  existing `copyFromBlob` call in the same file does. Phases 4-7 (load traversal, model API, optimizer
  state, quantized distribution artifact) are untouched.

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
- [x] **C4100 sweep in the op layer finished — 45 of the 47 remaining sites cleared, 13 files.** Done
  the way the previous attempt should have been: from the build's own file+line list, one site at a
  time, **not with a regex** (a pattern matching `<tokens> <name> ( ... ) {` also matches
  `if ( status != cudaSuccess ) {`, so it comments out tokens inside conditions and mangles default
  arguments (`= nullptr` -> `= /*nullptr*/`); that attempt corrupted 112 sites across 43 files and was
  reverted). Same census result as the dispatch layer: nearly every site is an unimplemented stub
  (`CudaSoftmaxOp` / `CudaSoftmaxCrossEntropyOp` half+float backward, four `CudaGqa.Dispatch` throw-only
  stubs), a no-op virtual hook (`Component::loadParameter` / `onBuilding` / `onTrainingModeChanging`,
  `Rope::onTrainingModeChanging`, `CpuGeluOp::build`, `SwigluConfig::fromMetadata`), or a parameter
  that is dead by the math (`alignment` in the three CUDA memory resources — CUDA over-aligns;
  `input` in `CpuAttentionOp::backward` — forward's cached permutation carries it; `input_grad` in
  `CudaTokenEmbeddingOp::backward` — token ids are non-differentiable). `input_A`/`input_B` in
  `CudaResidualOp::backward` were held back pending their contract question, and unnamed once that
  question was answered (above).
- [x] **nvcc `#177-D` cleared (15 rows, 7 sites).** The 5 `CudaW4A16Gemm.Wmma.cu` symbols
  (`kWarpTilesM/N`, `kKSubTiles`, `fp4_e2m1_decode`, `loadTileAsync`) are **not** unreferenced because
  the TU is parked — a `__global__` is compiled and emitted whether or not anything launches it, and
  the file's other constants (`kBlockM`, `kBlockThreads`) never warn because the host-side launcher
  reads them. These five are referenced **only inside the kernel's `#if __CUDA_ARCH__ >= 800` guard**
  (BF16 WMMA fragments need SM80), so every sub-SM80 compilation pass sees them unreferenced. That is
  a conditional-use warning, independent of the parking, and `[[maybe_unused]]` states the condition at
  the declaration. (The parking itself is real and unchanged: `kUseFusedFp4Gemm = false` at
  `CudaLinearOp.ixx:142`, Stage 2 kept in-tree as the cp.async foundation.) The 10 `CudaAdamW.cu` rows are one
  pair of constants times 5 arch passes: `kNumParamsToPrint` pairs with the commented-out debug print
  (same strip-vs-gate decision as the surviving `printf`s, tracked above), and `kAdaptiveLRAbsLimit`
  is a **missing check** — the sequence runs Check 1,2,3,4,6 with no Check 5, so the adaptive update
  magnitude `m / (sqrtf( v ) + eps)` is computed and never bounds-checked. Left unenforced deliberately
  (behaviour change on a training path under an open decision); noted at the site.
- [x] **Final 12 warnings: 10 cleared, 2 left deliberately — the warning burn-down closes at 252 -> 2.**
  The 4 C4267 rows inside `<optional>`/`<xutility>` were **not** std noise: all four are the assign path
  and the construct path of one `std::optional<int>` (`GenerateParams.ixx:31`) being handed a `size_t` at
  `Chat.ixx:710` — the single assignment site of seven that lacked the `static_cast<int>` its siblings in
  `ProfileModel`, the Python bindings and the Gemma parity test all have. Also fixed: `CpuAttentionOp.ixx:105`
  implicit `dim_t`->`int` (the three lines above it cast explicitly); `CudaResidualOp.Dispatch.ixx:36` took
  `size_t N` while its `nv_bfloat16` sibling and its own `forward` took `int` — now `int`, per the rule that
  the `*.Dispatch` layer stays `int` (the caller already passed `static_cast<int>`); two C4456 shadows that
  were pure duplicates of an enclosing local (`Llama.ixx:471` re-declared `B = input_shape[0]` over the same
  value at `:422`, `TokenEmbedding.ixx:455` re-declared `device` over `:431`) — both deleted, not renamed;
  and `ZipSerializer.ixx:272` passed `MZ_DEFAULT_COMPRESSION` (-1, signed enum) into an `mz_uint`
  `level_and_flags`, round-tripping through `0xFFFFFFFF` and relying on miniz casting back to int. Verified
  against the fetched source: `miniz.h:275` documents `MZ_DEFAULT_COMPRESSION=MZ_DEFAULT_LEVEL` and
  `miniz_zip.c:3283` maps negatives to `MZ_DEFAULT_LEVEL`, so passing the level directly is the same
  setting with no sign round-trip. The last 2 (`CudaResidualOp` `input_A`/`input_B`) closed with the
  contract question above, taking first-party C4100 to zero. **The one warning left in the tree is
  `GroupedQueryAttention.ixx:216` C4702 — deliberate, an honest report that GQA backward is a stub.**
- [x] **`CpuAttentionOp::build` divided before it validated — fixed 2026-07-28, and it is hardening,
  not the live crash this entry previously claimed.** `:106` computed `HS_ = embedding_dim_ / NH_` and
  *then* `:108` checked `embedding_dim_ % NH_ != 0`. Both expressions divide by `NH_`, so a zero head
  count would have faulted inside the guard meant to catch it. The check now precedes the division and
  also rejects `NH_ <= 0`.
  **Correcting the original entry, which overstated the risk: this was unreachable, not latent-live.**
  `config_` is a `MultiHeadAttentionConfig`, the constructor calls `config_.validate()`, and that
  rejects `num_heads < 2` and `model_dim % num_heads != 0`; `validateInputShape` separately pins
  `qkv_dim == 3 * model_dim`, so `embedding_dim_ == model_dim` exactly and no truncation was possible
  either. The op-level guard is therefore redundant with the config — kept and ordered correctly
  because it costs nothing and the op should not depend on an invariant it does not state, but it is
  not defending against anything reachable today. No test added: the enforcement point is already
  covered by `MultiHeadAttentionConfig.cpp` (`Validate_ThrowsForTooFewHeads`,
  `Validate_ThrowsWhenModelDimNotDivisibleByHeads`), and a test at the op level could only re-assert
  what the config already rejects. **Relevant to the API Coherence "redundant defensive checks"
  group under `## Future`** — this is an instance of that question, not a defect independent of it.
- [ ] `CudaManagedMemoryResource.ixx:85` builds a detailed `errorMsg` on `cudaMallocManaged` failure
  then throws a bare `std::bad_alloc()`, discarding it — the diagnostic never reaches the caller
  (and the dead local is itself a C4189). `CudaPinnedMemoryResource.ixx:101` throws bare `std::bad_alloc`
  with no message at all. `CudaDeviceMemoryResource` gets this right: `throw CudaBadAlloc( errorMsg )`.
  Align the other two on `CudaBadAlloc` so an OOM says which device, which size, and which resource.
- [ ] `GroupedQueryAttention.ixx:216` C4702 left deliberately — **the one warning remaining in the
  tree.** The `return` is unreachable because `CudaGqaOp::backward` unconditionally throws
  (`CudaGqaOp.ixx:334`). Honest reporting of a known-aspirational path: it self-clears when the GQA
  training path is built, whereas a suppression would have to be remembered and removed, and the only
  mechanisms available are both worse — `#pragma warning` in module code is ruled out by the ratchet's
  constraint (e), and a per-file CMake suppression would blind a file that is mostly live inference
  logic. Same root cause as the GQA standalone-`forward()` stub above. **Note for the warnings-as-errors
  decision: a blanket `/WX` would force this one to be silenced to get a green build; escalating only
  the defect-class codes leaves it visible.**

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
- [x] **The quantization policies were public API in practice but absent from the umbrella — fixed
  2026-07-28.** A consumer writing `Linear<Cuda, BF16, PerChannelFp8<>>` — the compile-time
  weight-quantization axis CLAUDE.md documents as a headline design feature — had to
  `import Dnn.Quantization.Weight.Policies` directly, because `Mila.ixx` re-exported neither it nor
  `Dnn.Quantization.KvCache.Policy`. **The governing rule is the one already recorded when the export
  surface was frozen: a type in a public template's interface must be visible, not merely reachable,
  at instantiation.** `TWeightQuantization` is `Linear`'s third template parameter and `TKvPolicy` is
  GQA's, so the omission contradicted that principle rather than expressing it. Both are now
  `export import`ed.
  **Why it hid for so long — it failed asymmetrically, and per compiler.** `Linear<Cuda, BF16>`
  compiled fine, because a *default* template argument only needs its type reachable; only the
  explicit spelling broke. And MSVC surfaced the policies transitively through `import Mila` in any TU
  that instantiated a component, while clang did not (non-export imports are not transitive) — so six
  test files carried an explicit import as a **clang portability workaround**, each with a comment
  saying so. The real export removes the workaround on both compilers; all six were dropped, keeping
  the `Serialization.Tensor` half of two of those comments, which is a separate un-exported module and
  still needed. Surfaced by a build break in the new dispatch contract tests, which name a policy
  without instantiating a component and so hit the gap on MSVC too.
  **Deliberately NOT bundled with the `IExecutionContext` item below.** They look like one question
  but resolve in opposite directions: policies needed *adding*, whereas `IExecutionContext` is
  exported with no consumer path and wants either removal or a real `fromPretrained` overload.
  Widening carries none of the risk measured for narrowing (Production Hardening).
  Remaining, if the umbrella is ever audited as a whole: `Serialization.Tensor` and
  `Compute.ExecutionContext` are both imported directly by consumer tests for the same class of
  reason. **The audit needs to separate two cases that look alike from the outside.** A backend
  module being absent is correct: `Compute.CudaAdamWOptimizer` is not exported, and should not be —
  `Dnn.Optimizers.AdamW`'s device-agnostic `AdamWOptimizer<TDeviceType, TPrecision>` wrapper is the
  public entry and takes `IExecutionContext*`, so a consumer never names the CUDA type. The policies
  were the other case: a type appearing *in a public template's interface*, which must be visible.
  The test for which case applies is whether a consumer has to spell the name to use the documented
  API, not whether the name happens to be needed by some in-tree file.
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
- [x] MIS `top_p` dropped before the sampler — closed when the binding was wired for it; verified
  2026-07-28 end to end (`protocols/*` → `routes/factory.py` → `model_worker` → `Mila_py.cpp` →
  `params.sampling.top_p`).
- [ ] Refine: buffer Gemma Anthropic streaming only when tools are present.
- [x] **CUDA DLL registration hoisted out of `main.py`** — found 2026-07-28 while validating the
  Python samples, fixed the same day. Since Python 3.8, **PATH is not searched when resolving an
  extension module's DLL dependencies** — only system directories, the extension's own directory, and
  directories registered with `os.add_dll_directory`. The binding links `cublasLt64_13.dll` and
  `curand64_10.dll` from the CUDA Toolkit, so a machine with CUDA correctly on PATH still fails with a
  bare `DLL load failed while importing mila`. `main.py` handled this inline, which left three gaps:
  every other entry point into the binding (`model_worker.py`, `routes/chat.py`,
  `routes/completions.py`, `routes/factory.py`) failed when imported without `main` — including the
  README's own verification step, `python -c "import mila"`, which did not work; only `bin\x64` was
  registered, so a pre-CUDA-13 layout (DLLs directly in `bin`) still failed; and a stale `CUDA_PATH`
  raised `FileNotFoundError` out of `os.add_dll_directory` rather than falling back. Now a
  dependency-free `cuda_runtime.py` imported ahead of `mila` in all five modules: registers one
  toolkit (`CUDA_PATH` if real, else the newest installed), both layouts, skipping what does not
  exist, and exposes `CUDA_DLL_DIRECTORIES` for diagnosis. README verification step corrected.
  Verified against the MIS venv with `CUDA_PATH` set, unset, and stale; 31 MIS tests green.
- [x] Neutral binding output location — `MilaPy` now publishes to `<build dir>/python/`, a directory
  holding nothing but the extension, so any consumer can put it on `sys.path` without dragging that
  consumer's sources along. The MIS convenience copy is retained (its run instructions depend on it),
  but it is no longer the only place the extension lands.
- [x] Python binding defect repair — `LlamaModel.from_pretrained(quantize_fp8=True)` was accepted and
  silently ignored (`(void)quantize_fp8;`); it now applies `WeightQuantization::FP8` to the weights
  (the KV cache stays uncompressed — `withFP8Quantization()` would also request an FP8 KV cache, which
  `LlamaModel::fromPretrained` does not implement). The stale module docstring ("Llama 3.2 3B Instruct
  on CUDA BF16", the first thing `help(mila)` prints, while Gemma is bound and is the flagship) is
  rewritten. Verified on the built extension 2026-07-28, Llama 3.2 3B Instruct on the 4070:
  **54.1 tok/s BF16 -> 78.2 tok/s FP8** (+45%), output coherent, `buildCublasLtPlans` logging the FP8
  dequant plan per projection.
- [~] **`mila-llm` wheel (PythonBinding.md Tier 3)** — started 2026-07-28 by explicit direction,
  overriding the spec's "post-v0.20" deferral: a pip-installable binding is what the Python-first
  audience actually needs. Landed: the extension renamed to `mila._mila` behind a `mila/__init__.py`
  that registers the CUDA DLL directories before the extension loads (the only place that can run
  early enough — this single-sources the fix for samples, MIS and wheel alike); `Bindings/Package/`
  with `pyproject.toml` + a `setup.py` whose sole job is `has_ext_modules()` so the wheel gets a real
  `cp313-cp313-win_amd64` tag instead of `py3-none-any`; CMake staging the extension into the package
  source tree, the neutral build directory, and MIS, all three now as package DIRECTORIES.
  CUDA ships as `nvidia-cublas` + `nvidia-curand` dependencies so no Toolkit is required — note the
  `-cu13` spellings are deprecated stubs and both live packages do publish win_amd64 wheels.
  **Verified end to end 2026-07-28** on the rebuilt extension: `mila_llm-0.20.0b2-cp313-cp313-win_amd64.whl`
  builds, installs into a clean venv, imports, and runs a sample (Llama 3.2 3B, 54.1 tok/s) against
  the *installed* wheel rather than the build tree. MIS survived the rename — `import mila` resolves
  to the staged package, 31 tests green. Stale flat `mila.*.pyd`/`.so` artifacts removed from the
  Server and build directories.
  **The verification earned its keep: NVIDIA's CUDA 13 wheels lay their DLLs out at
  `nvidia/cu13/bin/x86_64`, not the `nvidia/<library>/bin` of the CUDA 12 wheels.** The first
  implementation globbed `nvidia/*/bin`, which matched a directory containing only the `x86_64`
  subdirectory and no DLLs — and then returned early, skipping the working Toolkit fallback. Now it
  finds directories by searching for `*.dll` (survives the next reorganisation and the aarch64 split)
  and registers the Toolkit *as well as* the wheels, so a partial wheel install degrades instead of
  failing.
  **TestPyPI dry run done 2026-07-28** — `mila-llm` 0.20.0b2 uploaded, metadata correct (markdown
  README, MIT expression, `requires-python`, all three URLs), and installed from the index into a
  clean venv, which pulls `nvidia-cublas` 13.6.0.2 (394 MB), `nvidia-curand` 10.4.3.29 and the
  transitive `nvidia-cuda-nvrtc` 13.3.33, then runs a model.
  **Second bug the verification caught, and it inverted the design comment:** registering both the
  wheel directories and the Toolkit is NOT enough to make the wheel's copies win —
  `os.add_dll_directory` searches added directories in an *unspecified* order, and measurement
  (`GetModuleFileNameW` on the live process) showed `cublasLt64_13.dll` resolving to the machine's
  CUDA v13.3 rather than site-packages. A wheel install silently binding to whatever toolkit is
  present is what the dependency pins exist to prevent. Fixed by *loading* the wheel's DLLs at import
  rather than merely making them findable — Windows resolves dependencies by base name against what
  is already in the process, the same approach PyTorch takes. Re-measured: both now resolve to
  site-packages, and generation runs on cuBLAS 13.6 rather than the Toolkit's 13.3.
  **PUBLISHED to PyPI 2026-07-29** as `mila-llm` 0.20.0b2.dev20. Version mapping settled:
  `Version.txt` `0.20.0-beta.2+N` -> `0.20.0b2.devN`, **not** plain `0.20.0b2` — that number belongs
  to the upcoming beta.2 *release*, and a published version can never be reused, so a snapshot taken
  under it would have forced the real release to ship as a post-release of a snapshot. PEP 541 claim
  on the derelict bare `mila` filed as pypi/support#11675.
  **Remaining:** the wheel version is still hand-maintained in `pyproject.toml` *and* `Version.txt`
  and will drift at the next bump — derive it at build time; the Linux CUDA preload path is written
  but has never been exercised (needs a WSL build); the manylinux glibc floor is undecided (CI's
  Ubuntu 26.04 yields ~`manylinux_2_43`, which reaches almost nobody, and the standard
  `manylinux_2_28` image ships GCC 12 and cannot compile C++23 modules); `auditwheel` must be given
  an explicit `--exclude` for the CUDA libraries or it will vendor 400 MB of cuBLAS into the wheel
  and defeat the dependency design. Also open: whether the samples ship inside the wheel as
  `mila-chat` / `mila-generate` console scripts, Trusted Publishing from CI, and the cp314 matrix.
  Naming rationale and the multi-backend argument are in the spec.
- [ ] **Tier 2 — weights, the real product gap.** `pip install mila-llm` today gives a runtime with
  nothing to run: no weights ship, none download, and producing a `.bin` means cloning the repo and
  running the torch/transformers converters. **Settled 2026-07-29 (details in the spec): Mila hosts
  nothing large** — `google/gemma-4-12B-it` is `gated: false` + Apache 2.0, so the user fetches from
  Google over `urllib` and converts locally, which costs no storage and incurs no redistribution
  obligations. **Gemma 4 E2B/E4B are ruled out** as a smaller first-run candidate: they carry
  Per-Layer Embeddings on every layer plus cross-layer KV sharing, a different architecture rather
  than a smaller config. The `mila-llm` Hub organisation exists for tokenizers and the org card.
  The work is a **torch-free safetensors -> Mila converter**: feasible on the standard library alone
  (header is a u64 length + JSON; BF16 is a byte copy; the fused `[Q|K|V]` and `[gate|up]` are dim-0
  concatenations; `v_norm` is synthesised ones; `layer_scalar` is one BF16->FP32 widen).
- [x] Python samples — `Mila/Samples/Python/`: `chat.py` (Gemma 4 streaming chat: instruct template,
  token loop, channel filter, cooperative Ctrl-C through `StopController`), `generate.py` (tokenizer
  round-trip + sampling knobs, Gemma or Llama, `--fp8` exercising the defect fix above), `common.py`
  (extension + weight discovery, with an ABI-tag mismatch diagnostic), and a README stating what the
  binding does **and does not** expose. Standard library only — no `requirements.txt`. Tier 1 of
  `Specifications/PythonBinding.md`; Tiers 2 and 3 remain in `## Future`.

---

## Future

Uncommitted / next-cycle work. Coarse by design — detailed tasking happens only when an item promotes
to the current release.

- **Warnings-as-errors ratchet** — prevent the 252-warning re-accumulation that the v0.20 sweep cleared.
  Constraints worth keeping, decided 2026-07-27: **(a)** requires the `/external:W0` isolation above first;
  **(b)** enforce in **CI only**, never locally — the existing `cpu-only-tests` anti-rot job is the same
  pattern, and a local `/WX` that blocks bisecting gets quietly disabled within months; **(c)** ratchet on
  **warning count not increasing** before demanding zero, so a compiler update raises the ceiling
  deliberately instead of felling the build; **(d)** **MSVC first** — `/WX` across MSVC + Clang + GCC means
  the union of three compilers' opinions must be zero, and cross-compiler builds already surface real
  portability deltas; **(e)** dormant-but-retained code (GQA training substrate, AdamW debug constants)
  warns *by nature* and the warnings usefully mark it — suppress per-file in CMake pointing at the owning
  task, **not** with `#pragma warning` inside module code. Where the unreferenced symbol is *conditionally*
  used rather than dormant, `[[maybe_unused]]` at the declaration is better than either: it is local,
  standard, and states the condition at the site. **Land the blocking gate
  after v0.20 ships**: it fits the freeze as hardening, but a newly-added gate that can fail the build does
  not belong on the runway to a first production release.

- **Qwen 3** (presumptive next release) — the dense decoder, thinking-mode suppression, model-agnostic
  tool calling, and FP8 KV cache (`PerChannelKvFp8<>`); the `OperationTraits<GqaOp, Cuda, BF16,
  PerChannelKvFp8<>>` specialization lands here.
- **v0.20 feature-frozen tails** — the Generation API surface tail (`SamplerConfig` rename, Llama/Gpt
  seedable sampling, eager sampler, config-accessor propagation, `contextLength()` hoist), the
  Sample-API device-sampler migration for Llama/Gpt, the Optimizer-dispatch migration onto
  `OperationTraits`, and the unspecced **Chat** feature milestone.
- **Ministral** — SWA transformer; reuses the Llama foundation, Qwen 3 tool-calling, and the Gemma 4
  SWA mask + bounded-KV ring.
- **The gradient-write contract: assign vs accumulate** — moved here from Test Suite Revival
  2026-07-28 after investigation showed it is **not a defect**. It was filed as "`Residual::backward`
  zeroes both buffers citing accumulation — true of CPU, false of CUDA; pick one", which read as a
  latent bug. It is not: every gradient buffer has exactly **one producer** and is pre-zeroed, so
  assign and accumulate-into-zero are identical. Where the residual stream forks, the summation is
  **explicit in the owning block** — `GptBlock::backward` does
  `zero( d_res1_accum_ ); add( d_res1_from_res2, d_res1_from_ln2, ... )` and the same for `d_input_`.
  Bard converging to coherent Shakespeare is the end-to-end evidence.
  What is real is an inconsistency worth settling once, pre-1.0. **Four of the six components
  carrying the comment "backend ops use accumulation (atomicAdd/+=) which requires pre-zeroed
  buffers" were wrong about their own backend**, in two different ways, and all four were corrected
  in place 2026-07-28:
  *The CUDA op assigns, so the zero is redundant there and needed only for the CPU op* — `Residual`
  (`dA[idx] = grad`), `Gelu` (`dX[i] = local_grad * dY[i]`), and `Linear`'s input gradient
  (cuBLASLt `beta = 0.0`).
  *The op never writes the buffer at all* — `Lpe`, whose input is token indices:
  `CudaLpeOp::backward` documents `input_grad` as "Unused (non-differentiable input)" and never
  touches it, so **the zero is the only thing standing between the caller and uninitialized memory**,
  which is the opposite of the stated reason. Its `atomicAdd` is on `wte_grad_`/`wpe_grad_`, the
  parameter gradients.
  Left alone as accurate: `LayerNorm` (its CUDA kernel really does `atomicAdd( &dx[...] )` on the
  input gradient, lines 202/364) and `TokenEmbedding`, whose comment is the model for the rest — it
  scopes the `atomicAdd` claim to `wte_grad` and states plainly that the input gradient is a
  non-differentiable formality.
  **The framing that resolves it — two kinds of gradient buffer, which want opposite contracts:**
  *parameter* gradients (weight, bias) must accumulate across backward calls, since that is what makes
  gradient accumulation over micro-batches work, and are cleared by `zeroGradients()` between
  optimizer steps; *input/activation* gradients are per-call, single-producer, and can simply be
  assigned. `Linear` already implements exactly this split and is the reference. Deciding it project-
  wide means either (a) every op accumulates and components pre-zero — uniform, keeps the
  micro-batch option open, costs a zeroing pass plus a read-modify-write — or (b) ops declare which
  they do and components zero only for accumulating ops — faster, needs the contract stated per op.
  Today the tree states (a) and half-implements (b), which pays (a)'s cost without its uniformity.
  Also worth folding in: the `zero()` on a full-overwrite CUDA op is dead work (two extra kernel
  launches per Residual backward, ~24 per GPT-2 step).
- **safetensors slice 1 DONE and GREEN 2026-07-29 — the container, both directions.** Clean rebuild
  and full ctest green. **One build round, cost by the known MSVC C2079 `basic_istream::sentry`
  defect**: `SafeTensors.Cpu.cpp` was the first `.cpp` in the tree ever to instantiate
  `readTensorBlob<MR>`, which dragged `istream::seekg` into a consumer TU. Fixed by converting
  `PretrainedReader.ixx` off `std::ifstream` to C stdio (`unique_ptr<FILE, int(*)(FILE*)>`, a
  `readExact()` helper that now names the field it failed on, and a chunked `seekToOffset()` stepping
  1 GiB at a time because `fseek`'s `long` is 32-bit on Win64) — the MnistDataLoader /
  TokenSequenceLoader precedent. `SafeTensors.ixx` and the test fixtures were converted off stream
  I/O pre-emptively for the same reason. **Durable lesson: the C2079 trigger is a TEMPLATE member
  doing stream I/O in a module — it stays invisible until some `.cpp` instantiates it, so a module can
  sit green for months with the trap armed.**
  **MILA path re-confirmed end to end 2026-07-29: Chat on Gemma 4 FP4 green and coherent** after the
  C stdio rewrite, which is the only oracle that covers a real 22 GB load through mmap plus pinned
  staging (no ctest does).
- **safetensors slice 2 DONE and GREEN 2026-07-29 — the component-level quantized save.** Clean
  rebuild, full ctest, and Chat on Gemma 4 FP4 all green; no build round needed.
  `Component::saveParameterToWriter()` is the safetensors twin of `saveParameterToArchive`: same
  pinned-staging reasoning (FP8/FP4/BF16 are all `is_device_only`, so
  `Tensor<TPrecision, CpuMemoryResource>` is not a valid template-id), with the staging tensor scoped
  to a single write so only one parameter is resident. `Linear::saveFlatTensors( writer, prefix, pass )`
  emits `<prefix>.weight`, `<prefix>.weight.scales` on the quantized path, and `<prefix>.bias`.
  **One ordered body driven twice via `TensorSavePass`**, not two walks — the writer requires bodies in
  declaration order and two walks could drift with no diagnostic until read-back.
  **The scales stay OUT of `getParameterNames()` deliberately.** That vector is the join between the
  archive's `save_` and `load_`, and the `blob_count == getParameters().size()` invariant the Phase 4-5
  tests rest on would break. The flat path expresses the pairing as sibling tensor names instead, which
  is the ecosystem convention and what the reader already handles.
  Three tests added **in place** to `Linear.Cuda.cpp`: FP8 per-channel (structural, plus a host
  `w ~= float(fp8) * scale[row]` dequantization check — the only assertion that catches scales landing
  against the wrong rows, which passes every structural check), FP4 per-group (physical halved column
  count, `[out, K/128]` scales), and the unquantized path (weight + bias, no scales).
  **SCOPE NOTE — the whole-model dump tool is NOT here and is slice 3.** Two gaps to close, not the
  three first recorded: (1) **no writer for `PretrainedMetadata`** — there is a hand-rolled
  `parseMetadataJSON` and no inverse, so an artifact can be inspectable but not loadable; (2) the
  traversal needs root-prefix stripping to match `.bin` flat naming (`findComponent` strips it on the
  way in; nothing does on the way out).
  **CORRECTED 2026-07-29 — the third "gap" was not one.** It was recorded as *no runtime
  host-accessibility query* (`MemoryResource::is_host_accessible` is `static constexpr` at
  `MemoryResource.ixx:37`, not virtual, so a type-erased walk over `ITensor*` cannot decide whether to
  stage). That only binds if the traversal is type-erased, and it need not be: promote
  `saveFlatTensors` to a **virtual on `Component`** and let each of the five parameter-owning
  components implement it with concrete types, exactly as `save_` already does. `Linear`'s
  implementation is written and green. **No addition to any exported core type is required** — do not
  add a virtual to `MemoryResource` for this.
  So slice 3 is: the virtual, four more implementations (`RmsNorm`, `LayerNorm`, `TokenEmbedding`,
  `Lpe`), composite recursion building the dotted prefix, root-prefix stripping, the metadata writer,
  and the tool.
- **safetensors slice 3 PART A WRITTEN 2026-07-29, UNBUILT — the traversal.** `Component::saveFlatTensors`
  is now virtual with a default that **throws when `parameterCount() > 0`** rather than writing
  nothing (Phase 0's lesson: a component that contributes silently produces an artifact that loads,
  runs, and generates garbage). Implemented on all five parameter-owning leaves — `Linear` (weight,
  weight.scales, bias), `RmsNorm`/`LayerNorm` (weight, bias), `TokenEmbedding` (wte, wte.scales),
  `Lpe` (wte, wpe). `CompositeComponent::saveFlatTensors` recurses over `child_components_` (ordered —
  the map is unordered and the writer demands declaration order) and builds the dotted prefix via a
  new `childFlatPrefix()`: children carry fully qualified names, so stripping the parent's name yields
  the relative segment, and calling the root with an **empty prefix drops the model name**, producing
  `tf_layer_0.qkv_proj` exactly as the converter names it. Placed **public** on the composite,
  deliberately not repeating the `save_` public-on-Component / protected-on-Composite asymmetry
  already filed as a defect. `toMetadataJSON()` added to `PretrainedReader.ixx` — the inverse
  `parseMetadataJSON` never had, closing the "inspectable but not loadable" gap. Two CPU tests added:
  full 27-field metadata write/read cycle, and the container cases from slice 1.
  **Part B WRITTEN 2026-07-29, UNBUILT — the tool.** `GemmaModel::saveArtifact( path )` drives the
  two passes over `getLanguageNetwork()` with an **empty root prefix**, writing whatever precision the
  weights currently sit at — so a model loaded FP4 emits a pre-quantized artifact at roughly a third
  the BF16 size. The source artifact's `PretrainedMetadata` is now **retained on the model**
  (`source_metadata_`, threaded through the private constructor) and written back verbatim rather than
  reconstructed from `GemmaConfig`, which would be free to drift. A `mila_quantization` metadata key
  records the policy (`per_group_fp4_128` / `per_channel_fp8_e4m3` / `none`) so the load side can
  refuse an artifact whose packing disagrees with the build's compile-time `TWeightQuantization`.
  New CUDA-only target `Mila/Tools/ExportArtifact` — work confined to `ExportArtifact.ixx` with a thin
  `.cpp`, the ProfileModel arrangement, because a plain `.cpp` instantiating the model-load templates
  trips the C2079. It reopens what it wrote before reporting success, since a header that disagrees
  with its data region produces a file that looks finished and fails at load.
  **DEFECT IN SHIPPED SLICE 2, FOUND AND FIXED 2026-07-29 BEFORE THE LOAD SIDE WAS WRITTEN: the
  scales name was unloadable.** `parseParameterPath()` (`Gemma.ixx:956`, mirrored in Llama and
  GptTransformer) splits a flat name on its **last dot**, so the emitted
  `tf_layer_0.qkv_proj.weight.scales` resolved to a component named `tf_layer_0.qkv_proj.weight`,
  which does not exist — the artifact could be written and inspected but never read back. Renamed to
  **`weight_scale`** and **`wte_scale`** (underscore), which splits correctly *and* matches the
  compressed-tensors spelling, so the name is conventional as well as loadable. Tests updated.
  **This is why the writer was verified against the reader's own splitting rule before the load path
  was built on top of it** — every structural test passed, and numpy would have opened the file
  happily, so nothing would have surfaced this until slice 4 failed with a confusing component-lookup
  error.
  **Runtime defect found by RUNNING the tool, fixed 2026-07-29 (needs a rebuild):** `runExport` never
  called `Mila::initialize()`, so the export died before reading a byte with "log call before
  initializeLogger()" — the model load path logs and the logger is not implicitly created. Both
  existing entry points do it (`ProfileModel.ixx:645`, `Chat/Src/main.cpp:265`); a new one has no
  compiler or test that notices. Textbook compile-hides-link-hides-runtime: the target built green and
  passed every ctest.
  **Export defect found by building the tool, fixed 2026-07-29.** `GemmaModelConfig`,
  `LlamaModelConfig` and `GptModelConfig` each `import Dnn.LanguageModelConfig;` without
  re-exporting, yet their own public setters take `WeightQuantization` and `KvCacheCompression` from
  it — so any consumer outside the library could not name the types its API requires
  (`error C3646: unknown override specifier`). Changed to `export import` in all three. Same class as
  the `Serialization.Tensor` note in `Mila.ixx`: a module whose public interface names another
  module's types must re-export it. **Only the tool caught this because the tool is the first
  out-of-library consumer of those configs** — Chat builds its own config path, and every in-tree
  caller imports `Dnn.LanguageModelConfig` directly.
  **NOT DONE — the load side.** Nothing yet consumes `mila_quantization`: `Linear.ixx:477` still
  always calls `operation_->quantize()`, so a pre-quantized artifact would be re-quantized as if it
  were BF16. **The tool's output is therefore write-and-inspect only until slice 4.**
  **TWO DEFECTS FOUND BY RUNNING THE TOOL AND DIFFING THE ARTIFACT AGAINST THE SOURCE INDEX
  (2026-07-29). The export completed, self-verified, and reported success with both present.**
  First run: 23.8 GB source -> 7.27 GB FP4 artifact, 725 tensors, header 8-byte aligned, spans
  contiguous and covering EOF exactly, `mila_config` + `mila_quantization` intact. Structurally
  perfect, and wrong in two ways no structural check can see:
  **(1) All 48 `tf_layer_N.layer_scalar` tensors were silently dropped.** This CORRECTS the earlier
  claim here, which said their absence was fine because `layer_scalar` was archive-only and "the
  converter never writes it to `.bin`". **It does.** A direct index dump of the source shows 578
  tensors including `tf_layer_{0..47}.layer_scalar`, so it is squarely in the flat vocabulary. This is
  exactly the silent omission the base-class throw was added to prevent, bypassed because
  `CompositeComponent::saveFlatTensors` overrides that default and cannot distinguish its own
  parameters from its children's via `parameterCount()`. `Gemma.Block` is the one composite in the
  tree that owns a parameter and must override and extend the recursion.
  **(2) The tied `lm_head.weight` is written as a 0.94 GB byte-identical duplicate of `temb.wte`**
  (verified identical over both the first and last 1 MB) -- 13% of the artifact, and absent from the
  source by design since the head is tied at load time. The traversal walks the LIVE component tree,
  where the tied head and the embedding hold the same tensor through a shared_ptr, so both emit it.
  Loading it back would defeat tying and re-allocate the ~1 GB the weight-tying memory gate exists to
  save. `Linear` already knows this case as `weight_installed_`.
  **BOTH FIXED AND VERIFIED GREEN 2026-07-29.** `Gemma.Block::saveFlatTensors` calls the base
  recursion then emits its own `layer_scalar`; `Linear::saveFlatTensors` skips a weight when
  `weight_installed_` is set, which is exactly the tied-head flag. Re-export reconciles **exactly**:
  578 source tensors -> 771 artifact tensors (193 scale companions), **nothing missing, nothing
  extra**, 48 `layer_scalar` present as `F32 [1]`, zero `lm_head.*` tensors, `temb.wte` +
  `temb.wte_scale` only. Size fell 7.27 -> **6.33 GB**, the 0.94 GB duplicate gone to the byte; from a
  23.8 GB source that is a **3.76x** reduction.
  **The tool now reconciles against its own source** (`compareAgainstSource`): every source tensor must
  appear in the artifact and nothing may appear beyond `_scale` companions, exit 3 otherwise.
  **Durable lesson: structural self-verification is not verification.** The first export reopened
  cleanly, parsed, tiled its data region contiguously and covered EOF to the byte -- while missing 48
  parameters and carrying a gigabyte of the same table twice. Diffing the OUTPUT against the INPUT is
  what caught it, and that check now lives in the tool.
- **safetensors slice 4 WRITTEN 2026-07-29, UNBUILT — the pre-quantized LOAD side.**
  **The branch needs no flag: the blob's own dtype discriminates.** `Linear::loadParameter("weight")`
  compares `blob.getMetadata().dtype` against `kWeightDtype` — storage dtype means the bytes are
  already packed and the scales arrive as their own tensor, compute precision means quantize-on-load
  as before. Same in `TokenEmbedding` against `kTableDtype`. Feeding packed nibbles through
  `quantize()` would read them as BF16 and yield a model that runs and is wrong, so the two paths must
  never be confused. Pre-quantized weights validate against `weight_->shape()`, not the config shape —
  a packed FP4 weight is physically `[out, in/2]`.
  New `"weight_scale"` / `"wte_scale"` parameter names accepted by `loadParameter`, which is why the
  underscore rename mattered: `parseParameterPath` splits on the last dot and routes them to the right
  component. On an **unquantized** build both **throw** rather than drop the scales and leave the
  weights silently unscaled.
  `PretrainedModelReader::getWeightQuantization()` surfaces the `mila_quantization` key, normalizing
  `"none"` and an absent key to empty so callers have one test for "quantize on load"; every `.bin`
  therefore reports empty and keeps its existing behaviour. `GemmaModel::fromPretrainedImpl` refuses a
  policy mismatch — **the dtype cannot catch this**, since FP4 at group 128 and group 64 are both U8,
  so only the declared string can.
  Tests: three CPU cases for the metadata key (declared / `"none"` / legacy `.bin`), and two CUDA
  cases — a full **quantize-on-load -> export -> import -> re-export** cycle asserting the two
  artifacts match **byte for byte**, and a scales-on-unquantized-build refusal.
  **Weight tying needs no new work**: the artifact omits `lm_head.weight` exactly as the `.bin` does,
  so the existing tying path applies unchanged.
  **VERIFIED END TO END 2026-07-29 by feeding the tool its own artifact.** Re-exporting the 6.33 GB
  pre-quantized FP4 artifact loaded it through the new path and wrote it back: 771 source tensors ->
  771 artifact tensors, **0 scale companions added** (the source already carries them), and the two
  files are **SHA-256 identical** (`d49c6c16...`). Byte identity is the proof the load did not
  re-quantize: a second quantization pass over packed nibbles cannot reproduce the input. The policy
  guard was checked the same way — requesting `--quantization fp8` against the FP4 artifact refuses
  with "is pre-quantized as 'per_group_fp4_128' but this load requested 'per_channel_fp8_e4m3'",
  exit 1, and **writes no file**.
  Phase 7 is functionally complete: Mila writes a pre-quantized safetensors artifact, reads it back
  without re-quantizing, and refuses a mismatched one. **Still outstanding: Chat has not been pointed
  at a `.safetensors` artifact** — the round trip proves the bytes survive, not that the model
  generates coherently from them.
  **Full suite and Chat green 2026-07-29** — but note precisely what Chat proved: its catalog
  hardcodes `gemma/gemma4_12b_it_bf16.bin`, so a coherent session confirms the **`.bin` path is
  unregressed** (the `loadParameter` dtype branch did not disturb quantize-on-load on a live 12B
  model), not that the artifact generates. Added a `gemma-12b-packed` catalog entry pointing at
  `gemma/gemma4_12b_it_fp4.safetensors` so `/model gemma-12b-packed` closes that last gap; array size
  8 -> 9, and every consumer iterates with a range-for so nothing else changes.
  **PHASE 7 COMPLETE AND PROVEN 2026-07-29.** `/model gemma-12b-packed` loads the 6.33 GB
  pre-quantized artifact and Gemma 4 12B answers coherently. Every leg is now closed: writes a
  correct artifact (reconciles exactly against its source, opens in any safetensors reader), reads it
  back without re-quantizing (SHA-256 identical on a full 12B re-export), refuses a mismatched policy
  (exit 1, no file written), leaves the `.bin` path unregressed (full ctest suite plus a coherent
  `gemma-12b` session), and generates from the artifact. The distribution-artifact goal that ruled out
  hosting in PythonBinding.md is met: **23.8 GB -> 6.33 GB, a 3.76x reduction.**
  Also found, not fixed: **`ITensor.ixx:34` documents `rawData()` as protected; it is public**
  (`:147`, `:154`).
  *(original entry)* New
  `Src/Dnn/Serialization/SafeTensors.ixx` (`Serialization.SafeTensors`): dtype naming both ways,
  `storageBytesPerElement`, and a **two-phase streaming `SafeTensorsWriter`** — declare every tensor,
  `beginData()` emits the header, then bodies stream in declaration order. Two-phase is forced by the
  container (the header records byte ranges, so shapes must be known first) and streaming is what lets
  a 22 GB model be written with one staged tensor resident. `PretrainedModelReader` now **sniffs** the
  container: `MILA` magic takes the existing path completely untouched, anything else parses a
  safetensors header into the same `tensor_index_`, rebasing data-relative offsets to absolute so
  `mapFile`/`buildOffsetOrder`/`streamTensorBlobs`/the pinned staging producer never learn which
  container was opened. Wire codes 4-7 added for `UINT8`/`FP8_E4M3`/`FP8_E5M2`/`INT8`; codes 0-3 are
  frozen by every `.bin` on disk. The sniff is sound by construction — the header-length cap
  (128 MB) is below `MAGIC` (0x4D494C41), so a valid safetensors length can never alias it.
  **New test `Dnn/Serialization/SafeTensors.Cpu.cpp` (11 cases), including two that build a MILA file
  byte by byte from the format definition** — the legacy path had no coverage at all because every
  existing test of it needs a converted model this suite does not have, so adding a second container
  was otherwise a silent-regression risk. NEXT: the quantize-and-dump tool and the `Linear` save side
  (slice 2), then the pre-quantized load branch (slice 3).
- **safetensors: DECIDED 2026-07-29 — Phase 7 emits safetensors, not `MILA`.** Full analysis in
  [ModelSerialization.md](Mila/Specifications/ModelSerialization.md#safetensors-compatibility).
  **Goal is inspectability and trust** (a professional tooling need: any reader can verify a Mila file
  without running Mila). **Consumption by `transformers`/vLLM is an explicit NON-GOAL** — recorded so
  it is not rediscovered as a gap. Three capabilities: **(W)** write safetensors — decided, carries no
  schedule of its own since it is a property of the Phase 7 writer; **(R)** read unconverted HF
  snapshots — open, vNext, the larger item; **(C)** checkpoint — rejected, `ModelArchive` stays, since
  step/epoch/RNG have no representation in a string-to-string `__metadata__`.
  **Why W is free rather than a trade.** Phase 7 must build a C++ tensor-file writer regardless —
  **there is none today**: `PretrainedReader.ixx` is read-only, the only `MILA` magic under `Mila/Src`
  is the unrelated dataset header in `Data/Core/FileHeader.ixx`, and the flat format is written
  exclusively by Python `MilaWeightWriter`. Quantized bytes exist only *after* a device-side
  `operation_->quantize()`, so producing them in Python would mean a second absmax/packing
  implementation to keep in agreement. The writer gets built once; the only question is what it emits.
  The container is nearly isomorphic anyway (`u64` header length + JSON index + packed data versus
  `MILA` magic + binary index + packed data), so on the read side `readHeader`/`readMetadata`/
  `readTensorIndex` collapse to one nlohmann parse and the tuned parts — `mapFile`, `buildOffsetOrder`,
  offset-ordered `streamTensorBlobs`, the pinned double-buffer producer — are untouched.
  **Everything it costs, having gone looking:** no format version of its own (goes in `__metadata__`
  and must be validated deliberately — this is Open Decision 3); no padding for alignment (the
  reference validator expects contiguous tiling, so aligned starts for a direct mapped-view
  `cudaMemcpy` are unavailable — costs nothing today, but confirm against the reference validator);
  `__metadata__` is string-to-string, so the config is a stringified blob or a sidecar. No new
  third-party dependency — a writer is a `u64`, an nlohmann dump, and the blobs.
  **Tier 3 findings, recorded so the group-size question is not reopened casually.** Mila fails HF
  loadability on structure before quantization: projections are **fused** (`fc_qkv_proj.weight`,
  `fc_gate_up.weight`) where HF ships them separate, plus Mila names and Mila config schema. On
  quantization the split is **FP8 interoperates, FP4 matches nothing**: Mila FP8 is `FP8_E4M3` + FP32
  per-channel scales, which is substantively compressed-tensors `float-quantized`/`channel`; Mila FP4
  is `UINT8`-packed E2M1 + FP32 scales at **group 128**, against NVFP4 (group 16, FP8 scales), MXFP4
  (group 32, UE8M0 scales) and GPTQ/AWQ (int4 in int32). Matching would reach into the W4A16
  tile-load dequant and the decode matvec and move the measured FP4 numbers — **never make that
  change as a side effect of a format choice.** Tier 3, if ever wanted, is a separate **export** path
  (unfuse, rename, HF `config.json` + `compressed-tensors` block), not a property the distribution
  format acquires by adopting the container.
  **R, when it promotes.** Four obstacles: the fused projections against the one-blob-to-one-parameter
  contract; GPT-2's `Conv1D` transpose; a per-architecture `config.json` mapper in C++ tracking HF key
  drift — **the piece worth pushing back on**, since it compiles a dependency on someone else's
  evolving schema into the artifact strangers read, and is severable by keeping the config in a
  converter-written sidecar; and sharding (22.2 GB Gemma is ~5 GB shards + an index file, against a
  single-file single-mmap reader). Not an obstacle: the numerics already match — Gemma norms stored
  raw with the `+1` at the kernel, no Llama q/k permute. Payoff is parity by construction, retiring
  the converter-bug class the Llama HF-parity test exists to catch — and inbound is the more valuable
  direction regardless, since the Hub already carries thousands of pre-quantized checkpoints and
  reading them means never asking anyone to adopt Mila's artifacts.
- **Serialization Phases 4-5 DONE and verified 2026-07-29** — build green, full ctest green, including
  a real two-layer GPT round trip. Phase 6 (optimizer state) remains unstarted.
  **Phase 6 readiness, checked 2026-07-29: ready, and mechanically the easier of the two remaining
  phases.** All state is FP32 — `m_states_`, `v_states_`, `master_params_` are `Tensor<FP32, MR>`
  (`CudaAdamWOptimizer.ixx:427`, `:428`, `:433`) and `step_count_` is a `size_t` — so FP32 is not
  `is_device_only` and the Phase 3 staging helper works unmodified; the constraint that made Phase 3
  awkward does not bite here. **The one real design risk is that state is keyed by REGISTRATION ORDER,
  not name** — `addParameter( ITensor*, ITensor* )` (`:113`) has no name to key on — so the checkpoint
  must record parameter count plus per-slot shapes and validate on load, or a differently-built graph
  silently mismatches, and mismatched Adam moments produce a run that trains and is wrong. Plumbing:
  the optimizer is not a `Component`, so it cannot reach the staging writer at `Component.ixx:923`/
  `:940`; hoist that into a small new module imported by both (new `.ixx` needs CMake registration).
  Acceptance test is already specified — Phase 5's resume showing no loss discontinuity at the seam.
  **Sequencing call: Phase 7 first.** 7 turns 22.2 GB into ~7 GB and unparks packaging; 6's acceptance
  test is a long-running empirical training run and the training path is not on the v0.20 critical
  path the way distribution is.
  **What the green run actually establishes, and what it does not.** Verified end to end on the **GPT
  stack only**: `Lpe`, `LayerNorm`, `Linear`, `GptBlock`, `MLP`, `GptTransformer` — round trip
  bit-exact, one blob per parameter, config recovered including the two fields the old hand-rolled
  block got wrong, and a throw on a mismatched archive. **The `blob_count == getParameters().size()`
  assertion passing is the real prize**: it is empirical proof that `getParameterNames()` and
  `getParameters()` agree for every component in that stack, which is the Phase 2 criterion for three
  of the five components, obtained as a runtime check rather than a restatement.
  **Gaps closed 2026-07-29 (written, unbuilt):** `Gemma.Block.Cuda.cpp` — `layer_scalar` written to the
  archive and restored, verified *through the archive* since the block exposes no accessor (save with a
  set value, reload into a fresh block whose default is 1.0f, re-save, read back: 2.5 means restored,
  1.0 means `load_` did nothing), plus a blob count proving the sixteen children did not collapse onto
  one scope. `Llama.Cuda.cpp` — full network round trip with one blob per parameter, which is also the
  only coverage of `RmsNorm` and `TokenEmbedding` `getParameterNames()`, unused by the GPT stack.
  **New `Dnn/Models/GptModel.Cpu.cpp`** — Phase 5's public API at last: `fromCheckpoint` reconstructing
  a model from an archive alone, and the full cycle load → `saveCheckpoint` → compare **archives blob
  for blob**. That last oracle exists because `getLanguageNetwork()` is protected and widening the
  public API for a test would be the wrong trade; comparing archives is stronger anyway, since it
  covers both legs.
  **Original gap list, for the record:** `Llama.Block` and `Gemma.Block`
  (both had hand-rolled overrides removed or rewritten, neither has a serialization test);
  `Gemma.Block::layer_scalar` save/load (**net-new code, never executed** — and it is the one composite
  that owns a parameter, so nothing else exercises that path); `RmsNorm` and `TokenEmbedding`
  `getParameterNames()` (the remaining two of the five); and **`GptModel::saveCheckpoint` /
  `fromCheckpoint` themselves** — the tests drive `GptTransformer::save`/`load` and
  `configFromArchive` directly, so Phase 5's actual public deliverable has no oracle. A Llama or Gemma
  CPU round trip plus a `GptModel`-level test would close all of it and is the obvious next coverage.
  *(original entry)*
- **Serialization Phases 4-5 written 2026-07-29.** Todd's call to take 4-6 out of freeze
  scope. Phase 6 (optimizer state) deliberately sequenced *after* a build round rather than written
  alongside — it adds a new module plus CPU and CUDA optimizer surfaces, and an export change is
  already in flight in this batch.
  **Phase 4 — the load traversal.** `Component::load_` is virtual with a **working default**, not pure:
  it walks the same `getParameterNames()` vector `save_` walked, reads each blob, and hands it to
  `loadParameter`, which already validates dtype and shape. That default is the entire implementation
  for all five parameter-owning leaves. A named-but-absent blob **throws** — skipping would leave the
  parameter at its initialized value and report success, which reads as a converged model that is not
  one. `CompositeComponent::load_` recurses under the scopes Phase 1 established; `Network::load()` +
  `loadComponentGraph()` mirror `save()` + `saveComponentGraph()`, iterating the **live** children
  rather than replaying a manifest (`saveComponentGraph` records names into per-child descriptor
  *filenames*, not a list, so the live graph is the only clean enumeration) and validating the recorded
  component count first. `Network::load_` is non-pure where `save_` is pure: a concrete network must
  write its config because nothing else can, but rarely needs to read it back, since the factory
  consumes it before construction.
  **Phase 5 — `GptModel::saveCheckpoint` / `fromCheckpoint`, cheaper than scoped.** `GptConfig`
  already had `toMetadata()`/`fromMetadata()`, so the declared no-config signature is achievable after
  all — `fromCheckpoint(path, device_id)` reads the config, rebuilds at the saved geometry, builds,
  then loads. **This surfaced a live defect in the hand-rolled block it replaces:
  `GptTransformer::save_` wrote `mlp_hidden_dim` where `GptConfig::fromMetadata` reads `hidden_dim`,
  and omitted `use_bias` entirely** — a round trip would have silently rebuilt with a defaulted bias
  flag and a hidden size guessed as 4x embedding (right for GPT-2 by coincidence, wrong in general).
  `save_` now writes `config_.toMetadata()`, one source of truth. `GptTransformer::load_` validates
  vocab/layers/embedding/heads so a mismatch names itself instead of surfacing as a shape error deep
  inside a component.
  **Tests, two layers of them.** In `Core/CompositeComponent.cpp`: round trip into a *separately
  constructed* graph pre-filled with a sentinel so "restored" cannot be confused with "already
  correct", distinct per-tensor values so a cross-wired restore is visible, and the missing-blob
  throw. In `Components/Transformers/Gpt/GptTransformer.Cpu.cpp` (added after the mock-based suite
  proved unable to see the composite defect): the same round trip on a **real two-layer network**,
  asserting **one blob per parameter** — the assertion the unscoped walk fails outright, since it
  produced a single blob regardless of parameter count — plus config recovery (pinning the two fields
  the hand-rolled metadata got wrong) and a throw when the archive describes a different network.
  Together these are the oracle the save side never had, and the first thing to exercise
  `getParameterNames()` as a shared vocabulary rather than two lists that happen to agree, closing the
  Phase 2-3 gap.
  **One assertion is deliberately stronger than needed:** `blob_count == getParameters().size()` will
  also fail if a component's `getParameterNames()` and `getParameters()` disagree. That is the Phase 2
  gap restated as a runtime check, so a failure there is information rather than a bad test.

- **Model serialization — the checkpoint round trip and the distribution artifact.** Specified
  2026-07-29 in **`Specifications/ModelSerialization.md`**, which carries the design, the defect
  analysis and a seven-phase build plan; this entry is the pointer. Phases 0-1 are freeze-compatible
  defect repair and are filed under Production Hardening above; Phases 4-7 are the vNext feature.
  **The three things worth knowing without opening the spec:**
  **(a) The load side is much further along than "absent" — a first pass called it missing and that was
  wrong in a way that shrinks the work.** `ITensorBlob` (`Tensor.Serialization.ixx:53`) is already the
  type-erased boundary between a byte source and a component; `readTensorBlob( archive, prefix )`
  (`:172`) already produces one *from an archive*; and `loadParameter` is already implemented by all
  five parameter-owning components. What is missing is the **traversal** that walks an archive and
  drives them — the flat path has one (`Gemma.ixx:383`, `Llama.ixx:377`, `GptTransformer.ixx:532`, each
  welded to `PretrainedModelReader&`), the archive path has none. A default `Component::load_` iterating
  `getParameterNames()` covers all five leaves with no per-component code.
  **(b) `getParameterNames()` has ZERO overrides** (`Component.ixx:518`), though its own docstring calls
  it "the canonical parameter name list in the same stable order used by `save_()` and
  `loadParameter()`". That order does not exist, so save and load have no agreed vocabulary.
  Implementing it is the precondition for both sides, not a tidy-up after.
  **(c) `Checkpoint` and `WeightsOnly` are two artifacts, not two flags.** Training checkpoint =
  parameters + optimizer moments + step/RNG, written repeatedly, read by the same build — `ModelArchive`
  suits it. Distribution artifact = weights only, mmap-able, forward-compatible — that is the flat
  `.bin` + `PretrainedModelReader`, and routing distribution through the archive would replace a
  memory-mapped read with zip decompression into a heap buffer and produce a 22 GB zip. The small
  hostable artifact needs **quantized-tensor serialization in the flat format** (packed FP4 nibbles plus
  their per-group scale companion), which is what would take Gemma 12B from 22.2 GB to roughly 7 GB —
  precisely the number that ruled out hosting in `Specifications/PythonBinding.md`.
  **Also unbuilt and worth knowing: there is no optimizer state anywhere.** `SerializationMode` documents
  `Checkpoint` as "architecture + weights + optimizer state"; `AdamWConfig` serializes hyperparameters
  and the moments, step count and FP32 masters have no representation, so `Checkpoint` cannot currently
  mean what it says.
  **Naming is already settled by the codebase** — `fromPretrained` / `fromCheckpoint` as static
  factories, `saveCheckpoint` as a member (loading constructs, saving does not), both thin over
  `save( ModelArchive&, SerializationMode )`. Don't name the general API after one of three modes, and
  don't route concrete models through `NetworkFactory` — it is the string-keyed runtime registry the
  project is phasing out, and a model knows its own type at compile time.
  **Sequencing: the training checkpoint has a consumer today** — MNIST and Bard cannot resume a run, and
  the BF16 train-from-scratch path was just repaired in `e585be9d`; the distribution artifact has none
  until the Python/packaging work lands. Checkpoint first also leaves the working mmap inference path
  untouched. The acceptance test for the whole spine is Bard training N steps, checkpointing,
  restarting, and continuing on the same loss trajectory.

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
- **Python binding and samples** — scoped 2026-07-28 into
  `Specifications/PythonBinding.md`; read that before starting. Surface the `mila` binding as a
  consumable product for the Python-majority audience: it already drives MIS but had exactly one
  consumer, inside this repo, and no user-facing entry point.
  **All four freeze-compatible items landed 2026-07-28** and are tracked under the Adaptor Validation
  bucket above: the samples (`Mila/Samples/Python/`), the neutral extension output location, the
  `quantize_fp8` defect, and the stale module docstring. **Still open here:** Tier 2 and Tier 3
  below, binding `GptModel`, a precision parameter for `GemmaModel.from_pretrained`, and the product
  call on whether any of this promotes out of `## Future` into a v0.20 ROADMAP theme (it changes what
  v0.20 claims, so it is deliberate — the shipped work sits in the tree either way).
  **The "one command from a clean checkout" goal is tiered.** Tier 1 (one command given a built
  `.pyd`) is done. Tier 2 (weights fetched on first run) is **newly possible in Python and
  was not in C++** — the deferred zero-auth quick-start was blocked by needing a runtime HTTP
  dependency, and `urllib.request` is standard library while a sample is not the runtime.
  **Licensing is settled: Gemma 4 12B is Apache 2.0** — a change from the bespoke Gemma Terms of Use
  that governed earlier releases, so do not reason from those. Redistributing a converted FP4 `.bin`
  needs only the standard four (licence, `NOTICE`, retained attribution, statement of modification),
  with attribution going in root `NOTICE.md` per the existing licensing rule. A HuggingFace repo
  removes the hosting cost. Structurally this means Tier 2 needs **neither GPT-2 nor `GptModel`
  bound** and is freeze-compatible end to end. It also inverts the artifact question: Llama 3.2
  carries the Llama Community License with naming and threshold conditions while Gemma 4 carries
  none, so **Gemma 4 12B FP4 is both the flagship and the licensing-simplest choice** — the only
  argument against it is download size. Tier 3 (a published wheel, no build at all) is what actually
  reaches the stated audience and is a post-v0.20 piece of its own.
  Open decisions in the spec: first-run experience versus a multi-gigabyte download (is there a
  smaller Gemma 4 variant Mila could validate?), whether this promotes out of `## Future` into
  Production Hardening as a barrier lever beside the Docker image and CPU-only path, and wheel
  distribution.
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
  ALSO OPEN (2026-07-27): **`Web/static/achilles.png` is orphaned but still published.** Nothing
  under `Web/layouts/` references it -- it survives only as this entry and as the provenance comment
  in `mila-mark.svg:4` -- yet living in `static/` means Hugo copies it to the site root, so the
  retired mark is still served at `https://mila.toddt.me/achilles.png`. Keep it in the repo (it is
  the trace source the SVG cites); move it to a tracked path outside `Web/static/`. Not
  `.internal/Marketing/Brand/`, which is gitignored.
  NOT a defect, recorded so it is not rediscovered as one: Search Console's property switcher shows
  the old Achilles icon for `https://mila.toddt.me/`. That is Google's favicon cache from a crawl
  predating `6f11a5f1` (which is where `favicon.ico`/`icon.png` were first added -- before it the
  site's only icon was `achilles.png`). The live icons byte-match the repo; the favicon index
  refreshes on its own cadence, separate from the page index.
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
