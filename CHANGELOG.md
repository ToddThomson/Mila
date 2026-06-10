# Mila — Changelog

Completed, validated work, newest first.

- **Open tasks** live in [BACKLOG.md](BACKLOG.md).
- **Milestone vision and success criteria** live in [ROADMAP.md](ROADMAP.md).

Versions are the `Version.txt` stamp at the time the work landed. During alpha, releases
are tagged off `master` (see Release Assets in BACKLOG); completed phases below double as the
release notes.

---

## Alpha.6 — Consolidation (feature freeze + debt burndown; in progress)

The bridge from "the features work" to a tree honest enough to call beta. Milestone vision
is in ROADMAP; open triage buckets are in BACKLOG.

### Parameter initialization subsystem restored — DONE + VALIDATED (0.13.53-alpha.5)

The host `Tensor.Initializers` facade was deprecated (host-side init is wrong for CUDA) in
favor of per-device `TensorOps<device>` ops — but the CPU Random backend was never written
and most components' init calls were left commented (`FIXME`), so weight/gradient
initialization was silently disabled. Restored end to end, gated so inference is unaffected.

- `CpuTensorOps.Random` — new host `fill_normal`/`fill_uniform` (std distributions seeded
  from `Core::RandomGenerator`), closing the per-device parity gap; CUDA already had cuRAND
- `TensorOps.Random` — added device-agnostic `xavier()` (Glorot limit -> `fill_uniform`)
- `Kernels/Random.cu` registered in the build with its include corrected; the CUDA
  `fill_uniform` path (`launch_scale_shift`) was declared but never compiled/linked
- `GptModel::fromPretrained` now passes `initialize_parameters=false` (matching `LlamaModel`)
  so pretrained loads no longer run-then-discard initialization
- Gated value/grad init wired via live `zero`/`fill`/`xavier`/`fill_normal` in `Linear`,
  `RmsNorm`, `LayerNorm`, `Cpu`/`CudaAdamWOptimizer`, `TokenEmbedding`, `Lpe`; dead
  `Dnn.TensorInitializers` imports dropped. Each component gate-checked individually (most
  had ungated init — the GptModel/TokenEmbedding "takes too long" bug class)
- Validated: green build + coherent chat. Init is gated off on the inference path, so
  runtime correctness of train-from-scratch still awaits a unit test (first TDD-revival
  target). _Deferred (BACKLOG):_ CUDA `fill_normal`/`fill_uniform` are FP32-only (BF16
  train-from-scratch on CUDA corrupts); couple `initialize_parameters` default to `RuntimeMode`

### Multi-device `setCurrentDevice` re-enabled — DONE + VALIDATED (0.13.53-alpha.5)

- Enabled `Cuda::setCurrentDevice` at 9 sites in `CudaTensorOps.Transfer` (7) and
  `CudaTensorOps.Random` (2), restoring the `Cuda.Helpers` imports — needed for cross-device
  kernel launches and allocations on multi-GPU (no-op on single-GPU, device 0 always current)
- The original ICE was an old thread-local cache in the helper body, already removed; the
  helper is now a trivial `cudaSetDevice` wrapper. _Residual (BACKLOG/Blackwell):_
  `CudaTensorOps.Math` is internally inconsistent (5 live + 4 "redundant"); a scoped RAII
  device guard should replace the scattered bare calls when the dual-GPU rig validates them

## Alpha.5 — FP8/FP4 load-time quantization (in progress; completed work below)

Validated on Llama 3.2 3B and Llama 3.1 8B Instruct. Quantization is a compile-time
deployment decision via `TWeightQuant` on `Linear`/`CudaLinearOp`; the converter always
writes BF16 and quantization is entirely Mila's concern. Remaining Alpha.5 items are in
BACKLOG.

### Phase 6 — PretrainedModelReader bulk I/O — DONE (0.13.50-alpha.5)

The decisive insight: the former loop iterated tensors in `unordered_map` (hash) order, so
per-tensor `fstream` reads were effectively random seeks; consuming in ascending file offset
turns the load into a single sequential scan the OS reads ahead.

- `Serialization/Tensor.ixx` — added `TensorBlobView : ITensorBlob`, a non-owning view (metadata + borrowed `const void*` + size); the stable `ITensorBlob` surface is unchanged
- `PretrainedReader.ixx` — whole file memory-mapped at construction (`CreateFileMapping`/`MapViewOfFile` + `FILE_FLAG_SEQUENTIAL_SCAN` + best-effort `PrefetchVirtualMemory` on Windows; POSIX `mmap` + `madvise(SEQUENTIAL|WILLNEED)` fallback keeps Linux green). New `streamTensorBlobs<TStagingMemoryResource>(consume, device_id)` consumes blobs in file-offset order; legacy `readTensorBlob<MR>` retained as a random-access fallback. Reader stays free of `cuda_runtime.h`
- CUDA path runs a background producer thread staging each blob `mmap -> pinned` (double-buffered slots, 256 MB cap; oversized tensors bypass staging and stream directly from the pageable view) while the calling thread runs the consumer (H2D + quantize). Producer does only host `memcpy`. CPU path consumes mapped views directly, no staging or thread
- Quantize-on-load reuse-safety contract: the FP8/FP4 `quantize` path issues an async H2D and does NOT self-synchronize; since pinned slots are reused, the model's `consume` callback synchronizes its execution context after `loadParameter` so a slot is never overwritten mid-transfer (producer's next `memcpy` overlaps that sync, preserving disk/H2D overlap). `Llama.ixx` + `GptTransformer.ixx` `loadParameters` rewritten to drive `streamTensorBlobs`
- Validated on Llama 3.1 8B FP4: load dropped from ~8s to near the < 3s target; coherent output; no 3B regression
- _Deferred (BACKLOG):_ module hygiene `#ifdef` cleanup in `PretrainedReader.ixx`; Phase 6b H2D pipelining

### Phase 5 — Llama 3.1 8B Instruct @ FP8 — DONE

First target where FP8 is practically necessary (BF16 ~16 GB exceeds the RTX 4070 12 GB
budget; FP8 ~11.6 GB at context 8192). Production default is 8B at FP4 (~6 GB, ~57 tok/s
decode after the warp-per-row decode-softmax rewrite); FP8 is the validated alternative.

- `Llama.Presets.ixx` — `Llama3_1_8B()` preset (embedding=4096, layers=32, heads=32, kv_heads=8, hidden=14336, rope_theta=500000); `fromPretrained` reads architecture dims from checkpoint metadata
- `convert_llama_weights.py` — supports `meta-llama/Llama-3.1-8B[-Instruct]`; key mapping/gate-up concat identical to 3.2; `tie_word_embeddings=False` handled by existing lm_head fallback; output `llama31_8b_instruct_bf16.bin`
- `ChatConfig` — `ModelSize::B8`; `switchModel()` derives `llama31` family for B8 vs `llama32` for 1B/3B; `llama-8b`/`llama-8b-fp32` aliases
- `Llama.ixx` — `exec_context_` moved to last member so it destroys first; `cudaStreamSynchronize()` in `releaseResources()` fires before tensor `cudaFree()`, fixing UB during destruction
- `Chat.ixx` — `switchModel()` destroys the current model before allocating the replacement, eliminating the transient old+new VRAM peak (~12.83 GB) that caused WDDM spill and 4 tok/s warm-up
- `main.cpp` — `llama_weights_path()` uses `llama31` prefix for B8; chat default updated to Llama 3.1 8B FP4
- Prefill + greedy decode validated at FP8 (clean load and hot `/model llama-8b fp8` switch); stale-pointer root cause fixed (cached `dequant_weight_buffer_` dangled after `getDeviceScratchBuffer()` grow-realloc)

### Phase 4 — FP4 E2M1 weight quantization (storage) — DONE

4× VRAM reduction vs BF16; storage-only (native FP4 compute requires Blackwell SM 10.0).
Weights packed as nibbles, dequantized to BF16 on the W4A16 GEMM path.

- `Quantization/Weight/Policies.ixx` — `PerGroupFp4<GroupSize=128>` (`kStorageDtype=UINT8`, `kScaleDtype=Float32`, `kIsFp4E2M1=true`); `PerGroupInt4` gets `kIsFp4E2M1=false` for dispatch disambiguation; `PerGroupFp4<64>` variant; `static_assert`-verified
- FP4 E2M1 quantization kernel (`CudaFp4WeightQuantization.cu`) — BF16 → packed FP4 nibbles with per-group absmax scales (`/6.0f`); grid `(K/group_size, N)`; parallel absmax then E2M1 encode + nibble pack; `quantize_fp4_per_group()` bridge in `CudaLinearOp.Quantize.ixx`
- W4A16 tile dequant — `fused_w4a16_gemm_kernel` extended with `kIsFp4E2M1`; `if constexpr` branch unpacks nibble → `fp4_e2m1_decode()` LUT (`±{0,0.5,1,1.5,2,3,4,6}`) → group scale; `cuda_fp4a16_gemm()` host dispatch; `Linear.ixx` scale allocation corrected to 2D `[N, K/group_size]`
- `OperationTraits.Cuda.ixx` — `<Cuda, BF16, PerGroupFp4<128>>` and `<…, PerGroupFp4<64>>` LinearOp specializations
- `LlamaModel::fromPretrained()` — `WeightQuantization::FP4` dispatch updated to `PerGroupFp4<128>`
- FP4 E2M1 decode matvec — `matvec_decode_bf16_qfp4_kernel<kGroupSize>` in `CudaMatVecBias.Bf16.cu` for the outer_size==1 path; 44–48 tok/s vs 6–7 with the tiled GEMM (~7× improvement)
- Validated on Llama 3.2 3B Instruct — coherent generation; ~4 GB total VRAM

### Phase 3 — Llama 3.2 3B Instruct @ FP8 — DONE

- `ChatConfig` — `QuantizationMode` enum (`None`/`FP8`/`FP4`) orthogonal to `ModelPrecision` (compute type); `context_length` uncapped; `ModelSize::B8` added
- `LlamaModel::fromPretrained()` — `WeightQuantization::FP8` dispatches to `fromPretrainedImpl<PerChannelFp8<>, NoKvCompression>`; compile-time only
- `ConsoleRenderer` (`Chat.Renderer.ixx`) — standalone non-exported module; braille spinner with cursor hide/show; solid-color response blocks with word-wrap preserving leading indentation; Unicode welcome box; dim ANSI stats line; dynamic console width
- `Chat.ixx` — `/model <alias> [quant]` hot switching; `resolveAlias()` for `gpt2`/`llama-1b`/`llama-3b`/`llama-8b` + `-fp32` variants; `parseQuantization()`; context length preserved across same-architecture switches; all responses fully buffered (streaming removed from the hot path)
- Prefill + greedy decode validated at FP8 — coherent on Llama 3.2 3B Instruct; no catastrophic divergence vs BF16; TTFT ~2× faster than the W8A16 fused path

### Phase 2 — FP8 quantization infrastructure — DONE

- `Quantization/Weight/Policies.ixx` — `NoWeightQuant` identity; `PerChannelFp8<>` (`kStorageDtype=FP8_E4M3`, `kScaleDtype=Float32`, `kPerChannel=true`); `WeightQuantPolicy` concept; `static_assert`-verified
- `Quantization/KvCache/Policy.ixx` + `QuantPolicy.ixx` — `KvCachePolicy`/`QuantKvPolicy` concepts; `NoKvCompression` identity; `PerChannelKvFp8<>` (`kPerHeadPerToken=true`)
- `Linear.ixx` — `TWeightQuant` constrained to `WeightQuantPolicy`; `kIsQuantized`/`kWeightDtype` from policy; `loadParameter()` delegates to `operation_->quantize()` + `setWeightScales()` on the quantized path
- `CudaLinearOp.ixx` — `quantize()`/`setWeightScales()` gated on `requires kIsQuantized`; `supportsCuBLASLt()` SM ≥ 8.9 FP8 check; FP8 decode matvec kernel; FP8 batch-prefill 2-phase dequant (`cuda_fp8_dequantize_to_bf16` → staging → BF16×BF16 cuBLASLt NT GEMM, bias post-GEMM); staging buffer fetched at `forward()` from `getDeviceScratchBuffer()` (grow-on-demand shared scratch); stale-pointer bug fixed (caching the buffer at build time dangled after grow-realloc)
- W8A16 fused GEMM A/B path — `kUseW8A16Gemm` compile-time toggle; single kernel dequantizes inline in shared memory; benchmarked 2–3× slower than 2-phase (scalar cores vs cuBLASLt tensor cores), default `false`, retained as correctness reference

### Phase 1 — Compile-time operation dispatch — substantially DONE (0.13.51-alpha.5)

Replaced runtime `OperationRegistry` string-keyed lookup with compile-time
`OperationTraits<OperationType, TDeviceType, TPrecision, TPolicy>`. A missing specialization
is a compile error. (Remaining: token sampling, CPU Linear traits, Dropout, dead-file
deletion — see BACKLOG.)

- `OperationTraits.Template.ixx` — unified primary template; `LinearOpConcept`; transitive `export import` of `DeviceType`/`OperationType`/`TensorDataType`
- `OperationTraits.Cuda.ixx` — Linear `<Cuda, FP32/BF16, NoWeightQuant>` + `<Cuda, BF16, PerChannelFp8<>>`; GQA `<Cuda, FP32/BF16, NoKvCompression>`; policy-free CUDA specializations (FP32+BF16) for `Gelu`/`Residual`/`RmsNorm`/`Softmax`/`Swiglu`/`MultiHeadAttention`/`Rope`/`Lpe`/`TokenEmbedding`/`CrossEntropy`; `LayerNorm` (FP32/FP16)
- `OperationTraits.Cpu.ixx` — `:Cpu` partition; FP32 specializations for `Gelu`/`Residual`/`Softmax`/`MultiHeadAttention`/`Lpe`/`LayerNorm`
- Components migrated to `OperationTraits` dispatch: `Linear` (reference), `GroupedQueryAttention` (with `TKvPolicy`), all 9 policy-free components, `LayerNorm`, `MLP` (composite, dispatches nothing of its own), `SoftmaxCrossEntropy` (CUDA FP32/BF16 wired; CPU intentionally absent → deliberate hard compile error). Fixed latent `CudaTokenEmbeddingOp::setGradients` signature bug (non-virtual 1-arg hid the base 2-arg virtual)
- Retired the runtime dispatch scaffolding from the build — `OperationRegistry`/`OperationRegistryHelpers`/`OperationsRegistrar`/`OperationRegistrarHelpers`, the arity base classes (`UnaryOperation`/`BinaryOperation`/`PairedOperation`), and all legacy typemaps; no longer re-exported from `Mila.ixx`; `Mila::initialize()` no longer calls `OperationsRegistrar::instance()`. The unused `FusedComponent` marked `[[deprecated]]` and excluded
- Removed the unused `Dnn/Extensibility` plugin scaffolding (`IModulePlugin`/`PluginManager`/`PluginInfo`/`MyCustomPlugin`)

### Beta-track work landed during alpha

- License reconciliation (0.13.x, 2026-06-08) — repo had stated the license four ways (`License.md` MIT, source headers proprietary EULA, README + CONTRIBUTING Apache 2.0); all reconciled to **MIT**. SPDX header on the two public-entry source files (`Mila.ixx`, `Version.ixx`); README/CONTRIBUTING corrected and pointed at `License.md`; holder unified to `Todd J. Thomson` (dropped "Achilles Software"); copyright bumped to 2021..2026
- Formatter/linter config (2026-06-09) — `.editorconfig` (VS 2026 `cpp_*` keys, the Windows-authoritative formatter; fixed `indent_style` and added `charset`/`trim_trailing_whitespace`/`insert_final_newline`), `.clang-format` (best-effort for non-VS contributors; `ColumnLimit: 0`, Allman, spaces-in-parens), `.clang-tidy` (conservative). Conscious defers: `.clang-tidy` not a CI gate yet (needs module-aware `compile_commands.json`); `.clang-format` cannot express the no-abbreviation / blank-line / ASCII-only rules (review-time conventions, cross-referenced in CONTRIBUTING)
- GitHub community-health files (2026-06-09) — `CODE_OF_CONDUCT.md`, `SECURITY.md` (solo-maintainer private disclosure; scope = weight-blob/tokenizer parse paths), root `PULL_REQUEST_TEMPLATE.md`, and `.github/ISSUE_TEMPLATE/` (`bug_report`/`feature_request`/`config.yml`). config.yml contact links point at `dev` (flip to `master` at the beta default-branch switch)
- Docs publish via Actions-native Pages (2026-06-09) — replaced the `JamesIves`/`gh-pages` approach (silently failing inside the CUDA build container, site frozen ~12 months) with `upload-pages-artifact` + a git-free `deploy-pages` job; Pages source switched to "GitHub Actions"; `workflow_dispatch` added; validated on PR #14 merge, site live
- CI GPU-test honesty refactor (2026-06-07) — removed the old GPU-less `test` job that could false-green by silently skipping CUDA tests; CI is now **GPU-free by design** (a single clang-21 compile + packaging-gates job), with GPU correctness validated locally before commit. Also eliminated the gigabytes-large cross-runner `build/` tree handoff (consolidated to one job; `docs` is its own workflow)
- Packaging gates + CI bring-up — `find_package` and FetchContent gates green; CI = clang-21 + gcc-15 host on `cuda:13.3-ubuntu26.04`; CPU-only build (`MILA_ENABLE_CUDA=OFF`) green with ctests passing; WSL/Ubuntu 26.04 native Clang build green and chat-validated (use CUDA 13.3 — 13.0 fails on glibc 2.43 rsqrt)
- First proper tagged release `v0.13.46-alpha.5` (2026-06-08) — full `dev` → `master` → tag → release pipeline validated; master resynced from pre-Llama

---

## Alpha.4 — Instruction following & tool calling — Complete

Llama 3.2 3B Instruct at BF16. Structured message + tool-calling infrastructure in the Chat
layer; no model architecture changes.

- `ChatMessage` (role/content/optional tool calls); `MessageFormatter` applies the Llama 3.2 instruct chat template; stop on `<|eot_id|>`/`<|eom_id|>`; `Chat::run()` uses structured history
- `BpeTokenizer::loadLlama32` encodes special tokens as single atomic IDs
- `ToolDefinition`/`ToolCall`/`ToolCallParser` (detects `<|python_tag|>`, extracts + validates JSON); system-prompt builder injects active tools; `ChatConfig` gains `system_prompt`/`tools`; `Chat::registerTool()` binds handlers and dispatches the round-trip
- Validated end-to-end: model issues a tool call, result is fed back, final response correct

---

## Alpha.3 — BF16 compute backend — Complete

Greedy decode of `LlamaModel` matches HuggingFace token-for-token on Llama 3.2 3B at BF16.
BF16 is the primary reduced-precision target (matches FP32 exponent range, halves bandwidth);
FP16 is not a Mila target. CUDA BF16 kernels for the full GQA pipeline; BF16 dispatch wired;
converter extended for the 3.2 3B layout; prefill logits and full-network decode validated.

---

## Alpha.2 — Llama architecture — Complete

Greedy decode matches HuggingFace token-for-token on Llama 3.2 1B at FP32. Delivered
`TokenEmbedding`, `RoPE`, SiLU, SwiGLU MLP, `GroupedQueryAttention` (configurable
`num_kv_heads` + KV-cache), `LlamaBlock`, `LlamaTransformer`, `LlamaModel`, `LlamaConfig`;
`convert_llama_weights.py`; SentencePiece for Llama 3.x; prefill + full decode validated.

---

## Alpha.1 — GPT-2 inference — Complete

Full GPT-2 decoder stack, greedy decode matching HuggingFace token-for-token — establishes
the validation methodology all later architecture work follows. Core components (Linear,
LayerNorm, MHA, MLP, Residual, GELU) with CUDA + CPU kernels; `GptTransformer` (pre-LN);
`GptModel`; two-phase KV-cache (prefill + decode); HuggingFace GPT-2 converter; BPE
tokenizer; Chat CLI sample; AdamW optimizer + MNIST training loop.
