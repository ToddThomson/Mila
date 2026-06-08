# Mila — Roadmap

---

## Versioning

| Stage | Version | Title |
|---|---|---|
| In Progress | 0.13.46-alpha.5 | FP8/FP4 quantization pipeline — Llama 3.2 3B and 3.1 8B Instruct |
| Planned | 0.2.1-beta | Public release |
| Planned | 0.2.2-beta.1 | Qwen 3 architecture + thinking mode — Qwen 3 8B Instruct |
| Planned | 0.2.3-beta.2 | Ministral architecture + SWA — Ministral 3B and 8B Instruct |

---

## Alpha.5 — In Progress

**FP8/FP4 load-time quantization pipeline, validated on Llama 3.2 3B and 3.1 8B Instruct.**

Quantization in Mila is a compile-time deployment decision, not a runtime configuration
concern. Weight precision is encoded as a template parameter `TWeightQuant` on `Linear`
and `CudaLinearOp`. When `TWeightQuant = PerChannelFp8<>`, the `Linear` component
quantizes its weights from BF16 to FP8_E4M3 during `loadParameter()`, computing
per-channel FP32 scales via `CudaLinearOp::quantize()`. No quantized checkpoint format
is required — the converter always writes BF16, and quantization is entirely Mila's
concern. The existing BF16 baseline validated in Alpha.3 is the correctness reference
for all FP8 validation.

Llama 3.2 3B Instruct is the initial validation target because its BF16 baseline is
already token-for-token correct, making it the cleanest possible foundation for
isolating precision regressions. Llama 3.1 8B Instruct extends validation to a scale
where FP8 is practically required — at BF16 the model exceeds the RTX 4070 12 GB VRAM
budget; at FP8 it fits comfortably at ~8 GB. The quantization infrastructure is
model-agnostic and carries forward to Qwen 3 in Beta.1.

Success criterion: Greedy decode of Llama 3.2 3B Instruct at FP8 produces no catastrophic
divergence from the BF16 baseline on standard prompts. Llama 3.1 8B Instruct at FP8 fits
within the RTX 4070 12 GB VRAM budget and produces output quality consistent with its
BF16 baseline.

### Phase 1 — Compile-Time Operation Dispatch (All Components)

Replace runtime `OperationRegistry` string-keyed lookup with unified compile-time traits
dispatch via `OperationTraits<OperationType, TDeviceType, TPrecision, TPolicy>`. The
`Linear` component is the completed reference implementation. A missing `OperationTraits`
specialization is a compile error — no runtime fallback, no string key, no hash map.

The unified `OperationTraits` primary template (keyed on the `OperationType` enum) supersedes
the earlier per-component `XxxOpTypeMap` design. All remaining components migrate to
`OperationTraits` specializations in `OperationTraits.Cuda.ixx` and `OperationTraits.Cpu.ixx`.

- [x] `OperationTraits.Template.ixx` — unified primary template `OperationTraits<OperationType TOp, DeviceType, TPrecision, TPolicy = void>`; `LinearOpConcept` defined for contract documentation; `export import` on `DeviceType`, `OperationType`, `TensorDataType` so all consumers get them transitively
- [x] `OperationTraits.Cuda.ixx` — `:Cuda` partition; `LinearOp` specializations for `<Cuda, FP32, NoWeightQuant>`, `<Cuda, BF16, NoWeightQuant>`, `<Cuda, BF16, PerChannelFp8<>>`
- [x] `Linear.ixx` — `using OpType = OperationTraits<LinearOp, TDeviceType, TComputePrecision, TWeightQuant>::type`; `createOperation()` uses `std::make_shared<OpType>`; dead `LinearOpTypeMap` import removed
- [x] `CudaLinearOpRegistrar` — removed (dead code, registration approach fully retired)
- [x] `OperationTraits.Cuda.ixx` — `GroupedQueryAttentionOp` specializations added: `<Cuda, FP32, NoKvCompression>` and `<Cuda, BF16, NoKvCompression>`; `GroupedQueryAttention` component migrated to `OperationTraits` dispatch; `TKvPolicy` template parameter wired with `kKvCompressed = TKvPolicy::kIsActive` and `kCacheDtype` derived from policy; `NoKvCompression` is the default
- [ ] `OperationTraits.Cuda.ixx` — `<Cuda, BF16, PerChannelKvFp8<>>` GQA specialization pending `CudaGqaOp` FP8 KV cache support (deferred to Beta.1 Phase 4)
- [ ] `OperationTraits.Cuda.ixx` — `SamplingOp` specializations: `<Cuda, FP32>` and `<Cuda, BF16>`; implement `TokenSampler` component and `CudaSamplingOp` per `TokenSampling.md`
- [x] `OperationTraits.Cuda.ixx` — CUDA policy-free op specializations added (FP32 + BF16 each): `GeluOp`, `ResidualOp`, `RmsNormOp`, `SoftmaxOp`, `SwigluOp`, `MultiHeadAttentionOp`, `RopeOp`, `LpeOp`, `TokenEmbeddingOp`, `CrossEntropyOp`; all 9 active components' `createOperation()` migrated to `OperationTraits` dispatch; fixed latent `CudaTokenEmbeddingOp::setGradients` signature bug (non-virtual 1-arg was hiding base class 2-arg virtual)
- [ ] `SoftmaxCrossEntropy` component — pending lifecycle API modernization before `OperationTraits` dispatch can be applied (`onBuilding(shape_t)`, raw `exec_context_*` member, and 4-param `BinaryOperation` do not match current `Component` API)
- [x] `OperationTraits.Cpu.ixx` — `:Cpu` partition created; active CPU op specializations added (FP32): `GeluOp`, `ResidualOp`, `SoftmaxOp`, `MultiHeadAttentionOp`, `LpeOp`; corresponding CPU component paths migrated
- _(not a gate)_ `OperationTraits.Cpu.ixx` — Llama-lineage CPU ops (`RmsNormOp`, `SwigluOp`, `RopeOp`, `TokenEmbeddingOp`, `CrossEntropyOp`) are intentionally **demand-driven**. The compile-time dispatch makes their absence zero-cost on the GPU path and surfaces a localized compile error if a `<Cpu, …>` Llama is instantiated — so they are filled in by a contributor when CPU Llama is actually wanted. See the Beta "Compute backend scope" note
- [ ] `LayerNorm`, `Dropout`, `MLP` components — `LayerNormOp` has CPU/CUDA registrars but no `OperationTraits` entries yet; `DropoutOp` same; `MLP` has stale `OperationRegistry` import to remove (before Beta.1)
- [ ] Remove `OperationRegistry`, `OperationRegistryHelpers`, `LinearOpTypeMap`, `GqaOpTypeMap`, and all legacy registrar files once all components are migrated

### Phase 2 — FP8 Quantization Infrastructure

Weight quantization is a compile-time template parameter `TWeightQuant` on `Linear` and
`CudaLinearOp`. `NoWeightQuant` is the default for all non-quantized paths; `PerChannelFp8<>`
activates the FP8 path. There is no runtime quantization config object — `kIsQuantized` is
`TWeightQuant::kIsQuantized`, a compile-time constant. `QuantizationConfig` and
`LinearConfig::withQuantization()` are explicitly not part of this design.

The quantization pipeline:
1. `Linear` is instantiated with `TWeightQuant = PerChannelFp8<>`
2. `loadParameter("weight", blob)` delegates to `operation_->quantize(blob, *weight_, *weight_scales_, shape)`
3. `quantize()` runs host-side: computes per-channel scales (`max(abs(W[o,:])) / 448.0f`), uploads FP8 weights and FP32 scales to device
4. `operation_->setWeightScales(weight_scales_.get())` binds the scale tensor to the cuBLASLt plan descriptor
5. On each forward pass, cuBLASLt executes the mixed-precision GEMM natively — no dequantization on the hot path

`quantize()` and `setWeightScales()` are concrete methods on `CudaLinearOp` only — not
on the operation base class. Non-quantized operations are entirely unaware they exist.

- [x] `Dnn/Quantization/Weight/Policies.ixx` — `NoWeightQuant` identity struct; `PerChannelFp8<>` policy struct with `kStorageDtype = FP8_E4M3`, `kScaleDtype = Float32`, `kPerChannel = true`; `WeightQuantPolicy` concept; both policies verified with `static_assert` at definition time
- [x] `Dnn/Quantization/KvCache/Policy.ixx` — `KvCachePolicy` concept; `NoKvCompression` identity struct; zero-cost, no behavior change to any existing GQA path
- [x] `Dnn/Quantization/KvCache/QuantPolicy.ixx` — `QuantKvPolicy` concept refinement; `PerChannelKvFp8<>` policy struct; both verified with `static_assert` at definition time
- [x] `OperationTraits.Cuda.ixx` — `<Cuda, BF16, NoWeightQuant>` and `<Cuda, BF16, PerChannelFp8<>>` Linear specializations (replaces `LinearOpTypeMap.Cuda.ixx` for the quantized path)
- [x] `Linear.ixx` — `TWeightQuant = NoWeightQuant` parameter constrained to `WeightQuantPolicy`; `kIsQuantized`, `kWeightDtype` derived from policy; `WeightTensorType` alias; `loadParameter()` delegates to `operation_->quantize()` and `operation_->setWeightScales()` on the quantized path
- [x] `CudaLinearOp.ixx` — `TWeightQuant` template parameter; `quantize()` and `setWeightScales()` gated on `requires kIsQuantized`; `supportsCuBLASLt()` SM ≥ 8.9 check for FP8; `getComputeTypes()` FP8 branch; FP8 decode matvec kernel (BF16 activation × FP8_E4M3 weight + FP32 scale → BF16 output)
- [ ] `OperationTraits.Cpu.ixx` — `<Cpu, FP32, NoWeightQuant>` Linear specialization (replaces `LinearOpTypeMap.Cpu.ixx`)
- [x] `CudaLinearOp.ixx` — FP8 batch prefill path: 2-phase dequantize (`cuda_fp8_dequantize_to_bf16` → BF16 staging buffer) followed by standard BF16×BF16 cuBLASLt NT GEMM; bias added post-GEMM by `cuda_add_bias` to avoid Ada epilogue constraint; staging buffer fetched at `forward()` call time from `context_->getDeviceScratchBuffer()` (grow-on-demand shared scratch owned by the execution context, freed in `releaseResources()`); per-layer `cudaMalloc` approach retired after causing OOM at 8B scale (~13 GB aggregate). Stale pointer bug fixed: caching the buffer pointer at build time caused dangling references after a grow-realloc; fetching at `forward()` time is safe because all ops share a single stream. Native FP8 cuBLASLt (separate `data_type_A`/`data_type_B` descriptor) deferred — 2-phase is the validated production path.
- [x] W8A16 fused GEMM A/B test path — `kUseW8A16Gemm` compile-time toggle in `CudaLinearOp`; single kernel reads FP8 weights once, dequantizes per-channel inline in shared memory, accumulates in float32, writes BF16 output; eliminates BF16 staging buffer. Benchmarked: 2–3× slower than 2-phase at target batch sizes (scalar float CUDA cores vs cuBLASLt tensor cores); `kUseW8A16Gemm = false` is the default. Kernel retained as correctness reference; tensor-core WMMA upgrade is the path to a real win.

### Phase 3 — Llama 3.2 3B Instruct @ FP8

- [x] `ChatConfig` — `QuantizationMode` enum (`None`, `FP8`, `FP4`) orthogonal to `ModelPrecision`; `QuantizationMode` is the runtime quantization selector, `ModelPrecision` remains the compute-type selector
- [x] `ChatConfig` — `context_length` uncapped; no FP8-specific ceiling; `ModelSize::B8` added to enum; caller sets context length directly
- [x] Wire FP8 through `LlamaModel::fromPretrained()` — `WeightQuantization::FP8` dispatches to `fromPretrainedImpl<PerChannelFp8<>, NoKvCompression>`; compile-time only, no runtime config object
- [x] `ConsoleRenderer` (`Chat.Renderer.ixx`) — standalone non-exported module; braille dot spinner (⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏) with cursor hide/show (`\x1b[?25l/h`) to suppress blink flicker; solid color response blocks (`bg(40,44,60)` / `fg(200,215,240)`) with uniform right-fill and word-wrap preserving leading indentation (tabs expanded, `at_line_start` tracking, spaces at line-start bypass `flush()`); Unicode welcome box (╭─╮/│/╰─╯); dim ANSI generation stats line (`\x1b[2m`, format: `ms │ tok/s │ tokens`); `printInfo()` / `printError()` for system messages; dynamic console width via `GetConsoleScreenBufferInfo`; ANSI RGB helpers (`fg`, `bg`, `reset`) as private statics
- [x] `Chat.ixx` — `/model <alias> [quant]` command for hot model switching; `resolveAlias()` covers `gpt2`, `llama-1b`, `llama-3b`, `llama-8b`, and `-fp32` variants; `parseQuantization()` dispatches `none`/`fp8`/`fp4`; context length preserved across same-architecture switches, reset on architecture change; all responses fully buffered before display (streaming removed from hot path); `printGenerationStatistics()` delegates to `ConsoleRenderer::printStats()`; `/model` with no args prints current model, precision, quantization, and instruct flag
- [x] Prefill pipeline validated at FP8 — 2-phase dequant+cuBLASLt path produces coherent generation on Llama 3.2 3B Instruct; Chat CLI demo confirmed correct; TTFT ~2× faster than W8A16 fused path at target batch sizes
- [x] Greedy decode validated on standard prompts — no catastrophic divergence vs BF16 baseline

### Phase 4 — FP4 E2M1 Weight Quantization (Storage)

FP4 E2M1 (2-exponent, 1-mantissa) weight quantization for a 4× VRAM reduction vs BF16.
Native FP4 compute requires Blackwell (SM 10.0); this phase is storage-only — weights are
packed as nibbles (2 per byte) and dequantized to BF16 at inference time on the existing
W4A16 GEMM path. The per-group float32 scale infrastructure from `PerGroupInt4` carries
over directly. The dequant step in the tile load changes from INT4 unpacking to E2M1
decode; all other kernel and dispatch plumbing is reused.

`QuantizationMode::FP4` in `ChatConfig` already maps to `PerGroupInt4<128>`. This phase
replaces that INT4 backing with true FP4 E2M1 storage for better weight-distribution
fidelity, while keeping the same kernel path and VRAM footprint.

- [x] `Dnn/Quantization/Weight/Policies.ixx` — `PerGroupFp4<GroupSize=128>` policy: `kStorageDtype = UINT8`, `kScaleDtype = Float32`, `kPerChannel = false`, `kQuantizationGroupSize = GroupSize`, `kIsFp4E2M1 = true`; satisfies `WeightQuantPolicy` concept; `PerGroupInt4` updated with `kIsFp4E2M1 = false` for dispatch disambiguation; `PerGroupFp4<64>` variant added; both verified with `static_assert`
- [x] FP4 E2M1 quantization kernel — `CudaFp4WeightQuantization.cu`; BF16 → packed FP4 nibbles with per-group absmax scales; `scale[n,g] = max(|W[n,g*gs..(g+1)*gs)|) / 6.0f`; two FP4 nibbles per byte (low=even col, high=odd col); grid `(K/group_size, N)`; phase 1 parallel absmax reduction, phase 2 E2M1 encode + nibble packing; `quantize_fp4_per_group()` bridge function in `CudaLinearOp.Quantize.ixx`
- [x] W4A16 tile dequant — `fused_w4a16_gemm_kernel` extended with `kIsFp4E2M1` template bool; `if constexpr (kIsFp4E2M1)` branch: unpack nibble → `fp4_e2m1_decode()` LUT lookup (`±{0, 0.5, 1, 1.5, 2, 3, 4, 6}`) → multiply by group scale; `cuda_fp4a16_gemm()` host function dispatches `<G, true>` instantiation; `Linear.ixx` scale allocation corrected to 2D `[N, K/group_size]` for all per-group paths
- [x] `OperationTraits.Cuda.ixx` — `<Cuda, BF16, PerGroupFp4<128>>` and `<Cuda, BF16, PerGroupFp4<64>>` LinearOp specializations added
- [x] `LlamaModel::fromPretrained()` — `WeightQuantization::FP4` dispatch updated from `PerGroupInt4<128>` to `PerGroupFp4<128>`
- [x] Validated on Llama 3.2 3B Instruct — FP4 E2M1 quantized model produces coherent generation; Chat CLI demo confirmed; ~4 GB total VRAM; FP4 weights are 2× smaller than FP8 weights (4-bit vs 8-bit storage)
- [x] FP4 E2M1 decode matvec — dedicated `matvec_decode_bf16_qfp4_kernel<kGroupSize>` in `CudaMatVecBias.Bf16.cu` for the outer_size==1 decode path; 32 threads per output channel, 8 nibbles (4 packed bytes) per iteration, one per-group scale per 8-element chunk (guaranteed by `kGroupSize % 8 == 0`), warp shuffle reduction; replaces M=1 tiled GEMM fallback; 44–48 tok/s measured vs 6–7 tok/s with the tiled GEMM (~7× improvement)

### Phase 5 — Llama 3.1 8B Instruct @ FP8

Llama 3.1 8B Instruct is the first validation target where FP8 weight quantization is
practically necessary rather than an optimization. At BF16 the model requires ~16 GB
VRAM, exceeding the RTX 4070 12 GB budget. At FP8 the total footprint (weights + KV
cache + runtime overhead) is ~11.6 GB at context_length 8192 on an RTX 4070, within
the 12 GB budget. The transformer architecture is identical to Llama 3.2 3B — no new
components are required, only the config preset and weight converter mapping need
verification at the 8B parameter scale. The production default is Llama 3.1 8B at FP4
(~6 GB, ~57 tok/s decode after the warp-per-row decode-softmax rewrite); FP8 is the
validated alternative for applications requiring finer weight precision within the same
VRAM budget.

- [x] `Llama.Presets.ixx` — `Llama3_1_8B()` preset: embedding=4096, layers=32, heads=32, kv_heads=8, hidden=14336, rope_theta=500000; `LlamaModel::fromPretrained` reads all architecture dimensions from checkpoint metadata so no preset wiring is required in the load path
- [x] `convert_llama_weights.py` — extended to support `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct`; key mapping and gate/up concatenation confirmed identical to Llama 3.2; `tie_word_embeddings=False` on 8B handled by existing lm_head fallback; `rope_scaling` (`rope_type="llama3"`) printed for reference but not written — standard RoPE with `rope_theta=500000` is accurate at context lengths ≤ 4096; output: `llama31_8b_instruct_bf16.bin`
- [x] `ChatConfig` — `ModelSize::B8` added; `Chat.ixx` `switchModel()` path generation fixed: `family_str` derives `llama31` for B8 and `llama32` for 1B/3B so the correct binary filename is constructed; `llama-8b` and `llama-8b-fp32` aliases wired in `resolveAlias()`
- [x] `Llama.ixx` — `exec_context_` moved to last member declaration so it is destroyed first; `cudaStreamSynchronize()` in `releaseResources()` now fires before any tensor `cudaFree()` calls, fixing undefined behaviour during model destruction where stream callbacks fired after their device allocations had been freed
- [x] `Chat.ixx` — `switchModel()` destroys the current model via `std::visit([]( auto& m ) { m.reset(); }, model_)` before allocating the replacement, eliminating the transient old+new VRAM peak (~12.83 GB for 3B BF16 + 8B FP4) that caused WDDM shared memory spill and 4 tok/s warm-up on RTX 4070
- [x] `main.cpp` — `llama_weights_path()` corrected to use `llama31` family prefix for `ModelSize::B8` (was always emitting `llama32_8b_…`, which does not exist); chat app default updated to Llama 3.1 8B FP4 (`session.json` and `kDefaultQuantizationMode` aligned)
- [x] Prefill pipeline validated at FP8 — coherent generation confirmed on Llama 3.1 8B Instruct; both clean initial load and hot `/model llama-8b fp8` switch validated on RTX 4070
- [x] Greedy decode validated on standard prompts — no catastrophic divergence vs BF16 baseline; stale pointer root cause identified and fixed (cached `dequant_weight_buffer_` dangled after `getDeviceScratchBuffer()` grow-realloc during layer construction)

### Phase 6 — PretrainedModelReader Bulk I/O

The serialized per-tensor `fstream` read loop in `PretrainedReader.ixx` issues one read
call per tensor blob (224+ for Llama 3.1 8B), capping effective throughput at ~2 GB/s
against a PCIe 4.0 NVMe floor of ~7 GB/s. Llama 3.1 8B (~15.7 GB) loads in ~8s; the
hardware minimum is ~2.2s. The fix applies to all models.

Replace with `CreateFileMapping` / `MapViewOfFile` (Windows primary target): tensor blobs
become zero-copy pointers into the mapped region with no seek-per-tensor overhead. The
`ITensorBlob` interface (`blob.data()`, `blob.getMetadata()`) must remain stable; only the
reader implementation changes.

- [ ] `PretrainedReader.ixx` — replace `std::fstream` per-tensor read loop with `CreateFileMapping` + `MapViewOfFile`; `TensorBlob::data()` returns a pointer into the mapped view; no heap allocation per tensor
- [ ] `CudaPinnedMemoryResource` path — confirm pinned host staging is still used for the async H2D DMA path; mapped memory itself need not be pinned if a single async `cudaMemcpyAsync` from the view is issued in `loadParameter()`
- [ ] Validated on Llama 3.1 8B FP4 — load time target: < 3s on PCIe 4.0 NVMe; no regression on 3B models

---

## Beta.1 — 0.2.2 — Planned

**Qwen 3 transformer architecture with thinking mode and model-agnostic tool calling,
validated on Qwen 3 8B Instruct at BF16 and FP8. FP8 KV cache compression introduced
and validated on Qwen 3 8B.**

Beta.1 adds Qwen 3 as Mila's second supported architecture family. The Qwen 3 dense
decoder shares Mila's existing building blocks — RMSNorm, SwiGLU, GQA, RoPE — so the
model component is a thin addition on the established Llama foundation. The primary new
work is in the Chat layer: the ChatML prompt template, model-agnostic `ToolCallParser`,
and thinking mode token suppression.

The FP8 quantization infrastructure delivered in Alpha.5 is exercised on Qwen 3 8B,
providing a second independent architecture validation at a scale where VRAM constraints
are meaningful. Qwen 3 8B at FP8 targets approximately 9–10 GB VRAM, within the RTX
4070 12 GB budget.

FP8 KV cache compression is introduced in Beta.1 as a symmetric K/V policy
(`PerChannelKvFp8<>`). The `KvCachePolicy` extension point and `PerChannelKvFp8<>`
policy struct are already in place from Alpha.5. Qwen 3 8B is the appropriate
validation target — at this scale and context length the KV cache is large enough for
compression to be practically meaningful. Combined with FP8 weight quantization, this
is the primary VRAM lever for fitting larger models at longer contexts.

Success criterion: Greedy decode of Qwen 3 8B Instruct at BF16 and FP8 each match
HuggingFace token-for-token on identical prompts. Tool calling validated end-to-end
using the model-agnostic pipeline. Thinking mode token suppression confirmed in the
Chat CLI. FP8 KV cache compression produces acceptable output quality degradation
relative to the BF16 baseline on Qwen 3 8B.

### Phase 1 — Qwen 3 Transformer Component

- [ ] `Qwen3Config` — extends `LlamaConfig`; presets for 8B: embedding=4096, layers=36, heads=32, kv_heads=8, hidden=14336, vocab=151936, rope_theta=1000000
- [ ] `Qwen3Transformer` — new decoder-only network component, peer to `LlamaTransformer`; reuses `LlamaBlock` unchanged
- [ ] `Qwen3Model` — `fromPretrained()` + `generate()`; mirrors `LlamaModel`
- [ ] `Qwen3.Presets.ixx` — `Qwen3_8B()` preset

### Phase 2 — Qwen 3 Tokenizer and Weight Converter

- [ ] `BpeTokenizer::loadQwen3` — tiktoken encoding with Qwen 3 vocabulary (`vocab_size=151936`); registers `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, `<tool_call>`, `</tool_call>` as atomic special tokens
- [ ] `convert_qwen3_weights.py` — HuggingFace Qwen 3 checkpoint key mapping to Mila binary format; always writes BF16
- [ ] `ChatConfig` — add `ModelType::Qwen3`; wire `BpeTokenizer::loadQwen3` and `Qwen3Model` selection

### Phase 3 — Model-Agnostic Tool Calling and ChatML Template

- [ ] `ToolCallParser` — refactored to support pluggable boundary strategies; `Llama32Strategy` wraps existing `<|python_tag|>` / `<|eom_id|>` logic; `Qwen3Strategy` detects `<tool_call>` / `</tool_call>` boundaries
- [ ] `MessageFormatter` — add `Qwen3ChatTemplate`; ChatML format: `<|im_start|>{role}\n{content}<|im_end|>`; tool call turns emit `<tool_call>{json}</tool_call>`; tool result turns emit `<|im_start|>tool\n<tool_response>{content}</tool_response><|im_end|>`
- [ ] Stop token handling — generation halts on `<|im_end|>`
- [ ] Thinking mode — `ThinkingFilter` streams tokens to the Chat CLI output, suppressing content between `<think>` and `</think>` inclusive; thinking content discarded by default, available via `--show-thinking` flag

### Phase 4 — FP8 KV Cache Compression

Symmetric per-head per-token FP8 compression of the K and V cache tensors in
`CudaGqaOp`. The `KvCachePolicy` extension point and `NoKvCompression` identity were
established in Alpha.5 Phase 1. This phase activates the first real policy.

Scale granularity is per-head per-token (one float32 scale per head per cached token
for both K and V). This is coarser than per-channel weight quantization but appropriate
for the dynamic, growing shape of the KV cache. K and V use the same policy
symmetrically — asymmetric compression is not a current target.

- [x] `Dnn/Quantization/KvCache/QuantPolicy.ixx` — `PerChannelKvFp8<>` policy struct: `kStorageDtype = FP8_E4M3`, `kScaleDtype = Float32`, `kPerHeadPerToken = true`; satisfies `KvCachePolicy` concept (completed in Alpha.5 Phase 2)
- [ ] `OperationTraits.Cuda.ixx` — `<Cuda, BF16, PerChannelKvFp8<>>` GQA specialization is already in place from Alpha.5 Phase 1; confirm dispatch wires through to `CudaGqaOp<BF16, PerChannelKvFp8<>>`
- [ ] `CudaGqaOp` — extend template signature to `<TComputePrecision, TKvPolicy>`; select quantized vs passthrough cache kernels via `if constexpr (TKvPolicy::kIsActive)`; non-quantized path is unchanged
- [ ] KV cache scale tensor allocation — `GroupedQueryAttention::build()` allocates `k_scale_` and `v_scale_` tensors (shape `[num_kv_heads, max_seq_len]`, dtype FP32) when `TKvPolicy::kIsActive`; lifetime mirrors the KV cache tensors
- [ ] KV cache write kernel (prefill) — quantizes K and V from BF16 to FP8_E4M3 on each prefill chunk write; computes `scale[head, token] = max(abs(x[head, token, :])) / 448.0f` per head per token; writes FP8 values and FP32 scales to cache
- [ ] KV cache write kernel (decode) — same quantization logic for the single-token append on each decode step; scale computation per head for the new token only
- [ ] KV cache read kernel — dequantizes K and V from FP8_E4M3 back to BF16 before attention score and weighted-sum computation; applies stored per-head per-token scales
- [ ] `CudaGqaOp::setParameters()` — accept optional `k_scale_` and `v_scale_` tensor pointers when `TKvPolicy::kIsActive`
- [ ] Validated on Qwen 3 8B Instruct — confirm VRAM reduction fits within 4070 12 GB budget with both weight FP8 and KV cache FP8 active; measure VRAM at BF16, weight-FP8-only, and weight-FP8 + KV-FP8 configurations; acceptable degradation criterion: no catastrophic token divergence on standard prompts

---

## Beta.2 — 0.2.3 — Planned

**Ministral transformer architecture with Sliding Window Attention, validated on Ministral
3B Instruct at BF16 and Ministral 8B Instruct at FP8.**

Beta.2 introduces the Ministral transformer as a new first-class component built on the
Llama 3.2 foundation. The primary architectural addition is Sliding Window Attention (SWA),
used on interleaved layers in the Ministral 8B model. The FP8 quantization infrastructure
delivered in Alpha.5 is applied directly to Ministral 8B, bringing it within the 12 GB
VRAM budget of consumer Ada Lovelace GPUs (validated at context_length = 2048: ~10.2 GB
total including KV cache and runtime overhead on an RTX 4070).

Ministral 3B has no SWA and uses standard global GQA, making it a clean BF16 validation
gate before the combined SWA + FP8 work is exercised on the 8B model. The model-agnostic
tool calling pipeline and `ToolCallParser` strategy pattern delivered in Beta.1 apply
directly here via a `MistralStrategy`.

Success criterion: Greedy decode of Ministral 3B Instruct at BF16 and Ministral 8B Instruct
at FP8 each match HuggingFace token-for-token on identical prompts. Tool calling validated
end-to-end on both models using the model-agnostic pipeline from Beta.1.

### Phase 1 — Ministral Transformer Component

- [ ] `MinistralConfig` — extends `LlamaConfig`; adds `withSlidingWindowSize()` setter and `getSlidingWindowSize()` getter; `0` sentinel disables SWA
- [ ] `MinistralBlock` — extends `LlamaBlock`; even-indexed layers use global GQA, odd-indexed layers use SWA
- [ ] `GroupedQueryAttention` — add SWA masking support; SDPA kernel restricts causal mask to last `W` tokens per query position; CPU and CUDA parity required
- [ ] `MinistralTransformer` — new decoder-only network component, peer to `LlamaTransformer`; alternates block construction based on layer index and `sliding_window` config value
- [ ] `Ministral.Presets.ixx` — `Ministral3B()` preset: embedding=2048, layers=36, heads=16, kv_heads=8, hidden=6144, vocab=32768, rope_theta=1000000, no SWA
- [ ] `Ministral.Presets.ixx` — `Ministral8B()` preset: embedding=4096, layers=36, heads=32, kv_heads=8, hidden=14336, vocab=32768, rope_theta=1000000, sliding_window=1024

### Phase 2 — Mistral Tokenizer and Weight Converter

- [ ] Mistral v3 tokenizer support — `vocab_size = 32768`; distinct from Llama and Qwen tiktoken vocabularies
- [ ] `convert_ministral_weights.py` — Mistral HuggingFace checkpoint key mapping; gate/up concatenation follows same pattern as Llama converter; always writes BF16
- [ ] `ChatConfig` — add `ModelType::Ministral`; wire Mistral instruct chat template and `MistralStrategy` for `ToolCallParser`

### Phase 3 — Ministral 3B Instruct BF16 Validation

- [ ] Prefill pipeline validated at BF16 — logits match HuggingFace on identical prompts
- [ ] Full-network greedy decode validated token-for-token against HuggingFace
- [ ] Tool calling validated end-to-end using model-agnostic pipeline from Beta.1

### Phase 4 — Ministral 8B Instruct FP8 Validation

- [ ] Prefill pipeline validated at FP8 — logits match BF16 baseline on identical prompts
- [ ] Full-network greedy decode validated token-for-token against BF16 baseline
- [ ] Tool calling validated end-to-end using model-agnostic pipeline from Beta.1

---

## Alpha.4 — Complete

**Instruction following and tool calling, validated on Llama 3.2 3B Instruct at BF16.**

Alpha.4 delivers the structured message and tool calling infrastructure in the Chat
application layer. No model architecture changes are required — Llama 3.2 3B Instruct
shares the same weight layout as the base model validated in Alpha.3, and the converter
already supports the instruct variant. The work is entirely in the Chat layer above the
model.

Success criterion: Llama 3.2 3B Instruct produces correct tool call responses
end-to-end through the structured message pipeline in the Chat application.

### Phase 1 — Structured Message Infrastructure

- [x] Verify `BpeTokenizer::loadLlama32` encodes Llama 3.2 special tokens as single atomic token IDs (`<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`, `<|eom_id|>`)
- [x] `ChatMessage` — role (system / user / assistant / tool), content, optional tool calls
- [x] `MessageFormatter` — applies Llama 3.2 instruct chat template to a message sequence
- [x] Stop token handling — generation halts on `<|eot_id|>` and `<|eom_id|>`
- [x] `Chat::run()` — replace raw string history with structured `ChatMessage` history

### Phase 2 — Tool Calling Framework

- [x] `ToolDefinition` — name, description, JSON schema parameters
- [x] `ToolCall` — parsed tool name and arguments extracted from model output
- [x] `ToolCallParser` — detects `<|python_tag|>` boundary, extracts and validates JSON
- [x] System prompt builder — injects active `ToolDefinition` list into the system message
- [x] `ChatConfig` — add optional `system_prompt` and `tools` list
- [x] `Chat::registerTool()` — bind named handlers; `Chat::run()` dispatches tool call round-trip

### Phase 3 — Llama 3.2 3B Instruct Validation

- [x] Convert Llama 3.2 3B Instruct weights — `convert_llama_weights.py` already supports the instruct variant, confirm output
- [x] Instruct format validated — greedy decode with structured prompt matches expected assistant response format
- [x] Tool call round-trip validated end-to-end — model issues a tool call, result is fed back, final response is correct

---

## Alpha.3 — Complete

**BF16 compute backend, validated against HuggingFace using the same methodology as FP32.**

BF16 is Mila's primary reduced-precision compute target. It matches FP32's exponent
range, avoiding the overflow and underflow risks of FP16, while halving memory
bandwidth relative to FP32. FP16 is not a Mila target.

Success criterion: Greedy decode of LlamaModel matches HuggingFace LlamaForCausalLM
token-for-token on identical prompts using Llama 3.2 3B weights at BF16.

| Item | Status |
|---|---|
| CUDA BF16 kernels for GQA pipeline components | Complete |
| BF16 dispatch wired through compute backend | Complete |
| convert_llama_weights.py — extend for Llama 3.2 3B weight layout | Complete |
| Prefill pipeline validated at BF16 — logits match HuggingFace on identical prompts | Complete |
| Full-network greedy decode validated token-for-token against HuggingFace | Complete |

---

## Alpha.2 — Complete

**Llama architecture, validated against HuggingFace using the same methodology.**

Success criterion: Greedy decode of LlamaModel matches HuggingFace LlamaForCausalLM
token-for-token on identical prompts using Llama 3.2 1B weights at FP32.

| Item | Status |
|---|---|
| TokenEmbedding — pure vocabulary lookup, wte only | Complete |
| RoPE — rotary positional encoding applied to Q and K inside attention | Complete |
| SiLU activation — forward + CUDA kernel | Complete |
| SwiGLU MLP — gate_proj * silu(up_proj) then down_proj | Complete |
| GroupedQueryAttention — GQA with configurable num_kv_heads and KV-cache path | Complete |
| LlamaBlock — pre-RMSNorm, GQA, SwiGLU MLP, residual connections | Complete |
| LlamaTransformer — decoder-only stack | Complete |
| LlamaModel — fromPretrained() + generate(), mirrors GptModel | Complete |
| LlamaConfig — rope_theta, rms_norm_eps, num_kv_heads, intermediate_size | Complete |
| convert_llama_weights.py — HuggingFace to Mila binary format | Complete |
| SentencePiece support for Llama 3.x tokenization | Complete |
| Prefill pipeline validated — logits match HuggingFace on identical prompts | Complete |
| Full-network greedy decode validated token-for-token against HuggingFace | Complete |

---

## Alpha.1 — Complete

**GPT-2 inference validated against HuggingFace.**

The full GPT-2 decoder stack is implemented and confirmed correct. Greedy decode
matches HuggingFace token-for-token. This establishes the validation methodology
that all subsequent architecture work follows.

| Item | Status |
|---|---|
| Core components — Linear, LayerNorm, MHA, MLP, Residual, GELU | Complete |
| CUDA and CPU kernels for all components | Complete |
| GptTransformer — decoder-only, pre-LN | Complete |
| GptModel — fromPretrained() + generate() | Complete |
| Two-phase KV-cache — prefill + decode | Complete |
| HuggingFace GPT-2 weight converter | Complete |
| BPE tokenizer | Complete |
| Greedy decode validated token-for-token against HuggingFace | Complete |
| Chat CLI sample | Complete |
| AdamW optimizer + MNIST training loop | Complete |

---

## Beta — 0.2.1

**Public release milestone.**

Beta is reached when GPT-2 and Llama inference are validated across FP32, BF16, FP8,
and FP4, tool calling is validated on Llama 3.2 3B and 3.1 8B Instruct, and the
library is stable enough for external contributors to work with confidently.

| Item | Required |
|---|---|
| Llama 3.2 1B FP32 validated against HuggingFace | Yes |
| Llama 3.2 3B BF16 validated against HuggingFace | Yes |
| Llama 3.1 8B Instruct FP8 validated against BF16 baseline | Yes |
| Tool calling validated on Llama 3.2 3B and 3.1 8B Instruct | Yes |
| API documentation complete and published | Yes |
| CPU reference ops — GPT-2 (Alpha.1) lineage only; Llama/Qwen/Ministral CPU ops are contributor-driven, not a gate | Scoped |
| Debug instrumentation fully gated or removed | Yes |
| Test coverage of core components | Yes |
| CONTRIBUTING.md with coding standards | Yes |
| getting-started.md onboarding guide (user-first, contributor superset) | Yes |
| `find_package(Mila)` packaging validated by an external consumer build | Yes |
| Published Docker runtime image (slim multi-stage GPU runtime, release-tagged) | Yes |
| Ungated GPT-2 quick-start path for zero-auth first run | Yes |
| good-first-issue labels on GitHub | Yes |

**Compute backend scope.** Mila is GPU-first. The CUDA backend is the validated, supported
inference path; correctness is established token-for-token against HuggingFace, which serves as
the reference oracle (the original llm.c-derived CPU path is no longer needed for that role). The
CPU backend is retained as the always-available baseline and contributor on-ramp, but full CPU
op parity across architectures is explicitly **not** a beta gate. GPT-2 (the Alpha.1 / llm.c
lineage) keeps its CPU reference; Llama, Qwen, and Ministral are CUDA-first, and their CPU ops are
filled in by contributors as demand arises. This is safe by construction: compile-time
`OperationTraits` dispatch means a missing CPU op costs nothing on the GPU path and produces a
localized compile error — never a silent wrong answer — if a CPU model is instantiated without it.
The coverage matrix below marks these gaps as intentional good-first-issue work rather than hidden
incompleteness, and the user-facing model paths stay GPU so an end user never hits an unimplemented
CPU op.

CPU op coverage by component (the CUDA backend implements every row). Legend: ✅ wired via
`OperationTraits`; ◐ implemented but still on the legacy registry/typemap dispatch (migration
tracked in Phase 1); — not implemented (contributor opportunity).

| Component | Lineage | CPU | Notes |
|---|---|---|---|
| GELU | GPT-2 / shared | ✅ | |
| Residual | GPT-2 / shared | ✅ | |
| Softmax | GPT-2 / shared | ✅ | |
| LayerNorm | GPT-2 | ◐ | `CpuLayerNormOp` exists; not yet `OperationTraits`-wired |
| Linear | GPT-2 / shared | ◐ | `CpuLinearOp` exists; still on `CpuLinearOpTypeMap` |
| MultiHeadAttention | GPT-2 | ✅ | |
| LPE (learned positional) | GPT-2 | ✅ | `CpuEncoderOp` |
| AdamW (optimizer) | shared | ✅ | `CpuAdamWOptimizer` |
| SoftmaxCrossEntropy (loss) | training | — | `CpuSoftmaxCrossEntropyOp` exists but is not built/wired |
| TokenEmbedding | Llama | — | GPT-2 embedding is folded into `CpuEncoderOp` |
| RoPE | Llama | — | good-first-issue |
| RmsNorm | Llama | — | good-first-issue |
| SwiGLU (SiLU) | Llama | — | good-first-issue |
| GroupedQueryAttention | Llama | — | good-first-issue |

Net: the GPT-2 lineage runs end-to-end on CPU; the Llama lineage is CUDA-only until a contributor
adds the five Llama CPU ops above.

**Distribution.** Beta introduces a published Docker image so users can run Mila without
standing up the bleeding-edge build toolchain. Two images, two roles: a slim multi-stage
**runtime** image (built in the CUDA `-devel` base, artifacts copied into a `-runtime`
base) for users, and the existing `-devel` **dev container** for contributors. Images are
published on release tags (not a rolling `:latest` divorced from releases) to keep
maintenance bounded for a solo maintainer. Gated model weights are never baked into the
image — they remain a user-supplied, offline conversion step mounted in at run time; the
ungated GPT-2 path provides an out-of-the-box first run with no HuggingFace auth.

Pre-converted Mila-format weights for permissively-licensed models (GPT-2 first; Qwen and
Mistral as they land) are hosted in a public Hugging Face repository and fetched on first
run via direct `resolve/` URLs over HTTPS — no Python, venv, or HuggingFace auth at runtime.
The weight blob already carries a format magic and `version` (see `PretrainedReader`), so
hosted artifacts are versioned against the Mila format and an incompatible build fails loudly
rather than mis-loading; the writer (`common.py`) and reader `VERSION` constants must be kept
in sync and a re-publish is required on any format bump. Llama and other gated weights stay a
user-supplied offline conversion step — redistributing them would transfer Meta's license
obligations (attribution, Acceptable Use Policy, gating) onto the project.

**Packaging.** A downstream app consuming Mila via `find_package(Mila)` currently fails to
build. Because C++23 module interface units cannot ship as portable BMIs, the consumer's
toolchain recompiles the installed `.ixx` units, and each one pulls its kernel header via a
file-relative quoted include (`#include "Kernels/Gelu.cuh"`, `"../Common/Kernels/CudaAttention.cuh"`,
`"../../Deps/nlohmann/json.hpp"`). On install these resolve against the wrong tree: the module
units land under `include/mila/modules/Src/...` while the kernel headers are installed by a
separate `install(DIRECTORY Src/ FILES_MATCHING *.cuh *.h)` glob under `include/Dnn/...`, and
the vendored `Deps/` plus the generated `Version.h` are not installed at all.

The include strings are only the visible symptom; the real defect is in how the `Mila` target
is composed in `Mila/CMakeLists.txt`. The kernel `.cuh`/`.h` headers are listed as raw
`add_library` sources, which carry no base directory, no install rule, and no usage
requirement — which is precisely why packaging had to bolt on the ad-hoc directory glob that
splits headers from modules. The CUDA `.cu`/`.cuh` sources are added unconditionally even
though `enable_language(CUDA)` and the CUDA `.ixx` module file set are both gated on
`MILA_HAS_CUDA`, so a CPU-only configure is incoherent. And three distinct categories are
flattened into one list: `.cu` files (per-precision explicit instantiations compiled into the
archive — private, link-only, must not ship), `.cuh`/`.h` headers (declarations the installed
`.ixx` units include at consumer-recompile time — must ship), and `.ixx` interface units (the
only category currently modeled, via a file set).

The fix is a single coherent restructuring, not a destination patch: model the headers as a
`FILE_SET HEADERS TYPE HEADERS BASE_DIRS Src` so they gain a base dir and install semantics;
migrate the file-relative quoted includes to angled includes anchored at that one `Src` root
(give vendored `Deps` its own root so nlohmann becomes `<nlohmann/json.hpp>`); set `BASE_DIRS`
on the `CXX_MODULES` file sets to the same `Src` root; move all CUDA `.cu`/`.cuh` sources under
the `if(MILA_HAS_CUDA)` block via `target_sources`; replace the `install(DIRECTORY …)` glob with
`install(TARGETS Mila … FILE_SET HEADERS)`. The single include root must be on Mila's *own*
build path (the current root at `Mila/CMakeLists.txt:128` is INTERFACE-only, so it has to become
PUBLIC or gain a PRIVATE entry, or the in-tree build breaks once includes are anchored), and the
generated `Version.h` and `Deps/` must be installed alongside the modules. Validate with a
throwaway `find_package(Mila)` + `import Mila;` consumer wired into CI — Mila's own CI stays green
throughout and will not catch packaging regressions on its own. Suggested sequencing: convert one
CUDA op to angled includes and get the in-tree build green first (proves the root/`-I` model), then
bulk-convert backend-by-backend (the compiler flags every missed header), then do the install-side
CMake and the consumer test last.

Deferred to later in the Beta push: whether the kernel `.cuh` *declarations* belong in the public
install surface at all. Because the kernels are explicitly instantiated per precision in `.cu`
files compiled into the archive, consumers link the kernel symbols and only need the declarations
to call the launch wrappers — so the shippable surface may be reducible. That is a separate
architectural decision and is intentionally out of scope for the packaging fix above.

**Module Hygiene — Includes/Imports and Doxygen.** Over the course of alpha the module surface
has accumulated `#include`s and `import`s that are no longer required, and Doxygen comments that
have drifted out of sync with the code. Both are large, mechanical, low-risk-per-edit but
high-volume diffs, and both were deferred until a cross-compiler build environment was stood up
— a hard prerequisite for the include work, not a convenience. As of 0.13.39-alpha.5 the native
WSL / Ubuntu 26.04 Clang build is green (Clang 21 + CUDA 13.3 + gcc-15 host), so the Clang oracle
now exists locally and these phases are un-gated; GCC 16 (the second oracle) and the dev-container
build remain to be validated. Current surface: 287 `.ixx` module units, ~1,810 `import` lines, ~1,419 `#include`
lines (252 files use a global module fragment), and ~1,950 `@brief` / ~1,100 `@param` / ~257
`@tparam` / ~218 `@file` Doxygen tags across 258 files.

*Includes and imports.* There is no reliable off-the-shelf tool for C++23 module `import`
cleanup — IWYU and clangd do not understand the module graph — so the compiler is the only
ground-truth oracle. The critical trap is MSVC transitive resolution: a line can be removed and
MSVC still compiles because the symbol arrives transitively, which means "still builds on MSVC"
does *not* prove the line was unused, and can silently convert a real dependency into a fragile
implicit one. The honest oracle is a **Clang or GCC** build, which is exactly why this work waited
for the Linux toolchain — now available via the green WSL Clang build, with GCC 16 still to come. The cruft is already real and visible — even `Linear.ixx`,
the dispatch reference file, imports `Dnn.TensorOps` twice. Phasing:

- [ ] Phase 0 — exact-duplicate `import`/`#include` dedup within each file; pure text analysis, scriptable across all 287 units, zero compile cost and zero risk
- [ ] Phase 1 — candidate report (no edits): heuristic scan flagging imports/includes whose symbols never appear in the file body; over-reports by design (cannot see macro/transitive use), so it is a worklist to size the job, not a verdict
- [ ] Phase 2 — compiler-verified removal, leaf modules first: scripted remove -> rebuild -> revert-on-failure, batched per file with binary-search on failures, verified against Clang/GCC rather than MSVC so visible cruft is not traded for invisible transitive coupling

*Doxygen staleness.* Stratified by confidence and tooling:

- [ ] Tier 1 — `@file` rename drift: 34 files whose `@file` tag does not match the actual filename (e.g. `RocmDevice.ixx` tagged `@file VulkanDevice.ixx`, `CudaMhaOp.ixx` tagged `@file CudaAttentionOp.ixx`, `Lpe.ixx` tagged `@file Gpt2Encoder.ixx`). Pure rename leftovers; the correct value is `basename`, so this is fully scriptable with no judgment
- [ ] Tier 2 — `@param`/`@tparam` name mismatches: documented parameter/template names that no longer appear in the signature (renamed or removed). Mechanical and high-confidence, but module/template signatures span lines, so build a detector that emits a candidate list for review before batch-fixing; the actual mismatches are a small fraction of the ~1,100 `@param` + ~257 `@tparam`
- [ ] Tier 3 — semantic staleness (needs judgment, per-subsystem): `@brief`/descriptions that still describe the retired world — components "registering with `OperationRegistry`" or "deriving from `UnaryOperation`/`BinaryOperation`", string-keyed dispatch references, naming drift (`TWeightQuant` in prose vs. the spelled-out style), and file-level `@brief`s exceeding the 1-3 sentence rule. Done one settled subsystem at a time; subsystems still mid-refactor (notably the `OperationTraits` dispatch migration) are left alone until the refactor lands, to avoid re-staling the prose

**Public API Surface — Narrowing the `Mila` Umbrella.** The supported public entry point is a
single `import Mila;` — confirmed as the sole public surface, by design. Consumers import the
umbrella and nothing else; the internal module names (`Dnn.*`, `Compute.*`, etc.) are an
implementation detail of the source tree, not part of the consumer contract, which is also why
they are intentionally *not* prefixed with `Mila.` (the namespace root `Mila::` already provides
symbol-level scoping; the module-name layer is a private implementation concern as long as the
umbrella is the only door). Tests and samples import submodules directly (14 direct imports in
the test tree today) and are explicitly not bound by the public contract.

The mechanism is correct; the open work is *scope*. At an API freeze the two failure modes are
asymmetric: an umbrella that is too narrow is widened later by *adding* exports (non-breaking),
while an umbrella that is too broad can only be corrected by *removing* exports (breaking every
consumer that reached for the symbol). Beta should therefore freeze the narrowest defensible
surface, not the widest. Today `Mila.ixx` re-exports essentially the entire module tree, which
locks in two costs: (1) every consumer recompiles the full re-exported transitive closure into
BMIs — inference-only adopters pay the compile cost of the training/visualization/serialization
subtrees, because BMIs are not portable (see Packaging) and `export import` pulls the whole graph;
(2) every re-exported symbol becomes a frozen compatibility promise, including the legacy paths
currently being deleted. The umbrella *is* the API specification — there is no "exported but not
really public" once it is frozen.

- [ ] Define an explicit public allowlist for `Mila.ixx` — the inference surface (models, components, tensors, execution context, `initialize`/`shutdown`, tokenizers) is what beta promises; treat the export list as the literal API spec
- [ ] Demote non-public modules to unexported internal modules (still directly importable by tests/samples, just not re-exported through the umbrella): `OperationRegistry`/`OperationRegistryHelpers`/`OperationsRegistrar`, `UnaryOperation`/`BinaryOperation` (both slated for removal), `Dnn.TensorBuffer` (marked "remove after testing"), and the per-device operation modules
- [ ] Stop re-exporting the vendored `nlohmann` module/namespace through the public surface — it hands a breaking change to a third party's release schedule; the Chat sample's direct `import nlohmann.json` is a sample-layer concern, not a Mila public-API one
- [ ] Domain-qualify generic single-segment module names that are global-collision magnets on co-link — `Core`, `Utils`, `Components`, `Profiling` (e.g. `Dnn.Core`, `Dnn.Utils`); this is targeted (a handful of renames) and independent of the no-`Mila.`-prefix rule, which stands for the specific multi-segment names
- [ ] Deferred-but-non-breaking: if training becomes a first-class public concern, add a separate `Mila.Training` umbrella rather than widening `Mila` — the additive direction keeps the inference surface tight

**Release Assets and CI.** Mila is a source-distributed C++ library: contributors clone, users
consume it via `find_package(Mila)` built from a source install. That distribution model means
most "release asset" machinery is unnecessary — GitHub auto-generates source `.zip`/`.tar.gz` for
every tag, so **tagging `master` is the release**; there is no need for a release workflow unless
prebuilt binaries are shipped. The release flow is a `dev` -> `master` PR (dev is the interim
workspace); CI validates on that PR, and the documentation site publishes only from `master`.

During alpha the **GitHub default branch is `dev`**: all work lands on dev (the CI-gated trunk),
releases are infrequent, and the audience is followers/contributors who already target dev, so a
no-argument clone and the repo home page should show the live project rather than a lagging master.
At beta this flips — **switch the default branch to `master`** so the front door is the stable,
released artifact that matches the canonical identity (tags, docs, and the CPM/FetchContent
by-semver path all key on master). The README and roadmap links are branch-agnostic, so the switch
needs no content changes.

The genuinely GitHub-bound deliverable is the **documentation site** (GitHub Pages can only serve
from a GitHub source). Decisions:

- [ ] Docs are generated by a GitHub Action, never committed to the source tree — Doxygen output for 287 modules with call graphs is thousands of files plus binary graph images; committing it per release poisons the 14.8 MB source repo with noisy, conflict-prone history
- [ ] The docs job is decoupled from the build — Doxygen is a source parser (`EXTRACT_ALL` reads `.ixx`/`.cuh` directly) and needs no compiled library, no CUDA, no GPU, no clang. The current job downloads the multi-GB build tree and runs `cmake --build --target docs` against a foreign CMakeCache; the correct job is checkout master -> install `doxygen`+`graphviz` -> run Doxygen -> publish, with no build dependency
- [ ] Publish via Actions-native Pages (`actions/upload-pages-artifact` + `actions/deploy-pages`) rather than the current `gh-pages` orphan branch + `JamesIves` action — no branch to manage, and avoids the missing `permissions: contents: write` that can silently no-op the deploy; trigger on push to `master`
- [ ] Narrow what the docs expose to match the public API surface — current Doxygen config sets `EXTRACT_ALL`/`EXTRACT_PRIVATE`/`EXTRACT_STATIC` recursively over all of `Mila/Src`, producing an undifferentiated dump where the `import Mila;` public surface is indistinguishable from internals (see Public API Surface item); the published docs should show the public API, not every private member of 287 modules
- [ ] Verify Doxygen renders C++23 module units faithfully — module support is young; `export module`/partitions/`import` may misrepresent structure. Depends on the Doxygen staleness pass (Module Hygiene) so the generated docs are not loud with `WARN_NO_PARAMDOC` warnings and stale `@param`/`@file` drift

CI correctness — the existing `build-pipeline.yml` produces a green badge that both overstates and
understates reality, which is the opposite of what a beta trust signal should do:

- [ ] GPU test honesty (highest priority) — the `test` job runs on a bare GitHub-hosted `ubuntu-24.04` runner with no NVIDIA GPU and only `libgtest-dev` installed, while the build ran inside the `nvidia/cuda` container. CUDA test executables either fail to load (`libcudart.so` absent) or fail at runtime (no device); if any are silently skipping on "no device" that is *false green*, worse than red. Fix: a self-hosted GPU runner, or explicitly partition CPU-runnable tests from GPU-required ones so the badge means what readers assume
- [ ] Stop passing the whole configured `build/` tree between jobs — it is gigabytes, environment-specific, and consumed cross-environment (the `docs` and `test` jobs run on differently provisioned runners against a CMakeCache full of absolute tool paths that do not exist there); decoupling docs from the build (above) removes most of this coupling
- [ ] Wire ccache as the compiler launcher (`-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`) — `CCACHE_DIR` is set and cached but never used, so builds are not actually accelerated
- [ ] Broaden compiler coverage toward the supported matrix — CI builds only Clang 19; the primary dev compiler (MSVC 2026) and the working GCC 16 path are untested, so the compiler that previously broke the build (the VS 2026 pre-18.6.2 module regression) is the one CI cannot catch. A multi-compiler CI is also the cross-compiler oracle the deferred include/import hygiene pass needs (see Module Hygiene)

Docker image publish is optional and only if the runtime image stays a beta deliverable (see
Distribution) — building and pushing a release-tagged image to GHCR is a natural CI job on tag,
but is equally a local `docker build && docker push`; it is automation-of-convenience, not a gate.
The `find_package` consumer build itself is tracked under Packaging, not here.

**Project Hygiene and Contributor Readiness.** A beta is a trust signal to users and contributors;
these items are about the project not contradicting itself or wasting a newcomer's first hour.

- [x] License reconciliation (must-fix) — DONE 2026-06-08. Repo had stated the license **four** different ways: `License.md` (MIT), `Mila/Src/Mila.ixx` + `Version.ixx` (proprietary "EULA / All rights reserved" headers), `README.md` ("Apache License 2.0"), and `CONTRIBUTING.md` ("Apache License 2.0"). All reconciled to MIT: the two public-entry source files now carry an SPDX header (`SPDX-License-Identifier: MIT` + `Copyright (c) 2021..2026 Todd J. Thomson`) plus the standard `@file`/`@brief` block they were previously missing; README and CONTRIBUTING corrected to MIT and pointed at `License.md` (was the non-existent `LICENSE`); holder unified to `Todd J. Thomson` (dropped retired "Achilles Software" trade name); `License.md` copyright bumped to `2021..2026`. SPDX-on-two-files is the chosen convention; all other source files remain header-free by convention (root `License.md` governs)
- [ ] Formatter/linter config (highest-ROI scaffolding) — there is no `.clang-format`, `.editorconfig`, or `.clang-tidy`, so the idiosyncratic style in `CLAUDE.md` (no column alignment, blank-line-before-control-flow, full-word identifiers, ASCII-only comments) is unenforceable and reviews drown in whitespace nits. Add `.clang-format` + `.editorconfig` (even if they cannot capture every rule) so style is machine-checkable in CI rather than tribal knowledge
- [ ] GitHub community-health files — `.github/` has only the workflow and copilot-instructions; add `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue templates, and a PR template to complete GitHub's community-standards checklist (pairs with the existing good-first-issue labels requirement)
- [ ] FIXME/TODO debt triage — the source carries ~71 `FIXME` + ~69 `REVIEW` + ~25 `TODO` markers (165 total); `FIXME` reads as "known broken" to anyone browsing, and several are commented-out core paths (weight initializers bypassed as "takes too long", commented `prefill`/`xavier`/`normal` calls). Triage before beta: fix the real ones, convert the rest to tracked GitHub issues, and do not ship literal "FIXME"s in public source. Distinct from the "debug instrumentation gated/removed" item, which is the `std::cout` (12 files) / `std::cerr` (5) / `printf` (6) usage
- [ ] Borderline (conscious calls, not assumed gates): convert the Beta table's "test coverage of core components" from a vibe into an actual audit listing components with zero tests; and add the Samples build to CI (currently only tests build) so a contributor's first sample build is not the thing that breaks. A `CHANGELOG` is judged unnecessary — completed ROADMAP sections plus GitHub auto-generated release notes cover it

---

## Post-Beta

Items deferred until the library has a stable contributor base.

**Training** — Full LLaMA fine-tuning pipeline. Loss function GPU migration.
Gradient checkpointing. Checkpoint save and restore.

**Architecture** — Mixture of Experts components. Speculative decoding.
Additional attention variants.

**Performance** — Flash Attention integration. Tensor parallelism.
Deterministic gradient accumulation for training reproducibility。
