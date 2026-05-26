# Mila — Roadmap

---

## Versioning

| Stage | Version | Title |
|---|---|---|
| In Progress | 0.13.32-alpha.5 | FP8/FP4 quantization pipeline — Llama 3.2 3B and 3.1 8B Instruct |
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
- [ ] `OperationTraits.Cuda.ixx` — add `GroupedQueryAttentionOp` specializations: `<Cuda, BF16, NoKvCompression>` and `<Cuda, BF16, PerChannelKvFp8<>>`; migrate `GroupedQueryAttention` component to `OperationTraits` dispatch
- [ ] `OperationTraits.Cuda.ixx` — add `SamplingOp` specializations: `<Cuda, FP32>` and `<Cuda, BF16>`; implement `TokenSampler` component and `CudaSamplingOp` per `TokenSampling.md`
- [ ] `OperationTraits.Cuda.ixx` — remaining policy-free ops: `GeluOp`, `ResidualOp`, `RmsNormOp`, `SoftmaxOp`, `SwiGluOp`, `MultiHeadAttentionOp`, `RopeOp`, `LpeOp`, `TokenEmbeddingOp`, `SoftmaxCrossEntropyOp`; migrate each component's `createOperation()` to `OperationTraits` dispatch
- [ ] `OperationTraits.Cpu.ixx` — `:Cpu` partition; matching specializations for all ops above; migrate CPU component paths
- [ ] `GroupedQueryAttention` — replace bare `TKvCache TensorDataType` parameter with `TKvPolicy` constrained to `KvCachePolicy`; `kKvCompressed = TKvPolicy::kIsActive`; `kCacheDtype` derived from policy; `NoKvCompression` is the default
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
- [x] `CudaLinearOp.ixx` — FP8 batch prefill path: 2-phase dequantize (`cuda_fp8_dequantize_to_bf16` → BF16 staging buffer) followed by standard BF16×BF16 cuBLASLt NT GEMM; bias added post-GEMM by `cuda_add_bias` to avoid Ada epilogue constraint; staging buffer allocated once in `buildCublasLtPlans()`. Native FP8 cuBLASLt (separate `data_type_A`/`data_type_B` descriptor) deferred — 2-phase is the validated production path.
- [x] W8A16 fused GEMM A/B test path — `kUseW8A16Gemm` compile-time toggle in `CudaLinearOp`; single kernel reads FP8 weights once, dequantizes per-channel inline in shared memory, accumulates in float32, writes BF16 output; eliminates BF16 staging buffer. Benchmarked: 2–3× slower than 2-phase at target batch sizes (scalar float CUDA cores vs cuBLASLt tensor cores); `kUseW8A16Gemm = false` is the default. Kernel retained as correctness reference; tensor-core WMMA upgrade is the path to a real win.

### Phase 3 — Llama 3.2 3B Instruct @ FP8

- [x] `ChatConfig` — `QuantizationMode` enum (`None`, `FP8`, `FP4`) orthogonal to `ModelPrecision`; `QuantizationMode` is the runtime quantization selector, `ModelPrecision` remains the compute-type selector
- [x] `ChatConfig` — `context_length` uncapped; no FP8-specific ceiling; `ModelSize::B8` added to enum; caller sets context length directly
- [x] Wire FP8 through `LlamaModel::fromPretrained()` — `WeightQuantization::FP8` dispatches to `fromPretrainedImpl<PerChannelFp8<>, NoKvCompression>`; compile-time only, no runtime config object
- [x] `ConsoleRenderer` (`Chat.Renderer.ixx`) — standalone non-exported module; braille dot spinner (⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏) with cursor hide/show (`\x1b[?25l/h`) to suppress blink flicker; solid color response blocks (`bg(40,44,60)` / `fg(200,215,240)`) with uniform right-fill and word-wrap preserving leading indentation (tabs expanded, `at_line_start` tracking, spaces at line-start bypass `flush()`); Unicode welcome box (╭─╮/│/╰─╯); dim ANSI generation stats line (`\x1b[2m`, format: `ms │ tok/s │ tokens`); `printInfo()` / `printError()` for system messages; dynamic console width via `GetConsoleScreenBufferInfo`; ANSI RGB helpers (`fg`, `bg`, `reset`) as private statics
- [x] `Chat.ixx` — `/model <alias> [quant]` command for hot model switching; `resolveAlias()` covers `gpt2`, `llama-1b`, `llama-3b`, `llama-8b`, and `-fp32` variants; `parseQuantization()` dispatches `none`/`fp8`/`fp4`; context length preserved across same-architecture switches, reset on architecture change; all responses fully buffered before display (streaming removed from hot path); `printGenerationStatistics()` delegates to `ConsoleRenderer::printStats()`; `/model` with no args prints current model, precision, quantization, and instruct flag
- [x] Prefill pipeline validated at FP8 — 2-phase dequant+cuBLASLt path produces coherent generation on Llama 3.2 3B Instruct; Chat CLI demo confirmed correct; TTFT ~2× faster than W8A16 fused path at target batch sizes
- [ ] Greedy decode validated on standard prompts — no catastrophic divergence vs BF16 baseline

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
VRAM, exceeding the RTX 4070 12 GB budget. At FP8 it fits comfortably at ~8 GB,
validating the quantization pipeline at a scale that reflects real-world deployment
constraints. The transformer architecture is identical to Llama 3.2 3B — no new
components are required, only the config preset and weight converter mapping need
verification at the 8B parameter scale.

- [x] `Llama.Presets.ixx` — `Llama3_1_8B()` preset: embedding=4096, layers=32, heads=32, kv_heads=8, hidden=14336, rope_theta=500000; `LlamaModel::fromPretrained` reads all architecture dimensions from checkpoint metadata so no preset wiring is required in the load path
- [x] `convert_llama_weights.py` — extended to support `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct`; key mapping and gate/up concatenation confirmed identical to Llama 3.2; `tie_word_embeddings=False` on 8B handled by existing lm_head fallback; `rope_scaling` (`rope_type="llama3"`) printed for reference but not written — standard RoPE with `rope_theta=500000` is accurate at context lengths ≤ 4096; output: `llama31_8b_instruct_bf16.bin`
- [x] `ChatConfig` — `ModelSize::B8` added; `Chat.ixx` `switchModel()` path generation fixed: `family_str` derives `llama31` for B8 and `llama32` for 1B/3B so the correct binary filename is constructed; `llama-8b` and `llama-8b-fp32` aliases wired in `resolveAlias()`
- [ ] Prefill pipeline validated at FP8 — logits match BF16 reference on identical prompts
- [ ] Greedy decode validated on standard prompts — no catastrophic divergence vs BF16 baseline

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
| CPU reference implementations for all Alpha.2 components | Yes |
| Debug instrumentation fully gated or removed | Yes |
| Test coverage of core components | Yes |
| CONTRIBUTING.md with coding standards | Yes |
| good-first-issue labels on GitHub | Yes |

---

## Post-Beta

Items deferred until the library has a stable contributor base.

**Training** — Full LLaMA fine-tuning pipeline. Loss function GPU migration.
Gradient checkpointing. Checkpoint save and restore.

**Architecture** — Mixture of Experts components. Speculative decoding.
Additional attention variants.

**Performance** — Flash Attention integration. Tensor parallelism.
Deterministic gradient accumulation for training reproducibility。
