# Mila — Roadmap

---

## Versioning

| Stage | Version | Title |
|---|---|---|
| In Progress | 0.12.0-alpha.4 | Instruction following and tool calling on Llama 3.2 3B Instruct |
| Planned | 0.13.0-alpha.5 | FP8 quantization pipeline — Llama 3.1 8B Instruct |
| Planned | 0.2.1-beta | Public release |

---

## Alpha.4 — In Progress

**Instruction following and tool calling, validated on Llama 3.2 3B Instruct at BF16.**

Alpha.4 delivers the structured message and tool calling infrastructure in the Chat
application layer. No model architecture changes are required — Llama 3.2 3B Instruct
shares the same weight layout as the base model validated in Alpha.3, and the converter
already supports the instruct variant. The work is entirely in the Chat layer above the
model.

Success criterion: Llama 3.2 3B Instruct produces correct tool call responses
end-to-end through the structured message pipeline in the Chat application.

### Phase 1 — Structured Message Infrastructure

- [ ] Verify `BpeTokenizer::loadLlama32` encodes Llama 3.2 special tokens as single atomic token IDs (`<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`, `<|eom_id|>`)
- [ ] `ChatMessage` — role (system / user / assistant / tool), content, optional tool calls
- [ ] `MessageFormatter` — applies Llama 3.2 instruct chat template to a message sequence
- [ ] Stop token handling — generation halts on `<|eot_id|>` and `<|eom_id|>`
- [ ] `Chat::run()` — replace raw string history with structured `ChatMessage` history

### Phase 2 — Tool Calling Framework

- [ ] `ToolDefinition` — name, description, JSON schema parameters
- [ ] `ToolCall` — parsed tool name and arguments extracted from model output
- [ ] `ToolCallParser` — detects `<|python_tag|>` boundary, extracts and validates JSON
- [ ] System prompt builder — injects active `ToolDefinition` list into the system message
- [ ] `ChatConfig` — add optional `system_prompt` and `tools` list

### Phase 3 — Llama 3.2 3B Instruct Validation

- [ ] Convert Llama 3.2 3B Instruct weights — `convert_llama_weights.py` already supports the instruct variant, confirm output
- [ ] Instruct format validated — greedy decode with structured prompt matches expected assistant response format
- [ ] Tool call round-trip validated end-to-end — model issues a tool call, result is fed back, final response is correct

---

## Alpha.5 — Planned

**FP8 quantization pipeline, validated end-to-end on Llama 3.1 8B Instruct.**

FP8 is Mila's quantization target for weight compression at inference time. Alpha.5
delivers the quantization infrastructure required to load and run 8B-class models
within a 12 GB VRAM budget, using Meta's W8A8 static quantization format with
per-channel weight scales and per-layer activation scales.

Success criterion: Greedy decode of Llama 3.1 8B Instruct at FP8 matches
HuggingFace token-for-token on identical prompts, tool calling validated end-to-end.

### Phase 1 — FP8 Quantization Infrastructure

- [ ] `QuantizedWeight` — FP8 weight buffer + FP32 per-channel scales, contiguous allocation via `Tensor` views
- [ ] `ScaleGranularity` — `PerChannel` descriptor; `PerBlock` stub for future FP4
- [ ] `QuantizationConfig` — model-level policy (`none()` / `fp8()` factory methods), wired into `fromPretrained()` API
- [ ] `LinearConfig::withWeightPrecision()` — fluent setter routing to the correct op registration
- [ ] `Linear` — optional `QuantizedWeight` member + `input_scale` scalar; `initializeParameters()` branches on config
- [ ] `CudaLinearOp<TPrecision, TWeightPrecision>` — second template parameter; `NativeWeightType` alias; mixed-precision cuBLASLt plan path
- [ ] `build_strided_plan` — separate `data_type_A` / `data_type_B` parameters for mixed-precision layouts
- [ ] FP8 decode matvec kernel — `cuda_matvec_impl` variant: BF16 activation × FP8 weight + FP32 scale → BF16 output
- [ ] `CudaLinearOp` registrar — register `<BF16, FP8_E4M3>` instantiation

### Phase 2 — Llama 3.1 8B Instruct @ FP8

- [ ] `convert_llama_weights.py` — extend for Llama 3.1 8B weight layout and FP8 checkpoint format (weight + activation scales)
- [ ] `ChatConfig` — add `ModelSize::B8`, `ModelPrecision::FP8`; default `context_length` for FP8 set to 2048
- [ ] Prefill pipeline validated at FP8 — logits match HuggingFace on identical prompts
- [ ] Full-network greedy decode validated token-for-token against HuggingFace
- [ ] Tool calling validated end-to-end using structured message pipeline from Alpha.4

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

Beta is reached when both GPT-2 and Llama inference are validated across FP32 and BF16,
and the library is stable enough for external contributors to work with confidently.

| Item | Required |
|---|---|
| Llama 3.2 1B FP32 validated against HuggingFace | Yes |
| Llama 3.2 3B BF16 validated against HuggingFace | Yes |
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
