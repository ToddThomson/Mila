# Mila — Roadmap

---

## Versioning

| Stage | Version |
|---|---|
| Current | 0.10.3-alpha.2 |
| Planned beta | 0.11.1-beta |

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

## Alpha.2 — In Progress

**Llama architecture, validated against HuggingFace using the same methodology.**

Success criterion: Greedy decode of LlamaModel matches HuggingFace LlamaForCausalLM
token-for-token on identical prompts using Llama 3.2 1B weights.

Alpha.2 completes directly into the beta milestone. No intermediate release is planned.

---

### Architecture Differences: Llama vs GPT-2

| Concern | GPT-2 | Llama |
|---|---|---|
| Token embedding | Learned Positional Embedding (Lpe) (fused wte + wpe) | TokenEmbedding (pure wte) |
| Positional encoding | Learned (fused into Lpe) | RoPE applied to Q, K in attention |
| Normalization | LayerNorm | RMSNorm |
| MLP activation | GELU | SiLU + gated SwiGLU |
| Attention | Multi-Head (MHA) | Grouped Query (GQA) |
| Tokenizer | BPE | SentencePiece |

---

### Tasks

#### Components

- [ ] TokenEmbedding — pure vocabulary lookup, wte only
- [ ] RoPE — rotary positional encoding applied to Q and K inside attention
- [ ] SiLU activation — forward + CUDA kernel
- [ ] SwiGLU MLP — gate_proj * silu(up_proj) then down_proj
- [ ] GroupedQueryAttention — GQA with configurable num_kv_heads and KV-cache path
- [ ] LlamaBlock — pre-RMSNorm, GQA, SwiGLU MLP, residual connections

#### Model

- [ ] LlamaTransformer — decoder-only stack
- [ ] LlamaModel — fromPretrained() + generate(), mirrors GptModel
- [ ] LlamaConfig — rope_theta, rms_norm_eps, num_kv_heads, intermediate_size

#### Weight Conversion

- [ ] convert_llama_weights.py — HuggingFace to Mila binary format
- [ ] Handle GQA weight layout and SwiGLU MLP layout
- [ ] Preserve rope_theta and rms_norm_eps in metadata

#### Tokenizer

- [ ] SentencePiece support for Llama 3.x tokenization
- [ ] Validated encode/decode round-trip against HuggingFace tokenizer

#### Validation

- [ ] Per-component numerical comparison vs HuggingFace — RMSNorm, RoPE, SwiGLU, GQA
- [ ] Full-network greedy decode comparison — LlamaModel vs LlamaForCausalLM
- [ ] Validation script mirroring hf_greedy_validation.py

#### Correctness Fixes (carried from Alpha.1)

- [ ] GELU backward — replace cosh path with 1 - tanh squared identity
- [ ] Fix loadParameter infinite recursion on unknown parameter names in Lpe and Linear

#### Cleanup (carried from Alpha.1)

- [ ] Remove debug instrumentation from GptTransformer, GptBlock, CudaMhaOp
- [ ] Remove debug logit inspection from GptModel::generate()
- [ ] Gate remaining diagnostics behind MILA_DEBUG compile flag

---

## Beta — 0.11.1

**Public release milestone.**

Beta is reached when both GPT-2 and Llama inference are validated and the library
is stable enough for external contributors to work with confidently.

| Item | Required |
|---|---|
| Llama 3.2 1B validated against HuggingFace | Yes |
| API documentation complete and published | Yes |
| CPU reference implementations for all Alpha.2 components | Yes |
| Debug instrumentation fully gated or removed | Yes |
| Test coverage of core components | Yes |
| CONTRIBUTING.md with coding standards | Yes |
| good-first-issue labels on GitHub | Yes |

---

## Post-Beta

Items deferred until the library has a stable contributor base.

**Precision** — FP16, BF16, and FP8 variants. The dispatch infrastructure
is already in place. Precision variants are expected to be largely mechanical
once the FP32 path is complete and stable.

**Training** — Full LLaMA fine-tuning pipeline. Loss function GPU migration.
Gradient checkpointing. Checkpoint save and restore.

**Architecture** — Mixture of Experts components. Speculative decoding.
Additional attention variants.

**Performance** — Flash Attention integration. Tensor parallelism.
Deterministic gradient accumulation for training reproducibility.
