# Mila Deep Learning Framework — Roadmap

---

## Versioning

| Stage | Version |
|---|---|
| Current | `0.10.1-alpha.2` |
| Next milestone | `0.11.1-beta` (post-Llama validation) |

---

## Alpha.1 — Completed ✅

**Focus:** GPT-2 inference validated against HuggingFace.

| Item | Status |
|---|---|
| Core NN components (Linear, LayerNorm, MHA, MLP, Residual, Gelu) | ✅ |
| CUDA + CPU kernels for all components | ✅ |
| GptTransformer (decoder-only, pre-LN) | ✅ |
| GptModel — inference-only wrapper with `fromPretrained()` + `generate()` | ✅ |
| Two-phase KV cache prefill + decode pipeline | ✅ |
| HuggingFace GPT-2 weight converter (`convert_gpt2_weights.py`) | ✅ |
| BPE tokenizer (`BpeTokenizer::loadGpt2`) | ✅ |
| Greedy decode validated token-for-token against HuggingFace GPT-2 | ✅ |
| Chat CLI sample (`Mila.Chat`) | ✅ |
| AdamW optimizer + MNIST training loop | ✅ |
| ~60% test coverage | ✅ |

---

## Alpha.2 — Current 🚧

**Focus:** Llama compatibility, validated against HuggingFace using the same
methodology applied to GPT-2.

**Success criterion:** Greedy decode of `LlamaModel` matches HuggingFace
`LlamaForCausalLM` token-for-token on identical prompts.

---

### Architecture Delta: Llama vs GPT-2

| Component | GPT-2 (alpha.1) | Llama (alpha.2) |
|---|---|---|
| Positional encoding | Learned (`Lpe`) | Rotary (`RoPE`) |
| Normalization | `LayerNorm` | `RMSNorm` |
| MLP activation | `GELU` (tanh-approx) | `SiLU` + gated (`SwiGLU`) |
| Attention | MHA (full) | GQA (Llama 2+) |
| Tokenizer | BPE (GPT-2) | SentencePiece / tiktoken |
| Weight tying | `lm_head` ↔ `wte` | None (independent `lm_head`) |

---

### Tasks

#### 1. New Components

- [ ] `RMSNorm` — forward + CUDA kernel, validated against HF `LlamaRMSNorm`
- [ ] `RotaryEmbedding` (`RoPE`) — forward, frequency computation, validated against HF
- [ ] `SiLU` activation — forward + CUDA kernel
- [ ] `SwiGLU` MLP block — `gate_proj × silu(up_proj)` then `down_proj`
- [ ] `GroupedQueryAttention` — GQA with configurable `num_kv_heads`; KV cache path required
- [ ] `LlamaBlock` — pre-RMSNorm, GQA, SwiGLU MLP, residual connections

#### 2. LlamaTransformer + LlamaModel

- [ ] `LlamaTransformer` — decoder-only network (mirrors `GptTransformer`)
- [ ] `LlamaModel` — inference wrapper with `fromPretrained()` + `generate()` (mirrors `GptModel`)
- [ ] `LlamaConfig` — `num_kv_heads`, `rope_theta`, `rms_norm_eps`, `intermediate_size`

#### 3. Weight Conversion

- [ ] `convert_llama_weights.py` — HuggingFace → Mila binary format
  - Handle GQA weight layout (`q_proj`, `k_proj`, `v_proj` split)
  - Handle `gate_proj` / `up_proj` / `down_proj` MLP layout
  - No Conv1D transposition (Llama uses standard `nn.Linear`)
  - Preserve `rope_theta` and `rms_norm_eps` in metadata

#### 4. Tokenizer

- [ ] SentencePiece tokenizer support (`LlamaTokenizer`) or tiktoken (Llama 3)
- [ ] Validate encode/decode round-trip against HuggingFace tokenizer

#### 5. Validation

- [ ] Per-component forward pass numerical comparison vs HF (RMSNorm, RoPE, SwiGLU, GQA)
- [ ] Full-network greedy decode comparison: `LlamaModel` vs `LlamaForCausalLM`
- [ ] Target model: **Llama 3.2 1B** (smallest publicly accessible variant)
- [ ] Validation script: `hf_llama_greedy_validation.py` (mirrors `hf_greedy_validation.py`)

#### 6. Cleanup (carried from alpha.1)

- [ ] Remove debug instrumentation from `GptTransformer::forward()`, `GptBlock::forward()`, `CudaMhaOp::forwardPrefill()`
- [ ] Remove debug logit inspection from `GptModel::generate()`
- [ ] Fix `Gelu` backward: replace `cosh` path with `1 - tanh²(z)` identity
- [ ] Fix `Lpe::loadParameter` / `Linear::loadParameter` infinite recursion in unknown-name fallback
- [ ] Remove `generate_no_decode()` (superseded by validated `generate()`)

---

### Out of Scope for Alpha.2

- Training pipeline changes
- Loss function GPU migration
- Checkpoint system
- Test coverage targets
- Documentation sprint
- Beta release preparation

These remain valid goals but are deferred until both GPT-2 and Llama inference
are stable and validated. The beta milestone will be re-scoped after alpha.2
completes.

---
