# Mila/Tools/Converters

Converts pretrained model weights and tokenizer assets from HuggingFace to Mila binary format.

## Structure

```
Converters/
  common.py          — shared writers and the HF -> Mila tensor name maps
  Gpt2/
    convert_weights.py
    convert_tokenizer.py
  Llama/
    convert_weights.py    — Llama 3.1 (8B) and Llama 3.2 (1B, 3B)
    convert_tokenizer.py
  Gemma/
    convert_weights.py    — Gemma 4 12B (dense text chassis)
    convert_tokenizer.py
  Qwen/
    convert_weights.py    — Qwen 3.8 27B (hybrid DeltaNet / full-attention stack)
```

## Setup

Requires Python 3.10 or newer (validated on 3.14.5). PyTorch and Transformers publish
wheels per Python minor version and can lag brand-new releases — if `pip install` cannot
find a torch wheel for your interpreter, create the venv with a slightly older minor
(e.g. 3.12).

Run once from the `Converters/` root directory:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU-accelerated PyTorch is not required — conversion runs on CPU. If you prefer a CUDA wheel, substitute the appropriate index URL in `requirements.txt` for your platform and CUDA version.

All scripts are run from the `Converters/` root with the virtual environment activated. Running from the root ensures `common.py` is on the Python path without any additional configuration.

---

## GPT-2

```powershell
python Gpt2/convert_weights.py --model gpt2 --output <weights-dir>/gpt2/gpt2_small_fp32.bin
python Gpt2/convert_weights.py --model gpt2-medium --output <weights-dir>/gpt2/gpt2_medium_fp32.bin
```

| Option | Values | Default |
|---|---|---|
| `--model` | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` | required |
| `--output` | path to write `.bin` file | required |
| `--dtype` | `float32`, `float16`, `bfloat16` | `float32` |

---

## Llama

Llama models are gated on HuggingFace. Accept Meta's license agreement and authenticate before running:

```powershell
hf auth login
```

### Supported models

| Model | Parameters |
|---|---|
| `meta-llama/Llama-3.2-1B` | 1B base |
| `meta-llama/Llama-3.2-3B` | 3B base |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B instruct |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B instruct |
| `meta-llama/Llama-3.1-8B` | 8B base |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B instruct |

Run the tokenizer conversion first, then the weights. The tokenizer binary is shared across all Llama 3.x variants — it only needs to be converted once.

```powershell
# Tokenizer (shared across all Llama 3.x variants)
python Llama/convert_tokenizer.py --model meta-llama/Llama-3.2-3B-Instruct --output <weights-dir>/llama/llama_tokenizer.bin

# Llama 3.2
python Llama/convert_weights.py --model meta-llama/Llama-3.2-1B-Instruct --output <weights-dir>/llama/llama32_1b_instruct_bf16.bin
python Llama/convert_weights.py --model meta-llama/Llama-3.2-3B-Instruct --output <weights-dir>/llama/llama32_3b_instruct_bf16.bin

# Llama 3.1 8B — load in bf16 to stay within host RAM; ~16 GB required
python Llama/convert_weights.py --model meta-llama/Llama-3.1-8B-Instruct --output <weights-dir>/llama/llama31_8b_instruct_bf16.bin
```

| Option | Values | Default |
|---|---|---|
| `--model` | any supported model name above | required |
| `--output` | path to write `.bin` file | required |
| `--dtype` | `float32`, `bfloat16` | `bfloat16` |

> **Llama 3.1 8B note:** `tie_word_embeddings=False` — `lm_head.weight` is a separate tensor and is written directly. Llama 3.2 1B/3B tie embeddings; the converter handles both cases automatically.

---

## Gemma

Gemma models are gated on HuggingFace. Accept Google's license agreement and authenticate before running:

```powershell
hf auth login
```

> **Requires a recent transformers.** Gemma 4 is the `gemma4_unified` architecture, which older
> `transformers` releases do not recognize. Validated on **5.12.1**; upgrade if a load fails with an
> unknown/`gemma4` model type: `pip install -U "transformers>=5.12.1"`.

Target is the **Gemma 4 12B Unified** dense text chassis (the 5:1 sliding/global layer interleave,
decoupled `head_dim`, K=V global layers, GeGLU FFN, sandwich norm, and QK-norm are all read from the
model config, so the converter adapts to the geometry rather than hardcoding it).

### Supported models

| Model | Parameters |
|---|---|
| `google/gemma-4-12b` | 12B base |
| `google/gemma-4-12b-it` | 12B instruct |

```powershell
# Tokenizer (shared across Gemma 4 variants)
python Gemma/convert_tokenizer.py --model google/gemma-4-12b-it --output <weights-dir>/gemma/gemma_tokenizer.bin

# Gemma 4 12B — load in bf16 to stay within host RAM; ~24 GB required
python Gemma/convert_weights.py --model google/gemma-4-12b-it --output <weights-dir>/gemma/gemma4_12b_it_bf16.bin
```

| Option | Values | Default |
|---|---|---|
| `--model` | any supported model name above | required |
| `--output` | path to write `.bin` file | required |
| `--dtype` | `float32`, `bfloat16` | `bfloat16` |

**Three Gemma-specific transforms are folded in at convert time** (so the Mila inference path stays
identical to Llama and needs no Gemma-only kernels):

1. **Embedding scale** — HF multiplies embedded hidden states by `sqrt(hidden_size)` at runtime, and so
   does Mila (`TokenEmbeddingConfig::embedding_scale`), so the table is written **raw**. With
   `tie_word_embeddings` (the Gemma 4 default) the `lm_head` blob is omitted entirely and
   `GemmaTransformer` aliases the shared table at load time.
2. **`(1 + weight)` RMSNorm** — Gemma's RMSNorm computes `x_norm * (1 + weight)`. The `+1` is applied at
   the kernel (`RmsNormConfig::withUnitOffset`), **not** folded in here, so every norm weight (all
   sandwich norms, both QK-norms, and the final norm) is written **raw** — byte-identical to the HF
   checkpoint and directly comparable against it.
3. **K=V global layers** — the 1-in-N global (full-attention) layers share K=V and have no `v_proj`, so
   their fused QKV blob is `[Q | K]` only; the sliding layers are the usual `[Q | K | V]`.

> **Gemma 4 verification:** the exact HuggingFace config attribute names and `state_dict` keys for
> Gemma 4 are read defensively with the `Gemma.md` design defaults as fallbacks. The script prints every
> resolved value on first run — verify them against the installed `transformers` Gemma 4 implementation.

> **Tokenizer:** `convert_tokenizer.py` writes the Gemma vocabulary in the shared Mila tokenizer binary
> format (same layout as the Llama tokenizer). Gemma uses a SentencePiece tokenizer (262K vocab) with
> `▁` (U+2581) spaces and byte fallback, so the Mila *runtime* loader (`BpeVocabulary::loadGemma`, with a
> SentencePiece pre-tokenization) is the piece that makes it usable end-to-end — that loader is the
> follow-up to this converter and is validated by an encode round-trip against the HF tokens the script
> prints.

---

## Qwen

Qwen 3.8 is Apache 2.0 and ungated — no authentication step.

> **Requires a recent transformers** for the reference implementation the conversion is checked
> against (`models/qwen3_5/`). Validated on **5.12.1**.

Target is **Qwen 3.8 27B**, text only: the 64-layer stack interleaving three Gated DeltaNet layers per
full-attention layer (`full_attention_interval: 4`). The vision tower and the MTP draft head are out of
scope for this chassis and are skipped at the checkpoint index.

### Supported models

| Model | Parameters |
|---|---|
| `Qwen/Qwen3.8-27B` | 26.9B text (51.8 GiB checkpoint) |

```powershell
# Qwen 3.8 27B — streams shard by shard; host RAM never holds more than one tensor
python Qwen/convert_weights.py --model Qwen/Qwen3.8-27B --output <weights-dir>/qwen/qwen38_27b_bf16.bin

# Structural smoke test — the first 4 layers (3 DeltaNet + 1 full attention)
python Qwen/convert_weights.py --model Qwen/Qwen3.8-27B --max-layers 4 --output <weights-dir>/qwen/qwen38_27b_l4_bf16.bin
```

| Option | Values | Default |
|---|---|---|
| `--model` | `Qwen/Qwen3.8-27B`, or a local checkpoint directory | required |
| `--output` | path to write `.bin` file | required |
| `--dtype` | `float32`, `bfloat16` | `bfloat16` |
| `--max-layers` | convert only the first N layers — a structural smoke test, **not a model** | all |

> **This converter streams, and it has to.** The checkpoint is 51.8 GiB against 31.8 GB of host RAM, so
> `from_pretrained` — which the other converters here use — cannot run at all. Shards are read through
> safetensors mmap one tensor at a time, and the output goes through `MilaStreamingWeightWriter`, whose
> index is declared from the checkpoint's own shard headers before any tensor data moves. Output is
> ~54 GB at `bfloat16`.

**Four Qwen-specific transforms**, each a contract stated in the consuming block's header:

1. **De-interleave `q_proj`.** `attn_output_gate` makes the query projection double width, and the
   checkpoint stores it interleaved per head, `[q_h0 | gate_h0 | q_h1 | gate_h1 | ...]`.
   `QwenAttentionBlock` takes `[query | gate | key | value]` with query and gate as **contiguous
   halves**, so the query half is de-interleaved before the three sources concatenate.
2. **Split `in_proj_qkv`** `[10240, 5120]` into `[q|k]` at 4096 and `[v]` at 6144. One tensor cannot
   carry two storage policies, and the precision plan puts DeltaNet q/k a half step above v.
3. **Split `conv1d.weight`** `[10240, 1, 4]` on the same boundary. Exact rather than approximate: the
   convolution is depthwise, so two convolutions over a partition of the channels compute what one over
   all of them computes.
4. **Norm weights are written RAW.** Qwen's stream norms scale by `(1 + weight)` with weights stored
   zero-centered, and Mila applies the `+1` at the kernel — so stored weights stay identical to the
   checkpoint. The DeltaNet mixer's gated norm (`linear_attn.norm`) is genuinely raw on both sides.
   **Both conventions live in one model**, and getting them the wrong way round is silent.

> **Nothing is dropped in silence.** Every checkpoint tensor is accounted for as consumed or explicitly
> skipped; an unrecognized tensor family fails the run rather than being ignored. Declared shapes are
> checked against the geometry the config implies before any data is written.

---

## common.py

`common.py` is shared infrastructure imported by every converter, not intended to be run directly. It
holds `MilaWeightWriter` (buffered) and `MilaStreamingWeightWriter` (declare-then-stream, for models
larger than host RAM), plus the HuggingFace -> Mila tensor name maps. The maps live here rather than in
each converter because the Qwen3.8 codebook packer names its quantized tensors from the same source —
the packer and the BF16 converter cannot be allowed to disagree about what fuses with what.
