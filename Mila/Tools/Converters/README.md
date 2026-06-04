# Mila/Tools/Converters

Converts pretrained model weights and tokenizer assets from HuggingFace to Mila binary format.

## Structure

```
Converters/
  common.py          — shared MilaWeightWriter used by all converters
  Gpt2/
    convert_weights.py
    convert_tokenizer.py
  Llama/
    convert_weights.py    — Llama 3.1 (8B) and Llama 3.2 (1B, 3B)
    convert_tokenizer.py
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

## common.py

`MilaWeightWriter` in `common.py` is shared infrastructure imported by every converter. It is not intended to be run directly.
