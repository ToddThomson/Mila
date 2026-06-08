# Models Directory

This directory contains pretrained open-source models converted to Mila binary format for use in C++ inference.

## Overview

All models in this directory are:
- **Pretrained checkpoints** from HuggingFace
- **Converted via** the tools in `Mila/Tools/Converters/`
- **Ready for C++ consumption** in Mila's native binary format
- **Not tracked in git** — generate locally by running the appropriate converter

## Current Models

### GPT-2

```
Gpt2/
  gpt2_small_fp32.bin
```

Converter: `Mila/Tools/Converters/Gpt2/convert_weights.py`

### Llama

```
Llama/
  llama_tokenizer.bin               — shared across all Llama 3.x variants
  llama32_1b_instruct_bf16.bin
  llama32_3b_instruct_bf16.bin
  llama31_8b_instruct_bf16.bin
```

Converter: `Mila/Tools/Converters/Llama/convert_weights.py` and `convert_tokenizer.py`

See `Mila/Tools/Converters/README.md` for full setup and usage instructions.

## Planned Models

- `Qwen3/` — Qwen 3 variants (Beta.1)
- `Mistral/` — Ministral 3B and 8B variants (Beta.2)

## Model Organization

Models are organized by family. As the collection grows, a versioning structure may be added to handle fine-tuned variants and quantized checkpoints.

## Notes

- Model files are large — ensure adequate storage before converting
- Original model licenses apply to converted formats; verify before any production use
- Quantized variants (FP8, FP4) are produced at load time by Mila — only BF16 source files are stored here
