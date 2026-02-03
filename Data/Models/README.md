# Models Directory

This directory contains pretrained open-source models that have been converted to Mila-compatible formats for use in C++ inference and fine-tuning.

## Overview

All models in this directory are:
- **Pretrained checkpoints** from open-source projects
- **Converted via Python scripts** located in `/Data/Scripts`
- **Ready for C++ consumption** in Mila's native format
- **Snapshot versions** representing specific model states

## Current Models

### GPT-2
Generative pretrained transformer model from OpenAI.
- Architecture: Decoder-only transformer
- Use cases: Text generation, language modeling

## Planned Models

### LLaMA
Meta's Large Language Model series.

### Mistral
Mistral AI's efficient language models.

## Model Organization

Currently, models are organized by family name at the top level. As the collection grows, a versioning structure will be implemented to handle:
- Multiple checkpoint versions
- Different model sizes (base, large, etc.)
- Fine-tuned variants
- Quantized versions

Expected future structure:
```
Models/
├── Gpt2/
│   ├── base/
│   ├── large/
│   └── checkpoints/
├── Llama/
│   └── ...
└── Mistral/
    └── ...
```

## Conversion Process

Models are converted from their original Python/PyTorch formats using scripts in `/Data/Scripts`. The conversion process typically includes:
1. Loading pretrained weights from HuggingFace or original sources
2. Converting weight formats and tensor layouts for C++ compatibility
3. Extracting model configuration and hyperparameters
4. Saving in Mila's binary format

## Usage

Model files in this directory are referenced by Mila's training and inference tools. Ensure the model format matches the version expected by your C++ codebase.

## Notes

- Model files can be large; ensure adequate storage
- Original model licenses apply to converted formats
- Verify model provenance and licensing before use in production