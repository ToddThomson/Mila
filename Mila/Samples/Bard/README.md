# Bard - Shakespeare Transformer Language Model

A transformer-based language model implementation built on Mila.

## Status

Revived to the current Mila API as part of v0.20 **Training Revival** — Bard builds green and trains
end-to-end (FP32 on CUDA). See Current Status below for what runs today. The MNIST sample is the
simpler MLP reference if you want to start there.

## Overview

Bard implements a GPT-style decoder-only transformer for character-level next-token prediction:

```
Token Embedding -> Positional Encoding -> N x Transformer Blocks -> LayerNorm -> LM Head
```

## Features

- **Device-agnostic**: Supports CPU and CUDA execution
- **Transformer architecture**: Multi-head self-attention with causal masking
- **Character-level**: Learns directly from raw text without tokenization
- **Preprocessing pipeline**: Efficient vocabulary building and data caching
- **AdamW optimizer**: Modern optimization with weight decay

## Prerequisites

### System Requirements
- **CMake**: >= 3.14
- **C++ Compiler**: C++23 support required
- **CUDA** (optional): For GPU acceleration
- **Mila Framework**: Built and installed

### Dataset

Provide any plain text file for training. Example using Tiny Shakespeare:

```
Data/
   DataSets/
      TinyShakespeare/
         input.txt
```

Download from: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

## Building

```bash
# From repository root
mkdir build && cd build
cmake ..
cmake --build . --target Bard
cmake --build . --target PreprocessText  # Optional preprocessing tool
```

## Usage

### Preprocessing (Required First Run)

```bash
# Preprocess text data (creates .vocab and .tokens files)
./Tools/PreprocessText data/input.txt
```

This creates:
- `data/input.txt.vocab` - Character vocabulary
- `data/input.txt.tokens` - Tokenized text data

### Training

```bash
# Train with default settings
./Bard --data-file data/input.txt

# Custom hyperparameters
./Bard --data-file data/input.txt \
         --batch-size 32 \
         --seq-length 128 \
         --epochs 10 \
         --learning-rate 3e-4
```

### Command-Line Options

```
Options:
  --data-file <path>      Path to text file
  --batch-size <int>      Batch size (default: 32)
  --seq-length <int>      Sequence length (default: 128)
  --epochs <int>          Number of epochs (default: 10)
  --learning-rate <float> Learning rate (default: 3e-4)
  --embedding-dim <int>   Embedding dimension (default: 256)
  --num-heads <int>       Number of attention heads (default: 4)
  --num-layers <int>      Number of transformer layers (default: 4)
  --device <string>       cpu or cuda (default: cuda)
  --help                  Show help message
```

## Architecture

### CharTransformer
- **Vocabulary**: Built from training text (typically ASCII extended)
- **Embedding Dimension**: 256 (configurable)
- **Attention Heads**: 4 (configurable)
- **Transformer Layers**: 4 (configurable)
- **MLP Hidden Dimension**: 1024 (4x embedding_dim)

### Data Pipeline
- **CharPreprocessor**: Builds vocabulary and tokenizes text (one-time)
- **CharVocabulary**: Character -> index mappings
- **CharDataLoader**: Efficient sliding-window sequence loading

## Expected Results

With Tiny Shakespeare (~1MB, 40K lines):
- **Vocabulary**: ~65 unique characters
- **Sequences**: ~8K sequences (seq_length=128, stride=64)
- **Training Time**: ~2-5 minutes per epoch (CUDA)
- **Perplexity**: Decreases over epochs as model learns

## Current Status

Revived and validated (VS2026 / CUDA): Bard builds green, trains FP32 GPT-2 on TinyShakespeare to
coherent Shakespeare-structured text (perplexity < 3 by ~epoch 17), and generates from the trained
model. The full training spine — `GptTransformer` (attention, MLP, positional encoding), the AdamW
optimizer, gradient flow, and train-from-scratch initialization — runs end-to-end; reviving the
sample surfaced and fixed three latent CUDA training-backward bugs.

Remaining: the primitive / component test coverage that proves the training path at the component
level (v0.20 Training Revival — the sample is the bug-discovery tool, not the test target). Checkpoint
save/restore is a Future item.

## Troubleshooting

### Preprocessed Files Missing
```
Error: Vocabulary file not found
```
**Solution**: Run `PreprocessText` on your data file first.

### CUDA Out of Memory
**Solution**: Reduce `--batch-size` or `--seq-length`.

### Poor Training Results
**Solution**:
- Increase `--num-layers` and `--embedding-dim`
- Train for more `--epochs`
- Use larger dataset

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Mila Framework Documentation](../../docs/README.md)

## License

This sample is part of the Mila framework and follows the project license.
