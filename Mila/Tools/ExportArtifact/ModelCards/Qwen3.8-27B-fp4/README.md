---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - mila
  - qwen
  - fp4
  - quantized
library_name: mila
---

# Qwen3.8 27B Instruct — FP4 for Mila

`Qwen/Qwen3.8-27B` quantized to FP4 for [Mila](https://github.com/ToddThomson/Mila), so 27B runs
on a single 16 GB card. Nothing was fine-tuned or distilled.

## Requirements

A 16 GB NVIDIA GPU, and Mila. The download is 15.1 GiB.

This is Mila's own FP4 format, so `transformers` and vLLM cannot load it.

## Use

Install it with the Mila command line, then run it in the Mila chat app:

```
mila install Qwen3.8-27B-fp4
mila-chat --model Qwen3.8-27B-fp4
```

## License

Apache 2.0, inherited from `Qwen/Qwen3.8-27B`. The full text is in `LICENSE`.
