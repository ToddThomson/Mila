---
title: "Mila"
description: "A C++23 library for open LLMs at the metal - run Gemma 4 12B and Llama 3.x locally on a 12 GB consumer GPU, with FP4 and FP8 quantization and hand-written CUDA kernels."
---

**A C++23 library for open LLMs — inference and training, built from explicit neural-network
components you can read and understand.**

Mila runs open large language models locally on a single consumer GPU. Gemma 4 12B, Llama 3.1 8B,
Llama 3.2 1B/3B and GPT-2 are each validated token-for-token against HuggingFace, with FP4 and FP8
weight quantization applied at load time so a 12 GB card runs the 12B flagship at long context. Think
of it as a C++ answer to llama.cpp in spirit rather than in scope: hand-written CUDA kernels — flash
attention, flash decoding, quantized matvec — and no autograd engine or runtime dispatch layer
standing between you and the math. The [blog]({{< relref "blog" >}}) covers how it was built.
