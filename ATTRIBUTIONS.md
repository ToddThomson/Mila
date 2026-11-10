# Attributions

Mila was inspired by the practical, educational work of Andrej Karpathy and his `llm.c` project. I have attempted to follow Andrej's philosophy of "understand it by building it from scratch" and have tried to take it in a meaningful direction.

- Inspiration: Andrej Karpathy — https://github.com/karpathy/llm.c

## Research Acknowledgements

Mila is built on the published research and open-source contributions of the machine learning systems community. This document acknowledges the ideas, algorithms, and software projects that have influenced Mila's implementation.

### FlashAttention

Mila's attention kernels are informed by the work of Tri Dao and collaborators on IO-aware exact attention. The impact on Mila is hard to overstate — it shapes both phases of inference:

- **Prefill:** the tensor-core flash kernels on Gemma's global and local sliding layers (`Gqa.Flash.Wmma.cu`, `Gqa.Flash.Fa2.cu`) apply FlashAttention's tiled streaming with online softmax, and FlashAttention-2's warp partitioning and work-partitioning ideas. Beyond speed, eliminating the O(sequence × context) score materialization is what makes long context fit on consumer VRAM at all.
- **Decode:** the fused single-token attention kernel (`Gqa.Decode.Bf16.cu`) applies the Flash-Decoding formulation — split-K parallelization of the key/value sequence across blocks with an online-softmax merge — which is what keeps every SM busy when a layer has few (or one) KV heads.

Both rest on the online normalizer calculation for softmax introduced by Milakov and Gimelshein.

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré.
  *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
  [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.*
  ICLR 2024. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Tri Dao, Daniel Haziza, Francisco Massa, and Grigory Sizov.
  *Flash-Decoding for long-context inference.* PyTorch blog, 2023.
  [pytorch.org/blog/flash-decoding](https://pytorch.org/blog/flash-decoding/)
- Maxim Milakov and Natalia Gimelshein. *Online normalizer calculation for softmax.* 2018.
  [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{dao2022flashattention,
  title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}

@inproceedings{dao2023flashattention2,
  title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}

@misc{dao2023flashdecoding,
  title={Flash-Decoding for long-context inference},
  author={Dao, Tri and Haziza, Daniel and Massa, Francisco and Sizov, Grigory},
  year={2023},
  howpublished={PyTorch blog},
  url={https://pytorch.org/blog/flash-decoding/}
}

@misc{milakov2018online,
  title={Online normalizer calculation for softmax},
  author={Milakov, Maxim and Gimelshein, Natalia},
  year={2018},
  eprint={1805.02867},
  archivePrefix={arXiv}
}
```

</details>

### MLX

Mila's Metal backend for Apple silicon takes several kernel designs from Apple's MLX, an MIT-licensed array framework
for Apple silicon. Mila writes its own kernels for its own weight layouts; the shapes below are MLX's, and each source
file names the MLX file it follows (ml-explore/mlx v0.32.4).

- **Decode attention** (`GroupedQueryAttention.metal`): the two-pass decode of `sdpa_vector.h`, the keys split across
  blocks and merged, as Flash-Decoding does above.
- **Prefill attention** (`GroupedQueryAttention.Flash.metal`): the online softmax in base 2 with the scale carrying
  log2(e), and the starting tile sizes, from `steel_attention.h`.
- **One-row Q4_0 product** (`LinearQuantized.metal`, `linear_quantized_row`): the shape of `qmv_fast_impl` in
  `quantized.h`, with the codes masked in place against prescaled activations.

- Awni Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. *MLX: efficient and flexible machine learning
  on Apple silicon.* Apple, 2023. [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)

### MARLIN

On CUDA, decoding several rows at once (`CudaDecodeRows.cu`) keeps a register accumulator per row, so the work per
weight is the same at every row count up to 8, as MARLIN does for small batches.

- Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, and Dan Alistarh. *MARLIN: Mixed-Precision
  Auto-Regressive Parallel Inference on Large Language Models.* 2024.
  [arXiv:2408.11743](https://arxiv.org/abs/2408.11743)

### Speculative decoding

Gemma 4 decodes with Google's drafter on CUDA and on a Mac; a drafted token is kept by the rule of Leviathan et al.,
which leaves the model's output distribution unchanged (`SpeculativeSampler.ixx`, `GemmaModel.ixx`).

- Yaniv Leviathan, Matan Kalman, and Yossi Matias. *Fast Inference from Transformers via Speculative Decoding.*
  ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

<details>
<summary>BibTeX</summary>

```bibtex
@software{mlx2023,
  title={{MLX}: efficient and flexible machine learning on Apple silicon},
  author={Hannun, Awni and Digani, Jagrit and Katharopoulos, Angelos and Collobert, Ronan},
  year={2023},
  url={https://github.com/ml-explore}
}

@misc{frantar2024marlin,
  title={{MARLIN}: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models},
  author={Frantar, Elias and Castro, Roberto L. and Chen, Jiale and Hoefler, Torsten and Alistarh, Dan},
  year={2024},
  eprint={2408.11743},
  archivePrefix={arXiv}
}

@inproceedings{leviathan2023fast,
  title={Fast Inference from Transformers via Speculative Decoding},
  author={Leviathan, Yaniv and Kalman, Matan and Matias, Yossi},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023}
}
```

</details>

## A Steam Shovel Named Mary Anne

Some of Mila's instincts predate any paper. Virginia Lee Burton's *Mike Mulligan and His Steam Shovel* (1939) — a favorite of mine as a boy — quietly shaped how I frame a solution.  Mike's loyalty to a machine he understands completely, in an age of flashier ones, is the mastery-over-novelty stance of a hand-built C++/CUDA runtime. And Burton's honest cutaway drawings — which let a child see exactly how the machine works — are the same promise as Mila's readable path from prompt to kernel, with no hidden engine. With gratitude to a steam shovel that dug this particular corner long before I did.
