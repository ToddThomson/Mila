# Mila Samples

Two kinds of thing live here, and the split is the directory layout.

## [QuickStart/](QuickStart/) — getting Mila running

One directory per path a newcomer can take to a first run: [Python](QuickStart/Python/) and
[C++](QuickStart/Cpp/). Start there. Its [README](QuickStart/README.md) says what each path
costs and what it gets you.

## Demonstrations — what Mila does

| | |
|---|---|
| [`MNIST/`](MNIST/) | Trains a 3-layer MLP to ~97.9% test accuracy. The full training loop: data loading, forward, loss, backward, AdamW step. |
| [`Bard/`](Bard/) | Trains a small GPT-2-style transformer on Tiny Shakespeare to coherent, Shakespeare-structured text — the transformer counterpart to MNIST's MLP. |

Both are C++ and both build with Mila (`MILA_ENABLE_SAMPLES`, CUDA-only today). Nothing under
`QuickStart/` builds with Mila — see [QuickStart's note in CMakeLists.txt](CMakeLists.txt).

The chat harness is not a sample: it graduated to a first-class adaptor at
`Mila/Adaptors/Chat`. See `Mila/Specifications/MilaProductFamily.md`.
