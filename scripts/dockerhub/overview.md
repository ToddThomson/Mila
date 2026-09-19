# Mila

Run open LLMs on your own NVIDIA GPU, from a C++23 and CUDA library built out of explicit
components you can read.

Two tags per release:

- **`<version>-runtime`** — the chat app and the model store. Pull a model, talk to it.
- **`<version>-devel`** — the same, plus the Mila source already built, and a starter CMake
  project to write your own program against it.

## What you need

An NVIDIA GPU from the RTX 40-series or newer, and a current driver.
**No CUDA Toolkit** — the image carries everything it needs.

On Windows, Docker Desktop with the WSL2 backend. Pass `--gpus all`.

## Run a model

```
docker run --rm --gpus all -v mila-store:/models toddthomson/mila-llm:0.20.0-beta.3-runtime install Llama-3.2-3B-Instruct-fp4
```

```
docker run --rm --gpus all -it -v mila-store:/models toddthomson/mila-llm:0.20.0-beta.3-runtime chat --model Llama-3.2-3B-Instruct-fp4
```

The first command downloads the model into a named volume, so later runs start with it already
there. Models are published at [huggingface.co/mila-llm](https://huggingface.co/mila-llm) —
Llama 3.2 3B, Gemma 4 12B and Qwen 3.8 27B today.

## Write against it

The `-devel` tag lands you in a shell in a built tree, with `~/myapp` configured against the
library:

```
docker run --gpus all -it -v mila-store:/models toddthomson/mila-llm:0.20.0-beta.3-devel
```

Mila is MIT licensed and its source is at
[github.com/ToddThomson/Mila](https://github.com/ToddThomson/Mila). Published models carry their
own upstream licences.
