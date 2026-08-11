---
title: "Running Gemma 4 12B in Docker Behind an OpenAI-Compatible API"
date: 2026-07-18
description: "One container and one command puts Gemma 4 12B behind an OpenAI/Anthropic-compatible endpoint, so Codex CLI and Claude Code can drive a local C++ inference runtime."
discussion: "https://github.com/ToddThomson/Mila/discussions/18"
---

This week Mila added a feature I think is really cool! Mila now **builds and runs the same way** on three surfaces — Windows/MSVC (where I develop), Clang on WSL/Linux, and a single Docker container — and from that container you can **drive Mila as the "brain" behind any OpenAI- or Anthropic-compatible harness**.

## One container, one command, a Gemma 4 12B model as a brain

The dev container is a completely known build environment — clone, one step, a running Mila. Beyond the interactive Chat CLI, it now stands up the **Mila Inference Server (MIS)**: a small FastAPI wire adaptor that serves the runtime over an OpenAI / Anthropic / Mila-native HTTP API.

```bash
# Build the mila binding + the server (once)
docker compose -f Docker/docker-compose.yml run --rm mila-dev mila-build-mis

# Run it — Gemma 4 12B (FP4) loads onto the GPU, served on :6452
docker compose -f Docker/docker-compose.yml run --rm --publish 6452:6452 \
    -e MILA_PORT=6452 mila-dev mila-mis
```

(There are `scripts/mis-build` and `scripts/mis-run` wrappers if you'd rather not type the compose line.)

Then point whatever agent you already use at it:

- Any **OpenAI-compatible** client → `http://localhost:6452/v1`
- **Claude Code / Anthropic**-shaped clients → the Messages path (launch MIS with `MILA_PROTOCOL=anthropic`)

Open `http://localhost:6452/docs` in a browser and you get the full interactive API. Under the hood that request is: your prompt → the `mila` binding → **Gemma 4 12B quantized to FP4** → running on a 12 GB consumer card (I develop on an RTX 4070). Tool calling is wired on both the OpenAI and Anthropic paths, so a CLI agent can actually *use tools* through it — Codex CLI and Claude Code CLI both drive it end-to-end.

The whole point: **Mila is the brain; the harness is whatever you already have.**

## Why this mattered to me

Mila has always been Windows-first — I build in Visual Studio. Making it build *and run, provably,* under Clang on Linux and inside a reproducible container is the difference between "works on my machine" and something anyone can clone and run.

C++23 named modules + CUDA + a Python binding is not a well-trodden cross-compiler path, and Clang surfaced a pile of genuine conformance issues that MSVC had been quietly papering over — position-independent code for the binding, module-import strictness, two-phase lookup for dependent templates. Fixing them made the tree more *honest*, not just more portable. I landed on a three-surface model I'm happy with:

- **CI** (GitHub Actions, Clang) is the portability tripwire — it compiles the whole tree on every push.
- **WSL / Clang** is the local fix-it loop and the correctness gate — it also runs the CUDA tests on real hardware.
- **The dev container** is end-user convenience — a known-good environment so getting Mila running isn't a toolchain fight.

## What Mila is (and isn't)

It isn't trying to be the fastest or the most complete inference server. It's a place to understand — by building it, by hand — how a modern LLM actually runs: quantization, attention, the KV cache, the compile-time dispatch, the kernels. If that's your kind of thing, the code is meant to be **read**.

Still alpha, still just me — but it's getting to be genuinely good, and this step felt worth sharing.
