---
title: "Sponsoring Mila"
description: "Mila accepts compute and hardware sponsorship only. An independent C++23 LLM library that can tell a hardware bug from a framework bug - evaluation loans welcome."
layout: "page"
---

Mila accepts **compute and hardware sponsorship only**.

Mila is an independent, from-scratch C++23 library for open LLMs — inference and training — built
from explicit components rather than on top of an existing framework. That independence is the offer,
and it is an exchange rather than philanthropy: Mila is neither vLLM nor llama.cpp but a separate
implementation of the same math, so in Mila's implementation a vendor can tell a *hardware* bug from
a *framework* bug. Evaluation loans are welcome — hardware does not have to be a gift.

Gemma 4 12B Instruct at FP4, measured against llama.cpp on a single 12 GB RTX 4070 — same card, same
prompt, 48K context and a 22,496-token prefill, llama.cpp under LM Studio at Q4_K_M:

| Workload | Mila | llama.cpp | llama.cpp ahead by |
|---|---|---|---|
| Prefill — 22,496 tokens | 1,817 tok/s | 2,063 tok/s | 1.14x |
| Decode — 32K context | 49.1 tok/s | 50.3–50.7 tok/s | 1.03x |

Method and derivation in [Discussion #17](https://github.com/ToddThomson/Mila/discussions/17).
That RTX 4070 is Mila's entire development and validation hardware, and therefore its ceiling. What
follows is what lies beyond it.

## What would help

**24 GB is the minimum that makes the next models runnable; 32 GB on Blackwell also reaches the
native FP4 compute path Mila cannot test today.**

The next models Mila plans to support are agentic ones — **Qwen 3.8 27B** and **Muse Glimmer 30B**,
both dense, both tuned for tool use, long tasks and failure recovery. At FP4 their weights alone are
roughly **13.5 GB and 16 GB**, before a single token of KV cache. The development rig is a 12 GB
RTX 4070. **Neither model loads on it, at any precision Mila supports** — which makes this a
precondition for the work rather than an improvement to it.

What 24 GB buys is development, benchmarking, and establishing what is practical on a single card. It
does not buy token-for-token validation against the HuggingFace reference: at BF16 those models are
~54 GB and ~63 GB and fit neither card. Mila's correctness bar is unchanged, and goes on being met on
the models that do fit.

## Bandwidth and capacity

Decode is bandwidth-bound, so the table above is a bandwidth result. The return here is direct — a
card that runs Mila well becomes a validation surface for that card, independently measurable against
llama.cpp on the same silicon.

**RTX 5090 (32 GB, sm_120)** — the headline ask, because it closes the capacity gap and the sm_120
gap together. Mila's FP4 paths are written and validated entirely on Ada (sm_89): FP4 is a storage
format there, dequantized before the GEMM, so native Blackwell FP4 compute is unreachable on the
current rig and the design is forward-compatible on paper only. The 32 GB also fits the sparse-MoE
class — Gemma 26B-A4B, Qwen3-30B-A3B, gpt-oss-20b at FP4 — and an unquantized BF16 12B quality
oracle on a single card.

**24 GB (RTX 4090, or any 24 GB card)** — the floor. It runs both named targets at FP4 with room for
context, which is what the 12 GB card cannot do at all. It leaves the sm_120 gap open.

**RTX PRO 6000 Blackwell (96 GB)** — the same GB202 die and the same 512-bit GDDR7; capacity is the
difference. Long-context validation beyond the consumer ceiling, and 70B-class weights. Kernel work
developed on a 5090 ports to it unchanged.

**Multi-GPU** — tensor-parallel paths untestable on a single card today. Consumer Blackwell has no
NVLink, so a 2x 5090 pair tests tensor parallelism over PCIe — which is the configuration a desktop
system would use.

## Platform coverage

Platforms Mila has never been compiled or run on. Not throughput — these are about whether the
abstractions hold somewhere other than where they were written.

**DGX Spark (GB10, 128 GB coherent, sm_121)** — Mila has never been built or run on aarch64; today it
is x86-64 Windows and x86-64 Linux only. DGX OS is also the natural reference for the container and
published-image path, and sm_121 would be a second compute capability alongside the rig's sm_89. The
deepest test is the memory model: Mila's memory resources assume discrete device VRAM with explicit
host-to-device staging, and GB10 is one coherent pool with nothing to copy into. Explicitly **not** a
performance machine — at ~273 GB/s it sits below the development rig's RTX 4070 (~504 GB/s), so
bandwidth-bound decode is slower there.

**AMD (ROCm)** and **Apple silicon (Metal)** — a second and third compute backend. Mila's device type
is a compile-time template parameter and `Rocm` and `Metal` are reserved alongside `Cpu` and `Cuda`;
neither backend is implemented. Backend work is gated on the hardware: a port cannot begin, and can
never be validated, on a machine that does not have the silicon in it.

For a backend Mila does not have yet, the exchange is more concrete: per-operation dispatch is
resolved at compile time from an explicit traits table, so a port is a readable, self-contained
account of what your hardware needs to run a modern LLM — not a diff buried in a framework.

## What Mila does not accept

Money, in any form — no GitHub Sponsors, no cash tiers, no donations. This is a craft project; the
only useful currency here is hardware and compute time.

## Getting in touch

For compute or hardware collaboration, contact **todd.thomson@me.com**.

Sponsors are named in the README and in the release notes for any release their hardware made
possible. Anonymous is fine too — say which you prefer.
