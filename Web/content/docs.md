---
title: "How Mila works"
description: "What Mila is and how it is built - a reference implementation of LLM inference with no execution engine between your program and the arithmetic."
---

Mila is a **reference implementation**: an LLM inference stack written to be read.

Most inference engines are built to be *used*. You hand a runtime a model and a configuration, and
it owns execution from there — a graph, a scheduler, a dispatch layer standing between your program
and the arithmetic. Mila has no execution engine. A model is ordinary C++ objects composed in a
translation unit, so the path from prompt to kernel is one you can follow by reading it. That is
what *at the metal* means here: not merely close to the hardware, but nothing hidden between you
and it.

## The type is the configuration

Device, precision, and weight quantization are template parameters, resolved at compile time:

```
using QuantizedProjection =
    Linear<DeviceType::Cuda, TensorDataType::BF16, PerGroupFp4<128>>;
```

There is no configuration object and no runtime dispatch table. A CPU tensor and a CUDA tensor are
different types, and mixing them is a compile error rather than a runtime surprise. A combination
that has no implementation does not fail at load time — it fails to build.

This is why quantization has no checkpoint format. The converter always writes BF16, and the
reduced-precision path is chosen by the type: `PerChannelFp8<>` and `PerGroupFp4<128>` quantize the
weights as they load.

## A library, not a framework

Your application owns `main()`, the loop, and the tools. Mila makes the model an ordinary C++ object
inside them, rather than an engine that owns execution and calls back into your code. The tempting
analogy — a game engine for inference — points the wrong way. Mila is raylib, not Unity.

The consequence is a boundary rule the codebase holds to: the runtime accepts only what is
model-intrinsic and consumer-blind. There is no services layer for chat sessions, conversation
history, or prompt builders, because that is how a hidden execution engine accretes one convenience
class at a time.

## Tractable, not just readable

Readable is the visible property. The useful one is that every layer can be taken apart.

Because there is no execution engine, a profile names your own code — every kernel in an Nsight trace
is a file in this repository, not a fused operation a graph compiler generated and named for you.
Because each component owns its parameters and gradients and shares no global state, one can be
constructed and exercised on its own. Because dispatch resolves at compile time, the kernel that ran
is knowable from the type that selected it.

The consequence is pace. A wrong number can be walked back to the component, the kernel and the line
that produced it, instead of being inferred from the output of a system you cannot open. Much of the
[blog]({{< relref "blog" >}}) is a record of exactly that — bugs isolated by taking a model apart one
component at a time and checking each against a reference.

## What that buys, and what it costs

Three architecture families — Gemma 4, Llama 3.x and GPT-2 — each reproduce a HuggingFace reference
token-for-token, on consumer hardware. Being fully explicit costs a few percent: Mila runs within
1.03x of llama.cpp on decode and 1.14x on prefill at long context.

It is not a breadth competitor. llama.cpp runs everything, everywhere; vLLM serves the datacenter.
Mila runs a short, curated set and holds each one to parity, with the whole path readable.

## Where to go next

- The [blog]({{< relref "blog" >}}) covers how the kernels were written, profiled and rewritten.
- [Get started]({{< relref "start" >}}) is the toolchain, the build, and a first local inference run.
- The [source](https://github.com/ToddThomson/Mila) is MIT licensed, and the design is open to argument.
