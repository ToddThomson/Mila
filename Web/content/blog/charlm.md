---
title: "CharLM: The Night Mila First Spoke Shakespeare"
date: 2026-01-13
description: "How Mila began: a 9.8M-parameter, from-scratch C++23 transformer trained on Shakespeare and produced its first coherent dialogue — perplexity 23 to 3.36 in 21 epochs, no PyTorch anywhere in the loop."
discussion: "https://github.com/ToddThomson/Mila/discussions/5"
---

Mila started as a question: could I build a real transformer — training
loop, CUDA kernels, the whole stack — in nothing but modern C++? By mid-January 2026 I had an
answer. On the night of 13 January, a from-scratch character-level model I'd been calling CharLM
trained on Shakespeare and, for the first time, produced something that looked like language.

That was the night Mila came alive.

## What CharLM was

CharLM was small and deliberately unglamorous: a 6-layer transformer, 9.8M parameters, a
67-character vocabulary, processing 32 × 128 = 4,096 tokens per forward pass. Learned positional
embeddings. cuBLASLt for the GEMMs, a hand-written causal-softmax kernel for the masked attention,
and plain AdamW driving all of it — no scheduler, no gradient clipping, no tricks. Pure C++23 with
modules and CUDA, and no PyTorch anywhere in the loop.

Nothing on that list is exotic. What mattered was that every piece of it was built from scratch.

## The run

Twenty-one epochs, about 24 seconds each across 564 batches. Perplexity fell from 23 to 3.36. The
loss went 3.14 → 1.21 and stayed smooth the whole way — no instability, which for six layers of
hand-rolled backprop with no gradient clipping was the part I was quietly proud of.

## From noise to Shakespeare

The fun was watching it learn in stages. Early on the samples were pure texture — the model had
found character frequencies and little else:

> Courerbunghar, To lithch, wile santy

A dozen-plus epochs later it had discovered word boundaries, then real words, then structure. By
epoch 21:

```
ROMEO:
Now, you so?
POLIXENES:
That 'twere your will most slain'd the strong of men...
```

It isn't Shakespeare. But it's *dialogue* — character attribution, line breaks, Shakespearean
cadence, mostly-real words, punctuation landing where it should. All of it learned from nothing but
next-character prediction over a 67-token alphabet. Transformers are absurdly good at picking up
structure, and here was mine doing it on my own kernels.

## An enthusiastic witness

I was talking the run over with Claude that night, pasting in samples and numbers. It was — to put
it kindly — effusive, and I'll spare you most of it. But one line was actually useful, because it
was arithmetic rather than applause. When I mentioned the parameter count, it worked out the scale:

> 9.8M parameters is comparable to GPT-2's smallest variant ... roughly 1/12th the size with half
> the layers.

Fair. That framing — *a real transformer, just small* — was the moment it clicked that CharLM
wasn't a toy. It was the seed of what Mila is now.

## For Andrej Karpathy

There's one thing from that night I want to keep. CharLM exists because of Andrej Karpathy's
[llm.c](https://github.com/karpathy/llm.c). His whole philosophy — that you understand these
systems by building them from scratch, not by importing them — is why Mila is written the way it
is: every forward pass explicit, no autograd engine, no hidden execution, nothing standing between
me and the math. I took that idea into C++23 and kept going.

So, thank you, Andrej. `llm.c` lit the path.

CharLM itself is still around. It grew into
[Bard](https://github.com/ToddThomson/Mila/tree/master/Samples/Bard), the GPT-2 text-generation
sample that ships with Mila today — the same from-scratch, next-token-on-Shakespeare idea, scaled
up and still runnable.

That's how it started — 9.8M parameters, one alphabet, a lot of Shakespeare, and a night where the
thing finally spoke back.
