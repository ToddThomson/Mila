---
title: "The Lobotomized Attention Head Bug: One Head Does All the Work, the Others Stare Into the Void"
date: 2026-02-26
description: "A transformer debugging rite of passage: the fast prefill-to-decode path looked correct, but only one attention head was doing anything."
discussion: "https://github.com/ToddThomson/Mila/discussions/6"
---

I just finished building the fast prefill → decode inference path in my Mila DNN library.
Everything seemed fine — the model produced coherent text, KV caching worked, and decode mode
looked solid.

But my transformer's residuals were way off compared to 🤗 HF GPT-2, and the hidden states just
felt *wrong*. Not exploding, not NaN'ing — just wrong. After hours of combing through attention
math, KV cache, QKV packing, LayerNorm, and positional encodings... I found it.

## 🧪 Root Cause

In the MHA prefill path, my `unpermute_output` kernel was wrong. It needed a padded variant
(`unpermute_output_padded`), and instead of writing *all* the attention heads back into the output
tensor, it wrote back exactly one.

It wrote back **one** head.\
All the other heads?\
Nowhere. Silent. Forgotten. **Lobotomized.**

## 🤡 The Symptoms (that still produced coherent text)

- Hidden states completely misaligned from HF
- Residuals with huge swings
- Prefill corrupted → decode still worked (go figure)
- Yet the model *still* produced coherent sentences

Transformers are absurdly resilient.

## 🔍 Why It Still "Worked"

- The decode path was correct, so per-token incremental attention was fine
- LayerNorm aggressively stabilized everything downstream
- The MLP and embeddings carried most of the workload
- Attention had quietly become "single-head attention + moral support"

## 🎉 The Lesson

If you ever see all of these at once:

- Prefill mismatch
- HF vs. your model drifting hard
- Residuals acting hyperactive
- ...yet decode still produces intelligible sentences

Check your unpermute logic. A single bad stride or head offset can quietly *turn off* most of
attention — and the rest of the transformer is resilient enough to hide it from you.

## 🪦 Memorial

> In loving memory of Attention Heads 1–11.\
> They attended every forward pass.\
> They contributed nothing.\
> They will be missed.
