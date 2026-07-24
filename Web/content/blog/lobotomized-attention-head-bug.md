---
title: "The Lobotomized Attention Head Bug: One Head Does All the Work, the Others Stare Into the Void"
date: 2026-02-26
description: "A transformer debugging rite of passage: the fast prefill-to-decode path looked correct, but only one attention head was doing anything."
discussion: "https://github.com/ToddThomson/Mila/discussions/6"
---

🧠 The Lobotomized Attention Head Bug — A Transformer Debugging Rite of Passage

I just finished building the fast prefill → decode inference path in my Mila DNN library.
Everything seemed fine — the model produced coherent text, KV caching worked, and decode mode looked solid.

But… my transformer’s residuals were way off compared to 🤗 HF GPT-2, and the hidden states just felt wrong.
Not exploding, not NaN’ing — just wrong.

After hours of combing through attention math, KV cache, QKV packing, layernorm, and positional encodings…

I found it.

🧪 Root Cause

In the MHA prefill path, my unpermute_output kernel was wrong.
It needed a padded variant (unpermute_output_padded), and instead of writing all attention heads back into the output tensor…

It only wrote back ONE head.
All the other heads?
Nowhere.
Silent.
Forgotten.
Lobotomized.

🤡 Symptoms (that still produced coherent text!)

Hidden states completely misaligned from HF

Residuals with huge swings

Prefill corrupted → Decode still worked (go figure!)

Yet… model still produced coherent sentences
(Transformers are absurdly resilient.)

🔍 Why it still “worked”

Decode path was correct (so per-token incremental attention was fine)

LayerNorm aggressively stabilized everything

MLP + embeddings carried most of the workload

Attention became “Single-Head Attention + Moral Support”

🎉 Lesson

If you ever see:

Prefill mismatch

HF vs your model drifting hard

Residuals acting hyperactive

Yet decode produces intelligible sentences…

Check your unpermute logic.
A single bad stride or head offset can quietly “turn off” most of attention.

