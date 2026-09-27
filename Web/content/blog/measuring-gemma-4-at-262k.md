---
title: "Measuring Gemma 4 at 262,144 Tokens: Four Times the Number Lied"
date: 2026-09-26
description: "I set out to answer one question -- does Gemma 4 12B's quality hold past 131,072 tokens? -- and the log-likelihood harness gave me a wrong answer four different ways before it gave me a real one."
draft: true
---

Gemma 4 12B's weights declare a context of 262,144 tokens. Mila's chat harness caps it at 131,072, and since the planner learned to read the weights, the Python binding does not: on a 16 GB RTX 5060 Ti the same model now opens at 262,144 from Python and at 131,072 from chat. One of those is wrong, and the only way to find out which is to measure quality across the whole range.

That sounds like an afternoon. Score some long text at a few context lengths, check the perplexity does not fall off a cliff past 131,072, delete the cap or keep it. It was not an afternoon, and every number the harness produced along the way was wrong in a different, instructive way.

## A Perplexity of 1,800

The first thing Gemma needed was a way to score text at all. Mila had one for Qwen: run a prefill over a sequence, evaluate the output head on a window of rows at a time, and accumulate the log-probability of each actual next token. Gemma got the same loop, plus one thing Qwen does not have -- a final-logit softcap, `30 * tanh(logit / 30)`, which Gemma applies before sampling and which therefore has to be applied before the log-softmax too.

The softcap is monotonic, so argmax cannot see it: a model scored without it picks exactly the same tokens and reports a slightly wrong number. The only gate that catches it is an absolute one. On a tiny routed-expert Gemma at FP32, Mila's total log-likelihood matches HuggingFace's -72.0917 to within 5e-7. Without the softcap it misses by 4.0e-3, so the gate can fail.

Then I pointed it at the real 12B and wikitext-2: 16,195 positions, a mean of 7.4960777 nats per token. A perplexity of 1,801. On Wikipedia. For a 12-billion-parameter model.

The first run had been worse -- **24,986** -- because Gemma is trained with `<bos>` at the start of every sequence and its tokenizer does not add one. Chat writes it into the prompt text; a harness that encodes raw text has to add it itself. With `<bos>`, 1,800.

So I ran HuggingFace's own BF16 model on the same 256 token ids. It read **2,182**. The tokenizers agree id for id. The measurement was right. Gemma 4 12B *it* is instruction-tuned, and raw wikitext is a long way from the text it was tuned on.

That is odd, but it is not the lie. The lie comes next.

## Rounding That Makes the Model Better

Mila runs Gemma's FP4 prefill as W4A8: activations are quantized to FP8 per token, and the FP4 weights are re-expressed in FP8 under one scale per tensor, which buys 1.285x prefill throughput over dequantizing to BF16. Decode uses W4A16. To see what W4A8 costs, I scored the same four segments (4,092 positions) every way I could, in both Mila and HuggingFace:

| Weights | Arithmetic | Mean nats/token |
|---|---|---|
| BF16 | HuggingFace | 6.388 |
| FP8 | HuggingFace | 6.523 |
| FP8 | Mila | 6.507 |
| FP4 | HuggingFace, BF16 activations (W4A16) | 7.580 |
| FP4 | HuggingFace, W4A8 emulated | 7.336 |
| FP4 | Mila (W4A8) | 7.253 |

The FP8 rows agree across implementations, which is what you want. The FP4 rows say **W4A8 is better than W4A16** on the same weights -- that adding a second quantization, to the activations, improves the model.

It does not. Recomputed in FP64 from the package's own FP4 bytes, Mila implements W4A8 to within 1e-4 per projection, and W4A8 itself sits 2.0% to 3.6% from W4A16 per projection. What happens is that on raw wikitext this model is *confidently wrong*: it puts most of its mass on tokens that do not occur. Extra rounding flattens its distributions, which raises the probability of the tokens that do occur, which lowers the loss.

**So raw-text perplexity cannot rank precisions on this model.** That decided something I had not expected it to. A separate open question is whether Google's quantization-aware 4-bit checkpoint beats Mila's FP4, and the plan was to compare their wikitext perplexities. That comparison would have rewarded whichever build was noisier. It is now a KL divergence from BF16's next-token distribution, which does not care how well the model fits the text.

## Two Answers From One Head

The harness evaluates the output head on a *window* of rows per pass. At a window of one row, Gemma's FP8 tied head runs the decode matvec. At 64 rows it runs the batched prefill GEMM. They should give the same log-likelihood, and I wrote down a bound of 1e-3 relative perplexity before the first run. It measured **1.06e-3**.

Computing the logits exactly in FP64 from the embedding table over 64 positions separated the two paths cleanly:

| Head path | Mean \|logit error\| vs FP64 | Perplexity vs exact |
|---|---|---|
| Window 1 (matvec) | 0.015564 | -6.4e-4 |
| Exact, rounded once to BF16 | 0.015564 | -- |
| Window 64 (staged GEMM) | 0.017216 | +4.9e-4 |
| Model: `bf16(fp8 * scale)` before the GEMM | 0.017215 | -- |

The matvec is exactly as good as a head that writes BF16 logits can be. The staged path does one extra thing -- it stores each `fp8 * scale` as BF16 before the GEMM, where the matvec applies the per-channel scale once after the dot product -- and a model of that one rounding reproduces its error to four digits. The two errors have opposite signs, so they differ by about 1.1e-3. The bound went to 2e-3 with the cause named in the test, and moving the scale after the dot product is its own change.

## Where the Long Text Comes From

For lengths past 131,072, wikitext-2 is useless -- its whole test split barely fills one window. I needed single documents longer than 262,144 tokens with a licence I could verify, and PG-19 is exactly that: DeepMind's set of Project Gutenberg books published before 1919, Apache 2.0, public domain texts. Its test split is 100 books, the longest 4.5 million characters, and a handful fill the window on their own.

Measuring by context length usually means building the model at each length and cutting the corpus into segments of that length. That is several deployments, several prefill-chunk choices, and a corpus spent many times over. There is a cheaper way. Scoring is causal: position *p*'s prediction depends only on tokens before *p*. If every prefix ends on a prefill-chunk boundary, the positions a shorter prefix scores come out identical in a longer one. So one network built at 262,144 can score each book at prefixes of 8K, 16K, ... 256K, and the difference between two prefixes is the log-likelihood of the band between them *given the whole book before it*:

```cpp
for ( std::size_t index = 0; index < prefixes.size(); ++index )
{
    const SequenceLogLikelihood scored = Common::sequenceLogLikelihoodOf( *network, prefix( prefixes[ index ] ) );

    const double band = scored.total_log_probability - previous.total_log_probability;
    const dim_t positions = scored.scored_positions - previous.scored_positions;

    band_log_probability[ index ] += band;
    band_positions[ index ] += positions;
    previous = scored;
}
```

The top band, 131,072 to 262,144, is the question itself. The cost is about twice the longest run. The planner confirmed the 5060 Ti holds the whole thing:

```
>>> mila.GemmaModel.from_store('gemma-4-12b-it-fp4', 'auto', 1).context_length
262144
```

(After `mila.initialize()`. Without it, the same call raises "CUDA device 1 does not report its free memory" -- on both cards, pointing at the GPU when the cause is the missing call. That message is now a bug.)

I wrote the gate down before the run: the top band's perplexity at most 1.05 times the band below it.

## A Perplexity of 10,000

```
window 64, context 262144, prefill chunk 1024
30312.txt: 9.2298 10.0243 9.3116 9.4595 10.6905
```

Nine to ten and a half nats per token, on 19th-century prose. That is a perplexity around 10,000 -- five times worse than wikitext, and not far from guessing uniformly over a 262,144-token vocabulary (12.5 nats). And no band improved on the one before it: 64K tokens of a book in context bought nothing.

A model that is predicting the text gets better as it reads more of it. This one was not predicting the text. It was about to take six to eight hours to tell me so, so I stopped it and asked a smaller question on the other card: is it the format? The same 32,000 characters, scored three ways:

```
as stored                     8039 tokens, 9.2561 nats/token (perplexity 10468.6), 2.3253 nats/character
wraps joined                  7428 tokens, 9.6149 nats/token (perplexity 14987.0), 2.2319 nats/character
wraps joined, in a turn       7428 tokens, 3.7867 nats/token (perplexity 44.1), 0.8790 nats/character
```

"In a turn" means the book is the model's reply:

```
<bos><|turn>user
Continue this book.<turn|>
<|turn>model
<|channel>thought
<channel|>...the book, wraps joined...
```

**From 10,000 to 44 by wrapping the text in a chat template.** Per character, which compares across the different tokenizations, loss drops from 2.23 to 0.88 nats. Joining Gutenberg's hard 70-column line wraps helps a little per character; the turn is nearly all of it. Outside its own turn format, this instruction-tuned model is barely a language model at all. That is also the whole explanation for wikitext's 1,800.

## A Perplexity That Rises

So G2 went into a turn and started again, on three books:

```
30312.txt: 3.7867 3.8163 4.1105 4.4483 4.7902 5.3056  (2694 s elapsed)
3608.txt: 3.7942 3.6700 3.6503 3.8337 4.1779
```

| Book | 0-8K | 8K-16K | 16K-32K | 32K-64K | 64K-128K | 128K-256K |
|---|---|---|---|---|---|---|
| 30312 | 3.787 | 3.816 | 4.111 | 4.448 | 4.790 | 5.306 |
| 3608 | 3.794 | 3.670 | 3.650 | 3.834 | 4.178 | -- |

The level is sane now, and the curve is wrong. Loss **rises** with context -- from about 16K on book 30312, from about 32K on book 3608. The top band fails the gate at 1.67 against 1.05. But that gate was written to catch a model falling off a cliff past its trained range, and this is not a cliff at 131,072: perplexity roughly doubles between 8K and 128K before the top band adds to it.

(The second book's 262,144 prefix then died of out-of-memory at the 71-minute mark, with another process holding memory on the card. 262,144 fits the 5060 Ti with very little to spare.)

There are three stories that produce a rising curve. The later text is harder. The model is fine with long context but has never produced a 100,000-token reply and degrades inside one. Or something in Mila degrades with length. One diagnostic, two arms, on the other card:

```
fresh         book tokens      0 + 8192: 3.7842 nats/token
fresh         book tokens   8192 + 8192: 3.6778 nats/token
fresh         book tokens  16384 + 8192: 3.7878 nats/token
fresh         book tokens  32768 + 8192: 3.7378 nats/token
fresh         book tokens  65536 + 8192: 3.7393 nats/token
fresh         book tokens  98304 + 8192: 3.7659 nats/token
conversation  book tokens      0 +16384: 3.7929 nats/token
conversation  book tokens  16384 +16384: 4.0324 nats/token
conversation  book tokens  32768 +16384: 4.1626 nats/token
conversation  book tokens  49152 +16384: 4.3144 nats/token
```

*Fresh* scores 8K stretches from deeper in the book, each alone in its own turn with no book before it. They are flat at 3.7 to 3.8: the text does not get harder. *Conversation* splits the first 64K into four 16K model turns, each after a user saying "Continue." It still rises, only a little less than one long turn does.

Put the two arms side by side and they say something stark: **with 64K of the book in context, the model predicts the next page worse than with none of it.** A model using its context correctly does not do that.

## Decode Agrees With Prefill

If Mila is at fault, the obvious suspect is the prefill attention kernel, which is the code that handles a long context in chunks. Decode is different code: one token at a time through a flash-decoding kernel. So score the same 257 targets both ways, deep in the book and near the start:

```
after  4096: decode 3.8834 nats/token over 257, prefill 3.9250 over 257
after 32768: decode 3.9042 nats/token over 257, prefill 3.9328 over 257
```

They agree, by the same small margin at 32K as at 4K -- the W4A8 gap from earlier. Whatever grows with length is something both paths share: the weights, the rotary position tables, the KV cache, or the model itself.

## What HuggingFace Says

**[Pending -- the run is in progress.]** HuggingFace scores the first 32,768 tokens of book 30312's turn, on the FP4 weights rebuilt exactly as Mila quantizes them, fed in 1024-token chunks through its own KV cache so no attention matrix is wider than one chunk against the context. Its prompt tokenizes to the same 17 ids as Mila's:

```
prompt ids: 2 105 2364 107 34230 672 2260 236761 106 107 105 4368 107 100 45518 107 101
```

If HuggingFace stays near 3.8 through 32K, where Mila reads 4.111 over 16K-32K, the defect is Mila's, and it lives in what decode and prefill share. If HuggingFace rises too, then Gemma 4 12B *it* genuinely predicts a book worse the more of it it has read -- and the question "does quality hold past 131,072?" needs a different instrument than perplexity.

## What I Took From It

- **Perplexity is a property of the model *and* the text's format.** For an instruction-tuned model, text outside its turn format is out of distribution, and the number measures that more than the model. Here it moved perplexity from 10,000 to 44.
- **A number that improves when you add noise is measuring the wrong thing.** W4A8 beating W4A16 was the tell that raw-text perplexity could not rank precisions on this model.
- **Separate the text from the position before blaming either.** Scoring the same stretch fresh and in context is two runs and settles whether the text got harder.
- **Prefix differences turn one long run into a whole curve**, as long as scoring is causal and every prefix ends on a chunk boundary.
- **Write the gate before the run.** The 1.05 bound failed for a reason it was not written to catch, and it was the rest of the table that said so.
