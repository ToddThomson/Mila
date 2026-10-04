# Gemma 4 MTP

Speculative decoding for Gemma 4 12B and 26B-A4B, each with the draft model Google trained for it.

*Status: design, 2026-10-04. Stage 1 is built and measured for both models (5.1): the drafter component, its
parity with HuggingFace, its acceptance and the verify cost; the loop (stage 2, 4.7) is not built. Supersedes `SpeculativeDecoding.md` for Gemma: that document
proposed a drafter with its own KV cache and a compile-time drafter axis, and Google's drafter is
neither. The work is `BACKLOG.md`, Gemma 4 Complete, "Nobody knows whether Google's drafters would make
Gemma 4 decode faster".*

---

## 1. What It Is

Google ships a small draft model (an "assistant") beside each Gemma 4 size; the quantization-aware 12B
has its own, `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant`, the pair this document builds. The
drafter proposes K tokens, the 12B checks all K in one forward, and every token the 12B agrees with is
kept. The 12B always contributes at least one token per round, so output follows the 12B's distribution
exactly; only the rate changes. Google calls this multi-token prediction (MTP); HuggingFace runs it as
`SinglePositionMultiTokenCandidateGenerator`.

It is for both models (section 6, item 4). The 12B's drafter is the one sections 2 to 5 read and measure
first; the 26B-A4B's is its own checkpoint, and the design carries over to it at the 26B's width. Google
says a mixture of experts gains little from drafting at batch 1; the 26B-A4B is measured rather than
excused on that account.

Google's claim (blog.google, "multi-token-prediction-gemma-4", read 2026-09-27): "up to a 3x speedup",
about 2.2x on Apple Silicon at batch 4 to 8. No batch-1 figure, no acceptance rates, nothing on quantized
targets or long context -- the cases this document is for, so the number is measured here (5.1).

A community report (carteakey.dev, "Gemma 4 26B QAT + MTP", 2026-06-12, read 2026-10-04 at Todd's pointer;
an AI-written post of single runs with no acceptance rates) runs llama.cpp mainline's drafter support
(`--spec-type draft-mtp`, a Q8_0 drafter GGUF) on the **26B-A4B**: 69.0 to 100.6 tokens/s on an RTX 4070, best at
2 drafts per round. Two of its findings bear on this design:

- **A quantized KV cache "degrades the MTP acceptance rate to near zero on Gemma 4"** (q8_0 there). The drafter
  attends the target's cache (2.1), so the cache's format is part of the drafter's input. Stage 1 measures
  acceptance at both of Mila's cache formats, BF16 and FP8 (5.1).
- **The 26B-A4B gained 1.46x**, against Google's account that a mixture of experts gains little at batch 1. Its
  target did not fit the 12 GB card and ran partly on the CPU, which makes every target step slow and drafting
  worth more; on the 16 GB card Mila's 26B-A4B is resident, and its global cache is FP8. Both are what its own
  measurement settles (section 6, item 4).

## 2. The Drafter

Read 2026-10-04 from transformers 5.12.1 (`gemma4_assistant/modeling_gemma4_assistant.py`,
`generation/candidate_generator.py`, `models/gemma4/modeling_gemma4.py`) and the drafter's `config.json`;
shapes from its safetensors header.

| Property | Value |
|---|---|
| Architecture | `Gemma4UnifiedAssistantForCausalLM` |
| Layers | 4: three sliding (window 1024), one full |
| Hidden size | 1024 |
| Query heads | 16; head size 256 sliding, 512 full |
| Key/value projections | **none** (`num_kv_shared_layers` 4) |
| MLP | GeGLU, 8192 |
| RoPE | as the 12B: theta 1e4 sliding; 1e6 proportional, factor 0.25, full |
| `pre_projection` | [1024, 7680]: 2 x 3840 in, 1024 out |
| `post_projection` | [3840, 1024]: back to the 12B's width |
| Head | its own tied table, [262144, 1024] |
| `use_ordered_embeddings` | **false**: the centroid head (2048 clusters, top 32) is not used |
| Precision | BF16 |

Parameters, derived from those shapes: three sliding layers 100.7M, the full layer 41.9M, the two
projections 11.8M, the head 268.4M -- **about 423M, 0.85 GB at BF16, 63% of it the head.**

### 2.1 What it reads

The drafter has no cache of its own. Each of its layers projects queries only, and attends the **12B's**
keys and values: its three sliding layers the cache of the 12B's last sliding layer, its full layer the
cache of the 12B's last global layer (layers 46 and 47 of 48). What it reads is exactly what Mila caches:
keys after `k_norm` and RoPE, values after `v_norm` (`modeling_gemma4.py`, `store_full_length_kv`).

Its input is the 12B's token embedding of the last token (with Gemma's sqrt(3840) scale) joined to the
12B's last hidden state **after the final norm** -- the row the 12B's head reads -- in that order, then
`pre_projection` to 1024.

### 2.2 One round

At the start of a round the 12B has accepted every token through position `p - 1` and has chosen the
token `t` at position `p`, which it has not yet processed. It hands over `h`, its final-normed hidden at
position `p - 1` (the row that predicted `t`).

1. **Draft.** For `i = 1..K`: `x = pre_projection([embed(t), h])`; run the four layers on `x` at
   position `p` -- **the same position every step** -- attending the two shared caches through position
   `p - 1`; `d_i = argmax(head(x))`; `h = post_projection(x)`; `t = d_i`.
2. **Verify.** The 12B processes `[t_0, d_1, ..., d_K]` at positions `p..p+K` (where `t_0` is the token
   the round started with), writing their keys and values, and produces logits at all `K + 1` rows.
3. **Accept.** Greedy: keep `d_i` while it equals the 12B's argmax at row `i - 1`; at the first
   mismatch, or after all K, the 12B's own token at that row is the bonus. Sampling: section 4.4.
4. **Rewind.** With `m` drafts accepted, the cache is valid through position `p + m`; the rows written
   past it are discarded. The next round starts at `p + m + 1` with the bonus token and the 12B's hidden
   at row `m`.

Only step 1 is new arithmetic. Steps 2-4 are the 12B's forward, its head over several rows, and the
rewind `GemmaModel` already uses for prompt reuse.

## 3. The Cost Model

A round produces `m + 1` tokens for K drafter steps and one `(K + 1)`-row verify:

```
speedup = (E[m] + 1) x T_decode / (K x T_draft + T_verify(K + 1))
```

- `T_draft`: the drafter reads about 0.85 GB a step at BF16 -- the head 0.54 of it. With the head at the
  12B's own table format (six bits per 32, `Quantization.md` Part II) the head is about 0.22 GB.
  Derived, not measured; the measurement replaces it.
- `T_verify(K + 1)` against `T_decode`: decode is bandwidth-bound, so a verify that reads the weights
  once for K + 1 rows can cost near one decode. Whether it does depends on which path runs it (4.2).
- `E[m]`: the acceptance length, measured in Mila (5.1).

If the measurement says the product is not a gain, that is the recorded result and the loop stays out
(the BACKLOG entry's stop condition).

**The 26B-A4B's terms differ in two ways** (derived from its shapes, not measured). A decode reads its 8 routed
experts per layer -- about 0.8 GB at Q4_0 -- beside about 1.4 GB of attention, dense branch and head, some 2.2 GB in
all. A verify's K + 1 rows each route their own 8, so it reads the union of their experts, up to 40 per layer at
K = 4: a 5-row verify reads the shared bytes once and roughly three times the expert bytes, about 1.9 decodes against
the 12B's near 1. And its drafter, the same 0.85 GB a step at BF16, is about 0.35 to 0.4 of its faster decode against
0.11 of the 12B's, so the six-bit head (decision 2) weighs more there. Both are what its stage 1 measures; the expert
union is a term of its multi-row decode.

## 4. Design

### 4.1 A feature, built or not

The drafter is a component the network builds or does not, selected per deployment (`ModelHandle.md`
3.8, `Deployment.md` 2.1): never a template axis, so selecting it is not a new instantiation of the
12B. Not selected, it allocates nothing and the decode loop is today's. It is priced by the planner like
any other selection: its weights and its scratch -- it has no KV cache.

It is a sibling of the blocks, not a block: `GemmaDrafter`, a Gemma 4 transformer component the
`GemmaTransformer` owns as a child when it is selected, owning its four layers, both projections and its head.

**It runs on the target's execution context** -- one stream, one decode position, one scratch buffer. It reads
the target's caches and final-normed hidden state in stream order with no handshake; its query rotates at the
context's decode position, from which its attention derives the position before it on the device; and its
decode-attention scratch is the context's shared buffer, so the network reserves the drafter's request with its
own and the planner prices it. A context of its own would cost an event pair every draft step and put its
launches outside the target's recording. A drafter on a second card is `LayerSplit.md`'s, after this release.

Its layers are Gemma blocks without key and value projections. The target gives it read access to its last sliding
and last global layer's caches and to its final-norm output; that is the only coupling.

### 4.2 Verify: which path runs K + 1 rows

The verify is the one decision that decides both the cost and what "lossless" means.

- **(a) The prefill path.** `prefillFrom` at offset `p` over K + 1 tokens, with the head at every row (as
  `sequenceLogLikelihoodFrom` already evaluates it). It exists today. But prefill is not decode's
  arithmetic: the Q4_0 prefill multiplies INT8 activations (`Quantization.md`, Q4_0 decision 4), decode
  multiplies BF16 activations, and the attention kernels differ. Greedy output with the drafter could
  then differ from greedy output without it wherever the 12B's top two tokens are nearly tied. It is also
  a GEMM path tuned for 1024 rows, run at 5.
- **(b) A multi-row decode.** The decode path at `outer_size` K + 1: each layer reads its weights once and
  applies decode's per-row arithmetic to every row; decode attention runs each row at its own position,
  causal among the K + 1. Every row then computes exactly what a decode at that position computes, so
  greedy output is token-for-token the plain loop's **by construction**, and at small row counts this is
  the bandwidth-bound kernel shape anyway. It is new: decode is `outer_size == 1` throughout today, and
  each decode op needs a K + 1 form.

**Decided: (b), with (a) as the measurement's first arm** (section 6) -- (a) needs no code, so it prices
the loop before (b) is built, and (b) is built only if the measurement says the loop pays. Stage 1 says it pays
(5.1); stage 2 builds it, designed in 4.7.

### 4.3 The drafter's attention

A drafter step is a decode-shaped attention that **reads the 12B's cache and writes nothing**: a query at
position `p`, keys and values at positions up to `p - 1`, in whatever format the 12B caches them (the
sliding ring; BF16 or FP8). The decode attention kernels already read both formats; the step needs their
read without the write. The drafter's grouping follows the 12B's caches: 16 query heads over 8 key/value
heads sliding, over 1 global.

If the 12B's global layers come to store values only and build keys on read (`RopeInAttention.md`), the
drafter's full layer reads through the same path. The two land in either order; the second adapts.

### 4.4 Sampling

Chat samples by default (temperature 0.8), so a greedy-only loop would not reach Chat's default use. The
lossless sampled rule (Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*,
arXiv 2211.17192; HuggingFace `_speculative_sampling`): accept `d_i` with probability
`min(1, p_i(d_i) / q_i(d_i))`, where `p` is the 12B's distribution and `q` the drafter's after the same
temperature, top-k and top-p; on the first rejection, sample from `normalize(max(0, p_i - q_i))`; if all
K are accepted, sample the bonus from `p_K`. It needs the drafter's logits as well as its tokens, and
runs on the device sampler (`TokenSampling.md`) with host-drawn uniforms, so a seeded run stays
reproducible. Greedy lands first and gates the loop; sampling follows it before the feature is offered.

### 4.5 Rewind and the sliding ring

The rewind is short and always safe. A sliding ring holds `window + prefill_chunk - 1` rows
(`SlidingWindowKvCache.md` D2), and `CudaGqaOp::rewindKvCache` accepts a rewind of up to
`prefill_chunk - 1` positions. A verify writes K + 1 rows past `p - 1`, so every rewind the loop makes is
accepted whenever `K + 1 < prefill_chunk` -- K is a handful, the chunk is hundreds. The ring's save and
restore (BACKLOG, "A Gemma reply longer than about 1,024 tokens...") is not needed here.

### 4.6 Decode replay

The drafter's K steps are decode-shaped and replayable (`DecodeGraph.md`): within a round the position
is constant, across rounds it is not, so it is read from the execution context like any decode position,
and the step must otherwise be a pure function of device memory. A multi-row verify (4.2 b) is its own
recording. The acceptance decision needs the tokens on the host once per round, not once per token.

### 4.7 The multi-row decode

*Design, 2026-10-04; for Todd's review before code.* A verify is a decode of R = K + 1 rows at positions p to
p + K: one forward in which **every row computes exactly what a decode at its position computes**, and every weight
is read once for all R. Family-neutral by construction: it is the decode path widened, not a Gemma path.

**The entry.** `LanguageModelNetwork::decodeRows( tokens [1, R], position )` beside `decode`, returning logits
[1, R, vocab]; it sets the context's decode position to p, as `decode` does, and advances the cached length to
p + R. Each family implements `onDecodeRows`, Gemma first; a family without one refuses. `ITransformerBlock` gains
`decodeRows( input [1, R, model_dim], position )`. The final-norm output keeps all R rows, since the drafter's next
round reads the row at the last accepted position.

**What each operation does with R rows, and what changes:**

| Operation | Today at one row | At R rows |
|---|---|---|
| Norms, residuals, scale, GeGLU, embedding | Row-count blind | Unchanged |
| `Linear`, every weight format (Q4_0, BF16, FP8, FP4, six-bit head, codebook) | One warp per output channel, lanes striding the input, a fixed shuffle tree | An R-row matvec: each weight word unpacked once, R accumulators, each following today's order exactly -- so each row is bit-identical to a one-row decode, and the weights are read once. R up to 9, a template parameter, so register pressure is per instantiation |
| RoPE | Reads the decode position for its one row | Row r at `*position + r` (one-line change; row 0 is today's) |
| KV write | Writes one row at the decode position | Writes R rows from it (the same one-line change) |
| Decode attention (BF16 and FP8 caches) | One query at the decode position, the band ending there, split-K chosen on the device by band length | R queries, row r's band ending at p + r, its split count chosen as a one-row decode at p + r would; the scratch is R times one row's. A grid dimension, not a new kernel |
| Expert bank (26B-A4B) | Gathers the token's 8 experts | Groups the R rows' (row, expert) pairs by expert, so each chosen expert's weights are read once for every row that chose it; each row combines its experts in its own routing order, as a one-row decode does. This is the expert union section 3 prices |
| Head (tied six-bit table) | The six-bit matvec | The R-row form of it, as every `Linear` |

**Lossless by construction, and gated as such**: greedy output with the drafter equal to without it, token for
token, because each row's arithmetic is a one-row decode's; the R-row kernels are tested bit for bit against R
calls of the one-row kernel, format by format, before the loop is.

**Replay.** The verify is a recording of its own (`DecodeGraph.md`), keyed by R: its token buffer and the context's
decode position are device memory, so one recording replays every round at a fixed K.

**Order of work**, each step gated before the next: (1) the R-row Q4_0 matvec and its bit-for-bit test, and a
microbenchmark of R = 1 to 9 against one row -- the first number that prices (b), with no loop built; (2) every other
`Linear` format and the head; (3) RoPE, KV write and attention at R rows; (4) `decodeRows` through Gemma's blocks,
gated equal to R decodes; (5) the expert bank's union and its measured cost on the 26B-A4B; (6) the loop, greedy,
then sampled (4.4).

## 5. Gates

Each stage advances only when its gate holds.

### 5.1 Stage 1 -- the drafter's forward, and the measurement

- **Parity:** the drafter's logits against HuggingFace's `Gemma4AssistantForCausalLM.forward` on
  **explicit inputs** -- a token, a hidden state and the two caches, dumped from a Mila 12B run at BF16
  cache -- not against HuggingFace's generation loop. The drafter fits any card and runs on the CPU, so
  the oracle needs no 12B in HuggingFace.
- **The loop is not the oracle**, because two things in it are not what this document specifies, and
  neither has been confirmed by a run: after a partly rejected round, HuggingFace cuts the shared caches
  at the new sequence length, which keeps the rejected draft's row at the bonus token's position; and on
  the first round after the prompt it appears to take the hidden state of the prompt's **first** row
  (`n_last_matches` 0 against a full-prompt output). Mila attends exactly the accepted rows and hands
  over the row that predicted the bonus.
- **Acceptance length:** `E[m]` for K = 1..8, measured in Mila with the drafter's forward and the 12B's
  existing paths, on prompts of the kinds Chat and agents run (conversation, code, tool calls), greedy
  and at Chat's sampling settings, with the 12B's cache at BF16 and at FP8 (section 1). Measured in Mila
  because the BF16 12B does not fit the 16 GB card. It needs no loop: at each position of a plain greedy
  run the drafter drafts K tokens from the 12B's state there, and `m` is how many match the 12B's own
  continuation; sampled, the expected acceptance of draft i is the sum over tokens of `min(p_i, q_i)`.
- **llama.cpp's rate with the same drafter**, as the comparison point it already is for decode, once
  stage 2 has a rate to set beside it.
- **Verify cost:** `T_verify(K + 1)` through the prefill path against `T_decode`, on the 5060 Ti, at
  8K and 64K.

With these three, section 3 predicts the speedup, and the result is recorded either way.

**The drafter's forward -- gated 2026-10-04** (`Tools/Drafting drafter-parity`, then
`Tools/Converters/Gemma/hf_gemma_drafter_reference.py`; files `D:\Claude\drafting\parity`). The 12B Q4_0 with a BF16
cache read 1536 tokens of the first PG-19 test book; the drafter then ran 8 steps from its state, and HuggingFace's
`Gemma4UnifiedAssistantForCausalLM` ran each step on the same inputs -- the embedding row, the hidden state, 1024
sliding and 1536 global keys -- in FP32 and in BF16. The argmax agreed at every step. Mila's mean logit difference
from the FP32 reference was 0.018 to 0.067, against 0.018 to 0.064 for HuggingFace's own BF16 run (0.8x to 1.25x of
it, logits up to 35). The gate written before the run -- the same five highest tokens at every step -- failed at two
steps, on near-ties at fifth place (gaps 0.066 and 0.029, under the per-token difference); it was replaced after the
run by the yardstick above: mean difference within twice HuggingFace's BF16 run's, argmax equal. Recorded as such.
Not yet settled by it: whether HuggingFace's generation hands its drafter 1023 or 1024 sliding keys; on explicit
inputs both attend all they are given.

**Acceptance length -- measured 2026-10-04** (`Tools/Drafting acceptance --positions 512`, RTX 5060 Ti, the 12B
Q4_0; files `D:\Claude\drafting\acceptance_chat_{bf16,fp8}.txt`). Three chat turns in Gemma's template, each
answered greedily for 512 tokens -- none had ended -- with a draft round of 8 from the 12B's state before every
token: continue a novel's opening (1571-token prompt), explain and improve a C++ file (1578), explain a concept (44).
The decoded replies are coherent and do not repeat. A first run on raw-text continuations was discarded: an
instruct model continuing raw text greedily falls into repetition, which any drafter predicts (code read 7.35 of 8).

| Cache | Prompt | E[accepted], K = 1 / 2 / 3 / 4 / 6 / 8 | First draft sampled at 0.8, sum min(p, q) |
|---|---|---|---|
| BF16 | prose | 0.62 / 0.96 / 1.13 / 1.23 / 1.29 / 1.30 | 0.604 |
| BF16 | code | 0.76 / 1.31 / 1.71 / 2.00 / 2.38 / 2.59 | 0.750 |
| BF16 | chat | 0.79 / 1.42 / 1.92 / 2.31 / 2.81 / 3.05 | 0.802 |
| FP8 global | prose | 0.57 / 0.85 / 0.97 / 1.05 / 1.13 / 1.16 | 0.585 |
| FP8 global | code | 0.76 / 1.31 / 1.69 / 1.96 / 2.23 / 2.35 | 0.759 |
| FP8 global | chat | 0.80 / 1.43 / 1.90 / 2.26 / 2.74 / 3.01 | 0.798 |

A draft step costs 2.34 ms against a 21.3 ms decode, **0.110 of a decode** (BF16 drafter; section 3 derived 0.13).
The first draft's sampled acceptance is the greedy one's within 0.02 everywhere, so sampling costs the first draft
nothing. **Mila's FP8 cache does not collapse acceptance**, unlike the q8_0 report in section 1: chat and code are
within 0.05 of BF16 at every K, and prose, whose target reply differs between the two caches, 0.14 lower at K = 8.
Measured at contexts under 2K only; long context is unmeasured.

**Predicted speedup**, section 3, with the drafter at 0.110 of a decode and a verify of K + 1 rows costing v decodes:

| Prompt | v = 1.0 (4.2 b, ideal) | v = 1.3 | Prefill path (4.2 a, v about 2.6 here) |
|---|---|---|---|
| chat | 2.3x at K = 4 to 5 | 1.95x at K = 5 | 1.09x at K = 4 |
| code | 2.1x at K = 4 | 1.7x at K = 4 | under 1x |
| prose | 1.6x at K = 2 to 3 | 1.3x at K = 2 | 0.70x at K = 2 |

**The 26B-A4B and its drafter -- measured 2026-10-04**, the same prompts and tool (`--target 26b`: Q4_0 quantized on
load from `gemma4_26b_a4b_it_qat_bf16.bin`; files `D:\Claude\drafting\acceptance_26b_{fp8,bf16}.txt`,
`parity_26b`). Its drafter's forward passes the same parity gate: argmax equal at all 8 steps, mean difference 0.7x to
1.2x HuggingFace's BF16 run's, reading the 26B's two global key/value heads. Replies coherent, none ended by 512.

| Cache | Prompt | E[accepted], K = 1 / 2 / 3 / 4 / 6 / 8 | First draft, sampled |
|---|---|---|---|
| FP8 global | prose | 0.65 / 0.98 / 1.14 / 1.22 / 1.32 / 1.37 | 0.634 |
| FP8 global | code | 0.78 / 1.38 / 1.83 / 2.16 / 2.54 / 2.70 | 0.771 |
| FP8 global | chat | 0.82 / 1.48 / 2.01 / 2.44 / 3.06 / 3.41 | 0.806 |
| BF16 | prose | 0.64 / 1.01 / 1.21 / 1.31 / 1.43 / 1.49 | 0.640 |
| BF16 | code | 0.76 / 1.32 / 1.73 / 2.03 / 2.39 / 2.51 | 0.749 |
| BF16 | chat | 0.81 / 1.44 / 1.93 / 2.31 / 2.77 / 3.02 | 0.803 |

Acceptance is the 12B's or a little above. A draft step is 2.30 ms against a 10.3 ms decode, **0.223 of a decode** --
twice the 12B's ratio, though under the 0.35 to 0.4 derived in section 3. Its verify reads the union of the rows'
experts, unmeasured; at the derived 1.9 decodes for 5 rows, chat at K = 4 predicts (1 + 2.44) / (1.9 + 4 x 0.223) =
1.23x, and at an ideal 1.0, 1.82x. So the 26B-A4B's answer turns on the expert union, which its multi-row decode
measures first.

**Recorded result: the loop pays through the multi-row decode (4.2 b) and not through the prefill path.** On
replies of the kinds measured, K = 3 to 4 predicts 1.3x to 2.3x by prompt and verify cost; the number is the
measured one once (b) is built, and stage 2 builds it.

**Verify cost through the prefill path -- measured 2026-10-04, RTX 5060 Ti, Gemma 4 12B Q4_0, BF16 cache**
(`Tools/Drafting verify-cost`; median of 20 repeats, 10 at the shallower depths; the decode is the replayed step,
64 in a row; files `D:\Claude\drafting`):

| Depth | Decode | Verify, last-row head | Verify, head per row | Per row / decode |
|---|---|---|---|---|
| 1024 | 18.97 ms | 41.5 ms | 48.5 ms | 2.56 |
| 4096 | 19.18 ms | 44.4 ms | 51.5 ms | 2.69 |
| 8192 | 19.31 ms | 47.6 ms | 54.6 ms | 2.83 |
| 16384 | 19.66 ms | 55.2 ms | 62.4 ms | 3.18 |
| 65536 | 21.53 ms | 99.3 ms | 106.3 ms | 4.94 |

Values at K = 1; every K up to 8 is within 1% of its row, so the cost is all fixed. About 40 ms of it is depth-free --
the prefill GEMMs at two to nine rows, against 19 ms for a decode -- and about 0.9 ms more per 1024 tokens of context,
which is the flash prefill: its grid is one block per 64 query rows per KV head, so a global layer's few rows walk the
whole context on one SM, where decode attention splits it. The head over every row adds 7 ms.

**So arm (a) pays only at short context with high acceptance.** A verify through the prefill path costs 2.6 to 4.9
decodes whatever K is. With the drafter's step at about 0.13 of a decode (section 3, derived), a round at K = 4 costs
about 3.4 decodes at 8K, so it breaks even only if more than 2.4 of 4 drafts are accepted, and at 64K it costs 5.5,
more than the 5 tokens a fully accepted round yields. The loop rests on 4.2 (b), whose cost is near one decode by
construction -- weights read once, decode attention split over keys -- and is measured when it is built. Stage 1
continues with the acceptance length.

### 5.2 Stage 2 -- the loop, greedy

- With the drafter selected, greedy output equals greedy output without it, token for token, over the
  same prompt set and lengths; under 4.2 (b) exactly, by construction.
- A forced partial accept leaves the caches and the next logits equal to a run that never drafted past
  it, including a rewind across a ring wrap.
- The measured speedup set beside section 3's prediction; a gap between them is explained before stage 3.

### 5.3 Stage 3 -- sampling, and the applications

- Seeded runs: given the same uniforms, the sampled rule selects what a reference implementation of the
  rule selects from the same `p` and `q`.
- The planner prices the selection and `PlanEqualsBuild` holds with the drafter built.
- Chat, the inference server and the Python binding select it through a deployment request.

## 6. Decisions

Decided 2026-10-04 (Todd, as recommended):

1. **The verify path** (4.2): (b), a multi-row decode, built only if the loop pays; (a), the prefill path,
   is the measurement's first arm, since it needs no code.
2. **The drafter's format.** BF16 as shipped to start; its head at the 12B's six-bit table format (about
   2.5x fewer head bytes) is measured beside it, and kept if acceptance length holds.
3. **K and its schedule.** Settled by the stage 1 measurement: fixed K against HuggingFace's heuristic
   (raise K after a fully accepted round, lower it after a rejection).

4. **Both Gemma 4 models get their drafter** (Todd, 2026-10-04). The 26B-A4B is paired with its own, against
   Google's account that a mixture of experts gains little at batch 1 -- the community report in section 1
   measured 1.46x on it -- and its gate is the 12B's: measured on the 16 GB card, where it is resident, and in
   the deployment only where it pays. Its verify also routes K + 1 rows, whose experts are the union of each
   row's, so the multi-row decode (4.2 b) reads more of the expert bank than one decode does; that is part of
   what its measurement prices.

## 7. Non-Goals

- Prompt-lookup drafting, EAGLE and Medusa (`SpeculativeDecoding.md`). Google's drafter is trained for
  this model and needs none of their machinery.
- The centroid head: the 12B's drafter ships it switched off.
- Batch above 1.
