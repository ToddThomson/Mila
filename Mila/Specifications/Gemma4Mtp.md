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

Chat samples by default (temperature 0.8, top-k 40), so a greedy-only loop would not reach Chat's default use.
The lossless sampled rule (Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*,
arXiv 2211.17192): accept `d_i` with probability `min(1, p_i(d_i) / q_i(d_i))`, where `p` is the 12B's
distribution and `q` the drafter's; on the first rejection, sample from `normalize(max(0, p_i - q_i))`; if all
K are accepted, sample the bonus from `p_K`.

**Decided 2026-10-05 (Todd): the draft stays greedy, and every verified row is sampled with the caller's own
settings, a draft kept while it is the token the 12B sampled.** With a greedy draft `q_i` is all on `d_i`, so the
rule above keeps `d_i` with probability `p_i(d_i)` -- the chance the 12B's own draw is `d_i` -- and on a rejection
draws from `p_i` without `d_i`, which is what a draw that came out other than `d_i` is. So the two are the same
rule, and sampled output is distributed as it is without the drafter. It needs no drafter logits and no residual
kernel: the device sampler (`TokenSampling.md`) runs at every row with host-drawn uniforms, so a seeded run stays
reproducible. The gate is statistical: the fraction of rounds whose first draft is kept matches the mean of
`p(d_1)` over the same rounds, within its standard error (`Tools/Drafting speculate --sample`).

A drafter that samples its own draft is kept with probability `1 - TV(p, q)`, the sum of `min(p, q)`, where a greedy
one is kept with `p(argmax q)`; neither bounds the other, and sampling-aware drafting is where Xia et al.,
*Acceptance-Aware Draft Model Training for Speculative Decoding* (arXiv 2609.24150), find their gains. **Measured
2026-10-05, it is worth little here:** at temperature 0.8 the greedy draft's first-draft acceptance, the mean of
`p(d_1)`, is chat 0.73 to 0.79, code 0.70 to 0.75, prose 0.55 to 0.62 (4.7, the sampled loop), against stage 1's
`sum min(p, q)` of 0.80, 0.75 and 0.60 (5.1) -- a few hundredths at most, measured on other positions and without
top-k there. So the draft stays greedy; sampling it would need the drafter's logits kept per step and a residual
draw from `max(0, p - q)`.

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

*Design, 2026-10-04; revised 2026-10-05 with Todd: the `Linear` rows are a tensor-core product, not a matvec, and
the entry is `decodeTokens`, not `prefillFrom` routing a short sequence -- the verify is decode's arithmetic, and
decode's operations read the position from the execution context where prefill's take it as a launch argument, which
a replayed step may not.* A verify is a decode of R = K + 1 tokens at positions p to p + K: one forward in which
every row runs decode's arithmetic at its own position, and every weight is read once for all R. Family-neutral by
construction: it is the decode path widened, not a Gemma path.

**The entry.** `LanguageModelNetwork::decodeTokens( tokens [1, R], position )` beside `decode`, returning logits
[1, R, vocab]; it sets the context's decode position to p, as `decode` does, and advances the cached length to
p + R. It is not tied to speculation: any run of known tokens -- a template's fixed tokens after a reply -- can go
through it. Each family implements `onDecodeTokens`, Gemma first; a family without one refuses. `ITransformerBlock`
gains `decodeTokens( input [1, R, model_dim], position )`. The final-norm output keeps all R rows, since the
drafter's next round reads the row at the last accepted position.

**What each operation does with R rows, and what changes:**

| Operation | Today at one row | At R rows |
|---|---|---|
| Norms, residuals, scale, GeGLU, embedding | Row-count blind | Unchanged |
| `Linear`, every weight format (Q4_0, BF16, FP8, FP4, six-bit head, codebook) | One warp per output channel, lanes striding the input, a fixed shuffle tree | A tensor-core product with the R rows as the `mma` n = 8 operand: each weight widened to BF16 once in registers, the group scales applied to the accumulator, split-K across a block's warps and reduced in a fixed order. Its work per weight does not grow with R, so R up to 8 costs one decode's bandwidth (measured below). Not bit-identical to today's matvec -- see "Equal to decode" |
| RoPE | Reads the decode position for its one row | Row r at `*position + r` (one-line change; row 0 is today's) |
| KV write | Writes one row at the decode position | Writes R rows from it (the same one-line change) |
| Decode attention (BF16 and FP8 caches) | One query at the decode position, the band ending there, split-K chosen on the device by band length | The tile's rows are (token, query head) pairs of one KV head, so a KV head's keys stream once for every pair in the tile -- all R tokens of a sliding layer (2 heads a KV head), one token a tile on a global layer (16), the R tiles reading the same keys. Row r masks to its own band, ending at p + r; splits are cut over the tokens' union band, so a token's sums run in another order than its one-token decode's. The scratch is R times one token's. The same kernel at R = 1 is today's decode exactly |
| Expert bank (26B-A4B) | Gathers the token's 8 experts | Groups the R rows' (row, expert) pairs by expert, so each chosen expert's weights are read once for every row that chose it; each row combines its experts in its own routing order, as a one-row decode does. This is the expert union section 3 prices |
| Head (tied six-bit table) | The six-bit matvec | The R-row form of it, as every `Linear` |

**Equal to decode.** A first draft of this design kept each row in the matvec's own order, so that each row was
bit-identical to a one-row decode. That ties the verify to CUDA-core work that grows with R, and it was measured
(below): 1.9 decodes at R = 5. The tensor-core product sums in another order: on the 12B's shapes 0.01 to 0.05% of
its outputs differ from the matvec's, by 1 to 6 BF16 ulps, at the same error against an FP64 reference. Two ways to
keep greedy output with the drafter equal to greedy output without it:

- **(a)** decode itself becomes the R = 1 case of the same kernel, and a verify row equals a decode because it is one.
  Every decode `Linear` of the moved formats changes, and decode's rate must hold. Attention needs the same: its
  multi-token splits are cut over the tokens' union band, so equality there means each token keeping its own split
  boundaries -- its own blocks, reading the keys once per token.
- **(b)** decode keeps its matvecs and only the verify uses the product. Greedy output can then differ at near-ties,
  and the gate is the verify's logits within decode's own numerical noise, not token equality.

(b) is built first and is a prefix of (a): the kernels are the same, and moving to (a) is one dispatch per format,
taken if R = 1 holds decode's rate end to end. The R-row kernels are tested against R one-row decodes, format by
format, before the loop is.

**Measured 2026-10-05** (`Profiling/Microbenchmarks/VerifyRows.cu`; files `D:\Claude\verify_rows`): the 12B's six
Q4_0 Linear shapes, DRAM-resident, in CUDA graphs, weighted per token (40 sliding layers, 8 global, 48 FFN). Each
entry is the Linear time of one token's R rows as a multiple of one decode's:

| Card | Arm | R = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| RTX 5060 Ti | matvec, R rows in today's order | 0.99 | 1.05 | 1.28 | 1.60 | 1.91 | 2.26 | 2.59 | 2.93 |
| RTX 5060 Ti | tensor-core product | 1.01 | 1.01 | 1.01 | 1.01 | 1.01 | 1.01 | 1.02 | 1.02 |
| RTX 4070 | matvec, R rows in today's order | 1.00 | 1.11 | 1.22 | 1.47 | 1.78 | 2.20 | 2.42 | 2.88 |
| RTX 4070 | tensor-core product | 1.04 | 1.02 | 1.04 | 1.05 | 1.06 | 1.06 | 1.06 | 1.02 |

One decode's Q4_0 Linear time is 14.96 ms on the 5060 Ti, 13.49 ms on the 4070 (which also drives a display, so its
rows are noisier). The product runs 384 to 419 GB/s on the 5060 Ti against decode's 384 to 418; its one slower shape
is the FFN down projection (C = 15360), 1.06x decode on the 5060 Ti and 1.11x on the 4070, which is tuning left in
its split. The gate written before the run -- R = 5 within 1.3x of decode -- holds at 1.01x; R = 1 within 5% of
decode, which makes (a) worth pricing end to end, holds on the 5060 Ti. Two CuTe arms were measured against it
(CUTLASS 4.8.0, Todd's question): its canonical sm80 structure at 1.32x, occupancy-bound, and the product's own
structure written in CuTe at 1.41x, losing on delivering 4-bit operands (byte loads where the product loads words);
the product stays hand-written (`Profiling/Microbenchmarks/README.md`).

**Attention at R tokens -- measured 2026-10-05** (`CudaGqaDecodeRate.DISABLED_GemmaSeveralTokens`, RTX 5060 Ti, one
call DRAM-resident and replayed; correctness in `CudaGqaOp.DecodeTokens.Cuda.cpp`, which a token taking another
token's causal end fails). One call of R tokens as a multiple of one token's:

| Layer of the 12B | Depth | 1 token | 2 | 4 | 5 | 8 |
|---|---|---|---|---|---|---|
| sliding (2 heads a KV head) | 8192 | 28.8 us | 1.04 | 1.09 | 1.15 | 1.23 |
| global (16 heads a KV head) | 1024 | 15.0 us | 1.05 | 1.39 | 2.21 | 2.46 |
| global | 8192 | 51.0 us | 1.12 | 1.97 | 2.79 | 3.80 |
| global | 32768 | 174 us | 1.13 | 2.18 | 2.80 | 4.30 |

A sliding layer's tokens share one tile and read its keys once. A global layer's 16 heads fill a tile per token, so R
tokens are R times the blocks over the same splits; their keys come from L2 (the row tile is the fastest grid axis),
but each block's walk is latency-bound, so the extra tiles cost waves. Per token of the 12B at R = 5 that is about
0.17 ms on the sliding layers and 0.7 ms on the global layers at 8K (2.5 ms at 32K), against a 19 to 21 ms decode.
The remedy, not built: a block that stages each key tile once for all its row tiles, a warp group per tile. The one-
token kernel is a separate instantiation and runs at its rate before this change (the per-row band arithmetic cost
0.4 us a call on short bands when it was shared). Not yet measured: attention at R queries,
the six-bit head, and the 26B-A4B's expert union.

**Replay.** The verify is a recording of its own (`DecodeGraph.md`), keyed by R: its token buffer and the context's
decode position are device memory, so one recording replays every round at a fixed K.

**Order of work**, each step gated before the next: (1) the R-row Q4_0 product and a microbenchmark of R = 1 to 8
against one row -- the first number that prices (b), with no loop built (done, above); (2) every other
`Linear` format and the head (done 2026-10-05: `Linear::decode`, `Kernels/DecodeRows`, all seven formats tested
against one-row decodes in `Linear.Decode.Cuda.cpp`; in the library the 12B's gate and up projection costs 1.01 to
1.03x one row at R = 2 to 8); (3) RoPE, KV write and attention at R rows (done 2026-10-05, measured below); (4)
`decodeTokens` through Gemma's blocks, gated equal to R decodes, with the network reserving R tokens' attention
scratch -- a recorded decode step must not see the buffer it captured grow (done 2026-10-05, below); (5) the expert
bank's union and its measured cost on the 26B-A4B; (6) the loop, greedy, then sampled (4.4).

**`decodeTokens` -- built 2026-10-05.** `GemmaConfig::withDecodeTokens( n )`, 1 to 8 and 1 by default, is the run
capacity a deployment selecting the drafter raises to K + 1; it reaches the build as `BuildContext::withDecodeTokens`
and sizes the head's logit rows, attention's decode output and the decode split scratch, so it is priced and reserved
with everything else, and an attention op refuses a decode call of more tokens than it was built for. The routed
feed-forward (26B-A4B) refuses until step 5; Llama and Qwen refuse (a family without one). `decodeTokens` is called,
never replayed: a decode recording holds one token's launches, and the verify's own recording is the loop's (step 6).
Gate (`Tests/Dnn/Models/DecodeTokens.Cuda.cpp`, the seeded tiny Gemma, BF16): five tokens in one call against five
decodes differ by 5.0e-2 of the logits' RMS on a 150-token prompt, against 3.8e-2 for a prefill of the same tokens --
the network carries any summation-order change that far -- and the next decode after the call is bit-identical, so
the caches it leaves are exact. The budget is twice the prefill yardstick. The test runs a 12-token prompt, where the
call is bit-identical to the decodes (prefill 6.3e-2) and a token attending past itself measured 2.3 times the RMS; at
150 the diffuse attention of random weights put that same error at only 9.1e-2. A decode replayed across a `decodeTokens` call stays bit-identical to the
same steps called. For the loop: after `decodeTokens` the final-normed hidden state holds all T rows, and a replayed
decode updates no host state, so the drafter reads the row it needs by position rather than trusting the last
call's shape.

**The greedy loop -- measured 2026-10-05** (`Tools/Drafting speculate`, RTX 5060 Ti, the 12B Q4_0, BF16 cache, the
acceptance prompts, 512 tokens after the prompt's first, median of 3 runs; files `D:\Claude\drafting\speculate`).
Each token is chosen on the device into the slot the next pass reads (`TokenSampler::enqueueSampleOnDevice`), so a
round waits on the host once: K draft steps, one `decodeTokens` of K + 1, an argmax per row, then the host keeps the
agreed drafts and the target's next token, rewinds past the rest, and the drafter's next round reads the row of the last
token kept. Plain decoding is the replayed decode with the same device argmax.

| Prompt | K | E[accepted] | ms/token, plain / drafted | Speedup | Predicted (section 3) | Same tokens as plain |
|---|---|---|---|---|---|---|
| prose | 2 | 0.94 | 19.07 / 14.29 | 1.33x | 1.35x | first 23 |
| prose | 4 | 1.16 | 19.07 / 15.30 | 1.25x | 1.27x | first 23 |
| code | 3 | 1.55 | 19.07 / 11.83 | 1.61x | 1.62x | first 310 |
| code | 4 | 1.80 | 19.07 / 11.77 | 1.62x | 1.64x | all 513 |
| chat | 3 | 1.83 | 18.46 / 10.35 | 1.78x | 1.80x | first 402 |
| chat | 4 | 2.12 | 18.46 / 10.26 | 1.80x | 1.82x | first 402 |

The prediction uses the run's own acceptance and the step costs measured at the prompt's depth, and holds within 1 to
3% in every cell (K = 2 to 5): a round costs 0.3 to 1 ms more than its priced parts, the host's wait and the argmaxes.
Where a drafted run parts from plain, the plain run's top two logits differ by 0, 0.125 or 0.25 -- one BF16 step at
logits of 16 to 64 -- so every parting is a tie the two arithmetics round differently (5.2's first gate). On the same
point, over six tokens at the prompt's depth the verify's logits lie from decode's by 1.3e-1 (code, chat) to 1.7e-1
(prose) of their RMS, and the prefill path's by 3.2e-1, 3.0e-1 and 1.6e-1 -- the verify above that third arithmetic
only on prose, by 7% -- no argmax differing in either. The cache gate is `Gemma_RejectedTokensLeaveNoTraceAfterTheRewind`
(`DecodeTokens.Cuda.cpp`): verifies that differ only in rejected tokens leave the kept tokens' logits and every later
pass bit-identical, across the ring's wrap.

Acceptance in the loop is below stage 1's at the same K (chat K = 4: 2.12 against 2.31; code 1.80 against 2.00).
Stage 1 drafted from every position of a plain run; the loop drafts only from the positions it lands on. Not yet
explained further.

**The verify costs 1.19 to 1.23 decodes, not 1.0**, flat from 2 to 6 rows: it is called while the decode beside it is
replayed. A called decode costs 1.12 of a replayed one (21.4 ms against 19.0 at 1.5K), so about 2.4 ms of the verify's
3.6 ms over a decode is launch gaps, and its rows cost 1.07 called decodes. The draft step is called too. Recording the
verify and the draft steps (4.6, 4.7 "Replay") is the next term; with the verify at 1.07 decodes and the draft step
unchanged, section 3 gives chat 2.0x, code 1.8x, prose 1.5x (derived, not measured).

**The loop in the library -- built 2026-10-05 (6b-1).** A deployment selects the drafter with
`DeploymentRequest::withSpeculativeDecode( draft_model, draft_tokens )`, K from 1 to 7 and required: the best K
differs by kind of text (4 for chat and code above, 2 for prose), so a use case supplies it, not the library. The
plan carries the selection to the load. `GemmaModel` builds the network with K + 1 decode tokens and
`GemmaTransformer::addDrafter` attaches the drafter as a child (`<network>.drafter`, on the network's context), so
the footprint prices it with no planner change; a tiny network with a drafter prices exactly what it builds,
category by category (`Tests/Dnn/Models/Gemma/Gemma.DraftModel.Cuda.cpp`). The round is
`LanguageModelNetwork::draftTokens( tokens, position, hidden_row )` -- slot 0 known, slots 1 to K drafted on the
device, the hidden state taken from the row of the last pass the caller names -- then `decodeTokens`, then the
sampler at every row (4.4); `GemmaModel::generate` runs it whenever the plan selected a drafter. The draft's argmax
op is created at the first draft, as the model's own sampler is at its first token, so its vocabulary-wide scratch
(about 1 MB) is outside the footprint like the sampler's. Llama and Qwen refuse the request; so does the 26B-A4B
until its expert bank decodes several tokens (step 5). Greedy generation of 256 tokens through `GemmaModel` with the
drafter equals it without (the same test file, the 12B). Through `GemmaModel::generate` (`Tools/Drafting generate`,
512 tokens, median of 3, context 3072): chat 1.81x at K = 5, code 1.70x at K = 4, prose 1.39x at K = 2 -- the tool's
loop's rates within the spread the replies' own partings cause. Chat parts from plain at token 16 for K = 2 to 4
but not at K = 5, so at that token the drafted runs differ among themselves: a near-tie the orders round
differently, the plain model here being another build (one decode token) than the tool's shared network.

**The sampled loop -- measured 2026-10-05** (4.4's rule; `Tools/Drafting speculate --sample` and `generate --sample`,
Chat's sampling: temperature 0.8, top-k 40; the same prompts, 512 tokens, median of 3; files
`D:\Claude\drafting\speculate\*_sampled.txt`). The rule's gate holds in all twelve cells (three prompts, K = 2 to
5): the fraction of rounds whose first draft is kept matches the mean of `p(d_1)` within two standard errors, most
within one (largest gaps: prose K = 5, 0.596 against 0.617 +- 0.014; chat K = 5, 0.790 against 0.763 +- 0.017).
Through `GemmaModel::generate`:

| Prompt | Best K | Sampled speedup | Greedy speedup (above) |
|---|---|---|---|
| chat | 3 to 5 | 1.65x | 1.81x |
| code | 4 | 1.58x | 1.70x |
| prose | 2 to 3 | 1.31x | 1.39x |

The tool's loop predicts its own sampled rounds within 1 to 10% (section 3, each run's acceptance); the wider
cells are where a run's acceptance moves between runs, since each run samples another reply.

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

- With the drafter selected, the verify's logits are within decode's own numerical noise over the same
  prompt set and lengths, and greedy output differs from greedy output without it only at near-ties; token
  for token exactly if decode moves to the same kernels (4.7, option (a)).
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
