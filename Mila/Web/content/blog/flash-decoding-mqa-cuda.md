---
title: "Flash-Decoding for MQA in CUDA: One KV Head, Split-K, and an FP4 Matvec at 90% of DRAM Peak"
date: 2026-07-16
description: "Hand-writing a fused flash-decoding kernel for Gemma 4's single-KV-head global layers, plus an FP4 matvec taken from 63% to 90% of DRAM peak. 38.65 to 49.09 tok/s on an RTX 4070."
discussion: "https://github.com/ToddThomson/Mila/discussions/17"
---

*Part 1 — [Hand-Writing Gemma 4 FlashAttention + an FP8 GEMM Bonus](https://github.com/ToddThomson/Mila/discussions/16) —
took ten sessions to drag prefill from 1.95x behind llama.cpp to 1.14x. This part took **one
session** to drag decode from 1.30x behind to **1.03x**: Gemma 4 12B FP4 on the same RTX 4070,
38.65 → 49.09 tok/s at a 32K context, against llama.cpp's measured 50.3–50.7.*

*It didn't take one session because decode is easier. It took one session because Part 1 paid
the tuition — the parity-oracle ladder, the compute-sanitizer discipline, the online-softmax
kernel DNA, and above all the reflex of profiling before believing — and this campaign got to
spend the education.*

## The starting point

An nsys decomposition of one decode token (32K allocated context, early positions) split the
24.9 ms wall like this:

| Bucket | ms/token | share |
|---|---|---|
| FP4 weight matvecs (190 launches) | 15.17 | 61% |
| GQA attention (QK GEMM, softmax, AV GEMM, permutes) | 4.63 | 18.6% |
| RMSNorm (337 launches!) | 2.47 | 10% |
| lm_head over the 262K vocab | 2.16 | 9% |
| everything else + launch gaps | ~1.5 | — |

One experiment turned that table into an indictment: decoding the *same* token positions at
different context **allocations** gave 40.15 tok/s at 4K → 39.82 at 16K → 38.65 at 32K.
Token #100 got slower because we allocated more KV cache. The decode GEMM plans were built
once with `N = cache_capacity`, so the eight global attention layers streamed the entire
allocated cache — 537 MB of mostly-uninitialized VRAM per token at 32K — and let the softmax
mask the garbage afterward.

## The assumption that failed the profiler

Part 1 had "the bug that passed every test." This part had the assumption that failed the
profiler: *"the FP4 matvecs are memory-bound, therefore done."* ncu disagreed:

| Shape | DRAM % of peak | SM % of peak |
|---|---|---|
| gate_up (C=3840) | 77.8 | **82.4** |
| local qkv (C=3840) | 67.2 | **77.0** |
| fc_down (C=15360) | 68.2 | 69.7 |
| o_proj (C=4096) | 63.1 | 67.9 |
| *lm_head FP8 matvec, same run* | **97.2** | — |

SM throughput *above* DRAM throughput on every FP4 shape means the dequant arithmetic was
co-limiting a kernel that should be a pure weight stream — while the FP8 lm_head, one FMA per
byte with no dequant, proved the same GPU streams at 97% in the same token loop.

The diet, one kernel file: **PRMT byte-permute decode** replacing a dynamically-indexed
constant-memory LUT (all eight FP4-E2M1 magnitudes are exact in BF16 — one gotcha: PRMT's
selector bit 3 is its sign-replicate mode, and the FP4 sign bit *is* bit 3, so selectors get
masked and the sign is injected into BF16 bit 15 separately); the **group scale folded** out
of the per-weight math into one FMA per iteration over dual raw accumulator chains; and the
weight/activation/scale loads **software-pipelined one iteration ahead** for the 7–8-iteration
short shapes. Result: 63–78% → **79–90% of DRAM peak**, SM% below DRAM% everywhere, decode
40.15 → 43.24 tok/s. That was the bonus. The headline came next.

## One KV head breaks the textbook

Part 1's "head dim 512 breaks the textbook" has a decode twin. Gemma 4's global layers are
MQA — **one** KV head. A textbook fused decode kernel assigns a block per KV head, which here
means one block and 45 idle SMs. The fix is the Flash-Decoding formulation: split the key
sequence across blocks, each writing an unnormalized `(O, m, l)` partial, with a small fixup
kernel applying the online-softmax merge. Split-K isn't an optimization on this architecture —
it's the difference between using the GPU and not.

The kernel that replaced the five-launch cuBLASLt pipeline (QK GEMM → softmax → AV GEMM plus
two launches that were *byte-identity copies* — for a single token, `[B,1,NH*HS]` and
`[B*NKV,GS,HS]` are the same linear layout, so the Q-permute and output-unpermute did
nothing):

- **Walks absolute positions, not cache slots.** `slot = position % capacity` reads exactly
  the rows the ring softmax reconstructed, and is the identity for the unbounded cache — one
  code path for Gemma's sliding ring, its global layers, and Llama:

| Geometry | KV heads | Q rows/group | head dim |
|---|---|---|---|
| Gemma global (MQA) | **1** | 16 | 512 |
| Gemma local (sliding ring) | 8 | 2 | 256 |
| Llama | 8 | 4 | 128 |

- **Reads only the live band** — `[max(0, len − window), len)` — never the allocation.
- **cp.async double-buffered K/V tiles with tile-granular rescale**: the first version
  updated softmax state per position and serialized each warp on its shuffle reduce; batching
  scores per 8-position tile with one `(m, l)` rescale per tile took the local layers from
  22.7 to 14.5 µs/layer — 5.3x over the old pipeline. Familiar DNA: this is Part 1's flash
  prefill machinery, one token tall.

The GQA decode bucket fell **4.63 → 0.37 ms/token (12.5x)**, and the allocation experiment
now reads 48.87 @4K / 49.09 @32K — the spread didn't close, it inverted into noise.

## The test that catches what parity misses

Part 1 learned the hard way that a parity oracle can pass over undefined behavior. This
campaign institutionalized the lesson — both kernels were validated *before the first project
build* with a standalone nvcc harness that `#include`s the production `.cu`:

- an **FP64 host reference** over the same bf16-rounded inputs, 19 fixtures pinning split
  boundaries, pre-window positions, ring wrap, and batch > 1;
- **NaN poisoning** — every cache slot *outside* the live band is filled with NaN, so a single
  out-of-band read poisons the output and fails the run. The band-limiting claim is proved,
  not asserted;
- the **compute-sanitizer trio** (memcheck, initcheck, racecheck) over the harness — racecheck
  is the one that catches double-buffer barrier mistakes, initcheck the one that catches
  unwritten split partials.

All clean before the build; the in-tree parity oracles (fused vs. cuBLASLt on the same op
instance) and model-level token parity then confirmed on the real path.

## Where it landed

| | 4K alloc | 32K alloc |
|---|---|---|
| Session start | 40.15 tok/s | 38.65 |
| After the FP4 matvec diet | 43.24 | 41.26 |
| After fused flash-decoding | **48.87** | **49.09** |
