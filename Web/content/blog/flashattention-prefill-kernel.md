---
title: "Writing a FlashAttention Prefill Kernel for Gemma 4: Head Dim 512, Tensor Cores and an FP8 GEMM"
date: 2026-07-15
description: "Hand-writing a tensor-core flash-attention prefill kernel for Gemma 4's 512-wide heads, plus an FP8 activation GEMM. Ten sessions from 1.95x behind llama.cpp to 1.14x."
discussion: "https://github.com/ToddThomson/Mila/discussions/16"
---

Adding FlashAttention to Mila's GQA started as a **memory** project. It was the last big lever in optimizing memory utilization across the Gemma 4 pipeline — with weight tying, FP8 KV, and bounded sliding-window KV already in place, the attention path was the last thing still materializing large intermediates. A classical attention writes a `[chunk × attended_len]` score matrix to global memory, and its workspace has to be sized to the *full* context width; at 64K that's ~1 GB per buffer of pure throwaway. FlashAttention's online softmax removes it entirely, and that reclaim — **~1-2 GB at long context** — is what took Gemma 4 12B from a VRAM cliff at 32K to **running 64K on a 12 GB RTX 4070**. Speed was the second prize, and it's the one that turned into a ten-session campaign.

Ten sessions ago, Mila's Gemma 4 12B prefill on an RTX 4070 was ~1.95x slower than llama.cpp at 48K context. As of `0.20.0-alpha.6+104` it's **1.14x** — near parity on the same hardware, same model, same context. This is the log of how we got there: a hand-written FlashAttention kernel, an FP8 GEMM, and a trail of dead ends that taught us more than the wins did.

No CUTLASS. No CuTe. Just CUDA C++, raw `mma.sync` PTX, and `ncu`. Mila is a craft project — the point was to *understand* every cycle, not to link a library that already does.

## The starting point

Gemma 4 12B, FP4 weights, 48K context (22,496 prompt tokens), RTX 4070 (Ada, sm_89):

| | wall @48K | tok/s | vs llama.cpp |
|---|---|---|---|
| llama.cpp (reference) | 10,903 ms | 2,063 | 1.00x |
| Mila, start of campaign | ~21,300 ms | ~1,056 | **1.95x** |

Profiling the baseline told us *where* the time went, and it wasn't where we assumed. At 8K the linear/FFN GEMMs dominated (68%), but at 32K the **softmax score-matrix materialization** became the single biggest kernel — a clean O(S²) that grew 15x for a 4x context bump. That's the long-context cliff, and it's exactly what FlashAttention's online softmax removes: it never writes the `[chunk × attended_len]` score matrix to global memory at all.

## How the memory actually got reclaimed

The mechanism is worth spelling out, because the reclaim wasn't automatic. Gemma 4 interleaves 8 global layers with 40 window-bounded sliding layers, but the attention score workspace was a *single shared buffer* sized to the widest consumer — the global layer at full context width. Once the global layers moved onto flash and stopped touching that buffer, it could be resized down to what only the sliding layers need, which is window-bounded and tiny by comparison.

Before: ~300 MiB free at 32K — a VRAM cliff. After: 64K context fits, with the score buffer down from a full-context ~1 GB to the sliding-window need. The savings scale with context, which is exactly the regime that was breaking, so flash is as much a *fit* enabler as a speed one.

One insight unlocked the whole effort: **attention has no weights.** Q/K/V are activations, and Mila's KV cache is BF16. FP4 is weight-only (linears/FFN). So flash attention is BF16×BF16 tensor-core work — *identical* to what llama.cpp runs over its FP16 KV cache. There was no structural barrier to matching it; our gap was pure unfinished optimization.

## The kernel, iteration by iteration

Every step was gated on a parity oracle (flash-on vs cuBLASLt, atol 3e-2) and `ncu`. Per-instance kernel time, 8K global-layer attention:

| Stage | What changed | ms/inst | Verdict |
|---|---|---|---|
| Iter 1 | Naive scalar, warp-per-row, no shared mem | 127 | **74% regression** |
| Iter 2 | Shared-memory K/V tiling (the "real" flash algo) | 103 | Non-fix — falsified our own premise |
| 2a | WMMA tensor cores, HS-split across warps | 47.5 | Real step, still 3.5x off |
| 2b | Register-resident output via raw `mma.sync` PTX | 26.9 | Half the gap gone |
| 2c | `cp.async` double-buffered software pipeline | 22.8 | **The win — beats cuBLASLt at 32K** |
| 2d | QK→`mma.sync`, distributed softmax, `ldmatrix.x4` | 19.1 | Fastest global kernel |

The two "failures" at the top were the most valuable measurements in the project.

**Iter 1** was the naive thing everyone writes first: one warp per query row, stream all of K/V from global memory. Profiling before committing caught a 74% regression — a ~100x memory trap. Good; that's what the profiler is for.

**Iter 2** is the one that mattered. We added proper shared-memory K/V tiling — the textbook flash algorithm — and it cut global K/V traffic ~16x. It bought **1.2x.** That one fact falsified our entire mental model. `ncu` corrected us: the kernel was never DRAM-bound (L2 hit 99.7%). The scalar warp-per-row formulation issues **one shared-memory load per FMA** — a 1-FLOP-per-2-byte ratio that saturates the load/store unit's *instruction issue rate*, not bandwidth. Tiling moved the loads from L2 to shared without cutting their *count*. The entire scalar family was structurally dead, and no amount of tiling, occupancy, or spill-fixing could escape the ceiling. The only way out was tensor cores.

## Head dim 512 breaks the textbook

Standard FlashAttention-2 splits the query rows across warps and keeps the output tile `O[Br × HS]` in registers. At Gemma's global head dim of **512**, that accumulator is 256 floats per thread — it won't fit; it spills unrecoverably. So we did something bespoke: **split the head dimension across warps** instead of the rows. Warp *w* owns output columns `[w·64, (w+1)·64)`, the QK becomes a split-K contraction with a tiny shared-memory reduction, and PV writes disjoint column slices with no cross-warp sync.

That forced us down to raw `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32` PTX, because the online-softmax **per-row rescale on a tensor-core accumulator** cannot be expressed through the `wmma` convenience API — it hides the fragment→(row,col) thread mapping. The `mma.sync` layout is documented, so each thread knows which two rows its accumulators hold and applies the right `alpha` before each accumulate. This is the CUTLASS/FA-2 way, hand-written.

## The bug that passed every test

Two lessons from this project are burned in permanently.

**The single-warp softmax was the critical path all along.** When we converted QK to `mma.sync` in Stage 2d, the first attempt *regressed 57%*. The cause: we'd fused an 8-warp reduction into a 16-lane softmax section that ran on half of one warp while 7.5 warps idled. Every earlier flat experiment — padding, more warps, fewer barriers — had been flat because they were all attacking secondary costs. The serial softmax was the real ceiling. Distributing it across warps (each owns 2 rows, lane-groups sum partials in parallel, `__shfl_xor` for row max/sum) was the actual fix: 22.8 → 20.6 ms.

**A green oracle is not a correct kernel.** When we added `ldmatrix.x4`, the parity oracle passed, chat was coherent — and `ProfileModel` crashed with `cudaErrorIllegalAddress` at scale. `compute-sanitizer` found it: the `ldmatrix` PTX was missing the `.shared` qualifier, so a shared-memory offset was being dereferenced as a *generic* address. At small smem offsets that bogus address happened to alias the correct data, so the oracle stayed green over undefined behavior. The fix was `cvta.to.shared.u64` **and** `.shared` on the instruction — the canonical CUTLASS pattern, for exactly this reason. New rule: **every PTX-level change now gates on `compute-sanitizer memcheck`, not just parity.**

## The strategic pivot: flash wasn't the whole gap

Here's the honest part. Partway through, a hard `ncu` session forced a reckoning: even a *perfect* flash kernel does the same tensor-core QK/AV compute as cuBLASLt — it only removes the score materialization. That's a real prize at long context, but it was never going to be the whole 4-5x. The rest of the gap lived in the **quantized matmul**: our FP4 weights were dequantized to BF16 and run through a cuBLASLt BF16 GEMM, while llama.cpp runs integer/FP8 tensor cores directly.

So we built the other lever: **W4A8-FP8 prefill** — native FP8×FP8 tensor cores on the linear GEMMs. It was the single biggest jump of the campaign (1,372 → 1,763 tok/s), and it came with its own hard-won lesson. The first cut passed the per-layer oracle at 5e-2 tolerance and generated *garbage* over 48 layers. The oracle measured one layer; the error compounded across all of them. Root cause was a stale, uninitialized scale factor (`sB`) read before the quantizer had filled it — invisible to a numerics oracle, deterministic under `compute-sanitizer --tool initcheck`. **Validate the generation, not just the oracle.**

## Where it landed

The full campaign scoreboard @48K:

| Version | Lever | tok/s | vs llama.cpp |
|---|---|---|---|
| start | — | ~1,056 | 1.95x |
| +100 | Stage 2d: distributed-softmax mma + ldmatrix | 1,205 | 1.71x |
| +101 | Bounded-ring flash on the 40 local sliding layers | 1,372 | 1.50x |
| +103 | W4A8-FP8 prefill GEMM (FP8 tensor cores) | 1,763 | 1.17x |
| +104 | Row-split FA-2 kernel for the local layers | 1,817 | **1.14x** |

Two flash kernels ship: an **HS-split** kernel for the 8 global layers (head dim 512, at its Ada floor — the accumulator and the K/V double-buffer slam *both* occupancy doors at once, a direct consequence of the head dim, not a tuning gap), and a **row-split FA-2** kernel for the 40 local sliding layers (head dim 256, which *does* fit a register-resident output and gets 2 blocks/SM). Same idea, two geometries, because the hardware constraint is different at each head dim.

## External validation

Midway through, we found an excellent blog post by Thien Tran [Writing Speed-Of-Light FlashAttention for 5090 using C++](https://gau-nernst.github.io/fa-5090/) — a hand-rolled flash kernel, no CUTLASS, one hardware generation ahead of us. Same trajectory, same `mma.sync.m16n8k16` choice, and their final hand-written kernel hits 94% of peak and *beats* the official FlashAttention library. It was a good gut-check that the approach — understand the metal, write the PTX yourself — actually reaches the frontier.

## What we'd tell the next person

- **Profile before you commit.** Two of our biggest "insights" were regressions the profiler caught. A naive kernel is not a valid baseline.
- **The profiler will falsify your mental model.** We were sure we were memory-bound. We were instruction-issue-bound, then latency-bound, then occupancy-bound — a different regime at every stage, and only `ncu` could tell us which.
- **A passing oracle can hide UB and can hide compounding error.** Add `compute-sanitizer` (`memcheck` for PTX, `initcheck` for uninitialized reads) and validate real generation, not just a per-layer tolerance.
- **Know what your lever can and can't buy.** Flash removes score materialization; it was never the whole gap. The quantized GEMM was the other half. Measuring that early saved us from over-investing in one kernel.
- **The textbook assumes head dim ≤ 128.** At 512 you write a different kernel. The 128-dim playbook does not transfer verbatim.

This was never about beating llama.cpp — Mila is a mastery project, not a production runtime. But getting from 1.95x to 1.14x by hand, and understanding *exactly* why every cycle went where it did, was the whole point.

