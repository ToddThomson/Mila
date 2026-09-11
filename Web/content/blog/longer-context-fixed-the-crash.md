---
title: "When a Longer Context Fixed the Crash"
date: 2026-08-15
description: "Kicking the tires on Mila's container onboarding turned up a CUDA illegal address that got better when I asked for more memory, not less. The cause was a GEMM plan cache rounding up."
---

Mila is in beta for a reason, and this week a cuBLASLt crash justified the label.

I was doing an unglamorous pass over the Docker onboarding path — the sequence a new user actually walks: pull the image, install a model, ask it something. Nothing about it was supposed to be interesting. Then a run with the tool-calling system prompt died like this:

```
CUDA runtime API error cudaErrorIllegalAddress (700) at
CudaDeviceMemoryResource.ixx:135 in do_deallocate(void*, size_t, size_t):
an illegal memory access was encountered (ptr: 0xc05bff000)
```

An illegal address reported by a *Tensor deallocation*. Whatever actually went wrong happened earlier and asynchronously — and this was not bad luck about where it surfaced. `do_deallocate` calls `cudaCheckLastError` before it frees, so a tensor teardown is where a sticky error gets noticed. The fault had been sitting on the device since some kernel long since retired, waiting for someone to ask.

## The Symptom Made No Sense

The first thing I do with a crash like this is find its edges. Same model, same prompt, only the context length changing:

| model | context | prefill chunk | result |
|---|---|---|---|
| Llama 3.2 3B FP4 | 8192 | 512 | crash |
| Llama 3.2 3B FP4 | 31744 | 512 | crash |
| Llama 3.2 3B FP4 | 32768 | 256 | works |
| Llama 3.2 3B FP4 | 43008 | 256 | works |
| Gemma 4 12B FP4 | 8192 | 1024 | crash |
| Gemma 4 12B FP4 | 56320 | 1024 | crash |
| Gemma 4 12B FP4 | 95232 | 64 | works |

Read the middle two rows again. A context of 31744 crashed and a context of 32768 was fine. **Asking for more memory made the bug go away.** That kills the obvious theory — this is not running out of anything.

The column that does line up is the prefill chunk. Both families slice a long prompt into chunks and size their scratch against a fixed activation budget; as the context grows, the KV cache eats into that budget and the chunk walks down through a ladder of rungs. Every crash was at a large chunk. Every survivor was at a small one.

So it was a shape bug. Except it wasn't only a shape bug, because a roughly 1000-token *user* prompt at chunk 512 worked perfectly, while a roughly 300-token *system* prompt at the same chunk crashed every time. Longer input, no crash. Shorter input, crash.

## Two False Starts

The standard move for an async CUDA fault is `CUDA_LAUNCH_BLOCKING=1`, which serializes launches so the error surfaces at the kernel that caused it. I set it. The error appeared in exactly the same place, at the same deallocation. That follows from the point above and is worth stating as its own rule: launch-blocking changes *when* the failure happens, not *who asks*. The asking still happens where the code asks, which here was the next tensor teardown — exactly as before.

Second false start, and the more embarrassing one: the fault said it landed 1,973 bytes past an allocation of 16,574,540 bytes, and I spent a while trying to work out which buffer has exactly that size. Weights? Scales? Some staging tensor? It was a dead end, and I should have skipped it.

## One Command Named It

```
compute-sanitizer --tool memcheck ./mila-chat --model Llama-3.2-3B-Instruct-fp4 \
    --system-prompt tools --context-length 8192 -p "What is 2+2?"
```

```
Invalid __global__ read of size 16 bytes
    at sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x64x64_...
    Access to 0xc97bcf000 is out of bounds
    Host Frame: cublasLtMatmul
```

A hundred of them, all in one cuBLASLt FP8 GEMM. That single line did what an hour of reasoning had not: it named the kernel, the call, and the fact that this was a *read* off the end of a buffer.

## The Root Cause

For FP4 models, Mila's prefill runs W4A8: the FP4 weights are upcast to FP8 and the BF16 activations are quantized to FP8, both staged into a shared scratch buffer, and then a native FP8 GEMM runs on the tensor cores.

cuBLASLt plans are expensive to build, so they are cached. The cache is keyed on batch size in buckets, and the lookup rounds **up**:

```cpp
// Get the plan for the smallest bucket >= batch_size.
const TPlan& get( int batch_size ) const
```

On Ada with a 512-row prefill chunk the buckets are `1, 16, 32, ..., 128, 256, 512`. Ask for 300 rows and you get the plan built for **512**. That is deliberate and fine: the kernel computes 212 rows of garbage past the real data, and nobody ever reads those outputs.

The staging buffer, however, was sized like this:

```cpp
const size_t activation_fp8_bytes =
    static_cast<size_t>( outer_size ) * cached_in_features_;
```

`outer_size` is 300. The kernel reads 512. So the GEMM walked 212 × 3072 = 651,264 bytes off the end of the activation region, every single time a prompt length was not itself a bucket.

## Why It Hid Behind the Prompt

This is the part I find genuinely instructive.

The scratch buffer is grow-only — it is shared across layers, and a request smaller than the current size is handed the existing allocation unchanged. So the overrun only *faults* when the buffer has no slack left beyond the activation region. When an earlier, larger layer had already grown it, the same out-of-bounds read landed quietly inside the allocation and nothing happened at all.

That is what turned a deterministic shape bug into one that appeared to depend on the prompt:

- A 40-token prompt takes bucket 48 — an eight-row overrun, absorbed by slack.
- A 300-token prompt takes bucket 512 — a 212-row overrun, off the end.
- A 1000-token prompt fills whole 512-row chunks — exact bucket, no overrun at all.
- Gemma at chunk 64 slices the same prompt into 64-row chunks and a small remainder — every chunk either an exact bucket or a few rows over, which is why the *largest* context in the table was the one that worked.

Every "works" in that table was either an exact bucket or an overrun small enough to hide. Nothing about it was random; it just looked that way from the outside.

## The Fix

Size the staging by what the plan will actually read, not by what the caller asked for. The cache now reports the M it rounded to:

```cpp
int bucketFor( int batch_size ) const
{
    return getBucket( buckets_, batch_size );
}
```

and the staging uses it:

```cpp
const int plan_rows = fp8_forward_plan_cache_.bucketFor( outer_size );
const size_t activation_fp8_bytes =
    static_cast<size_t>( plan_rows ) * cached_in_features_;
```

I checked the other four plan lookups in the same file. They hand cuBLASLt real tensors that were allocated at the full prefill chunk extent, and a bucket never exceeds that chunk, so they were always in bounds. This path was the only one staging activations into shared scratch, and so the only one that could be short.

Verification: compute-sanitizer goes from 100 invalid reads to **0 errors**, all three reproducers answer, and the full suite passes at 1606 tests with one pre-existing skip.

## What Found It

It needed nothing new to reproduce — any FP4 model with a tools prompt at a large prefill chunk would have hit it. The git history dates it exactly, and the dating is more interesting than I expected.

The W4A8-FP8 prefill path landed on 12 July, and the short staging buffer was in it from the first commit. It was switched off again the next day, 13 July, because it *generated incoherently* — a numerical bug in the scale handling, spotted within a day because the model started talking nonsense. That got fixed, and the path shipped on for good on 14 July.

Two defects, then, in the same handful of commits. The one that corrupted the output was caught in **one day**. The one that read 651,264 bytes off the end of a buffer survived **32 days**, through a beta and a release cycle, because it corrupted nothing anybody could see: the rows the caller asked for were always computed correctly, and the garbage rows past them were always discarded. It only ever announced itself when the read happened to cross the end of an allocation.

That is the uncomfortable shape of this class of bug. Wrong numbers get caught fast — every test in the suite is looking for them. A read that is out of bounds but harmless is invisible to all of them.

What surfaced it was a change in *defaults*. I had just taught Mila's automatic context sizing to stop at the largest context that still prefills at a full chunk, rather than the largest that merely fits in memory. On a 12 GB card that moves Gemma 4 12B from 95232 down to 56320, and Llama 3.2 3B from 43008 down to 31744. Smaller contexts, deliberately: at 95232 Gemma prefills 64 rows at a time, and at 56320 it prefills 1024 — a chunk sixteen times larger, worth 3.7x on prefill and nothing on decode, bought by giving up context the card could technically have held.

It also moved both models' default straight onto the broken rung. The change that was supposed to make the common path faster made it crash instead, and that is how a month-old bug finally got caught: not by a test, but by walking the new-user path end to end and watching what actually happened.

## Lessons

**Reach for compute-sanitizer first.** On an illegal address it is not a last resort, it is the cheapest step available. It cost one command and named the kernel, the call site, and the direction of the overrun.

**`CUDA_LAUNCH_BLOCKING` changes when the failure happens, not who asks.** A sticky error sits on the device until code asks for it, so a fault will appear to come from wherever your asking happens to live — however synchronous you made the launches. Know where your code asks before you reach for the flag.

**A grow-only buffer turns an out-of-bounds read into an intermittent one.** Slack is what converts a deterministic bug into a flaky one, and flakiness is what makes it look like the wrong variable is to blame. It looked like the prompt. It was the shape.

**A test suite finds wrong answers, not wrong addresses.** Both bugs in this path were shipped in the same week; the one that changed the output lasted a day, and the one that only read out of bounds lasted a month. If a memory error does not move a number, no amount of numerical testing will find it — that job belongs to a sanitizer, and it has to be run on purpose.

**When a cache rounds your request up, size everything downstream by what it returned.** The bucket, not the batch. That is the whole bug in one sentence, and it is a shape of mistake that generalizes well beyond CUDA.

**Kicking the tires works.** Not the tests, not the benchmarks — a run through the same door a new user comes in.
