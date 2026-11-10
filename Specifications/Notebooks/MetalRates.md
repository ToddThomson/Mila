# Metal: rates next to llama.cpp and MLX, and where a decode step's time goes

Notebook. The M1 Pro's rates for Llama 3.2 3B, Llama 3.1 8B and Gemma 4 12B against llama.cpp and MLX on the same Mac,
measured 2026-11-07 at `0.21.0-dev+127`; then Gemma's decode step timed kernel by kernel, and the two changes that
took Gemma's decode at its own sampling from 3.0 to 15.2 tokens a second. Raw logs: `D:\Claude\metal\bench\logs`.

## 1. Method

Same machine and rules as `MetalFirstModel.md` section 4: the M1 Pro, on AC, nothing else running, one model at a time.
Shapes are llama-bench's: a prompt of 512 tokens (prefill), 128 tokens decoded from a one-token prompt (decode), and
128 decoded after a 4096-token prompt; context 8192; greedy.

- **Mila:** `mila-metal-model-timing --precision fp16`, three loads of five runs, the median of fifteen. The weights'
  own stored format; FP16, which is what the M1 runs (`Metal/MetalBackend.md` section 3).
- **llama.cpp b12138:** `llama-bench -p 512 -n 128 -d 0,4096 -ngl 99 -fa 0,1 -ctk f16 -ctv f16 -r 5`, the faster
  of flash attention off and on per cell, which was on in every cell but one.
- **MLX:** `mlx-lm` 0.32.2 on `mlx` 0.32.4, `python -m mlx_lm benchmark -p 512|4096 -g 128 -n 5`, the mean of five.
  Its "after 4096" is the decode that follows a 4096-token prompt, the same shape as Mila's.

**What is like for like.** Llama 3.2 3B is the same values in all three: BF16 weights at FP16 in Mila, an F16 GGUF, and
MLX's conversion to float16 of the same checkpoint. Llama 3.1 8B in Mila's Q4_0 and llama.cpp's is the same bytes (the
GGUF is Mila's own package). Mila's FP4, MLX's 4-bit (an affine format, groups of 64, 4.2 GB) and Gemma's formats are
each the nearest the engine has. Gemma 4 12B: Mila's Q4_0 package and Google's QAT Q4_0 GGUF have the same body codes,
with a six-bit table in Mila and Q6_K in llama.cpp; MLX's plain 4-bit build is 6.3 GB, and its QAT build stores every MLP
projection at 8 bits, 10.3 GB.

## 2. Rates at `+127`, tokens a second

| Model | Engine | Prefill 512 | Decode | Decode after 4096 |
|---|---|---|---|---|
| Llama 3.2 3B, 16-bit | llama.cpp | 627 | 25.3 | 23.3 |
| | MLX | 584 | 24.3 | 22.3 |
| | Mila | 511 | 19.1 | 17.5 |
| Llama 3.1 8B | llama.cpp, Q4_0 | 256 | 32.5 | 28.0 |
| | MLX, 4-bit | 236 | 36.7 | 32.6 |
| | Mila, Q4_0 | 223 | 21.3 | 18.9 |
| | Mila, FP4 | 221 | 24.2 | 21.0 |
| Gemma 4 12B | llama.cpp, QAT Q4_0 | 158 | 22.0 | 20.0 |
| | MLX, 4-bit | 106 | 19.8 | 19.5 |
| | MLX, QAT 4-bit | 110 | 14.3 | out of memory |
| | Mila, Q4_0 | 142 | 12.2 | 11.1 |

Prefill was 0.82 to 0.90 of llama.cpp's; decode 0.55 to 0.76. MLX's QAT Gemma peaked at 11.7 GB of the 11.8 GB working
set at 512 and failed at 4096. Mila on Llama 3.2 3B was 5 to 8% below its own rates of 2026-11-05 (555, 19.4, 18.65)
while llama.cpp held, filed in `Issues/Untriaged.md`.

## 3. Gemma's decode, kernel by kernel

`LanguageModel::setDispatchProfiling` encodes every dispatch apart with a GPU timestamp at each end
(`MetalExecutionContext`), and `dispatchProfile()` returns each kernel's GPU time; `--profile-dispatches` prints it per
token over one decode of 128. A Metal System Trace of the same decode first showed the GPU busy about 97% of each token,
with about 2 ms idle between command buffers: the host's encoding was not what held it.

At `+127`, about 79.5 ms of GPU time a token with profiling on (75.4 without):

| Kernel | ms a token | Share | Dispatches a token |
|---|---|---|---|
| Q4_0 Linear, own lanes | 50.2 | 63% | 192 |
| RMSNorm | 17.0 | 21% | 337 |
| Six-bit tied head | 6.0 | 7.5% | 1 |
| Everything else | about 6 | 8% | |

## 4. What changed

Each is its own entry in the design of record; here only the rate it moved, one run each, Gemma at its recommended
sampling (temperature 1, top-k 64, top-p 0.95), which is what Chat runs.

| Change | Gemma decode | Decode after 1024 |
|---|---|---|
| At `+127` | 3.0 | |
| The sampler across many threadgroups, its cut by radix selection on the logits' bits | 12.9 | 12.3 |
| RMSNorm across a threadgroup a row (17.0 to 4.3 ms a token) | 15.2 | 14.3 |

The sampler was the whole of Chat's 2 tokens a second: it gave one SIMD-group 85 passes over Gemma's 262,144 logits.
The profile at the end is 66.7 ms a token: Q4_0 50.2, the head 6.1, RMSNorm 4.3, the rest about 6. The Q4_0 product is
what remains, and a faster one is open work (`MetalMatrixPath.md` section 7).

## 5. Prefill after a held prompt, before fused attention

Measured 2026-11-08 at `0.21.0-dev+128`, as the before-number for a fused prefill (`Metal/MetalBackend.md` section 10).
A prompt of 512 tokens added after a held prompt of 0, 4096 or 16384 tokens, llama-bench's `-p 512 -d N`. Mila:
`mila-metal-model-timing --precision fp16 --context 20480 --prefill-depths 0,4096,16384`, one load, one run after one
warm-up per depth, the held prefix reused (the library logged "skipped 4096 of 4608"); FP16. llama.cpp b12138:
`llama-bench -p 512 -n 0 -d 0,4096,16384 -fa 0|1 -ctk f16 -ctv f16 -r 2`, flash attention off and on. Same Mac, on AC;
Activity Monitor was open throughout.

Tokens a second for the 512 added. llama.cpp's faster setting is bold, and the ratio is against it:

| Model | Held | Mila, unfused | llama.cpp, flash off | llama.cpp, flash on | Mila / llama.cpp |
|---|---|---|---|---|---|
| Gemma 4 12B Q4_0 | 0 | 127.5 | **151.5** | 131.3 | 0.84 |
| | 4096 | 115.2 | **137.8** | 92.3 | 0.84 |
| | 16384 | 96.0 | **115.8** | 45.7 | 0.83 |
| Llama 3.1 8B Q4_0 | 0 | 244.3 | 231.3 | **235.6** | 1.04 |
| | 4096 | 195.7 | 184.0 | **189.3** | 1.03 |
| | 16384 | 120.5 | 97.8 | **116.7** | 1.03 |

Mila's prefill of the held prompt itself, from empty: Gemma 116.8 at 4096 and 106.3 at 16384; Llama 228.1 and 137.7.

llama.cpp's Gemma is faster with flash attention off at every depth, by 2.5 times at 16384; its Llama is faster with it
on. Against llama.cpp's faster setting, Mila holds 0.83 to 0.84 on Gemma and 1.03 to 1.04 on Llama at every depth. Both
engines' Gemma rates from empty are below section 2's (Mila 142, llama.cpp 158), which took fifteen samples; not
explained. One run per Mila cell.

**Where Mila's time goes.** Llama 3.1 8B, the 512 added, every dispatch timed by the GPU (`--profile-dispatches`; the
profiled rates, 205.5 and 102.5 tokens a second, are below the unprofiled ones). Q x K^T runs through the Linear product,
so it is inside `linear_tiles`:

| Kernel | After 0, ms | After 16384, ms | Grows by |
|---|---|---|---|
| `linear_tiles` (Linears and Q x K^T) | 1967 | 2717 | 750 |
| `gqa_prefill_softmax` | 25 | 965 | 940 |
| `gqa_value_tiles` (attention x V) | 28 | 927 | 899 |
| Everything else | 185 | 289 | 104 |
| Total | 2205 | 4899 | 2694 |

Attention after 16384 is about 2.6 s of 4.9. The softmax, which only reads and writes the scores, costs more than
Q x K^T: it moves about 35 GB a prompt, 512 rows by 16,896 columns by 32 heads in FP16, read once and written once, in
each of 32 layers, at about 37 GB/s of the M1 Pro's 200 (arithmetic from the shape). Attention x V does the same number
of multiplies as Q x K^T and takes 24% longer.

## 6. The fused prefill's first build, against the unfused pipeline

Measured 2026-11-08, the build of `Metal/MetalBackend.md` 4.5 before any tuning. The Mac was not idle: the Aerials
wallpaper and its video decoder ran throughout, WindowServer at 25 to 47% of a core, so these rates sit about 15%
under section 5's and only the comparison within this section is fair.

**Memory.** The planner's context on the M1 Pro, `--planner-only`, FP16, the same build with the fused prefill
refused and allowed:

| Model | Unfused | Fused | Limited by, fused |
|---|---|---|---|
| Gemma 4 12B Q4_0 | 57,344 | 262,144 | the trained maximum |
| Llama 3.1 8B Q4_0 | 24,576 | 49,152 | a longer context would shrink the prefill chunk |

At a context of 20,480 the plan is 7.71 GiB for Gemma (8.96 unfused) and 8.33 for Llama (10.83).

**Speed.** Llama 3.1 8B, the two builds alternated twice, 512 tokens added, tokens a second:

| Held | Fused | Unfused |
|---|---|---|
| 0 | 208.1, 197.8 | 210.6, 209.4 |
| 4096 | 127.2, 127.8 | 159.3, 157.6 |

Profiled after 4096: `gqa_flash_prefill_half_128` takes 1,104 ms a prompt, where the unfused pipeline's attention takes
about 655 ms (Q x K^T about 178, the extra in `linear_tiles`; the softmax 227; attention x V 230; the permutes about 20).
The fused kernel is 1.7 times slower than the pipeline it replaces. Gemma's rates in the same run, not alternated:
128.9, 88.4 and 59.2 after 0, 4096 and 16,384, against section 5's 127.5, 115.2 and 96.0.

### 6.1 Tuning, on an idle Mac

Each arm built, its fused tests passed (24 of 24), then the fused kernel's GPU time profiled for 512 tokens after 4,096 held,
FP16. The unfused pipeline's attention on the same idle Mac: Llama 3.1 8B about 590 ms a prompt (softmax 215, attention x V
224, Q x K^T about 150 inside the Linear tiles); Gemma 4 12B about 515 ms, by the whole prompts' difference.

| Arm | Llama, head 128 | Gemma, heads 256 + 512 | Kept |
|---|---|---|---|
| 1: 64 keys a tile at head 128 (32 elsewhere) | 861 ms | 646 + 514 ms | yes |
| 2: no mask on a tile every row attends whole, the ring's modulo only once it has wrapped, the row maximum before scaling with the scale in the exponent, the output rescaled only when a row's maximum moved | 626 ms | 444 + 383 ms | yes |
| 2b: P kept in registers through `get_left_input_cooperative_tensor`, which keeps FP32, so P x V multiplies FP32 by FP16 | 1,024 ms | 494 + 389 ms | no |
| 2c: 64 keys a tile at every head size | 625 ms | 394 + 308 ms | yes |
| 3: each key and value tile copied once to threadgroup memory for the whole threadgroup (32 keys at 128, 16 at 256), eight SIMD-groups reading it | 982 ms | 798 + 313 ms | no |
| 4: 32 query rows a SIMD-group at head 128, four SIMD-groups | 1,234 ms | unchanged | no |

With arm 2c the whole Llama prompt took 2,661 ms against the unfused build's 2,649, and Gemma's 3,888 against 3,696.

Rates at depth with arm 2c, one run each, the fused and unfused builds alternated, idle Mac, tokens a second for 512 added:

| Held | Llama fused | Llama unfused | Gemma fused | Gemma unfused |
|---|---|---|---|---|
| 0 | 247.9 | 213.5 | 148.4 | 131.5 |
| 4096 | 154.0 | 165.6 | 109.6 | 120.3 |
| 16384 | 86.6 | 102.2 | 107.9 | 99.9 |

The fused prefill is faster from an empty cache, by 13 to 16%, and slower after a held prompt on Llama, by 7% after 4,096 and
15% after 16,384. Gemma's fused cell at 16,384 equal to its cell at 4,096 is not explained, and is one run.

## 7. Decode at `+130` against MLX's bar

The bar is 0.85 of MLX's rate on the same Mac, model and shape (`Metal/MetalBackend.md` section 6, Todd, 2026-11-09).
Run 2026-11-09, 11:38 to 12:05, on AC, `0.21.0-dev+130`, by `D:\Claude\metal\decode130\run.sh` (log beside it). FP16,
greedy, context 8192, 128 tokens decoded from a one-token prompt and after a 4096-token one. The sample is smaller than
section 2's, the configuration the same: Mila two rounds of two runs, the median of four, every spread under 1%; MLX
`mlx_lm benchmark -n 2`, the mean of two. MLX's decode from empty now follows a one-token prompt, as Mila's does, where
section 2's followed 512. Swap rose from 0.99 to 1.46 GB over the run; MLX's QAT Gemma again ran out of memory at 4096.

| Model | Mila | MLX | Ratio | Mila after 4096 | MLX after 4096 | Ratio |
|---|---|---|---|---|---|---|
| Llama 3.2 3B, 16-bit | 20.88 | 24.56 | **0.85** | 18.88 | 22.23 | **0.85** |
| Llama 3.1 8B, Mila Q4_0 / MLX 4-bit | 23.65 | 37.56 | 0.63 | 20.49 | 32.76 | 0.63 |
| Llama 3.1 8B, Mila FP4 / MLX 4-bit | 27.20 | 37.56 | 0.72 | 23.11 | 32.76 | 0.71 |
| Gemma 4 12B, Mila Q4_0 / MLX 4-bit | 15.14 | 20.65 | 0.73 | 13.31 | 19.53 | 0.68 |
| Gemma 4 12B, Mila Q4_0 / MLX QAT 4-bit | 15.14 | 14.62 | 1.04 | 13.31 | out of memory | |

Since section 2, Mila's 3B decode is 9% faster (19.1 to 20.88) and the drop filed in `Issues/Untriaged.md` is gone from
decode; its prefill is still 512 tokens a second against 555 at `+122`. The 8B is 11% faster on Q4_0 and 12% on FP4, Gemma
24% (12.2 to 15.14), greedy, after section 4's sampler and RMSNorm changes.

**Where a step goes**, `--profile-dispatches`, decode from empty, GPU time a token:

| Model | GPU ms | Largest kernel | Its ms | Share | Next |
|---|---|---|---|---|---|
| Llama 3.2 3B | 49.0 | 16-bit Linear lanes (body and tied head) | 44.4 | 91% | RMSNorm 0.9, split 0.8 |
| Llama 3.1 8B Q4_0 | 42.8 | Q4_0 lanes (body) | 32.1 | 75% | 16-bit head 5.8, RMSNorm 1.2 |
| Llama 3.1 8B FP4 | 37.5 | FP4 lanes (body) | 26.9 | 72% | 16-bit head 5.8, RMSNorm 1.2 |
| Gemma 4 12B Q4_0 | 67.5 | Q4_0 lanes (body) | 50.5 | 75% | six-bit head 6.1, RMSNorm 4.5 |

Attention's decode from empty is 1 to 3% of each. What the weights read at, arithmetic from the stored bytes: the 3B's
16-bit lanes about 145 GB/s; the 8B's Q4_0 body (3.93 GB) about 122 GB/s, its FP4 body (3.71 GB) about 138 GB/s, and its
16-bit head (1.05 GB) about 182 GB/s; Gemma's Q4_0 body about 116 GB/s, as in section 3. The M1 Pro's is 200 GB/s.

**What reaching the bar takes, by arithmetic, not measured.** If the quantized one-row product read at 170 GB/s, near the
182 GB/s the 16-bit lanes reach on the 8B's head:

- Gemma 4 12B: Q4_0 50.5 to about 34.4 ms, about 20.0 tokens a second, 0.97 of MLX. Past the bar.
- Llama 3.1 8B FP4: 26.9 to about 21.8 ms, about 31.5 tokens a second, 0.84.
- Llama 3.1 8B Q4_0: 32.1 to about 23.1 ms, about 30.0 tokens a second, 0.80. Short of the bar by its format, not its
  kernel: Mila's Q4_0 package keeps a 16-bit head of 1.05 GB, and MLX's 4-bit build quantizes the head too, so Mila reads
  about 5.0 GB a token to MLX's about 3.9. llama.cpp reads Mila's own bytes (section 1), and is the like-for-like
  comparison for this row.

So one kernel family, the quantized one-row product, carries every model below the bar. Gemma also loses more than MLX
after 4096 held tokens: its step grows 9.1 ms where MLX's grows 2.8, so decode attention at depth on Gemma's head sizes
is the second gap. Its profile at depth has not been taken; the tool profiles a decode from empty.

### 7.1 After the one-row Q4_0 kernel

Same day, same method, after Q4_0's one row moved to MLX's one-row shape (`Metal/MetalBackend.md` 4.2,
`Notebooks/MetalMatrixPath.md` 7.1); logs `D:\Claude\metal\row\row-rates-2.log`. MLX's figures are section 7's.

| Model | Decode | Ratio | Was | After 4096 | Ratio | Was |
|---|---|---|---|---|---|---|
| Llama 3.1 8B Q4_0 | 30.22 | 0.80 | 0.63 | 25.26 | 0.77 | 0.63 |
| Llama 3.1 8B FP4 | 27.17 | 0.72 | 0.72 | 23.13 | 0.71 | 0.71 |
| Gemma 4 12B Q4_0 | 19.14 | **0.93** | 0.73 | 16.28 | 0.83 | 0.68 |

The Q4_0 product went from 32.1 to 23.5 ms a token on the 8B and from 50.5 to 37.1 ms on Gemma, about 167 GB/s from 122
(arithmetic from the stored bytes). Prefill did not move. Gemma clears the bar from an empty cache and not after 4096;
the 8B's Q4_0 is short of it by its 16-bit head, as section 7 found; FP4 is unchanged.

**Gemma 4 12B against llama.cpp and MLX, the same afternoon** (`D:\Claude\metal\row\gemma-peers.sh`): llama.cpp b12138
on Google's QAT Q4_0 GGUF, flash attention off and on, the faster cell taken, two repetitions; MLX `-p 512 -n 2`.

| | Prefill 512 | Decode | Decode after 4096 |
|---|---|---|---|
| Mila, Q4_0 | 130.9 | 19.14 | 16.28 |
| llama.cpp | 151.4 (fa off) | 21.09 (fa on) | 19.18 (fa on) |
| MLX, 4-bit | 105.7 | 19.88 after 512 | 19.53 |
| MLX, QAT 4-bit | 108.4 | 14.13 after 512 | out of memory |
| Mila against llama.cpp | 0.86 | 0.91 | 0.85 |

Swap stood at 1.4 to 1.6 GB through this run, against 1.0 in section 7's.

### 7.2 Gemma's decode at depth, kernel by kernel

`mila-metal-model-timing --profile-dispatches --profile-depths 0,1024,4096,16384` (the tool's profile now starts at the
first generated token, so a held prompt's prefill is not in it), FP16, context 20480, 127 tokens decoded after each held
prompt, one run; log `~/mila-work/gemma-depth-profile.log` on the Mac. GPU ms a token:

| Held | Total | Q4_0 | Six-bit head | RMSNorm | Partial, sliding (256) | Partial, global (512) | Merges |
|---|---|---|---|---|---|---|---|
| 1 | 53.1 | 36.6 | 6.1 | 4.3 | 0.74 | 0.36 | 1.0 |
| 1024 | 57.4 | 36.5 | 6.2 | 4.4 | 3.55 | 1.66 | 1.1 |
| 4096 | 61.8 | 36.8 | 6.1 | 4.4 | 3.55 | 5.79 | 1.1 |
| 16384 | 76.4 | 35.0 | 5.9 | 4.2 | 3.50 | 23.27 | 1.0 |

Everything but the partial pass is flat. What the partial pass reads, arithmetic from the cache's shape: the 40 sliding
layers' full window, 1024 rows of 8 KV heads of 256 for K and for V in FP16, is 336 MB a token, so about 95 GB/s; the 8
global layers' cache, one KV head of 512 held as K and as V, is 67 MB a token at 4096 and 268 MB at 16384, so about
11.5 GB/s at both. The global layers' grid is 32 blocks of one KV head, 32 threadgroups of 16 SIMD-groups, each SIMD-group
walking its block's keys one after another; their time is linear in the keys held, about 5.7 us a key a block.

**MLX's block count and read-ahead did not move it** (2026-11-09, the change of `Metal/MetalBackend.md` 4.4). The same build
with and without them, alternated A B A B in one session, Gemma decode tokens a second: from empty 19.96 and 19.95
without, 19.71 and 19.71 with; after 4096, 17.15 and 17.15 against 16.95 and 16.94; after 16384, 13.13 and 13.17 against
13.73 and 13.75. Prefill unchanged at 156.7. Earlier runs the same afternoon had shown Mila's prefill 18% lower and its
decode a few percent lower, with more swap: rates from different sessions are not compared.

**What holds the global layers' kernel: its loads, not memory** (Metal System Trace with the Metal GPU Counters
instrument, the last 30 seconds of a profiled decode after 4096, one dispatch per encoder, `~/mila-work/trace` on the
Mac). Counters averaged over one dispatch of each kernel, the samples inside it:

| Kernel | Buffer load utilization | Buffer read limiter | Last level cache limiter | Compute occupancy | ALU limiter |
|---|---|---|---|---|---|
| Decode partial, head 512 (global) | 91% | 92% | 31% | 14% | 17% |
| Decode partial, head 256 (sliding) | 54% | 93% | 14% | 16% | 20% |
| Q4_0 one-row product, for contrast | 12% | 94% | 42% | 30% | 43% |

No register spills in the window. The read limiter is high in all three; what sets the head-512 kernel apart is its load
unit, busy 91% of the time while DRAM is barely read, against 12% for the Q4_0 product, which reads near the chip's
bandwidth. So the kernel issues too many loads for the bytes it brings: each lane loads its 16 channels one value at a
time, and every one of a threadgroup's 16 SIMD-groups loads the same key and value rows. One dispatch a kernel and four
to 34 samples each; the counters' read bandwidth is not taken as an absolute rate.

**Wide loads** (`Metal/MetalBackend.md` 4.4): each lane reads its run of channels 16 bytes a load, two loads at a head of
512 and one at 256. Profiled, GPU ms a token: the global layers' partial pass 5.79 to 1.93 after 4096 and 23.27 to 7.07
after 16384, about 37 GB/s now; the sliding layers' 3.55 to 2.30, about 146 GB/s; the whole step 61.8 to 53.9 and 76.4
to 59.1. Alternated A C A C in one session, tokens a second:

| Gemma 4 12B | Without | With | |
|---|---|---|---|
| Decode from empty | 19.95, 19.96 | 20.04, 20.04 | +0.4% |
| Decode after 4096 | 17.14, 17.16 | 18.79, 18.79 | +9.6% |
| Decode after 16384 | 13.19, 13.19 | 17.13, 17.12 | +30% |
| Prefill 512 | 156.7 | 156.7 | |

After 4096 that is 0.98 of llama.cpp's 19.18 and 0.96 of MLX's 19.53 the same afternoon (section 7.1). The global
layers' pass still reads at about a fifth of the chip's bandwidth: every one of a threadgroup's 16 SIMD-groups still
loads the same rows, which one copy shared through threadgroup memory would remove.

Gate: the 132 attention tests pass under Metal's API validation, the sums in the order they were; 7 of 7 forced failures
are caught (`D:\Claude\metal\attention2\mutatewide.sh`: a pair's halves swapped, the first word reread, FP8's bytes
taken from the wrong word or place, a lane's values read from its keys, a lane's run from channel 0); Gemma 4 12B's
HuggingFace gate on Metal passes in FP16 and BF16 (BF16's window edge 2.36e-2 against 2.54e-2), and the drafter's
greedy check gives 128 of 128.

## 8. What this cannot say

One Mac, one run per change in section 4, and short contexts. Section 2's rates are from before section 4's changes;
section 7 measures decode again at `+130`, not prefill. The BF16 check across Gemma's window edge moved with the RMSNorm
change (`BACKLOG.md`, *Gemma 4 12B in BF16 strays ten times further from HuggingFace*).
