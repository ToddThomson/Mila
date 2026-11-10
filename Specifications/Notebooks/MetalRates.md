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

**Prefill, Mila and llama.cpp alternated in one session** (2026-11-09 22:51, `0.21.0-dev+131`, Mila's `--prefill-depths
0,4096` and llama-bench's `-p 512 -d 0,4096`, both flash attention settings, two runs each). Tokens a second for 512 added:

| Held | Mila | llama.cpp, faster cell | Ratio |
|---|---|---|---|
| 0 | 140.2, 140.8 | 157.7, 157.7 (fa off) | 0.89 |
| 4096 | 110.0, 105.7 | 143.1, 143.4 (fa off) | 0.75 |

llama.cpp with flash attention on took 99.2 and 100.4 after 4096. Mila's prefill from empty measured 156.7 by the plain
shapes in other sessions the same evening; the depth mode and that session are not separated.

**Measured again, and that session's Mila rates do not repeat** (2026-11-10 11:46, `0.21.0-dev+131`, one session; log
`D:\Claude\metal\global\peers.log`). Mila at the tool's default context of 8192, as the run above had it, then llama.cpp,
then Mila at 20,480, then at 8192 again; two runs a cell:

| Held | Mila, 8192 | Mila, 20,480 | Mila, 8192 again | llama.cpp, faster cell | Ratio |
|---|---|---|---|---|---|
| 0 | 156.7 | 156.6 | 156.6 | 158.9 (fa off) | 0.99 |
| 4096 | 133.0 | 133.0 | 133.0 | 146.0 (fa off) | 0.91 |

llama.cpp is where it was the night before, 157.7 and 143.2; Mila is not, 140 and 106 to 110 then. The context is not the
cause. Mila's binary here is the one built from `+131` on 2026-11-10 06:47 and used as the baseline of section 7.3's arms;
the binary of the night before was the build directory's of that hour and is not kept, and the Mac's source tree was found
holding an arm's attention files on 2026-11-10, so what that binary held is not known. Section 7.3's profile of the same
night, 3.89 s of GPU time after 4096, is 132 tokens a second, which agrees with today and not with that table. These are
the ratios to use.

**After 16,384 too, and with the global layers unfused** (2026-11-10 12:30, one session; logs
`D:\Claude\metal\global\peers16k.log` and `peers16k_recheck.log`). llama.cpp with flash attention off, two repetitions;
Mila as it is; and the arm of section 7.3 with the global layers unfused, a whole prompt a block. Tokens a second for 512
added:

| Held | llama.cpp | Mila | Ratio | Mila, global layers unfused | Ratio |
|---|---|---|---|---|---|
| 4096 | 145.9 | 133.0 | 0.91 | 136.4 | 0.93 |
| 16384 | 122.0 | 107.8 | 0.88 | 118.8 | 0.97 |

One reading a Mila cell, except Mila after 4096: its one reading was 117.0, and four more of a fresh load were 106.5,
133.0, 133.0 and 133.0, so the cell is the 133.0 that this morning's eight readings also gave. The other three cells
are the kernels' sums of section 7.3 to within 1%: 4762 ms is 107.5, 3785 is 135.3 and 4329 is 118.3. Swap in use was
1.1 GB here against 0.5 GB in the morning, after a planner load at the full context; a slow first reading under memory
pressure is filed in `Issues/Untriaged.md`, and is one more thing the night before's table may have been.

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

**Sharing the rows was built and not kept** (2026-11-09, `0.21.0-dev+131`): the threadgroup loaded four rows of keys and
values at a time into threadgroup memory, a barrier on each side, and every query head read them there. Profiled on Gemma
4 12B, GPU ms a token, wide loads alone against rows shared: the global layers 1.93 against 2.40 after 4096 and 7.07
against 9.25 after 16384; the sliding layers 2.30 against 3.10. Slower on both of Gemma's shapes, so the reads repeated
across query heads are not what holds the global layers at about 37 GB/s; what does is not known.

Gate: the 132 attention tests pass under Metal's API validation, the sums in the order they were; 7 of 7 forced failures
are caught (`D:\Claude\metal\attention2\mutatewide.sh`: a pair's halves swapped, the first word reread, FP8's bytes
taken from the wrong word or place, a lane's values read from its keys, a lane's run from channel 0); Gemma 4 12B's
HuggingFace gate on Metal passes in FP16 and BF16 (BF16's window edge 2.36e-2 against 2.54e-2), and the drafter's
greedy check gives 128 of 128.

### 7.3 Gemma's prefill at depth, kernel by kernel

`mila-metal-model-timing --family gemma --precision fp16 --no-planner --context 20480 --prefill-depths 0,4096,16384
--profile-dispatches` (2026-11-09 23:23, `0.21.0-dev+131`, Gemma 4 12B Q4_0, one profiled prompt of 512 at each depth,
log `D:\Claude\metal\prefill2\profile.log`). GPU ms for the prompt of 512:

| Kernel | After 0 | After 4096 | After 16384 |
|---|---|---|---|
| Matrix product, `linear_tiles_half_128_16_4` | 2929 | 2923 | 2950 |
| Q4_0 expanded to FP16 for it, `linear_quantized_expand_int4_half` | 149 | 149 | 161 |
| Attention, sliding layers (head 256, 40 layers) | 93 | 396 | 402 |
| Attention, global layers (head 512, 8 layers) | 23 | 318 | 1553 |
| Everything else | 105 | 102 | 107 |
| Total | 3299 | 3888 | 5174 |
| Profiled rate, tokens a second | 145.2 | 126.3 | 96.2 |

Everything but attention is flat with depth, so attention is the whole of the fall: 116 ms of the prompt from empty,
714 after 4096, 1955 after 16384. The sliding layers stop growing once the window is full; the global layers grow about
0.10 ms of the prompt for each held token, 12.6 us a held token in each layer.

Arithmetic against section 7.1's llama.cpp rates, taken in another session: 157.7 tokens a second is 3.25 s a prompt from
empty and 143.2 is 3.58 s after 4096, so llama.cpp's prompt grows about 0.33 s from empty to 4096 held where Mila's GPU
time grows 0.59 s. From empty Mila's whole prompt is 3.30 s of GPU time, its matrix product alone 2.93 s and the expansion
of Q4_0 that feeds it 0.15 s. This session's profiled rate after 4096, 126.3, is higher than section 7.1's unprofiled
106 to 110; the two sessions are not separated (section 1).

**The GPU counters, kernel by kernel** (2026-11-09 23:50 to 2026-11-10 02:20, the same run under `xctrace record
--template "Metal System Trace" --instrument "Metal GPU Counters"`, the whole run recorded, counters joined to one
profiled dispatch of each kernel by `D:\Claude\metal\attention2\join2.py`; log `D:\Claude\metal\prefill2\traces_full.log`).
A windowed recording (`--window 30s`) kept only about a quarter of a second of counters, the tool's exit, and none of
the prompt; it suits decode, whose kernels repeat to the last step, and not a prefill. Means over the dispatch, percent:

| Kernel, held | Occupancy | ALU limiter | Buffer read limiter | Last level cache limiter |
|---|---|---|---|---|
| Matrix product, 4096 | 24 | 92 | 67 | 56 |
| Q4_0 expansion, 4096 (2 samples) | 59 | 81 | 4 | 27 |
| Q4_0 expansion, 0 (4 samples) | 59 | 100 | 6 | 29 |
| Attention head 512, 4096 | 11 | 44 | 57 | 51 |
| Attention head 512, 0 | 11 | 42 | 47 | 50 |
| Attention head 256, 4096 | 10 | 47 | 32 | 42 |
| Attention head 256, 0 | 7 | 27 | 20 | 29 |

The matrix product runs at its ALU limit, 89% utilized, with depth or without. Both attention kernels hold about a tenth
of the GPU's threads and no limiter passes 57%: they wait, on too few threads to cover it. The expansion is limited by its
arithmetic, not by memory. The compiler reported register spills in the prompt's command buffers, 48 in the depth-0 run;
the trace does not say which kernel's.

Reading the code against them (inference, not measured): a SIMD-group of the fused kernel holds its whole output row
block in FP32 registers, four 8-by-128 slices at a head of 512 and two 16-by-128 at 256, 4096 values, which is 128
registers a lane before the scores; that is the likeliest cause of the occupancy. The expansion computes its output, chunk
and scale group by an integer division and two remainders a thread for eight weights.

Proposals, none built:

1. **Attention: split a query head's channels across SIMD-groups.** Two (head 256) or four (head 512) SIMD-groups share
   one query head's rows; each multiplies its own 128-channel slices of Q and K, the partial scores are summed through
   threadgroup memory, the softmax is taken once, P is already staged in threadgroup memory, and each SIMD-group adds P x V
   into its own 128 output columns only. No product is repeated; a lane holds 32 output registers in place of 128; the
   cost is a threadgroup barrier and a 2 KB sum each tile. Built as arms first, measured on Gemma's two shapes at 0, 4096
   and 16384 held, the head-128 kernel (Llama) left as it is. Attention is 714 ms of a prompt after 4096 and 1955 after
   16384; what the split would recover is not known until the arms run.
2. **Q4_0 into the tiles directly.** The tiles kernel would read Q4_0 and expand a tile into threadgroup memory before its
   product, in place of a pass that writes the whole block in FP16 and a product that reads it back; the fused attention
   kernel already multiplies a threadgroup tensor. At most the expansion's 149 ms, 4.5% of a prompt from empty; the
   product is at its ALU limit, so its own reads are not what it waits on. A smaller step on the way: the expansion's grid
   in two dimensions and the group as a shift, which removes its divisions.
3. **The matrix product: nothing to propose.** At 89% of its ALU it has no slack this kernel can take. Arithmetic:
   llama.cpp's whole prompt from empty, 3.25 s, is about Mila's product alone, 2.93 s, plus 0.3 s; how llama.cpp spends its
   time is not studied here (CLAUDE.md *Sources*). MLX's Gemma 4-bit prefill in section 2, 106 tokens a second, is below
   Mila's.

**The four-row check could not be built** (2026-11-10, `0.21.0-dev+131`): the head-512 kernel at four query rows a
SIMD-group, to halve its output registers, failed to compile, `matmul2d` asserting "M must be a multiple of 8 or 16"
(`MPPTensorOpsMatMul2dImpl.h`, macOS 27.2 SDK). Eight rows is the fewest a TensorOps product takes, so a lane's output
registers can only fall by splitting the channels (proposal 1). The unchanged kernel, profiled twice in that session:
306.7 and 335.7 ms a prompt after 4096, 1290.9 and 1294.5 after 16384 (section 7.3's run gave 318 and 1553 another
session). Log `D:\Claude\metal\prefill2\rows4.log`.

**The channel split, built and slower** (2026-11-10, `0.21.0-dev+131` plus the arm; log `D:\Claude\metal\prefill2\split.log`).
`gqa_flash_prefill_split` at head 512: four SIMD-groups to a query head, each one 128-channel slice of Q, K, V and the
output, partial scores exchanged through double-buffered threadgroup memory with one threadgroup barrier a tile, the
softmax repeated by each; 4 SIMD-groups (128 threads) a threadgroup, one query head, 8 positions; about 20 KB of
threadgroup memory against today's 8.5 KB per 256 threads. All 128 `GqaMetal*` tests pass on it. Profiled alternated
A S A S in one session, global attention's GPU ms a prompt of 512:

| Held | Today's kernel | Split | Split against today |
|---|---|---|---|
| 0 | 23.3, 23.1 | 30.4, 30.4 | +31% |
| 4096 | 306.1, 322.7 | 389.1, 387.8 | +24% |
| 16384 | 1266.8, 1278.1 | 1705.9, 1701.6 | +34% |

Occupancy fell, 8.1% against today's 11% from empty (one dispatch traced); ALU limiter 36, buffer read 34, last level
cache 43, threadgroup load 4.7. Cutting the output to a quarter did not raise the threads in flight, so the logical
accumulator count is not what sets this kernel's occupancy, or the threadgroup memory the split added now does. Which
is not known: the compiler's register and threadgroup-memory statistics for both kernels decide it before another arm.

**The compiler's statistics** (2026-11-10, Xcode 27.2 replaying a GPU capture of `GqaMetalFused/Fp16.GemmasGlobalLayers`
with profiling, Performance, Shaders; capture recipe in `D:\Claude\metal\prefill2\capture.mm`):

| Kernel | Allocated registers | High register | Spilled |
|---|---|---|---|
| `gqa_flash_prefill_half_512`, today's | 254 | 254 | 1.14 KiB |
| `gqa_flash_prefill_half_512`, split | 254 | 254 | 256 bytes |
| `gqa_decode_partial_half_512` | 104 | 104 | 0 |
| `gqa_decode_merge_half_512` | 80 | 80 | 0 |
| `gqa_write_prefill_half`, `gqa_write_decode_half` | 42 | 42 | 0 |

The head-512 prefill kernel is at the register ceiling and spills, which fits the 384-thread limit and the 11%
occupancy. The split cut its spill by three quarters and left its registers at 254: the output accumulator is not what
fills them. Arithmetic, not measured: a 64-key tile of K or V over one 128-channel slice is 16 KB in FP16, 128 32-bit
registers a lane if a TensorOp holds its operand in registers, so the tile's operands, not the output, would account for
the count. Smaller tiles are not simply the way out: section 6.1's arm 2c took head 512 from 32 keys a tile to 64 and
cut its prompt from 383 to 308 ms, so fewer registers at 32 keys, if it had them, did not buy speed then.

**What sets the limit, compiled only** (2026-11-10; `D:\Claude\metal\prefill2\limits.mm` compiles Mila's kernel source as
`MetalKernelLibrary` does and prints each pipeline's `maxTotalThreadsPerThreadgroup`, which the registers set; 254
registers is 384 threads). Every fused prefill kernel, heads 128, 256 and 512, is at 384. The head-512 kernel with parts
removed (wrong results; registers only): 32 or 16 keys a tile, 384; one output slice with either product, 384; four
output slices and neither product, 384; one output slice and neither product, 704. Two causes, each enough alone: the
four FP32 output slices, and a TensorOp product. One SIMD-group's TensorOp alone, at the fused kernel's shapes:

| Product | Right operand in device memory | In threadgroup memory |
|---|---|---|
| Q x K^T, 8 x 64 x 128 | 832 | 1024 |
| Q x K^T, 8 x 16 x 128 | 1024 | 1024 |
| P x V, 8 x 128 x 64 | **448** | 1024 |
| P x V, 8 x 128 x 16 | 1024 | 1024 |

A TensorOp reading its right operand from device memory holds it in registers, the cost growing with the tile; read from
threadgroup memory it costs none of that. So staging K and V removes the products' share, and at head 512 the output's
four slices still cap the kernel at 384 on their own: the head needs both staging and fewer output slices a SIMD-group.
Heads 128 and 256 hold one and two slices, so staging alone may lift them; that is a prediction to probe.

**Staged tiles, built and timed** (2026-11-10, `0.21.0-dev+131` plus the arm; log `D:\Claude\metal\prefill2\staged.log`).
Every head size reads K and V from a tile copied once to threadgroup memory for the threadgroup. Probed first: at 16 rows
a SIMD-group head 128 stayed at 384 threads; at 8 rows it reached 576 (32 keys) and 640 (16 keys); head 256 stayed at 384
unsplit even at 8 rows and reached 576 split in two; head 512 split in four reached 576 at 16 keys (32 keys needs 39 KB of
threadgroup memory, over the 32 KB allowed). Built: head 128 unsplit, 8 rows, 32 keys; heads 256 and 512 split, 8 rows,
16 keys, 8 SIMD-groups a threadgroup; FP16 576 threads each, BF16 still 384. All 128 `GqaMetal*` tests pass. Profiled
alternated A S A S, GPU ms a prompt of 512:

| Kernel | Held | Today's | Staged | Change |
|---|---|---|---|---|
| Llama 3.1 8B, head 128 | 0 | 53.7, 53.8 | 42.1, 42.3 | -21% |
| | 4096 | 627.5, 627.8 | 544.0, 545.1 | -13% |
| Gemma, head 256 (split) | 0 | 93.1, 93.0 | 86.7, 86.7 | -7% |
| | 4096 | 396.9, 396.9 | 572.4, 572.0 | +44% |
| | 16384 | 396.5, 396.3 | 572.2, 572.0 | +44% |
| Gemma, head 512 (split) | 0 | 23.2, 23.2 | 32.9, 32.8 | +42% |
| | 4096 | 323.2, 320.5 | 489.1, 489.3 | +52% |
| | 16384 | 1450.9, 1475.5 | 2022.6, 2032.7 | +39% |

Llama's whole prompt after 4096: 186 to 192 tokens a second. Staging with one SIMD-group a head wins; the split, at 16
keys a tile and four threadgroup barriers a tile, loses even with its threads lifted from 384 to 576, so more threads in
flight is not by itself the measure.

**Staged without the split, heads 256 and 512** (2026-11-10, log `D:\Claude\metal\prefill2\unsplit.log`): one SIMD-group a
head as head 128's winning kernel, 8 rows, each 128-channel slice of K and of V staged in turn, two threadgroup barriers a
slice; probed at 384 threads, as predicted. All 128 tests pass at 32 and 64 keys. GPU ms a prompt, today's run twice,
the arms once each (the last arm's second run stopped once the result was plain):

| Kernel | Held | Today's | Staged, 32 keys | Staged, 64 keys |
|---|---|---|---|---|
| Head 512 | 0 | 23.8, 23.3 | 61.2, 61.3 | 35.1 |
| | 4096 | 324.7, 304.7 | 870.8, 870.1 | 476.5 |
| | 16384 | 1197.2, 1187.5 | 3313.8, 3307.7 | 1849.3 |
| Head 256 | 4096 | 396.9, 397.3 | 742.5, 741.9 | 620.8 |

Sharing the staged tile does not pay at 384 threads: its barriers, 16 a tile at head 512, cost more than the reads it
saves, more so as tiles shrink. Head 128's win came with its threads lifted to 576. At heads 256 and 512 the FP32
output a SIMD-group holds keeps the kernel at 384 whatever the operands: the split lifts it and pays in its exchange,
staging alone pays in barriers. CUDA's head-512 kernel (`GqaFlashAttention.md` 5.8) also holds 128 FP32 output registers
a warp and keeps FP32 accumulation, declining FP16's rate as a precision trade.

**The matrix product's thread limit** (2026-11-10, the compile-only probe): `linear_tiles_half_128_16_4` is given 832
threads a threadgroup and `gqa_prefill_softmax_half` 1024, against the fused prefill kernels' 384. 832 / 384 is 2.17, and
the counters' occupancy above is 24% against 11%, 2.18: occupancy follows the thread limit. The product's operands and
destination are device tensors and it holds nothing between tiles.

**The global layers unfused, the sliding layers still fused** (2026-11-10 11:31, `0.21.0-dev+131`, one session, today's
binary then a build whose `MetalGqaOp::supportsFlashPrefill` refuses a head of 512 and is otherwise the same; one
profiled prompt of 512 at each depth, context 20480, prefill chunk 1024; log `D:\Claude\metal\global\global.log`). The
unfused pipeline is Phase 3's: the query permuted, Q x K^T through the matrix product, the causal softmax, attention x V,
the output permuted back. GPU ms a prompt:

| Kernel | After 0 | After 4096 | After 16384 |
|---|---|---|---|
| Fused, `gqa_flash_prefill_half_512` | 23.1 | 306.5 | 1189.7 |
| Unfused: Q x K^T, the matrix product's growth over the fused run | 10.5 | 76.4 | 275.5 |
| Unfused: `gqa_value_tiles_half_128_16_4` | 10.4 | 101.3 | 381.4 |
| Unfused: `gqa_prefill_softmax_half` | 3.0 | 27.2 | 99.9 |
| Unfused: both permutes, and the cache write's growth | 8.4 | 8.7 | 8.4 |
| Unfused, all of it | 32.4 | 213.7 | 765.3 |
| Unfused against fused | +40% | -30% | -36% |
| Whole prompt, fused | 3285.4 | 3871.7 | 4762.3 |
| Whole prompt, unfused | 3296.7 | 3790.4 | 4346.4 |
| Profiled rate, fused, tokens a second | 146.8 | 126.6 | 103.4 |
| Profiled rate, unfused | 146.0 | 129.4 | 112.9 |

The controls held: the sliding layers' kernel took 92.5, 395.1 and 394.5 ms in the fused run and 92.8, 397.5 and 396.3
in the unfused, and the matrix product from empty 2923.7 against 2934.3. Arithmetic from the shape, 8 layers of 8192
rows by 16,896 columns by 512 channels: unfused Q x K^T runs at about 4.1 TFLOPS and attention x V at about 3.0, the
softmax at about 0.09 ns a score, and the fused kernel at about 1.9 TFLOPS over both products. The plan is 8.956 GiB
unfused against 7.706: the two score buffers, a chunk of 1024 rows by 16 heads by the 20,480 context in FP16, 1.25 GiB.
One run each; the same kernels in the four arms before it in this session repeated within 1%.

**The profiled prompt against the unprofiled one** (same session, 11:23, today's binary alternated U P U P, two unprofiled
runs an arm). Unprofiled, 512 tokens took 156.6 tokens a second from empty and 132.9 to 133.1 after 4096, which is 3.27 s
and 3.85 s; the profiled GPU time was 3.29 to 3.31 s and 3.87 to 3.88 s, and the profiled wall rate 146.2 to 146.8 and
126.1 to 126.7. So the unprofiled prompt is the kernels' sum to within 1%, and profiling adds about 0.2 s of its own.

**The unfused global layers a block of query positions at a time** (2026-11-10 12:03, `0.21.0-dev+131` plus an arm, one
process; log `D:\Claude\metal\global\arm.log`, the arm's two files in `D:\Claude\metal\global\arm`). The operation's
unfused prefill is called a block at a time, each block a prefill of its own rows at its own offset through the same
kernels; the score buffers stay a whole chunk's, so this times the blocks and not their memory. All 128 `GqaMetal*`
tests pass with the unfused reference whole, in blocks of 8 and in blocks of 24. One held prompt in the cache, then for
each block size a fresh 512 tokens prefilled after it twice, the second profiled; 512 again last. GPU ms a prompt:

| Positions a block | Whole prompt, after 4096 | Global layers | Whole prompt, after 16384 | Global layers |
|---|---|---|---|---|
| Fused (the 11:31 run above) | 3871.7 | 306.5 | 4762.3 | 1189.7 |
| 512 | 3783.1, 3786.2 | 219 (-28%) | 4325.8, 4332.3 | 756 (-36%) |
| 128 | 3822.1 | 257 (-16%) | 4521.0 | 948 (-20%) |
| 32 | 3835.1 | 270 (-12%) | 4580.3 | 1008 (-15%) |
| 8 | 3907.0 | 342 (+12%) | 4811.7 | 1239 (+4%) |

The global layers' column is the whole prompt less everything else in the fused run, 3565.2 and 3572.6 ms (arithmetic
across the two runs of one session; the sliding layers' kernel took 391 to 394 ms in both). After 16,384, by kernel, at
512, 128, 32 and 8 positions: Q x K^T 268, 323, 363 and 382; attention x V 381, 445, 492 and 570; the softmax 100, 113,
138 and 274. A block of 8 positions is 128 rows of the group: one row of tiles for each product, so attention x V is 32
threadgroups a dispatch and the softmax 128, in 512 dispatches of each a prompt where the whole chunk took 8 (arithmetic
from the tile shapes). The small block loses to the kernels' dispatches being small, not to the products' rate.

## 8. What this cannot say

One Mac, one run per change in section 4, and short contexts. Section 2's rates are from before section 4's changes;
section 7 measures decode again at `+130`, not prefill. The BF16 check across Gemma's window edge moved with the RMSNorm
change (`BACKLOG.md`, *Gemma 4 12B in BF16 strays ten times further from HuggingFace*).
