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

## 7. What this cannot say

One Mac, one run per change in section 4, and short contexts. Mila's rates in section 2 are from before section 4's
changes; the table has not been measured again. The BF16 check across Gemma's window edge moved with the RMSNorm
change (`BACKLOG.md`, *Gemma 4 12B in BF16 strays ten times further from HuggingFace*).
