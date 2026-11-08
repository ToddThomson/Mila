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

## 5. What this cannot say

One Mac, one run per change in section 4, and short contexts. Mila's rates in section 2 are from before section 4's
changes; the table has not been measured again. The BF16 check across Gemma's window edge moved with the RMSNorm
change (`BACKLOG.md`, *Gemma 4 12B in BF16 strays ten times further from HuggingFace*).
