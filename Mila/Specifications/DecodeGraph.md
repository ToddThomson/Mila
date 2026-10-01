# Decode Graph

Specification for replaying a decode step as one recorded CUDA graph: what the recording is, what every
decode op must honour for a recording to stay correct, and the guardrails that keep a violation from
producing silent wrong output.

Written 2026-09-29, before implementation, on the measurements in `ModelFamilyParity.md` 8.4 L5. Agreed
with Todd the same day, including the split-count design (6.2, option C) and the guardrails (section 5).

---

## 1. Problem

A decode step launches one kernel per operation: 451 for Llama 3.1 8B, 964 for Gemma 4 12B. Between two
consecutive kernels the GPU is idle for 2-3 us while it retires one launch and starts the next, even when
the host has already queued both. After the host left the token loop (decode-ahead in Gemma and Qwen, L5
for Llama) that gap is the only idle time left, and it is measured, not inferred
(`Tests/Dnn/Models/DecodeLaunch.Cuda.cpp`, one decode step at a fixed position, launched and then replayed
as one captured graph):

| One decode step | RTX 5060 Ti launched -> graph | RTX 4070 launched -> graph |
|---|---|---|
| Llama 3.1 8B Q4_0 | 13.28 -> 12.33 ms (75.3 -> 81.1 tokens/s) | 12.60 -> 11.49 ms (79.4 -> 87.1) |
| Gemma 4 12B Q4_0 | 20.77 -> 18.45 ms (48.2 -> 54.2) | 20.23 -> 17.32 ms (49.4 -> 57.7) |

The replay equals the kernels' own time: a recording removes the whole gap. Fusing kernels removes it only
for the launches fusion deletes (about 200 of Llama's 451), and does so by merging components the reader
is meant to see separately.

Capture needed no change for that measurement because it replays one position. Generation moves the
position every token, and four things in today's decode path depend on it on the host (section 3).

---

## 2. What The Recording Is

**The call structure is the specification of a decode step; the recording is a cache of the launches
it produces.** `LlamaBlock::decode`, `GemmaBlock::decode` and the Qwen blocks stay the code a reader reads
to learn what a step does. The ops and kernels are the same ones in both modes -- there is no second
implementation of anything. The recording is made by running that code once under stream capture, and it
is replayed only while nothing it recorded can have changed.

The cost is to a developer stepping through decode: after the second step a breakpoint in a block's
`decode` is not hit and host-side output from inside the step stops, because the step is replayed rather
than called. Turning replay off (5.2) restores the called path exactly; observing a network (5.4) does so
automatically.

---

## 3. What Depends On The Position Today

Traced through all three families' decode paths (Llama; Gemma dense, sliding and MoE; Qwen attention,
DeltaNet and convolution layers):

| Per-token dependency | Where | Effect on a recording |
|---|---|---|
| `position` as a launch argument | RoPE decode (BF16, FP32); KV-cache write (BF16, FP8) | frozen at the recorded value |
| `actual_len = position + 1` as a launch argument | fused decode attention (BF16, FP8 cache); the cuBLASLt fallback's decode softmax | frozen |
| Split count chosen on the host from the band length | `launchDecodeAttention` (`Gqa.Decode.Bf16.cu`) | changes the grid, and whether the fix-up runs, until the band saturates |
| `cached_seq_len_` advanced on the host | `CudaGqaOp` decode | not advanced by a replay; read by `rewindKvCache` (Gemma's prompt-prefix reuse) |

Not dependent, checked: RoPE calculates its angles from the device position and holds no table; network scratch is reserved
at build and a larger request throws rather than moving it (`CudaExecutionContext::getDeviceScratchBuffer`);
the one lazy allocation (`CudaLinearOp`'s FP8 unit scale) happens on the first called step; DeltaNet and
convolution state update in place with no pointer swap; host flags set during decode are set-once
(`state_primed_`, `decode_active_`); no op reads device memory back to the host during decode.

---

## 4. Design

### 4.1 The position lives on the device

The CUDA execution context owns one device `int`, the decode position. At the start of every decode step
the network writes it with a one-thread kernel whose argument is the position -- a launch argument is
copied at launch, so the write cannot race the decode-ahead loop that enqueues step t+1 while step t runs.

RoPE decode, the KV-cache write, fused decode attention and the fallback decode softmax read the position
from that `int` in both modes. One kernel per op, called or replayed. The host still passes `position` down
the call chain, for range checks and for ops that have no CUDA kernel.

A layer-split network's stages each have their own context; the network writes the position into every
stage's context.

### 4.2 The split count is chosen on the device (option C)

Split-K exists because Gemma's global layers have one KV head: without splitting the band, one block would
read the whole cache. The policy today, on the host, every token:

```
splits = max( 1, min( ceil( 128 / NKV ), ceil( band / 64 ), 128 ) )
```

It grows with the band until it reaches the grid-filling target -- 16 splits for Llama 8B and Gemma's
sliding layers (NKV 8) from about 1,000 positions, 128 for Gemma's global layers (NKV 1) from about 8,100.

The launcher sizes the grid for the most splits the op can ever use (from the build's context length, or
the window for a sliding layer), and every block computes today's policy itself from the device-resident
length. Blocks whose split index is past the live count exit at once. The fix-up is launched whenever the
maximum is above one, reads the same count, and exits at once when it is one, since the main kernel then
wrote the output. The policy is one `__host__ __device__` function, so the host (sizing the grid) and the
device (choosing the live count) cannot disagree.

Same splits, same chunk boundaries, same summation order as today: decode logits are **bit-identical** to
today's at every position. The costs are idle blocks that exit immediately and, below saturation, a fix-up
launch where there was none. Past saturation the launched grid is exactly today's. Scratch is already sized
for 128 splits.

### 4.3 The network owns the cached length

The number of positions the KV caches hold is the same for every layer, so the network tracks it: set by
prefill (offset + chunk), advanced by decode (position + 1), zeroed by reset, set by rewind.
`rewindKvCache( position )` becomes `rewindKvCache( position, cached_length )` through the blocks and ops,
and each op's `cached_seq_len_` is removed. A replayed step leaves no host state behind, and the cached
length has one source.

### 4.4 The decode wrapper

`LanguageModelNetwork::decode( input, position )` becomes non-virtual; each transformer's decode becomes the
override `onDecode`. The wrapper:

1. Writes the position (4.1) and advances the cached length (4.3).
2. Runs `onDecode` -- the called path -- when replay is off, when anything is observing (5.4), or before a
   recording exists.
3. Otherwise launches the recording and returns the logits tensor it was recorded against.

A recording's life, owned by a CUDA-only type `CudaDecodeGraph` in its own module:

| Step after load | What runs |
|---|---|
| 1st | Called. Performs the lazy setup and touches every buffer the step uses. |
| 2nd | `onDecode` under stream capture (thread-local mode), instantiated, then launched to produce this step's logits. |
| 3rd | The self-check (5.1). |
| Every later step | One `cudaGraphLaunch`. |

The recording holds the input token tensor's address and is recorded again if it changes; the logits tensor is
the network's own and cannot move, which the self-check confirms once. So a caller that decodes from a new token tensor every step is recorded every step and never
replays; the models, and the tests, keep one tensor and write each token into it. The network reaches the
recording through `IExecutionContext::createDecodeRecording()` (`Compute.IDecodeRecording`), which is null on a
context that cannot record, so `LanguageModelNetwork` names no CUDA type. It survives `generate()` calls and chat turns: prefill does not touch any buffer the decode
step uses, and scratch cannot move after build. A rebuild or a reset of the network discards it.

### 4.5 The sampler stays outside

The model enqueues sampling after the decode step, as today, as one called launch. Its per-token uniform and
its top-k/top-p settings are ordinary launch arguments, so nothing about sampling constrains the recording.

### 4.6 Who turns it on

`LlamaModel`, `GemmaModel` and `QwenModel` turn replay on at load. A network built directly -- every
component and network test -- is off unless the test turns it on, so a test that switches a component after
build (flash on or off, `setState`) can never replay a recording of the old configuration.

Never replayed: `GptModel` (learned position table and the MHA op, not covered by 4.1), and a layer-split
deployment (two devices, two streams; a later change if measured to pay).

---

## 5. Guardrails

### 5.1 The self-check turns a silent error into a detected one

A decode op that passes a per-token value as a launch argument produces correct output on the called path
and wrong output under replay from the third token on, with nothing raised. The self-check runs once per
recording, at the third step (position p + 1, the recording having been made at p):

1. Run the step called, and copy its logits to the host.
2. Replay the recording at the same position, and copy its logits to the host. Writing the same K and V to
   the same cache slot twice leaves the cache unchanged, so the replay sees exactly what the called step saw.
   A recurrent state is the exception: Qwen's DeltaNet layers and their convolution windows advance in place, so
   running a step twice would advance them twice and the replay would see a different state. The network copies
   that state to the host before the called step and puts it back before the replay (`holdDecodeState` /
   `restoreDecodeState`, over the blocks' existing snapshots), then frees the copy. A network with only KV caches
   copies nothing.
3. Compare bit for bit. Equal: replay is on for this network's life. Different: replay is turned off for this
   network's life, and a warning names the network and says the called path is in use.

It costs one extra decode step and two logits copies per load, and it catches the whole class, including in a
network a user composed from their own components. It needs decode to be deterministic -- the same step twice
gives the same bits -- which is itself a gate (7.2, B1). A capture that fails outright takes the same exit:
replay off, one warning.

### 5.2 The off switch

`LanguageModel::setDecodeReplay( bool )`, on the model, so it reaches every load path (`load( path, plan )`,
`load( path, config, device )`) without passing through the plan. Debugging a decode step means turning
replay off first; the spec for each family's decode points here. Exposure in Chat, the binding and MIS is not
part of this change.

### 5.3 The rule for decode ops

Written into CLAUDE.md's Architecture section in the change that makes replay the default, beside the
scratch-buffer trap it resembles:

> **A decode step must be a pure function of device memory.** Under replay (`DecodeGraph.md`) the launches
> of a decode step are recorded once and replayed every token. So a decode op may not pass a per-token value
> as a launch argument (read the decode position from the execution context), may not update host state that
> a later call reads, and may not allocate or read device memory back to the host after its first call. A
> violation is caught by the self-check on a network's third decode step, which turns replay off and warns.

### 5.4 Observation takes the called path

Publications to observers are host callbacks, which a recording cannot make. A step with an activation
observer installed on the network's context runs called; the recording is kept and used again once observing
stops. An observed generation is therefore slower than an unobserved one; `Observability.md` gains that
sentence in Phase B.

### 5.5 Scope

Decode only: prefill runs thousands of rows per launch and has no launch gap worth a recording. One wrapper,
one `CudaDecodeGraph` module, the device position and the device split count. Nothing in any block or
component changes except the rename to `onDecode` and the rewind signature.

---

## 6. Alternatives Considered

### 6.1 Fusion instead

Residual + norm, split + RoPE + cache write, the gated activation into its projection, and the fix-up into
attention remove about 200 of Llama's 451 launches (~0.42 ms, ~78 tokens a second) and about 500 of Gemma's
964 (~1.2 ms, ~51). It stops short of the ceiling and merges components across the boundaries a reader follows.
Fusions that also cut memory traffic remain worth doing on their own merits; they are not this change.

### 6.2 The split count

- **A. Fixed at the saturated count.** Static, but below saturation the summation order changes, so decode
  logits differ from today's by rounding at short lengths. Rejected for C, which costs the same and changes
  nothing.
- **B. One recording per split count.** Today's numbers, but Gemma's global layers pass through up to 128
  counts while the band grows to 8K: up to 128 captures and a selection among them. Rejected.
- **C. Maximum grid, count chosen on the device.** Chosen (4.2).

### 6.3 Patching launch arguments in the recording

`cudaGraphExecKernelNodeSetParams` could rewrite each node's position argument every token, leaving the
kernels as they are. It needs the recording to know which nodes carry a position and where in each argument
list -- a map of kernel signatures kept in step with every kernel by hand. Rejected: the device position is
one read in each kernel, visible where it happens.

---

## 7. Phases And Gates

### 7.1 Phase A -- position and split count on the device, cached length in the network

No recording yet; every step is called.

- **A1. Bit-identical.** Decode logits at positions spanning split-count changes (1, 17, 63, 64, 65, 1000,
  8200) and past a sliding window's wrap, on the tiny Llama, Gemma and Qwen fixtures and on the Llama 8B and
  Gemma 12B Q4_0 packages, equal a reference dumped by the build before the change, bit for bit.
- **A2. Rewind.** Gemma's prompt-prefix reuse accepts and refuses exactly as before (existing tests), with the
  cached length from the network.
- **A3. The cost of C.** Decode rate at depths 16, 256, 1024 and 8192 on both packages against the build before
  the change; a regression below saturation is reported with its size before Phase B starts.

**Result (2026-09-29, `+19`, RTX 5060 Ti).** A1: all eight arms of `Tests/Dnn/Models/DecodeReference.Cuda.cpp`
(tiny Llama BF16, BF16 with FP8 KV and FP32; tiny Gemma across its window; tiny Qwen; Llama 3.1 8B Q4_0 with BF16
and FP8 KV; Gemma 4 12B Q4_0) equal the `+18` reference at every step from 1 to 8200, bit for bit; every arm
first reproduced itself at `+18`, so the called path was already deterministic. A2: the prefix-reuse tests pass
with the cached length from the network. A3, tokens a second, `+18` (mean of two runs of three, measured before
and after) against Phase A, one build each, 129 tokens:

| Depth | Llama 3.1 8B | Gemma 4 12B |
|---|---|---|
| 16 | 74.72 -> 74.17 (+0.10 ms) | 47.64 -> 47.42 (+0.10 ms) |
| 256 | 73.64 -> 73.29 (+0.06 ms) | 46.88 -> 46.81 (+0.03 ms) |
| 1024 | 71.65 -> 71.37 (+0.05 ms) | 45.98 -> 45.78 (+0.10 ms) |
| 8192 | 59.41 -> 59.22 (+0.05 ms) | 44.33 -> 44.14 (+0.10 ms) |

0.1-0.7% at every depth, and about half of it past saturation, where the launched grid is today's -- so that part
is not option C's idle blocks and fix-up. It is consistent with the dependent read of the device position at the
head of each position-reading kernel and the one-thread write per step; inferred, not profiled. Both are small
against the ~1 ms (Llama) and ~2.3 ms (Gemma) the recording removes.

### 7.2 Phase B -- the recording

- **B1. Determinism.** The same called step twice gives bit-identical logits, per family.
- **B2. Replay is the called path.** A greedy generation with replay on equals one with replay off in every
  token and every step's logits, bit for bit: tiny Llama (FP32, the cuBLASLt fallback), tiny Qwen, tiny Gemma
  past its window, and the Llama 8B and Gemma 12B packages.
- **B3. The self-check catches a baked-in position.** A test op that reads the host position as a launch
  argument, in a test network, is caught at load: replay off, warning raised, output still correct.
- **B4. Observation.** With an observer installed, `generate()` delivers every decode publication.
- **B5. Rewind after replay.** A Gemma chat continuation after replayed steps reuses the prefix as the called
  path does.
- **B6. Rate.** `DecodeLaunch.Cuda.cpp`, the comparison table (`Mila/Profiling/Benchmarks/benchmark_comparison.py`)
  on the 5060 Ti, and the full suite on both cards.

**Result (2026-09-29, `+19`, RTX 5060 Ti).** B1: every arm reproduced itself at `+18` (7.1). B2: the eight arms of
`DecodeReference.Cuda.cpp`, replayed, equal the called `+18` reference at every step from 1 to 8200, each with
exactly three called steps (priming, recording, self-check); `DecodeReplay.Cuda.cpp` repeats it at 300 steps in
the suite. Its Qwen arm exposed that a build initialized `CausalConv1d` to zero, which silenced every DeltaNet layer
of a network built without weights -- its recurrent state never reached the logits, so the A1 and B2 Qwen arms
covered only the attention layer. The convolution now takes PyTorch's Conv1d default, a test asserts the fixture's
logits depend on the recurrent state, and the tiny Qwen reference was rewritten from `+19`.
B3: a step that sets a logit byte from the position through a memset's value is caught and runs called, output
correct. B4: an observed run calls every step and delivers every decode publication. B5: prefix reuse after
replayed steps equals the called path. Without the recurrent-state restore of 5.1 the Qwen self-check fails --
demonstrated before it was kept. Full suite: 2087 tests on the 5060 Ti, 2083 on the 4070, clean build of every
target, no warnings. Rates, `ProfileModel` through `generate()`, 129 tokens, tokens a second:

| Depth | Llama 3.1 8B Q4_0 | Gemma 4 12B Q4_0 |
|---|---|---|
| 16 | 74.72 -> 80.20 (-0.91 ms) | 47.64 -> 53.43 (-2.27 ms) |
| 256 | 73.64 -> 79.45 (-0.99 ms) | 46.88 -> 52.79 (-2.39 ms) |
| 1024 | 71.65 -> 77.10 (-0.99 ms) | 45.98 -> 51.03 (-2.15 ms) |
| 8192 | 59.41 -> 63.28 (-1.03 ms) | 44.33 -> 49.07 (-2.18 ms) |

The saving is the one section 1 measured, at every depth. `DecodeLaunch.Cuda.cpp` on the same build: Llama 484
kernels a step, 13.41 -> 12.38 ms; Gemma 1013, 21.04 -> 18.53 ms.

The comparison table's head-to-head rows on the same build against llama.cpp b11216 (driver 617.14), generation
of 128 tokens, Mila / llama.cpp: Llama 3.1 8B Q4_0 80 / 79 (1.02x), 63 / 65 (0.97x), 39 / 39 (1.00x) at depth 0,
8K and 32K; Gemma 4 12B Q4_0 54 / 54 (0.99x), 49 / 51 (0.97x), 45 / 48 (0.94x). Section 8's projection held to
within a token a second. What remains is attention at depth; the Gemma row also carries the output-head difference
section 8 states as a condition of the comparison.

---

## 8. Expected Result

The recording saves about 1 ms a token for Llama and about 2.3 ms for Gemma at every depth. Projected on the
RTX 5060 Ti from the measurements before it, against llama.cpp on the same GGUF weights:

| Tokens a second | depth 0 | 8K | 32K |
|---|---|---|---|
| Llama 3.1 8B Q4_0 (llama.cpp) | ~81 (78) | ~63 (65) | ~39 (39) |
| Gemma 4 12B Q4_0 (llama.cpp) | ~54 (54) | ~49 (50) | ~45 (48) |

So the recording brings Llama ahead with no context and level at 32K, and Gemma level with no context. What
remains is attention at depth -- Llama's decode attention reads an 8K cache at about 70% of bandwidth, and
Gemma's head-size-512 global layers fall away with depth. That is the next item, and it is not this change (done at
`+20`: `GqaDecodeAttention.md`). One
condition of the Gemma row: the compared weights differ in the output head, FP8 in Mila's package (1.0 GB read a
token) and Q6_K in the GGUF (0.83 GB).

---

## 9. Open

1. Chat, binding and MIS exposure of the off switch -- decided when a user needs it, not before.
2. A layer-split recording (two streams, one graph per stage or one multi-stream graph) -- only if measured to
   pay on a split deployment.
