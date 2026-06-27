# Token Sampling

Implementation Contract for Device-Side Logit Sampling in Mila's Decode Loop

---

## 1. Overview

Every `LanguageModel`'s `onGenerating` (`LlamaModel`, `GemmaModel`, `GptModel`) currently
performs token sampling entirely on the CPU via `sampleFromLogits` → `sampleToken` — three
near-identical copies of the same host code. Each requires a device-to-host transfer of the
full logits tensor (`[1, 1, vocab_size]`) on every decode step, followed by CPU-side softmax,
top-k sort, and weighted sampling. The transfer is vocab-dependent: ~512 KB for Llama
(~128 K vocab × FP32), ~1 MB for Gemma 4 (~256 K vocab) — the default chat model.

This specification introduces a `TokenSampler` **orchestrator tool** (owned by the
`LanguageModel` base, not a graph `Component`) and corresponding backend operations
(`CudaSamplingOp`, `CpuSamplingOp`) that move sampling to the compute device, reducing the
per-step D2H transfer from the full logits tensor to 4 bytes (single int32 token).

The sampled token is written directly into the decode input buffer (`decode_token_device_`)
on the device, eliminating the H2D path for the next decode step entirely. A 4-byte D2H
copy retrieves the token for stop-sequence detection and the `on_token` callback.

---

## 2. Key Insight — Device-to-Device Token Flow

The current decode loop transfers data in both directions per step:

```
GPU: decode → logits [512 KB]
              ↓ D2H (512 KB)
CPU: sample → next_token [int32]
              ↓ H2D (4 bytes via decode_token_staging_)
GPU: decode ...
```

After this change:

```
GPU: decode → logits
GPU: sample → decode_token_device_ [int32, in-place]
              ↓ D2H (4 bytes)
CPU: stop check / on_token callback
GPU: decode (reads decode_token_device_ already in place) ...
```

The logits tensor never leaves the device. `logits_staging_` (the 512 KB pinned host
buffer added in the decode-loop refactor) is removed. `decode_token_staging_` is renamed
`next_token_staging_` and narrows to a 1-element INT32 pinned buffer used only for the
4-byte D2H read.

**Transfer summary:**

| Path | Current D2H | After |
|---|---|---|
| Greedy (temperature ≤ 0 or top_k == 1) | full logits | 4 bytes |
| Stochastic top-k | full logits | 4 bytes |

---

## 3. Architecture — Dispatch and Ownership

### 3.1 An orchestrator tool, not a graph component

`TokenSampler` is an orchestration tool owned by the `LanguageModel` base, **not** a `Component`:

- it owns no parameters or gradients and has no position in the module/operation graph;
- it runs *after* `lm_head` produces logits — post-graph, never inside `forward()`;
- it is invoked by the generation loop (`onGenerating`), exactly as an `Optimizer` is invoked
  by the training loop.

It is therefore the structural sibling of `Optimizer`: both take the model's
`IExecutionContext*` (shared, never owned) and are model-level, not graph-level. A sampler that
owned its own `ExecutionContext`/stream would force a per-step cross-stream sync and defeat the
in-place device-token flow of Section 2 — the sampler **must share the model's context** so it
reads `decode_logits` and writes `decode_token_device_` on the same stream as `decode`.

The `LanguageModel` base owns the single `TokenSampler`, replacing the three copied host
`sampleToken` implementations in `LlamaModel` / `GemmaModel` / `GptModel`.

### 3.2 Compile-time dispatch via OperationTraits

The device implementation is resolved through the unified `OperationTraits` table — the same
mechanism graph operations use, **not** the legacy `std::conditional_t` + `#ifdef MILA_HAS_CUDA`
facade dispatch that `AdamWOptimizer` currently uses (that path is being migrated onto
`OperationTraits`; see BACKLOG → "Migrate Optimizer dispatch onto OperationTraits"):

```cpp
using SamplingOpType = typename OperationTraits<
    OperationType::SamplingOp, TDeviceType, TPrecision>::type;
```

- `OperationType::SamplingOp` is the dispatch key (policy-free; `TPolicy = void`). Already in
  the enum.
- `SamplingOpConcept` (in `OperationTraits.Template.ixx`) enforces the op's method contract.
- Specializations live in the `:Cuda` / `:Cpu` partitions:
  `OperationTraits<SamplingOp, Cuda, BF16>::type = CudaSamplingOp<BF16>`, etc.
- A missing specialization is a hard compile error.

This makes `OperationTraits` the single compile-time dispatch for every device-backed compute
unit — graph ops **and** model-level orchestrator tools alike. The `MILA_HAS_CUDA` guard lives
only in the `OperationTraits.ixx` aggregator; the sampler and its facade carry none.

### 3.3 One sampler, filters as parameters

There is a single concrete `TokenSampler`, **not** a class per strategy (`TopK`, `TopP`, ...).
top-k, top-p, min-p, and repetition penalty are *composable filters* over one shared pipeline:

```
softcap -> temperature -> [top-k] -> [top-p] -> [min-p / repetition penalty] -> normalize -> draw
```

They stack (AND), not choose (XOR) — inference routinely applies `temperature`, `top_k`, and
`top_p` together. Only the masking step differs; softcap, softmax, the multinomial draw, RNG, the
device buffers, and the int32 output are identical. Greedy is the degenerate
`temperature <= 0 || top_k == 1` case inside the same pipeline. So filters are per-call
parameters, not types, and there is one `OperationType::SamplingOp` key.

The `Sampler<TDeviceType, TPrecision>` base is retained as the seam for a genuinely different
*future* strategy that carries cross-step state (e.g. Mirostat's running `mu`). Such a strategy
would be a sibling class with its own key — the test is **"different state/math ⇒ class;
composable mask over the same softmax+draw ⇒ parameter."** Beam search is out of scope: it
restructures the generation loop with multiple hypotheses and is not a per-token sampler.

### 3.4 Module layout

Mirrors the `Optimizer` / `AdamW` file structure, dispatched through `OperationTraits`:

| Role | Type | Module | Location |
|---|---|---|---|
| Base interface | `Sampler<Device,Precision>` | `Compute.SamplerBase` | `Core/SamplerBase.ixx` |
| Facade (owned by model) | `TokenSampler<Device,Precision>` | `Dnn.Samplers.TokenSampler` | `Samplers/TokenSampler.ixx` |
| Construction-time config | `SamplingConfig` | `Dnn.Samplers.SamplingConfig` | `Samplers/SamplingConfig.ixx` |
| Device op (CUDA) | `CudaSamplingOp<Precision>` | `Compute.CudaSamplingOp` | `Compute/Devices/Cuda/Operations/Sampling/` |
| Device op (CPU) | `CpuSamplingOp<Precision>` | `Compute.CpuSamplingOp` | `Compute/Devices/Cpu/Operations/Sampling/` |

This **retires the skeleton's `Dnn/Decoders/` naming**: `Decoder` base → `Sampler`,
`TopKDecoder` → `TokenSampler`, `TopKConfig` → `SamplingConfig`. ("Decoder" collides with the
network's `decode()` step; "Sampler" is unambiguous.)

`TokenSampler` retains the `TPrecision` axis because its input (logits) is a
`Tensor<TPrecision, MR>` (BF16 for Gemma), even though internal math runs in FP32.

---

## 4. Interface Contract

### 4.1 Construction-time vs per-call split

Unlike `AdamWConfig` (all hyperparameters fixed at construction), sampling parameters are
per-request and vary every call. The split:

- **`SamplingConfig`** (construction-time): `vocab_size`, `final_logit_softcap` — fixed by the
  model.
- **`SamplingParams`** (per-call): `temperature`, `top_k`, `top_p`, ... — the per-call slice of
  `GenerateParams` (`Dnn.GenerateParams`); the two should unify rather than duplicate.

### 4.2 The op contract

`SamplingOpConcept` currently takes loose scalars `(logits, token_out, temperature, top_k)`.
Because the filter set will grow (top_p / min_p / repetition penalty are already pencilled into
`GenerateParams`), the contract takes the params struct by const-ref so adding a filter does not
churn the concept or every call site:

```cpp
op.forward( const TLogits& logits, TToken& token_out,
            const SamplingParams& params, float random_uniform );
```

- `logits` — device tensor `[1, 1, vocab_size]` at model precision, read in place (never copied
  to host).
- `token_out` — device INT32 `[1, 1]`, written in place (the model's `decode_token_device_`).
- `random_uniform` — a single host-drawn uniform in `[0, 1)`; see 4.3.
- Non-const `op` is permitted (`CpuSamplingOp` may hold scratch); the CUDA op is effectively
  stateless.

The op writes the device token only. The 4-byte D2H readback and the per-step `synchronize()`
are the `TokenSampler` facade's responsibility, keeping sync placement in the orchestrator's
control.

### 4.3 RNG

Randomness stays on the host: the `TokenSampler` facade owns the `std::mt19937` (seeded per
`SamplingParams::seed`, or a time seed) and draws a single uniform `r` per step, passing it to
`forward()` as a scalar. The device op is then **pure and deterministic** given
`(logits, params, r)`. This avoids device RNG state and makes the parity oracle trivial (inject a
fixed `r`). The multinomial draw is a prefix-sum threshold against `r`.

### 4.4 Softcap

Gemma applies a final logit softcap `c * tanh(logits / c)` (`c = final_logit_softcap`) that the
current host `sampleToken` **drops** — harmless for argmax (the softcap is monotonic) but wrong
for stochastic sampling, where it changes the distribution after temperature. The `SamplingOp`
applies softcap **first, before temperature**, whenever `final_logit_softcap > 0`. This is a
latent correctness fix, not only a perf rewrite.

---

## 5. Device Algorithm (per branch)

The op branches on `SamplingParams`; internal math is FP32 regardless of `TPrecision`:

- **Greedy** (`temperature <= 0 || top_k == 1`): block-wide argmax reduction over vocab.
  Deterministic, no RNG. Exactly validatable against the host argmax.
- **Full multinomial** (`top_k == 0`, `top_p >= 1`): softcap → temperature → softmax →
  prefix-sum threshold against `r`.
- **Truncated** (`0 < top_k < vocab` and/or `top_p < 1`): the hard case — partial selection over
  a large vocab (~256 K for Gemma). Candidate approaches (chosen in the implementation plan, not
  here): radix-select, iterative value-threshold (histogram/bisection), or per-block top-k +
  merge. top-p is a cumulative-probability cutoff applied after the k-mask on the sorted
  survivors.

---

## 6. Buffer and Decode-Loop Changes

Per Section 2:

- **Remove `logits_staging_`** (the full-vocab pinned host buffer) — logits never leave the
  device.
- **Rename `decode_token_staging_` → `next_token_staging_`**, narrowed to a 1-element INT32
  pinned buffer used only for the 4-byte D2H readback.
- `decode_token_device_` is written in place by the op and read directly by the next `decode()`
  — the per-step H2D restage is eliminated.
- The per-step `synchronize()` **remains** (the host still needs the token for stop-sequence
  detection and the `on_token` callback).

---

## 7. Validation

- **Greedy path first**: deterministic, validated token-for-token against the existing host
  argmax over the same logits. This is the first milestone and protects the Gemma parity already
  achieved.
- **Stochastic path**: validated with a fixed seed and injected `r` — given identical
  `(logits, params, r)`, the device pipeline must select the same token as a reference host
  pipeline applying the same softcap / temperature / top-k / top-p. A statistical check (KL of the
  sampled histogram vs the reference distribution) is a secondary oracle.

---

## 8. Implementation Plan

### 8.1 A/B transition path (divergence guard)

The device sampler (**path B**) lands **alongside** the validated host `sampleToken` (**path A**),
not replacing it — moving sampling to the device risks a silent divergence from the token-for-token
Gemma parity already achieved. A two-state compile-time toggle selects the path:

```cpp
enum class SamplingPath { HostA, DeviceB };
constexpr SamplingPath kSamplingPath = SamplingPath::HostA;   // flip to DeviceB to test B
```

- **`HostA`** — current behavior, the unchanged baseline.
- **`DeviceB`** — the new `TokenSampler` device path.

Validation is **across builds, not in-process**: build with `HostA`, run the Chat harness on the
sample prompt and capture the output; rebuild with `DeviceB` and run the same prompt; compare. For the
greedy gate the decoded text must be identical token-for-token. This matches the project's
edit-a-constant-then-rebuild workflow and side-steps the shared-RNG plumbing an in-process comparator
would need. The rigorous stochastic check (fixed seed + injected `r` against a host reference) is the
unit oracle in Section 7, not this toggle.

**Retirement** — once `DeviceB` reproduces `HostA` across the validation prompts, delete path A
(`sampleFromLogits` / `sampleToken`), apply the Section 6 buffer changes, and remove the toggle, all in
a follow-up commit (Phase D). The toggle starts in `GemmaModel` (Phase A's target) and moves to the
`LanguageModel` base when the sampler is hoisted there.

### 8.2 Buffer-change sequencing

The Section 6 buffer changes are **deferred to A-retirement**, not done up front: `CompareAB` needs
path A's host logits, so `logits_staging_` is retained through the transition and removed only when A
is deleted. Phase A adds `decode_token_device_` in-place writes for B without yet removing A's
buffers.

### 8.3 Phases (each advances only when its build-vs-build parity gate is clean)

- **Phase A — greedy on-device, end-to-end.** `Sampler` base (`Core/SamplerBase.ixx`, rewriting the
  `Dnn/Decoders` skeleton), `SamplingConfig`, `SamplingParams`; `CudaSamplingOp` + `CpuSamplingOp`
  with the argmax branch only; `OperationTraits<SamplingOp, {Cuda,Cpu}, {FP32,BF16}>`
  specializations; `TokenSampler` facade (resolves the trait, owns RNG + the 4-byte readback); wired
  into `GemmaModel::onGenerating` behind the toggle. **Gate:** a `DeviceB` build reproduces the
  `HostA` build token-for-token on the Gemma sample prompt via the Chat harness.
- **Phase B — full multinomial** (`top_k == 0`): softcap → temperature → softmax → prefix-sum
  threshold against `r` in `CudaSamplingOp`; closes the softcap correctness gap. **Gate:** injected-`r`
  token match vs the host reference.
- **Phase C — truncated top-k / top-p:** device selection over the ~256 K vocab (correctness-first
  simple kernel; algorithm choice — radix-select / threshold-iteration / per-block top-k+merge —
  deferred and revisited only if profiled), top-p as a cumulative cutoff on the survivors. **Gate:**
  same injected-`r` parity.
- **Phase D — generalize + clean up:** hoist `TokenSampler` ownership to the `LanguageModel` base;
  wire `LlamaModel` + `GptModel` `onGenerating`; delete the three copied `sampleToken`s; flip to
  `DeviceB`, remove path A and apply the Section 6 buffer changes; retire the `Dnn/Decoders/`
  skeleton.

### 8.4 Locked decisions

- **Config naming:** `SamplingConfig` (construction-time: `vocab_size`, `final_logit_softcap`).
- **Per-call params:** extend `GenerateParams` (add `top_p`, `seed`) and pass it as `SamplingParams`
  — no parallel struct.
- **`CpuSamplingOp`** is included from Phase A (nearly free — it is the existing host `sampleToken`)
  so the base-class wiring is device-agnostic for Phase D.
- **Op signature:** `forward(const TLogits& logits, TToken& token_out, const SamplingParams&, float r)`
  — widening the current scalar `SamplingOpConcept`; op writes the device token only, facade owns the
  D2H readback + the per-step `synchronize()`.
- **Sequencing note:** the Optimizer→`OperationTraits` migration follows Phase A (not before) — the
  AdamW tests are disabled, so an optimizer-first migration proves only compile-time dispatch, whereas
  Phase A's greedy slice proves the orchestrator-on-traits pattern including runtime, validated
  against parity.

### 8.5 Build / boundary notes

Core `Mila/Src/` work touching **parity-protected** code (`GemmaModel::onGenerating`, the decode
loop, the buffer members) — gated behind the greedy `CompareAB` result before anything stochastic.
New module files and the `OperationTraits` partition specializations are registered in
`Mila/CMakeLists.txt`; the user builds and runs in VS 2026 and reports results.
