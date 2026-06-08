# Token Sampling

Implementation Contract for Device-Side Logit Sampling in Mila's Decode Loop

---

## 1. Overview

`LlamaModel::onGenerating` currently performs token sampling entirely on the CPU via
`sampleFromLogits` → `sampleToken`. This requires a 512 KB device-to-host transfer of
the full logits tensor (`[1, 1, vocab_size]`) on every decode step, followed by CPU-side
softmax, top-k sort, and weighted sampling.

This specification introduces a `TokenSampler` component and corresponding backend
operations (`CudaSamplingOp`, `CpuSamplingOp`) that move sampling to the compute device,
reducing the per-step D2H transfer from 512 KB (full logits) to 4 bytes (single int32 token).

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
| Greedy (temperature ≤ 0 or top_k == 1) | 512 KB | 4 bytes |
| Stochastic top-k | 512 KB | 4 bytes |

---

## 3. Architecture

Mila's component/operation split applies directly:

- **`TokenSampler`** — hardware-agnostic component, lives in the `Dnn` layer.
  Holds an `ISamplingOperation`, delegates all compute to it.
- **`ISamplingOperation`** — compute-layer interface, defines the device contract.
- **`CudaSamplingOp`** — CUDA backend; implements argmax via CUB and stochastic
  sampling via softmax + cuRAND.
- **`CpuSamplingOp`** — CPU backend; implements the same logic with pre-allocated
  scratch buffers.

---

## 4. Design

### 4.1 `SamplerConfig`

```cpp
class SamplerConfig : public ComponentConfig
{
public:
    explicit SamplerConfig( int vocab_size, int max_top_k = 100 )
        : vocab_size_( vocab_size ), max_top_k_( max_top_k ) {}

    int getVocabSize() const noexcept { return vocab_size_; }
    int getMaxTopK()   const noexcept { return max_top_k_; }

private:
    int vocab_size_;
    int max_top_k_;
};
```

`vocab_size` and `max_top_k` are fixed at build time. They determine device buffer
allocation in `build()`. Per-request `temperature` and `top_k` are passed via
`configure()` before each `forward()` call (§4.2).

### 4.2 `ISamplingOperation` Interface

Sampling does not fit `UnaryOperation` because: (a) the output data type is always
`INT32` regardless of logits precision, and (b) per-request parameters must be set
between calls without rebuilding. A new interface is introduced:

```cpp
export struct ISamplingOperation
{
    /// Set per-request sampling parameters. Must be called before each forward().
    /// temperature: <= 0 selects argmax; > 0 enables stochastic sampling.
    /// top_k:       0 or vocab_size disables top-k filtering.
    virtual void configure( float temperature, int top_k ) = 0;

    /// Sample one token index from logits on the compute device.
    /// logits:    device tensor [1, 1, vocab_size] in model compute precision.
    /// token_out: device tensor [1, 1] of INT32 — written in-place on the device.
    virtual void forward( const ITensor& logits, ITensor& token_out ) = 0;

    virtual ~ISamplingOperation() = default;
};
```

`configure()` + `forward()` are always called sequentially on the same thread.
`token_out` is the caller-provided device INT32 buffer; in `LlamaModel` this is
`decode_token_device_`, enabling the device-to-device token flow described in §2.

### 4.3 `TokenSampler` Component

**File:** `Mila/Src/Dnn/Components/Sampling/TokenSampler.ixx`

```
TokenSampler<TDeviceType, TPrecision>
  Members:
    SamplerConfig config_
    shared_ptr<ISamplingOperation> operation_

  Hooks (follow Softmax component pattern):
    onExecutionContextSet() → creates operation via OperationRegistry
    onBuilding( BuildContext ) → calls operation_->build()

  Public:
    void sample( const ITensor& logits,
                 float temperature, int top_k,
                 ITensor& token_device_out )
      operation_->configure( temperature, top_k )
      operation_->forward( logits, token_device_out )
```

Construction follows the same standalone/shared context pattern as other components.
`sample()` is the single public method — `configure` + `forward` are always paired.

### 4.4 `CudaSamplingOp` — CUDA Backend

**File:** `Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Sampling/CudaSamplingOp.ixx`

#### build() — cold path

Pre-allocates all device buffers so the hot path is allocation-free:

```
// ArgMax / greedy path
argmax_result_device_   : Tensor<INT32,    CudaDeviceMR> [1]     — CUB ArgMax output
cub_argmax_temp_        : device byte buffer               — CUB temp storage (dry-run sized)

// Stochastic path
softmax_scratch_device_ : Tensor<FP32,     CudaDeviceMR> [vocab_size]  — scaled + softmaxed probs
random_uniform_device_  : Tensor<FP32,     CudaDeviceMR> [1]           — cuRAND uniform draw
sort_keys_in_           : Tensor<FP32,     CudaDeviceMR> [vocab_size]  — logit values for sort
sort_keys_out_          : Tensor<FP32,     CudaDeviceMR> [vocab_size]  — sorted logit values
sort_vals_in_           : Tensor<INT32,    CudaDeviceMR> [vocab_size]  — logit indices for sort
sort_vals_out_          : Tensor<INT32,    CudaDeviceMR> [vocab_size]  — sorted indices
cub_sort_temp_          : device byte buffer               — CUB temp storage (dry-run sized)
```

CUB temp storage sizes are determined via the standard dry-run pattern
(`nullptr` output, `nullptr` temp_storage, size query) during `build()`.

`max_top_k` from `SamplerConfig` determines the sort buffer size. If `top_k == 0` at
runtime, the full vocab sort is used (buffers already sized to `vocab_size`).

#### configure() — sets cached per-request state

```cpp
void configure( float temperature, int top_k ) override
{
    temperature_ = temperature;
    top_k_       = top_k;
}
```

#### forward() — hot path

**Greedy branch** (`temperature_ <= 0.0f || top_k_ == 1`):

```
cub::DeviceReduce::ArgMax(
    cub_argmax_temp_, cub_argmax_temp_bytes_,
    logits_ptr, argmax_result_device_.data(),
    vocab_size_, stream_ )

// Write key (index) from KeyValuePair result to token_out
cuda_write_int32_kernel<<<1,1,0,stream_>>>(
    token_out_ptr, argmax_result_device_.data() )
```

D2H on the hot path: zero bytes. The token lives in `token_out` on the device.

**Stochastic branch** (`temperature_ > 0.0f && top_k_ != 1`):

```
Step 1 — Temperature scaling + softmax (custom kernel, full vocab):
  softmax_scratch_[i] = exp( (logits[i] - max_logit) / temperature ) / sum

Step 2 — Top-k filtering (when top_k_ > 0 && top_k_ < vocab_size):
  // Populate sort input buffers: keys = softmax_scratch_, vals = [0..vocab_size)
  cub::DeviceRadixSort::SortPairsDescending(
      cub_sort_temp_, sort_keys_in_, sort_keys_out_,
      sort_vals_in_, sort_vals_out_, vocab_size_, stream_ )
  // Zero out softmax_scratch_ for non-top-k positions (custom kernel)
  // Renormalize top-k probabilities in-place (custom kernel)

Step 3 — GPU uniform draw:
  curandGenerateUniform( context_->getCurandGenerator(),
      random_uniform_device_.data(), 1 )

Step 4 — Prefix-sum sample (custom kernel):
  // Sequential scan: cumsum over softmax_scratch_ until cumsum >= random_uniform
  // Writes winning index to token_out
```

All four steps execute on the same CUDA stream. No synchronization between steps.

#### CUB dependency

CUB is a header-only library bundled with the CUDA Toolkit (≥ 11.0). Required headers:

```cpp
#include <cub/device/device_reduce.cuh>    // DeviceReduce::ArgMax
#include <cub/device/device_radix_sort.cuh> // DeviceRadixSort::SortPairsDescending
```

No additional link target is required.

### 4.5 `CpuSamplingOp` — CPU Backend

**File:** `Mila/Src/Dnn/Compute/Devices/Cpu/Operations/Sampling/CpuSamplingOp.ixx`

Implements the same algorithm as the removed `sampleToken` but with pre-allocated scratch
buffers eliminating the three per-call heap allocations:

```
build():
  probs_scratch_   : std::vector<float>  [vocab_size]   — softmax probabilities
  indices_scratch_ : std::vector<size_t> [max_top_k]    — top-k index candidates
  rng_             : std::mt19937        — moved from LlamaModel::onGenerating

forward():
  Greedy: std::max_element over logits → write index to token_out
  Stochastic:
    temperature scaling + softmax into probs_scratch_
    if top_k: std::partial_sort_copy into indices_scratch_ (size k, not vocab_size)
              renormalize over k entries only
    std::uniform_real_distribution draw from rng_
    linear scan of probs_scratch_ for cumsum threshold → write index to token_out
```

The `indices_scratch_` size is `max_top_k`, not `vocab_size` — fixing the current
implementation which allocates a full-vocab index vector.

### 4.6 Registration

Follow the `CudaSoftmaxOpRegistrar` / `CpuSoftmaxOpRegistrar` pattern.
Register under the name `"SamplingOp"` with a new `OperationType::SamplingOp` enum entry
and corresponding `OperationNames::Sampling` string constant.

`TokenSampler::onExecutionContextSet()` creates the operation via:

```cpp
operation_ = OperationRegistry::instance()
    .createSamplingOperation<TDeviceType, TPrecision>(
        OperationNames::Sampling,
        this->getExecutionContext(),
        config_ );
```

A `createSamplingOperation` overload is added to `OperationRegistry` alongside the
existing `createUnaryOperation` / `createBinaryOperation` overloads, keyed on
`(DeviceType, ComputePrecision)`.

---

## 5. `LlamaModel` Changes

### 5.1 New member

```cpp
TokenSampler<TDeviceType, TPrecision> token_sampler_;
```

Initialized in the constructor after `config_` (requires `vocab_size`):

```cpp
, token_sampler_( "token_sampler",
    SamplerConfig{ static_cast<int>( config.getVocabSize() ), /*max_top_k=*/ 100 } )
```

Built in `onGenerating` on first call, or preferably in `fromPretrainedImpl` alongside
the network build.

### 5.2 Members removed

| Member | Reason |
|---|---|
| `logits_staging_` (`Tensor<FP32, StagingMR> [1,1,vocab_size]`) | Logits no longer leave the device |
| `sampleFromLogits()` | Replaced by `token_sampler_.sample()` |
| `sampleToken()` | Logic moved into `CpuSamplingOp` |

### 5.3 Member renamed

`decode_token_staging_` → `next_token_staging_` to reflect its narrowed role:
a 1-element pinned INT32 buffer used only for the 4-byte D2H copy needed by
stop-token detection and the `on_token` callback.

### 5.4 Updated `onGenerating` decode loop

```
// Phase 1 — prefill
prefill_input  = makeTokenTensor( prefill_tokens )
logits         = network.prefill( prefill_input )
context.synchronize()
token_sampler_.sample( logits, temperature, top_k, decode_token_device_ )
context.synchronize()
copy( decode_token_device_, next_token_staging_ )   // 4-byte D2H
if stop_ids.contains( next_token_staging_[0] ): return
on_token( next_token_staging_[0] )

// Phase 2 — decode loop
for each step:
    logits = network.decode( decode_token_device_, position )
    context.synchronize()
    token_sampler_.sample( logits, temperature, top_k, decode_token_device_ )
    context.synchronize()
    copy( decode_token_device_, next_token_staging_ )   // 4-byte D2H
    if stop_ids.contains( next_token_staging_[0] ): break
    on_token( next_token_staging_[0] )
    ++position
```

`std::mt19937 rng` is removed from `onGenerating`; random state lives in `CpuSamplingOp`.

---

## 6. Invariants

| Invariant | Notes |
|---|---|
| `configure()` always precedes `forward()` within `sample()` | Enforced by `TokenSampler::sample()` — callers never call these separately |
| `token_out` shape is `[1, 1]` INT32 | Validated in `forward()` pre-condition |
| CUB sort buffers are sized to `vocab_size` at build time | Runtime `top_k` is always ≤ `vocab_size`; no runtime reallocation |
| cuRAND generator is stream-bound | `getCurandGenerator()` lazy-initializes on first call; always returns a generator bound to the execution context's stream |
| No D2H of logits on the hot path | Correctness depends on sampling staying entirely on device; `logits_staging_` must not be reintroduced |
| CPU and CUDA paths produce equivalent distributions | Both implement temperature-scaled softmax → top-k mask → weighted sample; test parity at fixed seed |

---

## 7. Open Questions

| # | Question | Impact |
|---|---|---|
| 1 | Is `max_top_k = 100` a reasonable default for sort buffer pre-allocation, or should it match a project-wide convention? | Affects `SamplerConfig` default |
| 2 | Should `TokenSampler` be built during `fromPretrainedImpl` (alongside the network) or lazily on first `onGenerating` call? | Affects where the CUDA device context is available for `token_sampler_.build()` |
| 3 | Is the prefix-sum scan in Step 4 of the stochastic path acceptable as a sequential GPU kernel for small vocab sizes, or should a parallel CUB `DeviceScan` be used? | Affects stochastic path kernel complexity |

---

## 8. Files Created / Modified

### Created

| File | Purpose |
|---|---|
| `Mila/Src/Dnn/Components/Sampling/TokenSampler.ixx` | Hardware-agnostic component |
| `Mila/Src/Dnn/Compute/Operations/ISamplingOperation.ixx` | Compute-layer interface |
| `Mila/Src/Dnn/Compute/Operations/SamplerConfig.ixx` | Build-time configuration |
| `Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Sampling/CudaSamplingOp.ixx` | CUDA backend |
| `Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Sampling/CudaSamplingOpRegistrar.ixx` | CUDA registrar |
| `Mila/Src/Dnn/Compute/Devices/Cpu/Operations/Sampling/CpuSamplingOp.ixx` | CPU backend |
| `Mila/Src/Dnn/Compute/Devices/Cpu/Operations/Sampling/CpuSamplingOpRegistrar.ixx` | CPU registrar |
| `Mila/Specifications/TokenSampling.md` | This specification |

### Modified

| File | Change |
|---|---|
| `Mila/Src/Dnn/Compute/Operations/OperationType.ixx` | Add `SamplingOp` enum entry and `OperationNames::Sampling` constant |
| `Mila/Src/Dnn/Compute/Operations/OperationRegistry.ixx` | Add `createSamplingOperation` overload |
| `Mila/Src/Dnn/Models/LlamaModel.ixx` | Add `TokenSampler` member; remove `logits_staging_`, `sampleFromLogits`, `sampleToken`; rename `decode_token_staging_` → `next_token_staging_`; update `onGenerating` |
