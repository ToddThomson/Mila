#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace Mila::Dnn::Compute::Cuda::Convolution
{
    /**
     * @brief Depthwise causal 1-D convolution over the sequence axis.
     *
     * Tensors are [B, T, C] and the filter is [C, K], one length-K filter per channel:
     *
     *   out[b, t, c] = bias[c] + sum(i = 0 .. K-1) weight[c, i] * x[b, t - (K-1) + i, c]
     *
     * Positions left of the chunk come from @p state, which holds the previous K-1 input
     * rows as [B, K-1, C] with row j at relative position -(K-1)+j. Passing a null state
     * means the sequence starts at position 0 and the missing rows are zero -- which is
     * what left zero-padding means, and is why the caller must not zero-fill a state
     * buffer to fake a fresh sequence: an all-zero state is indistinguishable from one,
     * but only if it really is the start.
     *
     * Accumulation is in float regardless of storage type. A 4-tap dot product in BF16
     * loses more than the conversion costs.
     */
    void cuda_causal_conv1d_forward_fp32(
        float* out,
        const float* x,
        const float* state,
        const float* weight,
        const float* bias,
        int B, int T, int C, int K,
        cudaStream_t stream );

    void cuda_causal_conv1d_forward_bf16(
        __nv_bfloat16* out,
        const __nv_bfloat16* x,
        const __nv_bfloat16* state,
        const __nv_bfloat16* weight,
        const __nv_bfloat16* bias,
        int B, int T, int C, int K,
        cudaStream_t stream );

    /**
     * @brief Refresh the conv state to the last K-1 rows of [state ; x].
     *
     * Run AFTER the forward pass -- it overwrites the left context that pass reads.
     * Handles T < K-1 (the decode case, T == 1) by keeping the still-relevant tail of
     * the old state, so one launcher covers prefill and decode.
     *
     * One thread owns all K-1 destination rows for a given (b, c) and stages them in
     * registers before writing, which is what makes the in-place shift safe.
     */
    void cuda_causal_conv1d_update_state_fp32(
        float* state,
        const float* x,
        int B, int T, int C, int K,
        cudaStream_t stream );

    void cuda_causal_conv1d_update_state_bf16(
        __nv_bfloat16* state,
        const __nv_bfloat16* x,
        int B, int T, int C, int K,
        cudaStream_t stream );
}
