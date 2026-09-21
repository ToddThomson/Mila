#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Routing
{
    /// Largest top_k the kernel holds in registers per row.
    inline constexpr int kMaximumTopK = 16;

    /**
     * @brief Mixture-of-experts selection over @p rows rows of @p experts router logits.
     *
     * Per row: softmax over all experts, keep the top_k by logit (equal logits keep the lower
     * expert index), renormalize those to sum to one, multiply each by per_expert_scale[expert].
     * Writes top_k weights and INT32 expert indices per row, in descending-logit order.
     *
     * PARALLELISM. One thread per row, and nothing shared: a row's 128 logits and its top_k
     * candidates stay in registers, so the kernel has no thread sync and no control-flow
     * uniformity to keep. Probabilities are accumulated in double, as the CPU reference does.
     */
    void cuda_router_select_fp32(
        const float* logits,
        const float* per_expert_scale,
        float* weights,
        int32_t* indices,
        int rows, int experts, int top_k,
        cudaStream_t stream );

    void cuda_router_select_bf16(
        const __nv_bfloat16* logits,
        const __nv_bfloat16* per_expert_scale,
        __nv_bfloat16* weights,
        int32_t* indices,
        int rows, int experts, int top_k,
        cudaStream_t stream );
}
