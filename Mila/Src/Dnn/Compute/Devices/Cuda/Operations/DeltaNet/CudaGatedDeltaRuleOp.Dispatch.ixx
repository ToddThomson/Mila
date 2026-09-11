module;
#include <cuda_bf16.h>
#include <cstdint>
#include "Kernels/GatedDeltaRule.cuh"

export module Compute.CudaGatedDeltaRuleOp:Dispatch;

namespace Mila::Dnn::Compute::Cuda::DeltaNet
{
    namespace Detail
    {
        template<typename TElementType>
        struct cuda_gated_delta_rule_impl;

        template<>
        struct cuda_gated_delta_rule_impl<float>
        {
            static inline void forward(
                float* out, const float* q, const float* k, const float* v,
                const float* a, const float* b,
                const float* A_log, const float* dt_bias,
                float* state,
                int batch, int steps, int num_k_heads, int num_v_heads,
                int head_k_dim, int head_v_dim, cudaStream_t stream )
            {
                cuda_gated_delta_rule_fp32( out, q, k, v, a, b, A_log, dt_bias, state,
                    batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim, stream );
            }
        };

        template<>
        struct cuda_gated_delta_rule_impl<nv_bfloat16>
        {
            static inline void forward(
                nv_bfloat16* out, const nv_bfloat16* q, const nv_bfloat16* k, const nv_bfloat16* v,
                const nv_bfloat16* a, const nv_bfloat16* b,
                const nv_bfloat16* A_log, const nv_bfloat16* dt_bias,
                float* state,
                int batch, int steps, int num_k_heads, int num_v_heads,
                int head_k_dim, int head_v_dim, cudaStream_t stream )
            {
                cuda_gated_delta_rule_bf16( out, q, k, v, a, b, A_log, dt_bias, state,
                    batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim, stream );
            }
        };
    }
}
