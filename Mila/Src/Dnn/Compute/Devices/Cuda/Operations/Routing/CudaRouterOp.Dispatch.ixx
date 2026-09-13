module;
#include <cuda_bf16.h>
#include <cstdint>
#include "Kernels/Router.cuh"

export module Compute.CudaRouterOp:Dispatch;

namespace Mila::Dnn::Compute::Cuda::Routing
{
    namespace Detail
    {
        template<typename TElementType>
        struct cuda_router_select_impl;

        template<>
        struct cuda_router_select_impl<float>
        {
            static inline void forward(
                const float* logits, const float* per_expert_scale,
                float* weights, int32_t* indices,
                int rows, int experts, int top_k, cudaStream_t stream )
            {
                cuda_router_select_fp32( logits, per_expert_scale, weights, indices, rows, experts, top_k, stream );
            }
        };

        template<>
        struct cuda_router_select_impl<nv_bfloat16>
        {
            static inline void forward(
                const nv_bfloat16* logits, const nv_bfloat16* per_expert_scale,
                nv_bfloat16* weights, int32_t* indices,
                int rows, int experts, int top_k, cudaStream_t stream )
            {
                cuda_router_select_bf16( logits, per_expert_scale, weights, indices, rows, experts, top_k, stream );
            }
        };
    }
}
