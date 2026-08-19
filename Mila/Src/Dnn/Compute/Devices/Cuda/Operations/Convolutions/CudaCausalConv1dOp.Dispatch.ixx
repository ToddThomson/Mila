module;
#include <cuda_bf16.h>
#include <cstdint>
#include "Kernels/CausalConv1d.cuh"

export module Compute.CudaCausalConv1dOp:Dispatch;

namespace Mila::Dnn::Compute::Cuda::Convolution
{
    /**
     * @brief CUDA causal-convolution kernel dispatch, specialized on native element type.
     */
    namespace Detail
    {
        template<typename TElementType>
        struct cuda_causal_conv1d_impl;

        template<>
        struct cuda_causal_conv1d_impl<float>
        {
            static inline void forward(
                float* out, const float* x, const float* state,
                const float* weight, const float* bias,
                int B, int T, int C, int K, cudaStream_t stream )
            {
                cuda_causal_conv1d_forward_fp32( out, x, state, weight, bias, B, T, C, K, stream );
            }

            static inline void updateState(
                float* state, const float* x,
                int B, int T, int C, int K, cudaStream_t stream )
            {
                cuda_causal_conv1d_update_state_fp32( state, x, B, T, C, K, stream );
            }
        };

        template<>
        struct cuda_causal_conv1d_impl<nv_bfloat16>
        {
            static inline void forward(
                nv_bfloat16* out, const nv_bfloat16* x, const nv_bfloat16* state,
                const nv_bfloat16* weight, const nv_bfloat16* bias,
                int B, int T, int C, int K, cudaStream_t stream )
            {
                cuda_causal_conv1d_forward_bf16( out, x, state, weight, bias, B, T, C, K, stream );
            }

            static inline void updateState(
                nv_bfloat16* state, const nv_bfloat16* x,
                int B, int T, int C, int K, cudaStream_t stream )
            {
                cuda_causal_conv1d_update_state_bf16( state, x, B, T, C, K, stream );
            }
        };
    }
}
