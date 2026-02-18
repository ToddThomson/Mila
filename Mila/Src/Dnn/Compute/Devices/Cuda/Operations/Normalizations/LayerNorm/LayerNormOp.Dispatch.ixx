module;
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <sstream>
#include <cassert>
#include "Kernels/LayerNorm.cuh"

export module Compute.CudaLayerNormOp:Dispatch;

namespace Mila::Dnn::Compute::Cuda::LayerNorm
{
    namespace Detail
    {
        /**
         * @brief CUDA kernel dispatcher for LayerNorm operations.
         *
         * Specialized for float (FP32) and half (FP16) native CUDA types.
         */
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_layernorm_impl;

        template <>
        struct cuda_layernorm_impl<float>
        {
            cuda_layernorm_impl() = default;

            static inline void forward(
                float* Y, const float* X,
                const float* weight, const float* bias,
                float* mean, float* rstd,
                int outer_size, int inner_size, int norm_dim,
                float epsilon,
                cudaStream_t stream )
            {
                cuda_layernorm_forward_fp32( Y, mean, rstd, X, weight, bias, outer_size, inner_size, norm_dim, epsilon, stream );
            }

            static inline void forward_fast(
                float* Y, const float* X,
                const float* weight, const float* bias,
                float* mean, float* rstd,
                int num_slices, int norm_dim,
                float epsilon,
                cudaStream_t stream )
            {
                cuda_layernorm_forward_fp32_contig( Y, mean, rstd, X, weight, bias, num_slices, norm_dim, epsilon, stream );
            }

            static inline void backward(
                float* dX, float* dweight, float* dbias,
                const float* dY, const float* X, const float* weight,
                const float* mean, const float* rstd,
                int outer_size, int inner_size, int norm_dim,
                cudaStream_t stream )
            {
                cuda_layernorm_backward_fp32( dX, dweight, dbias, dY, X, weight, mean, rstd, outer_size, inner_size, norm_dim, stream );
            }

            static inline void backward_fast(
                float* dX, float* dweight, float* dbias,
                const float* dY, const float* X, const float* weight,
                const float* mean, const float* rstd,
                int num_slices, int norm_dim,
                cudaStream_t stream )
            {
                cuda_layernorm_backward_fp32_contig( dX, dweight, dbias, dY, X, weight, mean, rstd, num_slices, norm_dim, stream );
            }
        };

        template <>
        struct cuda_layernorm_impl<half>
        {
            cuda_layernorm_impl() = default;

            static inline void forward(
                half* Y, const half* X,
                const half* weight, const half* bias,
                half* mean, half* rstd,
                int outer_size, int inner_size, int norm_dim,
                float epsilon,
                cudaStream_t stream )
            {
                cuda_layernorm_forward_fp16( Y, mean, rstd, X, weight, bias, outer_size, inner_size, norm_dim, epsilon, stream );
            }

            static inline void forward_fast(
                half* Y, const half* X,
                const half* weight, const half* bias,
                half* mean, half* rstd,
                int num_slices, int norm_dim,
                float epsilon,
                cudaStream_t stream )
            {
                cuda_layernorm_forward_fp16_contig( Y, mean, rstd, X, weight, bias, num_slices, norm_dim, epsilon, stream );
            }

            static inline void backward(
                half* dX, half* dweight, half* dbias,
                const half* dY, const half* X, const half* weight,
                const half* mean, const half* rstd,
                int outer_size, int inner_size, int norm_dim,
                cudaStream_t stream )
            {
                cuda_layernorm_backward_fp16( dX, dweight, dbias, dY, X, weight, mean, rstd, outer_size, inner_size, norm_dim, stream );
            }

            static inline void backward_fast(
                half* dX, half* dweight, half* dbias,
                const half* dY, const half* X, const half* weight,
                const half* mean, const half* rstd,
                int num_slices, int norm_dim,
                cudaStream_t stream )
            {
                cuda_layernorm_backward_fp16_contig( dX, dweight, dbias, dY, X, weight, mean, rstd, num_slices, norm_dim, stream );
            }
        };
    }
}