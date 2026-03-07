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
#include "Kernels/RmsNorm.cuh"

export module Compute.CudaRmsNormOp:Dispatch;

namespace Mila::Dnn::Compute::Cuda::RmsNorm
{
    namespace Detail
    {
        /**
         * @brief CUDA kernel dispatcher for RMSNorm operations.
         *
         * Specialized for float (FP32) and half (FP16) native CUDA types.
         */
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_rmsnorm_impl;

        template <>
        struct cuda_rmsnorm_impl<float>
        {
            cuda_rmsnorm_impl() = default;

            static inline void forward(
                float* Y, const float* X,
                const float* weight, const float* bias,
                float* rstd,
                int outer_size, int inner_size, int norm_dim,
                float epsilon,
                cudaStream_t stream )
            {
                cuda_rmsnorm_forward_fp32( Y, rstd, X, weight, bias, outer_size, inner_size, norm_dim, epsilon, stream );
            }

            static inline void backward(
                float* dX, float* dweight, float* dbias,
                const float* dY, const float* X, const float* weight,
                const float* rstd,
                int outer_size, int inner_size, int norm_dim,
                cudaStream_t stream )
            {
                cuda_rmsnorm_backward_fp32( dX, dweight, dbias, dY, X, weight, rstd, outer_size, inner_size, norm_dim, stream );
            }
        };

        template <>
        struct cuda_rmsnorm_impl<half>
        {
            cuda_rmsnorm_impl() = default;

            static inline void forward(
                half* Y, const half* X,
                const half* weight, const half* bias,
                half* rstd,
                int outer_size, int inner_size, int norm_dim,
                float epsilon,
                cudaStream_t stream )
            {
                // TODO: cuda_rmsnorm_forward_fp16( Y, rstd, X, weight, bias, outer_size, inner_size, norm_dim, epsilon, stream );
            }

            static inline void backward(
                half* dX, half* dweight, half* dbias,
                const half* dY, const half* X, const half* weight,
                const half* rstd,
                int outer_size, int inner_size, int norm_dim,
                cudaStream_t stream )
            {
                // TODO: cuda_rmsnorm_backward_fp16( dX, dweight, dbias, dY, X, weight, rstd, outer_size, inner_size, norm_dim, stream );
            }
        };
    }
}