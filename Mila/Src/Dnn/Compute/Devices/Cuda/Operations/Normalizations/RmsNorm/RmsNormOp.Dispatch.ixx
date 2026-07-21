module;
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
         * Specialized for float (FP32) and half (BF16) native CUDA types.
         */
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, nv_bfloat16>
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
                float epsilon, float weight_offset,
                cudaStream_t stream )
            {
                cuda_rmsnorm_forward_fp32( Y, rstd, X, weight, bias, outer_size, inner_size, norm_dim, epsilon, weight_offset, stream );
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
        struct cuda_rmsnorm_impl<nv_bfloat16>
        {
            cuda_rmsnorm_impl() = default;

            static inline void forward(
                nv_bfloat16* Y, const nv_bfloat16* X,
                const nv_bfloat16* weight, const nv_bfloat16* bias,
                nv_bfloat16* rstd,
                int outer_size, int inner_size, int norm_dim,
                float epsilon, float weight_offset,
                cudaStream_t stream )
            {
                cuda_rmsnorm_forward_bf16( Y, rstd, X, weight, bias, outer_size, inner_size, norm_dim, epsilon, weight_offset, stream );
            }

            static inline void backward(
                nv_bfloat16* dX, nv_bfloat16* dweight, nv_bfloat16* dbias,
                const nv_bfloat16* dY, const nv_bfloat16* X, const nv_bfloat16* weight,
                const nv_bfloat16* rstd,
                int outer_size, int inner_size, int norm_dim,
                cudaStream_t stream )
            {
                cuda_rmsnorm_backward_bf16( dX, dweight, dbias, dY, X, weight, rstd, outer_size, inner_size, norm_dim, stream );
            }
        };
    }
}