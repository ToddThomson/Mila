module;
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <sstream>
#include <cassert>
#include "Kernels/Residual.cuh"

export module Compute.CudaResidualOp:Dispatch;

namespace Mila::Dnn::Compute::Cuda::Residual
{
    /**
     * @brief CUDA residual kernel dispatch implementations.
     *
     * Specializations provided for float and half native types.
     */
    namespace Detail
    {
        template<typename TElementType>
        struct cuda_residual_impl;

        template<>
        struct cuda_residual_impl<float>
        {
            static inline void forward( float* Y, const float* X1, const float* X2, float scale, int N, cudaStream_t stream )
            {
                cuda_residual_forward_fp32( Y, X1, X2, scale, N, stream );
            }

            static inline void backward( float* dX1, float* dX2, const float* dY, int N, cudaStream_t stream )
            {
                cuda_residual_backward_fp32( dX1, dX2, dY, N, stream );
            }
        };

        template<>
        struct cuda_residual_impl<nv_bfloat16>
        {
            static inline void forward( nv_bfloat16* Y, const nv_bfloat16* X1, const nv_bfloat16* X2, float scale, int N, cudaStream_t stream )
            {
                cuda_residual_forward_bf16( Y, X1, X2, scale, N, stream );
            }

            static inline void backward( nv_bfloat16* dX1, nv_bfloat16* dX2, const nv_bfloat16* dY, int N, cudaStream_t stream )
            {
                cuda_residual_backward_bf16( dX1, dX2, dY, N, stream );
            }
        };
    }
}
