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

            static inline void backward( float* dX1, float* dX2, const float* dY, size_t N, cudaStream_t stream )
            {
                cuda_residual_backward_fp32( dX1, dX2, dY, N, stream );
            }
        };

        template<>
        struct cuda_residual_impl<half>
        {
            static inline void forward( half* Y, const half* X1, const half* X2, float scale, int N, cudaStream_t stream )
            {
                cuda_residual_forward_fp16( Y, X1, X2, scale, N, stream );
            }

            static inline void backward( half* dX1, half* dX2, const half* dY, size_t N, cudaStream_t stream )
            {
                cuda_residual_backward_fp16( dX1, dX2, dY, N, stream );
            }
        };
    }
}
