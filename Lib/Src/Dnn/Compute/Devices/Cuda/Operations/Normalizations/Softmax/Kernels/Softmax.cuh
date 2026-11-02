#pragma once

// Minimal header for Softmax host launchers (FP32/FP16) and header-visible templated dispatchers.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace Mila::Dnn::Compute::Cuda::Softmax
{
    // Host wrappers implemented in the CUDA translation units (Softmax.Fp32.cu / Softmax.Fp16.cu).

    void cuda_softmax_forward_fp32(
        float* Y,
        const float* X,
        int N,
        int C,
        cudaStream_t stream );

    void cuda_softmax_forward_fp16(
        half* Y,
        const half* X,
        int N,
        int C,
        cudaStream_t stream );

    void cuda_softmax_forward_general_fp32(
        float* Y,
        const float* X,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream );

    void cuda_softmax_forward_general_fp16(
        half* Y,
        const half* X,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream );

    // Backward host wrappers (implemented in the FP32/FP16 .cu files)
    void cuda_softmax_backward_fp32(
        float* dX,
        const float* dY,
        const float* Y,
        int N,
        int C,
        cudaStream_t stream );

    void cuda_softmax_backward_fp16(
        half* dX,
        const half* dY,
        const half* Y,
        int N,
        int C,
        cudaStream_t stream );

    void cuda_softmax_backward_general_fp32(
        float* dX,
        const float* dY,
        const float* Y,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream );

    void cuda_softmax_backward_general_fp16(
        half* dX,
        const half* dY,
        const half* Y,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream );

    // Header-visible templated dispatchers.
    template <typename TPrecision>
    inline void cuda_softmax_forward(
        TPrecision* Y,
        const TPrecision* X,
        int N,
        int C,
        cudaStream_t stream )
    {
        static_assert( std::is_same_v<TPrecision, float> || std::is_same_v<TPrecision, half>,
                       "cuda_softmax_forward: unsupported precision" );

        if constexpr ( std::is_same_v<TPrecision, float> ) {
            cuda_softmax_forward_fp32( Y, X, N, C, stream );
        }
        else {
            cuda_softmax_forward_fp16( reinterpret_cast<half*>(Y), reinterpret_cast<const half*>(X), N, C, stream );
        }
    }

    template <typename TPrecision>
    inline void cuda_softmax_forward_general(
        TPrecision* Y,
        const TPrecision* X,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream )
    {
        static_assert( std::is_same_v<TPrecision, float> || std::is_same_v<TPrecision, half>,
                       "cuda_softmax_forward_general: unsupported precision" );

        if constexpr ( std::is_same_v<TPrecision, float> ) {
            cuda_softmax_forward_general_fp32( Y, X, outer_size, dim_size, inner_size, stream );
        }
        else {
            cuda_softmax_forward_general_fp16( reinterpret_cast<half*>(Y), reinterpret_cast<const half*>(X), outer_size, dim_size, inner_size, stream );
        }
    }

    // Templated backward dispatchers
    template <typename TPrecision>
    inline void cuda_softmax_backward(
        TPrecision* dX,
        const TPrecision* dY,
        const TPrecision* Y,
        int N,
        int C,
        cudaStream_t stream )
    {
        static_assert( std::is_same_v<TPrecision, float> || std::is_same_v<TPrecision, half>,
                       "cuda_softmax_backward: unsupported precision" );

        if constexpr ( std::is_same_v<TPrecision, float> ) {
            cuda_softmax_backward_fp32( dX, dY, Y, N, C, stream );
        }
        else {
            cuda_softmax_backward_fp16( reinterpret_cast<half*>(dX), reinterpret_cast<const half*>(dY), reinterpret_cast<const half*>(Y), N, C, stream );
        }
    }

    template <typename TPrecision>
    inline void cuda_softmax_backward_general(
        TPrecision* dX,
        const TPrecision* dY,
        const TPrecision* Y,
        int outer_size,
        int dim_size,
        int inner_size,
        cudaStream_t stream )
    {
        static_assert( std::is_same_v<TPrecision, float> || std::is_same_v<TPrecision, half>,
                       "cuda_softmax_backward_general: unsupported precision" );

        if constexpr ( std::is_same_v<TPrecision, float> ) {
            cuda_softmax_backward_general_fp32( dX, dY, Y, outer_size, dim_size, inner_size, stream );
        }
        else {
            cuda_softmax_backward_general_fp16( reinterpret_cast<half*>(dX), reinterpret_cast<const half*>(dY), reinterpret_cast<const half*>(Y), outer_size, dim_size, inner_size, stream );
        }
    }
}