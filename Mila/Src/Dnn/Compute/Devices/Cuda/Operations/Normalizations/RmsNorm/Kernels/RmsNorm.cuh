/**
 * @file RmsNorm.cuh
 * @brief CUDA kernel declarations for RMS normalization (FP32/FP16) launchers.
 *
 * Declares host-launchable functions implemented in RmsNorm.Fp32.cu (and FP16 variant).
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace Mila::Dnn::Compute::Cuda::RmsNorm
{
    export inline constexpr int WARP_SIZE = 32;

    template<typename T>
    struct VecType
    {
        using type = float4;
        static constexpr int size = 4;
    };

    template<>
    struct VecType<half>
    {
        using type = half2;
        static constexpr int size = 2;
    };

    template<>
    struct VecType<__nv_fp8_e4m3>
    {
        using type = float4;
        static constexpr int size = 4;
    };

    __device__ __forceinline__ float to_float( float x )
    {
        return x;
    }

    __device__ __forceinline__ float to_float( half x )
    {
        return __half2float( x );
    }

    __device__ __forceinline__ float to_float( __nv_fp8_e4m3 x )
    {
        return float( x );
    }

    __device__ __forceinline__ float from_float( float x )
    {
        return x;
    }

    __device__ __forceinline__ half from_float_to_half( float x )
    {
        return __float2half( x );
    }

    __device__ __forceinline__ __nv_fp8_e4m3 from_float_to_fp8( float x )
    {
        return __nv_fp8_e4m3( x );
    }

    /**
     * @brief FP32 RMS normalization forward launcher.
     *
     * @param Y Output tensor (FP32)
     * @param rstd Per-slice reciprocal sqrt of mean-squared values (FP32, outer_size * inner_size elements)
     * @param X Input tensor (FP32)
     * @param weight Per-channel scaling parameters (norm_dim elements) or nullptr
     * @param bias Per-channel bias parameters (norm_dim elements) or nullptr
     * @param outer_size Product of dims before normalization axis
     * @param inner_size Product of dims after normalization axis
     * @param norm_dim Size of the normalized dimension
     * @param epsilon Numerical stability constant
     * @param stream CUDA stream for async execution
     */
    void cuda_rmsnorm_forward_fp32(
        float* Y,
        float* rstd,
        const float* X,
        const float* weight, const float* bias,
        int outer_size, int inner_size, int norm_dim,
        float epsilon,
        cudaStream_t stream );

    /**
     * @brief FP32 RMS normalization backward launcher.
     *
     * @param dX Output input-gradient buffer (FP32)
     * @param dweight Accumulator for weight gradients (norm_dim elements) or nullptr
     * @param dbias Accumulator for bias gradients (norm_dim elements) or nullptr
     * @param dY Output gradient (FP32)
     * @param X Original forward input (FP32)
     * @param weight Forward pass weight parameters (FP32) or nullptr
     * @param rstd Per-slice reciprocal sqrt computed during forward (FP32)
     * @param outer_size Product of dims before normalization axis
     * @param inner_size Product of dims after normalization axis
     * @param norm_dim Size of the normalized dimension
     * @param stream CUDA stream for async execution
     */
    void cuda_rmsnorm_backward_fp32(
        float* dX,
        float* dweight, float* dbias,
        const float* dY,
        const float* X,
        const float* weight,
        const float* rstd,
        int outer_size, int inner_size, int norm_dim,
        cudaStream_t stream );

    /**
     * @brief FP16 RMS normalization forward launcher.
     *
     * Statistics (rstd) may be stored in FP32 for numerical stability depending on implementation.
     */
    void cuda_rmsnorm_forward_fp16(
        half* Y,
        half* rstd,
        const half* X,
        const half* weight, const half* bias,
        int outer_size, int inner_size, int norm_dim,
        float epsilon,
        cudaStream_t stream );

    /**
     * @brief FP16 RMS normalization backward launcher.
     *
     * Parameter gradients may be accumulated in higher precision internally.
     */
    void cuda_rmsnorm_backward_fp16(
        half* dX,
        half* dweight, half* dbias,
        const half* dY,
        const half* X,
        const half* weight,
        const half* rstd,
        int outer_size, int inner_size, int norm_dim,
        cudaStream_t stream );
}