#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace Mila::Dnn::Compute::Cuda::LayerNorm
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
     * @brief FP32 layer normalization forward pass.
     *
     * @param Y Normalized output
     * @param mean Per-slice mean statistics (outer_size * inner_size elements)
     * @param rstd Per-slice reciprocal standard deviation (outer_size * inner_size elements)
     * @param X Input tensor
     * @param weight Scaling parameters (norm_dim elements)
     * @param bias Shift parameters (norm_dim elements)
     * @param outer_size Product of dimensions before normalization axis
     * @param inner_size Product of dimensions after normalization axis
     * @param norm_dim Size of normalized dimension
     * @param epsilon Small constant for numerical stability
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_layernorm_forward_fp32(
        float* Y,
        float* mean, float* rstd,
        const float* X,
        const float* weight, const float* bias,
        int outer_size, int inner_size, int norm_dim,
        float epsilon,
        cudaStream_t stream );

    /**
     * @brief FP32 layer normalization backward pass.
     *
     * @param dX Input gradient (same shape as X)
     * @param dweight Weight parameter gradient accumulator (norm_dim elements)
     * @param dbias Bias parameter gradient accumulator (norm_dim elements)
     * @param dY Output gradient
     * @param X Original forward pass input
     * @param weight Forward pass weight parameters
     * @param mean Forward pass mean statistics
     * @param rstd Forward pass reciprocal standard deviation
     * @param outer_size Product of dimensions before normalization axis
     * @param inner_size Product of dimensions after normalization axis
     * @param norm_dim Size of normalized dimension
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_layernorm_backward_fp32(
        float* dX,
        float* dweight, float* dbias,
        const float* dY,
        const float* X,
        const float* weight,
        const float* mean, const float* rstd,
        int outer_size, int inner_size, int norm_dim,
        cudaStream_t stream );

    /**
     * @brief FP16 layer normalization forward pass.
     *
     * Statistics (mean/rstd) are computed and stored in FP32 for numerical stability.
     *
     * @param Y Normalized output (FP16)
     * @param mean Per-slice mean statistics in FP32 (outer_size * inner_size elements)
     * @param rstd Per-slice reciprocal standard deviation in FP32 (outer_size * inner_size elements)
     * @param X Input tensor (FP16)
     * @param weight Scaling parameters (FP16, norm_dim elements)
     * @param bias Shift parameters (FP16, norm_dim elements)
     * @param outer_size Product of dimensions before normalization axis
     * @param inner_size Product of dimensions after normalization axis
     * @param norm_dim Size of normalized dimension
     * @param epsilon Small constant for numerical stability
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_layernorm_forward_fp16(
        half* Y,
        half* mean, half* rstd,
        const half* X,
        const half* weight, const half* bias,
        int outer_size, int inner_size, int norm_dim,
        float epsilon,
        cudaStream_t stream );

    /**
     * @brief FP16 layer normalization backward pass.
     *
     * Parameter gradients (dweight/dbias) are accumulated in FP32 for numerical stability.
     *
     * @param dX Input gradient (FP16, same shape as X)
     * @param dweight Weight parameter gradient accumulator in FP32 (norm_dim elements)
     * @param dbias Bias parameter gradient accumulator in FP32 (norm_dim elements)
     * @param dY Output gradient (FP16)
     * @param X Original forward pass input (FP16)
     * @param weight Forward pass weight parameters (FP16)
     * @param mean Forward pass mean statistics (FP32)
     * @param rstd Forward pass reciprocal standard deviation (FP32)
     * @param outer_size Product of dimensions before normalization axis
     * @param inner_size Product of dimensions after normalization axis
     * @param norm_dim Size of normalized dimension
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_layernorm_backward_fp16(
        half* dX,
        half* dweight, half* dbias,
        const half* dY,
        const half* X,
        const half* weight,
        const half* mean, const half* rstd,
        int outer_size, int inner_size, int norm_dim,
        cudaStream_t stream );
}