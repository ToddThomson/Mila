#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute::Cuda::Residual
{
    void cuda_residual_forward_fp32(
        float* Y,
        const float* X1, const float* X2,
        float scale,
        int N,
        cudaStream_t stream );

    /**
     * @brief Host function to launch residual backward kernel with full precision (FP32)
     *
     * Propagates gradients through the residual connection. Both inputs receive
     * the same gradient since the forward pass is simple addition.
     *
     * Formula: dX1 += dY, dX2 += dY
     *
     * @param dX1 Gradient tensor for first input
     * @param dX2 Gradient tensor for second input
     * @param dY Gradient from downstream layers
     * @param N Total number of elements in the tensors
     * @param stream CUDA stream for asynchronous execution
     */
    void cuda_residual_backward_fp32(
        float* dX1,
        float* dX2,
        const float* dY,
        int N,
        cudaStream_t stream );

    void cuda_residual_forward_bf16(
        __nv_bfloat16* Y,
        const __nv_bfloat16* X1, const __nv_bfloat16* X2,
        float scale,
        int N,
        cudaStream_t stream );

    void cuda_residual_backward_bf16(
        __nv_bfloat16* dX1,
        __nv_bfloat16* dX2,
        const __nv_bfloat16* dY,
        int N,
        cudaStream_t stream );
}