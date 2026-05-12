#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    // ========================================================================
    // FP32 Kernels
    // ========================================================================
    void cuda_matvec_decode_fp32(
        float* y,
        const float* x,
        const float* weight,
        const float* bias,
        int C,
        int OC,
        cudaStream_t stream );

    // Reduction kernels
    void cuda_reduce_sum_batch_fp32(
        float* dBias,
        const float* dY,
        int outer_size,
        int out_features,
        cudaStream_t stream );

    // ========================================================================
    // BF16 Kernels
    // ========================================================================
    void cuda_matvec_decode_bf16(
    __nv_bfloat16* y,
        const __nv_bfloat16* x,
        const __nv_bfloat16* weight,
        const __nv_bfloat16* bias,
        int C,
        int OC,
        cudaStream_t stream );

    /**
     * @brief BF16 activation x FP8-E4M3 weight decode-path matvec with per-channel dequantization.
     *
     * Computes y[oc] = scale[oc] * sum(x[c] * (float)weight[oc, c]) + bias[oc].
     * Requires C % 8 == 0 (8 FP8 elements per int2 load).
     */
    void cuda_matvec_decode_bf16_qfp8(
        __nv_bfloat16* y,
        const __nv_bfloat16* x,
        const __nv_fp8_e4m3* weight,
        const float* scales,
        const __nv_bfloat16* bias,
        int C,
        int OC,
        cudaStream_t stream );

    // TODO: Enable these functions when implemented
    //void cuda_reduce_sum_batch_fp16(
    //    half* dBias,
    //    const half* dY,
    //    int batch_size,
    //    int out_features,
    //    cudaStream_t stream );

    //void cuda_reduce_sum_batch_bfp16(
    //    __nv_bfloat16* dBias,
    //    const __nv_bfloat16* dY,
    //    int batch_size,
    //    int out_features,
    //    cudaStream_t stream );

    // Matmul functions
    void cuda_matmul_forward_fp32(
        float* Y,
        const float* X,
        const float* weight, const float* bias,
        int outer_size, int C, int OC,
        cudaStream_t stream );

    void cuda_matmul_backward_fp32(
        float* dX,
        float* dW,
        float* dB,
        const float* dY,
        const float* X,
        const float* weight,
        int outer_size, int C, int OC,
        cudaStream_t stream );

    void cuda_matmul_forward_fp16(
        half* Y, const half* X,
        const half* weight, const half* bias,
        int outer_size, int C, int OC,
        cudaStream_t stream );
}