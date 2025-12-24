#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    // Reduction kernels
    void cuda_reduce_sum_batch_fp32(
        float* dBias,
        const float* dY,
        int outer_size,
        int out_features,
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

    void cuda_matmul_forward_fp16(
        half* Y, const half* X,
        const half* weight, const half* bias,
        int outer_size, int C, int OC,
        cudaStream_t stream );
}