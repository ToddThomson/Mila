
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute::Cuda::Swiglu
{    
    // ------------------------------------------------------------------------
    // FP32 implementation
    // ------------------------------------------------------------------------

    void cuda_swiglu_forward_fp32(
        float* Y,
        const float* X,
        int N, int half_width,
        cudaStream_t stream );

    void cuda_swiglu_backward_fp32(
        float* dX,
        const float* X,
        const float* dY,
        const int N, int half_width,
        cudaStream_t stream );

    // ------------------------------------------------------------------------
    // BF16 implementation
    // ------------------------------------------------------------------------

    void cuda_swiglu_forward_bf16(
        __nv_bfloat16* Y,
        const __nv_bfloat16* X,
        int N, int half_width,
        cudaStream_t stream );

    void cuda_swiglu_backward_bf16(
        float* dX,
        const __nv_bfloat16* X,
        const float* dY,
        int N, int half_width,
        cudaStream_t stream );
}