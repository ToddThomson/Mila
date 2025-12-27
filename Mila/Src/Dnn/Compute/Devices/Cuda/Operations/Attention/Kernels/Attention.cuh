/**
 * @file Attention.cuh
 * @brief CUDA kernel declarations for Multi-Head Attention operations.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute
{
    // ========================================================================
    // FP32 Kernel Functions
    // ========================================================================

    void cuda_permute_qkv_fp32(
        float* q, float* k, float* v,
        const float* inp,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

    void cuda_unpermute_output_fp32(
        const float* vaccum, float* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

    void cuda_softmax_forward_fp32(
        float* att, float scale, const float* preatt,
        int B, int NH, int T,
        cudaStream_t stream );

    void cuda_softmax_backward_fp32(
        float* dpreatt, const float* datt, const float* att,
        int B, int NH, int T,
        cudaStream_t stream );

    void cuda_unpermute_backward_fp32(
        float* dvaccum, const float* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

    void cuda_permute_backward_fp32(
        float* dinp,
        const float* dq, const float* dk, const float* dv,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

    // ========================================================================
    // FP16 Kernel Functions
    // ========================================================================

    void cuda_permute_qkv_fp16(
        half* q, half* k, half* v,
        const half* inp,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

    void cuda_unpermute_output_fp16(
        const half* vaccum, half* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

    void cuda_softmax_forward_fp16(
        half* att, float scale, const half* preatt,
        int B, int NH, int T,
        cudaStream_t stream );

    void cuda_softmax_backward_fp16(
        half* dpreatt, const half* datt, const half* att,
        int B, int NH, int T,
        cudaStream_t stream );

    void cuda_unpermute_backward_fp16(
        half* dvaccum, const half* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

    void cuda_permute_backward_fp16(
        half* dinp,
        const half* dq, const half* dk, const half* dv,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

    // ========================================================================
    // Fallback implementations (legacy cuBLAS)
    // ========================================================================

    void cuda_mha_forward_fallback(
        float* out,
        float* q, float* k, float* v,
        float* preatt, float* att, float* vaccum,
        const float* inp,
        int B, int T, int C, int NH,
        cudaStream_t stream );

    void cuda_mha_backward_fallback(
        float* dinp,
        float* dq, float* dk, float* dv,
        float* dpreatt, float* datt, float* dvaccum,
        const float* dout,
        const float* inp,
        const float* att,
        const float* q, const float* k, const float* v,
        int B, int T, int C, int NH,
        cudaStream_t stream );

    // cuBLAS handle (defined in CudaDevice or CudaDeviceResources)
    extern cublasHandle_t cublas_handle;
}