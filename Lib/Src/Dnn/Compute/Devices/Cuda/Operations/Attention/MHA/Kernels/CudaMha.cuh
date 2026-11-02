/**
 * @file Attention.cuh
 * @brief CUDA kernel declarations for Multi-Head Attention operations.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute::Cuda::MultiHeadAttention
{
    // ========================================================================
    // FP32 Kernel Functions
    // ========================================================================

    void cuda_permute_qkv_fp32(
        float* q, float* k, float* v,
        const float* inp,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

    void cuda_permute_qkv_padded_fp32(
        float* q, float* k, float* v,
        const float* inp,
        int B, int input_T, int output_T, int NH, int HS,
        cudaStream_t stream );

    void cuda_permute_qkv_decode_fp32(
        float* q, float* k, float* v,
        const float* inp,
        int B, int position, int cache_T, int NH, int HS,
        cudaStream_t stream );

    void cuda_unpermute_output_fp32(
        const float* vaccum, float* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

    void cuda_unpermute_output_padded_fp32(
        const float* vaccum, float* out,
        int B, int actual_T, int T, int NH, int HS,
        cudaStream_t stream );

    void cuda_softmax_forward_fp32(
        float* att, float scale, const float* preatt,
        int B, int NH, int T,
        cudaStream_t stream );

    void cuda_softmax_padded_forward_fp32(
        float* att, float scale, const float* preatt,
        int B, int NH, int max_T, int actual_T,
        cudaStream_t stream );

    void cuda_softmax_decode_forward_fp32(
        float* att, float scale, const float* preatt,
        int B, int NH, int max_len, int actual_len,
        cudaStream_t stream );

    void cuda_softmax_backward_fp32(
        float* dpreatt, const float* datt, const float* att,
        float scale,
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

    void cuda_permute_qkv_padded_fp16(
        half* q, half* k, half* v,
        const half* inp,
        int B, int input_T, int output_T, int NH, int HS,
        cudaStream_t stream );

    void cuda_permute_qkv_decode_fp16(
        half* q, half* k, half* v,
        const half* inp,
        int B, int position, int cache_T, int NH, int HS,
        cudaStream_t stream );

    void cuda_unpermute_output_fp16(
        const half* vaccum, half* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream );

    void cuda_softmax_forward_fp16(
        half* att, float scale, const half* preatt,
        int B, int NH, int T,
        cudaStream_t stream );

    void cuda_softmax_padded_forward_fp16(
        half* att, float scale, const half* preatt,
        int B, int NH, int max_T, int actual_T,
        cudaStream_t stream );

    void cuda_softmax_decode_forward_fp16(
        half* att, float scale, const half* preatt,
        int B, int NH, int max_len, int actual_len,
        cudaStream_t stream );

    void cuda_softmax_backward_fp16(
        half* dpreatt, const half* datt, const half* att,
        float scale,
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
}