#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace Mila::Dnn::Compute::Cuda::TokenEmbedding
{
    // ========================================================================
    // FP32
    // ========================================================================
    void cuda_token_embedding_forward_fp32(
        float* Y, const int* X, const float* wte,
        int B, int T, int C, cudaStream_t stream );

    void cuda_token_embedding_backward_fp32(
        float* dwte, const float* dY, const int* X,
        int B, int T, int C, cudaStream_t stream );

    void cuda_token_embedding_decode_fp32(
        float* Y, const int* X, const float* wte,
        int B, int C, cudaStream_t stream );

    // ========================================================================
    // BF16
    // ========================================================================
    void cuda_token_embedding_forward_bf16(
        __nv_bfloat16* Y, const int* X, const __nv_bfloat16* wte,
        int B, int T, int C, cudaStream_t stream );

    void cuda_token_embedding_backward_bf16(
        __nv_bfloat16* dwte, const __nv_bfloat16* dY, const int* X,
        int B, int T, int C, cudaStream_t stream );

    void cuda_token_embedding_decode_bf16(
        __nv_bfloat16* Y, const int* X, const __nv_bfloat16* wte,
        int B, int C, cudaStream_t stream );

    // ========================================================================
    // BF16 output, FP8_E4M3 table (D4 Design B tied-table quantization)
    //
    // Gather-dequant: Y[b,t,:] = float(wte_fp8[X[b,t],:]) * scales[X[b,t]].
    // One float32 scale per vocabulary row -- the same axis as the tied
    // lm_head's per-output-channel scales, so one scale tensor serves both.
    // The table pointer is void* so this header stays safe for cl.exe TUs
    // that do not include cuda_fp8.h.
    // ========================================================================
    void cuda_token_embedding_forward_bf16_qfp8(
        __nv_bfloat16* Y, const int* X, const void* wte_fp8, const float* scales,
        int B, int T, int C, cudaStream_t stream );

    void cuda_token_embedding_decode_bf16_qfp8(
        __nv_bfloat16* Y, const int* X, const void* wte_fp8, const float* scales,
        int B, int C, cudaStream_t stream );
}