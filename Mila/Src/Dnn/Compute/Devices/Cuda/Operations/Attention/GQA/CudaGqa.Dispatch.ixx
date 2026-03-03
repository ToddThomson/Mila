module;
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <sstream>
#include <cassert>
#include "Kernels/CudaGqa.cuh"
#include "../Common/Kernels/CudaAttention.cuh"

export module Compute.CudaGroupedQueryAttentionOp:Dispatch;

namespace Mila::Dnn::Compute::Cuda::GroupedQueryAttention
{
    namespace Detail
    {
        template<typename T>
        struct cuda_gqa_kernels;

        // ====================================================================
        // FP32 specialization
        // ====================================================================

        template<>
        struct cuda_gqa_kernels<float>
        {
            // ----------------------------------------------------------------
            // GQA-specific: permute / expand / reduce
            // ----------------------------------------------------------------

            static void permute_qkv(
                float* Q, float* K, float* V, const float* X,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                cuda_gqa_permute_qkv_fp32( Q, K, V, X, B, T, NH, NKV, HS, s );
            }

            static void permute_qkv_padded(
                float* Q, float* K, float* V, const float* X,
                int B, int input_T, int output_T, int NH, int NKV, int HS, cudaStream_t s )
            {
                cuda_gqa_permute_qkv_padded_fp32( Q, K, V, X, B, input_T, output_T, NH, NKV, HS, s );
            }

            static void permute_qkv_decode(
                float* Q, float* K, float* V, const float* X,
                int B, int position, int cache_T, int NH, int NKV, int HS, cudaStream_t s )
            {
                cuda_gqa_permute_qkv_decode_fp32( Q, K, V, X, B, position, cache_T, NH, NKV, HS, s );
            }

            static void expand_kv(
                float* k_exp, float* v_exp,
                const float* k_compact, const float* v_compact,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                cuda_gqa_expand_kv_fp32( k_exp, v_exp, k_compact, v_compact, B, T, NH, NKV, HS, s );
            }

            static void reduce_kv_grad(
                float* dk_compact, float* dv_compact,
                const float* dk_exp, const float* dv_exp,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                cuda_gqa_reduce_kv_grad_fp32( dk_compact, dv_compact, dk_exp, dv_exp, B, T, NH, NKV, HS, s );
            }

            static void permute_backward(
                float* dX,
                const float* dQ, const float* dK, const float* dV,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                cuda_gqa_permute_backward_fp32( dX, dQ, dK, dV, B, T, NH, NKV, HS, s );
            }

            // ----------------------------------------------------------------
            // Common: softmax
            // ----------------------------------------------------------------

            static void softmax_forward(
                float* att, float scale, const float* preatt,
                int B, int NH, int T, cudaStream_t s )
            {
                Attention::Common::cuda_attention_softmax_forward_fp32(
                    att, scale, preatt, B, NH, T, s );
            }

            static void softmax_padded_forward(
                float* att, float scale, const float* preatt,
                int B, int NH, int max_T, int actual_T, cudaStream_t s )
            {
                Attention::Common::cuda_attention_softmax_padded_forward_fp32(
                    att, scale, preatt, B, NH, max_T, actual_T, s );
            }

            static void softmax_decode_forward(
                float* att, float scale, const float* preatt,
                int B, int NH, int max_len, int actual_len, cudaStream_t s )
            {
                Attention::Common::cuda_attention_softmax_decode_forward_fp32(
                    att, scale, preatt, B, NH, max_len, actual_len, s );
            }

            static void softmax_backward(
                float* dpreatt, const float* datt, const float* att,
                float scale, int B, int NH, int T, cudaStream_t s )
            {
                Attention::Common::cuda_attention_softmax_backward_fp32(
                    dpreatt, datt, att, scale, B, NH, T, s );
            }

            // ----------------------------------------------------------------
            // Common: unpermute
            // ----------------------------------------------------------------

            static void unpermute_output(
                const float* vaccum, float* out,
                int B, int T, int NH, int HS, cudaStream_t s )
            {
                Attention::Common::cuda_attention_unpermute_output_fp32(
                    vaccum, out, B, T, NH, HS, s );
            }

            static void unpermute_output_padded(
                const float* vaccum, float* out,
                int B, int actual_T, int padded_T, int NH, int HS, cudaStream_t s )
            {
                Attention::Common::cuda_attention_unpermute_output_padded_fp32(
                    vaccum, out, B, actual_T, padded_T, NH, HS, s );
            }

            static void unpermute_backward(
                float* dvaccum, const float* dout,
                int B, int T, int NH, int HS, cudaStream_t s )
            {
                Attention::Common::cuda_attention_unpermute_backward_fp32(
                    dvaccum, dout, B, T, NH, HS, s );
            }
        };

        // ====================================================================
        // FP16 specialization
        // ====================================================================

        template<>
        struct cuda_gqa_kernels<half>
        {
            // ----------------------------------------------------------------
            // GQA-specific: permute / expand / reduce
            // ----------------------------------------------------------------

            static void permute_qkv(
                half* Q, half* K, half* V, const half* X,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                cuda_gqa_permute_qkv_fp16( Q, K, V, X, B, T, NH, NKV, HS, s );
            }

            static void permute_qkv_padded(
                half* Q, half* K, half* V, const half* X,
                int B, int input_T, int output_T, int NH, int NKV, int HS, cudaStream_t s )
            {
                cuda_gqa_permute_qkv_padded_fp16( Q, K, V, X, B, input_T, output_T, NH, NKV, HS, s );
            }

            static void permute_qkv_decode(
                half* Q, half* K, half* V, const half* X,
                int B, int position, int cache_T, int NH, int NKV, int HS, cudaStream_t s )
            {
                cuda_gqa_permute_qkv_decode_fp16( Q, K, V, X, B, position, cache_T, NH, NKV, HS, s );
            }

            static void expand_kv(
                half* k_exp, half* v_exp,
                const half* k_compact, const half* v_compact,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                cuda_gqa_expand_kv_fp16( k_exp, v_exp, k_compact, v_compact, B, T, NH, NKV, HS, s );
            }

            static void reduce_kv_grad(
                half* dk_compact, half* dv_compact,
                const half* dk_exp, const half* dv_exp,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                cuda_gqa_reduce_kv_grad_fp16( dk_compact, dv_compact, dk_exp, dv_exp, B, T, NH, NKV, HS, s );
            }

            static void permute_backward(
                half* dX,
                const half* dQ, const half* dK, const half* dV,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                cuda_gqa_permute_backward_fp16( dX, dQ, dK, dV, B, T, NH, NKV, HS, s );
            }

            // ----------------------------------------------------------------
            // Common: softmax
            // ----------------------------------------------------------------

            static void softmax_forward(
                half* att, float scale, const half* preatt,
                int B, int NH, int T, cudaStream_t s )
            {
                Attention::Common::cuda_attention_softmax_forward_fp16(
                    att, scale, preatt, B, NH, T, s );
            }

            static void softmax_padded_forward(
                half* att, float scale, const half* preatt,
                int B, int NH, int max_T, int actual_T, cudaStream_t s )
            {
                Attention::Common::cuda_attention_softmax_padded_forward_fp16(
                    att, scale, preatt, B, NH, max_T, actual_T, s );
            }

            static void softmax_decode_forward(
                half* att, float scale, const half* preatt,
                int B, int NH, int max_len, int actual_len, cudaStream_t s )
            {
                Attention::Common::cuda_attention_softmax_decode_forward_fp16(
                    att, scale, preatt, B, NH, max_len, actual_len, s );
            }

            static void softmax_backward(
                half* dpreatt, const half* datt, const half* att,
                float scale, int B, int NH, int T, cudaStream_t s )
            {
                Attention::Common::cuda_attention_softmax_backward_fp16(
                    dpreatt, datt, att, scale, B, NH, T, s );
            }

            // ----------------------------------------------------------------
            // Common: unpermute
            // ----------------------------------------------------------------

            static void unpermute_output(
                const half* vaccum, half* out,
                int B, int T, int NH, int HS, cudaStream_t s )
            {
                Attention::Common::cuda_attention_unpermute_output_fp16(
                    vaccum, out, B, T, NH, HS, s );
            }

            static void unpermute_output_padded(
                const half* vaccum, half* out,
                int B, int actual_T, int padded_T, int NH, int HS, cudaStream_t s )
            {
                Attention::Common::cuda_attention_unpermute_output_padded_fp16(
                    vaccum, out, B, actual_T, padded_T, NH, HS, s );
            }

            static void unpermute_backward(
                half* dvaccum, const half* dout,
                int B, int T, int NH, int HS, cudaStream_t s )
            {
                Attention::Common::cuda_attention_unpermute_backward_fp16(
                    dvaccum, dout, B, T, NH, HS, s );
            }
        };

    } // namespace Detail
} // namespace Mila::Dnn::Compute::Cuda::GroupedQueryAttention
