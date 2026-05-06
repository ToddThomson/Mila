module;
#include <cublasLt.h>
#include <cuda_bf16.h>
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
            // kvcache related
            // ----------------------------------------------------------------

            static void kvcache_write_q(
                float* Q, const float* Xq,
                int B, int chunk_len, int NH, int HS,
                int position_offset, int T, cudaStream_t stream )
            {
                cuda_gqa_kvcache_write_q_fp32(
                    Q, Xq, B, chunk_len, NH, HS, position_offset, T, stream );
            }

            static void kvcache_write_kv(
                float* K, float* V, const float* Xk, const float* Xv,
                int B, int chunk_len, int NKV, int HS,
                int position_offset, int T, cudaStream_t stream )
            {
                cuda_gqa_kvcache_write_kv_fp32(
                    K, V, Xk, Xv, B, chunk_len, NKV, HS, position_offset, T, stream );
            }

            static void kvcache_expand_kv(
                float* k_exp, float* v_exp,
                const float* k_compact, const float* v_compact,
                int B, int chunk_len, int T, int NH, int NKV, int HS, int position_offset, cudaStream_t stream )
            {
                cuda_gqa_kvcache_expand_kv_fp32( k_exp, v_exp, k_compact, v_compact, B, chunk_len, T, NH, NKV, HS, position_offset, stream );
            }

            // TEMP: Optimized path — compact Q permute (Phase 1).
            // Remove with legacy path once validated.
            static void permute_q_compact(
                float* Q, const float* X,
                int B, int chunk_len, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_gqa_permute_q_compact_fp32( Q, X, B, chunk_len, NH, HS, stream );
            }

            static void permute_qkv(
                float* Q, float* K, float* V, const float* X,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                // FIXME: cuda_gqa_permute_q_fp32( Q, K, V, X, B, T, NH, NKV, HS, s );
            }

            static void permute_kv(
                float* Q, float* K, float* V, const float* X,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                // FIXME: cuda_gqa_permute_kv_fp32( Q, K, V, X, B, T, NH, NKV, HS, s );
            }

            static void permute_qkv_padded(
                float* Q, float* K, float* V, const float* X,
                int B, int input_T, int output_T, int NH, int NKV, int HS, cudaStream_t stream )
            {
                cuda_gqa_permute_qkv_padded_fp32( Q, K, V, X, B, input_T, output_T, NH, NKV, HS, stream );
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
                int B, int T, int NH, int NKV, int HS, cudaStream_t stream )
            {
                cuda_gqa_expand_kv_fp32( k_exp, v_exp, k_compact, v_compact, B, T, NH, NKV, HS, stream );
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
            // GQA Softmax  - prefill
            // ----------------------------------------------------------------

            static void prefill_softmax(
                float* att, const float* preatt,
                int B, int NH, int T_stride, int chunk_stride, int chunk_len, int position_offset,
                cudaStream_t stream )
            {
                cuda_gqa_prefill_softmax_fp32(
                    att, preatt, B, NH, T_stride, chunk_stride, chunk_len, position_offset, stream );
            }

            /*static void prefill_permute_qkv(
                float* Q, float* K, float* V, const float* X,
                int B, int chunk_len, int T, int NH, int NKV, int HS, int position_offset,
                cudaStream_t stream )
            {
                cuda_gqa_prefill_permute_qkv_fp32( Q, K, V, X, B, chunk_len, T, NH, NKV, HS, position_offset, stream );
            }*/

            static void prefill_unpermute_output_padded(
                const float* vaccum, float* out,
                int B, int chunk_len, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_gqa_prefill_unpermute_output_padded_fp32( vaccum, out, B, chunk_len, T, NH, HS, stream );
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
        // BF16 specialization
        // ====================================================================

        template<>
        struct cuda_gqa_kernels<nv_bfloat16>
        {
            // ----------------------------------------------------------------
            // kvcache BF16 dispatchers
            // ----------------------------------------------------------------

            static void kvcache_write_q(
                nv_bfloat16* Q, const nv_bfloat16* Xq,
                int B, int chunk_len, int NH, int HS,
                int position_offset, int T, cudaStream_t stream )
            {
                cuda_gqa_kvcache_write_q_bf16( Q, Xq, B, chunk_len, NH, HS, position_offset, T, stream );
            }

            static void kvcache_write_kv(
                nv_bfloat16* K, nv_bfloat16* V, const nv_bfloat16* Xk, const nv_bfloat16* Xv,
                int B, int chunk_len, int NKV, int HS,
                int position_offset, int T, cudaStream_t stream )
            {
                cuda_gqa_kvcache_write_kv_bf16( K, V, Xk, Xv, B, chunk_len, NKV, HS, position_offset, T, stream );
            }

            static void kvcache_expand_kv(
                nv_bfloat16* k_exp, nv_bfloat16* v_exp,
                const nv_bfloat16* k_compact, const nv_bfloat16* v_compact,
                int B, int chunk_len, int T, int NH, int NKV, int HS, int position_offset, cudaStream_t stream )
            {
                cuda_gqa_kvcache_expand_kv_bf16( k_exp, v_exp, k_compact, v_compact, B, chunk_len, T, NH, NKV, HS, position_offset, stream );
            }

            // TEMP: Optimized path — compact Q permute (Phase 1).
            // Remove with legacy path once validated.
            static void permute_q_compact(
                nv_bfloat16* Q, const nv_bfloat16* X,
                int B, int chunk_len, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_gqa_permute_q_compact_bf16( Q, X, B, chunk_len, NH, HS, stream );
            }

            // ----------------------------------------------------------------
            // TODO: Reorganize after here
            // ----------------------------------------------------------------

            static void permute_qkv(
                nv_bfloat16* Q, nv_bfloat16* K, nv_bfloat16* V, 
                const nv_bfloat16* X,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                //cuda_gqa_permute_qkv_fp16( Q, K, V, X, B, T, NH, NKV, HS, s );
            }

            static void permute_qkv_padded(
                nv_bfloat16* Q, nv_bfloat16* K, nv_bfloat16* V, const nv_bfloat16* X,
                int B, int input_T, int output_T, int NH, int NKV, int HS, cudaStream_t s )
            {
                //cuda_gqa_permute_qkv_padded_fp16( Q, K, V, X, B, input_T, output_T, NH, NKV, HS, s );
            }

            static void permute_qkv_decode(
                half* Q, half* K, half* V, const half* X,
                int B, int position, int cache_T, int NH, int NKV, int HS, cudaStream_t s )
            {
                cuda_gqa_permute_qkv_decode_fp16( Q, K, V, X, B, position, cache_T, NH, NKV, HS, s );
            }

            static void prefill_permute_qkv(
                nv_bfloat16* Q, nv_bfloat16* K, nv_bfloat16* V, 
                const nv_bfloat16* X,
                int B, int chunk_len, int T, int NH, int NKV, int HS, int position_offset,
                cudaStream_t stream )
            {
                // TODO: cuda_gqa_prefill_permute_qkv_fp16( Q, K, V, X, B, chunk_len, T, NH, NKV, HS, position_offset, stream );
            }
            

            static void expand_kv(
                nv_bfloat16* k_exp, nv_bfloat16* v_exp,
                const nv_bfloat16* k_compact, const nv_bfloat16* v_compact,
                int B, int T, int NH, int NKV, int HS, cudaStream_t stream )
            {
                // TODO: cuda_gqa_expand_kv_fp16( k_exp, v_exp, k_compact, v_compact, B, T, NH, NKV, HS, stream );
            }

            static void reduce_kv_grad(
                nv_bfloat16* dk_compact, nv_bfloat16* dv_compact,
                const nv_bfloat16* dk_exp, const nv_bfloat16* dv_exp,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                // FIXME: cuda_gqa_reduce_kv_grad_fp16( dk_compact, dv_compact, dk_exp, dv_exp, B, T, NH, NKV, HS, s );
            }

            static void permute_backward(
                nv_bfloat16* dX,
                const nv_bfloat16* dQ, const nv_bfloat16* dK, const nv_bfloat16* dV,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                // FIXME: cuda_gqa_permute_backward_fp16( dX, dQ, dK, dV, B, T, NH, NKV, HS, s );
            }

            // ----------------------------------------------------------------
            // GQA prefill
            // ----------------------------------------------------------------

            static void prefill_softmax(
                nv_bfloat16* att, const nv_bfloat16* preatt,
                int B_NH, int T, int T_stride, int chunk_stride, int chunk_len, int position_offset,
                cudaStream_t stream )
            {
                cuda_gqa_prefill_softmax_bf16(
                    att, preatt, 
                    B_NH, T, T_stride, chunk_stride, chunk_len, position_offset, 
                    stream );
            }

            static void prefill_unpermute_output_padded(
                const nv_bfloat16* vaccum, nv_bfloat16* out,
                int B, int chunk_len, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_gqa_prefill_unpermute_output_padded_bf16( vaccum, out, B, chunk_len, T, NH, HS, stream );
            }

            // ----------------------------------------------------------------
            // Common: softmax
            // ----------------------------------------------------------------

            static void softmax_forward(
                nv_bfloat16* att, float scale, const nv_bfloat16* preatt,
                int B, int NH, int T, cudaStream_t stream )
            {
                Attention::Common::cuda_attention_softmax_forward_bf16( att, scale, preatt, B, NH, T, stream );
            }

            static void softmax_padded_forward(
                nv_bfloat16* att, float scale, const nv_bfloat16* preatt,
                int B, int NH, int max_T, int actual_T, cudaStream_t s )
            {
                Attention::Common::cuda_attention_softmax_padded_forward_bf16(
                    att, scale, preatt, B, NH, max_T, actual_T, s );
            }

            static void softmax_decode_forward(
                nv_bfloat16* att, float scale, const nv_bfloat16* preatt,
                int B, int NH, int max_len, int actual_len, cudaStream_t stream )
            {
                Attention::Common::cuda_attention_softmax_decode_forward_bf16( att, scale, preatt, B, NH, max_len, actual_len, stream );
            }

            static void softmax_backward(
                nv_bfloat16* dpreatt, const nv_bfloat16* datt, const nv_bfloat16* att,
                float scale, int B, int NH, int T, cudaStream_t s )
            {
                Attention::Common::cuda_attention_softmax_backward_bf16( dpreatt, datt, att, scale, B, NH, T, s );
            }

            // ----------------------------------------------------------------
            // Common: unpermute
            // ----------------------------------------------------------------

            static void unpermute_output(
                const nv_bfloat16* vaccum, nv_bfloat16* out,
                int B, int T, int NH, int HS, cudaStream_t s )
            {
                Attention::Common::cuda_attention_unpermute_output_bf16( vaccum, out, B, T, NH, HS, s );
            }

            static void unpermute_output_padded(
                const nv_bfloat16* vaccum, nv_bfloat16* out,
                int B, int actual_T, int padded_T, int NH, int HS, cudaStream_t s )
            {
                //Attention::Common::cuda_attention_unpermute_output_padded_bf16( vaccum, out, B, actual_T, padded_T, NH, HS, s );
            }

            static void unpermute_backward(
                nv_bfloat16* dvaccum, const nv_bfloat16* dout,
                int B, int T, int NH, int HS, cudaStream_t s )
            {
                // FIXME: Attention::Common::cuda_attention_unpermute_backward_fp16( dvaccum, dout, B, T, NH, HS, s );
            }
        };

    }
}
