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

export module Compute.CudaGqaOp:Dispatch;

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    namespace Detail
    {
        template<typename T>
        struct cuda_gqa_kernels;

        // ====================================================================
        // Live vs dormant kernel dispatchers
        //
        // CudaGqaOp's inference path (prefill/decode) calls only:
        //   kvcache_write_kv, permute_q_compact, prefill_softmax,
        //   prefill_unpermute_output_padded, softmax_decode_forward, unpermute_output.
        //
        // The remaining dispatchers (kvcache_write_q, kvcache_expand_kv, expand_kv,
        // permute_qkv*, reduce_kv_grad, permute_backward, softmax_forward/backward,
        // unpermute_*_padded/backward) are DORMANT -- no longer called after the
        // compact NKV-layout cleanup. They are retained, NOT deleted, as the
        // substrate for a future GQA training path (expanded-layout forward/backward
        // derived from a working MHA). This banner resolves the per-method "REVIEW:
        // legacy / cleanup analysis required" markers below: the disposition is
        // "retire in place as dormant training substrate." The FP32 dispatchers are
        // intact; several BF16 ones are stubs to restore when training is built.
        // See BACKLOG "Retire the CudaGqaOp legacy A/B path" and GqaMemory.md.
        // ====================================================================

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

            // TEMP: Optimized path -- compact Q permute (Phase 1).
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
                // REVIEW: legacy path for correctness validation. This kernel is no longer called.
                // Remove once permute_q_compact is validated ( which it has )
                // cuda_gqa_permute_q_fp32( Q, K, V, X, B, T, NH, NKV, HS, s );
                throw std::runtime_error( "permute_qkv is deprecated and should not be called." );
            }

            static void permute_kv(
                float* Q, float* K, float* V, const float* X,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                // REVIEW: legacy path for correctness validation. This kernel is no longer called.
                // cuda_gqa_permute_kv_fp32( Q, K, V, X, B, T, NH, NKV, HS, s );
                throw std::runtime_error( "permute_kv is deprecated and should not be called." );
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
                int B, int NH, int T_stride, int attended_len, int chunk_stride, int chunk_len, int position_offset,
                int window, cudaStream_t stream )
            {
                cuda_gqa_prefill_softmax_fp32(
                    att, preatt, B, NH, T_stride, attended_len, chunk_stride, chunk_len, position_offset, window, stream );
            }

            static void prefill_softmax_ring(
                float* att, const float* preatt,
                int B, int NH, int capacity, int chunk_len, int position_offset,
                int window, cudaStream_t stream )
            {
                cuda_gqa_prefill_softmax_ring_fp32(
                    att, preatt, B, NH, capacity, chunk_len, position_offset, window, stream );
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
                int B, int NH, int max_len, int actual_len, int window, cudaStream_t s )
            {
                Attention::Common::cuda_attention_softmax_decode_forward_fp32(
                    att, scale, preatt, B, NH, max_len, actual_len, s, window );
            }

            static void softmax_decode_ring_forward(
                float* att, float scale, const float* preatt,
                int B, int NH, int capacity, int actual_len, int window, cudaStream_t s )
            {
                Attention::Common::cuda_attention_softmax_decode_ring_forward_fp32(
                    att, scale, preatt, B, NH, capacity, actual_len, s, window );
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

            // TEMP: Optimized path -- compact Q permute (Phase 1).
            // Remove with legacy path once validated.
            static void permute_q_compact(
                nv_bfloat16* Q, const nv_bfloat16* X,
                int B, int chunk_len, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_gqa_permute_q_compact_bf16( Q, X, B, chunk_len, NH, HS, stream );
            }

            // ----------------------------------------------------------------
            // REVIEW: Reorganize/DEPRECATE:/RETIRE: after here ( analysis required )
            // ----------------------------------------------------------------

            static void permute_qkv(
                nv_bfloat16* Q, nv_bfloat16* K, nv_bfloat16* V, 
                const nv_bfloat16* X,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                // REVIEW: Cleanup analysis required
                // cuda_gqa_permute_qkv_fp16( Q, K, V, X, B, T, NH, NKV, HS, s );
            }

            static void permute_qkv_padded(
                nv_bfloat16* Q, nv_bfloat16* K, nv_bfloat16* V, const nv_bfloat16* X,
                int B, int input_T, int output_T, int NH, int NKV, int HS, cudaStream_t s )
            {
                // REVIEW: Cleanup analysis required
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
                // REVIEW: Cleanup analysis required
                // cuda_gqa_prefill_permute_qkv_fp16( Q, K, V, X, B, chunk_len, T, NH, NKV, HS, position_offset, stream );
            }

            static void expand_kv(
                nv_bfloat16* k_exp, nv_bfloat16* v_exp,
                const nv_bfloat16* k_compact, const nv_bfloat16* v_compact,
                int B, int T, int NH, int NKV, int HS, cudaStream_t stream )
            {
                // REVIEW: Cleanup analysis required
                // cuda_gqa_expand_kv_fp16( k_exp, v_exp, k_compact, v_compact, B, T, NH, NKV, HS, stream );
            }

            static void reduce_kv_grad(
                nv_bfloat16* dk_compact, nv_bfloat16* dv_compact,
                const nv_bfloat16* dk_exp, const nv_bfloat16* dv_exp,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                // REVIEW: This kernel has likely been deprecated with the optimized permute_q_compact path.
                // Needs triage to understand if this is still needed for correctness or if it can be removed.
                // cuda_gqa_reduce_kv_grad_fp16( dk_compact, dv_compact, dk_exp, dv_exp, B, T, NH, NKV, HS, s );
                throw std::runtime_error( "GQA reduce_kv_grad is not implemented for BF16. This likely needs triage to determine if it's still needed." );
            }

            static void permute_backward(
                nv_bfloat16* dX,
                const nv_bfloat16* dQ, const nv_bfloat16* dK, const nv_bfloat16* dV,
                int B, int T, int NH, int NKV, int HS, cudaStream_t s )
            {
                // REVIEW: This kernel has likely been deprecated with the optimized permute_q_compact path.
                // cuda_gqa_permute_backward_fp16( dX, dQ, dK, dV, B, T, NH, NKV, HS, s );
                throw std::runtime_error( "GQA permute_backward is not implemented for BF16. This likely needs triage to determine if it's still needed." );
            }

            // ----------------------------------------------------------------
            // GQA prefill
            // ----------------------------------------------------------------

            static void prefill_softmax(
                nv_bfloat16* att, const nv_bfloat16* preatt,
                int B_NH, int T, int T_stride, int attended_len, int chunk_stride, int chunk_len, int position_offset,
                int window, cudaStream_t stream )
            {
                cuda_gqa_prefill_softmax_bf16(
                    att, preatt,
                    B_NH, T, T_stride, attended_len, chunk_stride, chunk_len, position_offset, window,
                    stream );
            }

            static void prefill_softmax_ring(
                nv_bfloat16* att, const nv_bfloat16* preatt,
                int B, int NH, int capacity, int chunk_len, int position_offset,
                int window, cudaStream_t stream )
            {
                cuda_gqa_prefill_softmax_ring_bf16(
                    att, preatt, B, NH, capacity, chunk_len, position_offset, window, stream );
            }

            static void prefill_unpermute_output_padded(
                const nv_bfloat16* vaccum, nv_bfloat16* out,
                int B, int chunk_len, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_gqa_prefill_unpermute_output_padded_bf16( vaccum, out, B, chunk_len, T, NH, HS, stream );
            }

            // Fused FlashAttention prefill (Iteration 1). BF16-only; the caller gates on
            // NativeType == nv_bfloat16 so the FP32 specialization never needs this symbol.
            static void flash_prefill(
                const nv_bfloat16* Q, const nv_bfloat16* K, const nv_bfloat16* V,
                nv_bfloat16* Y,
                int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
                int position_offset, int window, float scale,
                cudaStream_t stream )
            {
                cuda_gqa_flash_prefill_bf16(
                    Q, K, V, Y, B, chunk_len, NH, NKV, HS, cache_capacity,
                    position_offset, window, scale, stream );
            }

            // Bounded sliding-window ring variant of flash_prefill (kBounded op). BF16-only,
            // same gating as flash_prefill.
            static void flash_prefill_ring(
                const nv_bfloat16* Q, const nv_bfloat16* K, const nv_bfloat16* V,
                nv_bfloat16* Y,
                int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
                int position_offset, int window, float scale,
                cudaStream_t stream )
            {
                cuda_gqa_flash_prefill_ring_bf16(
                    Q, K, V, Y, B, chunk_len, NH, NKV, HS, cache_capacity,
                    position_offset, window, scale, stream );
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
                int B, int NH, int max_len, int actual_len, int window, cudaStream_t stream )
            {
                Attention::Common::cuda_attention_softmax_decode_forward_bf16( att, scale, preatt, B, NH, max_len, actual_len, stream, window );
            }

            static void softmax_decode_ring_forward(
                nv_bfloat16* att, float scale, const nv_bfloat16* preatt,
                int B, int NH, int capacity, int actual_len, int window, cudaStream_t stream )
            {
                Attention::Common::cuda_attention_softmax_decode_ring_forward_bf16( att, scale, preatt, B, NH, capacity, actual_len, stream, window );
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
                // REVIEW:
                // Attention::Common::cuda_attention_unpermute_output_padded_bf16( vaccum, out, B, actual_T, padded_T, NH, HS, s );
                throw std::runtime_error( "GQA unpermute_output_padded is not implemented for BF16. This likely needs triage to determine if it's still needed." );
            }

            static void unpermute_backward(
                nv_bfloat16* dvaccum, const nv_bfloat16* dout,
                int B, int T, int NH, int HS, cudaStream_t s )
            {
                // REVIEW: 
                // Attention::Common::cuda_attention_unpermute_backward_fp16( dvaccum, dout, B, T, NH, HS, s );
                throw std::runtime_error( "GQA unpermute_backward is not implemented for BF16. This likely needs triage to determine if it's still needed." );
            }
        };

    }
}
