module;
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>

export module Compute.CublasLtAttention;

import Dnn.ITensor;
import Dnn.TensorDataTypeTraits;
import Compute.Precision;
import CublasLt.Error;
import Utils.Logger;

namespace Mila::Dnn::Compute
{
    export class CublasLtAttentionKernel
    {
    private:
        cublasLtHandle_t cublaslt_handle_;

        // Operation descriptors
        cublasLtMatmulDesc_t qk_matmul_desc_;
        cublasLtMatmulDesc_t av_matmul_desc_;

        // Matrix layout descriptors
        cublasLtMatrixLayout_t q_layout_;
        cublasLtMatrixLayout_t k_layout_;
        cublasLtMatrixLayout_t v_layout_;
        cublasLtMatrixLayout_t scores_layout_;
        cublasLtMatrixLayout_t output_layout_;

        // Best algorithms (cached after first run)
        cublasLtMatmulAlgo_t qk_algo_;
        cublasLtMatmulAlgo_t av_algo_;
        bool algos_selected_ = false;

        void* workspace_ = nullptr;
        size_t workspace_size_ = 32 * 1024 * 1024; // 32MB workspace

        void setupCublasLt()
        {
            cublasLtCreate( &cublaslt_handle_ );

            // Allocate workspace for algorithm selection
            cudaMalloc( &workspace_, workspace_size_ );

            // Create matmul descriptors
            cublasLtMatmulDescCreate( &qk_matmul_desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F );
            cublasLtMatmulDescCreate( &av_matmul_desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F );

            // Set transpose operations
            cublasOperation_t trans = CUBLAS_OP_T;
            cublasOperation_t no_trans = CUBLAS_OP_N;

            // QK^T needs K transposed
            cublasLtMatmulDescSetAttribute(
                qk_matmul_desc_,
                CUBLASLT_MATMUL_DESC_TRANSA,
                &no_trans,
                sizeof( no_trans )
            );
            cublasLtMatmulDescSetAttribute(
                qk_matmul_desc_,
                CUBLASLT_MATMUL_DESC_TRANSB,
                &trans,
                sizeof( trans )
            );
        }

        void buildMatrixLayouts()
        {
            const int64_t B = cached_batch_size_;
            const int64_t T = cached_seq_length_;
            const int64_t NH = cached_num_heads_;
            const int64_t hs = cached_head_size_;
            const int64_t C = cached_embedding_dim_;
            const int64_t C3 = cached_qkv_dim_;

            // Total batch count for strided batched GEMM
            const int64_t batch_count = B * NH;

            // ================================================================
            // Q layout: [B*NH, T, hs]
            // Q is strided in the input tensor: stride between heads = hs
            // stride between batches = T * C3
            // ================================================================
            int64_t q_rows = T;
            int64_t q_cols = hs;
            int64_t q_ld = C3; // Leading dimension in source tensor
            int64_t q_stride_batch = T * C3; // Stride between batches

            cublasLtMatrixLayoutCreate( &q_layout_, CUDA_R_32F, q_rows, q_cols, q_ld );
            cublasLtMatrixLayoutSetAttribute(
                q_layout_,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch_count,
                sizeof( batch_count )
            );
            cublasLtMatrixLayoutSetAttribute(
                q_layout_,
                CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &q_stride_batch,
                sizeof( q_stride_batch )
            );

            // ================================================================
            // K layout: [B*NH, hs, T] (will be transposed in matmul)
            // K starts at offset C in the input tensor
            // ================================================================
            int64_t k_rows = T; // Before transpose
            int64_t k_cols = hs;
            int64_t k_ld = C3;
            int64_t k_stride_batch = T * C3;

            cublasLtMatrixLayoutCreate( &k_layout_, CUDA_R_32F, k_rows, k_cols, k_ld );
            cublasLtMatrixLayoutSetAttribute(
                k_layout_,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch_count,
                sizeof( batch_count )
            );
            cublasLtMatrixLayoutSetAttribute(
                k_layout_,
                CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &k_stride_batch,
                sizeof( k_stride_batch )
            );

            // ================================================================
            // Scores layout: [B*NH, T, T]
            // ================================================================
            int64_t scores_rows = T;
            int64_t scores_cols = T;
            int64_t scores_ld = T;
            int64_t scores_stride_batch = T * T;

            cublasLtMatrixLayoutCreate( &scores_layout_, CUDA_R_32F,
                scores_rows, scores_cols, scores_ld );
            cublasLtMatrixLayoutSetAttribute(
                scores_layout_,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch_count,
                sizeof( batch_count )
            );
            cublasLtMatrixLayoutSetAttribute(
                scores_layout_,
                CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &scores_stride_batch,
                sizeof( scores_stride_batch )
            );

            // Similar setup for V and output layouts...
        }

        void selectBestAlgorithms( const float* q_ptr, const float* k_ptr,
            float* scores_ptr )
        {
            if (algos_selected_) return;

            cublasLtMatmulPreference_t preference;
            cublasLtMatmulPreferenceCreate( &preference );
            cublasLtMatmulPreferenceSetAttribute(
                preference,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_size_,
                sizeof( workspace_size_ )
            );

            // Get heuristic for QK^T matmul
            cublasLtMatmulHeuristicResult_t heuristic_result[1];
            int returned_results = 0;

            const float scale = 1.0f / std::sqrt( static_cast<float>(cached_head_size_) );
            const float beta = 0.0f;

            cublasLtMatmulAlgoGetHeuristic(
                cublaslt_handle_,
                qk_matmul_desc_,
                q_layout_,
                k_layout_,
                scores_layout_,
                scores_layout_,
                preference,
                1,
                heuristic_result,
                &returned_results
            );

            if (returned_results > 0)
            {
                qk_algo_ = heuristic_result[0].algo;
            }

            // Similar for attention×V matmul...

            algos_selected_ = true;
            cublasLtMatmulPreferenceDestroy( preference );
        }

    public:
        
        void forward( const ITensor& input, ITensor& output ) const override
        {
            const float* X = static_cast<const float*>(input.rawData());
            float* Y = static_cast<float*>(output.rawData());

            const int64_t B = cached_batch_size_;
            const int64_t T = cached_seq_length_;
            const int64_t C = cached_embedding_dim_;
            const int64_t C3 = cached_qkv_dim_;
            const int64_t NH = cached_num_heads_;
            const int64_t hs = cached_head_size_;

            float* preatt_data = preatt_cache_->data();
            float* att_data = att_cache_->data();

            const float scale = 1.0f / std::sqrt( static_cast<float>(hs) );
            const float beta = 0.0f;

            // ============================================================
            // Step 1: Compute QK^T for all heads using cuBLASLt
            // Q: [B*NH, T, hs], K^T: [B*NH, hs, T] -> Scores: [B*NH, T, T]
            // ============================================================

            // We need to launch NH separate matmuls per batch, or reorganize data
            // The challenge: Q, K, V are interleaved by head within each position
            // We need to either:
            // 1. Reorganize data first (transpose to [B, NH, T, hs])
            // 2. Use pointer array batched GEMM with manual pointer setup
            // 3. Use custom striding (complex with interleaved layout)

            // Most practical: Reorganize Q, K, V first
            // Launch kernel to extract and transpose: [B,T,3*C] -> [B,NH,T,hs] for each of Q,K,V

            reorganizeQKV << <... >> > (X, q_buffer_, k_buffer_, v_buffer_,
                B, T, NH, hs, C, C3);

            // Now Q, K, V are in [B, NH, T, hs] layout with stride NH*T*hs between batches
            cublasLtMatmul(
                cublaslt_handle_,
                qk_matmul_desc_,
                &scale,  // alpha = 1/sqrt(hs)
                q_buffer_,  // Q: [B*NH, T, hs]
                q_layout_,
                k_buffer_,  // K: [B*NH, hs, T] (transposed)
                k_layout_,
                &beta,
                preatt_data,  // Output: [B*NH, T, T]
                scores_layout_,
                preatt_data,
                scores_layout_,
                &qk_algo_,
                workspace_,
                workspace_size_,
                stream_
            );

            // ============================================================
            // Step 2: Apply causal mask + softmax (custom CUDA kernel)
            // ============================================================
            causalMaskedSoftmax << <... >> > (preatt_data, att_data, B, NH, T);

            // ============================================================
            // Step 3: Attention × V using cuBLASLt
            // Att: [B*NH, T, T], V: [B*NH, T, hs] -> Out: [B*NH, T, hs]
            // ============================================================
            cublasLtMatmul(
                cublaslt_handle_,
                av_matmul_desc_,
                &alpha_one,
                att_data,
                scores_layout_,
                v_buffer_,
                v_layout_,
                &beta,
                output_buffer_,
                output_layout_,
                output_buffer_,
                output_layout_,
                &av_algo_,
                workspace_,
                workspace_size_,
                stream_
            );

            // ============================================================
            // Step 4: Reorganize output back to [B, T, C]
            // ============================================================
            reorganizeOutput << <... >> > (output_buffer_, Y, B, T, NH, hs);
        }
    };
}