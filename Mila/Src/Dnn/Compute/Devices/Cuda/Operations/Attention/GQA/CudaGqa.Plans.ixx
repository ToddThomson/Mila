/**
 * @file CudaGqaOp.Plans.ixx
 * @brief cuBLASLt matmul plan builders for the Grouped-Query Attention CUDA op.
 *
 * GQA plan construction mirrors MHA with one systematic difference: operations
 * that involve K or V use batch_count = B * NKV (not B * NH), while Q-only
 * operations keep batch_count = B * NH.
 *
 * Plan inventory
 * ──────────────
 * Forward prefill / training
 *   qk_score_plan_      : [B*NH,  1, T, HS] @ [B*NKV, T, HS]^T  → [B*NH,  T, T]
 *   att_value_plan_     : [B*NH,  T, T]     @ [B*NKV, T, HS]    → [B*NH,  T, HS]
 *
 * Forward decode (KV-cache)
 *   qk_decode_plan_     : [B*NH,  1, 1, HS] @ [B*NKV, T, HS]^T  → [B*NH,  1, T]
 *   att_value_decode_plan_: [B*NH, 1, T]    @ [B*NKV, T, HS]    → [B*NH,  1, HS]
 *
 * Backward (training only)
 *   backward_v_plan_    : [B*NKV, T, T]^T  @ [B*NH,  T, HS]    → [B*NKV, T, HS]  (dV)
 *   backward_att_plan_  : [B*NH,  T, HS]   @ [B*NKV, T, HS]^T  → [B*NH,  T, T]   (dAtt)
 *   backward_q_plan_    : [B*NH,  T, T]    @ [B*NKV, T, HS]    → [B*NH,  T, HS]  (dQ)
 *   backward_k_plan_    : [B*NH,  T, T]^T  @ [B*NH,  T, HS]    → [B*NKV, T, HS]  (dK)
 *
 * Note on KV-group broadcasting
 * ──────────────────────────────
 * cuBLASLt strided-batch gemm does not natively broadcast across batch
 * dimensions, so it cannot directly express the many-Q-to-one-KV grouping.
 * The approach taken here is to let the permute kernels (CudaGqa.cuh) expand
 * K and V into the full [B, NH, T, HS] layout before the matmuls, keeping
 * batch_count = B * NH throughout.  This trades a small memory overhead for
 * simplicity and maximum cuBLASLt throughput.
 *
 * For future optimisation, a custom grouped-batch kernel could avoid the
 * expansion entirely (see NVIDIA FasterTransformer / vLLM PagedAttention).
 */

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

export module Compute.CudaGroupedQueryAttentionOp:Plans;

import Compute.CublasLtPlan;
import Utils.Logger;

namespace Mila::Dnn::Compute::Cuda::GroupedQueryAttention
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute::Cuda;

    namespace Detail
    {
        template <typename TNative>
        using CublasLtMatMulPlan = CublasLtMatMulPlan<TNative>;

        // ====================================================================
        // Forward prefill / training plans
        // (After KV expansion: batch_count = B * NH for all plans)
        // ====================================================================

        /**
         * @brief Q @ K^T attention score plan (prefill / training).
         *
         * After KV expansion: K is [B, NH, T, HS] — same layout as MHA.
         * batch_count = B * NH.
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_qk_score_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,        // NH — total Q heads
            int seq_length,
            int head_size,
            cudaDataType_t       cuda_data_type,
            cublasComputeType_t  compute_type,
            cudaDataType_t       scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(seq_length) * head_size;   // Q
            const long long strideB = static_cast<long long>(seq_length) * head_size;   // K (expanded)
            const long long strideC = static_cast<long long>(seq_length) * seq_length;  // preatt

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, head_size, head_size, strideA,
                seq_length, head_size, head_size, strideB,
                seq_length, seq_length, seq_length, strideC,
                CUBLAS_OP_N, CUBLAS_OP_T,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Utils::Logger::warning(
                    "cuBLASLt GQA QK score plan built without algorithm (will use default)" );

            return plan;
        }

        /**
         * @brief Att @ V weighted-sum plan (prefill / training).
         *
         * After KV expansion: V is [B, NH, T, HS].
         * batch_count = B * NH.
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_att_value_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t       cuda_data_type,
            cublasComputeType_t  compute_type,
            cudaDataType_t       scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(seq_length) * seq_length;  // att [T,T]
            const long long strideB = static_cast<long long>(seq_length) * head_size;   // V   [T,HS]
            const long long strideC = static_cast<long long>(seq_length) * head_size;   // out [T,HS]

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,
                seq_length, head_size, head_size, strideB,
                seq_length, head_size, head_size, strideC,
                CUBLAS_OP_N, CUBLAS_OP_N,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Utils::Logger::warning(
                    "cuBLASLt GQA Att-Value plan built without algorithm (will use default)" );

            return plan;
        }

        // ====================================================================
        // Decode (KV-cache) plans
        // ====================================================================

        /**
         * @brief Single-token Q @ K^T decode plan.
         *
         * Q has one row (the current token); K is the full cached sequence.
         * After KV expansion: K is [B, NH, T, HS], batch_count = B * NH.
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_qk_decode_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int max_seq_length,
            int head_size,
            cudaDataType_t       cuda_data_type,
            cublasComputeType_t  compute_type,
            cudaDataType_t       scale_type )
        {
            const int batch_count = batch_size * num_heads;

            // Q slice: [1, HS] — single token; K cache: [T, HS]
            const long long strideA = static_cast<long long>(max_seq_length) * head_size; // full Q buf
            const long long strideB = static_cast<long long>(max_seq_length) * head_size; // K expanded
            const long long strideC = static_cast<long long>(max_seq_length);             // preatt [1,T]

            auto plan = build_strided_plan<TNative>(
                handle,
                1, head_size, head_size, strideA,
                max_seq_length, head_size, head_size, strideB,
                1, max_seq_length, max_seq_length, strideC,
                CUBLAS_OP_N, CUBLAS_OP_T,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Utils::Logger::warning(
                    "cuBLASLt GQA QK decode plan built without algorithm (will use default)" );

            return plan;
        }

        /**
         * @brief Single-token Att @ V decode plan.
         *
         * Att is [1, T] (softmax weights over cache); V is [T, HS].
         * batch_count = B * NH (after KV expansion).
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_att_value_decode_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int max_seq_length,
            int head_size,
            cudaDataType_t       cuda_data_type,
            cublasComputeType_t  compute_type,
            cudaDataType_t       scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(max_seq_length);            // att [1,T]
            const long long strideB = static_cast<long long>(max_seq_length) * head_size; // V  [T,HS]
            const long long strideC = static_cast<long long>(head_size);                 // out [1,HS]

            auto plan = build_strided_plan<TNative>(
                handle,
                1, max_seq_length, max_seq_length, strideA,
                max_seq_length, head_size, head_size, strideB,
                1, head_size, head_size, strideC,
                CUBLAS_OP_N, CUBLAS_OP_N,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Utils::Logger::warning(
                    "cuBLASLt GQA Att-Value decode plan built without algorithm (will use default)" );

            return plan;
        }

        // ====================================================================
        // Backward plans (training only)
        // All plans operate on expanded [B, NH, ...] layout.
        // ====================================================================

        /**
         * @brief dV = Att^T @ dVout.
         *
         * dV is accumulated across the Q group by the permute_backward kernel;
         * the cuBLASLt op itself works on the expanded NH dimension.
         * batch_count = B * NH.
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_backward_v_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t       cuda_data_type,
            cublasComputeType_t  compute_type,
            cudaDataType_t       scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(seq_length) * seq_length;  // att  [T,T]
            const long long strideB = static_cast<long long>(seq_length) * head_size;   // dVout[T,HS]
            const long long strideC = static_cast<long long>(seq_length) * head_size;   // dV   [T,HS]

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,
                seq_length, head_size, head_size, strideB,
                seq_length, head_size, head_size, strideC,
                CUBLAS_OP_T, CUBLAS_OP_N,   // dV = Att^T @ dVout
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Utils::Logger::warning(
                    "cuBLASLt GQA backward dV plan built without algorithm (will use default)" );

            return plan;
        }

        /**
         * @brief dAtt = dVout @ V^T.
         *
         * batch_count = B * NH.
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_backward_att_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t       cuda_data_type,
            cublasComputeType_t  compute_type,
            cudaDataType_t       scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(seq_length) * head_size;   // dVout[T,HS]
            const long long strideB = static_cast<long long>(seq_length) * head_size;   // V    [T,HS]
            const long long strideC = static_cast<long long>(seq_length) * seq_length;  // dAtt [T,T]

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, head_size, head_size, strideA,
                seq_length, head_size, head_size, strideB,
                seq_length, seq_length, seq_length, strideC,
                CUBLAS_OP_N, CUBLAS_OP_T,   // dAtt = dVout @ V^T
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Utils::Logger::warning(
                    "cuBLASLt GQA backward dAtt plan built without algorithm (will use default)" );

            return plan;
        }

        /**
         * @brief dQ = dPreatt @ K.
         *
         * K is the expanded KV buffer; batch_count = B * NH.
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_backward_q_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t       cuda_data_type,
            cublasComputeType_t  compute_type,
            cudaDataType_t       scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(seq_length) * seq_length;  // dPreatt[T,T]
            const long long strideB = static_cast<long long>(seq_length) * head_size;   // K      [T,HS]
            const long long strideC = static_cast<long long>(seq_length) * head_size;   // dQ     [T,HS]

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,
                seq_length, head_size, head_size, strideB,
                seq_length, head_size, head_size, strideC,
                CUBLAS_OP_N, CUBLAS_OP_N,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Utils::Logger::warning(
                    "cuBLASLt GQA backward dQ plan built without algorithm (will use default)" );

            return plan;
        }

        /**
         * @brief dK = dPreatt^T @ Q.
         *
         * dPreatt is [B, NH, T, T]; Q is [B, NH, T, HS].
         * The result dK is [B, NH, T, HS] in the expanded layout; the
         * permute_backward kernel subsequently reduces it to [B, NKV, T, HS].
         * batch_count = B * NH.
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_backward_k_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t       cuda_data_type,
            cublasComputeType_t  compute_type,
            cudaDataType_t       scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(seq_length) * seq_length;  // dPreatt[T,T]
            const long long strideB = static_cast<long long>(seq_length) * head_size;   // Q      [T,HS]
            const long long strideC = static_cast<long long>(seq_length) * head_size;   // dK_exp [T,HS]

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,
                seq_length, head_size, head_size, strideB,
                seq_length, head_size, head_size, strideC,
                CUBLAS_OP_T, CUBLAS_OP_N,   // dK = dPreatt^T @ Q
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Utils::Logger::warning(
                    "cuBLASLt GQA backward dK plan built without algorithm (will use default)" );

            return plan;
        }

    }
}
