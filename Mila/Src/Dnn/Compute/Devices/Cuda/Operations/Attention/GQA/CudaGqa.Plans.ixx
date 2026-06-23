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
 * Forward training (square, full T)
 *   qk_score_plan_         : [B*NH, T,     HS] @ [B*NH, T, HS]^T → [B*NH, T,     T]
 *   att_value_plan_        : [B*NH, T,     T]  @ [B*NH, T, HS]   → [B*NH, T,     HS]
 *
 * Forward prefill / chunked inference
 *   qk_prefill_plan_       : [B*NH, chunk, HS] @ [B*NH, T, HS]^T → [B*NH, chunk, T]
 *   att_value_prefill_plan_: [B*NH, chunk, T]  @ [B*NH, T, HS]   → [B*NH, chunk, HS]
 *
 * Forward decode (KV-cache)
 *   qk_decode_plan_        : [B*NH, 1,     HS] @ [B*NH, T, HS]^T → [B*NH, 1,     T]
 *   att_value_decode_plan_ : [B*NH, 1,     T]  @ [B*NH, T, HS]   → [B*NH, 1,     HS]
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
 *
 * Optimized NKV-layout plans (Phase 1)
 * ──────────────────────────────────────
 * The _optimized builders below use batch_count = B * NKV and fold the GS
 * Q heads into the M dimension, avoiding the expanded [B, NH, T, HS] buffers
 * entirely. K and V are read directly from their compact [B, NKV, T, HS] cache.
 *
 *   Q compact buffer        : [B, NH,  chunk, HS]  = [B*NKV, GS*chunk, HS]
 *   K/V cache (no expand)   : [B, NKV, T,     HS]  = [B*NKV, T,       HS]
 *   preatt / att            : [B, NH,  chunk, T]   = [B*NKV, GS*chunk, T]
 *   v_out                   : [B, NH,  chunk, HS]  = [B*NKV, GS*chunk, HS]
 *
 * Total shape of each output buffer is identical to the legacy equivalents,
 * so all downstream kernels (softmax, unpermute) are unchanged.
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

export module Compute.CudaGqaOp:Plans;

import Compute.CublasLtPlan;
import Logging.Logger;

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute::Cuda;

    namespace Detail
    {
        template <typename TNative>
        using CublasLtMatMulPlan = CublasLtMatMulPlan<TNative>;

        // ====================================================================
        // DORMANT — expanded [B,NH,T,HS]-layout plan builders.
        //
        // The live inference op (CudaGqaOp) builds only the _optimized (compact
        // NKV-layout) plans further below; the builders in this block are no
        // longer called. They are retained, NOT deleted, as the substrate for a
        // future GQA training path: the expanded-layout forward/backward derived
        // from a working MHA, and the clean expand_kv <-> reduce_kv_grad gradient
        // pairing makes the expanded layout the likely training vehicle. See the
        // BACKLOG "Retire the CudaGqaOp legacy A/B path" item and GqaMemory.md.
        // ====================================================================

        // ====================================================================
        // Forward training plans — square [T x T] attention matrices
        // ====================================================================

        /**
         * @brief Q @ K^T attention score plan (training, full sequence length).
         *
         * After KV expansion: K is [B, NH, T, HS] — same layout as MHA.
         * batch_count = B * NH.
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_qk_score_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t      cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t      scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(seq_length) * head_size;  // Q
            const long long strideB = static_cast<long long>(seq_length) * head_size;  // K (expanded)
            const long long strideC = static_cast<long long>(seq_length) * seq_length; // preatt

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, head_size,  head_size,  strideA,
                seq_length, head_size,  head_size,  strideB,
                seq_length, seq_length, seq_length, strideC,
                CUBLAS_OP_N, CUBLAS_OP_T,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Logging::Logger::warning(
                    "cuBLASLt GQA QK score plan built without algorithm (will use default)" );

            return plan;
        }

        /**
         * @brief Att @ V weighted-sum plan (training, full sequence length).
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
            cudaDataType_t      cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t      scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(seq_length) * seq_length; // att [T,T]
            const long long strideB = static_cast<long long>(seq_length) * head_size;  // V   [T,HS]
            const long long strideC = static_cast<long long>(seq_length) * head_size;  // out [T,HS]

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,
                seq_length, head_size,  head_size,  strideB,
                seq_length, head_size,  head_size,  strideC,
                CUBLAS_OP_N, CUBLAS_OP_N,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Logging::Logger::warning(
                    "cuBLASLt GQA Att-Value plan built without algorithm (will use default)" );

            return plan;
        }

        // ====================================================================
        // Forward prefill plans — rectangular [chunk x T] attention matrices
        //
        // Q has chunk_rows rows (kPrefillChunkSize); K and V cover the full
        // context [T, HS].  The A stride uses the full T-row Q-buffer offset
        // so that cuBLASLt advances correctly between (batch, head) slices.
        // ====================================================================

        /**
         * @brief Q @ K^T attention score plan for chunked prefill (inference).
         *
         * Computes preatt[B*NH, chunk_rows, max_seq_length] =
         *     Q[B*NH, chunk_rows, HS] @ K_exp^T[B*NH, max_seq_length, HS].
         *
         * strideA covers the full [T, HS] Q buffer so batch slices are spaced
         * correctly even though only chunk_rows rows are read per slice.
         *
         * @param chunk_rows     Number of Q-token rows (kPrefillChunkSize).
         * @param max_seq_length Full context / KV sequence length (T_).
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_qk_prefill_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int chunk_rows,
            int max_seq_length,
            int head_size,
            int prefill_window_size,
            cudaDataType_t      cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t      scale_type )
        {
            const int batch_count = batch_size * num_heads;

            // Q buffer is [B, NH, T, HS]; stride between slices is T*HS even
            // though only chunk_rows rows are read from each slice.
            const long long strideA = static_cast<long long>(max_seq_length) * head_size;
            const long long strideB = static_cast<long long>(max_seq_length) * head_size;
            const long long strideC = static_cast<long long>(chunk_rows) * max_seq_length;

            auto plan = build_strided_plan<TNative>(
                handle,
                chunk_rows,     head_size,       head_size,       strideA,
                max_seq_length, head_size,       head_size,       strideB,
                chunk_rows,     max_seq_length,  max_seq_length,  strideC,
                CUBLAS_OP_N, CUBLAS_OP_T,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Logging::Logger::warning(
                    "cuBLASLt GQA QK prefill plan built without algorithm (will use default)" );

            return plan;
        }

        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_att_value_prefill_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int chunk_rows,
            int max_seq_length,
            int head_size,
            int prefill_window_size,
            cudaDataType_t      cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t      scale_type )
        {
            const int batch_count = batch_size * num_heads;

            // Allocate buffers for prefill:
            //
            //const shape_t prefill_att_shape = { B_, NH_, kPrefillChunkSize, T_ };
            //const shape_t prefill_v_exp = { B_, NH_, T_, HS_ };
            //const shape_t prefill_vout_shape = { B_, NH_, kPrefillChunkSize, HS_ };

            // Row-major interpretation:
            // 
            // - A (att):   rows = chunk_rows, cols = T,  ldA = T,  strideA = Prefill_window_size * T
            // - B (v_exp): rows = T,          cols = HS, ldB = HS, strideB = T * HS
            // - C (v_out): rows = chunk_rows, cols = HS, ldC = HS, strideC = Prefill_window_size * HS
            //
            // Mathematical operation: v_out[chunk_rows, HS] = att[chunk_rows, T] @ v_exp[T, HS]
            //

            const long long strideA = static_cast<long long>(chunk_rows) * max_seq_length;
            const long long strideB = static_cast<long long>(max_seq_length) * head_size;
            const long long strideC = static_cast<long long>(chunk_rows) * head_size;

            auto plan = build_strided_plan<TNative>(
                handle,
                chunk_rows,     max_seq_length, max_seq_length, strideA,
                max_seq_length, head_size,      head_size,      strideB,
                chunk_rows,     head_size,      head_size,      strideC,
                CUBLAS_OP_N, CUBLAS_OP_N,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Logging::Logger::warning(
                    "cuBLASLt GQA Att-Value prefill plan built without algorithm (will use default)" );

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
            cudaDataType_t      cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t      scale_type )
        {
            const int batch_count = batch_size * num_heads;

            // Q slice: [1, HS] — single token; K cache: [T, HS]
            const long long strideA = static_cast<long long>(max_seq_length) * head_size;
            const long long strideB = static_cast<long long>(max_seq_length) * head_size;
            const long long strideC = static_cast<long long>(max_seq_length);

            auto plan = build_strided_plan<TNative>(
                handle,
                1,              head_size,       head_size,       strideA,
                max_seq_length, head_size,       head_size,       strideB,
                1,              max_seq_length,  max_seq_length,  strideC,
                CUBLAS_OP_N, CUBLAS_OP_T,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Logging::Logger::warning(
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
            cudaDataType_t      cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t      scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(max_seq_length);
            const long long strideB = static_cast<long long>(max_seq_length) * head_size;
            const long long strideC = static_cast<long long>(head_size);

            auto plan = build_strided_plan<TNative>(
                handle,
                1,              max_seq_length, max_seq_length, strideA,
                max_seq_length, head_size,      head_size,      strideB,
                1,              head_size,      head_size,      strideC,
                CUBLAS_OP_N, CUBLAS_OP_N,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Logging::Logger::warning(
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
            cudaDataType_t      cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t      scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(seq_length) * seq_length; // att  [T,T]
            const long long strideB = static_cast<long long>(seq_length) * head_size;  // dVout[T,HS]
            const long long strideC = static_cast<long long>(seq_length) * head_size;  // dV   [T,HS]

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,
                seq_length, head_size,  head_size,  strideB,
                seq_length, head_size,  head_size,  strideC,
                CUBLAS_OP_T, CUBLAS_OP_N,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Logging::Logger::warning(
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
            cudaDataType_t      cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t      scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(seq_length) * head_size;  // dVout[T,HS]
            const long long strideB = static_cast<long long>(seq_length) * head_size;  // V    [T,HS]
            const long long strideC = static_cast<long long>(seq_length) * seq_length; // dAtt [T,T]

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, head_size,  head_size,  strideA,
                seq_length, head_size,  head_size,  strideB,
                seq_length, seq_length, seq_length, strideC,
                CUBLAS_OP_N, CUBLAS_OP_T,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Logging::Logger::warning(
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
            cudaDataType_t      cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t      scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(seq_length) * seq_length; // dPreatt[T,T]
            const long long strideB = static_cast<long long>(seq_length) * head_size;  // K      [T,HS]
            const long long strideC = static_cast<long long>(seq_length) * head_size;  // dQ     [T,HS]

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,
                seq_length, head_size,  head_size,  strideB,
                seq_length, head_size,  head_size,  strideC,
                CUBLAS_OP_N, CUBLAS_OP_N,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Logging::Logger::warning(
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
            cudaDataType_t      cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t      scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(seq_length) * seq_length; // dPreatt[T,T]
            const long long strideB = static_cast<long long>(seq_length) * head_size;  // Q      [T,HS]
            const long long strideC = static_cast<long long>(seq_length) * head_size;  // dK_exp [T,HS]

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,
                seq_length, head_size,  head_size,  strideB,
                seq_length, head_size,  head_size,  strideC,
                CUBLAS_OP_T, CUBLAS_OP_N,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Logging::Logger::warning(
                    "cuBLASLt GQA backward dK plan built without algorithm (will use default)" );

            return plan;
        }

        // ====================================================================
        // Optimized NKV-layout plan builders (Phase 1)
        //
        // batch_count = B * NKV. The GS Q heads are folded into the M dimension
        // so K and V are read directly from their compact [B, NKV, T, HS] cache
        // without any expansion. Output buffer shapes are identical to the legacy
        // equivalents so all downstream kernels are unchanged.
        //
        // Geometry for Llama 3.2 3B: NH=24, NKV=8, GS=3, chunk=64, T=4096, HS=128
        //   batch_count = 1*8  = 8
        //   M (prefill) = 3*64 = 192  (GS * chunk)
        //   M (decode)  = 3*1  = 3    (GS * 1)
        //   K           = 128         (HS)
        //   N           = 4096        (T)
        //
        // LIVE: these compact-NKV builders are the only ones CudaGqaOp calls.
        // The expanded-layout builders above are dormant training substrate.
        // ====================================================================

        /**
         * @brief Q @ K^T prefill plan reading K directly from [B, NKV, T, HS] cache.
         *
         * Q compact buffer is [B, NH, chunk, HS] = [B*NKV, GS*chunk, HS].
         * K cache is          [B, NKV, T,     HS] = [B*NKV, T,       HS].
         * Output preatt is    [B, NH, chunk,  T]  = [B*NKV, GS*chunk, T].
         *
         * strideA = GS * chunk * HS  (compact Q head stride)
         * strideB = T * HS           (K cache head stride)
         * strideC = GS * chunk * T   (preatt head stride)
         *
         * @param group_size     NH / NKV — number of Q heads per KV head.
         * @param chunk_rows     Number of Q rows per chunk (kPrefillChunkSize for full chunks).
         * @param num_kv_heads   Number of KV heads (NKV).
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_qk_prefill_plan_optimized(
            cublasLtHandle_t handle,
            int batch_size,
            int num_kv_heads,
            int group_size,
            int chunk_rows,
            int max_seq_length,
            int head_size,
            cudaDataType_t      cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t      scale_type )
        {
            const int batch_count = batch_size * num_kv_heads;
            const int m_rows = group_size * chunk_rows;

            // Q compact: [B*NKV, GS*chunk, HS] — stride between KV-group slices
            const long long strideA = static_cast<long long>(m_rows) * head_size;
            // K cache:   [B*NKV, T, HS]
            const long long strideB = static_cast<long long>(max_seq_length) * head_size;
            // preatt:    [B*NKV, GS*chunk, T]
            const long long strideC = static_cast<long long>(m_rows) * max_seq_length;

            auto plan = build_strided_plan<TNative>(
                handle,
                m_rows,         head_size,      head_size,      strideA,
                max_seq_length, head_size,      head_size,      strideB,
                m_rows,         max_seq_length, max_seq_length, strideC,
                CUBLAS_OP_N, CUBLAS_OP_T,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Logging::Logger::warning(
                    "cuBLASLt GQA QK prefill optimized plan built without algorithm (will use default)" );

            return plan;
        }

        /**
         * @brief Att @ V prefill plan reading V directly from [B, NKV, T, HS] cache.
         *
         * att  is [B, NH, chunk, T]  = [B*NKV, GS*chunk, T].
         * V    is [B, NKV, T,    HS] = [B*NKV, T,        HS].
         * v_out is [B, NH, chunk, HS] = [B*NKV, GS*chunk, HS].
         *
         * strideA = GS * chunk * T   (att head stride)
         * strideB = T * HS           (V cache head stride)
         * strideC = GS * chunk * HS  (v_out head stride)
         *
         * @param group_size     NH / NKV.
         * @param chunk_rows     Number of Q rows per chunk (kPrefillChunkSize for full chunks).
         * @param num_kv_heads   Number of KV heads (NKV).
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_att_value_prefill_plan_optimized(
            cublasLtHandle_t handle,
            int batch_size,
            int num_kv_heads,
            int group_size,
            int chunk_rows,
            int max_seq_length,
            int head_size,
            cudaDataType_t      cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t      scale_type )
        {
            const int batch_count = batch_size * num_kv_heads;
            const int m_rows = group_size * chunk_rows;

            // att:   [B*NKV, GS*chunk, T]
            const long long strideA = static_cast<long long>(m_rows) * max_seq_length;
            // V cache: [B*NKV, T, HS]
            const long long strideB = static_cast<long long>(max_seq_length) * head_size;
            // v_out: [B*NKV, GS*chunk, HS]
            const long long strideC = static_cast<long long>(m_rows) * head_size;

            auto plan = build_strided_plan<TNative>(
                handle,
                m_rows,         max_seq_length, max_seq_length, strideA,
                max_seq_length, head_size,      head_size,      strideB,
                m_rows,         head_size,      head_size,      strideC,
                CUBLAS_OP_N, CUBLAS_OP_N,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Logging::Logger::warning(
                    "cuBLASLt GQA Att-Value prefill optimized plan built without algorithm (will use default)" );

            return plan;
        }

        /**
         * @brief Single-token Q @ K^T decode plan reading K from [B, NKV, T, HS] cache.
         *
         * Q compact is [B, NH, 1, HS]  = [B*NKV, GS, HS].
         * K cache is   [B, NKV, T, HS] = [B*NKV, T,  HS].
         * preatt_decode is [B, NH, 1, T] = [B*NKV, GS, T].
         *
         * strideA = GS * HS   (compact Q decode stride)
         * strideB = T * HS    (K cache head stride)
         * strideC = GS * T    (preatt_decode head stride)
         *
         * @param group_size     NH / NKV.
         * @param num_kv_heads   Number of KV heads (NKV).
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_qk_decode_plan_optimized(
            cublasLtHandle_t handle,
            int batch_size,
            int num_kv_heads,
            int group_size,
            int max_seq_length,
            int head_size,
            cudaDataType_t      cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t      scale_type )
        {
            const int batch_count = batch_size * num_kv_heads;
            const int m_rows = group_size; // GS * 1

            // Q compact decode: [B*NKV, GS, HS]
            const long long strideA = static_cast<long long>(m_rows) * head_size;
            // K cache: [B*NKV, T, HS]
            const long long strideB = static_cast<long long>(max_seq_length) * head_size;
            // preatt_decode: [B*NKV, GS, T]
            const long long strideC = static_cast<long long>(m_rows) * max_seq_length;

            auto plan = build_strided_plan<TNative>(
                handle,
                m_rows,         head_size,      head_size,      strideA,
                max_seq_length, head_size,      head_size,      strideB,
                m_rows,         max_seq_length, max_seq_length, strideC,
                CUBLAS_OP_N, CUBLAS_OP_T,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Logging::Logger::warning(
                    "cuBLASLt GQA QK decode optimized plan built without algorithm (will use default)" );

            return plan;
        }

        /**
         * @brief Single-token Att @ V decode plan reading V from [B, NKV, T, HS] cache.
         *
         * att_decode is [B, NH, 1, T]  = [B*NKV, GS, T].
         * V cache is   [B, NKV, T, HS] = [B*NKV, T,  HS].
         * v_out_decode is [B, NH, 1, HS] = [B*NKV, GS, HS].
         *
         * strideA = GS * T    (att_decode head stride)
         * strideB = T * HS    (V cache head stride)
         * strideC = GS * HS   (v_out_decode head stride)
         *
         * @param group_size     NH / NKV.
         * @param num_kv_heads   Number of KV heads (NKV).
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_att_value_decode_plan_optimized(
            cublasLtHandle_t handle,
            int batch_size,
            int num_kv_heads,
            int group_size,
            int max_seq_length,
            int head_size,
            cudaDataType_t      cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t      scale_type )
        {
            const int batch_count = batch_size * num_kv_heads;
            const int m_rows = group_size; // GS * 1

            // att_decode: [B*NKV, GS, T]
            const long long strideA = static_cast<long long>(m_rows) * max_seq_length;
            // V cache: [B*NKV, T, HS]
            const long long strideB = static_cast<long long>(max_seq_length) * head_size;
            // v_out_decode: [B*NKV, GS, HS]
            const long long strideC = static_cast<long long>(m_rows) * head_size;

            auto plan = build_strided_plan<TNative>(
                handle,
                m_rows,         max_seq_length, max_seq_length, strideA,
                max_seq_length, head_size,      head_size,      strideB,
                m_rows,         head_size,      head_size,      strideC,
                CUBLAS_OP_N, CUBLAS_OP_N,
                batch_count, false,
                compute_type, cuda_data_type, scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
                Logging::Logger::warning(
                    "cuBLASLt GQA Att-Value decode optimized plan built without algorithm (will use default)" );

            return plan;
        }

    }
}
