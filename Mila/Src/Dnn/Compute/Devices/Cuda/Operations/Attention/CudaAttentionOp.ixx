/**
 * @file CudaAttentionOp.ixx
 * @brief CUDA implementation of Multi-Head Attention with two-phase cuBLASLt optimization.
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
#include "Kernels/Attention.cuh"

export module Compute.CudaAttentionOp;

import Dnn.Components.Attention;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.Precision;
import Compute.OperationRegistry;
import Compute.Device;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;
import CublasLt.Error;
import CublasLtHelpers;
import Utils.Logger;

namespace Mila::Dnn::Compute::Cuda::Attention
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute::Cuda::Common::CublasLtHelpers;

    namespace Detail
    {
        /**
         * @brief cuBLASLt matmul execution plan for attention operations.
         *
         * Reuses the same RAII wrapper from CublasLtHelpers for consistency.
         * Plans are built once during build() and executed many times in forward/backward.
         */
        template <typename TNative>
        using CublasLtMatMulPlan = CublasLtMatMulPlan<TNative>;

        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_qk_score_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            CublasLtMatMulPlan<TNative> plan;

            const int batch_count = batch_size * num_heads;
            const int64_t strideQ = static_cast<int64_t>(seq_length) * head_size;
            const int64_t strideK = static_cast<int64_t>(seq_length) * head_size;
            const int64_t strideOut = static_cast<int64_t>(seq_length) * seq_length;

            cublasLtCheckStatus( cublasLtMatmulDescCreate( &plan.matmul_desc, compute_type, scale_type ) );

            cublasOperation_t opA = CUBLAS_OP_T;
            cublasOperation_t opB = CUBLAS_OP_N;

            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof( opA ) ) );
            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof( opB ) ) );

            // Q stored row-major [T, HS] -> cuBLAS sees col-major [HS, T]
            cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
                &plan.layoutA, cuda_data_type, head_size, seq_length, head_size ) );

            // K stored row-major [T, HS] -> cuBLAS sees col-major [HS, T]
            cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
                &plan.layoutB, cuda_data_type, head_size, seq_length, head_size ) );

            // Output preatt row-major [T, T] -> cuBLAS sees col-major [T, T]
            cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
                &plan.layoutC, cuda_data_type, seq_length, seq_length, seq_length ) );

            // Set batched attributes - stride in ELEMENTS (not bytes!)
            cublasLtCheckStatus( cublasLtMatrixLayoutSetAttribute(
                plan.layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof( batch_count ) ) );
            cublasLtCheckStatus( cublasLtMatrixLayoutSetAttribute(
                plan.layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideQ, sizeof( strideQ ) ) );

            cublasLtCheckStatus( cublasLtMatrixLayoutSetAttribute(
                plan.layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof( batch_count ) ) );
            cublasLtCheckStatus( cublasLtMatrixLayoutSetAttribute(
                plan.layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideK, sizeof( strideK ) ) );

            cublasLtCheckStatus( cublasLtMatrixLayoutSetAttribute(
                plan.layoutC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof( batch_count ) ) );
            cublasLtCheckStatus( cublasLtMatrixLayoutSetAttribute(
                plan.layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof( strideOut ) ) );

            cublasLtCheckStatus( cublasLtMatmulPreferenceCreate( &plan.preference ) );

            cublasLtMatmulHeuristicResult_t heuristic{};
            int returned_count = 0;

            cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
                handle, plan.matmul_desc,
                plan.layoutA, plan.layoutB, plan.layoutC, plan.layoutC,
                plan.preference, 1, &heuristic, &returned_count );

            if ( status == CUBLAS_STATUS_SUCCESS && returned_count > 0 )
            {
                plan.algorithm = heuristic.algo;
                plan.has_algorithm = true;
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt QK score plan: no algorithm found, using default" );
                plan.has_algorithm = false;
            }

            return plan;
        }


        /**
         * @brief Build cuBLASLt plan for Q·K^T attention score computation.
         *
         * Computes: preatt[B, NH, T, T] = Q[B, NH, T, HS] @ K^T[B, NH, HS, T]
         *
         * After permutation, Q and K are both [B, NH, T, HS] in memory.
         * We need to compute batched matmul: preatt = Q @ K^T for each (B, NH) pair.
         *
         * cuBLAS column-major interpretation:
         * - A (Q): [T × HS] per batch, ldA=T (leading dimension in elements)
         * - B (K): [HS × T] per batch, ldB=HS, will be transposed by opB=CUBLAS_OP_T
         * - C (preatt): [T × T] per batch, ldC=T
         *
         * Operation: C = A @ B^T
         *   [T × HS] @ [HS × T]^T = [T × HS] @ [T × HS]^T = [T × T]
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - strideA=T*HS elements, strideB=T*HS elements, strideC=T*T elements
         * - opA = CUBLAS_OP_N (no transpose Q)
         * - opB = CUBLAS_OP_T (transpose K)
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_qk_score_plan_using_helper(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            const int batch_count = batch_size * num_heads;
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);

            {
                std::ostringstream oss;
                oss << "build_qk_score_plan DEBUG: batch_size=" << batch_size
                    << " num_heads=" << num_heads
                    << " batch_count=" << batch_count
                    << " seq_length=" << seq_length
                    << " head_size=" << head_size
                    << " strideA_elems=" << strideA
                    << " strideB_elems=" << strideB
                    << " strideC_elems=" << strideC
                    << " opA=" << static_cast<int>(CUBLAS_OP_T)
                    << " opB=" << static_cast<int>(CUBLAS_OP_N);
                Utils::Logger::info( oss.str() );
            }

            /*auto plan = build_strided_plan<TNative>(
                handle,
                head_size, seq_length, head_size, strideA,
                head_size, seq_length, head_size, strideB,
                seq_length, seq_length, seq_length, strideC,
                CUBLAS_OP_T, CUBLAS_OP_N,
                compute_type,
                scale_type,
                cuda_data_type,
                batch_count,
                false );*/

            /*auto plan = build_strided_plan<TNative>(
                handle,
                head_size, seq_length, seq_length, strideA,
                head_size, seq_length, seq_length, strideB,
                seq_length, seq_length, seq_length, strideC,
                CUBLAS_OP_T, CUBLAS_OP_N,
                compute_type,
                scale_type,
                cuda_data_type,
                batch_count,
                false );*/

            // Row - major[ T, HS ] tensors appear as column - major[ HS, T ] to cuBLASLt
            // cuBLASLt: C = op(A) @ op(B) where A,B stored as [HS,T]
            // With opA=T: op(A) = [T,HS], with opB=N: op(B) = [HS,T]
            // Result: [T,HS] @ [HS,T] = [T,T] Good!
            auto plan = build_strided_plan<TNative>(
                handle,
                head_size, seq_length, head_size, strideA,   // A (Q): [HS × T], ldA=HS
                head_size, seq_length, head_size, strideB,   // B (K): [HS × T], ldB=HS  
                seq_length, seq_length, seq_length, strideC, // C (preatt): [T × T], ldC=T
                CUBLAS_OP_T, CUBLAS_OP_N,
                compute_type,
                scale_type,
                cuda_data_type,
                batch_count,
                false );

            if ( plan.isValid() && plan.has_algorithm )
            {
                //Utils::Logger::info( "cuBLASLt QK score plan built successfully" );
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt QK score plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for Att·V value computation.
         *
         * Computes: output[B, NH, T, HS] = V[B, NH, T, HS] @ Att[B, NH, T, T]
         *
         * In column-major cuBLAS notation: output = V @ Att
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A matrix (V): [HS × T] per batch, ldA=HS, strideA=T*HS elements
         * - B matrix (Att): [T × T] per batch, ldB=T, strideB=T*T elements
         * - C matrix (output): [HS × T] per batch, ldC=HS, strideC=T*HS elements
         * - opA = CUBLAS_OP_N
         * - opB = CUBLAS_OP_N
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_att_value_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            const int batch_count = batch_size * num_heads;
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size);

            auto plan = build_strided_plan<TNative>(
                handle,
                head_size, seq_length, head_size, strideA,
                seq_length, seq_length, seq_length, strideB,
                head_size, seq_length, head_size, strideC,
                CUBLAS_OP_N, CUBLAS_OP_N,
                compute_type,
                scale_type,
                cuda_data_type,
                batch_count,
                false );

            if ( plan.isValid() && plan.has_algorithm )
            {
                //Utils::Logger::info( "cuBLASLt Att-Value plan built successfully" );
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt Att-Value plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward dV computation.
         *
         * Computes: dV[B, NH, T, HS] = Att^T[B, NH, T, T] @ dvaccum[B, NH, T, HS]
         *
         * Memory layout (row-major storage):
         * - Att: [T, T] with stride HS between rows ? ldA = T
         * - dvaccum: [T, HS] with stride HS between rows ? ldB = HS
         * - dV: [T, HS] with stride HS between rows ? ldC = HS
         *
         * cuBLAS column-major interpretation:
         * - A (Att): rows=T, cols=T, ldA=T
         * - B (dvaccum): rows=HS, cols=T, ldB=HS
         * - C (dV): rows=HS, cols=T, ldC=HS
         * - opA = CUBLAS_OP_T, opB = CUBLAS_OP_N
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A matrix (Att): [T × T] per batch, ldA=T, strideA=T*T elements
         * - B matrix (dvaccum): [HS × T] per batch, ldB=HS, strideB=T*HS elements
         * - C matrix (dV): [HS × T] per batch, ldC=HS, strideC=T*HS elements
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_backward_v_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            const int batch_count = batch_size * num_heads;
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size);

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,
                seq_length, head_size, seq_length, strideB,
                seq_length, head_size, seq_length, strideC,
                CUBLAS_OP_T, CUBLAS_OP_N,
                compute_type,
                scale_type,
                cuda_data_type,
                batch_count,
                false );

            if ( plan.isValid() && plan.has_algorithm )
            {
                //Utils::Logger::info( "cuBLASLt backward dV plan built successfully" );
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt backward dV plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward dAtt computation.
         *
         * Computes: dAtt[B, NH, T, T] = dvaccum[B, NH, T, HS] @ V^T[B, NH, HS, T]
         *
         * Mathematical operation: dAtt[T, T] = dvaccum[T, HS] @ V^T[HS, T]
         *
         * In cuBLAS column-major (matrices stored transposed):
         * - dvaccum stored as [HS × T] ? interpret as [T × HS] with ldA=T when transposed
         * - V stored as [HS × T] ? interpret as [T × HS] with ldB=T when transposed
         * - dAtt stored as [T × T], ldC=T
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A matrix (dvaccum): [T × HS] per batch, ldA=T, strideA=T*HS elements (transposed view)
         * - B matrix (V): [HS × T] per batch, ldB=HS, strideB=T*HS elements (will be transposed)
         * - C matrix (dAtt): [T × T] per batch, ldC=T, strideC=T*T elements
         * - opA = CUBLAS_OP_T (transpose dvaccum for correct layout)
         * - opB = CUBLAS_OP_N (V is already in correct layout for transpose effect)
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_backward_att_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            const int batch_count = batch_size * num_heads;
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);

            // FIX: Change to CUBLAS_OP_N, CUBLAS_OP_T to transpose V instead of dvaccum
            // Mathematical operation: datt[T, T] = dvaccum[T, HS] @ V^T[HS, T]
            // Row-major storage -> column-major cuBLAS interpretation:
            // - A (dvaccum): [HS × T] (column-major view), opA=N -> no transpose
            // - B (V): [HS × T] (column-major view), opB=T -> transposed to [T × HS]
            // - C (datt): [T × T]
            /*auto plan = build_strided_plan<TNative>(
                handle,
                head_size, seq_length, head_size, strideA,
                head_size, seq_length, head_size, strideB,
                seq_length, seq_length, seq_length, strideC,
                CUBLAS_OP_T, CUBLAS_OP_N,
                compute_type,
                scale_type,
                cuda_data_type,
                batch_count,
                false );*/

            // 2nd try...

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, head_size, seq_length, strideA,  // A (dvaccum): [T × HS], ldA=T
                seq_length, head_size, seq_length, strideB,  // B (V): [T × HS], ldB=T
                seq_length, seq_length, seq_length, strideC, // C (datt): [T × T], ldC=T
                CUBLAS_OP_N, CUBLAS_OP_T,  // No transpose A, transpose B
                compute_type,
                scale_type,
                cuda_data_type,
                batch_count,
                false );

            if ( plan.isValid() && plan.has_algorithm )
            {
                //Utils::Logger::info( "cuBLASLt backward dAtt plan built successfully" );
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt backward dAtt plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward dQ computation.
         *
         * Computes: dQ[B, NH, T, HS] = dPreatt[B, NH, T, T] @ K[B, NH, T, HS]
         *
         * Mathematical operation: dQ[T, HS] = dPreatt[T, T] @ K[T, HS]
         *
         * In cuBLAS column-major (matrices stored transposed):
         * - dPreatt stored as [T × T], ldA=T
         * - K stored as [HS × T] ? interpret as [T × HS] with ldB=T when transposed
         * - dQ stored as [HS × T] ? result is [T × HS] with ldC=T when transposed
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A matrix (dPreatt): [T × T] per batch, ldA=T, strideA=T*T elements
         * - B matrix (K): [T × HS] per batch, ldB=T, strideB=T*HS elements (transposed view)
         * - C matrix (dQ): [T × HS] per batch, ldC=T, strideC=T*HS elements (transposed view)
         * - opA = CUBLAS_OP_N
         * - opB = CUBLAS_OP_T (transpose K for correct layout)
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_backward_q_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            const int batch_count = batch_size * num_heads;
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size);

            // TJT: This builds but does not compute proper results starting at offset 64
            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,
                head_size, seq_length, head_size, strideB,
                seq_length, head_size, seq_length, strideC,
                CUBLAS_OP_N, CUBLAS_OP_T,
                compute_type,
                scale_type,
                cuda_data_type,
                batch_count,
                false );

            if ( plan.isValid() && plan.has_algorithm )
            {
                //Utils::Logger::info( "cuBLASLt backward dQ plan built successfully" );
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt backward dQ plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward dK computation.
         *
         * Computes: dK[B, NH, T, HS] = dPreatt^T[B, NH, T, T] @ Q[B, NH, T, HS]
         *
         * Mathematical operation: dK[T, HS] = dPreatt^T[T, T] @ Q[T, HS]
         *
         * In cuBLAS column-major (matrices stored transposed):
         * - dPreatt stored as [T × T], ldA=T (will be transposed)
         * - Q stored as [HS × T] ? interpret as [T × HS] with ldB=T when transposed
         * - dK stored as [HS × T] ? result is [T × HS] with ldC=T when transposed
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A matrix (dPreatt): [T × T] per batch, ldA=T, strideA=T*T elements (will be transposed)
         * - B matrix (Q): [T × HS] per batch, ldB=T, strideB=T*HS elements (transposed view)
         * - C matrix (dK): [T × HS] per batch, ldC=T, strideC=T*HS elements (transposed view)
         * - opA = CUBLAS_OP_T (transpose dPreatt)
         * - opB = CUBLAS_OP_T (transpose Q for correct layout)
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_backward_k_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            const int batch_count = batch_size * num_heads;
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size);

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,
                seq_length, head_size, seq_length, strideB,
                seq_length, head_size, seq_length, strideC,
                CUBLAS_OP_T, CUBLAS_OP_N,
                compute_type,
                scale_type,
                cuda_data_type,
                batch_count,
                false );

            if ( plan.isValid() && plan.has_algorithm )
            {
                //Utils::Logger::info( "cuBLASLt backward dK plan built successfully" );
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt backward dK plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief CUDA kernel dispatcher for attention non-matmul operations.
         *
         * Handles permute, unpermute, softmax operations that cannot be done with cuBLASLt.
         */
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_mha_kernels;

        template <>
        struct cuda_mha_kernels<float>
        {
            cuda_mha_kernels() = default;

            static inline void permute_qkv(
                float* q, float* k, float* v,
                const float* inp,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_qkv_fp32( q, k, v, inp, B, T, NH, HS, stream );
            }

            static inline void unpermute_output(
                const float* vaccum, float* out,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_unpermute_output_fp32( vaccum, out, B, T, NH, HS, stream );
            }

            static inline void softmax_forward(
                float* att, float scale, const float* preatt,
                int B, int NH, int T,
                cudaStream_t stream )
            {
                cuda_softmax_forward_fp32( att, scale, preatt, B, NH, T, stream );
            }

            static inline void softmax_backward(
                float* dpreatt, const float* datt, const float* att,
                float scale,
                int B, int NH, int T,
                cudaStream_t stream )
            {
                cuda_softmax_backward_fp32( dpreatt, datt, att, scale, B, NH, T, stream );
            }

            static inline void permute_backward(
                float* dinp,
                const float* dq, const float* dk, const float* dv,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {

                cuda_permute_backward_fp32( dinp, dq, dk, dv, B, T, NH, HS, stream );
            }

            static inline void unpermute_backward(
                float* dvaccum, const float* dout,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_unpermute_backward_fp32( dvaccum, dout, B, T, NH, HS, stream );
            }
        };

        template <>
        struct cuda_mha_kernels<half>
        {
            cuda_mha_kernels() = default;

            static inline void permute_qkv(
                half* q, half* k, half* v,
                const half* inp,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_qkv_fp16( q, k, v, inp, B, T, NH, HS, stream );
            }

            static inline void unpermute_output(
                const half* vaccum, half* out,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_unpermute_output_fp16( vaccum, out, B, T, NH, HS, stream );
            }

            static inline void softmax_forward(
                half* att, float scale, const half* preatt,
                int B, int NH, int T,
                cudaStream_t stream )
            {
                cuda_softmax_forward_fp16( att, scale, preatt, B, NH, T, stream );
            }

            static inline void softmax_backward(
                half* dpreatt, const half* datt, const half* att,
                float scale,
                int B, int NH, int T,
                cudaStream_t stream )
            {
                cuda_softmax_backward_fp16( dpreatt, datt, att, scale, B, NH, T, stream );
            }

            static inline void permute_backward(
                half* dinp,
                const half* dq, const half* dk, const half* dv,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_backward_fp16( dinp, dq, dk, dv, B, T, NH, HS, stream );
            }

            static inline void unpermute_backward(
                half* dvaccum, const half* dout,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_unpermute_backward_fp16( dvaccum, dout, B, T, NH, HS, stream );
            }
        };
    }

    /**
     * @brief CUDA implementation of Multi-Head Attention using two-phase cuBLASLt optimization.
     *
     * Design philosophy (matches CudaLinearOp):
     * - Two-phase initialization: build() creates cuBLASLt plans, forward()/backward() execute them
     * - All dimension computation and algorithm selection happens once in build()
     * - Forward/backward are hot-path methods with zero setup overhead
     * - cuBLASLt plans cache descriptors, layouts, and optimal algorithms
     * - Custom CUDA kernels handle permute/unpermute and softmax operations
     *
     * Forward pass:
     *  1. Permute QKV from [B, T, 3*C] to separate Q, K, V [B, NH, T, HS]
     *  2. Compute attention scores: preatt = Q @ K^T
     *  3. Apply softmax with causal masking: att = softmax(preatt / sqrt(HS))
     *  4. Compute values: output = V @ att
     *  5. Unpermute output from [B, NH, T, HS] to [B, T, C]
     *
     * Backward pass:
     *  1. Unpermute output gradient
     *  2. Compute dV = Att^T @ dOut
     *  3. Compute dAtt = dOut @ V^T
     *  4. Softmax backward: dPreatt = softmax_backward(dAtt, Att)
     *  5. Compute dQ = dPreatt @ K
     *  6. Compute dK = dPreatt^T @ Q
     *  7. Permute gradients back to concatenated QKV format
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaAttentionOp : public UnaryOperation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = AttentionConfig;

        CudaAttentionOp( IExecutionContext* context, const AttentionConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaAttentionOp" ) ), config_( config )
        {
            config_.validate();
        }

        void setParameters( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {}

        void setGradients( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {}

        void build( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            batch_size_ = static_cast<int>(input_shape[ 0 ]);
            seq_length_ = static_cast<int>(input_shape[ 1 ]);
            qkv_dim_ = static_cast<int>(input_shape[ 2 ]);

            embedding_dim_ = qkv_dim_ / 3;
            num_heads_ = config_.getNumHeads();
            head_size_ = embedding_dim_ / num_heads_;

            if ( embedding_dim_ % num_heads_ != 0 )
            {
                throw std::invalid_argument(
                    "CudaAttentionOp: embedding_dim must be divisible by num_heads" );
            }

            allocateStateTensors();

            cublaslt_handle_ = context_->getCublasLtHandle();

            if ( cublaslt_handle_ == nullptr )
            {
                throw std::runtime_error(
                    "CudaAttentionOp requires cuBLASLt support. "
                    "Ensure CUDA 10.1 or newer is installed." );
            }

            precision_policy_ = config_.getPrecisionPolicy();

            buildCublasLtPlans();

            UnaryOperationBase::build( input_shape );
        }

        void forward( const ITensor& input, ITensor& output ) const override
        {
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;

            Detail::cuda_mha_kernels<NativeType>::permute_qkv(
                q_, k_, v_,
                X,
                batch_size_, seq_length_, num_heads_, head_size_,
                stream );

            // TJT: Verified. permute_qkv looks okay
            // DEBUG:
            //context_->synchronize();
            //{
            //    using HostTensorType = Tensor<TensorDataType::FP32, CpuMemoryResource>;

            //    HostTensorType host_q( Device::Cpu(), q_tensor_->shape() );
            //    host_q.setName( this->getName() + ".dbg.q.host_copy" );
            //    copy( *q_tensor_, host_q, context_ );

            //    HostTensorType host_k( Device::Cpu(), k_tensor_->shape() );
            //    host_k.setName( this->getName() + ".dbg.k.host_copy" );
            //    copy( *k_tensor_, host_k, context_ );

            //    // Zero check: count zeros in Q and K tensors
            //    const float* q_data = static_cast<const float*>(host_q.rawData());
            //    const float* k_data = static_cast<const float*>(host_k.rawData());
            //    const size_t total_elements = host_q.size();

            //    size_t q_zero_count = 0;
            //    size_t k_zero_count = 0;

            //    for ( size_t i = 0; i < total_elements; ++i )
            //    {
            //        if ( q_data[ i ] == 0.0f ) ++q_zero_count;
            //        if ( k_data[ i ] == 0.0f ) ++k_zero_count;
            //    }

            //    std::ostringstream oss;
            //    oss << "Q tensor: " << q_zero_count << " zeros out of " << total_elements
            //        << " elements (" << (100.0 * q_zero_count / total_elements) << "%)";
            //    Utils::Logger::info( oss.str() );

            //    oss.str( "" );
            //    oss << "K tensor: " << k_zero_count << " zeros out of " << total_elements
            //        << " elements (" << (100.0 * k_zero_count / total_elements) << "%)";
            //    Utils::Logger::info( oss.str() );

            //    // Only print full tensors if there's an unexpected number of zeros
            //    if ( q_zero_count > 0 || k_zero_count > 0 )
            //    {
            //        Utils::Logger::info( "Q and K tensors zeros found. Check QK matmul plan configuration." );
            //        Utils::Logger::info( this->getName() + ": dbg.q (host copy):\n" + host_q.toString( true ) );
            //        Utils::Logger::info( this->getName() + ": dbg.k (host copy):\n" + host_k.toString( true ) );
            //    }
            //    else
            //    {
            //        Utils::Logger::info( "Q and K tensors verified: no zeros found. QK matmul issue is in plan configuration." );
            //        Utils::Logger::info( this->getName() + ": dbg.q (host copy):\n" + host_q.toString( true ) );
            //        Utils::Logger::info( this->getName() + ": dbg.k (host copy):\n" + host_k.toString( true ) );
            //    }
            //}

            execute_plan<NativeType>(
                cublaslt_handle_,
                qk_score_plan_,
                &alpha,
                k_, q_, // swapped test fix
                &beta,
                preatt_,
                nullptr,
                stream );

            /*context_->synchronize();
            {
                using HostTensorType = Tensor<TensorDataType::FP32, CpuMemoryResource>;
                HostTensorType host_preatt( Device::Cpu(), preatt_tensor_->shape() );
                host_preatt.setName( this->getName() + ".dbg.preatt.host_copy" );
                copy( *preatt_tensor_, host_preatt );
                Utils::Logger::info( this->getName() + ": dbg.preatt (host copy):\n" + host_preatt.toString( true ) );
            }*/

            const float scale = 1.0f / sqrtf( static_cast<float>(head_size_) );

            Detail::cuda_mha_kernels<NativeType>::softmax_forward(
                att_, scale, preatt_,
                batch_size_, num_heads_, seq_length_,
                stream );

            /*context_->synchronize();
            {
                using HostTensorType = Tensor<TensorDataType::FP32, CpuMemoryResource>;
                HostTensorType host_att( Device::Cpu(), att_tensor_->shape() );
                host_att.setName( this->getName() + ".dbg.att.host_copy" );
                copy( *att_tensor_, host_att );
                Utils::Logger::info( this->getName() + ": dbg.att (host copy):\n" + host_att.toString( true ) );
            }*/

            execute_plan<NativeType>(
                cublaslt_handle_,
                att_value_plan_,
                &alpha,
                v_, att_,
                &beta,
                vaccum_,
                nullptr,
                stream );

            /*context_->synchronize();
            {
                using HostTensorType = Tensor<TensorDataType::FP32, CpuMemoryResource>;
                HostTensorType host_vaccum( Device::Cpu(), vaccum_tensor_->shape() );
                host_vaccum.setName( this->getName() + ".dbg.vaccum.host_copy" );
                copy( *vaccum_tensor_, host_vaccum );
                Utils::Logger::info( this->getName() + ": dbg.vaccum (host copy):\n" + host_vaccum.toString( true ) );
            }*/

            Detail::cuda_mha_kernels<NativeType>::unpermute_output(
                vaccum_, Y,
                batch_size_, seq_length_, num_heads_, head_size_,
                stream );

            /*context_->synchronize();
            {    
                using HostTensorType = Tensor<TensorDataType::FP32, CpuMemoryResource>;
                HostTensorType host_output( Device::Cpu(), output.shape() );
                host_output.setName( this->getName() + ".dbg.output.host_copy" );
                auto output_tensor = static_cast<const TensorType*>(&output);
                copy( *output_tensor, host_output );
                Utils::Logger::info( this->getName() + ": dbg.output (host copy):\n" + host_output.toString( true ) );
            }*/
        }

        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            assert( this->isBuilt() && "CudaAttentionOp must be built before calling backward()" );

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "CudaAttentionOp::backward called in inference mode" );
            }

            const NativeType* dY = static_cast<const NativeType*>(output_grad.rawData());
            NativeType* dX = static_cast<NativeType*>(input_grad.rawData());

            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(head_size_) );

            Detail::cuda_mha_kernels<NativeType>::unpermute_backward(
                dvaccum_, dY,
                batch_size_, seq_length_, num_heads_, head_size_,
                stream );

            using HostTensorType = Tensor<TensorDataType::FP32, CpuMemoryResource>;

            context_->synchronize();
            {
                
                HostTensorType host_dvaccum( Device::Cpu(), dvaccum_tensor_->shape() );
                host_dvaccum.setName( this->getName() + ".dbg.dvaccum.host_copy" );
                copy( *dvaccum_tensor_, host_dvaccum );
                Utils::Logger::info( this->getName() + ": dbg.dvaccum (host copy):\n" + host_dvaccum.toString( true ) );
            }

            context_->synchronize();
            {
                HostTensorType host_att( Device::Cpu(), att_tensor_->shape() );
                host_att.setName( this->getName() + ".dbg.att.host_copy" );
                copy( *att_tensor_, host_att );
                Utils::Logger::info( this->getName() + ": dbg.att (host copy):\n" + host_att.toString( true ) );
            }

            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_v_plan_,
                &alpha,
                att_, dvaccum_,
                &beta,
                dv_,
                nullptr,
                stream );

            context_->synchronize();
            {
                HostTensorType host_dv( Device::Cpu(), dv_tensor_->shape() );
                host_dv.setName( this->getName() + ".dbg.dv.host_copy" );
                copy( *dv_tensor_, host_dv );
                Utils::Logger::info( this->getName() + ": dbg.dv (host copy):\n" + host_dv.toString( true ) );
            }

            context_->synchronize();
            {
                HostTensorType host_v( Device::Cpu(), v_tensor_->shape() );
                host_v.setName( this->getName() + ".dbg.v.host_copy" );
                copy( *v_tensor_, host_v );
                Utils::Logger::info( this->getName() + ": dbg.v (host copy):\n" + host_v.toString( true ) );
            }

            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_att_plan_,
                &alpha,
                dvaccum_, v_,
                &beta,
                datt_,
                nullptr,
                stream );

            context_->synchronize();
            {
                HostTensorType host_datt( Device::Cpu(), datt_tensor_->shape() );
                host_datt.setName( this->getName() + ".dbg.datt.host_copy" );
                copy( *datt_tensor_, host_datt );
                Utils::Logger::info( this->getName() + ": dbg.datt (host copy):\n" + host_datt.toString( true ) );
            }

            Detail::cuda_mha_kernels<NativeType>::softmax_backward(
                dpreatt_, datt_, att_,
                scale,
                batch_size_, num_heads_, seq_length_,
                stream );

            context_->synchronize();
            {
                HostTensorType host_dpreatt( Device::Cpu(), dpreatt_tensor_->shape() );
                host_dpreatt.setName( this->getName() + ".dbg.dpreatt.host_copy" );
                copy( *dpreatt_tensor_, host_dpreatt );
                Utils::Logger::info( this->getName() + ": dbg.dpreatt (host copy):\n" + host_dpreatt.toString( true ) );
            }

            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_q_plan_,
                &alpha,
                k_, dpreatt_,
                &beta,
                dq_,
                nullptr,
                stream );

            context_->synchronize();
            {
                HostTensorType host_dq( Device::Cpu(), dq_tensor_->shape() );
                host_dq.setName( this->getName() + ".dbg.dq.host_copy" );
                copy( *dq_tensor_, host_dq );
                Utils::Logger::info( this->getName() + ": dbg.dq (host copy):\n" + host_dq.toString( true ) );
            }

            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_k_plan_,
                &alpha,
                dpreatt_, q_,
                &beta,
                dk_,
                nullptr,
                stream );

            context_->synchronize();
            {
                HostTensorType host_dk( Device::Cpu(), dk_tensor_->shape() );
                host_dk.setName( this->getName() + ".dbg.dk.host_copy" );
                copy( *dk_tensor_, host_dk );
                Utils::Logger::info( this->getName() + ": dbg.dk (host copy):\n" + host_dk.toString( true ) );
            }

            Detail::cuda_mha_kernels<NativeType>::permute_backward(
                dX,
                dq_, dk_, dv_,
                batch_size_, seq_length_, num_heads_, head_size_,
                stream );

            /*context_->synchronize();
            {
                HostTensorType host_dinput( Device::Cpu(), input_grad.shape() );
                host_dinput.setName( this->getName() + ".dbg.dinput.host_copy" );
                auto input_grad_tensor = static_cast<const TensorType*>(&input_grad);
                copy( *input_grad_tensor, host_dinput );
                Utils::Logger::info( this->getName() + ": dbg.dinput (host copy):\n" + host_dinput.toString( true ) );
            }*/
        }

        OperationType getOperationType() const override
        {
            return OperationType::AttentionOp;
        }

        std::string getName() const override
        {
            return "Cuda::AttentionOp";
        }

        const AttentionConfig& getConfig() const
        {
            return config_;
        }

    private:
        AttentionConfig config_;
        CudaExecutionContext* context_;

        int batch_size_{ 0 };
        int seq_length_{ 0 };
        int qkv_dim_{ 0 };
        int embedding_dim_{ 0 };
        int num_heads_{ 0 };
        int head_size_{ 0 };

        cublasLtHandle_t cublaslt_handle_{ nullptr };
        ComputePrecision::Policy precision_policy_;

        Detail::CublasLtMatMulPlan<NativeType> qk_score_plan_;
        Detail::CublasLtMatMulPlan<NativeType> att_value_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_v_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_att_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_q_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_k_plan_;

        std::shared_ptr<TensorType> q_tensor_;
        std::shared_ptr<TensorType> k_tensor_;
        std::shared_ptr<TensorType> v_tensor_;
        std::shared_ptr<TensorType> preatt_tensor_;
        std::shared_ptr<TensorType> att_tensor_;
        std::shared_ptr<TensorType> vaccum_tensor_;

        std::shared_ptr<TensorType> dq_tensor_;
        std::shared_ptr<TensorType> dk_tensor_;
        std::shared_ptr<TensorType> dv_tensor_;
        std::shared_ptr<TensorType> dpreatt_tensor_;
        std::shared_ptr<TensorType> datt_tensor_;
        std::shared_ptr<TensorType> dvaccum_tensor_;

        NativeType* q_{ nullptr };
        NativeType* k_{ nullptr };
        NativeType* v_{ nullptr };
        NativeType* preatt_{ nullptr };
        NativeType* att_{ nullptr };
        NativeType* vaccum_{ nullptr };

        NativeType* dq_{ nullptr };
        NativeType* dk_{ nullptr };
        NativeType* dv_{ nullptr };
        NativeType* dpreatt_{ nullptr };
        NativeType* datt_{ nullptr };
        NativeType* dvaccum_{ nullptr };

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 3 )
            {
                throw std::invalid_argument(
                    "CudaAttentionOp: input must have rank 3 (batch_size, seq_length, 3*embedding_dim)" );
            }

            const int64_t expected_qkv_dim = 3 * config_.getEmbeddingDim();

            if ( input_shape[ 2 ] != expected_qkv_dim )
            {
                throw std::invalid_argument(
                    "CudaAttentionOp: input last dimension must be 3*embedding_dim (Q, K, V concatenated)" );
            }
        }

        /**
         * @brief Allocate device tensors for intermediate state during forward/backward.
         *
         * Creates managed tensors for Q, K, V, attention scores, probabilities and gradients.
         * Ownership through shared_ptr ensures proper RAII semantics and automatic cleanup.
         * Raw pointers are cached for performance-critical forward/backward hot paths.
         */
        void allocateStateTensors()
        {
            auto device = context_->getDeviceId();

            shape_t qkv_shape = {
                static_cast<int64_t>(batch_size_),
                static_cast<int64_t>(num_heads_),
                static_cast<int64_t>(seq_length_),
                static_cast<int64_t>(head_size_)
            };

            q_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            q_tensor_->setName( "q_" );
            q_ = static_cast<NativeType*>(q_tensor_->rawData());

            k_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            k_tensor_->setName( "k_" );
            k_ = static_cast<NativeType*>(k_tensor_->rawData());

            v_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            v_tensor_->setName( "v_" );
            v_ = static_cast<NativeType*>(v_tensor_->rawData());

            shape_t att_shape = {
                static_cast<int64_t>(batch_size_),
                static_cast<int64_t>(num_heads_),
                static_cast<int64_t>(seq_length_),
                static_cast<int64_t>(seq_length_)
            };

            preatt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            preatt_tensor_->setName( "preatt_" );
            preatt_ = static_cast<NativeType*>(preatt_tensor_->rawData());

            att_tensor_ = std::make_shared<TensorType>( device, att_shape );
            att_tensor_->setName( "att_" );
            att_ = static_cast<NativeType*>(att_tensor_->rawData());

            vaccum_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            vaccum_tensor_->setName( "vaccum_" );
            vaccum_ = static_cast<NativeType*>(vaccum_tensor_->rawData());

            dq_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dq_tensor_->setName( "dq_" );
            dq_ = static_cast<NativeType*>(dq_tensor_->rawData());

            dk_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dk_tensor_->setName( "dk_" );
            dk_ = static_cast<NativeType*>(dk_tensor_->rawData());

            dv_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dv_tensor_->setName( "dv_" );
            dv_ = static_cast<NativeType*>(dv_tensor_->rawData());

            dpreatt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            dpreatt_tensor_->setName( "dpreatt_" );
            dpreatt_ = static_cast<NativeType*>(dpreatt_tensor_->rawData());

            datt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            datt_tensor_->setName( "datt_" );
            datt_ = static_cast<NativeType*>(datt_tensor_->rawData());

            dvaccum_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dvaccum_tensor_->setName( "dvaccum_" );
            dvaccum_ = static_cast<NativeType*>(dvaccum_tensor_->rawData());
        }

        void buildCublasLtPlans()
        {
            cudaDataType_t cuda_data_type = getCudaDataType();
            cublasComputeType_t compute_type;
            cudaDataType_t scale_type;

            getComputeTypes( compute_type, scale_type );

            qk_score_plan_ = Detail::build_qk_score_plan<NativeType>(
                cublaslt_handle_,
                batch_size_,
                num_heads_,
                seq_length_,
                head_size_,
                cuda_data_type,
                compute_type,
                scale_type );

            att_value_plan_ = Detail::build_att_value_plan<NativeType>(
                cublaslt_handle_,
                batch_size_,
                num_heads_,
                seq_length_,
                head_size_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_v_plan_ = Detail::build_backward_v_plan<NativeType>(
                cublaslt_handle_,
                batch_size_,
                num_heads_,
                seq_length_,
                head_size_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_att_plan_ = Detail::build_backward_att_plan<NativeType>(
                cublaslt_handle_,
                batch_size_,
                num_heads_,
                seq_length_,
                head_size_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_q_plan_ = Detail::build_backward_q_plan<NativeType>(
                cublaslt_handle_,
                batch_size_,
                num_heads_,
                seq_length_,
                head_size_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_k_plan_ = Detail::build_backward_k_plan<NativeType>(
                cublaslt_handle_,
                batch_size_,
                num_heads_,
                seq_length_,
                head_size_,
                cuda_data_type,
                compute_type,
                scale_type );
        }

        cudaDataType_t getCudaDataType() const
        {
            if constexpr ( std::is_same_v<NativeType, float> )
            {
                return CUDA_R_32F;
            }
            else if constexpr ( std::is_same_v<NativeType, half> )
            {
                return CUDA_R_16F;
            }
        }

        void getComputeTypes( cublasComputeType_t& compute_type, cudaDataType_t& scale_type ) const
        {
            scale_type = CUDA_R_32F;

            switch ( precision_policy_ )
            {
                case ComputePrecision::Policy::Native:
                case ComputePrecision::Policy::Accuracy:
                    if constexpr ( std::is_same_v<NativeType, half> )
                    {
                        compute_type = CUBLAS_COMPUTE_16F;
                    }
                    else
                    {
                        compute_type = CUBLAS_COMPUTE_32F;
                    }
                    break;

                case ComputePrecision::Policy::Performance:
                case ComputePrecision::Policy::Auto:
                default:
                    if constexpr ( std::is_same_v<NativeType, half> )
                    {
                        compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
                    }
                    else
                    {
                        compute_type = CUBLAS_COMPUTE_32F;
                    }
                    break;
            }
        }
    };

    export class CudaAttentionOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "AttentionOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::FP32>(
                opName,
                []( IExecutionContext* context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
                {
                    const auto& attentionConfig = static_cast<const AttentionConfig&>(config);
                    return std::make_shared<CudaAttentionOp<TensorDataType::FP32>>( context, attentionConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP16, TensorDataType::FP16>(
                opName,
                []( IExecutionContext* context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>>
                {
                    const auto& attentionConfig = static_cast<const AttentionConfig&>(config);
                    return std::make_shared<CudaAttentionOp<TensorDataType::FP16>>( context, attentionConfig );
                }
            );
        }
    };
}