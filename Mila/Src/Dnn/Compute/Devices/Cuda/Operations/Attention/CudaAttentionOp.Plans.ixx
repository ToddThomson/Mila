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
#include "Kernels/CudaAttention.cuh"

export module Compute.CudaAttentionOp:Plans;

import Compute.CublasLtPlan;
import Utils.Logger;

namespace Mila::Dnn::Compute::Cuda::Attention
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute::Cuda;

    namespace Detail
    {
        /**
         * @brief cuBLASLt matmul execution plan for attention operations.
         */
        template <typename TNative>
        using CublasLtMatMulPlan = CublasLtMatMulPlan<TNative>;

        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_qk_decode_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int max_seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(max_seq_length) * static_cast<long long>(head_size);
            const long long strideB = static_cast<long long>(max_seq_length) * static_cast<long long>(head_size);
            const long long strideC = static_cast<long long>(max_seq_length);

            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ 1, /*A_cols=*/ head_size, /*ldA=*/ head_size, /*strideA=*/ strideA,
                /*B_rows=*/ max_seq_length, /*B_cols=*/ head_size, /*ldB=*/ head_size, /*strideB=*/ strideB,
                /*C_rows=*/ 1, /*C_cols=*/ max_seq_length, /*ldC=*/ max_seq_length, /*strideC=*/ strideC,
                /*opA=*/ CUBLAS_OP_N, /*opB=*/ CUBLAS_OP_T,
                batch_count,
                false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt QK decode plan built without algorithm (will use default)" );
            }

            return plan;
        }

        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_att_value_decode_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int max_seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            const int batch_count = batch_size * num_heads;

            const long long strideA = static_cast<long long>(max_seq_length);
            const long long strideB = static_cast<long long>(max_seq_length) * static_cast<long long>(head_size);
            const long long strideC = static_cast<long long>(head_size);

            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ 1, /*A_cols=*/ max_seq_length, /*ldA=*/ max_seq_length, /*strideA=*/ strideA,
                /*B_rows=*/ max_seq_length, /*B_cols=*/ head_size, /*ldB=*/ head_size, /*strideB=*/ strideB,
                /*C_rows=*/ 1, /*C_cols=*/ head_size, /*ldC=*/ head_size, /*strideC=*/ strideC,
                /*opA=*/ CUBLAS_OP_N, /*opB=*/ CUBLAS_OP_N,
                batch_count,
                false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt Att-Value decode plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for Q·K^T attention score computation (row-major).
         *
         * Row-major storage: Q[K] and K[K] are stored as [T, HS] (rows = sequence length, cols = head size).
         * Mathematical operation: preatt[T, T] = Q[T, HS] @ K^T[HS, T]
         */
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
            const int batch_count = batch_size * num_heads;

            // Row-major element stride (elements per head per batch): rows * cols = T * HS
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);

            // Row-major interpretation:
            // - A (Q): rows = T, cols = HS, ldA = HS
            // - B (K): rows = T, cols = HS, ldB = HS (we will use K^T via opB = CUBLAS_OP_T)
            // - C (preatt): rows = T, cols = T, ldC = T
            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ seq_length, /*A_cols=*/ head_size, /*ldA=*/ head_size, /*strideA=*/ strideA,
                /*B_rows=*/ seq_length, /*B_cols=*/ head_size, /*ldB=*/ head_size, /*strideB=*/ strideB,
                /*C_rows=*/ seq_length, /*C_cols=*/ seq_length, /*ldC=*/ seq_length, /*strideC=*/ strideC,
                /*opA=*/ CUBLAS_OP_N, /*opB=*/ CUBLAS_OP_T,   // Q @ K^T
                batch_count,
                false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt QK score plan built without algorithm (will use default)" );
            }

            return plan;
        }

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

            //
            // Row-major interpretation:
            // - A (att): rows = T, cols = T, ldA = T, strideA = T * T
            // - B (V)  : rows = T, cols = HS, ldB = HS, strideB = T * HS
            // - C (v_out): rows = T, cols = HS, ldC = HS, strideC = T * HS
            //
            // Mathematical operation: v_out[T, HS] = att[T, T] @ V[T, HS]
            //

            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length); // T * T
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS

            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ seq_length, /*A_cols=*/ seq_length, /*ldA=*/ seq_length, /*strideA=*/ strideA, // att: [T, T]
                /*B_rows=*/ seq_length, /*B_cols=*/ head_size,  /*ldB=*/ head_size,  /*strideB=*/ strideB, // V:   [T, HS]
                /*C_rows=*/ seq_length, /*C_cols=*/ head_size,  /*ldC=*/ head_size,  /*strideC=*/ strideC, // v_out:[T, HS]
                /*opA=*/ CUBLAS_OP_N, /*opB=*/ CUBLAS_OP_N, // v_out = A @ B
                batch_count,
                false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt Att-Value plan built without algorithm (will use default)" );
            }

            return plan;
        }

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

            //
            // Row-major interpretation (refactored):
            // - A (att):   rows = T, cols = T,  ldA = T,  strideA = T * T
            // - B (dVout): rows = T, cols = HS, ldB = HS, strideB = T * HS
            // - C (dV):    rows = T, cols = HS, ldC = HS, strideC = T * HS
            //
            // Mathematical operation (row-major): dV[T,HS] = Att^T[T,T] @ dVout[T,HS]
            // Map to cuBLAS op flags: opA = CUBLAS_OP_T, opB = CUBLAS_OP_N
            //

            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length); // T * T
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS

            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ seq_length, /*A_cols=*/ seq_length, /*ldA=*/ seq_length, /*strideA=*/ strideA, // att: [T, T]
                /*B_rows=*/ seq_length, /*B_cols=*/ head_size,  /*ldB=*/ head_size,  /*strideB=*/ strideB, // dVout: [T, HS]
                /*C_rows=*/ seq_length, /*C_cols=*/ head_size,  /*ldC=*/ head_size,  /*strideC=*/ strideC, // dV: [T, HS]
                /*opA=*/ CUBLAS_OP_T, /*opB=*/ CUBLAS_OP_N, // dV = Att^T @ dVout
                batch_count,
                false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt backward dV plan built without algorithm (will use default)" );
            }

            return plan;
        }

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

            //
            // Row-major interpretation (refactored):
            // - A (dVout): rows = T, cols = HS, ldA = HS, strideA = T * HS
            // - B (V)    : rows = T, cols = HS, ldB = HS, strideB = T * HS
            // - C (dAtt) : rows = T, cols = T,  ldC = T,  strideC = T * T
            //
            // Mathematical operation: dAtt[T, T] = dVout[T, HS] @ V[T, HS]^T
            // cuBLAS mapping: opA = CUBLAS_OP_N, opB = CUBLAS_OP_T
            //

            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(seq_length); // T * T

            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ seq_length, /*A_cols=*/ head_size, /*ldA=*/ head_size, /*strideA=*/ strideA, // dVout: [T, HS]
                /*B_rows=*/ seq_length, /*B_cols=*/ head_size, /*ldB=*/ head_size, /*strideB=*/ strideB, // V:     [T, HS]
                /*C_rows=*/ seq_length, /*C_cols=*/ seq_length, /*ldC=*/ seq_length, /*strideC=*/ strideC, // dAtt:  [T, T]
                /*opA=*/ CUBLAS_OP_N, /*opB=*/ CUBLAS_OP_T, // dAtt = A @ B^T
                batch_count,
                false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt backward dAtt plan built without algorithm (will use default)" );
            }

            return plan;
        }

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

            // Row-major interpretation:
            // - A (dPreatt): rows = T, cols = T, ldA = T, strideA = T * T
            // - B (K):       rows = T, cols = HS, ldB = HS, strideB = T * HS
            // - C (dQ):      rows = T, cols = HS, ldC = HS, strideC = T * HS
            //
            // Mathematical operation (row-major): dQ[T,HS] = dPreatt[T,T] @ K[T,HS]
            // cuBLAS mapping: opA = CUBLAS_OP_N, opB = CUBLAS_OP_N

            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length); // T * T
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size);  // T * HS
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size);  // T * HS

            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ seq_length, /*A_cols=*/ seq_length, /*ldA=*/ seq_length, /*strideA=*/ strideA, // dPreatt: [T, T]
                /*B_rows=*/ seq_length, /*B_cols=*/ head_size,  /*ldB=*/ head_size,  /*strideB=*/ strideB, // K/q: [T, HS]
                /*C_rows=*/ seq_length, /*C_cols=*/ head_size,  /*ldC=*/ head_size,  /*strideC=*/ strideC, // dQ: [T, HS]
                /*opA=*/ CUBLAS_OP_N, /*opB=*/ CUBLAS_OP_N, // C = A @ B
                batch_count,
                false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt backward dQ plan built without algorithm (will use default)" );
            }

            return plan;
        }

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

            //
            // Row-major interpretation (refactored):
            // - A (dPreatt): rows = T, cols = T, ldA = T, strideA = T * T
            // - B (Q)      : rows = T, cols = HS, ldB = HS, strideB = T * HS
            // - C (dK)     : rows = T, cols = HS, ldC = HS, strideC = T * HS
            //
            // Mathematical operation: dK[T, HS] = dPreatt^T[T, T] @ Q[T, HS]
            // cuBLAS mapping: opA = CUBLAS_OP_T, opB = CUBLAS_OP_N
            //

            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length); // T * T
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS

            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ seq_length, /*A_cols=*/ seq_length, /*ldA=*/ seq_length, /*strideA=*/ strideA, // dPreatt: [T, T]
                /*B_rows=*/ seq_length, /*B_cols=*/ head_size,  /*ldB=*/ head_size,  /*strideB=*/ strideB, // Q:      [T, HS]
                /*C_rows=*/ seq_length, /*C_cols=*/ head_size,  /*ldC=*/ head_size,  /*strideC=*/ strideC, // dK:     [T, HS]
                /*opA=*/ CUBLAS_OP_T, /*opB=*/ CUBLAS_OP_N, // dK = A^T @ B
                batch_count,
                false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt backward dK plan built without algorithm (will use default)" );
            }

            return plan;
        }
    }
}