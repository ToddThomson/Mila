/**
 * @file CudaLinearOp.Plans.ixx
 * @brief cuBLASLt plan builders for CudaLinearOp forward and backward passes.
 */

module;
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <format>
#include <stdexcept>
#include "Kernels/Linear.cuh"

export module Compute.CudaLinearOp:Plans;

import Compute.CublasLtPlan;
import Utils.Logger;

namespace Mila::Dnn::Compute::Cuda::Linear
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute::Cuda;

    namespace Detail
    {
        template <typename TNative>
        using CublasLtMatMulPlan = Mila::Dnn::Compute::Cuda::CublasLtMatMulPlan<TNative>;

        /**
         * @brief Build cuBLASLt plan for the Linear forward pass.
         *
         * Computes output[batch, out] = input[batch, in] @ weight^T[in, out]
         *
         * Uses CUBLASLT_ORDER_COL (column-major) rather than the default ROW ordering.
         * Although Mila tensors are row-major throughout, col-major with transpose ops
         * gives equivalent row-major semantics for this operation:
         *
         *   weight: [in, out] col-major + opA=T  → cuBLASLt reads as [out, in]
         *   input:  [in, batch] col-major + opB=N
         *   output: [out, batch] col-major       → equivalent to [batch, out] row-major in memory
         *
         * ROW ordering is not used here because cuBLASLt requires a minimum tile size
         * (typically M=16 on Ampere+) for tensor core kernels with CUBLASLT_ORDER_ROW.
         * With M=1 (the decode path, batch_size=1), ROW ordering produces
         * CUBLAS_STATUS_NOT_SUPPORTED during plan building since no tensor core algorithm
         * applies. COL ordering allows cuBLASLt to select a SIMT fallback algorithm for
         * all batch sizes including M=1, at no performance cost since M=1 GEMMs are
         * memory-bandwidth bound regardless of ordering.
         *
         * Note: execute_plan must be called with A=weight, B=input to match this layout.
         * Note: When CudaExecutionContext gains a workspace buffer, revisit using
         *       CUBLASLT_ORDER_ROW with CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES for
         *       full consistency with the attention plans.
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_forward_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int in_features,
            int out_features,
            bool has_bias,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            auto plan = build_plan<TNative>(
                handle,
                batch_size,
                in_features,
                out_features,
                has_bias,
                cuda_data_type,
                compute_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt forward plan built without algorithm (will use default)" );
            }
            return plan;
        } 

        /**
         * @brief Build cuBLASLt plan for backward input gradient computation.
         *
         * Computes dX[batch, in] = dY[batch, out] @ weight[out, in]
         * Row-major layout, opA=N, opB=N, batch_count=1.
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_backward_input_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int in_features,
            int out_features,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ batch_size,   /*A_cols=*/ out_features, /*ldA=*/ out_features, /*strideA=*/ 0LL,
                /*B_rows=*/ out_features, /*B_cols=*/ in_features,  /*ldB=*/ in_features,  /*strideB=*/ 0LL,
                /*C_rows=*/ batch_size,   /*C_cols=*/ in_features,  /*ldC=*/ in_features,  /*strideC=*/ 0LL,
                /*opA=*/ CUBLAS_OP_N, /*opB=*/ CUBLAS_OP_N,
                /*batch_count=*/ 1,
                /*has_bias=*/ false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt backward input plan built without algorithm (will use default)" );
            }
            
            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward weight gradient computation.
         *
         * Computes dW[out, in] = dY^T[out, batch] @ X[batch, in]
         * Row-major layout, opA=T, opB=N, batch_count=1.
         * Note: always built at max batch_size — weight grad accumulates full batch.
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_backward_weight_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int in_features,
            int out_features,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ batch_size,   /*A_cols=*/ out_features, /*ldA=*/ out_features, /*strideA=*/ 0LL,
                /*B_rows=*/ batch_size,   /*B_cols=*/ in_features,  /*ldB=*/ in_features,  /*strideB=*/ 0LL,
                /*C_rows=*/ out_features, /*C_cols=*/ in_features,  /*ldC=*/ in_features,  /*strideC=*/ 0LL,
                /*opA=*/ CUBLAS_OP_T, /*opB=*/ CUBLAS_OP_N,
                /*batch_count=*/ 1,
                /*has_bias=*/ false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt backward weight plan built without algorithm (will use default)" );
            }
            return plan;
        }

        /**
         * @brief Compute bias gradient via reduction sum across batch dimension.
         * dB[out] = sum(dY[batch, out], dim=0)
         */
        template <typename TNative>
        void compute_bias_gradient(
            TNative* bias_grad,
            const TNative* output_grad,
            int batch_size,
            int out_features,
            cudaStream_t stream )
        {
            if constexpr ( std::is_same_v<TNative, float> )
            {
                cuda_reduce_sum_batch_fp32( bias_grad, output_grad, batch_size, out_features, stream );
            }
            else if constexpr ( std::is_same_v<TNative, half> )
            {
                throw std::logic_error( "Bias gradient for half not yet implemented" );
            }
            else if constexpr ( std::is_same_v<TNative, nv_bfloat16> )
            {
                throw std::logic_error( "Bias gradient for bfloat16 not yet implemented" );
            }
            else
            {
                static_assert(
                    std::is_same_v<TNative, float> ||
                    std::is_same_v<TNative, half> ||
                    std::is_same_v<TNative, nv_bfloat16>,
                    "compute_bias_gradient only supports float, half, and nv_bfloat16");
            }
        }
    }
}