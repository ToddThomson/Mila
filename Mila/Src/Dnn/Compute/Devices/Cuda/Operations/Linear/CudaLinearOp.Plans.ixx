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
import Logging.Logger;

namespace Mila::Dnn::Compute::Cuda::Linear
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute::Cuda;

    namespace Detail
    {
        template <typename TComputePrecision>
        using CublasLtMatMulPlan = Mila::Dnn::Compute::Cuda::CublasLtMatMulPlan<TComputePrecision>;

        template <typename TComputePrecision>
        CublasLtMatMulPlan<TComputePrecision> build_forward_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int in_features,
            int out_features,
            bool has_bias,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            auto plan = build_plan<TComputePrecision>(
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
                Logging::Logger::warning( "cuBLASLt forward plan built without algorithm (will use default)" );
            }
            
            return plan;
        } 

        /**
         * @brief Build cuBLASLt plan for backward input gradient computation.
         *
         * Computes dX[batch, in] = dY[batch, out] @ weight[out, in]
         * Row-major layout, opA=N, opB=N, batch_count=1.
         */
        template <typename TComputePrecision>
        CublasLtMatMulPlan<TComputePrecision> build_backward_input_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int in_features,
            int out_features,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            auto plan = build_strided_plan<TComputePrecision>(
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
                Logging::Logger::warning( "cuBLASLt backward input plan built without algorithm (will use default)" );
            }
            
            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward weight gradient computation.
         *
         * Computes dW[out, in] = dY^T[out, batch] @ X[batch, in]
         * Row-major layout, opA=T, opB=N, batch_count=1.
         * Note: always built at max batch_size -- weight grad accumulates full batch.
         */
        template <typename TComputePrecision>
        CublasLtMatMulPlan<TComputePrecision> build_backward_weight_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int in_features,
            int out_features,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            auto plan = build_strided_plan<TComputePrecision>(
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
                Logging::Logger::warning( "cuBLASLt backward weight plan built without algorithm (will use default)" );
            }
            return plan;
        }

        /**
         * @brief Compute bias gradient via reduction sum across batch dimension.
         * dB[out] = sum(dY[batch, out], dim=0)
         */
        template <typename TComputePrecision>
        void compute_bias_gradient(
            TComputePrecision* bias_grad,
            const TComputePrecision* output_grad,
            int batch_size,
            int out_features,
            cudaStream_t stream )
        {
            if constexpr ( std::is_same_v<TComputePrecision, float> )
            {
                cuda_reduce_sum_batch_fp32( bias_grad, output_grad, batch_size, out_features, stream );
            }
            else if constexpr ( std::is_same_v<TComputePrecision, half> )
            {
                throw std::logic_error( "Bias gradient for half not yet implemented" );
            }
            else if constexpr ( std::is_same_v<TComputePrecision, nv_bfloat16> )
            {
                throw std::logic_error( "Bias gradient for bfloat16 not yet implemented" );
            }
            else
            {
                static_assert(
                    std::is_same_v<TComputePrecision, float> ||
                    std::is_same_v<TComputePrecision, half> ||
                    std::is_same_v<TComputePrecision, nv_bfloat16>,
                    "compute_bias_gradient only supports float, half, and nv_bfloat16");
            }
        }
    }
}