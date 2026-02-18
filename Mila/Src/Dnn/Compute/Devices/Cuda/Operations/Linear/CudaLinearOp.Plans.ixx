/**
 * @file CudaLinearOp.ixx
 * @brief CUDA implementation of Linear operation with two-phase cuBLASLt optimization.
 */

module;
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector>
#include <memory>
#include <string>
#include <format>
#include <stdexcept>
#include <exception>
#include <cstdint>
#include <type_traits>
#include <sstream>
#include <cassert>
#include <algorithm>
#include "Kernels/Linear.cuh"

export module Compute.CudaLinearOp:Plans;

import Dnn.Components.Linear;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.Precision;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.ExecutionContextTemplate;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaDevice;
import Compute.CudaTensorDataType;
import CublasLt.Error;
import Utils.Logger;

// DEBUG:
import Dnn.TensorOps;
import Dnn.TensorHelpers;

namespace Mila::Dnn::Compute::Cuda::Linear
{
    using namespace Mila::Dnn;

    namespace Detail
    {
        /**
         * @brief cuBLASLt matmul execution plan for forward and backward operations.
         *
         * Caches all cuBLASLt descriptors, layouts, and selected algorithm.
         * Created once during build(), reused for all forward/backward calls.
         *
         * Default-constructed plan has all handles set to nullptr (invalid state).
         * After successful build, handles are valid and must be destroyed in destructor.
         */
        template <typename TNative>
        struct CublasLtMatMulPlan
        {
            mutable cublasLtMatmulDesc_t matmul_desc{ nullptr };
            cublasLtMatrixLayout_t weight_layout{ nullptr };
            cublasLtMatrixLayout_t input_layout{ nullptr };
            cublasLtMatrixLayout_t output_layout{ nullptr };
            cublasLtMatmulPreference_t preference{ nullptr };
            cublasLtMatmulAlgo_t algorithm{};

            bool has_algorithm{ false };
            bool has_bias_epilogue{ false };

            ~CublasLtMatMulPlan()
            {
                if ( matmul_desc ) cublasLtMatmulDescDestroy( matmul_desc );
                if ( weight_layout ) cublasLtMatrixLayoutDestroy( weight_layout );
                if ( input_layout ) cublasLtMatrixLayoutDestroy( input_layout );
                if ( output_layout ) cublasLtMatrixLayoutDestroy( output_layout );
                if ( preference ) cublasLtMatmulPreferenceDestroy( preference );
            }

            CublasLtMatMulPlan( const CublasLtMatMulPlan& ) = delete;
            CublasLtMatMulPlan& operator=( const CublasLtMatMulPlan& ) = delete;

            CublasLtMatMulPlan( CublasLtMatMulPlan&& other ) noexcept
                : matmul_desc( other.matmul_desc )
                , weight_layout( other.weight_layout )
                , input_layout( other.input_layout )
                , output_layout( other.output_layout )
                , preference( other.preference )
                , algorithm( other.algorithm )
                , has_algorithm( other.has_algorithm )
                , has_bias_epilogue( other.has_bias_epilogue )
            {
                other.matmul_desc = nullptr;
                other.weight_layout = nullptr;
                other.input_layout = nullptr;
                other.output_layout = nullptr;
                other.preference = nullptr;
            }

            CublasLtMatMulPlan& operator=( CublasLtMatMulPlan&& other ) noexcept
            {
                if ( this != &other )
                {
                    if ( matmul_desc ) cublasLtMatmulDescDestroy( matmul_desc );
                    if ( weight_layout ) cublasLtMatrixLayoutDestroy( weight_layout );
                    if ( input_layout ) cublasLtMatrixLayoutDestroy( input_layout );
                    if ( output_layout ) cublasLtMatrixLayoutDestroy( output_layout );
                    if ( preference ) cublasLtMatmulPreferenceDestroy( preference );

                    matmul_desc = other.matmul_desc;
                    weight_layout = other.weight_layout;
                    input_layout = other.input_layout;
                    output_layout = other.output_layout;
                    preference = other.preference;
                    algorithm = other.algorithm;
                    has_algorithm = other.has_algorithm;
                    has_bias_epilogue = other.has_bias_epilogue;

                    other.matmul_desc = nullptr;
                    other.weight_layout = nullptr;
                    other.input_layout = nullptr;
                    other.output_layout = nullptr;
                    other.preference = nullptr;
                }
                return *this;
            }

            CublasLtMatMulPlan() = default;

            bool isValid() const
            {
                return matmul_desc != nullptr;
            }
        };

        /**
         * @brief Build cuBLASLt plan for forward pass.
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
            CublasLtMatMulPlan<TNative> plan;
            plan.has_bias_epilogue = has_bias;

            cublasLtCheckStatus( cublasLtMatmulDescCreate( &plan.matmul_desc, compute_type, scale_type ) );

            cublasOperation_t op_transpose = CUBLAS_OP_T;
            cublasOperation_t op_no_transpose = CUBLAS_OP_N;

            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_transpose, sizeof( cublasOperation_t ) ) );

            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_no_transpose, sizeof( cublasOperation_t ) ) );

            cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
                &plan.weight_layout, cuda_data_type, in_features, out_features, in_features ) );

            cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
                &plan.input_layout, cuda_data_type, in_features, batch_size, in_features ) );

            cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
                &plan.output_layout, cuda_data_type, out_features, batch_size, out_features ) );

            if ( has_bias )
            {
                const int epilogue = CUBLASLT_EPILOGUE_BIAS;
                cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                    plan.matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                    &epilogue, sizeof( epilogue ) ) );

                // DEBUG: ADD THIS: Set the bias data type
                // Test result: no change
                cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                    plan.matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                    &cuda_data_type, sizeof( cuda_data_type ) ) );
            }

            cublasLtCheckStatus( cublasLtMatmulPreferenceCreate( &plan.preference ) );

            cublasLtMatmulHeuristicResult_t heuristic_result{};
            int returned_algo_count = 0;

            cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
                handle, plan.matmul_desc,
                plan.weight_layout, plan.input_layout, plan.output_layout, plan.output_layout,
                plan.preference, 1, &heuristic_result, &returned_algo_count );

            if ( status == CUBLAS_STATUS_SUCCESS && returned_algo_count > 0 )
            {
                plan.algorithm = heuristic_result.algo;
                plan.has_algorithm = true;
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt forward heuristic failed, will use default algorithm" );
                plan.algorithm = {};
                plan.has_algorithm = false;
            }

            // DEBUG: Start
            Utils::Logger::debug( std::format(
                "build_forward_plan: in={}, out={}, batch={}, has_bias={}, "
                "weight_layout dims=[{},{}] LD={}, "
                "TRANSA={}",
                in_features, out_features, batch_size, has_bias,
                in_features, out_features, in_features,
                "CUBLAS_OP_T" ) );
            // DEBUG: End

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward input gradient computation.
         *
         * Computes dX = W * dY (no transpose on either matrix)
         *
         * Reference cuBLAS call (from comment at line 236):
         * cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, OC,
         *             &one, weight, C, dout, OC, &zero, dinp, C)
         *
         * Where: C=in_features, OC=out_features, B*T=batch_size
         * - A (weight): [in_features × out_features] with ld=in_features (no transpose)
         * - B (dout/output_grad): [out_features × batch_size] with ld=out_features (no transpose)
         * - C (dinp/input_grad): [in_features × batch_size] with ld=in_features
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
            CublasLtMatMulPlan<TNative> plan;
            plan.has_bias_epilogue = false;

            cublasLtCheckStatus( cublasLtMatmulDescCreate( &plan.matmul_desc, compute_type, scale_type ) );

            cublasOperation_t op_no_transpose = CUBLAS_OP_N;

            // Both matrices: no transpose
            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_no_transpose, sizeof( cublasOperation_t ) ) );

            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_no_transpose, sizeof( cublasOperation_t ) ) );

            // A matrix (weight): [in_features × out_features] with ld=in_features
            cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
                &plan.weight_layout, cuda_data_type, in_features, out_features, in_features ) );

            // B matrix (output_grad): [out_features × batch_size] with ld=out_features
            cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
                &plan.input_layout, cuda_data_type, out_features, batch_size, out_features ) );

            // C matrix (input_grad): [in_features × batch_size] with ld=in_features
            cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
                &plan.output_layout, cuda_data_type, in_features, batch_size, in_features ) );

            cublasLtCheckStatus( cublasLtMatmulPreferenceCreate( &plan.preference ) );

            cublasLtMatmulHeuristicResult_t heuristic_result{};
            int returned_algo_count = 0;

            cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
                handle, plan.matmul_desc,
                plan.weight_layout, plan.input_layout, plan.output_layout, plan.output_layout,
                plan.preference, 1, &heuristic_result, &returned_algo_count );

            if ( status == CUBLAS_STATUS_SUCCESS && returned_algo_count > 0 )
            {
                plan.algorithm = heuristic_result.algo;
                plan.has_algorithm = true;
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt backward input heuristic failed, will use default algorithm" );
                plan.algorithm = {};
                plan.has_algorithm = false;
            }

            return plan;
        }

        /**
          * @brief Build cuBLASLt plan for backward weight gradient computation.
          *
          * Computes dW = X * dY^T
          * To compute dW[in_features, out_features] from:
          * - X[in_features, batch_size] (input)
          * - dY[out_features, batch_size] (output_grad)
          *
          * Reference cuBLAS call (from comment at line 305):
          * cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B*T,
          *             &one, inp, C, dout, OC, &one, dweight, C)
          *
          * Where: C=in_features, OC=out_features, B*T=batch_size
          * - A (inp/input): [in_features × batch_size] with ld=in_features (no transpose)
          * - B (dout/output_grad): [out_features × batch_size] with ld=out_features (transposed)
          * - C (dweight): [in_features × out_features] with ld=in_features
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
            CublasLtMatMulPlan<TNative> plan;
            plan.has_bias_epilogue = false;

            cublasLtCheckStatus( cublasLtMatmulDescCreate( &plan.matmul_desc, compute_type, scale_type ) );

            cublasOperation_t op_transpose = CUBLAS_OP_T;
            cublasOperation_t op_no_transpose = CUBLAS_OP_N;

            // A = input (inp), no transpose
            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_no_transpose, sizeof( cublasOperation_t ) ) );

            // B = output_grad (dout), transpose
            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                plan.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_transpose, sizeof( cublasOperation_t ) ) );

            // A matrix (input): [in_features × batch_size] with ld=in_features
            cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
                &plan.weight_layout, cuda_data_type, in_features, batch_size, in_features ) );

            // B matrix (output_grad): [out_features × batch_size] with ld=out_features, will be transposed
            cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
                &plan.input_layout, cuda_data_type, out_features, batch_size, out_features ) );

            // C matrix (weight_grad): [in_features × out_features] with ld=in_features
            cublasLtCheckStatus( cublasLtMatrixLayoutCreate(
                &plan.output_layout, cuda_data_type, in_features, out_features, in_features ) );

            cublasLtCheckStatus( cublasLtMatmulPreferenceCreate( &plan.preference ) );

            cublasLtMatmulHeuristicResult_t heuristic_result{};
            int returned_algo_count = 0;

            cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
                handle, plan.matmul_desc,
                plan.weight_layout, plan.input_layout, plan.output_layout, plan.output_layout,
                plan.preference, 1, &heuristic_result, &returned_algo_count );

            if ( status == CUBLAS_STATUS_SUCCESS && returned_algo_count > 0 )
            {
                plan.algorithm = heuristic_result.algo;
                plan.has_algorithm = true;
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt backward weight heuristic failed, will use default algorithm" );
                plan.algorithm = {};
                plan.has_algorithm = false;
            }

            return plan;
        }

        /**
         * @brief Compute bias gradient using CUDA reduction kernel.
         *
         * Performs a reduction sum across the batch dimension to compute the bias gradient.
         * In PyTorch notation: dB = dY.sum(dim=0), where all non-feature dimensions are
         * flattened into a single batch dimension.
         *
         * The function dispatches to type-specific CUDA kernels based on the native type:
         * - float (FP32): uses cuda_reduce_sum_batch_fp32
         * - half (FP16): not yet implemented (throws)
         * - nv_bfloat16 (BF16): not yet implemented (throws)
         *
         * @tparam TNative Native CUDA data type (float, half, or nv_bfloat16)
         *
         * @param bias_grad Output tensor for bias gradients [out_features].
         *                  Must be pre-allocated. Gradients are accumulated (+=) into this buffer.
         * @param output_grad Input tensor containing output gradients [batch_size, out_features].
         *                    Represents dL/dY from the downstream layer.
         * @param batch_size Total number of vectors across all batch dimensions.
         *                   Computed as the product of all tensor dimensions except the last
         *                   (feature) dimension. For example, a tensor of shape [B, T, C] has
         *                   batch_size = B × T.
         * @param out_features Number of output features (last dimension of output_grad).
         *                     Must be divisible by 32 for the FP32 kernel.
         * @param stream CUDA stream for asynchronous execution.
         *
         * @throws std::logic_error if TNative is half or nv_bfloat16 (not yet implemented)
         * @throws std::invalid_argument (from kernel) if out_features is not divisible by 32
         *
         * @note The reduction kernel requires out_features to be divisible by 32 due to
         *       warp-level coalesced memory access patterns.
         * @note This function accumulates into bias_grad; ensure it is zero-initialized
         *       before the first backward pass if not using gradient accumulation.
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
                throw std::logic_error( "Bias gradient computation for half is not yet implemented" );
                // FIXME: cuda_reduce_sum_batch_fp16( bias_grad, output_grad, batch_size, out_features, stream );
            }
            else if constexpr ( std::is_same_v<TNative, nv_bfloat16> )
            {
                throw std::logic_error( "Bias gradient computation for bfloat16 is not yet implemented" );
                // FIXME: cuda_reduce_sum_batch_bfp16( bias_grad, output_grad, batch_size, out_features, stream );
            }
            else
            {
                static_assert(std::is_same_v<TNative, float> ||
                    std::is_same_v<TNative, half> ||
                    std::is_same_v<TNative, nv_bfloat16>,
                    "compute_bias_gradient only supports float, half, and nv_bfloat16 types");
            }
        }

        /**
         * @brief Execute cuBLASLt matmul plan (generic version).
         *
         * This version handles both forward and backward passes.
         */
        template <typename TNative>
        void execute_cublaslt_plan(
            cublasLtHandle_t handle,
            const CublasLtMatMulPlan<TNative>& plan,
            const void* alpha,
            const TNative* A,
            const TNative* B,
            const void* beta,
            TNative* C,
            const TNative* bias,
            cudaStream_t stream )
        {
            // Validate inputs
            assert( handle != nullptr && "cuBLASLt handle must not be null" );
            assert( plan.isValid() && "cuBLASLt plan must be valid (built)" );
            assert( alpha != nullptr && "alpha scaling factor must not be null" );
            assert( beta != nullptr && "beta scaling factor must not be null" );
            assert( A != nullptr && "Input matrix A must not be null" );
            assert( B != nullptr && "Input matrix B must not be null" );
            assert( C != nullptr && "Output matrix C must not be null" );

            // Bias is optional - only validate if plan expects it
            assert( (!plan.has_bias_epilogue || bias != nullptr) &&
                "Bias pointer must not be null when plan has bias epilogue enabled" );

            if ( plan.has_bias_epilogue && bias != nullptr )
            {
                cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                    plan.matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    &bias, sizeof( bias ) ) );
            }

            cublasStatus_t status = cublasLtMatmul(
                handle,
                plan.matmul_desc,
                alpha,
                A, plan.weight_layout,
                B, plan.input_layout,
                beta,
                C, plan.output_layout,
                C, plan.output_layout,
                plan.has_algorithm ? &plan.algorithm : nullptr,
                nullptr, 0,
                stream );

            if ( status != CUBLAS_STATUS_SUCCESS )
            {
                throw CublasLtError( status );
            }
        }
    }
}