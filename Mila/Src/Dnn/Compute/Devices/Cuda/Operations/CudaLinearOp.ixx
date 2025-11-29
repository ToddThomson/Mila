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
#include <stdexcept>
#include <exception>
#include <cstdint>
#include <type_traits>
#include "Kernels/CudaOps.h"
#include <sstream>
#include <cassert>

export module Compute.CudaLinearOp;

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
//import Compute.CudaExecutionContext;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaDevice;
import Compute.CudaTensorDataType;
import CublasLt.Error;
import Utils.Logger;

namespace Mila::Dnn::Compute
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
                if (matmul_desc) cublasLtMatmulDescDestroy( matmul_desc );
                if (weight_layout) cublasLtMatrixLayoutDestroy( weight_layout );
                if (input_layout) cublasLtMatrixLayoutDestroy( input_layout );
                if (output_layout) cublasLtMatrixLayoutDestroy( output_layout );
                if (preference) cublasLtMatmulPreferenceDestroy( preference );
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
                if (this != &other)
                {
                    if (matmul_desc) cublasLtMatmulDescDestroy( matmul_desc );
                    if (weight_layout) cublasLtMatrixLayoutDestroy( weight_layout );
                    if (input_layout) cublasLtMatrixLayoutDestroy( input_layout );
                    if (output_layout) cublasLtMatrixLayoutDestroy( output_layout );
                    if (preference) cublasLtMatmulPreferenceDestroy( preference );

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
         *
         * Matches the exact layout from working CublasLtMatMulBias.ixx:
         * - Weight: [in_features × out_features] with ld=in_features, transposed
         * - Input: [in_features × batch_size] with ld=in_features
         * - Output: [out_features × batch_size] with ld=out_features
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

            if (has_bias)
            {
                const int epilogue = CUBLASLT_EPILOGUE_BIAS;
                cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                    plan.matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                    &epilogue, sizeof( epilogue ) ) );
            }

            cublasLtCheckStatus( cublasLtMatmulPreferenceCreate( &plan.preference ) );

            cublasLtMatmulHeuristicResult_t heuristic_result{};
            int returned_algo_count = 0;

            cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
                handle, plan.matmul_desc,
                plan.weight_layout, plan.input_layout, plan.output_layout, plan.output_layout,
                plan.preference, 1, &heuristic_result, &returned_algo_count );

            if (status == CUBLAS_STATUS_SUCCESS && returned_algo_count > 0)
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

            if (status == CUBLAS_STATUS_SUCCESS && returned_algo_count > 0)
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

            if (status == CUBLAS_STATUS_SUCCESS && returned_algo_count > 0)
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
            if constexpr (std::is_same_v<TNative, float>)
            {
                cuda_reduce_sum_batch_fp32( bias_grad, output_grad, batch_size, out_features, stream );
            }
            else if constexpr (std::is_same_v<TNative, half>)
            {
				throw std::logic_error( "Bias gradient computation for half is not yet implemented" );
                //cuda_reduce_sum_batch_fp16( bias_grad, output_grad, batch_size, out_features, stream );
            }
            else if constexpr (std::is_same_v<TNative, nv_bfloat16>)
            {
				throw std::logic_error( "Bias gradient computation for bfloat16 is not yet implemented" );
                //cuda_reduce_sum_batch_bfp16( bias_grad, output_grad, batch_size, out_features, stream );
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

            if (plan.has_bias_epilogue && bias != nullptr)
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

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                throw CublasLtError( status );
            }
        }

        /**
         * @brief CUDA kernel dispatcher for Linear operations.
         *
         * Specialized for float, half, bfloat16, and FP8 native CUDA types.
         */
        template <typename TNative>
            requires std::is_same_v<TNative, float> ||
        std::is_same_v<TNative, half> ||
            std::is_same_v<TNative, nv_bfloat16> ||
            std::is_same_v<TNative, __nv_fp8_e4m3> ||
            std::is_same_v<TNative, __nv_fp8_e5m2>
            struct cuda_matmul_impl;

        template <>
        struct cuda_matmul_impl<float>
        {
            cuda_matmul_impl() = default;

            static inline void forward(
                float* output,
                const float* input,
                const float* weight,
                const float* bias,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                cuda_matmul_forward_fp32( output, input, weight, bias, batch_size, in_features, out_features, stream );
            }

            static inline void backward(
                float* input_grad,
                float* weight_grad,
                float* bias_grad,
                const float* output_grad,
                const float* input,
                const float* weight,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::backward for FP32 is not yet implemented" );
            }
        };

        template <>
        struct cuda_matmul_impl<half>
        {
            cuda_matmul_impl() = default;

            static inline void forward(
                half* output,
                const half* input,
                const half* weight,
                const half* bias,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                cuda_matmul_forward_fp16( output, input, weight, bias, batch_size, in_features, out_features, stream );
            }

            static inline void backward(
                half* input_grad,
                half* weight_grad,
                half* bias_grad,
                const half* output_grad,
                const half* input,
                const half* weight,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::backward for FP16 is not yet implemented" );
            }
        };

        template <>
        struct cuda_matmul_impl<nv_bfloat16>
        {
            cuda_matmul_impl() = default;

            static inline void forward(
                nv_bfloat16* output,
                const nv_bfloat16* input,
                const nv_bfloat16* weight,
                const nv_bfloat16* bias,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::forward for BF16 is not yet implemented" );
            }

            static inline void backward(
                nv_bfloat16* input_grad,
                nv_bfloat16* weight_grad,
                nv_bfloat16* bias_grad,
                const nv_bfloat16* output_grad,
                const nv_bfloat16* input,
                const nv_bfloat16* weight,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::backward for BF16 is not yet implemented" );
            }
        };

        template <>
        struct cuda_matmul_impl<__nv_fp8_e4m3>
        {
            cuda_matmul_impl() = default;

            static inline void forward(
                __nv_fp8_e4m3* output,
                const __nv_fp8_e4m3* input,
                const __nv_fp8_e4m3* weight,
                const __nv_fp8_e4m3* bias,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::forward for FP8_E4M3 is not yet implemented" );
            }

            static inline void backward(
                __nv_fp8_e4m3* input_grad,
                __nv_fp8_e4m3* weight_grad,
                __nv_fp8_e4m3* bias_grad,
                const __nv_fp8_e4m3* output_grad,
                const __nv_fp8_e4m3* input,
                const __nv_fp8_e4m3* weight,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::backward for FP8_E4M3 is not yet implemented" );
            }
        };

        template <>
        struct cuda_matmul_impl<__nv_fp8_e5m2>
        {
            cuda_matmul_impl() = default;

            static inline void forward(
                __nv_fp8_e5m2* output,
                const __nv_fp8_e5m2* input,
                const __nv_fp8_e5m2* weight,
                const __nv_fp8_e5m2* bias,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::forward for FP8_E5M2 is not yet implemented" );
            }

            static inline void backward(
                __nv_fp8_e5m2* input_grad,
                __nv_fp8_e5m2* weight_grad,
                __nv_fp8_e5m2* bias_grad,
                const __nv_fp8_e5m2* output_grad,
                const __nv_fp8_e5m2* input,
                const __nv_fp8_e5m2* weight,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::backward for FP8_E5M2 is not yet implemented" );
            }
        };
    }

    /**
     * @brief CUDA implementation of Linear operation using two-phase cuBLASLt optimization.
     *
     * Design philosophy:
     * - Two-phase initialization: build() creates cuBLASLt plans, forward()/backward() execute them
     * - Module owns weight/bias parameters and binds them via setParameters()
     * - All dimension computation and algorithm selection happens once in build()
     * - Forward/backward are hot-path methods with zero setup overhead
     * - cuBLASLt plans cache descriptors, layouts, and optimal algorithms
     *
     * Forward: output = input * weight^T + bias
     * Backward:
     *  - input_grad = output_grad * weight
     *  - weight_grad = output_grad^T * input
     *  - bias_grad = sum(output_grad)
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaLinearOp : public UnaryOperation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaLinearOp( std::shared_ptr<CudaExecutionContext> context, const LinearConfig& config )
            : context_( context ), config_( config ), impl_()
        {
            if (!context_)
            {
                throw std::runtime_error( "CudaLinearOp requires a CUDA execution context" );
            }

            config_.validate();
        }

        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            if (!weight)
            {
                throw std::invalid_argument( "CudaLinearOp::setParameters - weight parameter is required" );
            }

            if (weight->getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument( "CudaLinearOp::setParameters - weight must be a CUDA tensor" );
            }

            weight_ = static_cast<const NativeType*>(weight->rawData());

            const auto& weight_shape = weight->shape();
            if (weight_shape.size() != 2)
            {
                throw std::invalid_argument( "CudaLinearOp::setParameters - weight must be 2D tensor" );
            }

            weight_out_features_ = weight_shape[0];
            weight_in_features_ = weight_shape[1];

            if (config_.hasBias())
            {
                if (!bias)
                {
                    throw std::invalid_argument( "CudaLinearOp::setParameters - bias parameter expected but null was provided" );
                }

                if (bias->getDeviceType() != DeviceType::Cuda)
                {
                    throw std::invalid_argument( "CudaLinearOp::setParameters - bias must be a CUDA tensor" );
                }

                bias_ = static_cast<const NativeType*>(bias->rawData());
            }
            else
            {
                bias_ = nullptr;
            }
        }

        void setGradients( ITensor* weight_grad, ITensor* bias_grad ) override
        {
            if (!weight_grad)
            {
                throw std::invalid_argument( "CudaLinearOp::setParameterGradients - weight gradient is required" );
            }

            if (weight_grad->getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument( "CudaLinearOp::setParameterGradients - weight gradient must be a CUDA tensor" );
            }

            weight_grad_ = static_cast<NativeType*>(weight_grad->rawData());

            if (config_.hasBias())
            {
                if (!bias_grad)
                {
                    throw std::invalid_argument( "CudaLinearOp::setParameterGradients - bias gradient expected but null was provided" );
                }

                if (bias_grad->getDeviceType() != DeviceType::Cuda)
                {
                    throw std::invalid_argument( "CudaLinearOp::setParameterGradients - bias gradient must be a CUDA tensor" );
                }

                bias_grad_ = static_cast<NativeType*>(bias_grad->rawData());
            }
            else
            {
                bias_grad_ = nullptr;
            }
        }

        void build( const shape_t& input_shape ) override
        {
            if (weight_ == nullptr)
            {
                throw std::runtime_error( "CudaLinearOp::build requires parameters bound via setParameters() before build()." );
            }

            if (config_.hasBias() && bias_ == nullptr)
            {
                throw std::runtime_error( "CudaLinearOp::build - bias expected by config but not bound via setParameters()." );
            }

            if (input_shape.empty())
            {
                throw std::invalid_argument( "CudaLinearOp::build - input shape cannot be empty" );
            }

            cached_in_features_ = static_cast<int>(input_shape.back());

            if (weight_out_features_ != config_.getOutputFeatures())
            {
                std::ostringstream oss;
                oss << "CudaLinearOp::build - weight output features mismatch. Expected "
                    << config_.getOutputFeatures() << ", got " << weight_out_features_;
                throw std::invalid_argument( oss.str() );
            }

            if (weight_in_features_ != cached_in_features_)
            {
                std::ostringstream oss;
                oss << "CudaLinearOp::build - weight input features mismatch. Expected "
                    << cached_in_features_ << ", got " << weight_in_features_;
                throw std::invalid_argument( oss.str() );
            }

			// TJT: Better here is outer_size_. The use of batch_size_ is misleading.
            cached_batch_size_ = 1;
            for (size_t i = 0; i + 1 < input_shape.size(); ++i)
            {
                cached_batch_size_ *= static_cast<int>(input_shape[i]);
            }

            cached_out_features_ = static_cast<int>(config_.getOutputFeatures());

            cached_cublaslt_handle_ = context_->getCublasLtHandle();
            use_cublaslt_ = (cached_cublaslt_handle_ != nullptr) && supportsCuBLASLt();

            cached_precision_policy_ = config_.getPrecisionPolicy();

            if (use_cublaslt_)
            {
                try
                {
                    buildCublasLtPlans();
                }
                catch (const std::exception& e)
                {
                    Utils::Logger::warning(
                        std::string( "Failed to build cuBLASLt plans, falling back to custom kernels: " ) + e.what() );
                    use_cublaslt_ = false;
                }
            }

            UnaryOperationBase::build( input_shape );
        }

        void forward( const ITensor& input, ITensor& output ) const override
        {
            const NativeType* input_ptr = static_cast<const NativeType*>(input.rawData());
            NativeType* output_ptr = static_cast<NativeType*>(output.rawData());

            cudaStream_t stream = context_->getStream();

            if (use_cublaslt_)
            {
                const float alpha = 1.0f;
                const float beta = 0.0f;

                Detail::execute_cublaslt_plan(
                    cached_cublaslt_handle_,
                    forward_plan_,
                    &alpha,
                    weight_, input_ptr,
                    &beta,
                    output_ptr,
                    bias_,
                    stream );

                return;
            }

            Detail::cuda_matmul_impl<NativeType>::forward(
                output_ptr, input_ptr,
                weight_, bias_,
                cached_batch_size_,
                cached_in_features_, cached_out_features_,
                stream );
        }

        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            // Validate backward pass preconditions
            assert( this->isBuilt() && "CudaLinearOp must be built before calling backward()" );
            assert( weight_ != nullptr && "Weight pointer must be set before backward pass" );
            assert( weight_grad_ != nullptr && "Weight gradient pointer must be set before backward pass" );

            // Bias gradient is optional (only required if bias exists)
            assert( (!config_.hasBias() || bias_grad_ != nullptr) &&
                "Bias gradient pointer must be set when bias is enabled" );

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "CudaLinearOp::backward called in inference mode" );
			}

            const NativeType* input_ptr = static_cast<const NativeType*>(input.rawData());
            const NativeType* output_grad_ptr = static_cast<const NativeType*>(output_grad.rawData());
            NativeType* input_grad_ptr = static_cast<NativeType*>(input_grad.rawData());

            // Validate tensor pointers
            assert( input_ptr != nullptr && "Input tensor data must not be null" );
            assert( output_grad_ptr != nullptr && "Output gradient tensor data must not be null" );
            assert( input_grad_ptr != nullptr && "Input gradient tensor data must not be null" );

            // Verify tensor shapes match expected dimensions
            assert( input.shape().size() >= 2 && "Input must have at least 2 dimensions" );
            assert( output_grad.shape().size() >= 2 && "Output grad must have at least 2 dimensions" );

            // Verify last dimension matches
            assert( input.shape().back() == cached_in_features_ &&
                "Input last dimension must match in_features" );
            assert( output_grad.shape().back() == cached_out_features_ &&
                "Output grad last dimension must match out_features" );

            cudaStream_t stream = context_->getStream();

            if (use_cublaslt_)
            {
                assert( cached_cublaslt_handle_ != nullptr && "cuBLASLt handle must be valid" );
                assert( forward_plan_.isValid() && "Forward plan must be valid" );
                assert( backward_input_plan_.isValid() && "Backward input plan must be valid" );
                assert( backward_weight_plan_.isValid() && "Backward weight plan must be valid" );

                const float alpha = 1.0f;

                // 1. Compute input gradient: dX = W * dY
                // OVERWRITE input gradient (beta=0), not accumulate
				const float beta_input = 0.0f; // This fixes the invalid arg we were having

                Detail::execute_cublaslt_plan(
                    cached_cublaslt_handle_,
                    backward_input_plan_,
                    &alpha,
                    weight_,
                    output_grad_ptr,
                    &beta_input,
                    input_grad_ptr,
					static_cast<const NativeType*>(nullptr), // no bias
                    stream );

                // 2. Compute weight gradient: dW = X * dY^T
				// ACCUMULATE into weight gradient (beta=1)
				const float beta_params = 1.0f;
                
                Detail::execute_cublaslt_plan(
                    cached_cublaslt_handle_,
                    backward_weight_plan_,
                    &alpha,
                    input_ptr,        // A matrix (input)
                    output_grad_ptr,  // B matrix (output_grad, will be transposed)
                    &beta_params,
                    weight_grad_,     // C matrix (output)
					static_cast<const NativeType*>(nullptr), // no bias
                    stream );

                // 3. Compute bias gradient: dB = sum(dY) (if bias exists)
                if (bias_grad_ != nullptr)
                {
                    Detail::compute_bias_gradient(
                        bias_grad_,
                        output_grad_ptr,
                        cached_batch_size_,
                        cached_out_features_,
                        stream );
                }

                return;
            }

            // Fallback to custom non-cublasLt kernels
            Detail::cuda_matmul_impl<NativeType>::backward(
                input_grad_ptr, weight_grad_, bias_grad_,
                output_grad_ptr, input_ptr, weight_,
                cached_batch_size_,
                cached_in_features_, cached_out_features_,
                stream );
        }

        OperationType getOperationType() const override
        {
            return OperationType::LinearOp;
        }

        std::string getName() const override
        {
            return "Cuda::LinearOp";
        }

        const LinearConfig& getConfig() const
        {
            return config_;
        }

    private:
        LinearConfig config_;
        std::shared_ptr<CudaExecutionContext> context_;
        Detail::cuda_matmul_impl<NativeType> impl_;

        const NativeType* weight_{ nullptr };
        const NativeType* bias_{ nullptr };

        NativeType* weight_grad_{ nullptr };
        NativeType* bias_grad_{ nullptr };

        int64_t weight_out_features_{ 0 };
        int64_t weight_in_features_{ 0 };

        int cached_batch_size_{ 0 };
        int cached_in_features_{ 0 };
        int cached_out_features_{ 0 };

        cublasLtHandle_t cached_cublaslt_handle_{ nullptr };
        bool use_cublaslt_{ false };
        ComputePrecision::Policy cached_precision_policy_;

        Detail::CublasLtMatMulPlan<NativeType> forward_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_input_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_weight_plan_;

        constexpr bool supportsCuBLASLt() const
        {
            return std::is_same_v<NativeType, float> ||
                std::is_same_v<NativeType, half> ||
                std::is_same_v<NativeType, nv_bfloat16>;
        }

        void buildCublasLtPlans()
        {
            cudaDataType_t cuda_data_type = getCudaDataType();
            cublasComputeType_t compute_type;
            cudaDataType_t scale_type;

            getComputeTypes( compute_type, scale_type );

            forward_plan_ = Detail::build_forward_plan<NativeType>(
                cached_cublaslt_handle_,
                cached_batch_size_,
                cached_in_features_,
                cached_out_features_,
                config_.hasBias(),
                cuda_data_type,
                compute_type,
                scale_type );

            backward_input_plan_ = Detail::build_backward_input_plan<NativeType>(
                cached_cublaslt_handle_,
                cached_batch_size_,
                cached_in_features_,
                cached_out_features_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_weight_plan_ = Detail::build_backward_weight_plan<NativeType>(
                cached_cublaslt_handle_,
                cached_batch_size_,
                cached_in_features_,
                cached_out_features_,
                cuda_data_type,
                compute_type,
                scale_type );
        }

        cudaDataType_t getCudaDataType() const
        {
            if constexpr (std::is_same_v<NativeType, float>)
            {
                return CUDA_R_32F;
            }
            else if constexpr (std::is_same_v<NativeType, half>)
            {
                return CUDA_R_16F;
            }
            else if constexpr (std::is_same_v<NativeType, nv_bfloat16>)
            {
                return CUDA_R_16BF;
            }
            else if constexpr (std::is_same_v<NativeType, __nv_fp8_e4m3>)
            {
                return CUDA_R_8F_E4M3;
            }
            else if constexpr (std::is_same_v<NativeType, __nv_fp8_e5m2>)
            {
                return CUDA_R_8F_E5M2;
            }
        }

        void getComputeTypes( cublasComputeType_t& compute_type, cudaDataType_t& scale_type ) const
        {
            scale_type = CUDA_R_32F;

            switch (cached_precision_policy_)
            {
                case ComputePrecision::Policy::Native:
                case ComputePrecision::Policy::Accuracy:
                    if constexpr (std::is_same_v<NativeType, half>)
                    {
                        compute_type = CUBLAS_COMPUTE_16F;
                    }
                    else if constexpr (std::is_same_v<NativeType, nv_bfloat16>)
                    {
                        compute_type = CUBLAS_COMPUTE_32F;
                    }
                    else
                    {
                        compute_type = CUBLAS_COMPUTE_32F;
                    }
                    break;

                case ComputePrecision::Policy::Performance:
                case ComputePrecision::Policy::Auto:
                default:
                    if constexpr (std::is_same_v<NativeType, half>)
                    {
                        compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
                    }
                    else if constexpr (std::is_same_v<NativeType, nv_bfloat16>)
                    {
                        compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
                    }
                    else
                    {
                        compute_type = CUBLAS_COMPUTE_32F;
                    }
                    break;
            }
        }
    };

    export class CudaLinearOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "LinearOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::FP32>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
                {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return std::make_shared<CudaLinearOp<TensorDataType::FP32>>( context, linearConfig );
                }
            );

            /*OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP16>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ModuleConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>>
                {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return std::make_shared<CudaLinearOp<TensorDataType::FP16>>( context, linearConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::BF16>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ModuleConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::BF16>>
                {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return std::make_shared<CudaLinearOp<TensorDataType::BF16>>( context, linearConfig );
                }
            );*/
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}