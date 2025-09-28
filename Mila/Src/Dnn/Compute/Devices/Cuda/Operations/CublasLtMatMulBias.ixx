/**
 * @file CublasLtMatMulBias.ixx
 * @brief CUDA-accelerated matrix multiplication with bias addition using cuBLASLt
 *
 * @details
 * This module provides high-performance matrix multiplication operations optimized for neural network
 * linear layers. It leverages NVIDIA's cuBLASLt library to efficiently execute matrix operations on
 * GPU tensor cores with configurable precision modes.
 *
 * Key features:
 * - Mixed precision computation capabilities (FP32, FP16, BF16, FP8)
 * - Optimized matrix multiplication with fused bias addition
 * - Support for adaptive precision based on computation policy
 * - Automatic algorithm selection via cuBLASLt heuristics
 *
 * The implementation handles various data types with appropriate compute precision selection
 * based on the provided ComputePrecision policy, including accuracy vs. performance tradeoffs.
 */

module;
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>

export module Compute.CublasLtMatMulBias;

import Dnn.TensorData;
import Dnn.TensorTraits;
import Compute.Precision;
import Cuda.DataTypeTraits;
import CublasLt.Error;
import Utils.Logger;

namespace Mila::Dnn::Compute
{
    template <typename T>
    constexpr bool always_false = false;

    /**
    * @brief cuBLASLt implementation of matrix multiplication with bias addition
    *
    * Performs Y = W·X + bias where W is the weight matrix, X is the input matrix,
    * and bias is optionally added to each output column. The operation uses cuBLASLt
    * for optimized matrix multiplication with configurable precision settings.
    *
    * @tparam TDataType Data type for inputs and outputs (float, half, bfloat16, fp8)
    * @param Y Output tensor pointer [OC × outer_size]
    * @param X Input tensor pointer [C × outer_size]
    * @param weight Weight tensor pointer [C × OC] (transposed internally)
    * @param bias Bias tensor pointer [OC] (can be nullptr for no bias)
    * @param outer_size Combined batch and sequence dimensions (B×T)
    * @param C Input feature dimension
    * @param OC Output feature dimension
    * @param stream CUDA stream for asynchronous execution
    * @param cublasLtHandle cuBLASLt library handle
    * @param precision_policy Policy controlling computation precision and performance tradeoffs
    */
    export template <typename TDataType>
        requires ValidFloatTensorType<TDataType>
    void cublaslt_matmul_forward(
        TDataType* Y, const TDataType* X, const TDataType& weight, void* bias,
        int outer_size, int C, int OC,
        cudaStream_t stream,
        cublasLtHandle_t cublasLtHandle,
        ComputePrecision::Policy precision_policy = ComputePrecision::Policy::Auto ) {

        // REVIEW: Tensors are aligned at construction. This should not be needed
        // 
        // Check alignment (some modes work unaligned but it always best to be aligned for performance)
        /*if ( ((uintptr_t)Y % 16) != 0 || ((uintptr_t)X % 16) != 0 || ((uintptr_t)weight % 16) != 0 || ((uintptr_t)bias % 16) != 0 ) {
            printf( "All cuBLASLt pointers must be aligned!\n" );
            exit( EXIT_FAILURE );
        }*/

        cudaDataType_t cuda_data_type;

		// The TDataType here is the Linear layer input/output type

        if constexpr ( std::is_same_v<TDataType, float> ) {
            cuda_data_type = CUDA_R_32F;
        }
        else if constexpr ( std::is_same_v<TDataType, half> ) {
            cuda_data_type = CUDA_R_16F;
        }
        else if constexpr ( std::is_same_v<TDataType, __nv_bfloat16> ) {
            cuda_data_type = CUDA_R_16BF;
        }
        else if constexpr ( std::is_same_v<TDataType, __nv_fp8_e4m3> ) {
            cuda_data_type = CUDA_R_8F_E4M3;
        }
        else if constexpr ( std::is_same_v<TDataType, __nv_fp8_e5m2> ) {
            cuda_data_type = CUDA_R_8F_E5M2;
        }
        else {
            static_assert(always_false<TDataType>, "Unsupported data type");
        }

        cublasComputeType_t compute_type;
        cudaDataType_t scale_type;

        switch ( precision_policy ) {
            case ComputePrecision::Policy::Native:
            case ComputePrecision::Policy::Accuracy:
                if constexpr ( std::is_same_v<TDataType, half> ) {
                    compute_type = CUBLAS_COMPUTE_16F;
                    scale_type = CUDA_R_16F;
                }
                else if constexpr ( std::is_same_v<TDataType, __nv_bfloat16> ) {
                    compute_type = CUBLAS_COMPUTE_32F;
                    scale_type = CUDA_R_32F;
                }
                else {
                    compute_type = CUBLAS_COMPUTE_32F;
                    scale_type = CUDA_R_32F;
                }
                break;
            case ComputePrecision::Policy::Performance:
            case ComputePrecision::Policy::Auto:
            default:
                if constexpr ( std::is_same_v<TDataType, half> ) {
                    compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
                    scale_type = CUDA_R_32F;
                }
                else if constexpr ( std::is_same_v<TDataType, __nv_bfloat16> ) {
                    compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
                    scale_type = CUDA_R_32F;
                }
                else {
                    compute_type = CUBLAS_COMPUTE_32F;
                    scale_type = CUDA_R_32F;
                }
                break;
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasLtMatmulDesc_t matmul_op;
        cublasLtCheckStatus( cublasLtMatmulDescCreate( &matmul_op, compute_type, scale_type ) );

        bool transA = true; // Transpose for weight
        bool transB = false; // Transpose for input

        cublasOperation_t opNoTranspose = CUBLAS_OP_N;
        cublasOperation_t opTranspose = CUBLAS_OP_T;
        cublasLtCheckStatus( cublasLtMatmulDescSetAttribute( matmul_op, CUBLASLT_MATMUL_DESC_TRANSA, (transA) ? &opTranspose : &opNoTranspose, sizeof( opTranspose ) ) );
        cublasLtCheckStatus( cublasLtMatmulDescSetAttribute( matmul_op, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose : &opNoTranspose, sizeof( opNoTranspose ) ) );

        // Create matrix descriptors
        cublasLtMatrixLayout_t weightLayout, inputLayout, outputLayout;
        // Create matrix descriptors (rows, cols, leading dimension)
        // m,n,k  = OC, B*T, C
        cublasLtCheckStatus( cublasLtMatrixLayoutCreate( &weightLayout, cuda_data_type, C, OC, C ) );
        cublasLtCheckStatus( cublasLtMatrixLayoutCreate( &inputLayout, cuda_data_type, C, outer_size, C ) );
        cublasLtCheckStatus( cublasLtMatrixLayoutCreate( &outputLayout, cuda_data_type, OC, outer_size, OC ) );

        // Set bias vector if provided
        if ( bias != nullptr ) {
            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                matmul_op,
                CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias,
                sizeof( bias ) ) );

            // Tell cuBLASLt to add bias after matmul
            const int bias_op = CUBLASLT_EPILOGUE_BIAS;
            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                matmul_op,
                CUBLASLT_MATMUL_DESC_EPILOGUE,
                &bias_op,
                sizeof( bias_op ) ) );
        }

        // Create preference handle with default settings
        cublasLtMatmulPreference_t preference;
        cublasLtCheckStatus( cublasLtMatmulPreferenceCreate( &preference ) );

        // Query for the best algorithm
        cublasLtMatmulHeuristicResult_t heuristicResult = {};
        int returnedAlgoCount = 0;

        cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
            cublasLtHandle, matmul_op,
            weightLayout, inputLayout, outputLayout, outputLayout,
            preference, 1, &heuristicResult, &returnedAlgoCount );

        if ( status != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0 ) {
            Utils::Logger::warning( "No cuBLASLt heuristic found. Falling back to default algorithm." );

            // Attempt default execution (may still fail if unsupported)
            heuristicResult.algo = {};
        }

        status = cublasLtMatmul(
            cublasLtHandle,
            matmul_op,
            &alpha,                // alpha scaling factor
            weight,                // A matrix (weight)
            weightLayout,          // A matrix layout
            X,                     // B matrix (input)
            inputLayout,           // B matrix layout
            &beta,                 // beta scaling factor
            nullptr,               // C matrix for accumulation (not used here)
            outputLayout,          // C matrix layout
            Y,                     // D matrix (output)
            outputLayout,          // D matrix layout
            &heuristicResult.algo, // Algorithm to use from heuristic result
            nullptr,               // Workspace (using default)
            0,                     // Workspace size
            stream                 // CUDA stream
        );

        // Clean up descriptors
        cublasLtMatrixLayoutDestroy( weightLayout );
        cublasLtMatrixLayoutDestroy( inputLayout );
        cublasLtMatrixLayoutDestroy( outputLayout );
        cublasLtMatmulDescDestroy( matmul_op );
        cublasLtMatmulPreferenceDestroy( preference );

        if ( status != CUBLAS_STATUS_SUCCESS ) {
            Utils::Logger::warning( "[F] cuBLASLt matmul failed!" );
            throw CublasLtError( status );
        }
    };
}