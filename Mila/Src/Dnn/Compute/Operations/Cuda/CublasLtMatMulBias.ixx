module;
#include <cublasLt.h>
//#include <cuda_fp16.h>
#include <type_traits>

export module Compute.CublasLtMatMulBias;

import Cuda.DataTypeTraits;
import CublasLt.Error;
import Utils.Logger;

namespace Mila::Dnn::Compute
{
    /**
    * @brief cuBLASLt implementation of matrix multiplication with bias addition
    *
    * @tparam TPrecision Data type for computation (float, half, etc.)
    * @param out Output tensor data pointer
    * @param inp Input tensor data pointer
    * @param weight Weight tensor data pointer
    * @param bias Bias tensor data pointer (can be nullptr)
    * @param B Batch size
    * @param TPrecision Sequence length
    * @param C Input channels
    * @param OC Output channels
    * @param stream CUDA stream
    */
    export template <typename TPrecision>
    void cublaslt_matmul_forward(
        TPrecision* out, const TPrecision* inp, const TPrecision* weight, const TPrecision* bias,
        int B, int T, int C, int OC,
        cudaStream_t stream,
        cublasLtHandle_t cublasLtHandle ) {

        // Check alignment (some modes work unaligned but it always best to be aligned for performance)
        if ( ((uintptr_t)out % 16) != 0 || ((uintptr_t)inp % 16) != 0 || ((uintptr_t)weight % 16) != 0 || ((uintptr_t)bias % 16) != 0 ) {
            printf( "All cuBLASLt pointers must be aligned!\n" );
            exit( EXIT_FAILURE );
        }

        // Get CUDA data type for the given C++ type
        constexpr cudaDataType_t cuda_data_type = CudaDataTypeMap<TPrecision>::value;
        constexpr cublasComputeType_t compute_type = CudaDataTypeMap<TPrecision>::compute_type;

        // Scale factors - use the same type as the computation
        const TPrecision alpha = static_cast<TPrecision>(1.0f);
        const TPrecision beta = static_cast<TPrecision>(0.0f);

        // Create the operation descriptor
        cublasLtMatmulDesc_t operationDesc;
        cublasLtCheckStatus( cublasLtMatmulDescCreate( &operationDesc, compute_type, cuda_data_type ) );

		bool transA = true; // Transpose for weight
		bool transB = false; // Transpose for input

        cublasOperation_t opNoTranspose = CUBLAS_OP_N;
        cublasOperation_t opTranspose = CUBLAS_OP_T;
        cublasLtCheckStatus( cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA) ? &opTranspose : &opNoTranspose, sizeof( opTranspose ) ) );
        cublasLtCheckStatus( cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose : &opNoTranspose, sizeof( opNoTranspose ) ) );
        
        
        // Create matrix descriptors
        cublasLtMatrixLayout_t weightLayout, inputLayout, outputLayout;
        // Create matrix descriptors (rows, cols, leading dimension)
		// m,n,k  = OC, B*T, C
        cublasLtCheckStatus( cublasLtMatrixLayoutCreate( &weightLayout, cuda_data_type, C, OC, C ) ); // OC, C, OC ) ); // [m, k, m] for non transposed, [k, m, k]
        cublasLtCheckStatus( cublasLtMatrixLayoutCreate( &inputLayout, cuda_data_type, C, B*T, C ) );
        cublasLtCheckStatus( cublasLtMatrixLayoutCreate( &outputLayout, cuda_data_type, OC, B*T, OC ) );

        // Set bias vector if provided
        if ( bias != nullptr ) {
            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                operationDesc,
                CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias,
                sizeof( bias ) ) );

            // Tell cuBLASLt to add bias after matmul
            const int bias_op = CUBLASLT_EPILOGUE_BIAS;
            cublasLtCheckStatus( cublasLtMatmulDescSetAttribute(
                operationDesc,
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
            cublasLtHandle, operationDesc,
            weightLayout, inputLayout, outputLayout, outputLayout,
            preference, 1, &heuristicResult, &returnedAlgoCount );

        if ( status != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0 ) {
            Utils::Logger::warning( "No cuBLASLt heuristic found. Falling back to default algorithm." );

            // Attempt default execution (may still fail if unsupported)
            heuristicResult.algo = {};
        }

        status = cublasLtMatmul(
            cublasLtHandle,
            operationDesc,
            &alpha,                // alpha scaling factor
            weight,                // A matrix (weight)
            weightLayout,          // A matrix layout
            inp,                   // B matrix (input)
            inputLayout,           // B matrix layout
            &beta,                 // beta scaling factor
            nullptr,               // C matrix for accumulation (not used here)
            outputLayout,          // C matrix layout
            out,                   // D matrix (output)
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
        cublasLtMatmulDescDestroy( operationDesc );
        cublasLtMatmulPreferenceDestroy( preference );

        if ( status != CUBLAS_STATUS_SUCCESS ) {
            Utils::Logger::warning( "[FusedLinear] cuBLASLt matmul failed!" );
            throw CublasLtError( status );
        }
    };

    //// Explicit instantiations for supported types
    //template void cublaslt_matmul_forward<float>(
    //    float* out, const float* inp, const float* weight, const float* bias,
    //    int B, int TPrecision, int C, int OC, cudaStream_t stream );
    //
    //template void cublaslt_matmul_forward<half>(
    //    half* out, const half* inp, const half* weight, const half* bias,
    //    int B, int TPrecision, int C, int OC, cudaStream_t stream );

    // Can add more instantiations for other types as needed
}