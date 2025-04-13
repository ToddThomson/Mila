#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <memory>
#include <vector>

template <typename T>
struct CudaTraits;

template <>
struct CudaTraits<float> {
    static constexpr cudaDataType_t Type = CUDA_R_32F;
};

template <>
struct CudaTraits<__half> {
    static constexpr cudaDataType_t Type = CUDA_R_16F;
};

template <>
struct CudaTraits<__nv_bfloat16> {
    static constexpr cudaDataType_t Type = CUDA_R_16BF;
};

// Forward fused matmul + bias + GELU
template <typename T>
void launchFusedMatmulBiasGelu(
    const T* A,        // [M x K]
    const T* B,        // [K x N]
    const T* bias,     // [N]
    T* C,              // [M x N]
    size_t M, size_t K, size_t N,
    cublasLtHandle_t ltHandle,
    cudaStream_t stream )
{
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasOperation_t opNonTranspose = CUBLAS_OP_N;

    // Create matmul descriptor
    cublasLtMatmulDesc_t matmulDesc = nullptr;
    cublasLtMatrixLayout_t aDesc = nullptr, bDesc = nullptr, cDesc = nullptr;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;

    cublasLtMatmulDescCreate( &matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F );
    cublasLtMatmulDescSetAttribute( matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opNonTranspose, sizeof( opNonTranspose ) );
    cublasLtMatmulDescSetAttribute( matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNonTranspose, sizeof( opNonTranspose ) );
    cublasLtMatmulDescSetAttribute( matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof( epilogue ) );
    cublasLtMatmulDescSetAttribute( matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof( bias ) );

    // Matrix layout: row-major
    cublasLtMatrixLayoutCreate( &aDesc, CudaTraits<T>::Type, M, K, K );
    cublasLtMatrixLayoutCreate( &bDesc, CudaTraits<T>::Type, K, N, N );
    cublasLtMatrixLayoutCreate( &cDesc, CudaTraits<T>::Type, M, N, N );

    float alpha = 1.0f;
    float beta = 0.0f;

    size_t workspaceSize = 1 << 22;  // 4MB
    void* workspace;
    cudaMalloc( &workspace, workspaceSize );

    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate( &preference );
    cublasLtMatmulPreferenceSetAttribute( preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize,
        sizeof( workspaceSize ) );

    cublasLtMatmulAlgo_t algo;
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    cublasLtMatmulAlgoGetHeuristic( ltHandle, matmulDesc,
        aDesc, bDesc, cDesc, cDesc,
        preference, 1, &heuristicResult, &returnedResults );

    if ( returnedResults == 0 ) {
        throw std::runtime_error( "No suitable cuBLASLt matmul algorithm found." );
    }

    algo = heuristicResult.algo;

    cublasLtMatmul( ltHandle,
        matmulDesc,
        &alpha,
        A, aDesc,
        B, bDesc,
        &beta,
        C, cDesc,
        C, cDesc,
        &algo,
        workspace,
        workspaceSize,
        stream );

    cudaFree( workspace );
    cublasLtMatmulPreferenceDestroy( preference );
    cublasLtMatrixLayoutDestroy( aDesc );
    cublasLtMatrixLayoutDestroy( bDesc );
    cublasLtMatrixLayoutDestroy( cDesc );
    cublasLtMatmulDescDestroy( matmulDesc );
}
