// Which cuBLASLt scale modes does this card actually accept, and what do they cost?
//
// Mila's FP8-activation prefill runs three kernels around one GEMM:
//
//   dequantize_fp4_to_fp8   ->  cuBLASLt FP8 x FP8 (scales = 1)  ->  apply_per_token_scales
//
// Both flanking passes exist only because the GEMM could not be told about the scales.
// Fp8ActivationPrefill.md records the reason: "Ada cuBLASLt accepts only per-tensor scale
// pointers (the outer-vector scale modes are Blackwell-era)". The ENUM has been present
// since at least CUDA 13.3, so the open question is not API availability but whether a
// given card returns an algorithm for it:
//
//   OUTER_VEC_32F   per-M and per-N FP32 vectors bound to the GEMM.
//                   If supported -> apply_per_token_scales (3.7% of prefill) DELETED.
//
//   VEC16_UE4M3     one UE4M3 scale per 16-element block -- NVFP4's native format, with
//                   A and B as CUDA_R_4F_E2M1.
//                   If supported -> dequantize_fp4_to_fp8 (7.5%, and far more at short
//                   prompts) DELETED, with no CUTLASS and no hand-written kernel.
//
// This probe answers support and cost together: a mode that works but runs at half the
// throughput of the shipped path is not a win, and the plumbing it removes is only worth
// 11-50% of prefill depending on prompt length.
//
// Build: nvcc -gencode=arch=compute_120f,code=sm_120f -O3 CublasLtScaleModes.cu \
//          -o CublasLtScaleModes.exe -lcublasLt
//
// Run on BOTH cards (pin by UUID) -- the Ada/Blackwell split is the whole question.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error line %d: %s\n", __LINE__, cudaGetErrorString(e)); return -1.0; } } while (0)

struct Shape
{
    const char* name;
    int m, n, k;
};

// D[m,n] = A[k,m]^T . B[k,n]. TN column-major is the only layout FP8 and FP4 accept.
struct Arm
{
    const char*                 name;
    cudaDataType_t              ab_type;
    cublasLtMatmulMatrixScale_t scale_mode;
};

// Returns TFLOP/s, -1 on error, -2 when no algorithm exists for the combination.
static double bench( cublasLtHandle_t lt, const Shape& s, const Arm& arm,
                     void* dA, void* dB, void* dD, void* workspace, size_t workspace_size,
                     void* scale_a, void* scale_b )
{
    cublasLtMatmulDesc_t op = nullptr;

    if ( cublasLtMatmulDescCreate( &op, CUBLAS_COMPUTE_32F, CUDA_R_32F ) != CUBLAS_STATUS_SUCCESS )
        return -1.0;

    cublasOperation_t ta = CUBLAS_OP_T;
    cublasOperation_t tb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute( op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof( ta ) );
    cublasLtMatmulDescSetAttribute( op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof( tb ) );

    // BF16 is the unscaled reference arm. Setting a scale mode on it makes cuBLASLt
    // report NO ALGORITHM -- which reads as "the card cannot do BF16" and is a lie the
    // harness told about itself before this guard existed.
    if ( arm.ab_type != CUDA_R_16BF )
    {
        const int32_t mode = static_cast<int32_t>( arm.scale_mode );
        const bool mode_ok =
            cublasLtMatmulDescSetAttribute( op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                            &mode, sizeof( mode ) ) == CUBLAS_STATUS_SUCCESS
            && cublasLtMatmulDescSetAttribute( op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                               &mode, sizeof( mode ) ) == CUBLAS_STATUS_SUCCESS;

        if ( !mode_ok )
        {
            cublasLtMatmulDescDestroy( op );
            return -2.0;
        }

        cublasLtMatmulDescSetAttribute( op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                        &scale_a, sizeof( scale_a ) );
        cublasLtMatmulDescSetAttribute( op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                        &scale_b, sizeof( scale_b ) );
    }

    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
    cublasLtMatrixLayoutCreate( &la, arm.ab_type, s.k, s.m, s.k );
    cublasLtMatrixLayoutCreate( &lb, arm.ab_type, s.k, s.n, s.k );
    cublasLtMatrixLayoutCreate( &ld, CUDA_R_16BF, s.m, s.n, s.m );

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate( &pref );
    cublasLtMatmulPreferenceSetAttribute( pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &workspace_size, sizeof( workspace_size ) );

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
        lt, op, la, lb, ld, ld, pref, 1, &heuristic, &returned );

    if ( hs != CUBLAS_STATUS_SUCCESS || returned == 0 )
    {
        cublasLtMatmulPreferenceDestroy( pref );
        cublasLtMatrixLayoutDestroy( la ); cublasLtMatrixLayoutDestroy( lb );
        cublasLtMatrixLayoutDestroy( ld ); cublasLtMatmulDescDestroy( op );
        return -2.0;  // the card has no kernel for this combination
    }

    const float alpha = 1.0f, beta = 0.0f;

    auto launch = [&]() {
        return cublasLtMatmul( lt, op, &alpha, dA, la, dB, lb, &beta, dD, ld, dD, ld,
                               &heuristic.algo, workspace, workspace_size, nullptr );
    };

    if ( launch() != CUBLAS_STATUS_SUCCESS )
    {
        cublasLtMatmulPreferenceDestroy( pref );
        cublasLtMatrixLayoutDestroy( la ); cublasLtMatrixLayoutDestroy( lb );
        cublasLtMatrixLayoutDestroy( ld ); cublasLtMatmulDescDestroy( op );
        return -3.0;  // heuristic offered an algorithm that then refused to run
    }

    CHECK( cudaDeviceSynchronize() );

    cudaEvent_t start, stop;
    CHECK( cudaEventCreate( &start ) );
    CHECK( cudaEventCreate( &stop ) );

    std::vector<double> samples;

    for ( int rep = 0; rep < 5; ++rep )
    {
        CHECK( cudaEventRecord( start ) );

        for ( int i = 0; i < 10; ++i )
            launch();

        CHECK( cudaEventRecord( stop ) );
        CHECK( cudaEventSynchronize( stop ) );

        float ms = 0.f;
        CHECK( cudaEventElapsedTime( &ms, start, stop ) );

        samples.push_back( 10.0 * 2.0 * s.m * s.n * s.k / ( ms * 1.0e-3 ) / 1.0e12 );
    }

    std::sort( samples.begin(), samples.end() );

    CHECK( cudaEventDestroy( start ) );
    CHECK( cudaEventDestroy( stop ) );
    cublasLtMatmulPreferenceDestroy( pref );
    cublasLtMatrixLayoutDestroy( la ); cublasLtMatrixLayoutDestroy( lb );
    cublasLtMatrixLayoutDestroy( ld ); cublasLtMatmulDescDestroy( op );

    return samples[samples.size() / 2];
}

int main()
{
    cudaDeviceProp prop{};
    cudaGetDeviceProperties( &prop, 0 );
    printf( "Device: %s  (SM %d.%d, %d SMs)\n\n", prop.name, prop.major, prop.minor,
            prop.multiProcessorCount );

    cublasLtHandle_t lt = nullptr;

    if ( cublasLtCreate( &lt ) != CUBLAS_STATUS_SUCCESS )
    {
        printf( "cublasLtCreate failed\n" );
        return 1;
    }

    const Shape shapes[] = {
        { "qkv      M1024 N8192  K3840",  1024,  8192,  3840 },
        { "ffn_up   M1024 N30720 K3840",  1024, 30720,  3840 },
        { "ffn_down M1024 N3840  K15360", 1024,  3840, 15360 },
    };

    const Arm arms[] = {
        { "BF16                 (reference)",  CUDA_R_16BF,    CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F },
        { "FP8  + SCALAR_32F    (ships today)", CUDA_R_8F_E4M3, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F },
        { "FP8  + OUTER_VEC_32F (kills epilogue)", CUDA_R_8F_E4M3, CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F },
        { "FP4  + VEC16_UE4M3   (kills dequant)",  CUDA_R_4F_E2M1, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 },
    };

    // Oversized for every arm: the widest operand is BF16 (2 bytes) and the scale planes
    // are far smaller than the data, so one generous allocation serves all of them.
    size_t max_a = 0, max_b = 0, max_d = 0;

    for ( const Shape& s : shapes )
    {
        max_a = std::max( max_a, (size_t)s.k * s.m );
        max_b = std::max( max_b, (size_t)s.k * s.n );
        max_d = std::max( max_d, (size_t)s.m * s.n );
    }

    void *dA = nullptr, *dB = nullptr, *dD = nullptr, *workspace = nullptr;
    void *scale_a = nullptr, *scale_b = nullptr;
    const size_t workspace_size = 64ull * 1024 * 1024;

    cudaMalloc( &dA, max_a * 2 );
    cudaMalloc( &dB, max_b * 2 );
    cudaMalloc( &dD, max_d * 2 );
    cudaMalloc( &workspace, workspace_size );
    cudaMemset( dA, 0x3c, max_a * 2 );
    cudaMemset( dB, 0x3c, max_b * 2 );

    // Big enough for the largest scale plane any mode asks for (one per 16 elements).
    cudaMalloc( &scale_a, max_a / 4 + 4096 );
    cudaMalloc( &scale_b, max_b / 4 + 4096 );
    cudaMemset( scale_a, 0x3f, max_a / 4 + 4096 );
    cudaMemset( scale_b, 0x3f, max_b / 4 + 4096 );

    for ( const Shape& s : shapes )
    {
        printf( "%s\n", s.name );

        for ( const Arm& arm : arms )
        {
            const double tf = bench( lt, s, arm, dA, dB, dD, workspace, workspace_size,
                                     scale_a, scale_b );

            if ( tf == -2.0 )
                printf( "  %-38s  NO ALGORITHM ON THIS CARD\n", arm.name );
            else if ( tf == -3.0 )
                printf( "  %-38s  heuristic offered, execution refused\n", arm.name );
            else if ( tf < 0.0 )
                printf( "  %-38s  error\n", arm.name );
            else
                printf( "  %-38s  %7.1f TFLOP/s\n", arm.name, tf );
        }

        printf( "\n" );
    }

    cublasLtDestroy( lt );

    return 0;
}
