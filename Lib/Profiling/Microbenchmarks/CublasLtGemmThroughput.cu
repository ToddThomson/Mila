// cuBLASLt achieved throughput on SM120, at real Gemma 4 12B prefill shapes.
//
// Pairs with mma_peak.cu: that probe gives the INSTRUCTION ceiling, this gives
// what the library actually banks today. The gap between them is the only thing
// that decides whether a hand-written block-scaled kernel is worth writing.
//
// Arms:
//   BF16 x BF16 -> BF16   (the 1x reference)
//   FP8  x FP8  -> BF16   (what kUseFp8ActivationPrefill runs through today)
//
// Matrices are sized well past L2 and re-touched between timed runs, so these
// are DRAM-resident numbers. Median of 7 reported with the spread.
//
// Build: nvcc -gencode=arch=compute_120f,code=sm_120f -O3 CublasLtGemmThroughput.cu \
//          -o CublasLtGemmThroughput.exe -lcublasLt

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error at line %d: %s\n", __LINE__, cudaGetErrorString(e)); \
    return -1.0; } } while (0)

#define CHECK_LT(x) do { cublasStatus_t status_ = (x); if (status_ != CUBLAS_STATUS_SUCCESS) { \
    printf("cuBLASLt error %d at line %d\n", (int)status_, __LINE__); \
    return -1.0; } } while (0)

struct Shape
{
    const char* name;
    int m, n, k;
};

// D[m,n] = A[k,m]^T . B[k,n]   -- TN, column-major, which is the only layout
// cuBLASLt FP8 accepts. m = tokens, n = out_features, k = in_features.
static double bench( cublasLtHandle_t lt, const Shape& s, bool fp8,
                     void* dA, void* dB, void* dD, void* workspace,
                     size_t workspace_size, float* d_scale_a, float* d_scale_b )
{
    const cudaDataType_t ab_type = fp8 ? CUDA_R_8F_E4M3 : CUDA_R_16BF;

    cublasLtMatmulDesc_t op = nullptr;
    CHECK_LT( cublasLtMatmulDescCreate( &op, CUBLAS_COMPUTE_32F, CUDA_R_32F ) );

    cublasOperation_t ta = CUBLAS_OP_T;
    cublasOperation_t tb = CUBLAS_OP_N;
    CHECK_LT( cublasLtMatmulDescSetAttribute( op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof( ta ) ) );
    CHECK_LT( cublasLtMatmulDescSetAttribute( op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof( tb ) ) );

    if ( fp8 )
    {
        CHECK_LT( cublasLtMatmulDescSetAttribute( op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                  &d_scale_a, sizeof( d_scale_a ) ) );
        CHECK_LT( cublasLtMatmulDescSetAttribute( op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                  &d_scale_b, sizeof( d_scale_b ) ) );
    }

    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
    CHECK_LT( cublasLtMatrixLayoutCreate( &la, ab_type, s.k, s.m, s.k ) );
    CHECK_LT( cublasLtMatrixLayoutCreate( &lb, ab_type, s.k, s.n, s.k ) );
    CHECK_LT( cublasLtMatrixLayoutCreate( &ld, CUDA_R_16BF, s.m, s.n, s.m ) );

    cublasLtMatmulPreference_t pref = nullptr;
    CHECK_LT( cublasLtMatmulPreferenceCreate( &pref ) );
    CHECK_LT( cublasLtMatmulPreferenceSetAttribute( pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &workspace_size, sizeof( workspace_size ) ) );

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic( lt, op, la, lb, ld, ld, pref,
                                                        1, &heuristic, &returned );

    if ( hs != CUBLAS_STATUS_SUCCESS || returned == 0 )
    {
        cublasLtMatmulPreferenceDestroy( pref );
        cublasLtMatrixLayoutDestroy( la ); cublasLtMatrixLayoutDestroy( lb );
        cublasLtMatrixLayoutDestroy( ld ); cublasLtMatmulDescDestroy( op );
        return -2.0;  // no algorithm for this combination
    }

    const float alpha = 1.0f, beta = 0.0f;

    auto launch = [&]() {
        return cublasLtMatmul( lt, op, &alpha, dA, la, dB, lb, &beta,
                               dD, ld, dD, ld, &heuristic.algo,
                               workspace, workspace_size, nullptr );
    };

    CHECK_LT( launch() );
    CHECK( cudaDeviceSynchronize() );

    cudaEvent_t start, stop;
    CHECK( cudaEventCreate( &start ) );
    CHECK( cudaEventCreate( &stop ) );

    std::vector<double> samples;

    for ( int rep = 0; rep < 7; ++rep )
    {
        CHECK( cudaEventRecord( start ) );

        // Several launches per timing so per-launch overhead does not dominate
        // the small-M shapes.
        for ( int i = 0; i < 10; ++i )
        {
            CHECK_LT( launch() );
        }

        CHECK( cudaEventRecord( stop ) );
        CHECK( cudaEventSynchronize( stop ) );

        float ms = 0.f;
        CHECK( cudaEventElapsedTime( &ms, start, stop ) );

        const double flops = 10.0 * 2.0 * s.m * s.n * s.k;
        samples.push_back( flops / ( ms * 1.0e-3 ) / 1.0e12 );
    }

    std::sort( samples.begin(), samples.end() );

    CHECK( cudaEventDestroy( start ) );
    CHECK( cudaEventDestroy( stop ) );
    cublasLtMatmulPreferenceDestroy( pref );
    cublasLtMatrixLayoutDestroy( la ); cublasLtMatrixLayoutDestroy( lb );
    cublasLtMatrixLayoutDestroy( ld ); cublasLtMatmulDescDestroy( op );

    // Report median; caller prints spread from min/max via the same vector shape.
    printf( "    (min %.1f  max %.1f)", samples.front(), samples.back() );

    return samples[samples.size() / 2];
}

int main()
{
    cudaDeviceProp prop{};
    CHECK( cudaGetDeviceProperties( &prop, 0 ) );
    printf( "Device: %s  (SM %d.%d, %d SMs)\n\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount );

    cublasLtHandle_t lt = nullptr;
    if ( cublasLtCreate( &lt ) != CUBLAS_STATUS_SUCCESS )
    {
        printf( "cublasLtCreate failed\n" );
        return 1;
    }

    // Gemma 4 12B: hidden 3840, GeGLU intermediate 15360, 16 Q heads x 256,
    // 8 KV heads -> packed QKV width 8192. M = 1024 = the prefill chunk the
    // heuristic actually picks at 48K context.
    const Shape shapes[] = {
        { "qkv      M1024 N8192  K3840",  1024,  8192,  3840 },
        { "o_proj   M1024 N3840  K4096",  1024,  3840,  4096 },
        { "ffn_up   M1024 N30720 K3840",  1024, 30720,  3840 },
        { "ffn_down M1024 N3840  K15360", 1024,  3840, 15360 },
        { "square   M4096 N4096  K4096",  4096,  4096,  4096 },
    };

    // Allocate once at the largest extent any shape needs.
    size_t max_a = 0, max_b = 0, max_d = 0;
    for ( const Shape& s : shapes )
    {
        max_a = std::max( max_a, (size_t)s.k * s.m );
        max_b = std::max( max_b, (size_t)s.k * s.n );
        max_d = std::max( max_d, (size_t)s.m * s.n );
    }

    void *dA = nullptr, *dB = nullptr, *dD = nullptr, *workspace = nullptr;
    const size_t workspace_size = 64ull * 1024 * 1024;

    CHECK( cudaMalloc( &dA, max_a * 2 ) );
    CHECK( cudaMalloc( &dB, max_b * 2 ) );
    CHECK( cudaMalloc( &dD, max_d * 2 ) );
    CHECK( cudaMalloc( &workspace, workspace_size ) );
    CHECK( cudaMemset( dA, 0x3c, max_a * 2 ) );
    CHECK( cudaMemset( dB, 0x3c, max_b * 2 ) );

    float *d_scale_a = nullptr, *d_scale_b = nullptr;
    CHECK( cudaMalloc( &d_scale_a, sizeof( float ) ) );
    CHECK( cudaMalloc( &d_scale_b, sizeof( float ) ) );
    const float one = 1.0f;
    CHECK( cudaMemcpy( d_scale_a, &one, sizeof( float ), cudaMemcpyHostToDevice ) );
    CHECK( cudaMemcpy( d_scale_b, &one, sizeof( float ), cudaMemcpyHostToDevice ) );

    printf( "%-28s %14s %14s %8s\n", "shape", "BF16 TFLOP/s", "FP8 TFLOP/s", "ratio" );

    for ( const Shape& s : shapes )
    {
        printf( "%-28s", s.name );

        printf( "\n  BF16:" );
        const double bf16 = bench( lt, s, false, dA, dB, dD, workspace,
                                   workspace_size, d_scale_a, d_scale_b );
        printf( " %8.1f TFLOP/s\n", bf16 );

        printf( "  FP8 :" );
        const double fp8 = bench( lt, s, true, dA, dB, dD, workspace,
                                  workspace_size, d_scale_a, d_scale_b );

        if ( fp8 == -2.0 )
        {
            printf( " NO ALGORITHM\n\n" );
        }
        else
        {
            printf( " %8.1f TFLOP/s   ratio %.2fx\n\n", fp8, fp8 / bf16 );
        }
    }

    cublasLtDestroy( lt );

    return 0;
}
