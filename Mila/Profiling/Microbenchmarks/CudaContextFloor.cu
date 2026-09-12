// What does a bare CUDA context cost on this card, before Mila exists?
// Mila's ProfileModel reports used=1139 MiB after initialize(); this splits that
// into the driver's floor vs everything Mila adds (fatbin module load, cuBLAS(Lt)
// kernel tables, any pool).
//
// Build BOTH ways to price the dual-arch fatbin (measured 2026-09-12 as identical, so
// the arch list costs nothing in VRAM and the 1138 MiB floor is the driver's):
//   nvcc -gencode=arch=compute_120f,code=sm_120f CudaContextFloor.cu -o CudaContextFloor1Arch.exe -lcublasLt
//   nvcc -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_120,code=sm_120 CudaContextFloor.cu -o CudaContextFloor2Arch.exe -lcublasLt

#include <cstdio>
#include <cuda_runtime.h>
#include <cublasLt.h>

static void report( const char* stage )
{
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo( &free_b, &total_b );

    printf( "  %-34s free=%5zu MiB  used=%5zu MiB\n",
            stage, free_b / ( 1024 * 1024 ),
            ( total_b - free_b ) / ( 1024 * 1024 ) );
}

__global__ void touch( float* p ) { if ( p ) p[0] = 1.0f; }

int main()
{
    cudaDeviceProp prop{};
    cudaGetDeviceProperties( &prop, 0 );
    printf( "%s (SM %d.%d)\n", prop.name, prop.major, prop.minor );

    // Force context creation.
    cudaFree( nullptr );
    report( "after CUDA context" );

    // Force this TU's module (both cubins, in the 2-arch build) to load.
    float* d = nullptr;
    cudaMalloc( &d, sizeof( float ) );
    touch<<<1, 1>>>( d );
    cudaDeviceSynchronize();
    report( "after kernel module load" );

    cublasLtHandle_t lt = nullptr;
    cublasLtCreate( &lt );
    report( "after cublasLtCreate" );

    // A real matmul forces cuBLASLt to page in its kernel tables.
    void *A = nullptr, *B = nullptr, *C = nullptr, *ws = nullptr;
    cudaMalloc( &A, 1024ull * 1024 * 2 );
    cudaMalloc( &B, 1024ull * 1024 * 2 );
    cudaMalloc( &C, 1024ull * 1024 * 2 );
    cudaMalloc( &ws, 32ull * 1024 * 1024 );

    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatmulDescCreate( &op, CUBLAS_COMPUTE_32F, CUDA_R_32F );

    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtMatrixLayoutCreate( &la, CUDA_R_16BF, 512, 512, 512 );
    cublasLtMatrixLayoutCreate( &lb, CUDA_R_16BF, 512, 512, 512 );
    cublasLtMatrixLayoutCreate( &lc, CUDA_R_16BF, 512, 512, 512 );

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate( &pref );
    size_t ws_size = 32ull * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute( pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &ws_size, sizeof( ws_size ) );

    cublasLtMatmulHeuristicResult_t h{};
    int got = 0;
    cublasLtMatmulAlgoGetHeuristic( lt, op, la, lb, lc, lc, pref, 1, &h, &got );

    const float alpha = 1.f, beta = 0.f;

    if ( got > 0 )
    {
        cublasLtMatmul( lt, op, &alpha, A, la, B, lb, &beta, C, lc, C, lc,
                        &h.algo, ws, ws_size, nullptr );
        cudaDeviceSynchronize();
    }

    report( "after first cuBLASLt matmul" );

    printf( "  (the 4 scratch allocations above total 38 MiB -- subtract them)\n" );

    return 0;
}
