// SM120 tensor-core instruction throughput probe.
//
// Prices the CEILING for the CudaLinearOp prefill ladder before any tiling work:
//   rung 0 (ships today) : BF16 x BF16, and FP8 x FP8
//   rung 1               : mxf8f6f4 block-scaled, e2m1 weights x e4m3 activations
//   rung 2               : mxf4nvf4 block-scaled, e2m1 x e2m1 (W4A4)
//
// Measures back-to-back mma.sync issue rate from registers -- NOT a GEMM. It is an
// upper bound no tiled kernel can exceed, which is exactly what we want to know
// before deciding whether the ladder is worth climbing.
//
// Build (see README -- the -gencode form is required, -arch=sm_120f silently drops
// the family suffix and ptxas then rejects every block-scaled instruction):
//
//   nvcc -gencode=arch=compute_120f,code=sm_120f -O3 MmaInstructionPeak.cu -o MmaInstructionPeak.exe
//
// kChains / kIters below are the knobs: raising kChains and lowering kIters checks
// whether a result is issue-limited or ILP-limited. Measured 2026-09-12 as identical
// at 4 and 8 chains, so the numbers are saturated issue rates.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
    return 1; } } while (0)

// Independent accumulator chains, to hide MMA latency and measure ISSUE rate
// rather than dependent-chain latency.
constexpr int kChains = 4;
constexpr int kIters  = 4096;

// ---------------------------------------------------------------------------
// BF16 m16n8k16 -- the 1x reference instruction
// ---------------------------------------------------------------------------
__global__ void mma_bf16_kernel( float* sink, int iters )
{
    uint32_t a0 = 0x3f803f80u, a1 = 0x3f803f80u, a2 = 0x3f803f80u, a3 = 0x3f803f80u;
    uint32_t b0 = 0x3f803f80u, b1 = 0x3f803f80u;

    float d[kChains][4];
#pragma unroll
    for ( int c = 0; c < kChains; ++c )
    {
        d[c][0] = 0.f; d[c][1] = 0.f; d[c][2] = 0.f; d[c][3] = 0.f;
    }

    for ( int i = 0; i < iters; ++i )
    {
#pragma unroll
        for ( int c = 0; c < kChains; ++c )
        {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1) );
        }
    }

    float acc = 0.f;
#pragma unroll
    for ( int c = 0; c < kChains; ++c )
    {
        acc += d[c][0] + d[c][1] + d[c][2] + d[c][3];
    }

    if ( acc == 1234.5678f ) sink[0] = acc;  // never true; defeats dead-code elimination
}

// ---------------------------------------------------------------------------
// FP8 e4m3 m16n8k32 -- what the shipped W4A8 prefill path runs through cuBLASLt
// ---------------------------------------------------------------------------
__global__ void mma_fp8_kernel( float* sink, int iters )
{
    uint32_t a0 = 0x38383838u, a1 = 0x38383838u, a2 = 0x38383838u, a3 = 0x38383838u;
    uint32_t b0 = 0x38383838u, b1 = 0x38383838u;

    float d[kChains][4];
#pragma unroll
    for ( int c = 0; c < kChains; ++c )
    {
        d[c][0] = 0.f; d[c][1] = 0.f; d[c][2] = 0.f; d[c][3] = 0.f;
    }

    for ( int i = 0; i < iters; ++i )
    {
#pragma unroll
        for ( int c = 0; c < kChains; ++c )
        {
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1) );
        }
    }

    float acc = 0.f;
#pragma unroll
    for ( int c = 0; c < kChains; ++c )
    {
        acc += d[c][0] + d[c][1] + d[c][2] + d[c][3];
    }

    if ( acc == 1234.5678f ) sink[0] = acc;
}

// ---------------------------------------------------------------------------
// RUNG 1: mxf8f6f4 block-scaled, e2m1 (weights) x e4m3 (activations), m16n8k32.
// Operand types are independent for this kind; f4/f6 values ride in 8-bit
// containers, so A is still 4 x .b32.
// ---------------------------------------------------------------------------
__global__ void mma_mxf8f6f4_kernel( float* sink, int iters )
{
    uint32_t a0 = 0x02020202u, a1 = 0x02020202u, a2 = 0x02020202u, a3 = 0x02020202u;
    uint32_t b0 = 0x38383838u, b1 = 0x38383838u;
    uint32_t sfa = 0x7f7f7f7fu;  // ue8m0 scale, exponent bias -> 1.0
    uint32_t sfb = 0x7f7f7f7fu;

    float d[kChains][4];
#pragma unroll
    for ( int c = 0; c < kChains; ++c )
    {
        d[c][0] = 0.f; d[c][1] = 0.f; d[c][2] = 0.f; d[c][3] = 0.f;
    }

    for ( int i = 0; i < iters; ++i )
    {
#pragma unroll
        for ( int c = 0; c < kChains; ++c )
        {
            asm volatile(
                "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                "m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                "%10, {0, 0}, %11, {0, 0};\n"
                : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "r"(sfa), "r"(sfb) );
        }
    }

    float acc = 0.f;
#pragma unroll
    for ( int c = 0; c < kChains; ++c )
    {
        acc += d[c][0] + d[c][1] + d[c][2] + d[c][3];
    }

    if ( acc == 1234.5678f ) sink[0] = acc;
}

// ---------------------------------------------------------------------------
// RUNG 2: mxf4nvf4 block-scaled, e2m1 x e2m1, m16n8k64, block16 / ue4m3 scales.
// True 4-bit packing here (2 values per byte), so A is 4 x .b32 for 32 values.
// ---------------------------------------------------------------------------
__global__ void mma_mxf4nvf4_kernel( float* sink, int iters )
{
    uint32_t a0 = 0x22222222u, a1 = 0x22222222u, a2 = 0x22222222u, a3 = 0x22222222u;
    uint32_t b0 = 0x22222222u, b1 = 0x22222222u;
    uint32_t sfa = 0x38383838u;  // ue4m3 scale
    uint32_t sfb = 0x38383838u;

    float d[kChains][4];
#pragma unroll
    for ( int c = 0; c < kChains; ++c )
    {
        d[c][0] = 0.f; d[c][1] = 0.f; d[c][2] = 0.f; d[c][3] = 0.f;
    }

    for ( int i = 0; i < iters; ++i )
    {
#pragma unroll
        for ( int c = 0; c < kChains; ++c )
        {
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                "%10, {0, 0}, %11, {0, 0};\n"
                : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "r"(sfa), "r"(sfb) );
        }
    }

    float acc = 0.f;
#pragma unroll
    for ( int c = 0; c < kChains; ++c )
    {
        acc += d[c][0] + d[c][1] + d[c][2] + d[c][3];
    }

    if ( acc == 1234.5678f ) sink[0] = acc;
}


// ---------------------------------------------------------------------------
// CONTROL: mxf8f6f4 with e4m3 x e4m3 -- same block-scaled datapath as rung 1 but
// NO FP4 operand. Separates "the Blackwell block-scaled datapath is fast" from
// "the FP4 operand is fast".
// ---------------------------------------------------------------------------
__global__ void mma_mxf8f6f4_fp8_kernel( float* sink, int iters )
{
    uint32_t a0 = 0x38383838u, a1 = 0x38383838u, a2 = 0x38383838u, a3 = 0x38383838u;
    uint32_t b0 = 0x38383838u, b1 = 0x38383838u;
    uint32_t sfa = 0x7f7f7f7fu;
    uint32_t sfb = 0x7f7f7f7fu;

    float d[kChains][4];
#pragma unroll
    for ( int c = 0; c < kChains; ++c )
    {
        d[c][0] = 0.f; d[c][1] = 0.f; d[c][2] = 0.f; d[c][3] = 0.f;
    }

    for ( int i = 0; i < iters; ++i )
    {
#pragma unroll
        for ( int c = 0; c < kChains; ++c )
        {
            asm volatile(
                "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                "%10, {0, 0}, %11, {0, 0};\n"
                : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "r"(sfa), "r"(sfb) );
        }
    }

    float acc = 0.f;
#pragma unroll
    for ( int c = 0; c < kChains; ++c )
    {
        acc += d[c][0] + d[c][1] + d[c][2] + d[c][3];
    }

    if ( acc == 1234.5678f ) sink[0] = acc;
}

// ---------------------------------------------------------------------------

struct Arm
{
    const char* name;
    void      (*kernel)( float*, int );
    int         m, n, k;
};

static int run_arm( const Arm& arm, float* sink, int blocks, int threads, int sm_count )
{
    const int warps_per_block = threads / 32;

    // Warm up, then take the MEDIAN of several timed runs. Report the spread so a
    // clock-ramp artefact is visible rather than silently averaged in.
    arm.kernel<<<blocks, threads>>>( sink, 256 );
    CHECK( cudaDeviceSynchronize() );

    cudaEvent_t start, stop;
    CHECK( cudaEventCreate( &start ) );
    CHECK( cudaEventCreate( &stop ) );

    std::vector<double> samples;

    for ( int rep = 0; rep < 7; ++rep )
    {
        CHECK( cudaEventRecord( start ) );
        arm.kernel<<<blocks, threads>>>( sink, kIters );
        CHECK( cudaEventRecord( stop ) );
        CHECK( cudaEventSynchronize( stop ) );

        float ms = 0.f;
        CHECK( cudaEventElapsedTime( &ms, start, stop ) );

        const double mma_count = static_cast<double>( blocks ) * warps_per_block
                               * kChains * kIters;
        const double flops = mma_count * 2.0 * arm.m * arm.n * arm.k;

        samples.push_back( flops / ( ms * 1.0e-3 ) / 1.0e12 );
    }

    std::sort( samples.begin(), samples.end() );

    const double median = samples[samples.size() / 2];
    const double lo     = samples.front();
    const double hi     = samples.back();

    printf( "  %-34s %8.1f TFLOP/s   (min %.1f  max %.1f  spread %.1f%%)\n",
            arm.name, median, lo, hi, 100.0 * ( hi - lo ) / median );

    CHECK( cudaEventDestroy( start ) );
    CHECK( cudaEventDestroy( stop ) );

    return 0;
}

int main()
{
    int device = 0;
    CHECK( cudaGetDevice( &device ) );

    cudaDeviceProp prop{};
    CHECK( cudaGetDeviceProperties( &prop, device ) );

    printf( "Device: %s  (SM %d.%d, %d SMs, %zu MiB)\n\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount,
            prop.totalGlobalMem / ( 1024 * 1024 ) );

    const int threads = 256;                             // 8 warps
    const int blocks  = prop.multiProcessorCount * 4;    // saturate issue

    float* sink = nullptr;
    CHECK( cudaMalloc( &sink, sizeof( float ) ) );

    const Arm arms[] = {
        { "BF16 m16n8k16 (rung 0 ref)",        mma_bf16_kernel,      16, 8, 16 },
        { "FP8 e4m3 m16n8k32 (rung 0)",        mma_fp8_kernel,       16, 8, 32 },
        { "mxf8f6f4 e4m3xe4m3 k32 (control)",  mma_mxf8f6f4_fp8_kernel, 16, 8, 32 },
        { "mxf8f6f4 e2m1xe4m3 k32 (rung 1)",   mma_mxf8f6f4_kernel,  16, 8, 32 },
        { "mxf4nvf4 e2m1xe2m1 k64 (rung 2)",   mma_mxf4nvf4_kernel,  16, 8, 64 },
    };

    printf( "Instruction issue rate, %d blocks x %d threads, %d chains x %d iters:\n",
            blocks, threads, kChains, kIters );

    for ( const Arm& arm : arms )
    {
        if ( run_arm( arm, sink, blocks, threads, prop.multiProcessorCount ) != 0 )
        {
            return 1;
        }
    }

    CHECK( cudaFree( sink ) );

    return 0;
}
