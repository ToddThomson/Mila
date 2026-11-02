// What a verify of R rows costs against one decode, for the Gemma 4 12B's Q4_0 weights (Gemma4Mtp.md 4.7).
//
// A speculative verify runs R = K + 1 tokens through decode's arithmetic. Every weight is read once either way, so
// the floor is one decode's bandwidth; what differs is the work per weight. The arms, each over the 12B's six
// Linear shapes, DRAM-resident (a rotation of copies larger than L2), each launch inside a CUDA graph:
//
//   decode       today's matvec_decode_bf16_qint4_kernel, one row (CudaMatVecBias.Bf16.cu, copied verbatim)
//   row-matvec   the same kernel with R accumulators: each weight unpacked once, every row in today's order, so each
//                row is bit-identical to a one-row decode; its CUDA-core work grows with R
//   row-mma      mma.m16n8k16 with the R rows as the n = 8 operand: codes widened to BF16 once, per-group scales on
//                the accumulator, split-K across a block's warps and reduced in a fixed order; its tensor-core work is
//                the same at every R up to 8 (rows as the narrow operand with weights widened in registers is the idea
//                of Frantar et al., "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language
//                Models", arXiv 2408.11743; its weight reorder and shared-memory pipeline are not used)
//
// row-mma's k order is permuted so that lane t of a quad owns code word t of each group and the matching 8
// activations: one 32-bit weight load per row and one 16-byte activation load per group, no shuffle.
//
//   cute         CuTe's TiledMMA of SM80_16x8x16 atoms in the canonical sm80 structure: a multi-stage cp.async
//                pipeline of packed weights, activations and scales into shared memory, a pass widening the codes
//                into a swizzled BF16 tile, ldmatrix into the fragments, cute::gemm per group of 32 with the scales on
//                the accumulator. A CUDA block owns 16 x warps output channels and the whole reduction. Compiled when
//                CUTLASS's include directory is on the path (CUTLASS has no sm_120 collective for INT4 x BF16, so the
//                kernel is written from CuTe's primitives).
//
// Direction, written before the run: the loop needs the per-token Linear time of R = 5 rows near one decode's
// (Gemma4Mtp.md 5.1 predicts from v = 1.0 to 1.3). row-mma at R = 5 within 1.3x of decode supports the design; its
// R = 1 within 5% of decode makes 4.7's option (a) -- decode moves to the same kernel -- worth pricing end to end.
//
//   nvcc -gencode=arch=compute_120,code=sm_120 -O3 VerifyRows.cu -o VerifyRows.exe
//   nvcc ... -std=c++17 --expt-relaxed-constexpr -Xcompiler=/Zc:__cplusplus -I<cutlass>/include ...   (adds cute)
//   CUDA_VISIBLE_DEVICES=GPU-<uuid> VerifyRows.exe

#if __has_include( <cute/tensor.hpp> )
#define VERIFY_ROWS_HAS_CUTE 1
#include <cute/tensor.hpp>
#endif

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (0)

constexpr int kGroupSize = 32;
constexpr int kMaxRows = 8;
constexpr size_t kRotationBytes = size_t( 512 ) << 20;    // 16x the 5060 Ti's 32 MB L2
constexpr int kTrials = 7;

// ---------------------------------------------------------------------------------------------------------------
// decode: today's kernel, verbatim apart from bias (Gemma's projections have none)
// ---------------------------------------------------------------------------------------------------------------

constexpr int kMatvecThreadsPerOC = 32;
constexpr int kMatvecBlockOC = 8;

__device__ __forceinline__ float int4_step( uint32_t word, int index )
{
    return __uint_as_float( 0x4B000000u | ( ( word >> ( 4 * index ) ) & 0xFu ) ) - 8388616.0f;
}

template<int kNibblesPerThread>
__global__ void __launch_bounds__( kMatvecThreadsPerOC* kMatvecBlockOC )
    matvec_decode_bf16_qint4_kernel(
        __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ weights_packed,
        const __half* __restrict__ scales, int C, int OC )
{
    constexpr int kSubWords = kNibblesPerThread / 8;

    const int oc = blockIdx.x * kMatvecBlockOC + threadIdx.y;

    if ( oc >= OC ) return;

    const uint8_t* w_row = weights_packed + static_cast<int64_t>( oc ) * ( C / 2 );
    const int num_groups = C / kGroupSize;

    const int c_start = threadIdx.x * kNibblesPerThread;
    const int c_step = kMatvecThreadsPerOC * kNibblesPerThread;

    uint32_t w_stage[ kSubWords ];
    int4 x_stage[ kSubWords ];
    __half scale_stage{};

    const auto stage = [&]( int chunk_c )
    {
        if constexpr ( kNibblesPerThread == 32 )
        {
            const int4 w_packed4 = *reinterpret_cast<const int4*>( w_row + chunk_c / 2 );
            w_stage[ 0 ] = static_cast<uint32_t>( w_packed4.x );
            w_stage[ 1 ] = static_cast<uint32_t>( w_packed4.y );
            w_stage[ 2 ] = static_cast<uint32_t>( w_packed4.z );
            w_stage[ 3 ] = static_cast<uint32_t>( w_packed4.w );
        }
        else
        {
            const int2 w_packed2 = *reinterpret_cast<const int2*>( w_row + chunk_c / 2 );
            w_stage[ 0 ] = static_cast<uint32_t>( w_packed2.x );
            w_stage[ 1 ] = static_cast<uint32_t>( w_packed2.y );
        }

#pragma unroll
        for ( int j = 0; j < kSubWords; ++j )
            x_stage[ j ] = *reinterpret_cast<const int4*>( x + chunk_c + j * 8 );

        scale_stage = scales[ static_cast<int64_t>( oc ) * num_groups + chunk_c / kGroupSize ];
    };

    float acc = 0.0f;

    if ( c_start < C )
        stage( c_start );

    for ( int c = c_start; c < C; c += c_step )
    {
        uint32_t w_words[ kSubWords ];
        int4 x_words[ kSubWords ];

#pragma unroll
        for ( int j = 0; j < kSubWords; ++j )
        {
            w_words[ j ] = w_stage[ j ];
            x_words[ j ] = x_stage[ j ];
        }

        const float scale = __half2float( scale_stage );

        if ( c + c_step < C )
            stage( c + c_step );

        float sub_even = 0.0f;
        float sub_odd = 0.0f;

#pragma unroll
        for ( int j = 0; j < kSubWords; ++j )
        {
            const __nv_bfloat162* x_pairs = reinterpret_cast<const __nv_bfloat162*>( &x_words[ j ] );

            const float2 x0 = __bfloat1622float2( x_pairs[ 0 ] );
            const float2 x1 = __bfloat1622float2( x_pairs[ 1 ] );
            const float2 x2 = __bfloat1622float2( x_pairs[ 2 ] );
            const float2 x3 = __bfloat1622float2( x_pairs[ 3 ] );

            const uint32_t w = w_words[ j ];
            float& sub = ( j % 2 == 0 ) ? sub_even : sub_odd;

            sub += x0.x * int4_step( w, 0 ) + x0.y * int4_step( w, 1 )
                 + x1.x * int4_step( w, 2 ) + x1.y * int4_step( w, 3 )
                 + x2.x * int4_step( w, 4 ) + x2.y * int4_step( w, 5 )
                 + x3.x * int4_step( w, 6 ) + x3.y * int4_step( w, 7 );
        }

        acc = fmaf( scale, sub_even + sub_odd, acc );
    }

#pragma unroll
    for ( int offset = kMatvecThreadsPerOC / 2; offset > 0; offset >>= 1 )
    {
        acc += __shfl_down_sync( 0xffffffff, acc, offset );
    }

    if ( threadIdx.x == 0 )
        y[ oc ] = __float2bfloat16( acc );
}

// ---------------------------------------------------------------------------------------------------------------
// row-matvec: today's kernel with R rows. Activations are read at use rather than staged (they are L1-resident and
// staging R rows would spill); the arithmetic per row is the one-row kernel's, expression for expression.
// ---------------------------------------------------------------------------------------------------------------

template<int kRows, int kNibblesPerThread>
__global__ void __launch_bounds__( kMatvecThreadsPerOC* kMatvecBlockOC )
    matvec_rows_bf16_qint4_kernel(
        __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ weights_packed,
        const __half* __restrict__ scales, int C, int OC )
{
    constexpr int kSubWords = kNibblesPerThread / 8;

    const int oc = blockIdx.x * kMatvecBlockOC + threadIdx.y;

    if ( oc >= OC ) return;

    const uint8_t* w_row = weights_packed + static_cast<int64_t>( oc ) * ( C / 2 );
    const int num_groups = C / kGroupSize;

    const int c_start = threadIdx.x * kNibblesPerThread;
    const int c_step = kMatvecThreadsPerOC * kNibblesPerThread;

    uint32_t w_stage[ kSubWords ];
    __half scale_stage{};

    const auto stage = [&]( int chunk_c )
    {
        if constexpr ( kNibblesPerThread == 32 )
        {
            const int4 w_packed4 = *reinterpret_cast<const int4*>( w_row + chunk_c / 2 );
            w_stage[ 0 ] = static_cast<uint32_t>( w_packed4.x );
            w_stage[ 1 ] = static_cast<uint32_t>( w_packed4.y );
            w_stage[ 2 ] = static_cast<uint32_t>( w_packed4.z );
            w_stage[ 3 ] = static_cast<uint32_t>( w_packed4.w );
        }
        else
        {
            const int2 w_packed2 = *reinterpret_cast<const int2*>( w_row + chunk_c / 2 );
            w_stage[ 0 ] = static_cast<uint32_t>( w_packed2.x );
            w_stage[ 1 ] = static_cast<uint32_t>( w_packed2.y );
        }

        scale_stage = scales[ static_cast<int64_t>( oc ) * num_groups + chunk_c / kGroupSize ];
    };

    float acc[ kRows ];

#pragma unroll
    for ( int r = 0; r < kRows; ++r )
        acc[ r ] = 0.0f;

    if ( c_start < C )
        stage( c_start );

    for ( int c = c_start; c < C; c += c_step )
    {
        uint32_t w_words[ kSubWords ];

#pragma unroll
        for ( int j = 0; j < kSubWords; ++j )
            w_words[ j ] = w_stage[ j ];

        const float scale = __half2float( scale_stage );

        if ( c + c_step < C )
            stage( c + c_step );

        float sub_even[ kRows ];
        float sub_odd[ kRows ];

#pragma unroll
        for ( int r = 0; r < kRows; ++r )
        {
            sub_even[ r ] = 0.0f;
            sub_odd[ r ] = 0.0f;
        }

#pragma unroll
        for ( int j = 0; j < kSubWords; ++j )
        {
            const uint32_t w = w_words[ j ];

#pragma unroll
            for ( int r = 0; r < kRows; ++r )
            {
                const int4 x_word = __ldg( reinterpret_cast<const int4*>( x + static_cast<int64_t>( r ) * C + c + j * 8 ) );
                const __nv_bfloat162* x_pairs = reinterpret_cast<const __nv_bfloat162*>( &x_word );

                const float2 x0 = __bfloat1622float2( x_pairs[ 0 ] );
                const float2 x1 = __bfloat1622float2( x_pairs[ 1 ] );
                const float2 x2 = __bfloat1622float2( x_pairs[ 2 ] );
                const float2 x3 = __bfloat1622float2( x_pairs[ 3 ] );

                float& sub = ( j % 2 == 0 ) ? sub_even[ r ] : sub_odd[ r ];

                sub += x0.x * int4_step( w, 0 ) + x0.y * int4_step( w, 1 )
                     + x1.x * int4_step( w, 2 ) + x1.y * int4_step( w, 3 )
                     + x2.x * int4_step( w, 4 ) + x2.y * int4_step( w, 5 )
                     + x3.x * int4_step( w, 6 ) + x3.y * int4_step( w, 7 );
            }
        }

#pragma unroll
        for ( int r = 0; r < kRows; ++r )
            acc[ r ] = fmaf( scale, sub_even[ r ] + sub_odd[ r ], acc[ r ] );
    }

#pragma unroll
    for ( int r = 0; r < kRows; ++r )
    {
#pragma unroll
        for ( int offset = kMatvecThreadsPerOC / 2; offset > 0; offset >>= 1 )
        {
            acc[ r ] += __shfl_down_sync( 0xffffffff, acc[ r ], offset );
        }
    }

    if ( threadIdx.x == 0 )
    {
#pragma unroll
        for ( int r = 0; r < kRows; ++r )
            y[ static_cast<int64_t>( r ) * OC + oc ] = __float2bfloat16( acc[ r ] );
    }
}

// ---------------------------------------------------------------------------------------------------------------
// row-mma
// ---------------------------------------------------------------------------------------------------------------

// Codes 2p (low half) and 2p + 1 (high half) of an 8-code word as BF16 steps, exactly: 0x43 0x0n is 128 + n.
__device__ __forceinline__ uint32_t int4_pair_bf16x2( uint32_t word, int pair )
{
    const uint32_t byte = ( word >> ( 8 * pair ) ) & 0xFFu;
    const uint32_t bits = ( byte & 0x0Fu ) | ( ( byte & 0xF0u ) << 12 ) | 0x43004300u;
    const __nv_bfloat162 biased = *reinterpret_cast<const __nv_bfloat162*>( &bits );
    const __nv_bfloat162 step = __hsub2( biased, __floats2bfloat162_rn( 136.0f, 136.0f ) );

    return *reinterpret_cast<const uint32_t*>( &step );
}

__device__ __forceinline__ void mma_bf16_16816( float ( &d )[ 4 ], const uint32_t ( &a )[ 4 ], uint32_t b0, uint32_t b1 )
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"( d[ 0 ] ), "+f"( d[ 1 ] ), "+f"( d[ 2 ] ), "+f"( d[ 3 ] )
        : "r"( a[ 0 ] ), "r"( a[ 1 ] ), "r"( a[ 2 ] ), "r"( a[ 3 ] ), "r"( b0 ), "r"( b1 ) );
}

// A block owns 16 * kTiles output channels; its kWarps warps split the reduction by chunks of two groups (64 codes,
// 32 bytes a row: one sector) and meet in shared memory. Lane (g, t) holds output channels g and g + 8 of each tile
// and, as the B operand, activation row g. x always holds kMaxRows rows; columns past R are computed and not stored.
template<int kTiles, int kWarps>
__global__ void __launch_bounds__( kWarps * 32 )
    rows_mma_bf16_qint4_kernel(
        __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ weights_packed,
        const __half* __restrict__ scales, int C, int OC, int R )
{
    __shared__ float partial[ kWarps ][ kTiles * 128 ];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int oc_base = blockIdx.x * 16 * kTiles;
    const int row_bytes = C / 2;
    const int num_groups = C / kGroupSize;
    const int chunks = C / 64;

    const __nv_bfloat16* x_row = x + static_cast<int64_t>( g ) * C;

    uint32_t w_stage[ kTiles ][ 4 ];      // row g group 0, row g group 1, row g + 8 group 0, row g + 8 group 1
    uint32_t s_stage[ kTiles ][ 2 ];      // half2 scales of the chunk's two groups: row g, row g + 8
    int4 x_stage[ 2 ];

    const auto stage = [&]( int chunk )
    {
        const int byte = chunk * 32 + 4 * t;

#pragma unroll
        for ( int i = 0; i < kTiles; ++i )
        {
            const int64_t low = static_cast<int64_t>( oc_base + 16 * i + g );
            const int64_t high = low + 8;
            const uint8_t* w_low = weights_packed + low * row_bytes + byte;
            const uint8_t* w_high = weights_packed + high * row_bytes + byte;

            w_stage[ i ][ 0 ] = __ldg( reinterpret_cast<const uint32_t*>( w_low ) );
            w_stage[ i ][ 1 ] = __ldg( reinterpret_cast<const uint32_t*>( w_low + 16 ) );
            w_stage[ i ][ 2 ] = __ldg( reinterpret_cast<const uint32_t*>( w_high ) );
            w_stage[ i ][ 3 ] = __ldg( reinterpret_cast<const uint32_t*>( w_high + 16 ) );

            s_stage[ i ][ 0 ] = __ldg( reinterpret_cast<const uint32_t*>( scales + low * num_groups + 2 * chunk ) );
            s_stage[ i ][ 1 ] = __ldg( reinterpret_cast<const uint32_t*>( scales + high * num_groups + 2 * chunk ) );
        }

        const int column = chunk * 64 + 8 * t;

        x_stage[ 0 ] = __ldg( reinterpret_cast<const int4*>( x_row + column ) );
        x_stage[ 1 ] = __ldg( reinterpret_cast<const int4*>( x_row + column + 32 ) );
    };

    float acc[ kTiles ][ 4 ];

#pragma unroll
    for ( int i = 0; i < kTiles; ++i )
        acc[ i ][ 0 ] = acc[ i ][ 1 ] = acc[ i ][ 2 ] = acc[ i ][ 3 ] = 0.0f;

    if ( warp < chunks )
        stage( warp );

    for ( int chunk = warp; chunk < chunks; chunk += kWarps )
    {
        uint32_t w_words[ kTiles ][ 4 ];
        uint32_t s_words[ kTiles ][ 2 ];
        int4 x_words[ 2 ];

#pragma unroll
        for ( int i = 0; i < kTiles; ++i )
        {
#pragma unroll
            for ( int k = 0; k < 4; ++k )
                w_words[ i ][ k ] = w_stage[ i ][ k ];

            s_words[ i ][ 0 ] = s_stage[ i ][ 0 ];
            s_words[ i ][ 1 ] = s_stage[ i ][ 1 ];
        }

        x_words[ 0 ] = x_stage[ 0 ];
        x_words[ 1 ] = x_stage[ 1 ];

        if ( chunk + kWarps < chunks )
            stage( chunk + kWarps );

#pragma unroll
        for ( int group = 0; group < 2; ++group )
        {
            // Logical k 2t + e of step s is code 4s + e of word t; logical k 2t + 8 + e is code 4s + 2 + e. The
            // activations follow the same map, so the B operand of the group is the lane's 8 contiguous columns.
            const uint32_t b[ 4 ] = {
                static_cast<uint32_t>( x_words[ group ].x ), static_cast<uint32_t>( x_words[ group ].y ),
                static_cast<uint32_t>( x_words[ group ].z ), static_cast<uint32_t>( x_words[ group ].w ) };

#pragma unroll
            for ( int i = 0; i < kTiles; ++i )
            {
                const uint32_t low = w_words[ i ][ group ];
                const uint32_t high = w_words[ i ][ 2 + group ];

                const uint32_t a0[ 4 ] = {
                    int4_pair_bf16x2( low, 0 ), int4_pair_bf16x2( high, 0 ),
                    int4_pair_bf16x2( low, 1 ), int4_pair_bf16x2( high, 1 ) };
                const uint32_t a1[ 4 ] = {
                    int4_pair_bf16x2( low, 2 ), int4_pair_bf16x2( high, 2 ),
                    int4_pair_bf16x2( low, 3 ), int4_pair_bf16x2( high, 3 ) };

                float raw[ 4 ] = { 0.0f, 0.0f, 0.0f, 0.0f };

                mma_bf16_16816( raw, a0, b[ 0 ], b[ 1 ] );
                mma_bf16_16816( raw, a1, b[ 2 ], b[ 3 ] );

                const __half2 scale_low = *reinterpret_cast<const __half2*>( &s_words[ i ][ 0 ] );
                const __half2 scale_high = *reinterpret_cast<const __half2*>( &s_words[ i ][ 1 ] );
                const float s_low = __half2float( group == 0 ? __low2half( scale_low ) : __high2half( scale_low ) );
                const float s_high = __half2float( group == 0 ? __low2half( scale_high ) : __high2half( scale_high ) );

                acc[ i ][ 0 ] = fmaf( s_low, raw[ 0 ], acc[ i ][ 0 ] );
                acc[ i ][ 1 ] = fmaf( s_low, raw[ 1 ], acc[ i ][ 1 ] );
                acc[ i ][ 2 ] = fmaf( s_high, raw[ 2 ], acc[ i ][ 2 ] );
                acc[ i ][ 3 ] = fmaf( s_high, raw[ 3 ], acc[ i ][ 3 ] );
            }
        }
    }

    // Accumulator element (channel m, row n) at m * 8 + n: c0, c1 are channel g rows 2t, 2t + 1; c2, c3 channel g + 8.
#pragma unroll
    for ( int i = 0; i < kTiles; ++i )
    {
        float* tile = &partial[ warp ][ i * 128 ];
        tile[ g * 8 + 2 * t ] = acc[ i ][ 0 ];
        tile[ g * 8 + 2 * t + 1 ] = acc[ i ][ 1 ];
        tile[ ( g + 8 ) * 8 + 2 * t ] = acc[ i ][ 2 ];
        tile[ ( g + 8 ) * 8 + 2 * t + 1 ] = acc[ i ][ 3 ];
    }

    __syncthreads();

    constexpr int kChannels = 16 * kTiles;

    for ( int o = threadIdx.x; o < kChannels * kMaxRows; o += kWarps * 32 )
    {
        const int m = o % kChannels;
        const int n = o / kChannels;

        if ( n >= R ) break;

        const int index = ( m / 16 ) * 128 + ( m % 16 ) * 8 + n;
        float sum = 0.0f;

#pragma unroll
        for ( int w = 0; w < kWarps; ++w )
            sum += partial[ w ][ index ];

        y[ static_cast<int64_t>( n ) * OC + oc_base + m ] = __float2bfloat16( sum );
    }
}

// ---------------------------------------------------------------------------------------------------------------
// cute
// ---------------------------------------------------------------------------------------------------------------

#ifdef VERIFY_ROWS_HAS_CUTE

template<int kWarpsM, int kStages>
struct CuteRows
{
    static constexpr int kThreads = 32 * kWarpsM;
    static constexpr int kBM = 16 * kWarpsM;
    static constexpr int kBK = 256;                     // columns a stage: 128 packed bytes a row
    static constexpr int kGroupsPerStage = kBK / kGroupSize;

    using TiledMma = decltype( cute::make_tiled_mma(
        cute::SM80_16x8x16_F32BF16BF16F32_TN{}, cute::Layout<cute::Shape<cute::Int<kWarpsM>, cute::_1, cute::_1>>{} ) );

    // 128-byte rows of BF16, swizzled so ldmatrix reads are free of bank conflicts.
    using SmemAtom = decltype( cute::composition( cute::Swizzle<3, 3, 3>{},
        cute::Layout<cute::Shape<cute::_8, cute::_64>, cute::Stride<cute::_64, cute::_1>>{} ) );
    using SmemLayoutA = decltype( cute::tile_to_shape( SmemAtom{}, cute::Shape<cute::Int<kBM>, cute::Int<kBK>>{} ) );
    using SmemLayoutX = decltype( cute::tile_to_shape( SmemAtom{},
        cute::Shape<cute::_8, cute::Int<kBK>, cute::Int<kStages>>{} ) );
    using SmemLayoutPacked = cute::Layout<cute::Shape<cute::Int<kBM>, cute::Int<kBK / 2>, cute::Int<kStages>>,
        cute::Stride<cute::Int<kBK / 2>, cute::_1, cute::Int<kBM * kBK / 2>>>;

    using CopyPacked = decltype( cute::make_tiled_copy(
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, uint8_t>{},
        cute::Layout<cute::Shape<cute::Int<kThreads / 8>, cute::_8>, cute::Stride<cute::_8, cute::_1>>{},
        cute::Layout<cute::Shape<cute::_1, cute::_16>>{} ) );
    using CopyX = decltype( cute::make_tiled_copy(
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::bfloat16_t>{},
        cute::Layout<cute::Shape<cute::Int<kThreads / 32>, cute::_32>, cute::Stride<cute::_32, cute::_1>>{},
        cute::Layout<cute::Shape<cute::_1, cute::_8>>{} ) );

    static constexpr size_t kBytesA = sizeof( cute::bfloat16_t ) * kBM * kBK;
    static constexpr size_t kBytesX = sizeof( cute::bfloat16_t ) * 8 * kBK * kStages;
    static constexpr size_t kBytesPacked = size_t( kBM ) * ( kBK / 2 ) * kStages;
    static constexpr size_t kBytesScales = sizeof( __half ) * kBM * kGroupsPerStage * kStages;
    static constexpr size_t kSmemBytes = kBytesA + kBytesX + kBytesPacked + kBytesScales;
};

template<int kWarpsM, int kStages>
__global__ void __launch_bounds__( 32 * kWarpsM )
    cute_rows_bf16_qint4_kernel(
        __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ weights_packed,
        const __half* __restrict__ scales, int C, int OC, int R )
{
    using namespace cute;
    using Cfg = CuteRows<kWarpsM, kStages>;
    constexpr int kBM = Cfg::kBM;
    constexpr int kBK = Cfg::kBK;
    constexpr int kGroups = Cfg::kGroupsPerStage;

    extern __shared__ __align__( 128 ) uint8_t smem[];
    auto* smem_a = reinterpret_cast<bfloat16_t*>( smem );
    auto* smem_x = reinterpret_cast<bfloat16_t*>( smem + Cfg::kBytesA );
    auto* smem_packed = smem + Cfg::kBytesA + Cfg::kBytesX;
    auto* smem_scales = reinterpret_cast<__half*>( smem + Cfg::kBytesA + Cfg::kBytesX + Cfg::kBytesPacked );

    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * kBM;
    const int k_tiles = C / kBK;
    const int num_groups = C / kGroupSize;

    Tensor g_packed = make_tensor( make_gmem_ptr( weights_packed + static_cast<int64_t>( m0 ) * ( C / 2 ) ),
        make_shape( Int<kBM>{}, C / 2 ), make_stride( C / 2, _1{} ) );
    Tensor g_packed_tiles = local_tile( g_packed, make_shape( Int<kBM>{}, Int<kBK / 2>{} ), make_coord( 0, _ ) );
    Tensor g_x = make_tensor( make_gmem_ptr( reinterpret_cast<const bfloat16_t*>( x ) ),
        make_shape( _8{}, C ), make_stride( C, _1{} ) );
    Tensor g_x_tiles = local_tile( g_x, make_shape( _8{}, Int<kBK>{} ), make_coord( 0, _ ) );

    Tensor s_packed = make_tensor( make_smem_ptr( smem_packed ), typename Cfg::SmemLayoutPacked{} );
    Tensor s_x = make_tensor( make_smem_ptr( smem_x ), typename Cfg::SmemLayoutX{} );
    Tensor s_a = make_tensor( make_smem_ptr( smem_a ), typename Cfg::SmemLayoutA{} );

    typename Cfg::CopyPacked copy_packed;
    auto thread_packed = copy_packed.get_slice( tid );
    Tensor packed_source = thread_packed.partition_S( g_packed_tiles );
    Tensor packed_destination = thread_packed.partition_D( s_packed );

    typename Cfg::CopyX copy_x;
    auto thread_x = copy_x.get_slice( tid );
    Tensor x_source = thread_x.partition_S( g_x_tiles );
    Tensor x_destination = thread_x.partition_D( s_x );

    const auto load_stage = [&]( int k_tile )
    {
        const int slot = k_tile % kStages;
        copy( copy_packed, packed_source( _, _, _, k_tile ), packed_destination( _, _, _, slot ) );
        copy( copy_x, x_source( _, _, _, k_tile ), x_destination( _, _, _, slot ) );

        // One 16-byte row of eight group scales per output channel.
        if ( tid < kBM )
        {
            SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>::copy(
                *reinterpret_cast<const uint128_t*>( scales + static_cast<int64_t>( m0 + tid ) * num_groups + k_tile * kGroups ),
                *reinterpret_cast<uint128_t*>( smem_scales + ( slot * kBM + tid ) * kGroups ) );
        }
    };

    typename Cfg::TiledMma mma;
    auto thread_mma = mma.get_slice( tid );
    Tensor fragment_a = thread_mma.partition_fragment_A( s_a );                       // (MMA, MMA_M, MMA_K)
    Tensor fragment_b = thread_mma.partition_fragment_B( s_x( _, _, 0 ) );            // (MMA, MMA_N, MMA_K)
    Tensor acc = partition_fragment_C( mma, cute::Shape<Int<kBM>, _8>{} );
    Tensor raw = make_fragment_like( acc );
    clear( acc );

    auto copy_a = make_tiled_copy_A( Copy_Atom<SM75_U32x4_LDSM_N, bfloat16_t>{}, mma );
    auto thread_a = copy_a.get_slice( tid );
    Tensor a_source = thread_a.partition_S( s_a );
    Tensor a_destination = thread_a.retile_D( fragment_a );

    auto copy_b = make_tiled_copy_B( Copy_Atom<SM75_U32x2_LDSM_N, bfloat16_t>{}, mma );
    auto thread_b = copy_b.get_slice( tid );
    Tensor b_source = thread_b.partition_S( s_x );
    Tensor b_destination = thread_b.retile_D( fragment_b );

    Tensor coordinates = thread_mma.partition_C( make_identity_tensor( cute::Shape<Int<kBM>, _8>{} ) );

#pragma unroll
    for ( int stage = 0; stage < kStages - 1; ++stage )
    {
        if ( stage < k_tiles )
            load_stage( stage );

        cp_async_fence();
    }

    for ( int k_tile = 0; k_tile < k_tiles; ++k_tile )
    {
        const int slot = k_tile % kStages;

        cp_async_wait<kStages - 2>();
        __syncthreads();

        // Codes to exact BF16 steps, eight per 32-bit word, into the swizzled tile.
        for ( int index = tid; index < kBM * kBK / 8; index += Cfg::kThreads )
        {
            const int m = index / ( kBK / 8 );
            const int k = ( index % ( kBK / 8 ) ) * 8;
            const uint32_t word = *reinterpret_cast<const uint32_t*>( &s_packed( m, k / 2, slot ) );
            const uint4 steps = make_uint4( int4_pair_bf16x2( word, 0 ), int4_pair_bf16x2( word, 1 ),
                int4_pair_bf16x2( word, 2 ), int4_pair_bf16x2( word, 3 ) );
            *reinterpret_cast<uint4*>( &s_a( m, k ) ) = steps;
        }

        if ( k_tile + kStages - 1 < k_tiles )
            load_stage( k_tile + kStages - 1 );

        cp_async_fence();
        __syncthreads();

#pragma unroll
        for ( int group = 0; group < kGroups; ++group )
        {
#pragma unroll
            for ( int step = 2 * group; step < 2 * group + 2; ++step )
            {
                copy( copy_a, a_source( _, _, step ), a_destination( _, _, step ) );
                copy( copy_b, b_source( _, _, step, slot ), b_destination( _, _, step ) );
            }

            clear( raw );
            gemm( mma, fragment_a( _, _, 2 * group ), fragment_b( _, _, 2 * group ), raw );
            gemm( mma, fragment_a( _, _, 2 * group + 1 ), fragment_b( _, _, 2 * group + 1 ), raw );

#pragma unroll
            for ( int i = 0; i < size( raw ); ++i )
            {
                const int m = get<0>( coordinates( i ) );
                const float scale = __half2float( smem_scales[ ( slot * kBM + m ) * kGroups + group ] );
                acc( i ) = fmaf( scale, raw( i ), acc( i ) );
            }
        }
    }

#pragma unroll
    for ( int i = 0; i < size( acc ); ++i )
    {
        const int m = get<0>( coordinates( i ) );
        const int n = get<1>( coordinates( i ) );

        if ( n < R )
            y[ static_cast<int64_t>( n ) * OC + m0 + m ] = __float2bfloat16( acc( i ) );
    }
}

// cute-split: row-mma's structure in CuTe. The k permutation is the TiledMMA's own PermutationMNK -- logical k
// e + 2t + 8h + 16s of a 32-column block is column 8t + 4s + 2h + e -- so partition_A and partition_B hand each lane
// its 8 contiguous columns, and cute::copy and cute::gemm do the rest. Warps split K as row-mma's do.
using CuteSplitPermutationK = cute::Layout<cute::Shape<cute::_2, cute::_4, cute::_2, cute::_2>,
    cute::Stride<cute::_1, cute::_8, cute::_2, cute::_4>>;

template<int kWarps>
__global__ void __launch_bounds__( kWarps * 32 )
    cute_split_rows_bf16_qint4_kernel(
        __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ weights_packed,
        const __half* __restrict__ scales, int C, int OC, int R )
{
    using namespace cute;

    __shared__ float partial[ kWarps ][ 128 ];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int m0 = blockIdx.x * 16;
    const int blocks = C / kGroupSize;
    const int chunks = blocks / 2;

    auto mma = make_tiled_mma( SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<cute::Shape<_1, _1, _1>>{},
        Tile<_16, _8, CuteSplitPermutationK>{} );
    auto thread_mma = mma.get_slice( lane );

    Tensor g_a = make_tensor( make_gmem_ptr<uint4_t>( weights_packed + static_cast<int64_t>( m0 ) * ( C / 2 ) ),
        make_shape( _16{}, C ), make_stride( C, _1{} ) );
    Tensor g_b = make_tensor( make_gmem_ptr( reinterpret_cast<const bfloat16_t*>( x ) ),
        make_shape( _8{}, C ), make_stride( C, _1{} ) );
    Tensor a_blocks = local_tile( g_a, cute::Shape<_16, _32>{}, make_coord( 0, _ ) );        // (16, 32, blocks)
    Tensor b_blocks = local_tile( g_b, cute::Shape<_8, _32>{}, make_coord( 0, _ ) );         // (8, 32, blocks)

    Tensor a_source = thread_mma.partition_A( a_blocks );                                       // (MMA, 1, 2, blocks)
    Tensor b_source = thread_mma.partition_B( b_blocks );

    Tensor codes = make_fragment_like( a_source( _, _, _, 0 ) );
    Tensor codes_next = make_fragment_like( a_source( _, _, _, 0 ) );
    Tensor codes_next2 = make_fragment_like( a_source( _, _, _, 0 ) );
    Tensor codes2 = make_fragment_like( a_source( _, _, _, 0 ) );
    Tensor activations = make_fragment_like( b_source( _, _, _, 0 ) );
    Tensor activations2 = make_fragment_like( b_source( _, _, _, 0 ) );
    Tensor activations_next = make_fragment_like( b_source( _, _, _, 0 ) );
    Tensor activations_next2 = make_fragment_like( b_source( _, _, _, 0 ) );
    Tensor steps = make_tensor<bfloat16_t>( codes.layout() );

    Tensor acc = partition_fragment_C( mma, cute::Shape<_16, _8>{} );
    Tensor raw = make_fragment_like( acc );
    clear( acc );

    Tensor coordinates = thread_mma.partition_C( make_identity_tensor( cute::Shape<_16, _8>{} ) );

    const auto load = [&]( int chunk, auto& codes_first, auto& codes_second, auto& activations_first, auto& activations_second )
    {
        copy( a_source( _, _, _, 2 * chunk ), codes_first );
        copy( a_source( _, _, _, 2 * chunk + 1 ), codes_second );
        copy( b_source( _, _, _, 2 * chunk ), activations_first );
        copy( b_source( _, _, _, 2 * chunk + 1 ), activations_second );
    };

    const auto compute = [&]( int block, auto& block_codes, auto& block_activations )
    {
        // Two codes to an exact BF16 pair each: the fragment's element order is the mma's register order.
        Tensor words = recast<uint32_t>( block_codes );
        Tensor pairs = recast<uint32_t>( steps );

#pragma unroll
        for ( int w = 0; w < size( words ); ++w )
        {
#pragma unroll
            for ( int p = 0; p < 4; ++p )
                pairs( 4 * w + p ) = int4_pair_bf16x2( words( w ), p );
        }

        clear( raw );
        gemm( mma, steps, block_activations, raw );

#pragma unroll
        for ( int i = 0; i < size( raw ); ++i )
        {
            const int m = get<0>( coordinates( i ) );
            const float scale = __half2float( scales[ static_cast<int64_t>( m0 + m ) * blocks + block ] );
            acc( i ) = fmaf( scale, raw( i ), acc( i ) );
        }
    };

    if ( warp < chunks )
        load( warp, codes_next, codes_next2, activations_next, activations_next2 );

    for ( int chunk = warp; chunk < chunks; chunk += kWarps )
    {
        copy( codes_next, codes );
        copy( codes_next2, codes2 );
        copy( activations_next, activations );
        copy( activations_next2, activations2 );

        if ( chunk + kWarps < chunks )
            load( chunk + kWarps, codes_next, codes_next2, activations_next, activations_next2 );

        compute( 2 * chunk, codes, activations );
        compute( 2 * chunk + 1, codes2, activations2 );
    }

#pragma unroll
    for ( int i = 0; i < size( acc ); ++i )
    {
        const int m = get<0>( coordinates( i ) );
        const int n = get<1>( coordinates( i ) );
        partial[ warp ][ m * 8 + n ] = acc( i );
    }

    __syncthreads();

    for ( int o = threadIdx.x; o < 16 * kMaxRows; o += kWarps * 32 )
    {
        const int m = o % 16;
        const int n = o / 16;

        if ( n >= R )
            break;

        float sum = 0.0f;

#pragma unroll
        for ( int w = 0; w < kWarps; ++w )
            sum += partial[ w ][ m * 8 + n ];

        y[ static_cast<int64_t>( n ) * OC + m0 + m ] = __float2bfloat16( sum );
    }
}

template<int kWarps>
void launch_cute_split_config( int R, int C, int OC, const uint8_t* w, const __half* sc, const __nv_bfloat16* x,
    __nv_bfloat16* y, cudaStream_t stream )
{
    cute_split_rows_bf16_qint4_kernel<kWarps><<<OC / 16, kWarps * 32, 0, stream>>>( y, x, w, sc, C, OC, R );
}

template<int kWarpsM, int kStages>
void launch_cute_config( int R, int C, int OC, const uint8_t* w, const __half* sc, const __nv_bfloat16* x,
    __nv_bfloat16* y, cudaStream_t stream )
{
    using Cfg = CuteRows<kWarpsM, kStages>;
    auto kernel = cute_rows_bf16_qint4_kernel<kWarpsM, kStages>;
    static bool configured = false;

    if ( !configured )
    {
        CHECK( cudaFuncSetAttribute( kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int( Cfg::kSmemBytes ) ) );
        configured = true;
    }

    kernel<<<OC / Cfg::kBM, Cfg::kThreads, Cfg::kSmemBytes, stream>>>( y, x, w, sc, C, OC, R );
}

#endif

// ---------------------------------------------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------------------------------------------

struct Shape
{
    const char* name;
    int OC;
    int C;
    int per_token;      // 40 sliding layers and 8 global layers; the FFN is in all 48
};

// Gemma.md section 8: local qkv 3840 -> 8192, o 4096 -> 3840; global qkv 3840 -> 8704, o 8192 -> 3840; GeGLU 15360.
const Shape kShapes[] = {
    { "sliding.qkv", 8192, 3840, 40 },
    { "sliding.o", 3840, 4096, 40 },
    { "global.qkv", 8704, 3840, 8 },
    { "global.o", 3840, 8192, 8 },
    { "ffn.gate_up", 30720, 3840, 48 },
    { "ffn.down", 3840, 15360, 48 },
};

enum class Arm { Decode, RowMatvec, RowMma, Cute, CuteSplit };

// cute-split: warps splitting K, as row-mma's kWarps.
const int kCuteSplitWarps[] = { 4, 8, 16 };
constexpr int kCuteSplitConfigCount = sizeof( kCuteSplitWarps ) / sizeof( kCuteSplitWarps[ 0 ] );

struct MmaConfig { int tiles; int warps; };
const MmaConfig kMmaConfigs[] = { { 1, 4 }, { 1, 8 }, { 1, 16 }, { 2, 4 }, { 2, 8 }, { 4, 4 } };
constexpr int kMmaConfigCount = sizeof( kMmaConfigs ) / sizeof( kMmaConfigs[ 0 ] );

// cute: warps along M (16 output channels each) and pipeline stages.
struct CuteConfig { int warps; int stages; };
const CuteConfig kCuteConfigs[] = { { 1, 4 }, { 1, 8 }, { 2, 4 }, { 2, 6 }, { 4, 3 } };
constexpr int kCuteConfigCount = sizeof( kCuteConfigs ) / sizeof( kCuteConfigs[ 0 ] );

#ifdef VERIFY_ROWS_HAS_CUTE
constexpr bool kHasCute = true;
#else
constexpr bool kHasCute = false;
#endif

struct Buffers
{
    std::vector<uint8_t*> weights;
    std::vector<__half*> scales;
    __nv_bfloat16* x = nullptr;
    __nv_bfloat16* y = nullptr;
};

void launch_decode( const Shape& s, const uint8_t* w, const __half* sc, const __nv_bfloat16* x, __nv_bfloat16* y,
    cudaStream_t stream )
{
    const dim3 block( kMatvecThreadsPerOC, kMatvecBlockOC );
    const dim3 grid( ( s.OC + kMatvecBlockOC - 1 ) / kMatvecBlockOC );

    if ( s.C >= 8192 )
        matvec_decode_bf16_qint4_kernel<32><<<grid, block, 0, stream>>>( y, x, w, sc, s.C, s.OC );
    else
        matvec_decode_bf16_qint4_kernel<16><<<grid, block, 0, stream>>>( y, x, w, sc, s.C, s.OC );
}

template<int kRows>
void launch_row_matvec_rows( const Shape& s, const uint8_t* w, const __half* sc, const __nv_bfloat16* x,
    __nv_bfloat16* y, cudaStream_t stream )
{
    const dim3 block( kMatvecThreadsPerOC, kMatvecBlockOC );
    const dim3 grid( ( s.OC + kMatvecBlockOC - 1 ) / kMatvecBlockOC );

    if ( s.C >= 8192 )
        matvec_rows_bf16_qint4_kernel<kRows, 32><<<grid, block, 0, stream>>>( y, x, w, sc, s.C, s.OC );
    else
        matvec_rows_bf16_qint4_kernel<kRows, 16><<<grid, block, 0, stream>>>( y, x, w, sc, s.C, s.OC );
}

void launch_row_matvec( int R, const Shape& s, const uint8_t* w, const __half* sc, const __nv_bfloat16* x,
    __nv_bfloat16* y, cudaStream_t stream )
{
    switch ( R )
    {
        case 1: launch_row_matvec_rows<1>( s, w, sc, x, y, stream ); break;
        case 2: launch_row_matvec_rows<2>( s, w, sc, x, y, stream ); break;
        case 3: launch_row_matvec_rows<3>( s, w, sc, x, y, stream ); break;
        case 4: launch_row_matvec_rows<4>( s, w, sc, x, y, stream ); break;
        case 5: launch_row_matvec_rows<5>( s, w, sc, x, y, stream ); break;
        case 6: launch_row_matvec_rows<6>( s, w, sc, x, y, stream ); break;
        case 7: launch_row_matvec_rows<7>( s, w, sc, x, y, stream ); break;
        case 8: launch_row_matvec_rows<8>( s, w, sc, x, y, stream ); break;
    }
}

template<int kTiles, int kWarps>
void launch_row_mma_config( int R, const Shape& s, const uint8_t* w, const __half* sc, const __nv_bfloat16* x,
    __nv_bfloat16* y, cudaStream_t stream )
{
    const dim3 grid( s.OC / ( 16 * kTiles ) );
    rows_mma_bf16_qint4_kernel<kTiles, kWarps><<<grid, kWarps * 32, 0, stream>>>( y, x, w, sc, s.C, s.OC, R );
}

void launch_row_mma( int config, int R, const Shape& s, const uint8_t* w, const __half* sc, const __nv_bfloat16* x,
    __nv_bfloat16* y, cudaStream_t stream )
{
    switch ( config )
    {
        case 0: launch_row_mma_config<1, 4>( R, s, w, sc, x, y, stream ); break;
        case 1: launch_row_mma_config<1, 8>( R, s, w, sc, x, y, stream ); break;
        case 2: launch_row_mma_config<1, 16>( R, s, w, sc, x, y, stream ); break;
        case 3: launch_row_mma_config<2, 4>( R, s, w, sc, x, y, stream ); break;
        case 4: launch_row_mma_config<2, 8>( R, s, w, sc, x, y, stream ); break;
        case 5: launch_row_mma_config<4, 4>( R, s, w, sc, x, y, stream ); break;
    }
}

void launch_cute( int config, int R, const Shape& s, const uint8_t* w, const __half* sc, const __nv_bfloat16* x,
    __nv_bfloat16* y, cudaStream_t stream )
{
#ifdef VERIFY_ROWS_HAS_CUTE
    switch ( config )
    {
        case 0: launch_cute_config<1, 4>( R, s.C, s.OC, w, sc, x, y, stream ); break;
        case 1: launch_cute_config<1, 8>( R, s.C, s.OC, w, sc, x, y, stream ); break;
        case 2: launch_cute_config<2, 4>( R, s.C, s.OC, w, sc, x, y, stream ); break;
        case 3: launch_cute_config<2, 6>( R, s.C, s.OC, w, sc, x, y, stream ); break;
        case 4: launch_cute_config<4, 3>( R, s.C, s.OC, w, sc, x, y, stream ); break;
    }
#endif
}

void launch_cute_split( int config, int R, const Shape& s, const uint8_t* w, const __half* sc, const __nv_bfloat16* x,
    __nv_bfloat16* y, cudaStream_t stream )
{
#ifdef VERIFY_ROWS_HAS_CUTE
    switch ( config )
    {
        case 0: launch_cute_split_config<4>( R, s.C, s.OC, w, sc, x, y, stream ); break;
        case 1: launch_cute_split_config<8>( R, s.C, s.OC, w, sc, x, y, stream ); break;
        case 2: launch_cute_split_config<16>( R, s.C, s.OC, w, sc, x, y, stream ); break;
    }
#endif
}

void launch( Arm arm, int config, int R, const Shape& s, const Buffers& b, int copy, cudaStream_t stream )
{
    switch ( arm )
    {
        case Arm::Decode: launch_decode( s, b.weights[ copy ], b.scales[ copy ], b.x, b.y, stream ); break;
        case Arm::RowMatvec: launch_row_matvec( R, s, b.weights[ copy ], b.scales[ copy ], b.x, b.y, stream ); break;
        case Arm::RowMma: launch_row_mma( config, R, s, b.weights[ copy ], b.scales[ copy ], b.x, b.y, stream ); break;
        case Arm::Cute: launch_cute( config, R, s, b.weights[ copy ], b.scales[ copy ], b.x, b.y, stream ); break;
        case Arm::CuteSplit: launch_cute_split( config, R, s, b.weights[ copy ], b.scales[ copy ], b.x, b.y, stream ); break;
    }
}

// Microseconds per launch: a graph of launches cycling the copies, median of kTrials replays.
double time_arm( Arm arm, int config, int R, const Shape& s, const Buffers& b, cudaStream_t stream )
{
    const int copies = static_cast<int>( b.weights.size() );
    const int launches = std::max( 64, 4 * copies );

    cudaGraph_t graph;
    cudaGraphExec_t exec;

    CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );

    for ( int i = 0; i < launches; ++i )
        launch( arm, config, R, s, b, i % copies, stream );

    CHECK( cudaStreamEndCapture( stream, &graph ) );
    CHECK( cudaGraphInstantiate( &exec, graph, 0 ) );

    cudaEvent_t start, stop;
    CHECK( cudaEventCreate( &start ) );
    CHECK( cudaEventCreate( &stop ) );

    CHECK( cudaGraphLaunch( exec, stream ) );
    CHECK( cudaStreamSynchronize( stream ) );

    std::vector<double> samples;

    for ( int trial = 0; trial < kTrials; ++trial )
    {
        CHECK( cudaEventRecord( start, stream ) );
        CHECK( cudaGraphLaunch( exec, stream ) );
        CHECK( cudaEventRecord( stop, stream ) );
        CHECK( cudaEventSynchronize( stop ) );

        float ms = 0.0f;
        CHECK( cudaEventElapsedTime( &ms, start, stop ) );
        samples.push_back( 1000.0 * ms / launches );
    }

    CHECK( cudaGraphExecDestroy( exec ) );
    CHECK( cudaGraphDestroy( graph ) );
    CHECK( cudaEventDestroy( start ) );
    CHECK( cudaEventDestroy( stop ) );

    std::sort( samples.begin(), samples.end() );

    return samples[ kTrials / 2 ];
}

uint64_t g_state = 0x9E3779B97F4A7C15ull;

uint64_t next_random()
{
    g_state ^= g_state << 13;
    g_state ^= g_state >> 7;
    g_state ^= g_state << 17;

    return g_state;
}

double uniform() { return ( next_random() >> 11 ) * ( 1.0 / 9007199254740992.0 ); }

struct Host
{
    std::vector<uint8_t> weights;
    std::vector<__half> scales;
    std::vector<__nv_bfloat16> x;
};

// Sampled channels against a double-precision reference: the largest error over the outputs' RMS.
struct Accuracy { double relative_max_error; };

std::vector<int> sample_channels( int OC )
{
    std::vector<int> channels;

    for ( int m = 0; m < OC; m += std::max( 1, OC / 512 ) )
        channels.push_back( m );

    return channels;
}

std::vector<double> reference( const Shape& s, const Host& h, const std::vector<int>& channels )
{
    const int num_groups = s.C / kGroupSize;
    std::vector<double> out( channels.size() * kMaxRows );

    for ( size_t k = 0; k < channels.size(); ++k )
    {
        const int m = channels[ k ];

        for ( int n = 0; n < kMaxRows; ++n )
        {
            double total = 0.0;

            for ( int group = 0; group < num_groups; ++group )
            {
                double partial = 0.0;

                for ( int c = group * kGroupSize; c < ( group + 1 ) * kGroupSize; ++c )
                {
                    const uint8_t byte = h.weights[ static_cast<size_t>( m ) * ( s.C / 2 ) + c / 2 ];
                    const int code = ( c % 2 == 0 ) ? ( byte & 0xF ) : ( byte >> 4 );
                    partial += double( __bfloat162float( h.x[ static_cast<size_t>( n ) * s.C + c ] ) ) * ( code - 8 );
                }

                total += double( __half2float( h.scales[ static_cast<size_t>( m ) * num_groups + group ] ) ) * partial;
            }

            out[ k * kMaxRows + n ] = total;
        }
    }

    return out;
}

double relative_error( const std::vector<__nv_bfloat16>& y, int OC, int R, const std::vector<int>& channels,
    const std::vector<double>& ref )
{
    double max_error = 0.0;
    double sum_square = 0.0;

    for ( size_t k = 0; k < channels.size(); ++k )
    {
        for ( int n = 0; n < R; ++n )
        {
            const double expected = ref[ k * kMaxRows + n ];
            const double got = __bfloat162float( y[ static_cast<size_t>( n ) * OC + channels[ k ] ] );
            max_error = std::max( max_error, std::fabs( got - expected ) );
            sum_square += expected * expected;
        }
    }

    return max_error / std::sqrt( sum_square / ( channels.size() * R ) );
}

int main()
{
    cudaDeviceProp prop;
    CHECK( cudaGetDeviceProperties( &prop, 0 ) );
    printf( "%s, %d SMs, L2 %d MB\n\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20 );

    cudaStream_t stream;
    CHECK( cudaStreamCreateWithFlags( &stream, cudaStreamNonBlocking ) );

    constexpr int kShapeCount = sizeof( kShapes ) / sizeof( kShapes[ 0 ] );

    // [shape][R] microseconds; mma is the best configuration per shape and R.
    double decode_us[ kShapeCount ];
    double matvec_us[ kShapeCount ][ kMaxRows + 1 ];
    double mma_us[ kShapeCount ][ kMaxRows + 1 ];
    double mma_config_us[ kShapeCount ][ kMmaConfigCount ];
    double cute_us[ kShapeCount ][ kMaxRows + 1 ] = {};
    double cute_config_us[ kShapeCount ][ kCuteConfigCount ] = {};
    double split_us[ kShapeCount ][ kMaxRows + 1 ] = {};
    double split_config_us[ kShapeCount ][ kCuteSplitConfigCount ] = {};

    for ( int si = 0; si < kShapeCount; ++si )
    {
        const Shape& s = kShapes[ si ];
        const int num_groups = s.C / kGroupSize;
        const size_t weight_bytes = static_cast<size_t>( s.OC ) * s.C / 2;
        const size_t scale_count = static_cast<size_t>( s.OC ) * num_groups;
        const size_t bytes = weight_bytes + scale_count * sizeof( __half );
        const int copies = static_cast<int>( std::max<size_t>( 2, ( kRotationBytes + bytes - 1 ) / bytes ) );

        Host h;
        h.weights.resize( weight_bytes );
        h.scales.resize( scale_count );
        h.x.resize( static_cast<size_t>( kMaxRows ) * s.C );

        for ( auto& byte : h.weights ) byte = static_cast<uint8_t>( next_random() );
        for ( auto& scale : h.scales ) scale = __float2half( float( 0.004 + 0.012 * uniform() ) );

        for ( auto& v : h.x )
        {
            // Roughly normal activations, so the sums cancel as a real projection's do.
            const double u = uniform() + uniform() + uniform() + uniform() - 2.0;
            v = __float2bfloat16( float( 1.7 * u ) );
        }

        Buffers b;

        for ( int k = 0; k < copies; ++k )
        {
            uint8_t* w;
            __half* sc;
            CHECK( cudaMalloc( &w, weight_bytes ) );
            CHECK( cudaMalloc( &sc, scale_count * sizeof( __half ) ) );
            CHECK( cudaMemcpy( w, h.weights.data(), weight_bytes, cudaMemcpyHostToDevice ) );
            CHECK( cudaMemcpy( sc, h.scales.data(), scale_count * sizeof( __half ), cudaMemcpyHostToDevice ) );
            b.weights.push_back( w );
            b.scales.push_back( sc );
        }

        CHECK( cudaMalloc( &b.x, h.x.size() * sizeof( __nv_bfloat16 ) ) );
        CHECK( cudaMalloc( &b.y, static_cast<size_t>( kMaxRows ) * s.OC * sizeof( __nv_bfloat16 ) ) );
        CHECK( cudaMemcpy( b.x, h.x.data(), h.x.size() * sizeof( __nv_bfloat16 ), cudaMemcpyHostToDevice ) );

        // Accuracy and identity, on copy 0 with all eight rows.
        const std::vector<int> channels = sample_channels( s.OC );
        const std::vector<double> ref = reference( s, h, channels );
        const size_t y_count = static_cast<size_t>( kMaxRows ) * s.OC;

        std::vector<__nv_bfloat16> one_row( y_count );

        for ( int n = 0; n < kMaxRows; ++n )
        {
            launch_decode( s, b.weights[ 0 ], b.scales[ 0 ], b.x + static_cast<size_t>( n ) * s.C,
                b.y + static_cast<size_t>( n ) * s.OC, stream );
        }

        CHECK( cudaStreamSynchronize( stream ) );
        CHECK( cudaMemcpy( one_row.data(), b.y, y_count * sizeof( __nv_bfloat16 ), cudaMemcpyDeviceToHost ) );

        const auto compare = [&]( const std::vector<__nv_bfloat16>& y, size_t& identical, double& max_ulps )
        {
            identical = 0;
            max_ulps = 0.0;

            for ( size_t i = 0; i < y_count; ++i )
            {
                const uint16_t a = *reinterpret_cast<const uint16_t*>( &y[ i ] );
                const uint16_t c = *reinterpret_cast<const uint16_t*>( &one_row[ i ] );
                identical += ( a == c );

                const double scale = std::ldexp( 1.0, std::ilogb( std::max( 1e-30, double( std::fabs(
                    __bfloat162float( one_row[ i ] ) ) ) ) ) - 7 );
                max_ulps = std::max( max_ulps,
                    std::fabs( double( __bfloat162float( y[ i ] ) ) - __bfloat162float( one_row[ i ] ) ) / scale );
            }
        };

        std::vector<__nv_bfloat16> rows( y_count );
        size_t identical;
        double max_ulps;

        const double decode_error = relative_error( one_row, s.OC, kMaxRows, channels, ref );

        printf( "%s  [%d x %d]  %.1f MB a copy, %d copies\n", s.name, s.OC, s.C, bytes / 1048576.0, copies );
        printf( "  accuracy, 8 rows, largest error / RMS against FP64:  decode %.2e", decode_error );

        // On the benchmark's own stream: a non-blocking stream does not order against a legacy-stream memset.
        CHECK( cudaMemsetAsync( b.y, 0, y_count * sizeof( __nv_bfloat16 ), stream ) );
        launch_row_matvec( kMaxRows, s, b.weights[ 0 ], b.scales[ 0 ], b.x, b.y, stream );
        CHECK( cudaGetLastError() );
        CHECK( cudaStreamSynchronize( stream ) );
        CHECK( cudaMemcpy( rows.data(), b.y, y_count * sizeof( __nv_bfloat16 ), cudaMemcpyDeviceToHost ) );
        compare( rows, identical, max_ulps );
        printf( "  row-matvec %.2e (%zu of %zu bit-identical to decode)",
            relative_error( rows, s.OC, kMaxRows, channels, ref ), identical, y_count );

        for ( int config = 0; config < kMmaConfigCount; ++config )
        {
            CHECK( cudaMemsetAsync( b.y, 0, y_count * sizeof( __nv_bfloat16 ), stream ) );
            launch_row_mma( config, kMaxRows, s, b.weights[ 0 ], b.scales[ 0 ], b.x, b.y, stream );
            CHECK( cudaGetLastError() );
            CHECK( cudaStreamSynchronize( stream ) );
            CHECK( cudaMemcpy( rows.data(), b.y, y_count * sizeof( __nv_bfloat16 ), cudaMemcpyDeviceToHost ) );
            compare( rows, identical, max_ulps );

            const double error = relative_error( rows, s.OC, kMaxRows, channels, ref );

            if ( config == 0 )
            {
                printf( "  row-mma %.2e (%zu of %zu bit-identical to decode, largest difference %.0f BF16 ulps)\n",
                    error, identical, y_count, max_ulps );
            }
            else if ( error > 1.5 * decode_error )
            {
                printf( "  row-mma %dx%d WRONG: %.2e\n", kMmaConfigs[ config ].tiles, kMmaConfigs[ config ].warps, error );
            }
        }

        for ( int config = 0; kHasCute && config < kCuteConfigCount; ++config )
        {
            CHECK( cudaMemsetAsync( b.y, 0, y_count * sizeof( __nv_bfloat16 ), stream ) );
            launch_cute( config, kMaxRows, s, b.weights[ 0 ], b.scales[ 0 ], b.x, b.y, stream );
            CHECK( cudaGetLastError() );
            CHECK( cudaStreamSynchronize( stream ) );
            CHECK( cudaMemcpy( rows.data(), b.y, y_count * sizeof( __nv_bfloat16 ), cudaMemcpyDeviceToHost ) );
            compare( rows, identical, max_ulps );

            const double error = relative_error( rows, s.OC, kMaxRows, channels, ref );

            if ( config == 0 )
            {
                printf( "  cute %.2e (%zu of %zu bit-identical to decode, largest difference %.0f BF16 ulps)\n",
                    error, identical, y_count, max_ulps );
            }
            else if ( error > 1.5 * decode_error )
            {
                printf( "  cute %dx%d WRONG: %.2e\n", kCuteConfigs[ config ].warps, kCuteConfigs[ config ].stages, error );
            }
        }

        for ( int config = 0; kHasCute && config < kCuteSplitConfigCount; ++config )
        {
            CHECK( cudaMemsetAsync( b.y, 0, y_count * sizeof( __nv_bfloat16 ), stream ) );
            launch_cute_split( config, kMaxRows, s, b.weights[ 0 ], b.scales[ 0 ], b.x, b.y, stream );
            CHECK( cudaGetLastError() );
            CHECK( cudaStreamSynchronize( stream ) );
            CHECK( cudaMemcpy( rows.data(), b.y, y_count * sizeof( __nv_bfloat16 ), cudaMemcpyDeviceToHost ) );
            compare( rows, identical, max_ulps );

            const double error = relative_error( rows, s.OC, kMaxRows, channels, ref );

            if ( config == 0 )
            {
                printf( "  cute-split %.2e (%zu of %zu bit-identical to decode, largest difference %.0f BF16 ulps)\n",
                    error, identical, y_count, max_ulps );
            }
            else if ( error > 1.5 * decode_error )
            {
                printf( "  cute-split %d warps WRONG: %.2e\n", kCuteSplitWarps[ config ], error );
            }
        }

        // Timing.
        decode_us[ si ] = time_arm( Arm::Decode, 0, 1, s, b, stream );

        for ( int config = 0; config < kMmaConfigCount; ++config )
            mma_config_us[ si ][ config ] = time_arm( Arm::RowMma, config, kMaxRows, s, b, stream );

        int best = 0;

        for ( int config = 1; config < kMmaConfigCount; ++config )
            if ( mma_config_us[ si ][ config ] < mma_config_us[ si ][ best ] ) best = config;

        int best_cute = 0;

        for ( int config = 0; kHasCute && config < kCuteConfigCount; ++config )
        {
            cute_config_us[ si ][ config ] = time_arm( Arm::Cute, config, kMaxRows, s, b, stream );

            if ( cute_config_us[ si ][ config ] < cute_config_us[ si ][ best_cute ] ) best_cute = config;
        }

        int best_split = 0;

        for ( int config = 0; kHasCute && config < kCuteSplitConfigCount; ++config )
        {
            split_config_us[ si ][ config ] = time_arm( Arm::CuteSplit, config, kMaxRows, s, b, stream );

            if ( split_config_us[ si ][ config ] < split_config_us[ si ][ best_split ] ) best_split = config;
        }

        for ( int R = 1; R <= kMaxRows; ++R )
        {
            matvec_us[ si ][ R ] = time_arm( Arm::RowMatvec, 0, R, s, b, stream );
            mma_us[ si ][ R ] = time_arm( Arm::RowMma, best, R, s, b, stream );
            cute_us[ si ][ R ] = kHasCute ? time_arm( Arm::Cute, best_cute, R, s, b, stream ) : 0.0;
            split_us[ si ][ R ] = kHasCute ? time_arm( Arm::CuteSplit, best_split, R, s, b, stream ) : 0.0;
        }

        printf( "  decode %.1f us = %.0f GB/s\n", decode_us[ si ], bytes / ( decode_us[ si ] * 1e3 ) );
        printf( "  row-mma configurations at R = 8 (tiles x warps):" );

        for ( int config = 0; config < kMmaConfigCount; ++config )
            printf( "  %dx%d %.1f", kMmaConfigs[ config ].tiles, kMmaConfigs[ config ].warps, mma_config_us[ si ][ config ] );

        printf( "  -> best %dx%d\n", kMmaConfigs[ best ].tiles, kMmaConfigs[ best ].warps );
        printf( "  R           " );

        for ( int R = 1; R <= kMaxRows; ++R ) printf( "%8d", R );

        printf( "\n  row-matvec  " );

        for ( int R = 1; R <= kMaxRows; ++R ) printf( "%8.2f", matvec_us[ si ][ R ] / decode_us[ si ] );

        printf( "   (x decode)\n  row-mma     " );

        for ( int R = 1; R <= kMaxRows; ++R ) printf( "%8.2f", mma_us[ si ][ R ] / decode_us[ si ] );

        printf( "   (x decode; %.0f GB/s at R = 1)\n", bytes / ( mma_us[ si ][ 1 ] * 1e3 ) );

        if ( kHasCute )
        {
            printf( "  cute        " );

            for ( int R = 1; R <= kMaxRows; ++R ) printf( "%8.2f", cute_us[ si ][ R ] / decode_us[ si ] );

            printf( "   (x decode; %.0f GB/s at R = 1)\n  cute configurations at R = 8 (warps x stages):",
                bytes / ( cute_us[ si ][ 1 ] * 1e3 ) );

            for ( int config = 0; config < kCuteConfigCount; ++config )
                printf( "  %dx%d %.1f", kCuteConfigs[ config ].warps, kCuteConfigs[ config ].stages, cute_config_us[ si ][ config ] );

            printf( "  -> best %dx%d\n", kCuteConfigs[ best_cute ].warps, kCuteConfigs[ best_cute ].stages );
            printf( "  cute-split  " );

            for ( int R = 1; R <= kMaxRows; ++R ) printf( "%8.2f", split_us[ si ][ R ] / decode_us[ si ] );

            printf( "   (x decode; %.0f GB/s at R = 1)\n  cute-split warps at R = 8:", bytes / ( split_us[ si ][ 1 ] * 1e3 ) );

            for ( int config = 0; config < kCuteSplitConfigCount; ++config )
                printf( "  %d %.1f", kCuteSplitWarps[ config ], split_config_us[ si ][ config ] );

            printf( "  -> best %d\n", kCuteSplitWarps[ best_split ] );
        }

        printf( "\n" );

        for ( int k = 0; k < copies; ++k )
        {
            CHECK( cudaFree( b.weights[ k ] ) );
            CHECK( cudaFree( b.scales[ k ] ) );
        }

        CHECK( cudaFree( b.x ) );
        CHECK( cudaFree( b.y ) );
    }

    // The 12B's Linear layers per token: 40 sliding, 8 global, 48 FFN.
    double decode_total = 0.0;

    for ( int si = 0; si < kShapeCount; ++si )
        decode_total += kShapes[ si ].per_token * decode_us[ si ];

    printf( "Gemma 4 12B, every Q4_0 Linear of one token (the head is not Q4_0 and is not here)\n" );
    printf( "  decode, one row: %.2f ms\n", decode_total / 1000.0 );
    printf( "  R                 " );

    for ( int R = 1; R <= kMaxRows; ++R ) printf( "%8d", R );

    const auto row = [&]( const char* name, double ( *table )[ kMaxRows + 1 ] )
    {
        printf( "\n  %-16s  ", name );

        for ( int R = 1; R <= kMaxRows; ++R )
        {
            double total = 0.0;

            for ( int si = 0; si < kShapeCount; ++si )
                total += kShapes[ si ].per_token * table[ si ][ R ];

            printf( "%8.2f", total / decode_total );
        }
    };

    row( "row-matvec", matvec_us );
    row( "row-mma", mma_us );

    if ( kHasCute )
    {
        row( "cute", cute_us );
        row( "cute-split", split_us );
    }
    printf( "\n  R decodes         " );

    for ( int R = 1; R <= kMaxRows; ++R ) printf( "%8.2f", double( R ) );

    printf( "\n  (each as a multiple of one decode's Linear time)\n" );

    CHECK( cudaStreamDestroy( stream ) );

    return 0;
}
