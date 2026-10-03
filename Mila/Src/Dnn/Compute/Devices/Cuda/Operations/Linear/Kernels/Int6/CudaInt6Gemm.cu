/**
 * @file CudaInt6Gemm.cu
 * @brief Batched INT6 forward on the BF16 tensor cores: a 128 x 128 tiled GEMM whose codes widen to BF16 exactly
 * in registers, two k16 MMAs per group, and each group's FP32 partial scaled once.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <format>
#include <stdexcept>
#include "CudaInt6Gemm.cuh"
#include "../Int4/Int4Int8Mma.cuh"

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace
    {
        constexpr int kTileM = 128;
        constexpr int kTileN = 128;
        constexpr int kTileK = kInt6GemmInFeaturesMultiple;
        constexpr int kGroupsPerTileK = kTileK / kInt6GemmGroupSize;
        constexpr int kStages = 2;
        constexpr int kThreads = 256;

        // Row strides padded so the fragment loads below are free of bank conflicts: a 128-bit activation load at
        // 16 * t in rows g and g + 1 lands in opposite halves of the 128-byte bank line; a 32-bit low-nibble load at
        // 4 * t and a 16-bit high-bit load at 2 * t in rows g = 0..7 land in distinct banks.
        constexpr int kActivationRowBytes = kTileK * 2 + 64;
        constexpr int kLowRowBytes = kTileK / 2 + 16;
        constexpr int kHighRowBytes = kTileK / 4 + 8;

        constexpr int kActivationTileBytes = kTileM * kActivationRowBytes;
        constexpr int kLowTileBytes = kTileN * kLowRowBytes;
        constexpr int kHighTileBytes = kTileN * kHighRowBytes;
        constexpr int kScaleTileBytes = kTileN * kGroupsPerTileK * static_cast<int>( sizeof( __half ) );
        constexpr int kStageBytes = kActivationTileBytes + kLowTileBytes + kHighTileBytes + kScaleTileBytes;
        constexpr int kSharedBytes = kStages * kStageBytes;

        static_assert( kStageBytes % 16 == 0 );
        static_assert( kTileM * ( kTileK * 2 / 16 ) % kThreads == 0, "every thread copies the same number of activation chunks" );
        static_assert( kTileN * ( kTileK / 2 / 16 ) == kThreads, "one low-nibble chunk a thread" );
        static_assert( kTileN * ( kTileK / 4 / 8 ) == kThreads, "one high-bit chunk a thread" );

        // Warps tile the block 2 (rows) x 4 (columns); each warp owns 64 x 32 outputs as 4 x 4 m16n8 tiles.
        constexpr int kWarpTilesM = 4;
        constexpr int kWarpTilesN = 4;

        using Int4Mma::copyAsync;
        using Int4Mma::commitAsync;
        using Int4Mma::waitAsync;

        __device__ __forceinline__ void mmaBf16( float ( &accumulator )[ 4 ],
            uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1 )
        {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"( accumulator[ 0 ] ), "+f"( accumulator[ 1 ] ), "+f"( accumulator[ 2 ] ), "+f"( accumulator[ 3 ] )
                : "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 ), "r"( b0 ), "r"( b1 ) );
        }

        // Codes 2p and 2p + 1 of a thread's eight as a BF16 pair of their signed steps, exactly: 128 + code is a
        // BF16 bit pattern 0x4300 | code, and subtracting 160 leaves code - 32.
        __device__ __forceinline__ uint32_t codePair( uint32_t low, uint32_t high, int pair )
        {
            const uint32_t even = ( ( low >> ( 8 * pair ) ) & 0xFu ) | ( ( ( high >> ( 4 * pair ) ) & 0x3u ) << 4 );
            const uint32_t odd = ( ( low >> ( 8 * pair + 4 ) ) & 0xFu ) | ( ( ( high >> ( 4 * pair + 2 ) ) & 0x3u ) << 4 );
            const uint32_t biased = 0x43004300u | even | ( odd << 16 );

            const __nv_bfloat162 steps = __hsub2(
                *reinterpret_cast<const __nv_bfloat162*>( &biased ), __floats2bfloat162_rn( 160.0f, 160.0f ) );

            return *reinterpret_cast<const uint32_t*>( &steps );
        }

        // Thread t of an MMA quad covers elements 8t .. 8t+7 of a group, read as one 16-byte activation load per
        // row and one byte-run of codes. The MMA's k positions 2t, 2t+1 and 2t+8, 2t+9 are mapped to elements
        // 8t .. 8t+3 for the first k16 MMA and 8t+4 .. 8t+7 for the second; A and B use the same mapping, and a dot
        // product does not depend on the order of its terms.
        __global__ void __launch_bounds__( kThreads )
        int6_bf16_gemm_kernel(
            __nv_bfloat16* __restrict__        output,
            const __nv_bfloat16* __restrict__  input,
            const uint8_t* __restrict__        weight_codes,
            const __half* __restrict__         weight_scales,
            const __nv_bfloat16* __restrict__  bias,
            int                                rows,
            int                                in_features,
            int                                out_features )
        {
            extern __shared__ __align__( 16 ) unsigned char shared[];

            const int thread = static_cast<int>( threadIdx.x );
            const int warp = thread / 32;
            const int lane = thread % 32;
            const int group = lane / 4;
            const int thread_in_group = lane % 4;
            const int warp_row = ( warp / 4 ) * ( kWarpTilesM * 16 );
            const int warp_column = ( warp % 4 ) * ( kWarpTilesN * 8 );

            const int tile_row = static_cast<int>( blockIdx.x ) * kTileM;
            const int tile_column = static_cast<int>( blockIdx.y ) * kTileN;

            const int64_t code_row_bytes = in_features / 2 + in_features / 4;
            const int groups_per_row = in_features / kInt6GemmGroupSize;
            const int k_tiles = in_features / kTileK;

            const auto loadStage = [&]( int stage, int k_tile )
            {
                unsigned char* base = shared + stage * kStageBytes;
                const int k = k_tile * kTileK;

                constexpr int kActivationChunksPerRow = kTileK * 2 / 16;

#pragma unroll
                for ( int part = 0; part < kTileM * kActivationChunksPerRow / kThreads; ++part )
                {
                    const int chunk = thread + part * kThreads;
                    const int row = chunk / kActivationChunksPerRow;
                    const int column = ( chunk % kActivationChunksPerRow ) * 8;
                    const int source_row = tile_row + row;
                    const bool valid = source_row < rows;

                    copyAsync<16>( base + row * kActivationRowBytes + column * 2,
                        input + static_cast<int64_t>( valid ? source_row : 0 ) * in_features + k + column,
                        valid );
                }

                const int code_row = thread / 2;
                const int source_code_row = tile_column + code_row;
                const bool code_valid = source_code_row < out_features;
                const uint8_t* source_codes = weight_codes + static_cast<int64_t>( code_valid ? source_code_row : 0 ) * code_row_bytes;

                // Two 16-byte low-nibble chunks and two 8-byte high-bit chunks per code row.
                copyAsync<16>( base + kActivationTileBytes + code_row * kLowRowBytes + ( thread % 2 ) * 16,
                    source_codes + k / 2 + ( thread % 2 ) * 16, code_valid );

                copyAsync<8>( base + kActivationTileBytes + kLowTileBytes + code_row * kHighRowBytes + ( thread % 2 ) * 8,
                    source_codes + in_features / 2 + k / 4 + ( thread % 2 ) * 8, code_valid );

                // A tile row's scales are contiguous, and aligned to their own size because in_features is a
                // multiple of kTileK.
                if ( thread < kTileN )
                {
                    const int source_row = tile_column + thread;
                    const bool valid = source_row < out_features;

                    copyAsync<kGroupsPerTileK * sizeof( __half )>(
                        base + kActivationTileBytes + kLowTileBytes + kHighTileBytes + thread * kGroupsPerTileK * sizeof( __half ),
                        weight_scales + static_cast<int64_t>( valid ? source_row : 0 ) * groups_per_row + k / kInt6GemmGroupSize,
                        valid );
                }
            };

            float accumulators[ kWarpTilesM ][ kWarpTilesN ][ 4 ];

#pragma unroll
            for ( int m = 0; m < kWarpTilesM; ++m )
#pragma unroll
                for ( int n = 0; n < kWarpTilesN; ++n )
#pragma unroll
                    for ( int i = 0; i < 4; ++i )
                        accumulators[ m ][ n ][ i ] = 0.0f;

#pragma unroll
            for ( int stage = 0; stage < kStages - 1; ++stage )
            {
                if ( stage < k_tiles )
                    loadStage( stage, stage );

                commitAsync();
            }

            for ( int k_tile = 0; k_tile < k_tiles; ++k_tile )
            {
                waitAsync<kStages - 2>();
                __syncthreads();

                const int next = k_tile + kStages - 1;

                if ( next < k_tiles )
                    loadStage( next % kStages, next );

                commitAsync();

                const unsigned char* base = shared + ( k_tile % kStages ) * kStageBytes;
                const unsigned char* activation_tile = base;
                const unsigned char* low_tile = base + kActivationTileBytes;
                const unsigned char* high_tile = low_tile + kLowTileBytes;
                const __half* scale_tile = reinterpret_cast<const __half*>( high_tile + kHighTileBytes );

#pragma unroll
                for ( int weight_group = 0; weight_group < kGroupsPerTileK; ++weight_group )
                {
                    uint4 upper[ kWarpTilesM ];
                    uint4 lower[ kWarpTilesM ];

#pragma unroll
                    for ( int m = 0; m < kWarpTilesM; ++m )
                    {
                        const int row = warp_row + m * 16 + group;
                        const int offset = weight_group * kInt6GemmGroupSize * 2 + 16 * thread_in_group;

                        upper[ m ] = *reinterpret_cast<const uint4*>( activation_tile + row * kActivationRowBytes + offset );
                        lower[ m ] = *reinterpret_cast<const uint4*>( activation_tile + ( row + 8 ) * kActivationRowBytes + offset );
                    }

#pragma unroll
                    for ( int n = 0; n < kWarpTilesN; ++n )
                    {
                        const int column = warp_column + n * 8;
                        const uint32_t low = *reinterpret_cast<const uint32_t*>(
                            low_tile + ( column + group ) * kLowRowBytes + weight_group * 16 + 4 * thread_in_group );
                        const uint32_t high = *reinterpret_cast<const uint16_t*>(
                            high_tile + ( column + group ) * kHighRowBytes + weight_group * 8 + 2 * thread_in_group );

                        const uint32_t b0 = codePair( low, high, 0 );
                        const uint32_t b1 = codePair( low, high, 1 );
                        const uint32_t b2 = codePair( low, high, 2 );
                        const uint32_t b3 = codePair( low, high, 3 );

                        const int output_column = column + 2 * thread_in_group;
                        const float scale_low = __half2float( scale_tile[ output_column * kGroupsPerTileK + weight_group ] );
                        const float scale_high = __half2float( scale_tile[ ( output_column + 1 ) * kGroupsPerTileK + weight_group ] );

#pragma unroll
                        for ( int m = 0; m < kWarpTilesM; ++m )
                        {
                            float partial[ 4 ] = { 0.0f, 0.0f, 0.0f, 0.0f };

                            mmaBf16( partial, upper[ m ].x, lower[ m ].x, upper[ m ].y, lower[ m ].y, b0, b1 );
                            mmaBf16( partial, upper[ m ].z, lower[ m ].z, upper[ m ].w, lower[ m ].w, b2, b3 );

                            float* accumulator = accumulators[ m ][ n ];
                            accumulator[ 0 ] = fmaf( partial[ 0 ], scale_low, accumulator[ 0 ] );
                            accumulator[ 1 ] = fmaf( partial[ 1 ], scale_high, accumulator[ 1 ] );
                            accumulator[ 2 ] = fmaf( partial[ 2 ], scale_low, accumulator[ 2 ] );
                            accumulator[ 3 ] = fmaf( partial[ 3 ], scale_high, accumulator[ 3 ] );
                        }
                    }
                }
            }

            waitAsync<0>();

#pragma unroll
            for ( int m = 0; m < kWarpTilesM; ++m )
            {
#pragma unroll
                for ( int n = 0; n < kWarpTilesN; ++n )
                {
                    const int column = tile_column + warp_column + n * 8 + 2 * thread_in_group;

                    if ( column >= out_features )
                        continue;

                    float bias_low = 0.0f;
                    float bias_high = 0.0f;

                    if ( bias != nullptr )
                    {
                        bias_low = __bfloat162float( bias[ column ] );
                        bias_high = __bfloat162float( bias[ column + 1 ] );
                    }

#pragma unroll
                    for ( int half_tile = 0; half_tile < 2; ++half_tile )
                    {
                        const int row = tile_row + warp_row + m * 16 + group + half_tile * 8;

                        if ( row < rows )
                        {
                            *reinterpret_cast<__nv_bfloat162*>( output + static_cast<int64_t>( row ) * out_features + column ) =
                                __floats2bfloat162_rn(
                                    accumulators[ m ][ n ][ 2 * half_tile ] + bias_low,
                                    accumulators[ m ][ n ][ 2 * half_tile + 1 ] + bias_high );
                        }
                    }
                }
            }
        }
    }

    void cuda_int6_bf16_gemm(
        __nv_bfloat16*       output,
        const __nv_bfloat16* input,
        const uint8_t*       weight_codes,
        const __half*        weight_scales,
        const __nv_bfloat16* bias,
        int                  rows,
        int                  in_features,
        int                  out_features,
        cudaStream_t         stream )
    {
        if ( in_features % kInt6GemmInFeaturesMultiple != 0 || out_features % 2 != 0 )
        {
            throw std::invalid_argument( std::format(
                "cuda_int6_bf16_gemm: in_features {} must be a multiple of {} and out_features {} even",
                in_features, kInt6GemmInFeaturesMultiple, out_features ) );
        }

        if ( rows == 0 || out_features == 0 )
            return;

        // The opt-in above 48 KiB of dynamic shared memory is per device, and a process may drive several.
        const cudaError_t attribute_status = cudaFuncSetAttribute(
            int6_bf16_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes );

        if ( attribute_status != cudaSuccess )
        {
            throw std::runtime_error( std::format(
                "cuda_int6_bf16_gemm: cannot reserve {} bytes of shared memory: {}",
                kSharedBytes, cudaGetErrorString( attribute_status ) ) );
        }

        const dim3 grid(
            static_cast<unsigned>( ( rows + kTileM - 1 ) / kTileM ),
            static_cast<unsigned>( ( out_features + kTileN - 1 ) / kTileN ) );

        int6_bf16_gemm_kernel<<<grid, kThreads, kSharedBytes, stream>>>(
            output, input, weight_codes, weight_scales, bias, rows, in_features, out_features );
    }

} // namespace Mila::Dnn::Compute::Cuda::Linear
