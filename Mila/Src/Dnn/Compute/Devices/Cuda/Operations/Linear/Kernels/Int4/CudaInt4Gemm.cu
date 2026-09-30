/**
 * @file CudaInt4Gemm.cu
 * @brief Q4_0 prefill on the INT8 tensor cores: a per-block activation quantizer and a 128 x 128 tiled GEMM whose
 * every k32 MMA is exactly one Q4_0 block.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <format>
#include <stdexcept>
#include "CudaInt4Gemm.cuh"

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace
    {
        // ---- activation quantizer ------------------------------------------------------------------------

        constexpr int kElementsPerThread = 8;
        constexpr int kThreadsPerBlock32 = kInt4GemmBlockSize / kElementsPerThread;

        __global__ void quantize_bf16_to_int8_per_block_kernel(
            int8_t* __restrict__               codes,
            float* __restrict__                scales,
            const __nv_bfloat16* __restrict__  input,
            int64_t                            thread_count )
        {
            const int64_t index = static_cast<int64_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
            const bool valid = index < thread_count;

            float values[ kElementsPerThread ];
            float largest = 0.0f;

            if ( valid )
            {
                const uint4 packed = reinterpret_cast<const uint4*>( input )[ index ];
                const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>( &packed );

#pragma unroll
                for ( int pair = 0; pair < kElementsPerThread / 2; ++pair )
                {
                    const float2 unpacked = __bfloat1622float2( pairs[ pair ] );
                    values[ 2 * pair ] = unpacked.x;
                    values[ 2 * pair + 1 ] = unpacked.y;
                }

#pragma unroll
                for ( int element = 0; element < kElementsPerThread; ++element )
                    largest = fmaxf( largest, fabsf( values[ element ] ) );
            }

            // The four threads of a block are adjacent lanes, and a block never straddles a warp because
            // in_features is a multiple of 32; every lane joins the shuffle.
#pragma unroll
            for ( int offset = 1; offset < kThreadsPerBlock32; offset <<= 1 )
                largest = fmaxf( largest, __shfl_xor_sync( 0xFFFFFFFFu, largest, offset ) );

            if ( !valid )
                return;

            const float inverse = largest > 0.0f ? 127.0f / largest : 0.0f;
            uint32_t words[ 2 ] = { 0u, 0u };

#pragma unroll
            for ( int element = 0; element < kElementsPerThread; ++element )
            {
                const int code = __float2int_rn( values[ element ] * inverse );
                words[ element / 4 ] |= ( static_cast<uint32_t>( code ) & 0xFFu ) << ( 8 * ( element % 4 ) );
            }

            reinterpret_cast<uint2*>( codes )[ index ] = make_uint2( words[ 0 ], words[ 1 ] );

            if ( ( threadIdx.x % kThreadsPerBlock32 ) == 0 )
                scales[ index / kThreadsPerBlock32 ] = largest / 127.0f;
        }

        // ---- GEMM ----------------------------------------------------------------------------------------

        constexpr int kTileM = 128;
        constexpr int kTileN = 128;
        constexpr int kStages = 2;
        constexpr int kThreads = 256;

        // Measured 2026-09-28 at Gemma 4 12B and Llama 3.1 8B shapes, 1024 rows: a 64-deep tile with 3 stages ran
        // 67-86 TFLOPS on the RTX 4070, 128 deep 91-108; a third stage at 128 changed nothing on either card. So
        // 128 wherever it divides the width, and 64 for a width it does not (the Gemma 4 26B-A4B's dense
        // feed-forward, 2112).
        constexpr int kDeepTileK = 128;
        constexpr int kShallowTileK = kInt4GemmInFeaturesMultiple;

        template <int kTileK>
        struct TileGeometry
        {
            static constexpr int kBlocksPerTileK = kTileK / kInt4GemmBlockSize;

            // Row strides padded so that, at the 128-deep tile, the fragment loads below are free of bank
            // conflicts: a 64-bit load at 8 * t in rows g = 0..3 and a 32-bit load at 4 * t in rows g = 0..7.
            static constexpr int kActivationRowBytes = kTileK + 32;
            static constexpr int kWeightRowBytes = kTileK / 2 + 16;

            static constexpr int kActivationTileBytes = kTileM * kActivationRowBytes;
            static constexpr int kWeightTileBytes = kTileN * kWeightRowBytes;
            static constexpr int kActivationScaleBytes = kTileM * kBlocksPerTileK * static_cast<int>( sizeof( float ) );
            static constexpr int kWeightScaleBytes = kTileN * kBlocksPerTileK * static_cast<int>( sizeof( __half ) );
            static constexpr int kStageBytes = kActivationTileBytes + kWeightTileBytes + kActivationScaleBytes + kWeightScaleBytes;
            static constexpr int kSharedBytes = kStages * kStageBytes;

            static_assert( kStageBytes % 16 == 0 );
            static_assert( kTileM * ( kTileK / 16 ) % kThreads == 0 && kTileN * ( kTileK / 32 ) % kThreads == 0,
                "every thread copies the same number of 16-byte chunks" );
        };

        // Warps tile the block 2 (rows) x 4 (columns); each warp owns 64 x 32 outputs as 4 x 4 m16n8 tiles.
        constexpr int kWarpTilesM = 4;
        constexpr int kWarpTilesN = 4;

        // An invalid source copies nothing and zero-fills the destination.
        template <int kBytes>
        __device__ __forceinline__ void copyAsync( void* shared, const void* global, bool valid )
        {
            const unsigned address = static_cast<unsigned>( __cvta_generic_to_shared( shared ) );

            if constexpr ( kBytes == 16 )
            {
                asm volatile( "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                    :: "r"( address ), "l"( global ), "r"( valid ? 16 : 0 ) );
            }
            else
            {
                asm volatile( "cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                    :: "r"( address ), "l"( global ), "n"( kBytes ), "r"( valid ? kBytes : 0 ) );
            }
        }

        __device__ __forceinline__ void commitAsync()
        {
            asm volatile( "cp.async.commit_group;\n" ::: "memory" );
        }

        template <int kPending>
        __device__ __forceinline__ void waitAsync()
        {
            asm volatile( "cp.async.wait_group %0;\n" :: "n"( kPending ) : "memory" );
        }

        // Eight packed codes, k = 8t .. 8t+7 of one block, become the two B registers of thread t. The MMA's
        // k positions 4t..4t+3 and 16+4t..16+4t+3 are mapped to the block's 8t..8t+7; the A fragment is loaded
        // under the same mapping, and a dot product does not depend on the order of its terms.
        // Per byte, (v + 0x78) ^ 0x80 is v - 8 in two's complement for v in [0, 15], with no carry between bytes.
        __device__ __forceinline__ void unpackWeightFragment( uint32_t word, uint32_t& low_half, uint32_t& high_half )
        {
            const uint32_t low_nibbles = ( ( word & 0x0F0F0F0Fu ) + 0x78787878u ) ^ 0x80808080u;
            const uint32_t high_nibbles = ( ( ( word >> 4 ) & 0x0F0F0F0Fu ) + 0x78787878u ) ^ 0x80808080u;

            low_half = __byte_perm( low_nibbles, high_nibbles, 0x5140 );
            high_half = __byte_perm( low_nibbles, high_nibbles, 0x7362 );
        }

        // The MMA starts each block's dot product at the bit pattern of 1.5 * 2^23, so its INT32 result read as
        // FP32 is exactly 1.5 * 2^23 + dot for |dot| < 2^22 -- a block's dot product is at most 32 * 127 * 8 in
        // magnitude. The conversion to float then costs nothing, and the offset cancels inside one FMA.
        constexpr int kFloatBias = 0x4B400000;
        constexpr float kFloatBiasValue = 12582912.0f;

        __device__ __forceinline__ void mmaInt8Biased( float ( &result )[ 4 ],
            uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1 )
        {
            int raw[ 4 ];
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                : "=r"( raw[ 0 ] ), "=r"( raw[ 1 ] ), "=r"( raw[ 2 ] ), "=r"( raw[ 3 ] )
                : "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 ), "r"( b0 ), "r"( b1 ), "r"( kFloatBias ) );

#pragma unroll
            for ( int i = 0; i < 4; ++i )
                result[ i ] = __int_as_float( raw[ i ] );
        }

        template <int kTileK>
        __global__ void __launch_bounds__( kThreads )
        int4_int8_gemm_kernel(
            __nv_bfloat16* __restrict__        output,
            const int8_t* __restrict__         activation_codes,
            const float* __restrict__          activation_scales,
            const uint8_t* __restrict__        weight_codes,
            const __half* __restrict__         weight_scales,
            const __nv_bfloat16* __restrict__  bias,
            int                                rows,
            int                                in_features,
            int                                out_features )
        {
            using Geometry = TileGeometry<kTileK>;

            constexpr int kBlocksPerTileK = Geometry::kBlocksPerTileK;
            constexpr int kActivationRowBytes = Geometry::kActivationRowBytes;
            constexpr int kWeightRowBytes = Geometry::kWeightRowBytes;
            constexpr int kActivationTileBytes = Geometry::kActivationTileBytes;
            constexpr int kWeightTileBytes = Geometry::kWeightTileBytes;
            constexpr int kActivationScaleBytes = Geometry::kActivationScaleBytes;
            constexpr int kStageBytes = Geometry::kStageBytes;

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

            const int blocks_per_row = in_features / kInt4GemmBlockSize;
            const int k_tiles = in_features / kTileK;

            const auto loadStage = [&]( int stage, int k_tile )
            {
                unsigned char* base = shared + stage * kStageBytes;
                const int k = k_tile * kTileK;

                constexpr int kActivationChunksPerRow = kTileK / 16;
                constexpr int kWeightChunksPerRow = kTileK / 32;

#pragma unroll
                for ( int part = 0; part < kTileM * kActivationChunksPerRow / kThreads; ++part )
                {
                    const int chunk = thread + part * kThreads;
                    const int row = chunk / kActivationChunksPerRow;
                    const int column = ( chunk % kActivationChunksPerRow ) * 16;
                    const int source_row = tile_row + row;
                    const bool valid = source_row < rows;

                    copyAsync<16>( base + row * kActivationRowBytes + column,
                        activation_codes + static_cast<int64_t>( valid ? source_row : 0 ) * in_features + k + column,
                        valid );
                }

#pragma unroll
                for ( int part = 0; part < kTileN * kWeightChunksPerRow / kThreads; ++part )
                {
                    const int chunk = thread + part * kThreads;
                    const int row = chunk / kWeightChunksPerRow;
                    const int column = ( chunk % kWeightChunksPerRow ) * 16;
                    const int source_row = tile_column + row;
                    const bool valid = source_row < out_features;

                    copyAsync<16>( base + kActivationTileBytes + row * kWeightRowBytes + column,
                        weight_codes + static_cast<int64_t>( valid ? source_row : 0 ) * ( in_features / 2 ) + k / 2 + column,
                        valid );
                }

                // A tile row's scales are contiguous, and aligned to their own size because in_features is a
                // multiple of kTileK.
                if ( thread < kTileM )
                {
                    const int source_row = tile_row + thread;
                    const bool valid = source_row < rows;

                    copyAsync<kBlocksPerTileK * sizeof( float )>(
                        base + kActivationTileBytes + kWeightTileBytes + thread * kBlocksPerTileK * sizeof( float ),
                        activation_scales + static_cast<int64_t>( valid ? source_row : 0 ) * blocks_per_row + k / kInt4GemmBlockSize,
                        valid );
                }
                else
                {
                    const int row = thread - kTileM;
                    const int source_row = tile_column + row;
                    const bool valid = source_row < out_features;

                    copyAsync<kBlocksPerTileK * sizeof( __half )>(
                        base + kActivationTileBytes + kWeightTileBytes + kActivationScaleBytes + row * kBlocksPerTileK * sizeof( __half ),
                        weight_scales + static_cast<int64_t>( valid ? source_row : 0 ) * blocks_per_row + k / kInt4GemmBlockSize,
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
                const unsigned char* weight_tile = base + kActivationTileBytes;
                const float* activation_scale_tile = reinterpret_cast<const float*>( base + kActivationTileBytes + kWeightTileBytes );
                const __half* weight_scale_tile = reinterpret_cast<const __half*>(
                    base + kActivationTileBytes + kWeightTileBytes + kActivationScaleBytes );

#pragma unroll
                for ( int block = 0; block < kBlocksPerTileK; ++block )
                {
                    uint32_t b[ kWarpTilesN ][ 2 ];
                    float weight_scale[ kWarpTilesN ][ 2 ];
                    float bias_times_scale[ kWarpTilesN ][ 2 ];

#pragma unroll
                    for ( int n = 0; n < kWarpTilesN; ++n )
                    {
                        const int column = warp_column + n * 8;
                        const uint32_t word = *reinterpret_cast<const uint32_t*>(
                            weight_tile + ( column + group ) * kWeightRowBytes + block * 16 + 4 * thread_in_group );
                        unpackWeightFragment( word, b[ n ][ 0 ], b[ n ][ 1 ] );

                        const int output_column = column + 2 * thread_in_group;
                        weight_scale[ n ][ 0 ] = __half2float( weight_scale_tile[ output_column * kBlocksPerTileK + block ] );
                        weight_scale[ n ][ 1 ] = __half2float( weight_scale_tile[ ( output_column + 1 ) * kBlocksPerTileK + block ] );
                        bias_times_scale[ n ][ 0 ] = -kFloatBiasValue * weight_scale[ n ][ 0 ];
                        bias_times_scale[ n ][ 1 ] = -kFloatBiasValue * weight_scale[ n ][ 1 ];
                    }

#pragma unroll
                    for ( int m = 0; m < kWarpTilesM; ++m )
                    {
                        const int row = warp_row + m * 16 + group;
                        const uint2 upper = *reinterpret_cast<const uint2*>(
                            activation_tile + row * kActivationRowBytes + block * 32 + 8 * thread_in_group );
                        const uint2 lower = *reinterpret_cast<const uint2*>(
                            activation_tile + ( row + 8 ) * kActivationRowBytes + block * 32 + 8 * thread_in_group );

                        const float upper_scale = activation_scale_tile[ row * kBlocksPerTileK + block ];
                        const float lower_scale = activation_scale_tile[ ( row + 8 ) * kBlocksPerTileK + block ];

#pragma unroll
                        for ( int n = 0; n < kWarpTilesN; ++n )
                        {
                            float biased[ 4 ];
                            mmaInt8Biased( biased, upper.x, lower.x, upper.y, lower.y, b[ n ][ 0 ], b[ n ][ 1 ] );

                            // fmaf( biased, w, -bias * w ) is exactly dot * w rounded once: the product is exact
                            // inside the FMA and the offset cancels before the rounding.
                            float* accumulator = accumulators[ m ][ n ];
                            accumulator[ 0 ] = fmaf( fmaf( biased[ 0 ], weight_scale[ n ][ 0 ], bias_times_scale[ n ][ 0 ] ), upper_scale, accumulator[ 0 ] );
                            accumulator[ 1 ] = fmaf( fmaf( biased[ 1 ], weight_scale[ n ][ 1 ], bias_times_scale[ n ][ 1 ] ), upper_scale, accumulator[ 1 ] );
                            accumulator[ 2 ] = fmaf( fmaf( biased[ 2 ], weight_scale[ n ][ 0 ], bias_times_scale[ n ][ 0 ] ), lower_scale, accumulator[ 2 ] );
                            accumulator[ 3 ] = fmaf( fmaf( biased[ 3 ], weight_scale[ n ][ 1 ], bias_times_scale[ n ][ 1 ] ), lower_scale, accumulator[ 3 ] );
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

    void cuda_quantize_bf16_to_int8_per_block(
        int8_t*              codes,
        float*               scales,
        const __nv_bfloat16* input,
        int                  rows,
        int                  in_features,
        cudaStream_t         stream )
    {
        if ( in_features % kInt4GemmBlockSize != 0 )
        {
            throw std::invalid_argument( std::format(
                "cuda_quantize_bf16_to_int8_per_block: in_features {} is not a multiple of {}",
                in_features, kInt4GemmBlockSize ) );
        }

        const int64_t thread_count = static_cast<int64_t>( rows ) * in_features / kElementsPerThread;

        if ( thread_count == 0 )
            return;

        constexpr int kBlockThreads = 256;
        const auto grid = static_cast<unsigned>( ( thread_count + kBlockThreads - 1 ) / kBlockThreads );

        quantize_bf16_to_int8_per_block_kernel<<<grid, kBlockThreads, 0, stream>>>( codes, scales, input, thread_count );
    }

    namespace
    {
        template <int kTileK>
        void launchInt4Int8Gemm(
            __nv_bfloat16* output, const int8_t* activation_codes, const float* activation_scales,
            const uint8_t* weight_codes, const __half* weight_scales, const __nv_bfloat16* bias,
            int rows, int in_features, int out_features, cudaStream_t stream )
        {
            constexpr int kSharedBytes = TileGeometry<kTileK>::kSharedBytes;

            // The opt-in above 48 KiB of dynamic shared memory is per device, and a process may drive several.
            const cudaError_t attribute_status = cudaFuncSetAttribute(
                int4_int8_gemm_kernel<kTileK>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes );

            if ( attribute_status != cudaSuccess )
            {
                throw std::runtime_error( std::format(
                    "cuda_int4_int8_gemm: cannot reserve {} bytes of shared memory: {}",
                    kSharedBytes, cudaGetErrorString( attribute_status ) ) );
            }

            const dim3 grid(
                static_cast<unsigned>( ( rows + kTileM - 1 ) / kTileM ),
                static_cast<unsigned>( ( out_features + kTileN - 1 ) / kTileN ) );

            int4_int8_gemm_kernel<kTileK><<<grid, kThreads, kSharedBytes, stream>>>(
                output, activation_codes, activation_scales, weight_codes, weight_scales, bias,
                rows, in_features, out_features );
        }
    }

    void cuda_int4_int8_gemm(
        __nv_bfloat16*       output,
        const int8_t*        activation_codes,
        const float*         activation_scales,
        const uint8_t*       weight_codes,
        const __half*        weight_scales,
        const __nv_bfloat16* bias,
        int                  rows,
        int                  in_features,
        int                  out_features,
        cudaStream_t         stream )
    {
        if ( in_features % kShallowTileK != 0 || out_features % 2 != 0 )
        {
            throw std::invalid_argument( std::format(
                "cuda_int4_int8_gemm: in_features {} must be a multiple of {} and out_features {} even",
                in_features, kShallowTileK, out_features ) );
        }

        if ( rows == 0 || out_features == 0 )
            return;

        if ( in_features % kDeepTileK == 0 )
        {
            launchInt4Int8Gemm<kDeepTileK>( output, activation_codes, activation_scales, weight_codes, weight_scales,
                bias, rows, in_features, out_features, stream );
        }
        else
        {
            launchInt4Int8Gemm<kShallowTileK>( output, activation_codes, activation_scales, weight_codes, weight_scales,
                bias, rows, in_features, out_features, stream );
        }
    }

} // namespace Mila::Dnn::Compute::Cuda::Linear
