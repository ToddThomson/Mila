// Q4_0 expert bank decode kernels: the gated and combine passes as gather-matvecs. See Moe.cuh.

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "Moe.cuh"
// Shared activation math source, at the depth-relative path the elementwise kernel uses.
#include "../../../../../../Components/Activations/Activation/Kernels/ElementwiseActivation.h"

namespace Mila::Dnn::Compute::Cuda::Moe
{
    namespace
    {
        // A Q4_0 group: 32 codes, 16 bytes, one lane's unit of work.
        constexpr int kGroup = 32;
        constexpr int kWarpsPerBlock = 8;

        // Outputs per warp. Register-limited to two blocks a multiprocessor, a warp with one row had one load in flight
        // per lane per step; these put two and four there, and the combine's staged values serve four columns.
        constexpr int kUnitsPerWarp = 2;
        constexpr int kColumnsPerWarp = 2;

        // A group's 32 FP32 gated values stride 36 floats in shared memory: the eight lanes of a quarter-warp read
        // float4 k of eight consecutive groups, and without the pad all eight land in one bank.
        constexpr int kPaddedGroup = 36;

        // ( code - 8 ) exactly: the code in the low mantissa bits of 2^23, minus 2^23 + 8.
        __device__ __forceinline__ float centered_code( uint32_t word, int index )
        {
            return __uint_as_float( 0x4B000000u | ( ( word >> ( 4 * index ) ) & 0xFu ) ) - 8388616.0f;
        }

        __device__ __forceinline__ float warp_sum( float value )
        {
#pragma unroll
            for ( int offset = 16; offset > 0; offset >>= 1 )
            {
                value += __shfl_xor_sync( 0xffffffff, value, offset );
            }

            return value;
        }

        // Both rows' group sums of x * ( code - 8 ) over the same 32 activations, low nibble the even column.
        __device__ __forceinline__ void group_dot_pair_bf16(
            const int4 gate_codes, const int4 up_codes, const __nv_bfloat16* x, float& gate_sum, float& up_sum )
        {
            const uint32_t gate_words[ 4 ] = {
                static_cast<uint32_t>( gate_codes.x ), static_cast<uint32_t>( gate_codes.y ),
                static_cast<uint32_t>( gate_codes.z ), static_cast<uint32_t>( gate_codes.w ) };
            const uint32_t up_words[ 4 ] = {
                static_cast<uint32_t>( up_codes.x ), static_cast<uint32_t>( up_codes.y ),
                static_cast<uint32_t>( up_codes.z ), static_cast<uint32_t>( up_codes.w ) };

            gate_sum = 0.0f;
            up_sum = 0.0f;

#pragma unroll
            for ( int word = 0; word < 4; ++word )
            {
                const int4 raw = *reinterpret_cast<const int4*>( x + word * 8 );
                const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>( &raw );

#pragma unroll
                for ( int pair = 0; pair < 4; ++pair )
                {
                    const float2 value = __bfloat1622float2( pairs[ pair ] );

                    gate_sum += value.x * centered_code( gate_words[ word ], 2 * pair )
                              + value.y * centered_code( gate_words[ word ], 2 * pair + 1 );
                    up_sum += value.x * centered_code( up_words[ word ], 2 * pair )
                            + value.y * centered_code( up_words[ word ], 2 * pair + 1 );
                }
            }
        }

        // A group's sum of g * ( code - 8 ), g 16-byte aligned in shared memory.
        __device__ __forceinline__ float group_dot_fp32( const int4 codes, const float* g )
        {
            const uint32_t words[ 4 ] = {
                static_cast<uint32_t>( codes.x ), static_cast<uint32_t>( codes.y ),
                static_cast<uint32_t>( codes.z ), static_cast<uint32_t>( codes.w ) };

            float even = 0.0f;
            float odd = 0.0f;

#pragma unroll
            for ( int word = 0; word < 4; ++word )
            {
                const float4 low = *reinterpret_cast<const float4*>( g + word * 8 );
                const float4 high = *reinterpret_cast<const float4*>( g + word * 8 + 4 );
                float& sum = ( word % 2 == 0 ) ? even : odd;

                sum += low.x * centered_code( words[ word ], 0 ) + low.y * centered_code( words[ word ], 1 )
                     + low.z * centered_code( words[ word ], 2 ) + low.w * centered_code( words[ word ], 3 )
                     + high.x * centered_code( words[ word ], 4 ) + high.y * centered_code( words[ word ], 5 )
                     + high.z * centered_code( words[ word ], 6 ) + high.w * centered_code( words[ word ], 7 );
            }

            return even + odd;
        }

        template<typename TFunctor>
        __global__ void __launch_bounds__( 32 * kWarpsPerBlock, 2 ) moe_gated_gather_int4_kernel(
            const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ gate_up,
            const __half* __restrict__ gate_up_scales, const int32_t* __restrict__ indices, float* __restrict__ gated,
            int hidden, int intermediate, int experts, int top_k, TFunctor functor )
        {
            const int first_unit = ( blockIdx.x * kWarpsPerBlock + threadIdx.y ) * kUnitsPerWarp;
            const int slot = blockIdx.y;
            const int64_t token = blockIdx.z;

            if ( first_unit >= intermediate )
            {
                return;
            }

            const int64_t first_cell = ( token * top_k + slot ) * intermediate + first_unit;
            const int32_t expert = indices[ token * top_k + slot ];

            if ( expert < 0 || expert >= experts )
            {
                if ( threadIdx.x == 0 )
                {
                    for ( int u = 0; u < kUnitsPerWarp && first_unit + u < intermediate; ++u )
                    {
                        gated[ first_cell + u ] = NAN;
                    }
                }

                return;
            }

            const int groups = hidden / kGroup;
            const __nv_bfloat16* x = input + token * hidden;

            // A unit past the end reads the last one's rows and is not written.
            const int4* gate_codes[ kUnitsPerWarp ];
            const int4* up_codes[ kUnitsPerWarp ];
            const __half* gate_scales[ kUnitsPerWarp ];
            const __half* up_scales[ kUnitsPerWarp ];

#pragma unroll
            for ( int u = 0; u < kUnitsPerWarp; ++u )
            {
                const int unit = min( first_unit + u, intermediate - 1 );
                const int64_t gate_row = static_cast<int64_t>( expert ) * 2 * intermediate + unit;
                const int64_t up_row = gate_row + intermediate;

                gate_codes[ u ] = reinterpret_cast<const int4*>( gate_up + gate_row * ( hidden / 2 ) );
                up_codes[ u ] = reinterpret_cast<const int4*>( gate_up + up_row * ( hidden / 2 ) );
                gate_scales[ u ] = gate_up_scales + gate_row * groups;
                up_scales[ u ] = gate_up_scales + up_row * groups;
            }

            float gate[ kUnitsPerWarp ] = {};
            float up[ kUnitsPerWarp ] = {};

            for ( int group = threadIdx.x; group < groups; group += 32 )
            {
                int4 gate_group[ kUnitsPerWarp ];
                int4 up_group[ kUnitsPerWarp ];

#pragma unroll
                for ( int u = 0; u < kUnitsPerWarp; ++u )
                {
                    gate_group[ u ] = gate_codes[ u ][ group ];
                    up_group[ u ] = up_codes[ u ][ group ];
                }

#pragma unroll
                for ( int u = 0; u < kUnitsPerWarp; ++u )
                {
                    float gate_sum;
                    float up_sum;

                    group_dot_pair_bf16( gate_group[ u ], up_group[ u ], x + group * kGroup, gate_sum, up_sum );

                    gate[ u ] = fmaf( __half2float( gate_scales[ u ][ group ] ), gate_sum, gate[ u ] );
                    up[ u ] = fmaf( __half2float( up_scales[ u ][ group ] ), up_sum, up[ u ] );
                }
            }

#pragma unroll
            for ( int u = 0; u < kUnitsPerWarp; ++u )
            {
                const float gate_total = warp_sum( gate[ u ] );
                const float up_total = warp_sum( up[ u ] );

                if ( threadIdx.x == 0 && first_unit + u < intermediate )
                {
                    gated[ first_cell + u ] = functor.fwd( gate_total ) * up_total;
                }
            }
        }

        __global__ void __launch_bounds__( 32 * kWarpsPerBlock, 2 ) moe_combine_gather_int4_kernel(
            const float* __restrict__ gated, const uint8_t* __restrict__ down, const __half* __restrict__ down_scales,
            const __nv_bfloat16* __restrict__ weights, const int32_t* __restrict__ indices,
            __nv_bfloat16* __restrict__ output, int hidden, int intermediate, int experts, int top_k )
        {
            extern __shared__ float4 shared_groups[];
            float* staged = reinterpret_cast<float*>( shared_groups );

            const int64_t token = blockIdx.y;
            const int groups_per_row = intermediate / kGroup;
            const int token_groups = top_k * groups_per_row;

            // The token's gated values, slot-major as the gated pass wrote them, one padded group at a time.
            const float4* source = reinterpret_cast<const float4*>( gated + token * top_k * intermediate );
            const int thread = threadIdx.y * 32 + threadIdx.x;

            for ( int quad = thread; quad < token_groups * ( kGroup / 4 ); quad += 32 * kWarpsPerBlock )
            {
                const int group = quad / ( kGroup / 4 );

                reinterpret_cast<float4*>( staged + group * kPaddedGroup )[ quad % ( kGroup / 4 ) ] = source[ quad ];
            }

            __syncthreads();

            const int first_column = ( blockIdx.x * kWarpsPerBlock + threadIdx.y ) * kColumnsPerWarp;

            if ( first_column >= hidden )
            {
                return;
            }

            float sum[ kColumnsPerWarp ] = {};

            for ( int group = threadIdx.x; group < token_groups; group += 32 )
            {
                const int slot = group / groups_per_row;
                const int within = group - slot * groups_per_row;
                const int32_t expert = indices[ token * top_k + slot ];

                if ( expert < 0 || expert >= experts )
                {
#pragma unroll
                    for ( int c = 0; c < kColumnsPerWarp; ++c )
                    {
                        sum[ c ] += NAN;
                    }

                    continue;
                }

                // A column past the end reads the last one's row and is not written.
                int4 codes[ kColumnsPerWarp ];
                float scale[ kColumnsPerWarp ];

#pragma unroll
                for ( int c = 0; c < kColumnsPerWarp; ++c )
                {
                    const int64_t row = static_cast<int64_t>( expert ) * hidden + min( first_column + c, hidden - 1 );

                    codes[ c ] = reinterpret_cast<const int4*>( down + row * ( intermediate / 2 ) )[ within ];
                    scale[ c ] = __half2float( down_scales[ row * groups_per_row + within ] );
                }

                const float combine = __bfloat162float( weights[ token * top_k + slot ] );

#pragma unroll
                for ( int c = 0; c < kColumnsPerWarp; ++c )
                {
                    sum[ c ] = fmaf( combine * scale[ c ], group_dot_fp32( codes[ c ], staged + group * kPaddedGroup ), sum[ c ] );
                }
            }

#pragma unroll
            for ( int c = 0; c < kColumnsPerWarp; ++c )
            {
                const float total = warp_sum( sum[ c ] );

                if ( threadIdx.x == 0 && first_column + c < hidden )
                {
                    output[ token * hidden + first_column + c ] = __float2bfloat16( total );
                }
            }
        }
    }

    template<typename TFunctor>
    void launch_moe_gated_gather_int4(
        const __nv_bfloat16* input, const uint8_t* gate_up, const __half* gate_up_scales, const int32_t* indices,
        float* gated, int tokens, int hidden, int intermediate, int experts, int top_k,
        TFunctor functor, cudaStream_t stream )
    {
        if ( tokens == 0 )
        {
            return;
        }

        const dim3 block( 32, kWarpsPerBlock );
        const int units_per_block = kWarpsPerBlock * kUnitsPerWarp;
        const dim3 grid( ( intermediate + units_per_block - 1 ) / units_per_block, top_k, tokens );

        moe_gated_gather_int4_kernel<TFunctor><<<grid, block, 0, stream>>>(
            input, gate_up, gate_up_scales, indices, gated, hidden, intermediate, experts, top_k, functor );

        cudaCheck( cudaGetLastError() );
    }

    void launch_moe_combine_gather_int4(
        const float* gated, const uint8_t* down, const __half* down_scales, const __nv_bfloat16* weights,
        const int32_t* indices, __nv_bfloat16* output,
        int tokens, int hidden, int intermediate, int experts, int top_k,
        cudaStream_t stream )
    {
        if ( tokens == 0 )
        {
            return;
        }

        const int shared_bytes = top_k * ( intermediate / kGroup ) * kPaddedGroup * static_cast<int>( sizeof( float ) );

        // Past the default 48 KB the kernel must ask; idempotent, as GatedDeltaRule.cu does per launch.
        if ( shared_bytes > 48 * 1024 )
        {
            cudaCheck( cudaFuncSetAttribute(
                moe_combine_gather_int4_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes ) );
        }

        const dim3 block( 32, kWarpsPerBlock );
        const int columns_per_block = kWarpsPerBlock * kColumnsPerWarp;
        const dim3 grid( ( hidden + columns_per_block - 1 ) / columns_per_block, tokens );

        moe_combine_gather_int4_kernel<<<grid, block, shared_bytes, stream>>>(
            gated, down, down_scales, weights, indices, output, hidden, intermediate, experts, top_k );

        cudaCheck( cudaGetLastError() );
    }

    template void launch_moe_gated_gather_int4<Mila::Dnn::Activations::GeluTanh>(
        const __nv_bfloat16*, const uint8_t*, const __half*, const int32_t*, float*, int, int, int, int, int,
        Mila::Dnn::Activations::GeluTanh, cudaStream_t );
    template void launch_moe_gated_gather_int4<Mila::Dnn::Activations::Silu>(
        const __nv_bfloat16*, const uint8_t*, const __half*, const int32_t*, float*, int, int, int, int, int,
        Mila::Dnn::Activations::Silu, cudaStream_t );
}
