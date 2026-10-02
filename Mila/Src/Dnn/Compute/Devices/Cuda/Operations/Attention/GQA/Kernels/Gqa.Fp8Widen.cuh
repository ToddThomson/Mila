#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    /**
     * @brief Widens sixteen E4M3 codes into sixteen BF16 values at destination, unscaled.
     *
     * Every E4M3 value is exact in BF16, so a staged FP8 tile widened here feeds the same MMAs as a BF16 tile. One
     * 16-byte read and two 16-byte writes: destination must be 16-byte aligned.
     */
    __device__ __forceinline__ void widen_sixteen_codes( const uint4 codes, __nv_bfloat16* destination )
    {
        const uint32_t words[ 4 ] = { codes.x, codes.y, codes.z, codes.w };
        uint32_t widened[ 8 ];

#pragma unroll
        for ( int word = 0; word < 4; ++word )
        {
#pragma unroll
            for ( int half = 0; half < 2; ++half )
            {
                const __nv_fp8x2_storage_t pair = static_cast<__nv_fp8x2_storage_t>( words[ word ] >> ( 16 * half ) );
                const __nv_bfloat162 value = __float22bfloat162_rn(
                    __half22float2( __half2( __nv_cvt_fp8x2_to_halfraw2( pair, __NV_E4M3 ) ) ) );
                widened[ 2 * word + half ] = *reinterpret_cast<const uint32_t*>( &value );
            }
        }

        reinterpret_cast<uint4*>( destination )[ 0 ] = make_uint4( widened[ 0 ], widened[ 1 ], widened[ 2 ], widened[ 3 ] );
        reinterpret_cast<uint4*>( destination )[ 1 ] = make_uint4( widened[ 4 ], widened[ 5 ], widened[ 6 ], widened[ 7 ] );
    }

    /**
     * @brief Widens sixteen E4M3 codes into sixteen FP16 values at destination, times their row's scale.
     *
     * Each code times the scale is formed in FP32 and rounded to FP16 once; with a power-of-two scale the value is
     * exact. destination must be 16-byte aligned.
     */
    __device__ __forceinline__ void widen_sixteen_codes_to_half( const uint4 codes, float scale, __half* destination )
    {
        const uint32_t words[ 4 ] = { codes.x, codes.y, codes.z, codes.w };
        uint32_t widened[ 8 ];

#pragma unroll
        for ( int word = 0; word < 4; ++word )
        {
#pragma unroll
            for ( int half = 0; half < 2; ++half )
            {
                const __nv_fp8x2_storage_t pair = static_cast<__nv_fp8x2_storage_t>( words[ word ] >> ( 16 * half ) );
                const float2 value = __half22float2( __half2( __nv_cvt_fp8x2_to_halfraw2( pair, __NV_E4M3 ) ) );
                const __half2 scaled = __floats2half2_rn( value.x * scale, value.y * scale );
                widened[ 2 * word + half ] = *reinterpret_cast<const uint32_t*>( &scaled );
            }
        }

        reinterpret_cast<uint4*>( destination )[ 0 ] = make_uint4( widened[ 0 ], widened[ 1 ], widened[ 2 ], widened[ 3 ] );
        reinterpret_cast<uint4*>( destination )[ 1 ] = make_uint4( widened[ 4 ], widened[ 5 ], widened[ 6 ], widened[ 7 ] );
    }
}
