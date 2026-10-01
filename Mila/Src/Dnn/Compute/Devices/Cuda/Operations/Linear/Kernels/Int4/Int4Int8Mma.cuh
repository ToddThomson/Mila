/**
 * @file Int4Int8Mma.cuh
 * @brief Device helpers of the Q4_0 INT8 tensor-core GEMMs: asynchronous tile copies, the packed-code fragment
 * unpack and the biased k32 MMA. Shared by Linear's prefill and the expert bank's grouped prefill.
 */

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Int4Mma
{
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
    inline constexpr int kFloatBias = 0x4B400000;
    inline constexpr float kFloatBiasValue = 12582912.0f;

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
}
