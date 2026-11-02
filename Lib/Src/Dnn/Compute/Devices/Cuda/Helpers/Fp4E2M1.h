/**
 * @file Fp4E2M1.h
 * @brief The FP4 E2M1 nibble decode every kernel that reads packed FP4 weights goes through.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda
{
    namespace
    {
        // bit3 = sign, bits[2:0] index the magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}; negatives are sign-magnitude.
        __device__ __forceinline__ float fp4_e2m1_decode( uint8_t nibble )
        {
            static constexpr float kLut[ 8 ] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
            const float magnitude = kLut[ nibble & 0x7u ];

            return ( nibble & 0x8u ) ? -magnitude : magnitude;
        }

        // Decodes 8 packed FP4-E2M1 nibbles (one uint32, low nibble = even column)
        // into four raw (unscaled) BF16 pairs via byte-permute table selects. All
        // eight E2M1 magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6} are exact in BF16, so
        // this is value-identical to fp4_e2m1_decode -- but with no dynamically
        // indexed constant-memory LUT, whose per-lane address divergence replays
        // the load up to 8 ways across a warp. The PRMT selectors must be masked
        // to 3 bits per nibble: selector bit 3 engages PRMT's sign-replicate mode,
        // and the FP4 sign bit is instead injected into BF16 bit 15 afterwards.
        __device__ __forceinline__ void fp4x8_decode_bf16x2(
            uint32_t w_packed,
            __nv_bfloat162 ( &w )[ 4 ] )
        {
            // Per-index byte tables of the BF16 magnitude patterns:
            // index:   0       1       2       3       4       5       6       7
            // bf16: 0x0000  0x3F00  0x3F80  0x3FC0  0x4000  0x4040  0x4080  0x40C0
            constexpr uint32_t kHighBytes0123 = 0x3F3F3F00u;
            constexpr uint32_t kHighBytes4567 = 0x40404040u;
            constexpr uint32_t kLowBytes0123 = 0xC0800000u;
            constexpr uint32_t kLowBytes4567 = 0xC0804000u;

            const uint32_t selector_lo = w_packed & 0x7777u;
            const uint32_t selector_hi = ( w_packed >> 16 ) & 0x7777u;

            const uint32_t high_lo4 = __byte_perm( kHighBytes0123, kHighBytes4567, selector_lo );
            const uint32_t low_lo4 = __byte_perm( kLowBytes0123, kLowBytes4567, selector_lo );
            const uint32_t high_hi4 = __byte_perm( kHighBytes0123, kHighBytes4567, selector_hi );
            const uint32_t low_hi4 = __byte_perm( kLowBytes0123, kLowBytes4567, selector_hi );

            // Interleave low/high bytes into two BF16 values per word, then inject
            // the FP4 sign bits (bit 4j+3 of w_packed) into BF16 bit 15.
            uint32_t pair01 = __byte_perm( low_lo4, high_lo4, 0x5140 );
            uint32_t pair23 = __byte_perm( low_lo4, high_lo4, 0x7362 );
            uint32_t pair45 = __byte_perm( low_hi4, high_hi4, 0x5140 );
            uint32_t pair67 = __byte_perm( low_hi4, high_hi4, 0x7362 );

            const uint32_t w_high = w_packed >> 16;
            pair01 |= ( ( w_packed & 0x8u ) << 12 ) | ( ( w_packed & 0x80u ) << 24 );
            pair23 |= ( ( w_packed & 0x800u ) << 4 ) | ( ( w_packed & 0x8000u ) << 16 );
            pair45 |= ( ( w_high & 0x8u ) << 12 ) | ( ( w_high & 0x80u ) << 24 );
            pair67 |= ( ( w_high & 0x800u ) << 4 ) | ( ( w_high & 0x8000u ) << 16 );

            w[ 0 ] = *reinterpret_cast<const __nv_bfloat162*>( &pair01 );
            w[ 1 ] = *reinterpret_cast<const __nv_bfloat162*>( &pair23 );
            w[ 2 ] = *reinterpret_cast<const __nv_bfloat162*>( &pair45 );
            w[ 3 ] = *reinterpret_cast<const __nv_bfloat162*>( &pair67 );
        }
    }
}
