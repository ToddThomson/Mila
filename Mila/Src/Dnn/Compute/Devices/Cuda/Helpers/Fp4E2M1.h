/**
 * @file Fp4E2M1.h
 * @brief The FP4 E2M1 nibble decode every kernel that reads packed FP4 weights goes through.
 */

#pragma once

#include <cuda_runtime.h>
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
    }
}
