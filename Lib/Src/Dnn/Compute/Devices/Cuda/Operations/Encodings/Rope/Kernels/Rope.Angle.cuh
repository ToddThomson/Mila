#pragma once
#include <cuda_runtime.h>
#include "Rope.cuh"

namespace Mila::Dnn::Compute::Cuda::Rope
{
    /**
     * @brief Llama 3's rescaling of one inverse frequency by its wavelength (RopeFrequencyScaling).
     *
     * Mirrors HuggingFace's `_compute_llama3_parameters`: long wavelengths are divided by the factor, short ones
     * kept, and the band between interpolated. An original context of 0 returns the frequency unchanged.
     */
    __device__ __forceinline__ float scale_inverse_frequency(
        float inverse_frequency,
        float factor,
        float low_frequency_factor,
        float high_frequency_factor,
        int original_context_length )
    {
        if ( original_context_length <= 0 )
            return inverse_frequency;

        constexpr float kPi = static_cast<float>( 3.14159265358979323846 );

        const float original = static_cast<float>( original_context_length );
        const float wavelength = 2.0f * kPi / inverse_frequency;

        if ( wavelength > original / low_frequency_factor )
            return inverse_frequency / factor;

        if ( wavelength < original / high_frequency_factor )
            return inverse_frequency;

        const float smooth = ( original / wavelength - low_frequency_factor ) / ( high_frequency_factor - low_frequency_factor );

        return ( 1.0f - smooth ) * inverse_frequency / factor + smooth * inverse_frequency;
    }

    /**
     * @brief cos and sin of the angle at (position, pair).
     *
     * The one definition every rotation uses; a kernel that fuses RoPE into another operation includes it rather
     * than restating it. A pair past rope_pairs carries zero frequency, so its rotation is the identity (cos 1,
     * sin 0).
     */
    __device__ __forceinline__ void rope_cos_sin(
        int position, int pair, const RopeAngleParameters& angles, float& cos_value, float& sin_value )
    {
        if ( pair < angles.rope_pairs )
        {
            const float theta = scale_inverse_frequency(
                __powf( angles.base, -2.0f * static_cast<float>( pair ) / static_cast<float>( angles.frequency_denominator ) ),
                angles.scaling_factor, angles.scaling_low_frequency_factor, angles.scaling_high_frequency_factor,
                angles.scaling_original_context_length );
            const float angle = static_cast<float>( position ) * theta;

            cos_value = cosf( angle );
            sin_value = sinf( angle );
        }
        else
        {
            cos_value = 1.0f;
            sin_value = 0.0f;
        }
    }
}
