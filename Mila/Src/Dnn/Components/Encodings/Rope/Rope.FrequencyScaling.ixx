/**
 * @file Rope.FrequencyScaling.ixx
 * @brief Llama 3's rotary frequency scaling: how a checkpoint trained at a short context reads a longer one.
 */

module;
#include <stdexcept>

export module Dnn.Components.RopeFrequencyScaling;

import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Rescales each rotary frequency by its wavelength against the context the checkpoint was first trained at.
     *
     * The rule HuggingFace names `llama3`. With L the original context, a frequency whose wavelength is longer than
     * L / low_frequency_factor is divided by `factor`; one shorter than L / high_frequency_factor is kept; one between
     * is interpolated, weighted by (L / wavelength - low_frequency_factor) / (high_frequency_factor - low_frequency_factor).
     * The scaled frequencies apply at every position, not only past L, so a checkpoint trained with them reads even a
     * short prompt slightly differently without them.
     *
     * The default, an original context of 0, is no scaling.
     */
    export struct RopeFrequencyScaling
    {
        float factor{ 1.0f };
        float low_frequency_factor{ 1.0f };
        float high_frequency_factor{ 1.0f };
        dim_t original_context_length{ 0 };

        bool isScaled() const noexcept
        {
            return original_context_length > 0;
        }

        /// @throws std::invalid_argument when a scaled rule could not be applied.
        void validate() const
        {
            if ( !isScaled() )
            {
                return;
            }

            if ( factor <= 0.0f )
            {
                throw std::invalid_argument( "RopeFrequencyScaling: factor must be > 0" );
            }

            if ( low_frequency_factor <= 0.0f || high_frequency_factor <= low_frequency_factor )
            {
                throw std::invalid_argument(
                    "RopeFrequencyScaling: need 0 < low_frequency_factor < high_frequency_factor" );
            }
        }

        bool operator==( const RopeFrequencyScaling& ) const = default;
    };
}
