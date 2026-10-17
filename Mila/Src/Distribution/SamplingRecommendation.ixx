/**
 * @file SamplingRecommendation.ixx
 * @brief One sampling setting a model's publisher recommends, and the configuration it applies to.
 *
 * See Specifications/ChatConfiguration.md section 5.
 */

export module Distribution.SamplingRecommendation;

import std;

namespace Mila::Distribution
{
    /**
     * @brief A setting the base model's publisher recommends, keyed by how the model is run.
     *
     * Each knob is present only when the publisher states it; an absent one is the sampler's own neutral value,
     * never another setting's. Thinking on or off is the only configuration any publisher keys on today.
     */
    export struct SamplingRecommendation
    {
        /// The thinking this setting applies to. Empty applies to either.
        std::optional<bool> thinking;

        std::optional<float> temperature;
        std::optional<float> top_p;
        std::optional<int> top_k;
        std::optional<float> presence_penalty;
    };

    /**
     * @brief The first setting that applies with thinking on or off, or null when none does.
     *
     * A model without a reasoning channel is run with thinking off.
     */
    export inline const SamplingRecommendation* matchingSampling(
        std::span<const SamplingRecommendation> settings, bool thinking ) noexcept
    {
        for ( const auto& setting : settings )
        {
            if ( !setting.thinking.has_value() || *setting.thinking == thinking )
            {
                return &setting;
            }
        }

        return nullptr;
    }
}
