/**
 * @file FamilySampling.ixx
 * @brief The sampling each family's publisher recommends, for a model whose manifest states none.
 *
 * See Specifications/ChatConfiguration.md section 5.
 */

export module Distribution.FamilySampling;

import std;
import Distribution.SamplingRecommendation;
import Distribution.ModelRecord;

namespace Mila::Distribution
{
    /**
     * @brief The settings the family's publisher recommends, by the architecture a manifest declares.
     *
     * Empty for an architecture whose publisher recommends none here, which leaves the sampler at its neutral
     * values. Read 2026-10-17 from each publisher's own source.
     */
    export inline std::vector<SamplingRecommendation> familySampling( std::string_view architecture )
    {
        // Qwen 3.8's card, Best Practices: one setting for thinking at every effort, one without it.
        if ( architecture == "qwen" )
        {
            return {
                { .thinking = true, .temperature = 1.0f, .top_p = 0.95f, .top_k = 20 },
                { .thinking = false, .temperature = 0.7f, .top_p = 0.80f, .top_k = 20, .presence_penalty = 1.5f } };
        }

        // Gemma 4's card, "across all use cases".
        if ( architecture == "gemma" )
        {
            return { { .temperature = 1.0f, .top_p = 0.95f, .top_k = 64 } };
        }

        // Llama 3.1 and 3.2 Instruct: the cards state none; generation_config.json gives these.
        if ( architecture == "llama" )
        {
            return { { .temperature = 0.6f, .top_p = 0.9f } };
        }

        return {};
    }

    /**
     * @brief What a session opens at for this installed model: its manifest's settings, or its family's.
     *
     * The manifest's settings replace the family's whole rather than knob by knob, since a mix of the two is a
     * setting no publisher recommended.
     */
    export inline std::vector<SamplingRecommendation> recommendedSampling( const ModelRecord& record )
    {
        return record.sampling.empty() ? familySampling( record.architecture ) : record.sampling;
    }
}
