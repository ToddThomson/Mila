/**
 * @file SamplingOverride.ixx
 * @brief The sampling settings a program asks for, each over what the model's publisher recommends.
 */

export module AI.SamplingOverride;

import std;

namespace Mila::AI
{
    /**
     * @brief Each setting set stands over the publisher's recommendation for the thinking in effect; each left empty
     *        is the publisher's, or the sampler's neutral value where the publisher recommends none.
     */
    export struct SamplingOverride
    {
        std::optional<float> temperature;

        /// 0 disables top-k truncation; 1 is greedy.
        std::optional<int> top_k;

        /// 1.0 disables nucleus truncation.
        std::optional<float> top_p;

        std::optional<float> presence_penalty;
    };
}
