/**
 * @file Chat.FootprintPrediction.ixx
 * @brief A footprint, or why there is not one.
 */

module;
#include <optional>
#include <string>

export module Chat.FootprintPrediction;

import Mila;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn;

    /**
     * @brief A footprint, or why there is not one.
     *
     * The two consumers need different halves of the same answer. A pre-flight proceeds silently
     * when nothing can be predicted, because it must never be the thing that stops a model from
     * being tried. `context_length: "auto"` cannot: a derived number with no provenance is the
     * complaint ChatConfiguration.md opens with, so it has to say what it fell back to and why.
     * Carrying the reason costs the silent caller nothing.
     */
    export struct FootprintPrediction
    {
        std::optional<MemoryStats> required;

        /// Why there is no prediction, phrased to read inside a parenthetical. Empty when
        /// required holds one.
        std::string unavailable_reason;

        /// How this context length would chunk its prefill. Zeroed when there is no prediction,
        /// and for a family whose transformer does not chunk.
        PrefillChunking prefill;
    };
}
