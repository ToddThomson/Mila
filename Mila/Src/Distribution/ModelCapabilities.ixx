/**
 * @file ModelCapabilities.ixx
 * @brief What a model was trained to do, as its manifest declares it.
 *
 * See Specifications/ModelDistribution.md, Capabilities and modalities.
 */

module;
#include <string>
#include <vector>

export module Distribution.ModelCapabilities;

namespace Mila::Distribution
{
    /**
     * @brief What a model was trained to do, and the input it reads besides text.
     *
     * Declared by the manifest, never inferred from the family, so two models of one family can differ. A manifest
     * written before these fields existed declares only `instruct`, and none of the rest.
     */
    export struct ModelCapabilities
    {
        /// Instruction-tuned: it takes turns and a chat template.
        bool instruct{ false };

        /// It has a reasoning channel the prompt can turn on.
        bool thinking{ false };

        /// Trained for tool calls, as the source checkpoint states.
        bool tools{ false };

        /**
         * @brief Mila's own verdict that the model works as an agent: it still finds facts, obeys its instructions and
         *        calls tools correctly deep into a conversation.
         *
         * Measured, not copied: written only for a model whose context profile has been measured
         * (Specifications/ContextProfile.md). A model can call tools and not be agentic.
         */
        bool agentic{ false };

        /**
         * @brief Input modalities besides text, as the manifest names them ("image", "video", "audio").
         *
         * Names, not a closed set: a manifest from a newer Mila may name one this build does not run, and the
         * installed model reports only those it does.
         */
        std::vector<std::string> modalities;
    };
}
