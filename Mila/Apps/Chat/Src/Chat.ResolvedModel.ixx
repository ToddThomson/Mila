/**
 * @file Chat.ResolvedModel.ixx
 * @brief Everything Chat needs to load a model, resolved from its store record.
 */

export module Chat.ResolvedModel;

import std;
import Chat.Config;
import Distribution.SamplingRecommendation;

namespace Mila::ChatApp
{
    /**
     * @brief Everything Chat needs to load a model, resolved from its store record.
     *
     * The family, the stored format and the capabilities are the library's answer (InstalledModel); the rest
     * is Chat's own: a quantization asked for on load.
     */
    export struct ResolvedModel
    {
        std::string name;

        std::filesystem::path weights;
        std::filesystem::path tokenizer;

        ModelType family{ ModelType::Gemma };
        QuantizationMode quantization{ QuantizationMode::None };

        /// Lineage from the record. Carried because a license that requires attribution requires
        /// it wherever the model is presented, and the session is one of those places.
        std::string base_model;
        std::string license;

        bool instruct{ false };

        /// Whether the model has a reasoning channel at all, as its manifest declares. Read as a capability so
        /// the session never offers a thinking mode the weights cannot produce.
        bool thinking_capable{ false };

        /// Trained for tool calls, and Mila's measured verdict that it works as an agent, as the manifest declares.
        bool tools_capable{ false };
        bool agentic{ false };

        /// What the model's publisher recommends sampling at: its manifest's settings, or its family's.
        std::vector<Distribution::SamplingRecommendation> sampling;

        /// True when the quantization is a load-time choice rather than what the weights already
        /// are. Both land in the same field, but only this one is a fact the model's name omits.
        bool quantization_applied_at_load{ false };
    };
}
