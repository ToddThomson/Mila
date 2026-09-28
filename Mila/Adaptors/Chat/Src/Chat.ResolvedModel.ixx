/**
 * @file Chat.ResolvedModel.ixx
 * @brief Everything Chat needs to load a model, resolved from its store record.
 */

module;
#include <filesystem>
#include <string>

export module Chat.ResolvedModel;

import Chat.Config;

namespace Mila::ChatApp
{
    /**
     * @brief Everything Chat needs to load a model, resolved from its store record.
     *
     * Assembled rather than looked up. Architecture and variant come from the record; the rest
     * are deployment decisions Chat owns, keyed on architecture.
     */
    export struct ResolvedModel
    {
        std::string name;

        std::filesystem::path weights;
        std::filesystem::path tokenizer;

        ModelType family{ ModelType::Gemma };
        ModelPrecision precision{ ModelPrecision::BF16 };
        QuantizationMode quantization{ QuantizationMode::None };

        /// Lineage from the record. Carried because a license that requires attribution requires
        /// it wherever the model is presented, and the session is one of those places.
        std::string base_model;
        std::string license;

        bool instruct{ false };
        bool streaming_capable{ false };

        /// Whether the model has a reasoning channel at all. Read as a capability so the session
        /// never offers a thinking mode the weights cannot produce.
        bool thinking_capable{ false };

        /// True when the quantization is a load-time choice rather than what the weights already
        /// are. Both land in the same field, but only this one is a fact the model's name omits.
        bool quantization_applied_at_load{ false };
    };
}
