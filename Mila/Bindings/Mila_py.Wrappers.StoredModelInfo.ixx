/**
 * @file Mila_py.Wrappers.StoredModelInfo.ixx
 * @brief One installed model, flattened to std types.
 */

export module Mila.Bindings:StoredModelInfo;

import std;

namespace Mila::Bindings
{
    /**
     * @brief One installed model, flattened to std types.
     *
     * Paths are strings rather than std::filesystem::path: this surface crosses into Python,
     * where a path is a string anyway.
     */
    export struct StoredModelInfo
    {
        std::string name;
        std::string architecture;

        /// The family the library loads it as -- "gemma", "llama" or "qwen" -- or empty when this build has none.
        std::string family;

        std::string variant;
        std::string weight_quantization;

        /// Instruction-tuned. Decides the prompt template a consumer applies.
        bool instruct{ false };

        /// A reasoning channel; trained for tool calls; Mila's measured verdict that it works as an agent.
        bool thinking{ false };
        bool tools{ false };
        bool agentic{ false };

        /// It carries a draft model this build decodes with.
        bool mtp{ false };

        /// Inputs besides text this build runs for it.
        std::vector<std::string> modalities;

        std::string base_model;
        std::string license;

        /// Provenance. Empty hub means the model was published from this machine.
        std::string hub;
        std::string owner;
        std::string repository;
        std::string revision;
        std::string installed_at;

        std::string weights_path;
        std::string tokenizer_path;

        /// False when a declared blob is missing, which makes the model unloadable.
        bool complete{ false };

        std::uint64_t bytes_on_disk{ 0 };
    };
}
