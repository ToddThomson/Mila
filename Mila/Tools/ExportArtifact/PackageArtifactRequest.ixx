/**
 * @file PackageArtifactRequest.ixx
 * @brief A package to assemble around weights that already exist.
 */

export module Tools.ExportArtifact.PackageArtifactRequest;

import std;
import Mila;

namespace Mila::Tools
{
    /**
     * @brief Package weights that already exist on disk.
     *
     * The counterpart to --package after an export: it takes finished weights rather than
     * producing them, so migrating a file, adding a licence, or fixing a model card costs a
     * manifest write instead of a multi-gigabyte reload on the GPU.
     */
    export struct PackageArtifactRequest
    {
        std::filesystem::path directory;
        std::filesystem::path weights;
        std::filesystem::path tokenizer;
        std::filesystem::path license;
        std::filesystem::path notice;
        std::filesystem::path model_card;

        /// Empty takes the package directory's name.
        std::string name;

        /// Empty derives the variant from the weights themselves.
        std::string variant;

        /// Which of the family's architectures, "3.1" for Llama 3.1. Names the chat template the model runs with.
        std::string architecture_version;

        /// Lineage, published with the model.
        std::string base_model;
        std::string license_id;

        /// What the model was trained to do, and the modalities besides text this Mila runs for it.
        Mila::Distribution::ModelCapabilities capabilities;

        /// The publisher's sampling settings, transcribed from the source model's card. Empty when it states none.
        std::filesystem::path sampling;

        /// The draft model the package carries for speculative decoding. Empty when it carries none.
        std::filesystem::path drafter;

        /// Build over a package directory that already describes a model.
        bool replace{ false };
    };
}
