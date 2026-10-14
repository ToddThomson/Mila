/**
 * @file PackageRequest.ixx
 * @brief One model's files and description, as a caller hands them to the packager.
 */

export module Distribution.PackageRequest;

import std;
export import Distribution.ModelCapabilities;

namespace Mila::Distribution
{
    /**
     * @brief One model's files, as a caller hands them to the packager.
     *
     * Paths outside the package directory are brought in; paths already inside it are left where
     * they are. A 6.8 GB weights file exported straight into the package therefore costs no copy.
     */
    export struct PackageRequest
    {
        std::filesystem::path directory;

        /// Empty takes the package directory's name.
        std::string name;

        std::string architecture;

        /// Which of the family's architectures, "3.1" for Llama 3.1. Empty for a family with only one.
        std::string architecture_version;

        std::string variant;
        std::string weight_quantization;
        std::string minimum_mila_version;

        /// Lineage, published with the model because attribution has to travel with the weights.
        std::string base_model;
        std::string license_id;

        /// What the model was trained to do, and the modalities besides text this Mila runs for it.
        ModelCapabilities capabilities;

        std::filesystem::path weights;
        std::filesystem::path tokenizer;

        /// The draft model the package carries for speculative decoding. Empty when it carries none.
        std::filesystem::path drafter;

        std::filesystem::path license;

        /// Some licenses require their attribution to travel in a file of its own rather than
        /// inside the license text -- Llama names a "Notice" file specifically. Empty when the
        /// source license asks for nothing beyond LICENSE.
        std::filesystem::path notice;

        std::filesystem::path model_card;

        /**
         * @brief Build over a directory that already describes a model.
         *
         * Off by default. One package is one model, so a manifest already there is either a
         * rebuild the caller meant or a directory they have forgotten holds something else, and
         * buildPackage cannot tell which -- the same reason InstallOptions::replace exists.
         */
        bool replace{ false };
    };
}
