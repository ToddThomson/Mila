/**
 * @file ExportOptions.ixx
 * @brief What one export loads, at which weight format, and where its package goes.
 */

module;
#include <filesystem>

export module Tools.ExportArtifact.ExportOptions;

import Mila;

namespace Mila::Tools
{
    using Mila::Dnn::WeightQuantization;

    export struct ExportOptions
    {
        std::filesystem::path source;
        std::filesystem::path destination;
        WeightQuantization quantization{ WeightQuantization::FP4 };

        /// Load only, print a logits fingerprint, and write nothing.
        bool fingerprint_only{ false };

        /**
         * @brief Where to assemble the package. Empty means none.
         *
         * Packaging hashes every file it declares, so exporting straight into the package
         * directory is the cheap arrangement: a file already there is described in place rather
         * than copied.
         */
        std::filesystem::path package_directory;

        /// Tokenizer to record in the manifest. Optional.
        std::filesystem::path tokenizer;

        /// Source model's license text, required by every license Mila republishes.
        std::filesystem::path license;

        /// Attribution the source license requires be carried in a file of its own -- Llama
        /// names a "Notice" file specifically. Optional; most licenses ask for nothing here.
        std::filesystem::path notice;

        /// Model card, carrying the statement that the weights were modified.
        std::filesystem::path model_card;

        /// Build over a package directory that already describes a model.
        bool replace_package{ false };
    };
}
