/**
 * @file Gemma.Architecture.ixx
 * @brief What a manifest's "gemma" names: the model type, its tokenizer and its chat template.
 */

export module Dnn.Models.GemmaArchitecture;

import std;
import Compute.DeviceType;
import Dnn.TensorDataType;
import Dnn.Models.Conversation.Protocol;
import Dnn.Models.GemmaModel;
import Dnn.Models.GemmaChatProtocol;
import Data.BpeTokenizer;

namespace Mila::Dnn
{
    /**
     * @brief Gemma 4's entry in the architecture list (Architectures.ixx, ModelHandle.md 3.3).
     */
    export struct GemmaArchitecture
    {
        /// What a manifest's `architecture` says.
        static constexpr std::string_view name = "gemma";

        /// The chat templates this build renders, as a manifest's `architecture_version` names them.
        static constexpr std::array<std::string_view, 1> template_versions{ "4" };

        /// Its packages may carry an MTP draft model, and this build decodes with one.
        static constexpr bool drafts = true;

        template<Compute::DeviceType TDeviceType, TensorDataType TPrecision>
        using Model = GemmaModel<TDeviceType, TPrecision>;

        static std::shared_ptr<Data::BpeTokenizer> loadTokenizer( const std::filesystem::path& path )
        {
            return Data::BpeTokenizer::loadGemma( path );
        }

        /// One template, so a manifest written before `architecture_version` existed means it. Null for any other.
        static std::unique_ptr<Conversation::Protocol> chatProtocol( std::string_view version )
        {
            if ( version.empty() || version == "4" )
            {
                return std::make_unique<Gemma::GemmaChatProtocol>();
            }

            return nullptr;
        }
    };
}
