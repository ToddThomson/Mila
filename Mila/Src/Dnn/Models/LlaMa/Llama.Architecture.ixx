/**
 * @file Llama.Architecture.ixx
 * @brief What a manifest's "llama" names: the model type, its tokenizer and its two chat templates.
 */

export module Dnn.Models.LlamaArchitecture;

import std;
import Compute.DeviceType;
import Dnn.TensorDataType;
import Dnn.Models.Conversation.Protocol;
import Dnn.Models.LlamaModel;
import Dnn.Models.LlamaChatProtocol;
import Data.BpeTokenizer;

namespace Mila::Dnn
{
    /**
     * @brief Llama 3.1's and 3.2's entry in the architecture list (Architectures.ixx, ModelHandle.md 3.3).
     */
    export struct LlamaArchitecture
    {
        /// What a manifest's `architecture` says.
        static constexpr std::string_view name = "llama";

        /// The chat templates this build renders, as a manifest's `architecture_version` names them.
        static constexpr std::array<std::string_view, 2> template_versions{ "3.1", "3.2" };

        static constexpr bool drafts = false;

        template<Compute::DeviceType TDeviceType, TensorDataType TPrecision>
        using Model = LlamaModel<TDeviceType, TPrecision>;

        static std::shared_ptr<Data::BpeTokenizer> loadTokenizer( const std::filesystem::path& path )
        {
            return Data::BpeTokenizer::loadLlama32( path );
        }

        /**
         * @brief The template a version names; null for a version this build lacks.
         *
         * @throws std::invalid_argument for no version at all: the two templates differ, and a guess would render
         *         one of them wrong.
         */
        static std::unique_ptr<Conversation::Protocol> chatProtocol( std::string_view version )
        {
            if ( version == "3.1" )
            {
                return std::make_unique<Llama::LlamaChatProtocol>( Llama::LlamaChatProtocol::llama31() );
            }

            if ( version == "3.2" )
            {
                return std::make_unique<Llama::LlamaChatProtocol>( Llama::LlamaChatProtocol::llama32() );
            }

            if ( version.empty() )
            {
                throw std::invalid_argument(
                    "a Llama model must declare its architecture_version (3.1 or 3.2): the two ship different chat templates" );
            }

            return nullptr;
        }
    };
}
