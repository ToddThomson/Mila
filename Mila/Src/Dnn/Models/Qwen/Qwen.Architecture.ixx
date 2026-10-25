/**
 * @file Qwen.Architecture.ixx
 * @brief What a manifest's "qwen" names: the model type, its tokenizer and its chat template.
 */

export module Dnn.Models.QwenArchitecture;

import std;
import Compute.DeviceType;
import Dnn.TensorDataType;
import Dnn.Models.Conversation.Protocol;
import Dnn.Models.QwenModel;
import Dnn.Models.QwenChatProtocol;
import Data.BpeTokenizer;

namespace Mila::Dnn
{
    /**
     * @brief Qwen 3.8's entry in the architecture list (Architectures.ixx, ModelHandle.md 3.3).
     */
    export struct QwenArchitecture
    {
        /// What a manifest's `architecture` says.
        static constexpr std::string_view name = "qwen";

        /// The chat templates this build renders, as a manifest's `architecture_version` names them.
        static constexpr std::array<std::string_view, 1> template_versions{ "3.8" };

        static constexpr bool drafts = false;

        template<Compute::DeviceType TDeviceType, TensorDataType TPrecision>
        using Model = QwenModel<TDeviceType, TPrecision>;

        static std::shared_ptr<Data::BpeTokenizer> loadTokenizer( const std::filesystem::path& path )
        {
            return Data::BpeTokenizer::loadQwen( path );
        }

        /// One template, so a manifest written before `architecture_version` existed means it. Null for any other.
        static std::unique_ptr<Conversation::Protocol> chatProtocol( std::string_view version )
        {
            if ( version.empty() || version == "3.8" )
            {
                return std::make_unique<Qwen::QwenChatProtocol>();
            }

            return nullptr;
        }
    };
}
