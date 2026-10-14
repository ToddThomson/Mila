/**
 * @file ChatProtocolForArchitecture.ixx
 * @brief The chat template a model runs with, from the architecture its manifest declares.
 */

export module Dnn.Models.ChatProtocolForArchitecture;

import std;
import Dnn.Models.Conversation;
import Dnn.Models.GemmaChatProtocol;
import Dnn.Models.LlamaChatProtocol;
import Dnn.Models.QwenChatProtocol;

namespace Mila::Dnn
{
    /**
     * @brief The protocol of the chat template that ships with a model of this architecture.
     *
     * Read from what the manifest declares, never from its name or lineage. A family with one template takes
     * an empty version as that one; Llama has two (3.1 and 3.2), so an empty or unknown version is refused.
     * Until the model handle's architecture list (ModelHandle.md 3.3) this is the one place an architecture
     * becomes a template.
     *
     * @param architecture The family, as the manifest's `architecture` writes it: gemma, llama or qwen.
     * @param architecture_version The manifest's `architecture_version`, empty when it declared none.
     * @throws std::invalid_argument for an architecture with no chat template here, naming the ones it has.
     */
    export inline std::unique_ptr<Conversation::Protocol> chatProtocolFor(
        std::string_view architecture, std::string_view architecture_version )
    {
        if ( architecture == "gemma" && ( architecture_version.empty() || architecture_version == "4" ) )
        {
            return std::make_unique<Gemma::GemmaChatProtocol>();
        }

        if ( architecture == "qwen" && ( architecture_version.empty() || architecture_version == "3.8" ) )
        {
            return std::make_unique<Qwen::QwenChatProtocol>();
        }

        if ( architecture == "llama" && architecture_version == "3.1" )
        {
            return std::make_unique<Llama::LlamaChatProtocol>( Llama::LlamaChatProtocol::llama31() );
        }

        if ( architecture == "llama" && architecture_version == "3.2" )
        {
            return std::make_unique<Llama::LlamaChatProtocol>( Llama::LlamaChatProtocol::llama32() );
        }

        if ( architecture == "llama" && architecture_version.empty() )
        {
            throw std::invalid_argument(
                "a Llama model must declare its architecture_version (3.1 or 3.2): the two ship different chat templates" );
        }

        throw std::invalid_argument( std::format(
            "no chat template for architecture '{}'{}; there are: gemma 4, llama 3.1, llama 3.2, qwen 3.8",
            architecture, architecture_version.empty() ? std::string{} : std::format( " {}", architecture_version ) ) );
    }
}
