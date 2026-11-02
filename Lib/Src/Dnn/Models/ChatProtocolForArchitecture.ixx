/**
 * @file ChatProtocolForArchitecture.ixx
 * @brief The chat template a model runs with, from the architecture its manifest declares.
 */

export module Dnn.Models.ChatProtocolForArchitecture;

import std;
import Dnn.Models.Conversation;
import Dnn.Models.Architectures;

namespace Mila::Dnn
{
    /**
     * @brief The protocol of the chat template that ships with a model of this architecture.
     *
     * Read from what the manifest declares, never from its name or lineage; each architecture decides what its
     * versions mean (Architectures.ixx). A family with one template takes an empty version as that one; Llama has
     * two (3.1 and 3.2), so an empty version is refused.
     *
     * @param architecture The family, as the manifest's `architecture` writes it.
     * @param architecture_version The manifest's `architecture_version`, empty when it declared none.
     * @throws std::invalid_argument for an architecture or version with no chat template here, naming the ones it has.
     */
    export inline std::unique_ptr<Conversation::Protocol> chatProtocolFor(
        std::string_view architecture, std::string_view architecture_version )
    {
        std::unique_ptr<Conversation::Protocol> protocol;

        Architectures::visit( architecture, [&]( auto named )
            {
                protocol = decltype( named )::type::chatProtocol( architecture_version );
            } );

        if ( protocol )
        {
            return protocol;
        }

        std::string templates;

        Architectures::forEach( [&]( auto listed )
            {
                using Architecture = typename decltype( listed )::type;

                for ( const std::string_view version : Architecture::template_versions )
                {
                    templates += std::format( "{}{} {}", templates.empty() ? "" : ", ", Architecture::name, version );
                }
            } );

        throw std::invalid_argument( std::format(
            "no chat template for architecture '{}'{}; there are: {}",
            architecture, architecture_version.empty() ? std::string{} : std::format( " {}", architecture_version ),
            templates ) );
    }
}
