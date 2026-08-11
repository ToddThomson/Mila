/**
 * @file ModelCoordinate.ixx
 * @brief How a hub repository is addressed: [hf:]<owner>/<repository>[@<revision>].
 *
 * This is the *fetch* grammar, not the store's. A model is named by a single flat name; the
 * owner is provenance that a consumer supplies and a user never types. Always compiled, because
 * naming a model is not a network operation. See Specifications/ModelDistribution.md.
 */

module;
#include <optional>
#include <string>
#include <string_view>

export module Distribution.ModelCoordinate;

namespace Mila::Distribution
{
    /**
     * @brief The owner Mila's published models live under.
     *
     * A default a consumer passes to the hub, not something the hub class knows: the hub is
     * "HuggingFace", the owner is Mila's, and baking the second into the first would make the
     * implementation Mila-specific and defeat the interface it sits behind.
     */
    export inline constexpr std::string_view kDefaultHubOwner = "mila-llm";

    /**
     * @brief Where a repository lives on a hub.
     *
     * There is no variant here. One repository is one model at one precision, which the model's
     * name already states -- the platform's own convention, and the reason a listing shows every
     * variant without a manifest fetch.
     */
    export struct ModelCoordinate
    {
        std::string organization;
        std::string repository;

        /// Branch, tag or commit. Defaults to main.
        std::string revision{ "main" };

        std::string toString() const
        {
            std::string text = organization + "/" + repository;

            if ( revision != "main" )
            {
                text += "@" + revision;
            }

            return text;
        }
    };

    /**
     * @brief Parse a coordinate, or return nothing if the spec is not one.
     *
     * Grammar: [hf:]<organization>/<repository>[@<revision>], where organization and repository
     * allow only characters HuggingFace permits in a namespace. Anything path-shaped is rejected
     * on purpose, so a mistyped path reads as a path mistake rather than as a request against a
     * repository that does not exist.
     */
    export std::optional<ModelCoordinate> parseCoordinate( std::string_view spec )
    {
        if ( spec.starts_with( "hf:" ) )
        {
            spec.remove_prefix( 3 );
        }

        if ( spec.empty() )
        {
            return std::nullopt;
        }

        ModelCoordinate coordinate;

        const auto at = spec.rfind( '@' );

        if ( at != std::string_view::npos )
        {
            coordinate.revision = std::string( spec.substr( at + 1 ) );
            spec = spec.substr( 0, at );

            if ( coordinate.revision.empty() )
            {
                return std::nullopt;
            }
        }

        const auto slash = spec.find( '/' );

        // Exactly one separator: a second would be a path, not a coordinate.
        if ( slash == std::string_view::npos || spec.find( '/', slash + 1 ) != std::string_view::npos )
        {
            return std::nullopt;
        }

        coordinate.organization = std::string( spec.substr( 0, slash ) );
        coordinate.repository = std::string( spec.substr( slash + 1 ) );

        if ( coordinate.organization.empty() || coordinate.repository.empty() )
        {
            return std::nullopt;
        }

        const auto isPermitted = []( std::string_view text )
            {
                for ( char character : text )
                {
                    const bool allowed =
                        ( character >= 'A' && character <= 'Z' ) ||
                        ( character >= 'a' && character <= 'z' ) ||
                        ( character >= '0' && character <= '9' ) ||
                        character == '.' || character == '_' || character == '-';

                    if ( !allowed )
                    {
                        return false;
                    }
                }

                return true;
            };

        if ( !isPermitted( coordinate.organization ) || !isPermitted( coordinate.repository ) )
        {
            return std::nullopt;
        }

        return coordinate;
    }
}
