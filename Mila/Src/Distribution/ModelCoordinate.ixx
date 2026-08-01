/**
 * @file ModelCoordinate.ixx
 * @brief How a model is named: [<hub>:]<owner>/<repository>[:<variant>][@<revision>].
 *
 * One naming scheme for every model, whether it came from a hub or was built here. Always
 * compiled, because naming and locating a model are not network operations.
 * See Specifications/ModelDistribution.md.
 */

module;
#include <optional>
#include <string>
#include <string_view>

export module Distribution.ModelCoordinate;

namespace Mila::Distribution
{
    /// The reserved owner for a model converted, trained or exported on this machine.
    export inline constexpr std::string_view kLocalOwner = "local";

    /**
     * @brief A model's name.
     *
     * Variant is separate from repository because variants share components: the FP4, FP8 and
     * BF16 builds of one model share a tokenizer, which a flat repository-variant name hides.
     */
    export struct ModelCoordinate
    {
        std::string organization;
        std::string repository;

        /// Empty means the manifest's declared default.
        std::string variant;

        /// Branch, tag or commit. Defaults to main.
        std::string revision{ "main" };

        /// True for the reserved `local` owner, which no hub serves.
        bool isLocal() const
        {
            return organization == kLocalOwner;
        }

        std::string toString() const
        {
            std::string text = organization + "/" + repository;

            if ( !variant.empty() )
            {
                text += ":" + variant;
            }

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
     * Grammar: [hf:]<organization>/<repository>[:<variant>][@<revision>], where organization
     * and repository allow only characters HuggingFace permits in a namespace. Anything
     * path-shaped is rejected on purpose, so a mistyped path reads as a path mistake rather
     * than as a request against a repository that does not exist.
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

        const auto colon = spec.rfind( ':' );

        if ( colon != std::string_view::npos )
        {
            coordinate.variant = std::string( spec.substr( colon + 1 ) );
            spec = spec.substr( 0, colon );

            if ( coordinate.variant.empty() )
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

        if ( !isPermitted( coordinate.organization ) || !isPermitted( coordinate.repository )
            || !isPermitted( coordinate.variant ) )
        {
            return std::nullopt;
        }

        return coordinate;
    }
}
