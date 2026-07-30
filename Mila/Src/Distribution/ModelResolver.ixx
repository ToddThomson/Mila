/**
 * @file ModelResolver.ixx
 * @brief Turns a model spec -- a coordinate or a local path -- into files on disk.
 *
 * Remote access is injected, so coordinate parsing, manifest validation and version-skew
 * refusal are all testable without a network. The libcurl wiring lives in one factory at the
 * edge. See Specifications/ModelDistribution.md.
 */

module;
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

export module Distribution.ModelResolver;

import nlohmann.json;
import Distribution.ModelCache;
import Distribution.HttpClient;
import Mila.Version;

namespace Mila::Distribution
{
    /**
     * @brief A published model, named as <organization>/<repository>:<variant>[@revision].
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
     * @brief Everything a loader needs after resolution.
     */
    export struct ResolvedModel
    {
        std::filesystem::path weights_path;

        /// Empty when the model carries no tokenizer, or when resolved from a bare local path.
        std::filesystem::path tokenizer_path;

        /// From the manifest; empty for a local path, whose metadata lives inside the file.
        std::string architecture;
        std::string weight_quantization;
        std::string variant;

        /// True when the spec named a file directly and no network or cache was involved.
        bool from_local_path{ false };
    };

    /**
     * @brief The two remote operations resolution needs.
     *
     * Injected rather than called directly so the resolver's logic can be tested offline.
     */
    export struct RemoteAccess
    {
        BlobFetcher fetch_blob;
        std::function<std::string( const std::string& url )> fetch_text;
    };

    /**
     * @brief Parse a coordinate, or return nothing if the spec is not one.
     *
     * Grammar: [hf:]<organization>/<repository>[:<variant>][@<revision>], where organization
     * and repository allow only characters HuggingFace permits in a namespace.
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

    /**
     * @brief HuggingFace token, or empty for anonymous access.
     *
     * First match wins: MILA_HF_TOKEN, HF_TOKEN, then the file huggingface-cli login writes.
     * Gemma 4 is Apache 2.0 and needs none of this; Llama 3.1 and 3.2 are gated and do.
     */
    export std::string discoverHuggingFaceToken()
    {
        for ( const char* name : { "MILA_HF_TOKEN", "HF_TOKEN" } )
        {
            if ( const char* value = std::getenv( name ) )
            {
                if ( *value != '\0' )
                {
                    return value;
                }
            }
        }

        std::filesystem::path home;

        if ( const char* user_profile = std::getenv( "USERPROFILE" ) )
        {
            home = user_profile;
        }
        else if ( const char* home_variable = std::getenv( "HOME" ) )
        {
            home = home_variable;
        }

        if ( home.empty() )
        {
            return {};
        }

        const auto token_path = home / ".cache" / "huggingface" / "token";

        std::unique_ptr<std::FILE, int( * )( std::FILE* )> file(
            std::fopen( token_path.string().c_str(), "rb" ), &std::fclose );

        if ( file == nullptr )
        {
            return {};
        }

        std::string token;
        char buffer[ 512 ];
        size_t read = 0;

        while ( ( read = std::fread( buffer, 1, sizeof( buffer ), file.get() ) ) > 0 )
        {
            token.append( buffer, read );
        }

        while ( !token.empty() &&
            ( token.back() == '\n' || token.back() == '\r' || token.back() == ' ' ) )
        {
            token.pop_back();
        }

        return token;
    }

    /**
     * @brief Wire libcurl into a RemoteAccess.
     *
     * The only place the HTTP client meets the resolver. Failure messages are mapped here so
     * a 401 and a 403 stay distinguishable -- one means "get a token", the other means "accept
     * the terms", and conflating them wastes an afternoon.
     */
    export RemoteAccess makeHuggingFaceRemoteAccess(
        std::string token = discoverHuggingFaceToken(),
        ProgressCallback progress = {} )
    {
        RemoteAccess access;

        access.fetch_blob = [token, progress](
            const std::string& url, uint64_t resume_from,
            const std::function<bool( const char*, size_t )>& sink ) -> FetchReport
            {
                HttpRequest request;
                request.url = url;
                request.token = token;
                request.resume_from = resume_from;

                const HttpResult result = httpGet( request, sink, progress );

                if ( result.status == HttpStatus::Ok )
                {
                    return { FetchOutcome::Complete, {} };
                }

                if ( result.status == HttpStatus::RangeIgnored )
                {
                    return { FetchOutcome::RangeIgnored, result.message };
                }

                return { FetchOutcome::Failed,
                    std::format( "{} ({})", result.message, toString( result.status ) ) };
            };

        access.fetch_text = [token]( const std::string& url ) -> std::string
            {
                HttpRequest request;
                request.url = url;
                request.token = token;

                std::string body;
                const HttpResult result = httpGetString( request, body );

                if ( result.ok() )
                {
                    return body;
                }

                if ( result.status == HttpStatus::Unauthorized )
                {
                    throw std::runtime_error( std::format(
                        "{}: no valid HuggingFace token. Set HF_TOKEN, or run "
                        "'huggingface-cli login'.", url ) );
                }

                if ( result.status == HttpStatus::Forbidden )
                {
                    throw std::runtime_error( std::format(
                        "{}: the token is valid but this repository's terms have not been "
                        "accepted. Open the model page on huggingface.co and accept them.",
                        url ) );
                }

                // The final URL, not the requested one: after a redirect they differ, and
                // naming the requested one hides which hop actually failed.
                const std::string& failed_at =
                    result.final_url.empty() ? url : result.final_url;

                throw std::runtime_error( std::format( "{}: {}", failed_at, result.message ) );
            };

        return access;
    }

    /**
     * @brief Resolves a model spec to files on disk, fetching what is missing.
     */
    export class ModelResolver
    {
    public:

        ModelResolver( ModelCache& cache, RemoteAccess access )
            : cache_( cache ), access_( std::move( access ) )
        {}

        /**
         * @brief Resolve a spec that is either a local path or a coordinate.
         *
         * A path is returned as-is with no network and no cache: a user who already has the
         * file must not be made to re-download it, which is the failure mode ollama's
         * content-addressed store has.
         *
         * @throws std::runtime_error if the spec is neither, if the manifest is malformed, or
         *         if the artifact requires a newer Mila.
         */
        ResolvedModel resolve( const std::string& spec )
        {
            std::error_code ignored;

            const bool forced_coordinate = spec.starts_with( "hf:" );

            if ( !forced_coordinate && std::filesystem::exists( spec, ignored ) )
            {
                ResolvedModel resolved;
                resolved.weights_path = spec;
                resolved.from_local_path = true;

                return resolved;
            }

            const auto coordinate = parseCoordinate( spec );

            if ( !coordinate.has_value() )
            {
                throw std::runtime_error( std::format(
                    "Model spec '{}' is neither an existing file nor a coordinate of the form "
                    "<organization>/<repository>[:<variant>][@<revision>]", spec ) );
            }

            return resolveCoordinate( *coordinate );
        }

        /**
         * @brief Resolve a parsed coordinate.
         */
        ResolvedModel resolveCoordinate( const ModelCoordinate& coordinate )
        {
            const std::string manifest_text = access_.fetch_text( manifestUrl( coordinate ) );

            nlohmann::json manifest = nlohmann::json::parse( manifest_text, nullptr, false );

            if ( manifest.is_discarded() || !manifest.is_object() )
            {
                throw std::runtime_error( std::format(
                    "{}: mila.json is not a JSON object", coordinate.toString() ) );
            }

            const std::string variant_name = selectVariant( manifest, coordinate );
            const nlohmann::json& variant = manifest[ "variants" ][ variant_name ];

            requireCompatibleVersion( coordinate, variant_name, variant );

            ResolvedModel resolved;
            resolved.variant = variant_name;
            resolved.architecture = manifest.value( "architecture", std::string{} );
            resolved.weight_quantization = variant.value( "weight_quantization", std::string{} );

            if ( !variant.contains( "files" ) || !variant[ "files" ].is_object() )
            {
                throw std::runtime_error( std::format(
                    "{}: variant '{}' declares no files", coordinate.toString(), variant_name ) );
            }

            const nlohmann::json& files = variant[ "files" ];

            resolved.weights_path = fetchDeclaredFile( coordinate, files, "weights", true );
            resolved.tokenizer_path = fetchDeclaredFile( coordinate, files, "tokenizer", false );

            return resolved;
        }

    private:

        static std::string manifestUrl( const ModelCoordinate& coordinate )
        {
            return std::format( "https://huggingface.co/{}/{}/resolve/{}/mila.json",
                coordinate.organization, coordinate.repository, coordinate.revision );
        }

        static std::string fileUrl( const ModelCoordinate& coordinate, const std::string& path )
        {
            return std::format( "https://huggingface.co/{}/{}/resolve/{}/{}",
                coordinate.organization, coordinate.repository, coordinate.revision, path );
        }

        static std::string selectVariant(
            const nlohmann::json& manifest, const ModelCoordinate& coordinate )
        {
            if ( !manifest.contains( "variants" ) || !manifest[ "variants" ].is_object() )
            {
                throw std::runtime_error( std::format(
                    "{}: mila.json declares no variants", coordinate.toString() ) );
            }

            const std::string requested = coordinate.variant.empty()
                ? manifest.value( "default_variant", std::string{} )
                : coordinate.variant;

            if ( requested.empty() )
            {
                throw std::runtime_error( std::format(
                    "{}: no variant given and mila.json declares no default_variant",
                    coordinate.toString() ) );
            }

            if ( !manifest[ "variants" ].contains( requested ) )
            {
                std::string available;

                for ( const auto& entry : manifest[ "variants" ].items() )
                {
                    available += available.empty() ? entry.key() : ", " + entry.key();
                }

                throw std::runtime_error( std::format(
                    "{}: no variant '{}'. Available: {}",
                    coordinate.toString(), requested, available ) );
            }

            return requested;
        }

        /**
         * @brief Refuse an artifact that needs a newer Mila.
         *
         * A version comparison here beats a parse error somewhere inside the tensor index,
         * which is what a format change would otherwise look like.
         */
        static void requireCompatibleVersion(
            const ModelCoordinate& coordinate,
            const std::string& variant_name,
            const nlohmann::json& variant )
        {
            const std::string minimum = variant.value( "minimum_mila_version", std::string{} );

            if ( minimum.empty() )
            {
                return;
            }

            int major = 0;
            int minor = 0;
            int patch = 0;

            if ( std::sscanf( minimum.c_str(), "%d.%d.%d", &major, &minor, &patch ) < 2 )
            {
                throw std::runtime_error( std::format(
                    "{}: variant '{}' has an unparseable minimum_mila_version '{}'",
                    coordinate.toString(), variant_name, minimum ) );
            }

            Version current = getAPIVersion();

            const long long required_ordinal =
                ( static_cast<long long>( major ) * 1000000 ) + ( minor * 1000 ) + patch;
            const long long current_ordinal =
                ( static_cast<long long>( current.getMajor() ) * 1000000 )
                + ( current.getMinor() * 1000 ) + current.getPatch();

            if ( current_ordinal < required_ordinal )
            {
                throw std::runtime_error( std::format(
                    "{}: variant '{}' requires Mila {} or newer; this build is {}",
                    coordinate.toString(), variant_name, minimum, current.toString() ) );
            }
        }

        std::filesystem::path fetchDeclaredFile(
            const ModelCoordinate& coordinate,
            const nlohmann::json& files,
            const std::string& role,
            bool required )
        {
            if ( !files.contains( role ) )
            {
                if ( required )
                {
                    throw std::runtime_error( std::format(
                        "{}: manifest declares no '{}' file", coordinate.toString(), role ) );
                }

                return {};
            }

            const nlohmann::json& entry = files[ role ];

            if ( !entry.is_object() || !entry.contains( "path" ) || !entry.contains( "sha256" ) )
            {
                throw std::runtime_error( std::format(
                    "{}: '{}' entry needs both a path and a sha256",
                    coordinate.toString(), role ) );
            }

            const std::string path = entry[ "path" ].get<std::string>();
            const std::string digest = entry[ "sha256" ].get<std::string>();

            return cache_.ensureBlob( fileUrl( coordinate, path ), digest, access_.fetch_blob );
        }

        ModelCache& cache_;
        RemoteAccess access_;
    };
}
