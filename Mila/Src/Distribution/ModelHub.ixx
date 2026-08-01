/**
 * @file ModelHub.ixx
 * @brief The abstraction over a non-local model hub, and its HuggingFace implementation.
 *
 * Everything that knows a URL shape, an authentication scheme or a listing API lives behind
 * IModelHub, so the store, the digest check and the resume protocol never learn one. See
 * Specifications/ModelDistribution.md.
 */

module;
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

export module Distribution.ModelHub;

import nlohmann.json;
import Distribution.Environment;
import Distribution.ModelCoordinate;
import Distribution.ModelStore;
import Distribution.HttpClient;

namespace Mila::Distribution
{
    /**
     * @brief One repository as a hub reports it, before any manifest is fetched.
     *
     * Everything here comes from the hub's own listing API. Variants are deliberately absent:
     * only mila.json knows them, and knowing that costs a further request per repository.
     */
    export struct HubModel
    {
        std::string owner;
        std::string repository;

        /// True when the terms must be accepted before the files can be fetched.
        bool gated{ false };

        /// The resolved commit of the default branch.
        std::string revision;

        std::string last_modified;

        /// The hub's library attribution. "mila" for a model this runtime can load.
        std::string library;

        std::vector<std::string> tags;
        std::vector<std::string> files;

        /// A repository is loadable by Mila only if it publishes a manifest.
        bool hasManifest() const
        {
            for ( const auto& file : files )
            {
                if ( file == "mila.json" )
                {
                    return true;
                }
            }

            return false;
        }

        std::string coordinate() const
        {
            return owner + "/" + repository;
        }
    };

    /**
     * @brief A remote that serves manifests and files.
     *
     * Narrow on purpose. What varies between hubs is URL construction, authentication and the
     * listing API; what does not vary is the manifest schema, the digest check, the blob store
     * and the resume protocol. fetchFile takes a resume offset rather than a URL so that a hub
     * which is not HTTP can still satisfy it.
     */
    export class IModelHub
    {
    public:

        virtual ~IModelHub() = default;

        /// Stable identifier recorded in a store record, e.g. "huggingface".
        virtual std::string name() const = 0;

        /**
         * @brief Every repository an owner publishes.
         *
         * The result is untrusted remote text authored by whoever owns the repository. It is
         * data: rendered, never interpreted as markup and never as instructions.
         */
        virtual std::vector<HubModel> listModels( const std::string& owner ) const = 0;

        /// The raw mila.json text for a coordinate.
        virtual std::string fetchManifest( const ModelCoordinate& coordinate ) const = 0;

        /// Stream one repository file, resuming from an offset.
        virtual FetchReport fetchFile(
            const ModelCoordinate& coordinate,
            const std::string& path,
            uint64_t resume_from,
            const std::function<bool( const char*, size_t )>& sink ) const = 0;
    };

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
            if ( const auto value = readEnvironmentVariable( name ) )
            {
                return *value;
            }
        }

        std::filesystem::path home;

        if ( const auto user_profile = readEnvironmentVariable( "USERPROFILE" ) )
        {
            home = *user_profile;
        }
        else if ( const auto home_variable = readEnvironmentVariable( "HOME" ) )
        {
            home = *home_variable;
        }

        if ( home.empty() )
        {
            return {};
        }

        std::ifstream file( home / ".cache" / "huggingface" / "token", std::ios::binary );

        if ( !file.is_open() )
        {
            return {};
        }

        std::string token;
        char buffer[ 512 ];

        for ( ;; )
        {
            file.read( buffer, sizeof( buffer ) );

            const auto read = static_cast<size_t>( file.gcount() );

            if ( read == 0 )
            {
                break;
            }

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
     * @brief Parse a HuggingFace models listing.
     *
     * Exposed so the shape can be tested against a recorded response rather than a live one.
     * Repositories that publish no mila.json are dropped: an owner may hold models this
     * runtime cannot load, and reporting them as available would be a lie.
     */
    export std::vector<HubModel> parseHuggingFaceListing( const std::string& body )
    {
        nlohmann::json json = nlohmann::json::parse( body, nullptr, false );

        std::vector<HubModel> models;

        if ( json.is_discarded() || !json.is_array() )
        {
            return models;
        }

        for ( const auto& entry : json )
        {
            if ( !entry.is_object() )
            {
                continue;
            }

            const std::string id = entry.value( "id", std::string{} );
            const auto slash = id.find( '/' );

            if ( slash == std::string::npos )
            {
                continue;
            }

            HubModel model;
            model.owner = id.substr( 0, slash );
            model.repository = id.substr( slash + 1 );
            model.revision = entry.value( "sha", std::string{} );
            model.last_modified = entry.value( "lastModified", std::string{} );
            model.library = entry.value( "library_name", std::string{} );

            // Absent on a public repository, and a string ("auto"/"manual") rather than a bool
            // when the repository is behind terms.
            if ( entry.contains( "gated" ) )
            {
                const auto& gated = entry[ "gated" ];

                model.gated = gated.is_boolean() ? gated.get<bool>() : !gated.is_null();
            }

            if ( entry.contains( "tags" ) && entry[ "tags" ].is_array() )
            {
                for ( const auto& tag : entry[ "tags" ] )
                {
                    if ( tag.is_string() )
                    {
                        model.tags.push_back( tag.get<std::string>() );
                    }
                }
            }

            if ( entry.contains( "siblings" ) && entry[ "siblings" ].is_array() )
            {
                for ( const auto& sibling : entry[ "siblings" ] )
                {
                    if ( sibling.is_object() && sibling.contains( "rfilename" ) )
                    {
                        model.files.push_back( sibling[ "rfilename" ].get<std::string>() );
                    }
                }
            }

            if ( !model.hasManifest() )
            {
                continue;
            }

            models.push_back( std::move( model ) );
        }

        return models;
    }

    /**
     * @brief The HuggingFace hub.
     *
     * The only class in the library that names a huggingface.co URL.
     */
    export class HuggingFaceHub : public IModelHub
    {
    public:

        explicit HuggingFaceHub(
            std::string token = discoverHuggingFaceToken(),
            ProgressCallback progress = {} )
            : token_( std::move( token ) ), progress_( std::move( progress ) )
        {}

        std::string name() const override { return "huggingface"; }

        std::vector<HubModel> listModels( const std::string& owner ) const override
        {
            // full=true is what makes one request enough: it adds gated, sha and siblings, so a
            // complete listing needs no per-repository follow-up.
            return parseHuggingFaceListing( fetchText( std::format(
                "https://huggingface.co/api/models?author={}&full=true", owner ) ) );
        }

        std::string fetchManifest( const ModelCoordinate& coordinate ) const override
        {
            return fetchText( fileUrl( coordinate, "mila.json" ) );
        }

        FetchReport fetchFile(
            const ModelCoordinate& coordinate,
            const std::string& path,
            uint64_t resume_from,
            const std::function<bool( const char*, size_t )>& sink ) const override
        {
            HttpRequest request;
            request.url = fileUrl( coordinate, path );
            request.token = token_;
            request.resume_from = resume_from;

            const HttpResult result = httpGet( request, sink, progress_ );

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
        }

    private:

        static std::string fileUrl( const ModelCoordinate& coordinate, const std::string& path )
        {
            return std::format( "https://huggingface.co/{}/{}/resolve/{}/{}",
                coordinate.organization, coordinate.repository, coordinate.revision, path );
        }

        /**
         * @brief GET a small document, mapping the two authentication failures apart.
         *
         * A 401 and a 403 need different messages -- one means "get a token", the other means
         * "accept the terms" -- and conflating them wastes an afternoon.
         */
        std::string fetchText( const std::string& url ) const
        {
            HttpRequest request;
            request.url = url;
            request.token = token_;

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
                    "accepted. Open the model page on huggingface.co and accept them.", url ) );
            }

            // The final URL, not the requested one: after a redirect they differ, and naming
            // the requested one hides which hop actually failed.
            const std::string& failed_at = result.final_url.empty() ? url : result.final_url;

            throw std::runtime_error( std::format( "{}: {}", failed_at, result.message ) );
        }

        std::string token_;
        ProgressCallback progress_;
    };
}
