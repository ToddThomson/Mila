/**
 * @file HuggingFaceHub.ixx
 * @brief The HuggingFace implementation of IModelHub, and its token discovery.
 *
 * The only place in the library that names a huggingface.co URL, and **always compiled**: URL
 * shapes, the token file, the listing quirks and the meaning of a 403 are knowledge, not
 * transport. It performs no I/O of its own -- an IHttpTransport does that -- so a build with
 * no libcurl still knows everything about HuggingFace and needs only bytes from elsewhere.
 * See Specifications/ModelDistribution.md.
 */

module;
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <ios>
// Required by every importer of nlohmann.json -- see Distribution/ModelStore.ixx.
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

export module Distribution.HuggingFaceHub;

import nlohmann.json;
import Distribution.Environment;
import Distribution.ModelCoordinate;
import Distribution.ModelStore;
import Distribution.ModelHub;
import Distribution.HttpTransport;
import Distribution.HttpClient;
import Distribution.HttpTransportBackend;

namespace Mila::Distribution
{
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
     * The only class in the library that names a huggingface.co URL. It composes an HttpClient
     * rather than a transport: redirects, the cross-host token rule and the Range protocol are
     * the client's business, and this class should no more reimplement them than it should
     * open a socket.
     */
    export class HuggingFaceHub : public IModelHub
    {
    public:

        explicit HuggingFaceHub(
            std::shared_ptr<const IHttpTransport> transport,
            std::string token = discoverHuggingFaceToken(),
            ProgressCallback progress = {} )
            : client_( std::move( transport ) ),
              token_( std::move( token ) ),
              progress_( std::move( progress ) )
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

            const HttpResult result = client_.get( request, sink, progress_ );

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
            const HttpResult result = client_.getString( request, body );

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

        HttpClient client_;
        std::string token_;
        ProgressCallback progress_;
    };

    /**
     * @brief The hub a consumer gets when it does not care how the bytes arrive.
     *
     * HuggingFace over whichever transport this build was compiled with. In a build without
     * libcurl the hub still knows every URL and rule -- what it gets back is a transport that
     * refuses, so the failure names the missing dependency rather than a network error.
     */
    export std::unique_ptr<IModelHub> makeDefaultModelHub( ProgressCallback progress = {} )
    {
        return std::make_unique<HuggingFaceHub>(
            makeDefaultHttpTransport(), discoverHuggingFaceToken(), std::move( progress ) );
    }
}
