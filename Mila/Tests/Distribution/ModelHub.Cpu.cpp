/**
 * @file ModelHub.Cpu.cpp
 * @brief Parsing a hub listing, pinned against a response recorded from the live API, and the
 * messages the hub gives when HuggingFace refuses a request.
 *
 * The listing body is the real one returned by
 * https://huggingface.co/api/models?author=mila-llm&full=true on 2026-08-01, so a change in
 * the shape Mila depends on fails here rather than at a user's first `/model list --online`.
 *
 * CPU only, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Distribution
{
    using namespace Mila::Distribution;

    namespace
    {
        /// Recorded verbatim from the live endpoint, trimmed only of fields Mila ignores.
        const std::string kRecordedListing = R"([
          {
            "_id": "6a6a85cc1b3eddc53549479e",
            "id": "mila-llm/gemma-4-12b-it",
            "author": "mila-llm",
            "gated": false,
            "lastModified": "2026-07-30T03:14:20.000Z",
            "likes": 1,
            "private": false,
            "sha": "570dbe0e5778c4a1ab96fb8ec2dcc626da828e37",
            "downloads": 0,
            "tags": ["mila", "gemma", "fp4", "quantized", "license:apache-2.0", "region:us"],
            "library_name": "mila",
            "createdAt": "2026-07-29T22:59:24.000Z",
            "modelId": "mila-llm/gemma-4-12b-it",
            "siblings": [
              { "rfilename": ".gitattributes" },
              { "rfilename": "LICENSE" },
              { "rfilename": "README.md" },
              { "rfilename": "gemma4_12b_it_fp4.safetensors" },
              { "rfilename": "gemma_tokenizer.bin" },
              { "rfilename": "mila.json" }
            ]
          }
        ])";
    }

    TEST( HuggingFaceListing, ParsesTheRecordedResponse )
    {
        const auto models = parseHuggingFaceListing( kRecordedListing );

        ASSERT_EQ( models.size(), 1u );

        const HubModel& model = models[ 0 ];

        EXPECT_EQ( model.owner, "mila-llm" );
        EXPECT_EQ( model.repository, "gemma-4-12b-it" );
        EXPECT_EQ( model.coordinate(), "mila-llm/gemma-4-12b-it" );
        EXPECT_EQ( model.library, "mila" );
        EXPECT_EQ( model.last_modified, "2026-07-30T03:14:20.000Z" );
        EXPECT_TRUE( model.hasManifest() );

        // Known before any file is requested, so a gated repository is reported as needing
        // accepted terms rather than discovered as a 403 partway into a transfer.
        EXPECT_FALSE( model.gated );

        // The resolved commit, which is what a store record has to persist.
        EXPECT_EQ( model.revision, "570dbe0e5778c4a1ab96fb8ec2dcc626da828e37" );

        EXPECT_NE( std::find( model.tags.begin(), model.tags.end(), "fp4" ), model.tags.end() );
    }

    TEST( HuggingFaceListing, TreatsAStringGatedFieldAsGated )
    {
        // Public repositories report false; repositories behind terms report "auto" or
        // "manual", not true. Reading that as a boolean would call Llama ungated.
        const auto models = parseHuggingFaceListing( R"([
          { "id": "meta-llama/Llama-3.2-3B-Instruct", "gated": "manual",
            "siblings": [ { "rfilename": "mila.json" } ] }
        ])" );

        ASSERT_EQ( models.size(), 1u );
        EXPECT_TRUE( models[ 0 ].gated );
    }

    TEST( HuggingFaceListing, DropsRepositoriesWithNoManifest )
    {
        // An owner may hold models this runtime cannot load. Listing them as available would
        // be a lie that only surfaces at the pull.
        const auto models = parseHuggingFaceListing( R"([
          { "id": "mila-llm/has-one", "siblings": [ { "rfilename": "mila.json" } ] },
          { "id": "mila-llm/has-none", "siblings": [ { "rfilename": "config.json" } ] },
          { "id": "mila-llm/no-siblings" }
        ])" );

        ASSERT_EQ( models.size(), 1u );
        EXPECT_EQ( models[ 0 ].repository, "has-one" );
    }

    TEST( HuggingFaceListing, SurvivesAMalformedBody )
    {
        // A hub is remote and may answer with anything. An empty listing is a legible outcome;
        // a parse exception on a startup path is not.
        EXPECT_TRUE( parseHuggingFaceListing( "not json at all" ).empty() );
        EXPECT_TRUE( parseHuggingFaceListing( R"({"error":"not found"})" ).empty() );
        EXPECT_TRUE( parseHuggingFaceListing( "[]" ).empty() );
        EXPECT_TRUE( parseHuggingFaceListing( R"([{"id":"no-slash-here"}])" ).empty() );
    }

    namespace
    {
        /// Answers every request with one status, as HuggingFace does for a name it will not
        /// confirm to the caller.
        class FixedStatusTransport : public IHttpTransport
        {
        public:

            explicit FixedStatusTransport( long http_code ) : http_code_( http_code ) {}

            std::string name() const override { return "fixed-status"; }

            HttpResponse fetch(
                const HttpFetch&,
                const SinkCallback&,
                const HeadersCallback& on_headers ) const override
            {
                HttpResponse response;
                response.http_code = http_code_;

                if ( on_headers )
                {
                    on_headers( response.http_code, 0 );
                }

                return response;
            }

        private:

            long http_code_;
        };

        std::string manifestFailure( const std::string& token )
        {
            const HuggingFaceHub hub( std::make_shared<FixedStatusTransport>( 401 ), token );

            try
            {
                hub.fetchManifest( { "mila-llm", "gemma-4-12b-itt" } );
            } catch ( const std::runtime_error& error ) {
                return error.what();
            }

            return {};
        }
    }

    TEST( HuggingFaceHubErrors, AnAnonymousUnauthorizedLeadsWithTheModelName )
    {
        // HuggingFace hides whether a repository exists from a caller with no token, so a
        // mistyped name arrives as a 401. Sending that user off to get a token is the
        // failure this pins.
        const std::string message = manifestFailure( "" );

        EXPECT_NE( message.find( "mila-llm/gemma-4-12b-itt" ), std::string::npos ) << message;
        EXPECT_NE( message.find( "not found" ), std::string::npos ) << message;
        EXPECT_EQ( message.find( "no valid HuggingFace token" ), std::string::npos ) << message;
    }

    TEST( HuggingFaceHubErrors, AnUnauthorizedTokenIsReportedAsTheToken )
    {
        // An authenticated caller gets a 404 for a missing name, so a 401 here really is the
        // token.
        const std::string message = manifestFailure( "hf_not_a_real_token" );

        EXPECT_NE( message.find( "no valid HuggingFace token" ), std::string::npos ) << message;
        EXPECT_EQ( message.find( "not found" ), std::string::npos ) << message;
    }
}
