/**
 * @file HttpClient.Cpu.cpp
 * @brief The HTTP client's policy, tested offline: redirects, the token rule, Range, status.
 *
 * These exist because the omission cost a live debugging round. Phase 1 shipped with no
 * tests on the argument that the client's contract was about live behaviour; the very first
 * real request then failed on relative-Location handling, which is pure string work and
 * needed no server at all.
 *
 * The policy cases below became possible only when the transport was split out
 * (Specifications/HttpClient.md): the hop limit, the cross-host token drop and the
 * 200-versus-206 rule used to live inside the libcurl implementation, where reaching them
 * meant a live server.
 *
 * CPU only, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

import Mila;

namespace Mila::Tests::Distribution
{
    using namespace Mila::Distribution;

    TEST( RedirectResolution, KeepsAnAbsoluteLocationUnchanged )
    {
        EXPECT_EQ(
            resolveRedirect( "https://huggingface.co/a/b/resolve/main/f.json",
                "https://cdn-lfs.hf.co/repos/xyz/f.json?token=abc" ),
            "https://cdn-lfs.hf.co/repos/xyz/f.json?token=abc" );
    }

    TEST( RedirectResolution, ResolvesTheRootRelativeFormHuggingFaceActuallySends )
    {
        // Verbatim shape of the 307 that broke the first live request: HuggingFace answers
        // a manifest fetch with a path, not a URL.
        const std::string base =
            "https://huggingface.co/mila-llm/gemma-4-12b-it/resolve/main/mila.json";
        const std::string location =
            "/api/resolve-cache/models/mila-llm/gemma-4-12b-it/2d88819/mila.json?etag=%22abc%22";

        EXPECT_EQ( resolveRedirect( base, location ),
            "https://huggingface.co"
            "/api/resolve-cache/models/mila-llm/gemma-4-12b-it/2d88819/mila.json?etag=%22abc%22" );
    }

    TEST( RedirectResolution, ResolvesAProtocolRelativeLocation )
    {
        EXPECT_EQ(
            resolveRedirect( "https://huggingface.co/a/b", "//cdn.example.com/x/y" ),
            "https://cdn.example.com/x/y" );
    }

    TEST( RedirectResolution, ResolvesAPathRelativeLocationAgainstTheCurrentDirectory )
    {
        EXPECT_EQ(
            resolveRedirect( "https://example.com/a/b/c.json", "d.json" ),
            "https://example.com/a/b/d.json" );

        // A query on the base must not leak into the resolved path.
        EXPECT_EQ(
            resolveRedirect( "https://example.com/a/b/c.json?x=1", "d.json" ),
            "https://example.com/a/b/d.json" );
    }

    TEST( RedirectResolution, HandlesABaseWithNoPath )
    {
        EXPECT_EQ( resolveRedirect( "https://example.com", "/x" ), "https://example.com/x" );
    }

    TEST( RedirectResolution, PreservesTheSchemeAcrossHosts )
    {
        // The host changes here, which is what drops the authorization header. The scheme
        // must survive so the next hop is still TLS.
        const std::string resolved =
            resolveRedirect( "https://huggingface.co/a", "//cdn-lfs.hf.co/b" );

        EXPECT_TRUE( resolved.starts_with( "https://" ) );
    }

    // ------------------------------------------------------------------------
    // HttpClient policy, against a scripted transport
    //
    // None of this could be tested before: the hop limit, the token rule and the Range
    // protocol lived inside the libcurl implementation and needed a live server to reach.
    // Splitting the transport out is what made them ordinary unit tests, and they compile
    // in every build rather than only where libcurl does.
    // ------------------------------------------------------------------------

    namespace
    {
        struct ScriptedResponse
        {
            long http_code{ 200 };
            std::string location;
            std::string body;
            bool transport_failed{ false };
        };

        /// Answers from a script and records exactly what it was asked to send.
        class ScriptedTransport : public IHttpTransport
        {
        public:

            std::string name() const override { return "scripted"; }

            HttpResponse fetch(
                const HttpFetch& request,
                const SinkCallback& sink,
                const HeadersCallback& on_headers ) const override
            {
                seen.push_back( request );

                const auto entry = script.find( request.url );

                const ScriptedResponse scripted = ( entry == script.end() )
                    ? ScriptedResponse{ 404, {}, {}, false }
                    : entry->second;

                HttpResponse response;
                response.http_code = scripted.http_code;
                response.location = scripted.location;
                response.content_length = scripted.body.size();
                response.transport_failed = scripted.transport_failed;

                if ( on_headers )
                {
                    on_headers( response.http_code, response.content_length );
                }

                // The contract every transport owes: a non-2xx body never reaches the sink.
                const bool successful =
                    scripted.http_code >= 200 && scripted.http_code < 300;

                if ( successful && !scripted.body.empty() )
                {
                    sink( scripted.body.data(), scripted.body.size() );
                }

                return response;
            }

            std::map<std::string, ScriptedResponse> script;
            mutable std::vector<HttpFetch> seen;
        };

        std::string headerValue( const HttpFetch& request, const std::string& name )
        {
            for ( const auto& header : request.headers )
            {
                if ( header.name == name )
                {
                    return header.value;
                }
            }

            return {};
        }

        HttpClient clientOver( std::shared_ptr<ScriptedTransport> transport )
        {
            return HttpClient( std::move( transport ) );
        }
    }

    TEST( HttpClientPolicy, DropsTheTokenWhenTheHostChanges )
    {
        // The rule this whole split exists to write once: HuggingFace redirects an LFS file
        // to a pre-signed CDN URL that carries its own authorization. Forwarding the bearer
        // token there hands it to whoever operates that host.
        auto transport = std::make_shared<ScriptedTransport>();

        transport->script[ "https://huggingface.co/repo/file" ] =
            { 302, "https://cdn-lfs.hf.co/blob?sig=xyz", {}, false };
        transport->script[ "https://cdn-lfs.hf.co/blob?sig=xyz" ] =
            { 200, {}, "PAYLOAD", false };

        HttpRequest request;
        request.url = "https://huggingface.co/repo/file";
        request.token = "hf_secret";

        std::string body;
        const HttpResult result = clientOver( transport ).getString( request, body );

        ASSERT_TRUE( result.ok() );
        EXPECT_EQ( body, "PAYLOAD" );

        ASSERT_EQ( transport->seen.size(), 2u );
        EXPECT_EQ( headerValue( transport->seen[ 0 ], "Authorization" ), "Bearer hf_secret" );
        EXPECT_EQ( headerValue( transport->seen[ 1 ], "Authorization" ), "" )
            << "the bearer token was forwarded across a change of host";
    }

    TEST( HttpClientPolicy, KeepsTheTokenOnASameHostRedirect )
    {
        auto transport = std::make_shared<ScriptedTransport>();

        transport->script[ "https://huggingface.co/a" ] = { 307, "/b", {}, false };
        transport->script[ "https://huggingface.co/b" ] = { 200, {}, "OK", false };

        HttpRequest request;
        request.url = "https://huggingface.co/a";
        request.token = "hf_secret";

        std::string body;
        ASSERT_TRUE( clientOver( transport ).getString( request, body ).ok() );

        ASSERT_EQ( transport->seen.size(), 2u );
        EXPECT_EQ( headerValue( transport->seen[ 1 ], "Authorization" ), "Bearer hf_secret" );
    }

    TEST( HttpClientPolicy, ReportsRangeIgnoredWhenAResumeDraws200 )
    {
        // The failure this catches is silent: appending a whole-file 200 onto an existing
        // partial concatenates, and only the digest check downstream would notice.
        auto transport = std::make_shared<ScriptedTransport>();

        transport->script[ "https://example.com/blob" ] = { 200, {}, "WHOLE FILE", false };

        HttpRequest request;
        request.url = "https://example.com/blob";
        request.resume_from = 512;

        std::string body;
        const HttpResult result = clientOver( transport ).getString( request, body );

        EXPECT_EQ( result.status, HttpStatus::RangeIgnored );

        ASSERT_EQ( transport->seen.size(), 1u );
        EXPECT_EQ( headerValue( transport->seen[ 0 ], "Range" ), "bytes=512-" );
    }

    TEST( HttpClientPolicy, Accepts206ForAResume )
    {
        auto transport = std::make_shared<ScriptedTransport>();

        transport->script[ "https://example.com/blob" ] = { 206, {}, "TAIL", false };

        HttpRequest request;
        request.url = "https://example.com/blob";
        request.resume_from = 4;

        std::string body;
        const HttpResult result = clientOver( transport ).getString( request, body );

        EXPECT_TRUE( result.ok() );
        EXPECT_EQ( body, "TAIL" );
    }

    TEST( HttpClientPolicy, SendsNoRangeHeaderWithoutAResume )
    {
        auto transport = std::make_shared<ScriptedTransport>();

        transport->script[ "https://example.com/blob" ] = { 200, {}, "ALL", false };

        HttpRequest request;
        request.url = "https://example.com/blob";

        std::string body;
        ASSERT_TRUE( clientOver( transport ).getString( request, body ).ok() );

        ASSERT_EQ( transport->seen.size(), 1u );
        EXPECT_EQ( headerValue( transport->seen[ 0 ], "Range" ), "" );
    }

    TEST( HttpClientPolicy, StopsAfterTheHopLimit )
    {
        auto transport = std::make_shared<ScriptedTransport>();

        // A self-referential redirect: without a limit this never returns.
        transport->script[ "https://example.com/loop" ] =
            { 302, "https://example.com/loop", {}, false };

        HttpRequest request;
        request.url = "https://example.com/loop";
        request.maximum_redirects = 3;

        std::string body;
        const HttpResult result = clientOver( transport ).getString( request, body );

        EXPECT_EQ( result.status, HttpStatus::TransportError );
        EXPECT_NE( result.message.find( "redirects" ), std::string::npos );
        EXPECT_EQ( transport->seen.size(), 4u ) << "one initial request plus three hops";
    }

    TEST( HttpClientPolicy, RefusesARedirectWithNoLocation )
    {
        auto transport = std::make_shared<ScriptedTransport>();

        transport->script[ "https://example.com/x" ] = { 302, {}, {}, false };

        HttpRequest request;
        request.url = "https://example.com/x";

        std::string body;
        const HttpResult result = clientOver( transport ).getString( request, body );

        EXPECT_EQ( result.status, HttpStatus::TransportError );
        EXPECT_NE( result.message.find( "Location" ), std::string::npos );
    }

    TEST( HttpClientPolicy, KeepsUnauthorizedAndForbiddenApart )
    {
        // One means "get a token", the other means "accept the terms". Conflating them
        // wastes an afternoon.
        auto transport = std::make_shared<ScriptedTransport>();

        transport->script[ "https://example.com/a" ] = { 401, {}, {}, false };
        transport->script[ "https://example.com/b" ] = { 403, {}, {}, false };

        HttpClient client = clientOver( transport );

        std::string body;

        HttpRequest unauthorized;
        unauthorized.url = "https://example.com/a";
        EXPECT_EQ( client.getString( unauthorized, body ).status, HttpStatus::Unauthorized );

        HttpRequest forbidden;
        forbidden.url = "https://example.com/b";
        EXPECT_EQ( client.getString( forbidden, body ).status, HttpStatus::Forbidden );
    }

    TEST( HttpClientPolicy, ReportsTheFinalUrlRatherThanTheRequestedOne )
    {
        // After a redirect the two differ, and naming the requested one hides which hop failed.
        auto transport = std::make_shared<ScriptedTransport>();

        transport->script[ "https://example.com/start" ] =
            { 302, "https://elsewhere.example/end", {}, false };
        transport->script[ "https://elsewhere.example/end" ] = { 500, {}, {}, false };

        HttpRequest request;
        request.url = "https://example.com/start";

        std::string body;
        const HttpResult result = clientOver( transport ).getString( request, body );

        EXPECT_EQ( result.status, HttpStatus::ServerError );
        EXPECT_EQ( result.final_url, "https://elsewhere.example/end" );
    }

    TEST( HttpClientPolicy, ReportsProgressAgainstTheWholeFileWhenResuming )
    {
        // content-length covers only the remaining bytes on a resumed transfer, so a total
        // taken straight from it would report a file smaller than what is already on disk.
        auto transport = std::make_shared<ScriptedTransport>();

        transport->script[ "https://example.com/blob" ] = { 206, {}, "9876543210", false };

        HttpRequest request;
        request.url = "https://example.com/blob";
        request.resume_from = 90;

        uint64_t last_received = 0;
        uint64_t last_total = 0;

        HttpRequest copy = request;
        const HttpResult result = clientOver( transport ).get( copy,
            []( const char*, size_t ) { return true; },
            [&]( uint64_t received, uint64_t total )
            {
                last_received = received;
                last_total = total;

                return true;
            } );

        EXPECT_TRUE( result.ok() );
        EXPECT_EQ( last_received, 100u );
        EXPECT_EQ( last_total, 100u );
    }
}
