/**
 * @file WebPageReader.cpp
 * @brief The web tool against a scripted transport: a page read as text and in parts, redirects followed and bounded,
 *        and what is not a page of text refused by name (ToolCalling.md section 5).
 */

#include <gtest/gtest.h>
#include <format>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

import Mila;
import Milo;

namespace Mila::Tests::AI
{
    using namespace Milo;
    using Mila::Distribution::HttpFetch;
    using Mila::Distribution::HttpResponse;
    using Mila::Distribution::HeadersCallback;
    using Mila::Distribution::IHttpTransport;
    using Mila::Distribution::SinkCallback;

    namespace
    {
        struct ScriptedPage
        {
            long code{ 200 };
            std::string location;
            std::string body;
            bool transport_failed{ false };
        };

        class ScriptedSite : public IHttpTransport
        {
        public:

            std::string name() const override { return "scripted"; }

            HttpResponse fetch( const HttpFetch& request, const SinkCallback& sink, const HeadersCallback& ) const override
            {
                const auto page = pages.find( request.url );
                const ScriptedPage scripted = page == pages.end() ? ScriptedPage{ 404 } : page->second;

                HttpResponse response;
                response.http_code = scripted.code;
                response.location = scripted.location;
                response.transport_failed = scripted.transport_failed;

                if ( scripted.code >= 200 && scripted.code < 300 && !scripted.body.empty() )
                {
                    sink( scripted.body.data(), scripted.body.size() );
                }

                return response;
            }

            std::map<std::string, ScriptedPage> pages;
        };

        std::string call( const std::string& url )
        {
            return std::format( R"({{"url":"{}"}})", url );
        }

        void expectRefused( const ScriptedSite& site, const std::string& arguments, std::string_view reason )
        {
            try
            {
                const std::string result = WebPageReader::read( arguments, site );
                ADD_FAILURE() << "read; expected a refusal naming '" << reason << "': " << result;
            }
            catch ( const std::runtime_error& error )
            {
                EXPECT_NE( std::string_view( error.what() ).find( reason ), std::string_view::npos ) << error.what();
            }
        }
    }

    TEST( WebPageReaderTool, CarriesWhatTheModelAndThePersonAreGivenAndNeedsTheNetworkAlone )
    {
        const Tool tool = WebPageReader::tool();

        EXPECT_EQ( tool.name, "read_web_page" );
        EXPECT_FALSE( tool.description.empty() );
        EXPECT_EQ( tool.summary, "Reading {url}" );
        EXPECT_EQ( tool.source, "Milo" );
        EXPECT_TRUE( tool.parameters.starts_with( R"({"type":"object","properties":{"url":)" ) ) << tool.parameters;
        EXPECT_TRUE( tool.needs.uses_network );
        EXPECT_FALSE( tool.needs.reads_files );
        EXPECT_FALSE( tool.needs.writes_files );
        EXPECT_FALSE( tool.needs.runs_programs );
    }

    TEST( WebPageReaderTool, AGrantWithNoNetworkIsRefused )
    {
        try
        {
            (void)WebPageReader::tool().run( call( "https://example.test/" ), ToolGrant{}, {} );
            ADD_FAILURE() << "ran with no network";
        }
        catch ( const std::runtime_error& error )
        {
            EXPECT_NE( std::string_view( error.what() ).find( "no network" ), std::string_view::npos ) << error.what();
        }
    }

    TEST( WebPageReader, APageIsReadAsTextWithItsTitleAndAddress )
    {
        ScriptedSite site;
        site.pages[ "https://example.test/" ] = { 200, {},
            "<html><head><title>The Title</title></head><body><script>hidden()</script><p>Shown text.</p></body></html>" };

        const std::string result = WebPageReader::read( call( "https://example.test/" ), site );

        EXPECT_NE( result.find( "Title: The Title" ), std::string::npos ) << result;
        EXPECT_NE( result.find( "Address: https://example.test/" ), std::string::npos ) << result;
        EXPECT_NE( result.find( "Shown text." ), std::string::npos ) << result;
        EXPECT_EQ( result.find( "hidden" ), std::string::npos ) << result;
    }

    TEST( WebPageReader, PlainTextIsReadAsItIs )
    {
        ScriptedSite site;
        site.pages[ "https://example.test/notes.txt" ] = { 200, {}, "<not html> as written" };

        EXPECT_NE( WebPageReader::read( call( "https://example.test/notes.txt" ), site ).find( "<not html> as written" ), std::string::npos );
    }

    TEST( WebPageReader, ALongPageIsReadInPartsNeverInsideACharacter )
    {
        // The two bytes of the e with an accent lie across the 40000 characters read at once.
        const std::string text = std::string( 39999, 'x' ) + "\xC3\xA9" + "the end";

        ScriptedSite site;
        site.pages[ "https://example.test/long.txt" ] = { 200, {}, text };

        const std::string first = WebPageReader::read( call( "https://example.test/long.txt" ), site );

        EXPECT_EQ( first.find( "\xC3" ), std::string::npos ) << "the first part ends before the character";
        EXPECT_NE( first.find( "read on with start 39999" ), std::string::npos );

        const std::string rest = WebPageReader::read( R"({"url":"https://example.test/long.txt","start":39999})", site );

        EXPECT_NE( rest.find( "\xC3\xA9the end" ), std::string::npos );
        EXPECT_EQ( rest.find( "read on" ), std::string::npos );
    }

    TEST( WebPageReader, ReadingFromPastTheEndIsRefused )
    {
        ScriptedSite site;
        site.pages[ "https://example.test/short.txt" ] = { 200, {}, "short" };

        expectRefused( site, R"({"url":"https://example.test/short.txt","start":5})", "nothing from 5" );
    }

    TEST( WebPageReader, ARedirectIsFollowedAndTheFinalAddressShown )
    {
        ScriptedSite site;
        site.pages[ "https://example.test/old" ] = { 301, "/new" };
        site.pages[ "https://example.test/new" ] = { 200, {}, "moved here" };

        const std::string result = WebPageReader::read( call( "https://example.test/old" ), site );

        EXPECT_NE( result.find( "Address: https://example.test/new" ), std::string::npos ) << result;
        EXPECT_NE( result.find( "moved here" ), std::string::npos ) << result;
    }

    TEST( WebPageReader, FiveRedirectsAreFollowedAndASixthRefused )
    {
        ScriptedSite site;

        for ( int hop = 0; hop < 5; ++hop )
        {
            site.pages[ std::format( "https://example.test/{}", hop ) ] = { 302, std::format( "/{}", hop + 1 ) };
        }

        site.pages[ "https://example.test/5" ] = { 200, {}, "arrived" };

        EXPECT_NE( WebPageReader::read( call( "https://example.test/0" ), site ).find( "arrived" ), std::string::npos );

        site.pages[ "https://example.test/5" ] = { 302, "/6" };
        site.pages[ "https://example.test/6" ] = { 200, {}, "too far" };

        expectRefused( site, call( "https://example.test/0" ), "redirected more than 5 times" );
    }

    TEST( WebPageReader, ARedirectToAnythingButTheWebIsRefused )
    {
        ScriptedSite site;
        site.pages[ "https://example.test/" ] = { 302, "file:///c:/secret.txt" };

        expectRefused( site, call( "https://example.test/" ), "not a web page" );
    }

    TEST( WebPageReader, AnAddressThatIsNotTheWebIsRefusedBeforeAnythingIsFetched )
    {
        expectRefused( ScriptedSite{}, call( "file:///c:/secret.txt" ), "is not a web address" );
    }

    TEST( WebPageReader, WhatIsNotTextIsRefused )
    {
        ScriptedSite site;
        site.pages[ "https://example.test/report.pdf" ] = { 200, {}, "%PDF-1.7 ..." };
        site.pages[ "https://example.test/picture" ] = { 200, {}, std::string( "ab\0cd", 5 ) };

        expectRefused( site, call( "https://example.test/report.pdf" ), "not a page of text" );
        expectRefused( site, call( "https://example.test/picture" ), "not a page of text" );
    }

    TEST( WebPageReader, AFailureSaysWhatHappened )
    {
        ScriptedSite site;
        site.pages[ "https://example.test/down" ] = { 0, {}, {}, true };

        expectRefused( site, call( "https://example.test/missing" ), "(no such page)" );
        expectRefused( site, call( "https://example.test/down" ), "could not be reached" );
    }
}
