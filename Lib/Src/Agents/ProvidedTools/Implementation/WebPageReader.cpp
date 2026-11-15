/**
 * @file WebPageReader.cpp
 * @brief The bodies of WebPageReader's members; the interface is WebPageReader.ixx.
 */

module Agents.WebPageReader;

import std;
import nlohmann.json;
import Networking.HttpClient;
import Networking.HttpTransport;
import Agents.HtmlText;
import Agents.Tool;
import Agents.ToolGrant;
import Agents.ToolNeeds;

namespace Mila::Agents
{
    namespace
    {
        using Mila::Networking::HttpClient;
        using Mila::Networking::HttpRequest;
        using Mila::Networking::HttpResult;
        using Mila::Networking::HttpStatus;
        using Mila::Networking::IHttpTransport;

        constexpr std::size_t kMostBytesFetched = 4u * 1024u * 1024u;
        constexpr std::size_t kMostCharactersRead = 40000;
        constexpr int kMostRedirects = 5;

        bool looksLikeHtml( std::string_view body )
        {
            std::string head( body.substr( 0, 1024 ) );

            for ( char& character : head )
                character = static_cast<char>( std::tolower( static_cast<unsigned char>( character ) ) );

            return head.find( "<!doctype html" ) != std::string::npos || head.find( "<html" ) != std::string::npos
                || head.find( "<head" ) != std::string::npos || head.find( "<body" ) != std::string::npos;
        }

        std::string_view statusMeaning( long code )
        {
            switch ( code )
            {
                case 401:
                case 403: return " (the site refused to show this page)";
                case 404: return " (no such page)";
                case 410: return " (the page was removed)";
                case 429: return " (the site asked for fewer requests)";
                default:  return code >= 500 ? " (the site had an error)" : "";
            }
        }

        /// The page's address after its redirects, and its bytes up to the cut.
        std::pair<std::string, std::string> fetch( const std::string& url, const IHttpTransport& transport )
        {
            // The transport is the caller's and outlives this call, so the client holds it without owning it.
            const HttpClient client( std::shared_ptr<const IHttpTransport>( std::shared_ptr<void>(), &transport ) );

            HttpRequest request;
            request.url = url;
            request.headers = { { "Accept", "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5" } };
            request.maximum_body_bytes = kMostBytesFetched;
            request.maximum_redirects = kMostRedirects;
            request.low_speed_timeout_seconds = 20;

            std::string body;
            const HttpResult result = client.fetchString( request, body );

            switch ( result.status )
            {
                case HttpStatus::Ok:
                    return { result.final_url, std::move( body ) };

                case HttpStatus::RedirectRefused:
                    throw std::runtime_error( std::format( "{} redirects to something that is not a web page", result.final_url ) );

                case HttpStatus::TooManyRedirects:
                    throw std::runtime_error( std::format( "{} redirected more than {} times", url, kMostRedirects ) );

                default:
                    break;
            }

            const long code = result.http_code;

            if ( code == 0 )
                throw std::runtime_error( std::format( "{} could not be reached: {}", result.final_url, result.message ) );

            throw std::runtime_error( std::format( "{} answered HTTP {}{}", result.final_url, code, statusMeaning( code ) ) );
        }
    }

    Tool WebPageReader::tool()
    {
        Tool tool;
        tool.name = "read_web_page";
        tool.description = "Read a web page as text. Links on the page are shown with their addresses, so you can read on from them. A long page is read in parts.";

        // Ordered, so the schema reaches the model with its keys as written here.
        tool.parameters = nlohmann::ordered_json::parse( R"json({
              "type": "object",
              "properties": {
                "url": {
                  "type": "string",
                  "description": "The page's address, starting http:// or https://."
                },
                "start": {
                  "type": "integer",
                  "description": "The character to start from, to read on in a long page. Default 0."
                }
              },
              "required": [
                "url"
              ]
            })json" ).dump();
        tool.run = []( std::string_view arguments, const ToolGrant& grant, std::stop_token )
            {
                return read( std::string( arguments ), *grant.transport() );
            };
        tool.source = "Mila";
        tool.summary = "Reading {url}";
        tool.needs.uses_network = true;

        return tool;
    }

    std::string WebPageReader::read( const std::string& arguments_text, const IHttpTransport& transport )
    {
        const auto arguments = nlohmann::json::parse( arguments_text );
        const std::string url = arguments.at( "url" ).get<std::string>();
        const std::size_t start = arguments.contains( "start" ) && arguments[ "start" ].is_number_integer()
            ? static_cast<std::size_t>( std::max<std::int64_t>( 0, arguments[ "start" ].get<std::int64_t>() ) ) : 0;

        if ( !Mila::Networking::isWebAddress( url ) )
            throw std::runtime_error( std::format( "'{}' is not a web address; give one starting http:// or https://", url ) );

        const auto [final_url, body] = fetch( url, transport );
        std::string title;
        std::string text;

        if ( body.substr( 0, 8192 ).find( '\0' ) != std::string::npos || body.starts_with( "%PDF" ) )
            throw std::runtime_error( std::format( "{} is not a page of text, and only text can be read", final_url ) );

        if ( looksLikeHtml( body ) )
        {
            title = pageTitle( body );
            text = readableText( body );
        }
        else
        {
            text = body;
        }

        if ( start >= text.size() && !text.empty() )
            throw std::runtime_error( std::format( "the page has {} characters, so there is nothing from {}", text.size(), start ) );

        std::size_t end = std::min( text.size(), start + kMostCharactersRead );

        // A part ends between characters, never inside one's UTF-8 bytes.
        while ( end < text.size() && ( static_cast<unsigned char>( text[ end ] ) & 0xC0 ) == 0x80 )
            --end;

        std::string result = std::format( "{}{}\n\n", title.empty() ? "" : "Title: " + title + "\n", "Address: " + final_url );

        if ( text.empty() )
            return result + "(the page has no text)";

        result.append( text, start, end - start );

        if ( end < text.size() )
            result += std::format( "\n\n(the page continues for {} more characters; read on with start {})", text.size() - end, end );

        return result;
    }
}
