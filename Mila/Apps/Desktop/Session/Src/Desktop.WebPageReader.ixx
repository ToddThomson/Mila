/**
 * @file Desktop.WebPageReader.ixx
 * @brief The desktop app's web tool: a page fetched over HTTP or HTTPS and given to the model as readable text
 *        (ToolCalling.md section 5).
 */

export module Desktop.WebPageReader;

import std;
import Chat.Json;
import Desktop.HtmlText;

import Mila;

namespace Mila::DesktopApp
{
    using Mila::Distribution::HttpFetch;
    using Mila::Distribution::HttpResponse;
    using Mila::Distribution::IHttpTransport;

    /**
     * @brief Reads one web page for the model, a part at a time.
     *
     * The page is fetched through Mila's own transport, following at most five redirects and only to http and
     * https addresses, and is cut at a size no page of text needs. Nothing here holds a fetch to public addresses:
     * that is the grant of ToolCalling.md 3.4, which does not exist yet, so every call asks (Untriaged.md, "The
     * desktop app's file tools open files directly"). A failure throws, and the model reads its message.
     */
    export class WebPageReader
    {
    public:

        /// The tool's name, as the research prompt defines it.
        static constexpr std::string_view kName = "read_web_page";

        explicit WebPageReader( std::shared_ptr<const IHttpTransport> transport )
            : transport_( std::move( transport ) )
        {
        }

        /// `url`, and optionally `start`, the character the part read begins at.
        [[nodiscard]] std::string read( const std::string& arguments_text ) const
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string url = arguments.at( "url" ).get<std::string>();
            const std::size_t start = arguments.contains( "start" ) && arguments[ "start" ].is_number_integer()
                ? static_cast<std::size_t>( std::max<std::int64_t>( 0, arguments[ "start" ].get<std::int64_t>() ) ) : 0;

            if ( !isWebAddress( url ) )
                throw std::runtime_error( std::format( "'{}' is not a web address; give one starting http:// or https://", url ) );

            const auto [final_url, body] = fetch( url );
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

    private:

        static constexpr std::size_t kMostBytesFetched = 4u * 1024u * 1024u;
        static constexpr std::size_t kMostCharactersRead = 40000;
        static constexpr int kMostRedirects = 5;

        static bool isWebAddress( std::string_view url )
        {
            return url.starts_with( "http://" ) || url.starts_with( "https://" );
        }

        static bool looksLikeHtml( std::string_view body )
        {
            std::string head( body.substr( 0, 1024 ) );

            for ( char& character : head )
                character = static_cast<char>( std::tolower( static_cast<unsigned char>( character ) ) );

            return head.find( "<!doctype html" ) != std::string::npos || head.find( "<html" ) != std::string::npos
                || head.find( "<head" ) != std::string::npos || head.find( "<body" ) != std::string::npos;
        }

        /// The page's address after its redirects, and its bytes up to the cut.
        std::pair<std::string, std::string> fetch( std::string url ) const
        {
            for ( int hop = 0; hop <= kMostRedirects; ++hop )
            {
                std::string body;
                bool cut = false;

                HttpFetch request;
                request.url = url;
                request.headers = { { "Accept", "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5" } };
                request.low_speed_timeout_seconds = 20;

                const HttpResponse response = transport_->fetch( request,
                    [&body, &cut]( const char* data, std::size_t length )
                    {
                        body.append( data, std::min( length, kMostBytesFetched - body.size() ) );
                        cut = body.size() >= kMostBytesFetched;

                        return !cut;
                    }, {} );

                if ( cut )
                    return { url, std::move( body ) };

                if ( response.transport_failed )
                    throw std::runtime_error( std::format( "{} could not be reached: {}", url, response.message ) );

                const long code = response.http_code;

                if ( code == 301 || code == 302 || code == 303 || code == 307 || code == 308 )
                {
                    const std::string next = Mila::Distribution::resolveRedirect( url, response.location );

                    if ( response.location.empty() || !isWebAddress( next ) )
                        throw std::runtime_error( std::format( "{} redirects to something that is not a web page", url ) );

                    url = next;

                    continue;
                }

                if ( code < 200 || code >= 300 )
                    throw std::runtime_error( std::format( "{} answered HTTP {}{}", url, code, statusMeaning( code ) ) );

                return { url, std::move( body ) };
            }

            throw std::runtime_error( std::format( "{} redirected more than {} times", url, kMostRedirects ) );
        }

        static std::string_view statusMeaning( long code )
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

        std::shared_ptr<const IHttpTransport> transport_;
    };
}
