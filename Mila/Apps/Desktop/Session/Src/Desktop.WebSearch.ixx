/**
 * @file Desktop.WebSearch.ixx
 * @brief The desktop app's web search: a query sent to Tavily with the person's own key, and the results the model
 *        reads (MilaApp.md 3.7).
 */

export module Desktop.WebSearch;

import std;
import Chat.Json;

import Mila;

namespace Mila::DesktopApp
{
    using Mila::Distribution::HttpFetch;
    using Mila::Distribution::HttpResponse;

    /**
     * @brief Searches the web for the model, one query at a time.
     *
     * The query leaves the machine for Tavily, which is why every search is asked about. The key is the person's,
     * sent only to Tavily and never written into a result, a log or a conversation. A basic search is asked for by
     * name, so a search never costs Tavily's two credits for an advanced one. The request goes through the call's
     * grant, as every connection a tool makes does. A failure throws, and the model reads its message.
     */
    export class WebSearch
    {
    public:

        /// The tool's name, as the research and tasks prompts define it.
        static constexpr std::string_view kName = "web_search";

        explicit WebSearch( std::string key )
            : key_( std::move( key ) )
        {
        }

        /// `query`, and optionally `max_results`, from 1 to 10.
        [[nodiscard]] std::string search( const std::string& arguments_text, const Mila::AI::ToolGrant& grant ) const
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string query = arguments.at( "query" ).get<std::string>();
            const int results = std::clamp( arguments.value( "max_results", kDefaultResults ), 1, kMostResults );

            if ( query.find_first_not_of( " \t\r\n" ) == std::string::npos )
                throw std::runtime_error( "the query is empty" );

            HttpFetch request;
            request.url = std::string( kEndpoint );
            request.headers = { { "Authorization", "Bearer " + key_ }, { "Content-Type", "application/json" } };
            request.body = nlohmann::json{ { "query", query }, { "search_depth", "basic" }, { "max_results", results } }.dump();
            request.low_speed_timeout_seconds = 20;

            std::string body;
            const HttpResponse response = grant.transport()->fetch( request,
                [&body]( const char* data, std::size_t length )
                {
                    body.append( data, length );

                    return body.size() < kMostBytes;
                }, {} );

            if ( response.transport_failed )
                throw std::runtime_error( std::format( "the search service could not be reached: {}", response.message ) );

            if ( response.http_code == 401 || response.http_code == 403 )
                throw std::runtime_error( "the search service refused the key; the user can check it in Settings" );

            if ( response.http_code == 429 )
                throw std::runtime_error( "the search service asked for fewer searches; try again shortly" );

            if ( response.http_code < 200 || response.http_code >= 300 )
                throw std::runtime_error( std::format( "the search service answered HTTP {}", response.http_code ) );

            return format( query, nlohmann::json::parse( body ) );
        }

    private:

        static constexpr std::string_view kEndpoint = "https://api.tavily.com/search";
        static constexpr int kDefaultResults = 5;
        static constexpr int kMostResults = 10;
        static constexpr std::size_t kMostBytes = 1024u * 1024u;

        /// The results as numbered entries the model can cite and read on from.
        static std::string format( const std::string& query, const nlohmann::json& answer )
        {
            const nlohmann::json results = answer.value( "results", nlohmann::json::array() );

            if ( !results.is_array() || results.empty() )
                return std::format( "No results for '{}'.", query );

            std::string text = std::format( "Results for '{}':\n", query );
            int number = 0;

            for ( const nlohmann::json& result : results )
            {
                text += std::format( "\n{}. {}\n{}\n{}\n", ++number, result.value( "title", std::string() ),
                    result.value( "url", std::string() ), result.value( "content", std::string() ) );
            }

            return text + "\nRead a page in full with read_web_page.";
        }

        std::string key_;
    };
}
