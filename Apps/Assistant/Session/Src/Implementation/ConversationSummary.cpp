/**
 * @file ConversationSummary.cpp
 * @brief Reading a stored conversation's plan, last reply and written files.
 *
 * Every field is derived from the stored messages and plan, so a conversation written before a field existed has it.
 */

module Assistant.ConversationSummary;

import std;
import nlohmann.json;
import Assistant.Creations;
import Assistant.Offers;
import Assistant.SetupSteps;

namespace Mila::Assistant
{
    namespace
    {
        constexpr std::size_t kMostFilesWritten = 20;
        constexpr std::size_t kMostThingsSent = 50;
        constexpr std::size_t kReplyStartBytes = 160;

        nlohmann::json planProgress( const nlohmann::json& plan )
        {
            if ( !plan.is_array() || plan.empty() )
            {
                return nullptr;
            }

            std::int64_t finished = 0;
            nlohmann::json in_progress = nullptr;
            nlohmann::json first_pending = nullptr;

            for ( const nlohmann::json& step : plan )
            {
                if ( !step.is_object() )
                {
                    continue;
                }

                const std::string status = step.value( "status", std::string( "pending" ) );
                const nlohmann::json text = step.value( "text", nlohmann::json( nullptr ) );

                if ( status == "done" || status == "skipped" )
                {
                    ++finished;
                }
                else if ( status == "in_progress" && in_progress.is_null() )
                {
                    in_progress = text;
                }
                else if ( status == "pending" && first_pending.is_null() )
                {
                    first_pending = text;
                }
            }

            const nlohmann::json current = in_progress.is_null() ? first_pending : in_progress;

            return { { "steps", plan.size() }, { "finished", finished }, { "current", current } };
        }

        nlohmann::json lastFinish( const nlohmann::json& messages )
        {
            for ( auto message = messages.rbegin(); message != messages.rend(); ++message )
            {
                if ( !message->is_object() || message->value( "role", std::string{} ) != "assistant" )
                {
                    continue;
                }

                const nlohmann::json stats = message->value( "stats", nlohmann::json( nullptr ) );

                if ( stats.is_object() && stats.value( "finish", nlohmann::json( nullptr ) ).is_string() )
                {
                    return stats[ "finish" ];
                }

                return nullptr;
            }

            return nullptr;
        }

        /// Runs of white space become one space, so a reply's paragraphs read as one line.
        std::string flattened( std::string_view text, std::size_t most_bytes )
        {
            std::string line;
            bool space = false;

            for ( const char c : text )
            {
                if ( line.size() >= most_bytes )
                {
                    break;
                }

                if ( c == ' ' || c == '\t' || c == '\r' || c == '\n' )
                {
                    space = !line.empty();

                    continue;
                }

                if ( space )
                {
                    line.push_back( ' ' );
                    space = false;
                }

                line.push_back( c );
            }

            // A UTF-8 character's continuation bytes are 10xxxxxx: one cut among them takes its lead byte too.
            if ( line.size() > most_bytes )
            {
                line.resize( most_bytes );
            }

            if ( !line.empty() && ( static_cast<unsigned char>( line.back() ) & 0x80u ) != 0 )
            {
                std::size_t lead = line.size() - 1;

                while ( lead > 0 && ( static_cast<unsigned char>( line[ lead ] ) & 0xC0u ) == 0x80u )
                {
                    --lead;
                }

                const unsigned char first = static_cast<unsigned char>( line[ lead ] );
                const std::size_t length = first >= 0xF0u ? 4 : first >= 0xE0u ? 3 : first >= 0xC0u ? 2 : 1;

                if ( lead + length > line.size() )
                {
                    line.resize( lead );
                }
            }

            return line;
        }

        nlohmann::json replyStart( const nlohmann::json& messages )
        {
            for ( auto message = messages.rbegin(); message != messages.rend(); ++message )
            {
                if ( !message->is_object() || message->value( "role", std::string{} ) != "assistant" )
                {
                    continue;
                }

                const nlohmann::json content = message->value( "content", nlohmann::json( nullptr ) );

                if ( !content.is_string() )
                {
                    return nullptr;
                }

                const std::string line = flattened( content.get<std::string>(), kReplyStartBytes );

                return line.empty() ? nlohmann::json( nullptr ) : nlohmann::json( line );
            }

            return nullptr;
        }

        nlohmann::json filesWritten( const nlohmann::json& messages )
        {
            struct Written
            {
                std::string path;
                std::string tool;
                nlohmann::json time;
                nlohmann::json record;
            };

            std::vector<Written> written;

            for ( const nlohmann::json& message : messages )
            {
                if ( !message.is_object() || !message.value( "tool_calls", nlohmann::json( nullptr ) ).is_array() )
                {
                    continue;
                }

                for ( const nlohmann::json& call : message[ "tool_calls" ] )
                {
                    const std::optional<std::string> path = writtenPath( call );

                    if ( !path.has_value() )
                    {
                        continue;
                    }

                    // A file written again moves to the end, where the most recent writes are.
                    std::erase_if( written, [ & ]( const Written& entry ) { return entry.path == *path; } );
                    written.push_back( { *path, call[ "name" ].get<std::string>(), call.value( "time", nlohmann::json( nullptr ) ),
                        call.value( "written", nlohmann::json( nullptr ) ) } );
                }
            }

            const std::size_t first = written.size() > kMostFilesWritten ? written.size() - kMostFilesWritten : 0;
            nlohmann::json files = nlohmann::json::array();

            for ( std::size_t index = first; index < written.size(); ++index )
            {
                files.push_back( { { "path", written[ index ].path }, { "tool", written[ index ].tool },
                    { "time", written[ index ].time }, { "written", written[ index ].record } } );
            }

            return files;
        }

        nlohmann::json repliesKept( const nlohmann::json& messages )
        {
            nlohmann::json kept = nlohmann::json::array();

            for ( std::size_t index = 0; index < messages.size(); ++index )
            {
                const nlohmann::json& message = messages[ index ];

                if ( !message.is_object() || message.value( "role", std::string{} ) != "assistant"
                    || !message.value( "kept", nlohmann::json( nullptr ) ).is_string() )
                {
                    continue;
                }

                const std::string line = flattened( message.value( "content", std::string{} ), kReplyStartBytes );

                kept.push_back( { { "message", index }, { "time", message[ "kept" ] },
                    { "start", line.empty() ? nlohmann::json( nullptr ) : nlohmann::json( line ) } } );
            }

            return kept;
        }

        /// The host of a web address, for saying where something went; the address itself when it has none.
        std::string hostIn( const std::string& address )
        {
            const std::size_t scheme = address.find( "://" );

            if ( scheme == std::string::npos )
            {
                return address;
            }

            const std::size_t start = scheme + 3;
            std::string authority = address.substr( start, address.find_first_of( "/?#", start ) - start );
            const std::size_t user = authority.rfind( '@' );

            return user == std::string::npos ? authority : authority.substr( user + 1 );
        }

        /// What a call sent and where, or null for a call that sends nothing.
        nlohmann::json sentByCall( const nlohmann::json& call )
        {
            if ( !call.is_object() )
            {
                return nullptr;
            }

            const nlohmann::json decision = call.value( "decision", nlohmann::json( nullptr ) );
            const bool answered = call.value( "result", nlohmann::json( nullptr ) ).is_string();
            const nlohmann::json arguments_text = call.value( "arguments", nlohmann::json( nullptr ) );

            // Declined, or stopped before it ran. A failed call still went out.
            if ( decision == "deny" || !answered || !arguments_text.is_string() )
            {
                return nullptr;
            }

            const nlohmann::json arguments = nlohmann::json::parse( arguments_text.get<std::string>(), nullptr, false );

            if ( !arguments.is_object() )
            {
                return nullptr;
            }

            const std::string name = call.value( "name", std::string{} );
            const auto text = [ &arguments ]( const char* key ) { return arguments.value( key, std::string{} ); };
            nlohmann::json time = call.value( "time", nlohmann::json( nullptr ) );

            if ( name == "read_web_page" && arguments.value( "url", nlohmann::json( nullptr ) ).is_string() )
            {
                return { { "kind", "page" }, { "what", text( "url" ) }, { "where", hostIn( text( "url" ) ) }, { "time", std::move( time ) } };
            }

            if ( name == "web_search" && arguments.value( "query", nlohmann::json( nullptr ) ).is_string() )
            {
                return { { "kind", "search" }, { "what", text( "query" ) }, { "where", "your search provider" }, { "time", std::move( time ) } };
            }

            if ( name == "run_command" && arguments.value( "network", nlohmann::json( false ) ) == true
                && arguments.value( "command", nlohmann::json( nullptr ) ).is_string() )
            {
                return { { "kind", "command" }, { "what", text( "command" ) }, { "where", "the internet" }, { "time", std::move( time ) } };
            }

            return nullptr;
        }

        nlohmann::json whatWasSent( const nlohmann::json& conversation, const nlohmann::json& messages )
        {
            std::vector<nlohmann::json> sent;

            for ( const nlohmann::json& message : messages )
            {
                if ( !message.is_object() || !message.value( "tool_calls", nlohmann::json( nullptr ) ).is_array() )
                {
                    continue;
                }

                for ( const nlohmann::json& call : message[ "tool_calls" ] )
                {
                    if ( nlohmann::json entry = sentByCall( call ); !entry.is_null() )
                    {
                        sent.push_back( std::move( entry ) );
                    }
                }
            }

            const nlohmann::json links = conversation.value( "links_opened", nlohmann::json( nullptr ) );

            if ( links.is_array() )
            {
                for ( const nlohmann::json& link : links )
                {
                    if ( link.is_object() && link.value( "url", nlohmann::json( nullptr ) ).is_string() )
                    {
                        const std::string url = link[ "url" ].get<std::string>();

                        sent.push_back( { { "kind", "link" }, { "what", url }, { "where", hostIn( url ) },
                            { "time", link.value( "time", nlohmann::json( nullptr ) ) } } );
                    }
                }
            }

            // ISO 8601 in UTC sorts as text; a call stored with no time keeps its place before those that have one.
            std::ranges::stable_sort( sent, {}, []( const nlohmann::json& entry )
                {
                    return entry[ "time" ].is_string() ? entry[ "time" ].get<std::string>() : std::string{};
                } );

            const std::size_t first = sent.size() > kMostThingsSent ? sent.size() - kMostThingsSent : 0;
            nlohmann::json kept = nlohmann::json::array();

            for ( std::size_t index = first; index < sent.size(); ++index )
            {
                kept.push_back( std::move( sent[ index ] ) );
            }

            return kept;
        }
    }

    nlohmann::json summarizeConversation( const nlohmann::json& conversation )
    {
        const nlohmann::json plan = conversation.value( "plan", nlohmann::json( nullptr ) );
        const nlohmann::json stored = conversation.value( "messages", nlohmann::json( nullptr ) );
        const nlohmann::json messages = stored.is_array() ? stored : nlohmann::json::array();

        return { { "progress", planProgress( plan ) }, { "finish", lastFinish( messages ) },
            { "reply_start", replyStart( messages ) }, { "wrote", filesWritten( messages ) },
            { "sent", whatWasSent( conversation, messages ) }, { "kept", repliesKept( messages ) },
            { "offers", offersIn( messages ) },
            { "setup", SetupSteps::outstandingIn( conversation.value( "setup", nlohmann::json( nullptr ) ) ) } };
    }
}
