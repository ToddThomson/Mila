/**
 * @file ConversationSummary.cpp
 * @brief Reading a stored conversation's plan, last reply and written files.
 *
 * Every field is derived from the stored messages and plan, so a conversation written before a field existed has it.
 */

module Desktop.ConversationSummary;

import std;
import nlohmann.json;

namespace Mila::Desktop
{
    namespace
    {
        constexpr std::size_t kMostFilesWritten = 20;
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

        /// The path a file tool's call named, or nothing when the call failed, did not run or named none.
        std::optional<std::string> writtenPath( const nlohmann::json& call )
        {
            if ( !call.is_object() )
            {
                return std::nullopt;
            }

            const std::string name = call.value( "name", std::string{} );
            const bool failed = call.value( "is_error", false );
            const bool ran = call.value( "result", nlohmann::json( nullptr ) ).is_string();

            if ( ( name != "write_file" && name != "edit_file" ) || failed || !ran )
            {
                return std::nullopt;
            }

            const nlohmann::json arguments_text = call.value( "arguments", nlohmann::json( nullptr ) );

            if ( !arguments_text.is_string() )
            {
                return std::nullopt;
            }

            const nlohmann::json arguments = nlohmann::json::parse( arguments_text.get<std::string>(), nullptr, false );

            if ( !arguments.is_object() || !arguments.value( "path", nlohmann::json( nullptr ) ).is_string() )
            {
                return std::nullopt;
            }

            return arguments[ "path" ].get<std::string>();
        }

        nlohmann::json filesWritten( const nlohmann::json& messages )
        {
            std::vector<std::pair<std::string, std::string>> written;

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
                    std::erase_if( written, [ & ]( const auto& entry ) { return entry.first == *path; } );
                    written.emplace_back( *path, call[ "name" ].get<std::string>() );
                }
            }

            const std::size_t first = written.size() > kMostFilesWritten ? written.size() - kMostFilesWritten : 0;
            nlohmann::json files = nlohmann::json::array();

            for ( std::size_t index = first; index < written.size(); ++index )
            {
                files.push_back( { { "path", written[ index ].first }, { "tool", written[ index ].second } } );
            }

            return files;
        }
    }

    nlohmann::json summarizeConversation( const nlohmann::json& conversation )
    {
        const nlohmann::json plan = conversation.value( "plan", nlohmann::json( nullptr ) );
        const nlohmann::json stored = conversation.value( "messages", nlohmann::json( nullptr ) );
        const nlohmann::json messages = stored.is_array() ? stored : nlohmann::json::array();

        return { { "progress", planProgress( plan ) }, { "finish", lastFinish( messages ) },
            { "reply_start", replyStart( messages ) }, { "wrote", filesWritten( messages ) } };
    }
}
