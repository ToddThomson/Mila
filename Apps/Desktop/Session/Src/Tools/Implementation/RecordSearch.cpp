/**
 * @file RecordSearch.cpp
 * @brief Searching the stored chats' messages for a person's words, and the passage around each match.
 */

module Desktop.RecordSearch;

import std;
import nlohmann.json;

import Mila;

namespace Mila::Desktop
{
    namespace
    {
        // Words every message may hold, which narrow nothing: a search for *the invoice from June* looks for the rest.
        constexpr std::array<std::string_view, 12> kCommonWords = {
            "the", "a", "an", "and", "or", "of", "to", "in", "on", "my", "from", "for" };

        std::string folded( std::string_view text )
        {
            std::string lower( text );

            // ASCII only, so a byte offset in the folded text is the same in the message.
            std::ranges::transform( lower, lower.begin(), []( unsigned char character )
                {
                    return static_cast<char>( std::tolower( character ) );
                } );

            return lower;
        }

        std::vector<std::string> searchWords( std::string_view text )
        {
            std::vector<std::string> words;
            std::string word;

            const auto finish = [&]
                {
                    if ( !word.empty() && std::ranges::find( kCommonWords, word ) == kCommonWords.end()
                        && std::ranges::find( words, word ) == words.end() )
                    {
                        words.push_back( word );
                    }

                    word.clear();
                };

            for ( const char character : folded( text ) )
            {
                if ( std::isalnum( static_cast<unsigned char>( character ) ) != 0 || static_cast<unsigned char>( character ) >= 0x80 )
                {
                    word += character;
                }
                else
                {
                    finish();
                }
            }

            finish();

            return words;
        }

        std::string text( const nlohmann::json& object, std::string_view key )
        {
            const auto value = object.find( key );

            return value != object.end() && value->is_string() ? value->get<std::string>() : std::string{};
        }

        bool continuationByte( char byte )
        {
            return ( static_cast<unsigned char>( byte ) & 0xC0 ) == 0x80;
        }

        /// The message around @p at, about kPassageCharacters long, never cut inside a character, on one line.
        std::string passageAround( const std::string& message, std::size_t at )
        {
            const std::size_t half = kPassageCharacters / 2;
            std::size_t first = at > half ? at - half : 0;
            std::size_t last = std::min( message.size(), at + half );

            const auto space = []( char character ) { return std::isspace( static_cast<unsigned char>( character ) ) != 0; };

            // In to a word's edge where one is between the cut and the match, then off any byte inside a character.
            if ( first > 0 && !space( message[ first - 1 ] ) )
            {
                for ( std::size_t edge = first; edge < at; ++edge )
                {
                    if ( space( message[ edge ] ) )
                    {
                        first = edge + 1;

                        break;
                    }
                }
            }

            if ( last < message.size() && !space( message[ last ] ) )
            {
                for ( std::size_t edge = last; edge > at; --edge )
                {
                    if ( space( message[ edge - 1 ] ) )
                    {
                        last = edge - 1;

                        break;
                    }
                }
            }

            while ( first > 0 && continuationByte( message[ first ] ) )
            {
                --first;
            }

            while ( last < message.size() && continuationByte( message[ last ] ) )
            {
                ++last;
            }

            std::string passage = message.substr( first, last - first );
            std::ranges::replace_if( passage, []( char character ) { return character == '\n' || character == '\r' || character == '\t'; }, ' ' );

            return std::format( "{}{}{}", first > 0 ? "..." : "", passage, last < message.size() ? "..." : "" );
        }

        std::string projectName( std::span<const nlohmann::json> projects, std::string_view id )
        {
            for ( const nlohmann::json& project : projects )
            {
                if ( !id.empty() && text( project, "id" ) == id )
                {
                    return text( project, "name" );
                }
            }

            return {};
        }

        struct Match
        {
            const nlohmann::json* conversation;

            /// Who said what: *the user wrote: "..." and Mila replied: "..."*.
            std::string said;
            bool exposed;
        };

        /// One message's part of a match: the passage around the first word it holds, or its start when it holds none.
        std::string passageOf( const std::string& content, std::span<const std::string> sought )
        {
            const std::string lower = folded( content );
            std::size_t at = std::string::npos;

            for ( const std::string& word : sought )
            {
                at = std::min( at, lower.find( word ) );
            }

            return passageAround( content, at == std::string::npos ? 0 : at );
        }
    }

    bool chatIsExposed( const nlohmann::json& conversation )
    {
        const auto exposure = conversation.find( "exposure" );

        if ( exposure != conversation.end() )
        {
            return exposure->is_array() && !exposure->empty();
        }

        // Stored before the mark was: every tool result counts, as Session::openConversation counts it.
        const auto turns = conversation.find( "turns" );

        if ( turns == conversation.end() || !turns->is_array() )
        {
            return false;
        }

        return std::ranges::any_of( *turns, []( const nlohmann::json& turn ) { return text( turn, "role" ) == "tool"; } );
    }

    Mila::Agents::ToolResult findRecords( std::string_view words, std::span<const nlohmann::json> conversations,
        std::span<const nlohmann::json> projects, std::string_view open_chat )
    {
        const std::vector<std::string> sought = searchWords( words );

        if ( sought.empty() )
        {
            return { "Give find_records the words to look for, such as a name, a subject or a phrase the user remembers.",
                Mila::Agents::ResultOrigin::Application };
        }

        std::vector<const nlohmann::json*> ordered;

        for ( const nlohmann::json& conversation : conversations )
        {
            if ( text( conversation, "id" ) != open_chat )
            {
                ordered.push_back( &conversation );
            }
        }

        std::ranges::stable_sort( ordered, std::greater<>{}, []( const nlohmann::json* conversation ) { return text( *conversation, "updated" ); } );

        std::vector<Match> matches;
        std::size_t more = 0;

        for ( const nlohmann::json* conversation : ordered )
        {
            const auto messages = conversation->find( "messages" );

            if ( messages == conversation->end() || !messages->is_array() )
            {
                continue;
            }

            const bool exposed = chatIsExposed( *conversation );

            // An exchange is the person's message and Mila's reply to it, searched together: an answer seldom repeats
            // the words of the question it answers.
            for ( std::size_t index = 0; index < messages->size(); ++index )
            {
                const nlohmann::json& message = ( *messages )[ index ];
                const std::string role = text( message, "role" );
                std::string asked;
                std::string replied;

                if ( role == "user" )
                {
                    asked = text( message, "content" );

                    if ( index + 1 < messages->size() && text( ( *messages )[ index + 1 ], "role" ) == "assistant" )
                    {
                        replied = text( ( *messages )[ ++index ], "content" );
                    }
                }
                else if ( role == "assistant" )
                {
                    replied = text( message, "content" );
                }

                const std::string lower = folded( asked + "\n" + replied );

                if ( ( asked.empty() && replied.empty() )
                    || !std::ranges::all_of( sought, [&lower]( const std::string& word ) { return lower.contains( word ); } ) )
                {
                    continue;
                }

                if ( matches.size() == kMostRecordsFound )
                {
                    ++more;

                    continue;
                }

                std::string said;

                if ( !asked.empty() )
                {
                    said = std::format( "the user wrote: \"{}\"", passageOf( asked, sought ) );
                }

                if ( !replied.empty() )
                {
                    said += std::format( "{}Mila replied: \"{}\"", said.empty() ? "" : " and ", passageOf( replied, sought ) );
                }

                matches.push_back( { conversation, std::move( said ), exposed } );
            }
        }

        std::string joined;

        for ( const std::string& word : sought )
        {
            joined += std::format( "{}{}", joined.empty() ? "" : " ", word );
        }

        if ( matches.empty() )
        {
            return { std::format( "Nothing in the user's other chats holds all of: {}. Say that nothing was found; "
                "do not guess what the chats held.", joined ), Mila::Agents::ResultOrigin::Application };
        }

        std::string result = std::format( "{} in the user's other chats holding {}, the most recently changed chat first:",
            matches.size() == 1 ? std::string( "1 exchange" ) : std::format( "{} exchanges", matches.size() ), joined );

        bool any_exposed = false;

        for ( std::size_t index = 0; index < matches.size(); ++index )
        {
            const Match& match = matches[ index ];
            const std::string project = projectName( projects, text( *match.conversation, "project" ) );
            const std::string changed = text( *match.conversation, "updated" ).substr( 0, 10 );

            result += std::format( "\n{}. In the chat \"{}\"{}{}, {}", index + 1, text( *match.conversation, "title" ),
                project.empty() ? std::string{} : ", in the project " + project,
                changed.empty() ? std::string{} : ", changed " + changed, match.said );

            if ( match.exposed )
            {
                result += " (that chat had read pages or files, so this may not be the user's own)";
            }

            any_exposed = any_exposed || match.exposed;
        }

        if ( more > 0 )
        {
            result += std::format( "\n...and {} more; narrow the words to find fewer.", more );
        }

        return { std::move( result ), any_exposed ? Mila::Agents::ResultOrigin::Outside : Mila::Agents::ResultOrigin::Application };
    }
}
