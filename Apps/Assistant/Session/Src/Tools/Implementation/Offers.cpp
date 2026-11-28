/**
 * @file Offers.cpp
 * @brief Making, finding and checking the offers kept on a chat's calls.
 */

module Assistant.Offers;

import std;
import nlohmann.json;
import Assistant.Places;

namespace Mila::Assistant
{
    namespace
    {
        // The most other chats one offer gathers, and the most names a refusal lists.
        constexpr std::size_t kMostChatsGathered = 20;

        std::string text( const nlohmann::json& object, std::string_view key )
        {
            if ( !object.is_object() )
            {
                return {};
            }

            const auto value = object.find( key );

            return value != object.end() && value->is_string() ? value->get<std::string>() : std::string{};
        }

        std::string trimmed( std::string_view line )
        {
            const auto first = line.find_first_not_of( " \t\r\n" );

            if ( first == std::string_view::npos )
            {
                return {};
            }

            const auto last = line.find_last_not_of( " \t\r\n" );

            return std::string( line.substr( first, last - first + 1 ) );
        }

        std::string folded( std::string_view words )
        {
            std::string lower( words );

            std::ranges::transform( lower, lower.begin(), []( unsigned char character )
                {
                    return static_cast<char>( std::tolower( character ) );
                } );

            return lower;
        }

        /// A line of `chats` as the model may write it: a bullet, a number or quotes around the title are not part of it.
        std::string chatWords( std::string_view line )
        {
            std::string words = trimmed( line );

            if ( words.starts_with( "- " ) || words.starts_with( "* " ) )
            {
                words = trimmed( std::string_view( words ).substr( 2 ) );
            }

            // A list's number, *2.* or *2)*, but not a title that begins with one, such as *2026 taxes*.
            const auto digits = words.find_first_not_of( "0123456789" );

            if ( digits != std::string::npos && digits > 0 && digits + 1 < words.size()
                && ( words[ digits ] == '.' || words[ digits ] == ')' ) && words[ digits + 1 ] == ' ' )
            {
                words = trimmed( std::string_view( words ).substr( digits + 1 ) );
            }

            while ( words.size() >= 2 && ( ( words.front() == '"' && words.back() == '"' ) || ( words.front() == '\'' && words.back() == '\'' ) ) )
            {
                words = trimmed( std::string_view( words ).substr( 1, words.size() - 2 ) );
            }

            return words;
        }

        const nlohmann::json* chatNamed( const nlohmann::json& chats, std::string_view id )
        {
            if ( !chats.is_array() )
            {
                return nullptr;
            }

            for ( const nlohmann::json& chat : chats )
            {
                if ( text( chat, "id" ) == id )
                {
                    return &chat;
                }
            }

            return nullptr;
        }

        bool inProject( const nlohmann::json& chat )
        {
            return !text( chat, "project" ).empty();
        }

        std::set<std::string> chatIds( const nlohmann::json& offer )
        {
            std::set<std::string> ids;

            for ( const nlohmann::json& chat : offer.value( "chats", nlohmann::json::array() ) )
            {
                ids.insert( text( chat, "id" ) );
            }

            return ids;
        }

        /// The project offers the stored chats hold, each once.
        std::vector<nlohmann::json> projectOffers( const nlohmann::json& chats )
        {
            std::vector<nlohmann::json> offers;

            if ( !chats.is_array() )
            {
                return offers;
            }

            for ( const nlohmann::json& chat : chats )
            {
                for ( const nlohmann::json& offer : chat.value( "offers", nlohmann::json::array() ) )
                {
                    if ( text( offer, "kind" ) == "project" )
                    {
                        offers.push_back( offer );
                    }
                }
            }

            return offers;
        }

        std::string quotedList( const std::vector<std::string>& names )
        {
            std::string joined;

            for ( const std::string& name : names )
            {
                joined += joined.empty() ? "" : "; ";
                joined += name;
            }

            return joined;
        }
    }

    nlohmann::json offerProject( std::string_view arguments, const nlohmann::json& chats,
        std::span<const nlohmann::json> projects, std::string_view open_chat, bool names_allowed )
    {
        const auto refused = []( std::string result ) -> nlohmann::json
            {
                return { { "offer", nullptr }, { "result", std::move( result ) } };
            };

        const nlohmann::json parsed = nlohmann::json::parse( arguments, nullptr, false );
        const std::string name = trimmed( text( parsed, "name" ) );
        const std::string purpose = trimmed( text( parsed, "purpose" ) );

        if ( name.empty() )
        {
            return refused( "Give propose_project the project's name; nothing was offered." );
        }

        for ( const nlohmann::json& project : projects )
        {
            if ( folded( text( project, "name" ) ) == folded( name ) )
            {
                return refused( std::format( "There is already a project called {}, so nothing was offered; "
                    "move_to_project moves this chat into it.", text( project, "name" ) ) );
            }
        }

        const nlohmann::json* asking = chatNamed( chats, open_chat );

        if ( asking == nullptr )
        {
            return refused( "This chat is not stored yet, so nothing was offered." );
        }

        if ( inProject( *asking ) )
        {
            return refused( "This chat is already in a project, so nothing was offered." );
        }

        nlohmann::json gathered = nlohmann::json::array( { { { "id", open_chat }, { "title", text( *asking, "title" ) } } } );
        std::vector<std::string> problems;

        // A list of titles, as the schema asks and Gemma 4 26B-A4B wrote it (Todd's walk, 2026-11-27), or one a line.
        std::vector<std::string> named;
        const nlohmann::json listed = parsed.is_object() ? parsed.value( "chats", nlohmann::json( nullptr ) ) : nlohmann::json( nullptr );

        if ( listed.is_array() )
        {
            for ( const nlohmann::json& title : listed )
            {
                if ( title.is_string() )
                {
                    named.push_back( title.get<std::string>() );
                }
            }
        }
        else
        {
            std::istringstream lines( text( parsed, "chats" ) );

            for ( std::string line; std::getline( lines, line ); )
            {
                named.push_back( line );
            }
        }

        for ( const std::string& line : named )
        {
            if ( gathered.size() > kMostChatsGathered )
            {
                break;
            }

            const std::string words = chatWords( line );

            if ( words.empty() )
            {
                continue;
            }

            nlohmann::json fit = nlohmann::json::array();

            for ( const nlohmann::json& place : findPlaces( words, {}, chats, open_chat ) )
            {
                if ( place[ "view" ] == "chat" )
                {
                    fit.push_back( place );
                }
            }

            if ( fit.empty() )
            {
                problems.push_back( std::format( "\"{}\" names no chat", words ) );

                continue;
            }

            if ( fit.size() > 1 )
            {
                std::vector<std::string> titles;

                for ( const nlohmann::json& place : fit )
                {
                    titles.push_back( text( place, "name" ) );
                }

                problems.push_back( names_allowed
                    ? std::format( "\"{}\" fits several chats: {}", words, quotedList( titles ) )
                    : std::format( "\"{}\" fits {} chats", words, fit.size() ) );

                continue;
            }

            const std::string id = text( fit[ 0 ], "id" );
            const nlohmann::json* chat = chatNamed( chats, id );

            if ( chat == nullptr || inProject( *chat ) )
            {
                problems.push_back( std::format( "\"{}\" is in a project, so it stays there", words ) );

                continue;
            }

            const bool already = std::ranges::any_of( gathered, [&id]( const nlohmann::json& held ) { return held[ "id" ] == id; } );

            if ( !already )
            {
                gathered.push_back( { { "id", id }, { "title", text( *chat, "title" ) } } );
            }
        }

        nlohmann::json offer = { { "kind", "project" }, { "state", "open" }, { "name", name }, { "purpose", purpose },
            { "chats", gathered }, { "in_window", !names_allowed } };

        const std::set<std::string> asked = chatIds( offer );

        for ( const nlohmann::json& earlier : projectOffers( chats ) )
        {
            const std::string state = text( earlier, "state" );
            const std::set<std::string> held = chatIds( earlier );

            if ( ( state == "open" || state == "undecided" ) && held == asked )
            {
                return refused( "A project for these chats was already offered and waits for the user's answer, so it "
                    "was not offered again." );
            }

            if ( state == "declined" && std::ranges::includes( held, asked ) )
            {
                return refused( "The user declined a project for these chats, so it was not offered again. Do not "
                    "suggest it again." );
            }
        }

        const std::size_t others = gathered.size() - 1;
        std::string gathers = others == 0 ? std::string( "this chat" )
            : std::format( "this chat and {} other{}", others, others == 1 ? "" : "s" );

        if ( names_allowed && others > 0 )
        {
            std::vector<std::string> titles;

            for ( std::size_t index = 1; index < gathered.size(); ++index )
            {
                titles.push_back( text( gathered[ index ], "title" ) );
            }

            gathers += ": " + quotedList( titles );
        }

        std::string result = std::format( "Offered the user a project called {} that gathers {}. {} Do not make the "
            "project yourself or offer it again, and do not plan or start the work it is for until they answer; in a "
            "sentence, say what you suggested.", name, gathers,
            names_allowed ? "The user answers in their next message." : "The user answers in the window." );

        if ( !problems.empty() )
        {
            result += std::format( " Not gathered: {}.", quotedList( problems ) );
        }

        return { { "offer", std::move( offer ) }, { "result", std::move( result ) } };
    }

    nlohmann::json projectForMove( std::string_view arguments, std::span<const nlohmann::json> projects, bool names_allowed )
    {
        const nlohmann::json parsed = nlohmann::json::parse( arguments, nullptr, false );
        const std::string words = trimmed( text( parsed, "project" ) );

        nlohmann::json fit = nlohmann::json::array();

        if ( !words.empty() )
        {
            for ( const nlohmann::json& place : findPlaces( words, projects, nlohmann::json::array() ) )
            {
                if ( place[ "view" ] == "project" )
                {
                    fit.push_back( place );
                }
            }
        }

        if ( fit.size() == 1 )
        {
            return { { "project", { { "id", fit[ 0 ][ "id" ] }, { "name", fit[ 0 ][ "name" ] } } } };
        }

        std::vector<std::string> names;

        if ( fit.empty() )
        {
            for ( const nlohmann::json& project : projects )
            {
                if ( names.size() < kMostNamesListed )
                {
                    names.push_back( text( project, "name" ) );
                }
            }

            if ( !names_allowed || names.empty() )
            {
                return { { "result", "No project is called that, and nothing was moved." } };
            }

            return { { "result", std::format( "No project is called that, and nothing was moved. The projects are: {}.",
                quotedList( names ) ) } };
        }

        if ( !names_allowed )
        {
            return { { "result", std::format( "{} projects fit, and nothing was moved.", fit.size() ) } };
        }

        for ( const nlohmann::json& place : fit )
        {
            names.push_back( text( place, "name" ) );
        }

        return { { "result", std::format( "Several projects fit: {}. Nothing was moved; ask the user which they meant.",
            quotedList( names ) ) } };
    }

    bool filingAllowed( std::span<const nlohmann::json> history )
    {
        return std::ranges::any_of( history, []( const nlohmann::json& event )
            {
                return text( event, "kind" ) == "chat.moved" && text( event, "suggested" ) == "move";
            } );
    }

    nlohmann::json offersIn( const nlohmann::json& messages )
    {
        nlohmann::json offers = nlohmann::json::array();

        if ( !messages.is_array() )
        {
            return offers;
        }

        for ( std::size_t index = 0; index < messages.size(); ++index )
        {
            const nlohmann::json calls = messages[ index ].is_object()
                ? messages[ index ].value( "tool_calls", nlohmann::json( nullptr ) ) : nlohmann::json( nullptr );

            if ( !calls.is_array() )
            {
                continue;
            }

            for ( const nlohmann::json& call : calls )
            {
                if ( !call.is_object() || !call.value( "offer", nlohmann::json( nullptr ) ).is_object() )
                {
                    continue;
                }

                nlohmann::json offer = call[ "offer" ];
                offer[ "message" ] = index;
                offer[ "call_id" ] = call.value( "call_id", nlohmann::json( nullptr ) );
                offers.push_back( std::move( offer ) );
            }
        }

        return offers;
    }

    nlohmann::json pendingOffer( const nlohmann::json& messages )
    {
        if ( !messages.is_array() )
        {
            return nullptr;
        }

        for ( std::size_t index = messages.size(); index-- > 0; )
        {
            const nlohmann::json& message = messages[ index ];

            if ( !message.is_object() || text( message, "role" ) != "assistant" )
            {
                continue;
            }

            const nlohmann::json calls = message.value( "tool_calls", nlohmann::json( nullptr ) );

            if ( !calls.is_array() )
            {
                return nullptr;
            }

            for ( std::size_t call = calls.size(); call-- > 0; )
            {
                const nlohmann::json offer = calls[ call ].is_object() ? calls[ call ].value( "offer", nlohmann::json( nullptr ) ) : nlohmann::json( nullptr );
                const std::string state = text( offer, "state" );

                if ( state == "open" || ( text( offer, "kind" ) == "move" && state == "accepted" && offer.value( "said", false ) == true ) )
                {
                    return { { "message", index }, { "call", call } };
                }
            }

            return nullptr;
        }

        return nullptr;
    }

    nlohmann::json* offerOnCall( nlohmann::json& messages, std::string_view call_id )
    {
        if ( !messages.is_array() || call_id.empty() )
        {
            return nullptr;
        }

        for ( nlohmann::json& message : messages )
        {
            if ( !message.is_object() || !message.contains( "tool_calls" ) || !message[ "tool_calls" ].is_array() )
            {
                continue;
            }

            for ( nlohmann::json& call : message[ "tool_calls" ] )
            {
                if ( call.is_object() && text( call, "call_id" ) == call_id && call.contains( "offer" ) && call[ "offer" ].is_object() )
                {
                    return &call[ "offer" ];
                }
            }
        }

        return nullptr;
    }

    bool projectUndoable( const nlohmann::json& project, std::span<const nlohmann::json> history,
        std::span<const nlohmann::json> projects )
    {
        if ( !project.is_object() || !project.value( "suggested", nlohmann::json( nullptr ) ).is_object() )
        {
            return false;
        }

        const std::string created = text( project, "created" );
        const std::string id = text( project, "id" );

        const bool made_since = std::ranges::any_of( projects, [&]( const nlohmann::json& other )
            {
                return text( other, "id" ) != id && text( other, "created" ) > created;
            } );

        if ( made_since )
        {
            return false;
        }

        return std::ranges::all_of( history, []( const nlohmann::json& event )
            {
                const std::string kind = text( event, "kind" );

                // Mila's offers are not the person's changes; a chat gathered into it is the making itself, and so is the
                // folder that chat held, which became the project's as it was made.
                return kind == "project.made" || kind == "suggestion.made" || kind == "suggestion.declined"
                    || ( ( kind == "chat.moved" || kind == "project.folder_added" ) && text( event, "suggested" ) == "project" );
            } );
    }
}
