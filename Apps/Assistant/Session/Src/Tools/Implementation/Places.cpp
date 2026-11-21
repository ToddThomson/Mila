/**
 * @file Places.cpp
 * @brief Matching a person's words against the workspace's views, projects and chats, and saying what a place shows.
 */

module Assistant.Places;

import std;
import nlohmann.json;

namespace Mila::Assistant
{
    namespace
    {
        // Words a request carries around a name that no name needs to hold: *open the calculator project*.
        constexpr std::array<std::string_view, 7> kFillerWords = { "the", "a", "an", "my", "our", "project", "chat" };

        // A reply that ended for one of these needs the person (AssistantInterface.md 4.3), as the page's needsYou says.
        constexpr std::array<std::string_view, 3> kAttentionEndings = { "length", "context", "error" };

        std::string folded( std::string_view text )
        {
            std::string lower( text );

            // ASCII only: a name in another script matches as written.
            std::ranges::transform( lower, lower.begin(), []( unsigned char character )
                {
                    return static_cast<char>( std::tolower( character ) );
                } );

            return lower;
        }

        std::string trimmed( std::string_view text )
        {
            const auto first = text.find_first_not_of( " \t\r\n.?!" );

            if ( first == std::string_view::npos )
            {
                return {};
            }

            const auto last = text.find_last_not_of( " \t\r\n.?!" );

            return std::string( text.substr( first, last - first + 1 ) );
        }

        std::vector<std::string> wordsOf( std::string_view text )
        {
            std::vector<std::string> words;
            std::string word;

            for ( const char character : folded( text ) )
            {
                if ( std::isalnum( static_cast<unsigned char>( character ) ) != 0 || static_cast<unsigned char>( character ) >= 0x80 )
                {
                    word += character;
                }
                else if ( !word.empty() )
                {
                    words.push_back( std::move( word ) );
                    word.clear();
                }
            }

            if ( !word.empty() )
            {
                words.push_back( std::move( word ) );
            }

            return words;
        }

        /// The view a request names by its own name, or empty.
        std::string viewNamed( std::string_view words )
        {
            std::string name = folded( trimmed( words ) );

            for ( const std::string_view leading : { "all ", "my ", "the ", "your " } )
            {
                if ( name.starts_with( leading ) )
                {
                    name.erase( 0, leading.size() );
                }
            }

            static const std::array<std::pair<std::string_view, std::string_view>, 8> kViews = { {
                { "workspace", "workspace" }, { "home", "workspace" }, { "projects", "projects" }, { "chats", "chats" },
                { "conversations", "chats" }, { "activity", "activity" }, { "reach", "reach" },
                { "what mila can reach", "reach" } } };

            for ( const auto& [ said, view ] : kViews )
            {
                if ( name == said )
                {
                    return std::string( view );
                }
            }

            return {};
        }

        bool holdsEvery( std::string_view name, std::span<const std::string> words )
        {
            if ( words.empty() )
            {
                return false;
            }

            const std::vector<std::string> held = wordsOf( name );

            return std::ranges::all_of( words, [&held]( const std::string& word )
                {
                    return std::ranges::any_of( held, [&word]( const std::string& part ) { return part.starts_with( word ); } );
                } );
        }

        std::string text( const nlohmann::json& object, std::string_view key )
        {
            const auto value = object.find( key );

            return value != object.end() && value->is_string() ? value->get<std::string>() : std::string{};
        }

        /// The chats newest first, as the lists show them.
        std::vector<const nlohmann::json*> chatsNewestFirst( const nlohmann::json& chats )
        {
            std::vector<const nlohmann::json*> ordered;

            if ( chats.is_array() )
            {
                for ( const nlohmann::json& chat : chats )
                {
                    ordered.push_back( &chat );
                }
            }

            std::ranges::stable_sort( ordered, std::greater<>{}, []( const nlohmann::json* chat ) { return text( *chat, "updated" ); } );

            return ordered;
        }

        std::vector<const nlohmann::json*> projectsNewestFirst( std::span<const nlohmann::json> projects )
        {
            std::vector<const nlohmann::json*> ordered;

            for ( const nlohmann::json& project : projects )
            {
                ordered.push_back( &project );
            }

            std::ranges::stable_sort( ordered, std::greater<>{}, []( const nlohmann::json* project ) { return text( *project, "updated" ); } );

            return ordered;
        }

        nlohmann::json placeOf( std::string_view view, std::string_view id, std::string_view name )
        {
            return { { "view", view }, { "id", id.empty() ? nlohmann::json( nullptr ) : nlohmann::json( id ) }, { "name", name } };
        }

        std::string projectName( std::span<const nlohmann::json> projects, std::string_view id )
        {
            for ( const nlohmann::json& project : projects )
            {
                if ( text( project, "id" ) == id )
                {
                    return text( project, "name" );
                }
            }

            return {};
        }

        bool projectWaits( std::string_view id, const nlohmann::json& chats, std::string_view open_chat )
        {
            for ( const nlohmann::json* chat : chatsNewestFirst( chats ) )
            {
                if ( text( *chat, "project" ) == id && text( *chat, "id" ) != open_chat && waitsOnPerson( *chat ) )
                {
                    return true;
                }
            }

            return false;
        }

        std::size_t chatsIn( std::string_view id, const nlohmann::json& chats )
        {
            std::size_t count = 0;

            for ( const nlohmann::json* chat : chatsNewestFirst( chats ) )
            {
                count += text( *chat, "project" ) == id ? 1 : 0;
            }

            return count;
        }

        std::string plural( std::size_t count, std::string_view one, std::string_view many )
        {
            return std::format( "{} {}", count, count == 1 ? one : many );
        }

        /// Each chat as a line: its title, its project, when it changed.
        std::string chatLines( const std::vector<const nlohmann::json*>& chats, std::span<const nlohmann::json> projects )
        {
            std::string lines;

            for ( std::size_t index = 0; index < chats.size() && index < kMostNamesListed; ++index )
            {
                const std::string project = projectName( projects, text( *chats[ index ], "project" ) );
                const std::string updated = text( *chats[ index ], "updated" ).substr( 0, 10 );

                lines += std::format( "\n- {}{}{}{}", text( *chats[ index ], "title" ),
                    project.empty() ? "" : ", in ", project, updated.empty() ? "" : ", changed " + updated );

                if ( waitsOnPerson( *chats[ index ] ) )
                {
                    lines += ", waiting on the user";
                }
            }

            if ( chats.size() > kMostNamesListed )
            {
                lines += std::format( "\n...and {} more.", chats.size() - kMostNamesListed );
            }

            return lines;
        }
    }

    bool waitsOnPerson( const nlohmann::json& chat )
    {
        const auto progress = chat.find( "progress" );

        if ( progress != chat.end() && progress->is_object() && progress->contains( "current" ) && !( *progress )[ "current" ].is_null() )
        {
            return true;
        }

        const std::string finish = text( chat, "finish" );

        return std::ranges::find( kAttentionEndings, finish ) != kAttentionEndings.end();
    }

    nlohmann::json findPlaces( std::string_view words, std::span<const nlohmann::json> projects, const nlohmann::json& chats,
        std::string_view open_chat )
    {
        nlohmann::json places = nlohmann::json::array();

        if ( const std::string view = viewNamed( words ); !view.empty() )
        {
            places.push_back( placeOf( view, {}, {} ) );

            return places;
        }

        const std::string exact = folded( trimmed( words ) );

        if ( exact.empty() )
        {
            return places;
        }

        const std::vector<const nlohmann::json*> ordered_projects = projectsNewestFirst( projects );
        std::vector<const nlohmann::json*> ordered_chats = chatsNewestFirst( chats );

        std::erase_if( ordered_chats, [open_chat]( const nlohmann::json* chat ) { return !open_chat.empty() && text( *chat, "id" ) == open_chat; } );

        // An exact name is the place, projects before chats.
        for ( const nlohmann::json* project : ordered_projects )
        {
            if ( folded( text( *project, "name" ) ) == exact )
            {
                places.push_back( placeOf( "project", text( *project, "id" ), text( *project, "name" ) ) );

                return places;
            }
        }

        for ( const nlohmann::json* chat : ordered_chats )
        {
            if ( folded( text( *chat, "title" ) ) == exact )
            {
                places.push_back( placeOf( "chat", text( *chat, "id" ), text( *chat, "title" ) ) );

                return places;
            }
        }

        std::vector<std::string> named;

        for ( std::string& word : wordsOf( words ) )
        {
            if ( std::ranges::find( kFillerWords, word ) == kFillerWords.end() )
            {
                named.push_back( std::move( word ) );
            }
        }

        for ( const nlohmann::json* project : ordered_projects )
        {
            if ( places.size() < kMostPlacesOffered && holdsEvery( text( *project, "name" ), named ) )
            {
                places.push_back( placeOf( "project", text( *project, "id" ), text( *project, "name" ) ) );
            }
        }

        for ( const nlohmann::json* chat : ordered_chats )
        {
            if ( places.size() < kMostPlacesOffered && holdsEvery( text( *chat, "title" ), named ) )
            {
                places.push_back( placeOf( "chat", text( *chat, "id" ), text( *chat, "title" ) ) );
            }
        }

        return places;
    }

    std::string describePlace( const nlohmann::json& place, std::span<const nlohmann::json> projects,
        const nlohmann::json& chats, std::string_view open_chat, bool names_allowed )
    {
        const std::string view = text( place, "view" );
        const std::vector<const nlohmann::json*> all_chats = chatsNewestFirst( chats );

        if ( view == "workspace" )
        {
            std::size_t waiting = 0;

            for ( const nlohmann::json& project : projects )
            {
                waiting += projectWaits( text( project, "id" ), chats, open_chat ) ? 1 : 0;
            }

            return std::format( "Opened the user's workspace: {} ({} waiting on the user) and {}.",
                plural( projects.size(), "project", "projects" ), waiting, plural( all_chats.size(), "chat", "chats" ) );
        }

        if ( view == "projects" )
        {
            std::string result = std::format( "Opened All projects: {}.", plural( projects.size(), "project", "projects" ) );

            if ( !names_allowed )
            {
                return result + " The user can see them.";
            }

            std::size_t listed = 0;

            for ( const nlohmann::json* project : projectsNewestFirst( projects ) )
            {
                if ( listed++ == kMostNamesListed )
                {
                    result += std::format( "\n...and {} more.", projects.size() - kMostNamesListed );

                    break;
                }

                const std::string id = text( *project, "id" );
                const std::string purpose = text( *project, "purpose" );

                result += std::format( "\n- {}: {}, {}{}", text( *project, "name" ),
                    projectWaits( id, chats, open_chat ) ? "waiting on the user" : "nothing waiting",
                    plural( chatsIn( id, chats ), "chat", "chats" ), purpose.empty() ? "" : ". Purpose: " + purpose );
            }

            return result;
        }

        if ( view == "chats" )
        {
            const std::string result = std::format( "Opened All chats: {}.", plural( all_chats.size(), "chat", "chats" ) );

            return names_allowed ? result + chatLines( all_chats, projects ) : result + " The user can see them.";
        }

        if ( view == "activity" )
        {
            return "Opened Activity, which shows what happened in the user's workspace, the newest first.";
        }

        if ( view == "reach" )
        {
            return "Opened What Mila can reach, which shows the folders each chat holds and what the user allowed.";
        }

        if ( view == "project" )
        {
            const std::string id = text( place, "id" );
            std::vector<const nlohmann::json*> in_project;

            for ( const nlohmann::json* chat : all_chats )
            {
                if ( text( *chat, "project" ) == id )
                {
                    in_project.push_back( chat );
                }
            }

            std::string purpose;

            for ( const nlohmann::json& project : projects )
            {
                if ( text( project, "id" ) == id )
                {
                    purpose = text( project, "purpose" );
                }
            }

            std::string result = std::format( "Opened the project {}{}: {}, {}.", text( place, "name" ),
                purpose.empty() ? "" : " (" + purpose + ")", plural( in_project.size(), "chat", "chats" ),
                projectWaits( id, chats, open_chat ) ? "waiting on the user" : "nothing waiting on the user" );

            return names_allowed ? result + chatLines( in_project, projects ) : result;
        }

        if ( view == "chat" )
        {
            return text( place, "id" ) == open_chat
                ? std::format( "The chat {} is this one, and is open.", text( place, "name" ) )
                : std::format( "The chat {} opens for the user when this reply ends.", text( place, "name" ) );
        }

        return "Nothing was opened.";
    }

    std::string severalPlaces( const nlohmann::json& places )
    {
        std::string result = std::format( "{} places fit, and none was opened. Ask the user which they meant:", places.size() );

        for ( const nlohmann::json& place : places )
        {
            result += std::format( "\n- the {} {}", text( place, "view" ), text( place, "name" ) );
        }

        return result;
    }

    std::string wentLine( const nlohmann::json& place )
    {
        const std::string view = text( place, "view" );

        if ( view == "project" || view == "chat" )
        {
            return "Opened " + text( place, "name" );
        }

        static const std::map<std::string, std::string, std::less<>> kViews = {
            { "workspace", "Opened your workspace" }, { "projects", "Opened All projects" }, { "chats", "Opened All chats" },
            { "activity", "Opened Activity" }, { "reach", "Opened What Mila can reach" } };

        const auto line = kViews.find( view );

        return line == kViews.end() ? std::string( "Opened" ) : line->second;
    }
}
