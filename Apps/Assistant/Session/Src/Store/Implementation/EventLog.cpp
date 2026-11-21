/**
 * @file EventLog.cpp
 * @brief The bodies of EventLog's members; the interface is Assistant.EventLog.ixx.
 */

module Assistant.EventLog;

import std;
import nlohmann.json;
import Assistant.StoreFile;

namespace Mila::Assistant
{
    EventLog::EventLog( std::filesystem::path path )
        : path_( std::move( path ) )
    {
    }

    void EventLog::append( const nlohmann::json& event ) const
    {
        appendStoreLine( path_, event );
    }

    std::vector<nlohmann::json> EventLog::read() const
    {
        return readStoreLines( path_ );
    }

    void EventLog::forget( std::string_view id ) const
    {
        std::vector<nlohmann::json> events = read();
        bool changed = false;

        const auto names = [id]( const nlohmann::json& event, const char* field )
            {
                return event.contains( field ) && event[ field ].is_string() && event[ field ].get<std::string>() == id;
            };

        for ( nlohmann::json& event : events )
        {
            // A move names the chat it moved, so the chat's title goes with its id; the projects it moved between stay.
            if ( names( event, "conversation" ) )
            {
                event.erase( "conversation" );
                event.erase( "name" );
                changed = true;
            }

            if ( names( event, "project" ) )
            {
                event.erase( "project" );

                if ( !event.contains( "conversation" ) )
                {
                    event.erase( "name" );
                    event.erase( "path" );
                }

                changed = true;
            }

            if ( names( event, "from" ) )
            {
                event.erase( "from" );
                changed = true;
            }
        }

        if ( changed )
        {
            writeStoreLines( path_, events );
        }
    }
}
