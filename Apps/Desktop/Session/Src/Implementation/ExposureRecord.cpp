/**
 * @file ExposureRecord.cpp
 * @brief Desktop.ExposureRecord's bodies.
 */

module Desktop.ExposureRecord;

import std;
import nlohmann.json;

import Mila;

namespace Mila::Desktop
{
    nlohmann::json exposureToJson( std::span<const Mila::Agents::ExposureSource> exposure )
    {
        nlohmann::json sources = nlohmann::json::array();

        for ( const Mila::Agents::ExposureSource& source : exposure )
        {
            sources.push_back( {
                { "tool", source.tool },
                { "arguments", source.arguments },
                { "time", source.when
                    ? nlohmann::json( std::format( "{:%FT%TZ}", std::chrono::floor<std::chrono::milliseconds>( *source.when ) ) )
                    : nlohmann::json( nullptr ) } } );
        }

        return sources;
    }

    std::vector<Mila::Agents::ExposureSource> exposureFromJson( const nlohmann::json& stored )
    {
        std::vector<Mila::Agents::ExposureSource> exposure;

        if ( stored.is_null() )
        {
            return exposure;
        }

        if ( !stored.is_array() )
        {
            exposure.push_back( { "unknown", {}, std::nullopt } );

            return exposure;
        }

        for ( const nlohmann::json& entry : stored )
        {
            Mila::Agents::ExposureSource source{ "unknown", {}, std::nullopt };

            if ( entry.is_object() )
            {
                if ( entry.contains( "tool" ) && entry[ "tool" ].is_string() )
                {
                    source.tool = entry[ "tool" ].get<std::string>();
                }

                if ( entry.contains( "arguments" ) && entry[ "arguments" ].is_string() )
                {
                    source.arguments = entry[ "arguments" ].get<std::string>();
                }

                if ( entry.contains( "time" ) && entry[ "time" ].is_string() )
                {
                    std::istringstream text( entry[ "time" ].get<std::string>() );
                    std::chrono::sys_time<std::chrono::milliseconds> when;

                    if ( text >> std::chrono::parse( "%FT%TZ", when ) )
                    {
                        source.when = when;
                    }
                }
            }

            exposure.push_back( std::move( source ) );
        }

        return exposure;
    }
}
