/**
 * @file ExposureRecord.cpp
 * @brief Assistant.ExposureRecord's bodies.
 */

module Assistant.ExposureRecord;

import std;
import nlohmann.json;

import Mila;

namespace Mila::Assistant
{
    namespace
    {
        bool readDigits( std::string_view text, std::size_t at, std::size_t digits, unsigned& value )
        {
            if ( at + digits > text.size() )
            {
                return false;
            }

            const char* first = text.data() + at;
            const auto [end, error] = std::from_chars( first, first + digits, value );

            return error == std::errc{} && end == first + digits;
        }

        /// A time as exposureToJson writes it, "2026-11-19T14:10:25.123Z", the fraction optional. Read by hand, since
        /// libc++ has no std::chrono::parse.
        std::optional<std::chrono::sys_time<std::chrono::milliseconds>> parseStoredTime( std::string_view text )
        {
            unsigned year = 0;
            unsigned month = 0;
            unsigned day = 0;
            unsigned hour = 0;
            unsigned minute = 0;
            unsigned second = 0;
            unsigned millisecond = 0;

            const bool fields = text.size() >= 20 && text[ 4 ] == '-' && text[ 7 ] == '-' && text[ 10 ] == 'T'
                && text[ 13 ] == ':' && text[ 16 ] == ':' && text.back() == 'Z'
                && readDigits( text, 0, 4, year ) && readDigits( text, 5, 2, month ) && readDigits( text, 8, 2, day )
                && readDigits( text, 11, 2, hour ) && readDigits( text, 14, 2, minute ) && readDigits( text, 17, 2, second );

            if ( !fields )
            {
                return std::nullopt;
            }

            const std::string_view fraction = text.substr( 19, text.size() - 20 );

            if ( !fraction.empty() && !( fraction.size() == 4 && fraction[ 0 ] == '.' && readDigits( fraction, 1, 3, millisecond ) ) )
            {
                return std::nullopt;
            }

            const std::chrono::year_month_day date{ std::chrono::year( static_cast<int>( year ) ), std::chrono::month( month ), std::chrono::day( day ) };

            if ( !date.ok() || hour > 23 || minute > 59 || second > 59 )
            {
                return std::nullopt;
            }

            return std::chrono::sys_days( date ) + std::chrono::hours( hour ) + std::chrono::minutes( minute )
                + std::chrono::seconds( second ) + std::chrono::milliseconds( millisecond );
        }
    }

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
                    if ( const auto when = parseStoredTime( entry[ "time" ].get<std::string>() ) )
                    {
                        source.when = *when;
                    }
                }
            }

            exposure.push_back( std::move( source ) );
        }

        return exposure;
    }
}
