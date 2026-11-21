/**
 * @file StoreFile.cpp
 * @brief Whole-file writes, records one per line, and the ids the workspace's files are named by.
 */

module Assistant.StoreFile;

import std;
import nlohmann.json;

namespace Mila::Assistant
{
    void writeStoreFile( const std::filesystem::path& path, std::string_view text )
    {
        std::filesystem::create_directories( path.parent_path() );

        std::filesystem::path partial_path = path;
        partial_path += ".partial";

        {
            std::ofstream file( partial_path, std::ios::binary | std::ios::trunc );

            if ( !file )
            {
                throw std::runtime_error( std::format( "cannot write {}", partial_path.string() ) );
            }

            file << text;
        }

        std::filesystem::rename( partial_path, path );
    }

    std::optional<nlohmann::json> readStoreObject( const std::filesystem::path& path )
    {
        std::ifstream file( path, std::ios::binary );

        if ( !file )
        {
            return std::nullopt;
        }

        try
        {
            nlohmann::json record = nlohmann::json::parse( file );

            if ( !record.is_object() )
            {
                return std::nullopt;
            }

            return record;
        }
        catch ( const nlohmann::json::parse_error& )
        {
            return std::nullopt;
        }
    }

    void appendStoreLine( const std::filesystem::path& path, const nlohmann::json& record )
    {
        std::filesystem::create_directories( path.parent_path() );

        std::ofstream file( path, std::ios::binary | std::ios::app );

        if ( !file )
        {
            throw std::runtime_error( std::format( "cannot write {}", path.string() ) );
        }

        file << record.dump() << '\n';
    }

    std::vector<nlohmann::json> readStoreLines( const std::filesystem::path& path )
    {
        std::vector<nlohmann::json> records;
        std::ifstream file( path, std::ios::binary );
        std::string line;

        while ( std::getline( file, line ) )
        {
            nlohmann::json record = nlohmann::json::parse( line, nullptr, false );

            if ( record.is_object() )
            {
                records.push_back( std::move( record ) );
            }
        }

        return records;
    }

    void writeStoreLines( const std::filesystem::path& path, const std::vector<nlohmann::json>& records )
    {
        std::string text;

        for ( const nlohmann::json& record : records )
        {
            text += record.dump();
            text += '\n';
        }

        writeStoreFile( path, text );
    }

    std::string newStoreIdentifier( std::string_view kind )
    {
        const auto now = std::chrono::floor<std::chrono::seconds>( std::chrono::system_clock::now() );

        std::random_device device;
        std::uniform_int_distribution<std::uint32_t> distribution( 0, 0xFFFFFF );

        return std::format( "{}-{:%Y%m%d%H%M%S}-{:06x}", kind, now, distribution( device ) );
    }

    bool isStoreIdentifier( std::string_view id )
    {
        return !id.empty() && id.size() <= 64
            && std::ranges::all_of( id, []( char c )
                {
                    return ( c >= 'a' && c <= 'z' ) || ( c >= '0' && c <= '9' ) || c == '-';
                } );
    }
}
