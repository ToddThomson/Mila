/**
 * @file MilaLicense.cpp
 * @brief Reading the licence the package carries.
 */

module Assistant.MilaLicense;

import std;
import Mila;

namespace Mila::Assistant
{
    std::expected<MilaLicense, std::string> MilaLicense::read( const std::filesystem::path& path )
    {
        // Without its licence the package is incomplete, and nothing may be used until one is accepted.
        constexpr std::string_view missing =
            "The licence that comes with Mila is missing, so Mila cannot start. Install Mila again.";

        std::ifstream file( path, std::ios::binary );

        if ( !file )
        {
            return std::unexpected( std::string( missing ) );
        }

        std::string text( ( std::istreambuf_iterator<char>( file ) ), std::istreambuf_iterator<char>() );

        if ( file.bad() || text.empty() )
        {
            return std::unexpected( std::string( missing ) );
        }

        std::string digest = Mila::Distribution::sha256Hex( text.data(), text.size() );

        return MilaLicense{ std::move( text ), std::move( digest ) };
    }
}
