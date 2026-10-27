/**
 * @file GrantPath.ixx
 * @brief The one form of path a grant accepts, checked as text before anything is opened.
 */

export module AI.GrantPath;

import std;

namespace Mila::AI
{
    namespace
    {
        bool isAsciiLetter( char character ) noexcept
        {
            return ( character >= 'A' && character <= 'Z' ) || ( character >= 'a' && character <= 'z' );
        }

        std::string upperAscii( std::string_view text )
        {
            std::string upper( text );

            for ( char& character : upper )
            {
                if ( character >= 'a' && character <= 'z' )
                {
                    character = static_cast<char>( character - 'a' + 'A' );
                }
            }

            return upper;
        }

        /// Whether a name opens a device rather than a file: Microsoft's "Naming Files, Paths, and Namespaces" lists
        /// CON, PRN, AUX, NUL, COM1-9, LPT1-9 and the superscript-digit forms, alone or before any extension.
        /// CONIN$ and CONOUT$ are the console's own, which CreateFile also opens as devices.
        bool isDeviceName( std::string_view name )
        {
            std::string_view base = name.substr( 0, name.find( '.' ) );

            // Windows drops the spaces that end a base name before it looks for a device, so "NUL .txt" is NUL.
            while ( base.ends_with( ' ' ) )
            {
                base.remove_suffix( 1 );
            }

            const std::string upper = upperAscii( base );

            if ( upper == "CON" || upper == "PRN" || upper == "AUX" || upper == "NUL" || upper == "CONIN$" || upper == "CONOUT$" )
            {
                return true;
            }

            if ( !upper.starts_with( "COM" ) && !upper.starts_with( "LPT" ) )
            {
                return false;
            }

            const std::string_view digit = std::string_view( upper ).substr( 3 );

            return ( digit.size() == 1 && digit[ 0 ] >= '1' && digit[ 0 ] <= '9' )
                || digit == "\xC2\xB9" || digit == "\xC2\xB2" || digit == "\xC2\xB3";
        }

        void checkName( std::string_view path, std::string_view name )
        {
            if ( name == ".." )
            {
                throw std::invalid_argument( std::format(
                    "'{}' leads up out of a folder with '..'; give the path from the project folder down", path ) );
            }

            for ( const char character : name )
            {
                const auto code = static_cast<unsigned char>( character );

                if ( code < 32 || std::string_view( "<>:\"|?*" ).find( character ) != std::string_view::npos )
                {
                    throw std::invalid_argument( std::format(
                        "'{}' has a character no file name may hold (< > : \" | ? * or a control character)", path ) );
                }
            }

            if ( name.ends_with( ' ' ) || name.ends_with( '.' ) )
            {
                throw std::invalid_argument( std::format( "in '{}', the name '{}' ends with a space or a period", path, name ) );
            }

            if ( isDeviceName( name ) )
            {
                throw std::invalid_argument( std::format( "in '{}', '{}' is the name of a device, not a file", path, name ) );
            }
        }
    }

    /**
     * @brief The names a path is made of, refused unless it is in the one form a grant accepts (ToolCalling.md 3.4).
     *
     * Accepted: a drive-letter path, whose first name is the drive ("D:"), or a path relative to the granted folder,
     * with / or \ between names. "." stands for the folder it is in and adds no name, so "." alone is the folder
     * itself. Every other name is an ordinary file or folder name, so nothing the operating system would read as
     * something else gets through: no network or device path (\\server, \\?\, \\.\), no path from the root of no
     * drive, no drive-relative "C:name", no "..", no ":" (an alternate stream), no reserved or control character,
     * no name ending in a space or period, and no device name such as NUL or COM1, with an extension or without.
     *
     * Passing is not permission: what a path leads to is decided when the grant opens it.
     *
     * @param path In UTF-8.
     * @throws std::invalid_argument naming what the path holds that a grant refuses.
     */
    export std::vector<std::string> grantPathNames( std::string_view path )
    {
        if ( path.empty() )
        {
            throw std::invalid_argument( "the path is empty" );
        }

        std::string text( path );
        std::ranges::replace( text, '\\', '/' );

        std::vector<std::string> names;
        std::string_view rest = text;

        if ( rest.starts_with( "//" ) )
        {
            throw std::invalid_argument( std::format( "'{}' is a network or device path, which a grant never opens", path ) );
        }

        if ( rest.starts_with( '/' ) )
        {
            throw std::invalid_argument( std::format(
                "'{}' starts at the root of no drive; give a drive letter, or a path from the project folder", path ) );
        }

        if ( rest.size() >= 2 && rest[ 1 ] == ':' && isAsciiLetter( rest[ 0 ] ) )
        {
            if ( rest.size() == 2 || rest[ 2 ] != '/' )
            {
                throw std::invalid_argument( std::format(
                    "'{}' names a drive but not where on it; write the drive as \"{}:\\\"", path, rest[ 0 ] ) );
            }

            names.push_back( upperAscii( rest.substr( 0, 2 ) ) );
            rest.remove_prefix( 3 );
        }

        for ( const auto part : std::views::split( rest, '/' ) )
        {
            const std::string_view name( part.begin(), part.end() );

            if ( name.empty() || name == "." )
            {
                continue;
            }

            checkName( path, name );
            names.emplace_back( name );
        }

        return names;
    }
}
