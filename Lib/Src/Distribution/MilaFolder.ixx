/**
 * @file MilaFolder.ixx
 * @brief The person's Mila folder, where each platform keeps an application's data, and the area each Mila program
 *        keeps in it: `Store` for the model store, `Assistant`, `Chat`, `Server`.
 */

export module Distribution.MilaFolder;

import std;

import Distribution.Environment;

namespace Mila::Distribution
{
    /**
     * @brief The person's Mila folder: `%LOCALAPPDATA%\Mila` on Windows, `~/Library/Application Support/Mila` on
     *        macOS, `$XDG_DATA_HOME/mila`, else `~/.local/share/mila`, on Linux.
     *
     * The system's temporary folder holds it when none of those variables answers, so a program always has a place.
     */
    export std::filesystem::path resolveMilaFolder()
    {
#if defined( _WIN32 )
        if ( const auto local_app_data = readEnvironmentVariable( "LOCALAPPDATA" ) )
        {
            return std::filesystem::path( *local_app_data ) / "Mila";
        }
#elif defined( __APPLE__ )
        if ( const auto home = readEnvironmentVariable( "HOME" ) )
        {
            return std::filesystem::path( *home ) / "Library" / "Application Support" / "Mila";
        }
#else
        if ( const auto xdg_data = readEnvironmentVariable( "XDG_DATA_HOME" ) )
        {
            return std::filesystem::path( *xdg_data ) / "mila";
        }

        if ( const auto home = readEnvironmentVariable( "HOME" ) )
        {
            return std::filesystem::path( *home ) / ".local" / "share" / "mila";
        }
#endif

        return std::filesystem::temp_directory_path() / "mila";
    }

    /**
     * @brief One area of a folder laid out as the Mila folder is, named as the platform names its own folders:
     *        `Assistant` on Windows and macOS, `assistant` on Linux.
     *
     * Given the folder, so a program whose configuration lives elsewhere (Chat's settings roam on Windows) lays out
     * its area there the same way.
     */
    export std::filesystem::path milaArea( const std::filesystem::path& folder, std::string_view area )
    {
#if defined( _WIN32 ) || defined( __APPLE__ )
        return folder / std::string( area );
#else
        std::string lower( area );
        const auto toLower = []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); };
        std::ranges::transform( lower, lower.begin(), toLower );

        return folder / lower;
#endif
    }

    /// One program's area of the person's Mila folder: `resolveMilaFolder()` and `milaArea()` together.
    export std::filesystem::path resolveMilaArea( std::string_view area )
    {
        return milaArea( resolveMilaFolder(), area );
    }
}
