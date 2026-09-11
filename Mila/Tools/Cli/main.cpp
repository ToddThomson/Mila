/**
 * @file main.cpp
 * @brief Entry point for the `mila` command. All work is in Tools.Cli.
 *
 * The image directory is resolved here rather than in the module: finding it is
 * per-platform, and module code carries no #ifdef.
 */

#include <filesystem>
#include <string_view>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

import Tools.Cli;

/**
 * @brief Directory containing the running executable, or empty when unavailable.
 *
 * Empty rather than the working directory: `mila` launches its siblings from
 * here, and a wrong answer would run whatever happens to sit in the directory
 * the user started from. The caller decides what to do with no answer.
 */
static std::filesystem::path imageDirectory()
{
#ifdef _WIN32
    wchar_t buffer[ MAX_PATH ];
    const DWORD length = GetModuleFileNameW( nullptr, buffer, MAX_PATH );

    if ( length > 0 && length < MAX_PATH )
        return std::filesystem::path( buffer ).parent_path();
#elif defined( __linux__ )
    std::error_code unresolved;
    const std::filesystem::path image =
        std::filesystem::read_symlink( "/proc/self/exe", unresolved );

    if ( !unresolved )
        return image.parent_path();
#endif
    return {};
}

int main( int argc, char** argv )
{
    std::vector<std::string_view> arguments;
    arguments.reserve( static_cast<std::size_t>( argc ) );

    for ( int index = 0; index < argc; ++index )
    {
        arguments.emplace_back( argv[ index ] );
    }

    return Mila::Tools::Cli::run( arguments, imageDirectory() );
}
