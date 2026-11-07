#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <optional>
#include <iostream>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

import Mila;
import Desktop.ConversationStore;
import Desktop.Models;
import Desktop.SessionProcess;
import Desktop.Settings;
import Desktop.SettingsFile;

using namespace Mila::Desktop;

/**
 * @brief Directory containing the running executable, where the shipped prompts are.
 *
 * The window starts this program from wherever it likes, so a prompt resolved against the working directory would be
 * found on one machine and not the next.
 */
static std::filesystem::path executable_directory()
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
#elif defined( __APPLE__ )
    // The path the program was started by, which can be a link; canonical resolves it to the real folder.
    std::uint32_t size = 0;
    _NSGetExecutablePath( nullptr, &size );
    std::string image( size, '\0' );

    if ( _NSGetExecutablePath( image.data(), &size ) == 0 )
    {
        std::error_code unresolved;
        const std::filesystem::path resolved = std::filesystem::canonical( image.c_str(), unresolved );

        if ( !unresolved )
            return resolved.parent_path();
    }
#endif
    return std::filesystem::current_path();
}

/// @brief A malformed command line: unknown flag, or a flag missing its value.
struct UsageError : std::invalid_argument
{
    using std::invalid_argument::invalid_argument;
};

/**
 * @brief The command line the window starts this program with: each flag given stands over the kept settings for
 *        this run.
 */
struct CommandLine
{
    std::optional<std::string> model;

    /// Set with --context-length: a length, or empty for auto.
    std::optional<std::optional<Mila::Dnn::dim_t>> context_length;

    /// Set with --device: the devices the model may run on, as written.
    std::optional<std::string> devices;
    std::optional<std::filesystem::path> conversations;
    bool show_help = false;
};

static std::string_view requireValue( int argc, char* argv[], int& index, std::string_view flag )
{
    if ( index + 1 >= argc )
        throw UsageError( std::format( "{} requires a value.", flag ) );

    return argv[ ++index ];
}

static CommandLine parseCommandLine( int argc, char* argv[] )
{
    CommandLine line;

    for ( int i = 1; i < argc; ++i )
    {
        const std::string_view arg = argv[ i ];

        if ( arg == "--help" || arg == "-h" )
        {
            line.show_help = true;
        }
        else if ( arg == "--model" )
        {
            line.model = std::string( requireValue( argc, argv, i, arg ) );
        }
        else if ( arg == "--context-length" )
        {
            const std::string_view value = requireValue( argc, argv, i, arg );
            Mila::Dnn::dim_t length = 0;
            const auto [stopped, error] = std::from_chars( value.data(), value.data() + value.size(), length );

            if ( value == "auto" )
                line.context_length.emplace( std::nullopt );
            else if ( error == std::errc{} && stopped == value.data() + value.size() && length > 0 )
                line.context_length.emplace( length );
            else
                throw UsageError( std::format( "{} expects a positive integer or auto, not '{}'.", arg, value ) );
        }
        else if ( arg == "--device" )
        {
            const std::string_view value = requireValue( argc, argv, i, arg );

            // A number alone meant a CUDA device only by assumption (Deployment.md 12.8).
            if ( !value.empty() && std::ranges::all_of( value, []( unsigned char c ) { return std::isdigit( c ) != 0; } ) )
                throw UsageError( std::format( "{} no longer takes a number alone: write cuda:{}.", arg, value ) );

            line.devices = std::string( value );
        }
        else if ( arg == "--conversations" )
        {
            line.conversations = std::filesystem::path( requireValue( argc, argv, i, arg ) );
        }
        else
        {
            throw UsageError( std::format( "Unknown option: '{}'. Run with --help for the available flags.", arg ) );
        }
    }

    return line;
}

static void printUsage( const char* program )
{
    std::cerr
        << "Usage: " << program << " [options]\n"
        << "\n"
        << "Mila Desktop's session: requests in as JSON lines on standard input, events out as JSON lines on\n"
        << "standard output (Specifications/MilaDesktop.md section 4). Log records go to standard error. End of input\n"
        << "ends it. Its settings are kept in settings.json beside the conversations folder.\n"
        << "\n"
        << "  --model <name>          Load this model, as 'mila-store models' lists it.\n"
        << "  --context-length <n>    Maximum sequence length, or auto.\n"
        << "  --device <devices>      The devices the model may run on: cuda, cuda:1, metal or a card's\n"
        << "                          identity, several separated by commas. Default: the first GPU.\n"
        << "  --conversations <dir>   Where conversations are stored. Default: "
        << ConversationStore::defaultDirectory().string() << "\n"
        << "  --help, -h              Show this message.\n";
}

int main( int argc, char* argv[] )
{
#ifdef _WIN32
    // The protocol is bytes: a newline is one byte each way, and no code page stands between the window and UTF-8.
    _setmode( _fileno( stdin ), _O_BINARY );
    _setmode( _fileno( stdout ), _O_BINARY );
#endif

    CommandLine line;

    try
    {
        line = parseCommandLine( argc, argv );
    }
    catch ( const UsageError& error )
    {
        std::cerr << error.what() << "\n";

        return 2;
    }

    if ( line.show_help )
    {
        printUsage( argv[ 0 ] );

        return 0;
    }

    // Standard output carries the protocol and nothing else, and not every writer that would share it is ours: the
    // library's console sink writes its records to std::cout. So std::cout is pointed at standard error for the
    // whole run, and the protocol is written through the buffer it used to hold (MilaDesktop.md 4, Transport).
    std::streambuf* const standard_output = std::cout.rdbuf( std::cerr.rdbuf() );
    std::ostream protocol_out( standard_output );

    auto sink = std::make_shared<Mila::Logging::ConsoleSink>( Mila::Logging::LogLevel::Warning );
    sink->setIncludeSourceLocation( false );
    Mila::initialize( 0, std::move( sink ) );

    try
    {
        const std::filesystem::path conversations = line.conversations.value_or( ConversationStore::defaultDirectory() );
        SettingsFile settings_file( conversations.parent_path() / "settings.json" );

        std::vector<std::string> problems;
        Settings settings = settings_file.read( problems );

        if ( line.context_length )
        {
            settings.context_length = *line.context_length;
        }

        // A model named on the command line is the one instruction the run was given, so one that will not open
        // ends it rather than opening a session on something else.
        if ( line.model )
        {
            if ( const auto refusal = Mila::Agents::Agent::check( *line.model ) )
            {
                std::cerr << refusalText( *refusal ) << "\n";

                return 4;
            }

            settings.model = *line.model;
        }

        // Unset, the first GPU this build sees (LayerSplit.md 12.3), or every device when there is none, so the load's
        // refusal says what is missing.
        Mila::Deployment::AllowedDevices devices;

        if ( line.devices )
        {
            try
            {
                devices = Mila::Deployment::AllowedDevices::parse( *line.devices );
            } catch ( const std::invalid_argument& )
            {
                std::cerr << std::format( "--device: '{}' is not a list of devices. Write cuda, cuda:1, metal or a "
                    "card's identity, several separated by commas.\n", *line.devices );

                return 2;
            }
        } else if ( const auto type = Mila::Deployment::ModelHandle::deviceTypeFor( {} ) )
        {
            devices = Mila::Deployment::AllowedDevices( Mila::Dnn::Compute::DeviceId{ *type, 0 } );
        }

        SessionProcess process( std::move( settings ), std::move( settings_file ), std::move( problems ),
            std::move( devices ), conversations, executable_directory() / "Prompts", protocol_out );

        return process.run( std::cin );
    }
    catch ( const std::exception& e )
    {
        std::cerr << "Fatal error: " << e.what() << "\n";

        return 1;
    }
}
