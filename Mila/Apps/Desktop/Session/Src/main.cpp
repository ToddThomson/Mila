#include <charconv>
#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>
#include <optional>
#include <iostream>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>
#include <functional>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

import Mila;
import Chat.Config;
import Chat.ConfigError;
import Chat.Json;
import Chat.ModelNotFound;
import Chat.SettingsResolution;
import Desktop.ConversationStore;
import Desktop.SessionProcess;

using namespace Mila::ChatApp;
using namespace Mila::DesktopApp;

/**
 * @brief Directory containing the running executable, where the shipped prompts are.
 *
 * The same answer mila-chat's has to give, for the same reason: the window starts this program from wherever it
 * likes, so a prompt resolved against the working directory would be found on one machine and not the next.
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
#endif
    return std::filesystem::current_path();
}

/// @brief A malformed command line: unknown flag, or a flag missing its value.
struct UsageError : std::invalid_argument
{
    using std::invalid_argument::invalid_argument;
};

/**
 * @brief The command line the window starts this program with.
 *
 * A flag that overrides a setting lands in @c overrides under the key it overrides, merged as mila-chat's flags
 * are (ChatConfiguration.md section 9).
 */
struct CommandLine
{
    nlohmann::json overrides = nlohmann::json::object();

    std::optional<std::filesystem::path> settings;
    std::optional<std::filesystem::path> conversations;
    bool model_given = false;
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
            line.overrides[ "model" ] = std::string( requireValue( argc, argv, i, arg ) );
            line.model_given = true;
        }
        else if ( arg == "--context-length" )
        {
            const std::string_view value = requireValue( argc, argv, i, arg );
            std::size_t length = 0;
            const auto [stopped, error] = std::from_chars( value.data(), value.data() + value.size(), length );

            if ( value == "auto" )
                line.overrides[ "context_length" ] = "auto";
            else if ( error == std::errc{} && stopped == value.data() + value.size() && length > 0 )
                line.overrides[ "context_length" ] = length;
            else
                throw UsageError( std::format( "{} expects a positive integer or auto, not '{}'.", arg, value ) );
        }
        else if ( arg == "--device" )
        {
            const std::string_view value = requireValue( argc, argv, i, arg );
            int device = 0;
            const auto [stopped, error] = std::from_chars( value.data(), value.data() + value.size(), device );

            if ( error != std::errc{} || stopped != value.data() + value.size() || device < 0 )
                throw UsageError( std::format( "{} expects a device index of 0 or more, not '{}'.", arg, value ) );

            line.overrides[ "device" ] = device;
        }
        else if ( arg == "--system-prompt" )
        {
            line.overrides[ "system_prompt_path" ] = std::string( requireValue( argc, argv, i, arg ) );
        }
        else if ( arg == "--settings" )
        {
            line.settings = std::filesystem::path( requireValue( argc, argv, i, arg ) );
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
        << "The Mila desktop app's session: requests in as JSON lines on standard input, events out as JSON lines on\n"
        << "standard output (Specifications/ChatDesktop.md section 4). Log records go to standard error. End of input\n"
        << "ends it. Settings resolve as mila-chat's do, from the same files.\n"
        << "\n"
        << "  --model <name>          Load this model, as 'mila-store models' lists it.\n"
        << "  --context-length <n>    Maximum sequence length, or auto.\n"
        << "  --device <n>            Which CUDA device to run on (the CUDA ordinal).\n"
        << "  --system-prompt <name>  A prompt name, or a path to a prompt file.\n"
        << "  --settings <path>       A JSON config file, read as part of this run.\n"
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
    // whole run, and the protocol is written through the buffer it used to hold (ChatDesktop.md 4, Transport).
    std::streambuf* const standard_output = std::cout.rdbuf( std::cerr.rdbuf() );
    std::ostream protocol_out( standard_output );

    auto sink = std::make_shared<Mila::Logging::ConsoleSink>( Mila::Logging::LogLevel::Warning );
    sink->setIncludeSourceLocation( false );
    Mila::initialize( 0, std::move( sink ) );

    try
    {
        ChatConfig config = resolveChatConfig( line.overrides, line.settings, line.model_given, executable_directory() );

        if ( config.detail == DetailLevel::All )
            Mila::Logging::Logger::defaultLogger().setLevel( Mila::Logging::LogLevel::Info );

        SessionProcess process(
            std::move( config ), line.conversations.value_or( ConversationStore::defaultDirectory() ), protocol_out );

        // The same canned reading as mila-chat's probe stub, so a reply can call a tool before the app has tools of
        // its own (ToolCalling.md section 5).
        process.registerTool( "get_weather", []( const std::string& ) -> std::string
        {
            return R"({"temperature_c": 18, "condition": "cloudy"})";
        } );

        return process.run( std::cin );
    }
    catch ( const ConfigError& e )
    {
        std::cerr << e.what() << "\n";

        return 3;
    }
    catch ( const ModelNotFound& e )
    {
        std::cerr << e.what() << "\n";

        return 4;
    }
    catch ( const std::exception& e )
    {
        std::cerr << "Fatal error: " << e.what() << "\n";

        return 1;
    }
}
