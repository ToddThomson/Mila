#include <charconv>
#include <cstddef>
#include <string>
#include <string_view>
#include <optional>
#include <iostream>
#include <fstream>
#include <iterator>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>
#include <functional>
#include <source_location>

#ifdef _WIN32
#include <windows.h>
#endif

import Mila;
import Chat.ConfigError;
import Chat.ConsoleFrontEnd;
import Chat.ModelCatalog;
import Chat.ModelNotFound;
import Chat.Session;
import Chat.SettingsResolution;
import Chat.Renderer;
import nlohmann.json;

using namespace Mila::ChatApp;

namespace
{
    /**
     * @brief A ConsoleSink that shares the console with the spinner instead of overwriting it.
     *
     * The stock sink writes the moment a record arrives, which during a model load is mid-frame
     * of a spinner that redraws every 80 ms. What an evaluator saw on the published quick start
     * was "Loading Llama-3.2-3B-Instruct-fp415:49:42.425 [WARN ] BpeTokenizer.ixx:378:...".
     *
     * Source location is off for the same audience reason: `file:line:function` is for whoever
     * is fixing Mila, not for someone deciding whether to try it. Timestamps stay -- they say
     * whether a warning belongs to the step still running.
     */
    class SpinnerAwareConsoleSink final : public Mila::Logging::ConsoleSink
    {
    public:

        explicit SpinnerAwareConsoleSink( Mila::Logging::LogLevel level )
            : ConsoleSink( level )
        {
            setIncludeSourceLocation( false );
        }

        void log_trace( std::string_view message, const std::source_location& location ) override
        {
            around( Mila::Logging::LogLevel::Trace,
                [&] { ConsoleSink::log_trace( message, location ); } );
        }

        void log_debug( std::string_view message, const std::source_location& location ) override
        {
            around( Mila::Logging::LogLevel::Debug,
                [&] { ConsoleSink::log_debug( message, location ); } );
        }

        void log_info( std::string_view message, const std::source_location& location ) override
        {
            around( Mila::Logging::LogLevel::Info,
                [&] { ConsoleSink::log_info( message, location ); } );
        }

        void log_warning( std::string_view message, const std::source_location& location ) override
        {
            around( Mila::Logging::LogLevel::Warning,
                [&] { ConsoleSink::log_warning( message, location ); } );
        }

        void log_error( std::string_view message, const std::source_location& location ) override
        {
            around( Mila::Logging::LogLevel::Error,
                [&] { ConsoleSink::log_error( message, location ); } );
        }

        void log_critical( std::string_view message, const std::source_location& location ) override
        {
            around( Mila::Logging::LogLevel::Critical,
                [&] { ConsoleSink::log_critical( message, location ); } );
        }

        void log( std::string_view message, Mila::Logging::LogLevel level,
            const std::source_location& location ) override
        {
            around( level, [&] { ConsoleSink::log( message, level, location ); } );
        }

    private:

        /**
         * @brief Erase the spinner's line only for a record that will actually be written.
         *
         * The base sink drops records below its level, but it drops them AFTER being called --
         * so routing every record through the erase blanked the spinner for each discarded one,
         * leaving the line empty until the next frame up to 80 ms later. Measured as runs of
         * bare CR-plus-erase in the raw output. Filtering here means a suppressed record costs
         * nothing and never touches the console.
         */
        void around( Mila::Logging::LogLevel level, const std::function<void()>& emit )
        {
            if ( !isEnabled( level ) )
                return;

            ConsoleRenderer::writeAroundSpinner( emit );
        }
    };
}

/**
 * @brief Directory containing the running executable.
 *
 * The POST_BUILD step copies Prompts/ next to the executable, and a config may be found there
 * too, so resolving both against this directory makes them independent of the process working
 * directory -- which the IDE and debugger set to the source tree or the repo root rather than
 * the binary output directory.
 *
 * Every supported platform must answer this for itself. Falling through to the working directory
 * does not implement the guarantee above, it silently drops it -- which is how a container ended
 * up loading no config at all. See Specifications/ChatConfiguration.md sections 2 and 8.
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

/**
 * @brief A malformed command line: unknown flag, or a flag missing its value.
 *
 * Separate from a bad configuration VALUE so the two can exit with different codes -- a
 * caller can tell "you typed it wrong" from "the file says something impossible".
 */
struct UsageError : std::invalid_argument
{
    using std::invalid_argument::invalid_argument;
};

/**
 * @brief The command line, parsed once, before anything else happens.
 *
 * Flags are the last word over every configuration file -- see ChatConfiguration.md section 9.
 * They are collected here rather than applied as they are read, so that the order they were
 * typed in cannot change what they mean.
 *
 * A flag that overrides a SETTING lands in @c overrides under the key it overrides, and is merged
 * like any other layer rather than assigned to a ChatConfig member. Adding one is then a branch
 * here and nothing else: no precedence rule of its own, and origin recording for free. The plain
 * members below are the other kind of flag -- what to do this run, not how to be configured.
 */
struct CommandLine
{
    nlohmann::json overrides = nlohmann::json::object();

    std::filesystem::path settings;
    bool settings_given = false;

    /// Whether --model was typed. The name itself lives in @c overrides; this says it was an
    /// instruction rather than something a file happened to mention, which is what makes a name
    /// that does not resolve fatal instead of a session that opens on something else.
    bool model_given = false;

    std::string prompt;
    bool one_shot = false;

    bool json_output = false;
    bool show_version = false;
    bool show_help = false;
};

static std::string_view requireValue( int argc, char* argv[], int& index, std::string_view flag )
{
    if ( index + 1 >= argc )
        throw UsageError( std::format( "{} requires a value.", flag ) );

    return argv[ ++index ];
}

static std::size_t requireCount( std::string_view value, std::string_view flag )
{
    std::size_t parsed = 0;
    const char* const end = value.data() + value.size();
    const auto [stopped, error] = std::from_chars( value.data(), end, parsed );

    if ( error != std::errc{} || stopped != end || parsed == 0 )
        throw UsageError( std::format(
            "{} expects a positive integer or auto, not '{}'.", flag, value ) );

    return parsed;
}

/// Like requireCount, but zero is a value rather than an error: device 0 is the first card, and
/// on a single-card machine it is the only one.
static int requireIndex( std::string_view value, std::string_view flag )
{
    int parsed = 0;
    const char* const end = value.data() + value.size();
    const auto [stopped, error] = std::from_chars( value.data(), end, parsed );

    if ( error != std::errc{} || stopped != end || parsed < 0 )
        throw UsageError( std::format(
            "{} expects a device index of 0 or more, not '{}'.", flag, value ) );

    return parsed;
}

/**
 * @brief The whole of a file, as one prompt.
 *
 * A command line holds about 32 KB on Windows, which is a fraction of what a long-context model
 * accepts, so a prompt that exercises one cannot be typed as an argument at all.
 *
 * @throws UsageError when the file cannot be read, or holds nothing.
 */
static std::string readPromptFile( std::string_view path )
{
    std::ifstream file( std::filesystem::path( path ), std::ios::binary );

    if ( !file )
        throw UsageError( std::format( "--prompt-file: cannot read '{}'.", path ) );

    const std::istreambuf_iterator<char> first( file );
    const std::istreambuf_iterator<char> last;
    std::string prompt( first, last );

    if ( prompt.empty() )
        throw UsageError( std::format( "--prompt-file: '{}' is empty.", path ) );

    return prompt;
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
        else if ( arg == "--version" )
        {
            line.show_version = true;
        }
        else if ( arg == "--settings" )
        {
            line.settings = requireValue( argc, argv, i, arg );
            line.settings_given = true;
        }
        else if ( arg == "--model" )
        {
            line.overrides[ "model" ] = std::string( requireValue( argc, argv, i, arg ) );
            line.model_given = true;
        }
        else if ( arg == "--context-length" )
        {
            // "auto" because layer 6 must be able to express everything layer 1 can. A flag set
            // that cannot say what the compiled default says is an incomplete override layer.
            const std::string_view value = requireValue( argc, argv, i, arg );

            if ( value == "auto" )
                line.overrides[ "context_length" ] = "auto";
            else
                line.overrides[ "context_length" ] = requireCount( value, arg );
        }
        else if ( arg == "--device" )
        {
            // The CUDA ordinal, which is NOT nvidia-smi's index -- the two orders differ on a
            // mixed-generation rig, and picking the wrong one is an out-of-memory abort rather
            // than a message naming the card.
            line.overrides[ "device" ] = requireIndex(
                requireValue( argc, argv, i, arg ), arg );
        }
        else if ( arg == "--system-prompt" )
        {
            line.overrides[ "system_prompt_path" ] =
                std::string( requireValue( argc, argv, i, arg ) );
        }
        else if ( arg == "-p" )
        {
            if ( line.one_shot )
                throw UsageError( "-p and --prompt-file both name the prompt. Use one." );

            line.prompt = requireValue( argc, argv, i, arg );
            line.one_shot = true;
        }
        else if ( arg == "--prompt-file" )
        {
            if ( line.one_shot )
                throw UsageError( "-p and --prompt-file both name the prompt. Use one." );

            line.prompt = readPromptFile( requireValue( argc, argv, i, arg ) );
            line.one_shot = true;
        }
        else if ( arg == "--output-format" )
        {
            const std::string_view value = requireValue( argc, argv, i, arg );

            if ( value == "json" )
                line.json_output = true;
            else if ( value != "text" )
                throw UsageError( std::format(
                    "Unknown output format '{}'. Use text or json.", value ) );
        }
        else
        {
            throw UsageError( std::format(
                "Unknown option: '{}'. Run with --help for the available flags.", arg ) );
        }
    }

    // Rejected rather than ignored: a caller that asked for JSON and got a painted session
    // would discover the mistake by parsing a welcome box.
    if ( line.json_output && !line.one_shot )
        throw UsageError( "--output-format applies to -p only; an interactive session has no "
                          "single response to format." );

    return line;
}

static void printUsage( const char* prog_name )
{
    std::cerr
        << "Usage: " << prog_name << " [options]\n"
        << "\n"
        << "  --prompt-file <path>   Answer the prompt held in this file and exit, for a prompt\n"
        << "                         too long to pass as an argument. Not with -p.\n"
        << "  -p <prompt>            Answer one prompt and exit. The answer is the whole of\n"
        << "                         standard output, so it can be piped or redirected.\n"
        << "  --model <name>         Load this model, as 'mila-store models' lists it.\n"
        << "  --context-length <n>   Maximum sequence length to build for, or auto to take the\n"
        << "                         largest that fits this device with room to spare.\n"
        << "  --device <n>           Which CUDA device to run on. This is the CUDA ordinal,\n"
        << "                         which need not match the order nvidia-smi lists.\n"
        << "  --system-prompt <name> A prompt name, or a path to a JSON file holding a\n"
        << "                         system_prompt and optional tools. A path resolves\n"
        << "                         against the working directory.\n"
        << "  --settings <path>      A JSON config file, read as part of this run.\n"
        << "  --output-format <fmt>  text (default) or json. With -p only.\n"
        << "  --version              Print the Mila version and exit.\n"
        << "  --help, -h             Show this message.\n"
        << "\n"
        << "Settings merge key by key -- the compiled defaults, then your own config, then the\n"
        << "one beside the executable, then the model last chosen with /model, then a file named\n"
        << "with --settings, then the flags above. Setting one key inherits the rest, so a config\n"
        << "file need only hold what it changes. Overriding is not setting: a flag applies to\n"
        << "this run and writes nothing back.\n"
        << "\n"
        << "Your config is read from, and a named prompt is searched for under:\n"
        << "  " << ( resolveConfigRoot().empty()
                        ? std::string( "(no config directory on this platform)" )
                        : ( resolveConfigRoot() / "chat.json" ).string() ) << "\n"
        << "\n"
        << "Config keys:\n"
        << "  model              Installed model name, used until one is chosen with /model load\n"
        << "                     or /model install. There is no default: a fresh store has no model.\n"
        << "  context_length     Maximum sequence length, or \"auto\" (the default) to measure\n"
        << "                     the largest that fits this device. An explicit number is\n"
        << "                     used as written, and a model that does not fit at it is not loaded.\n"
        << "  kv_cache           bf16 (the default) | fp8. fp8 holds the context in half the\n"
        << "                     memory, so \"auto\" fits a longer one.\n"
        << "  thinking_effort    1-5 token-budget scale for reasoning (default 3 = balanced).\n"
        << "                     Reasoning is surfaced whenever the model has that channel.\n"
        << "  verbose            display detail: off | thoughts | all (default off).\n"
        << "  system_prompt_path A prompt name (assistant, tools, tools-weather,\n"
        << "                     tools-single), or a path. A relative path resolves against\n"
        << "                     the directory of the file that set it.\n"
        << "  temperature, top_k, top_p, presence_penalty\n"
        << "                     Sampling. Each one you leave out follows the model's\n"
        << "                     recommended setting, which changes with /thinking.\n"
        << "  max_new_tokens\n"
        << "\n"
        << "  quantization       none | fp8 | fp4 | q4_0. Quantizes unquantized weights on load;\n"
        << "                     a name ending -fp4/-fp8/-q4_0 is already quantized and refuses it.\n"
        << "\n";

    // What is loadable is what is installed, so the list is read rather than compiled in.
    // No budget: this runs before Mila::initialize(), so there is no device to cost against.
    const ModelListing listing = describeInstalledModels();

    for ( const auto& line : listing.table )
    {
        std::cerr << line << "\n";
    }

    for ( const auto& line : listing.notes )
    {
        std::cerr << "  " << line << "\n";
    }
}

int main( int argc, char* argv[] )
{

#ifdef _WIN32
    SetConsoleOutputCP( CP_UTF8 );
    SetConsoleCP( CP_UTF8 );
    HANDLE hOut = GetStdHandle( STD_OUTPUT_HANDLE );
    if ( hOut != INVALID_HANDLE_VALUE )
    {
        DWORD mode = 0;
        if ( GetConsoleMode( hOut, &mode ) )
            SetConsoleMode( hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING );
    }
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

    // Answered before anything is initialized: a container user's first command is often one of
    // these, and neither has an opinion about whether a model store or a device exists.
    if ( line.show_version )
    {
        std::cout << Mila::getAPIVersion().toString() << "\n";

        return 0;
    }

    if ( line.show_help )
    {
        printUsage( argv[ 0 ] );

        return 0;
    }

    // In one shot, standard output belongs to the answer alone, and the writers that would
    // otherwise share it are not all ours: ConsoleSink sends every record below Error to
    // std::cout, and that is library code. So std::cout is pointed at standard error for the
    // whole run and the answer is written through the buffer it used to hold. One redirect
    // covers every writer, including ones added later; suppressing call sites one at a time
    // could not have covered the tokenizer warning that found this.
    std::streambuf* const standard_output = line.one_shot
        ? std::cout.rdbuf( std::cerr.rdbuf() )
        : std::cout.rdbuf();

    std::ostream answer_out( standard_output );

    // Quiet by default — only warnings and errors. "verbose" in the session config
    // raises the level to Info to show tokenizer/model load logging.
    auto sink = std::make_shared<SpinnerAwareConsoleSink>( Mila::Logging::LogLevel::Warning );
    Mila::initialize( 0, std::move( sink ) );

    try
    {
        ChatConfig config = resolveChatConfig(
            line.overrides,
            line.settings_given ? std::optional<std::filesystem::path>( line.settings ) : std::nullopt,
            line.model_given,
            executable_directory() );

        if ( config.detail == DetailLevel::All )
            Mila::Logging::Logger::defaultLogger().setLevel( Mila::Logging::LogLevel::Info );

        // Only when a model resolved: with nothing selected there are no paths to check, and
        // the session opens anyway so /model install can be reached.
        if ( !config.model_name.empty() )
        {
            // The store named these, so a missing one is a broken installation rather than a
            // typo. Not usage: printing the flag list here answered a question nobody asked.
            if ( !std::filesystem::exists( config.model_path ) )
            {
                std::cerr << "Error: Model file not found: " << config.model_path << "\n";

                return 4;
            }

            if ( !std::filesystem::exists( config.tokenizer_path ) )
            {
                std::cerr << "Error: Tokenizer file not found: " << config.tokenizer_path << "\n";

                return 4;
            }
        }

        ConsoleFrontEnd chat( std::move( config ) );

        // Probe stub for the Gemma 4 native tool-call format experiment
        // (GemmaChatProtocol.md): returns a canned reading, no real lookup.
        chat.registerTool( "get_weather", []( const std::string& arguments ) -> std::string
        {
            return R"({"temperature_c": 18, "condition": "cloudy"})";
        } );

        if ( line.one_shot )
        {
            return chat.runOnce( line.prompt, line.json_output, answer_out );
        }

        chat.run();

        return 0;
    }
    catch ( const UsageError& e )
    {
        std::cerr << e.what() << "\n";

        return 2;
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
