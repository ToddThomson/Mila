#include <string>
#include <string_view>
#include <optional>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#endif

import Mila;
import Mila.Chat;
import nlohmann.json;

using namespace Mila::ChatApp;

/**
 * @brief Directory containing the running executable.
 *
 * The POST_BUILD step copies Data/ next to the executable, so resolving the
 * default session config against this directory makes the default model
 * independent of the process working directory (which the IDE/debugger may set
 * to the source tree or repo root rather than the binary output directory).
 *
 * Every supported platform must answer this for itself. Falling through to the
 * working directory does not implement the guarantee above, it silently drops
 * it -- which is how a container ended up loading no config at all, and why the
 * image documents its WORKDIR as a correctness requirement. See
 * Specifications/ChatConfiguration.md section 2.
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

/// @brief A configuration that parsed but cannot be honoured.
struct ConfigError : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

/// @brief A model name that named nothing the store holds.
struct ModelNotFound : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

/**
 * @brief The command line, parsed once, before anything else happens.
 *
 * Flags are the last word over every configuration file -- see ChatConfiguration.md section 9.
 * They are collected here rather than applied as they are read, so that the order they were
 * typed in cannot change what they mean.
 */
struct CommandLine
{
    std::filesystem::path settings;
    bool settings_given = false;

    std::string model;
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
            line.model = requireValue( argc, argv, i, arg );
            line.model_given = true;
        }
        else if ( arg == "-p" )
        {
            line.prompt = requireValue( argc, argv, i, arg );
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
        << "  -p <prompt>            Answer one prompt and exit. The answer is the whole of\n"
        << "                         standard output, so it can be piped or redirected.\n"
        << "  --model <name>         Load this model, as 'mila models' lists it.\n"
        << "  --settings <path>      Session JSON config (default: Data/session.json).\n"
        << "  --output-format <fmt>  text (default) or json. With -p only.\n"
        << "  --version              Print the Mila version and exit.\n"
        << "  --help, -h             Show this message.\n"
        << "\n"
        << "Flags win over the session config. Remaining settings live in that file:\n"
        << "\n"
        << "Session config keys:\n"
        << "  model              Installed model name, used until one is chosen with /model or\n"
        << "                     /install. There is no default: a fresh store has no model.\n"
        << "  context_length     Maximum sequence length. Default: the model's own default.\n"
        << "  thinking_effort    1-5 token-budget scale for reasoning (default 3 = balanced).\n"
        << "                     Reasoning is surfaced whenever the model has that channel.\n"
        << "  verbose            display detail: off | thoughts | all (default off).\n"
        << "  temperature, top_k, top_p, max_new_tokens, system_prompt_path.\n"
        << "\n"
        << "  quantization       none | fp8 | fp4. Quantizes an unquantized artifact on load;\n"
        << "                     a name ending -fp4/-fp8 is already quantized and refuses it.\n"
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

/**
 * @brief Resolve the session config path, build a ChatConfig from it.
 *
 * The session JSON names a model, which the catalog resolves into the
 * family/size/precision/paths, plus optional overrides (quantization, context length) and
 * runtime preferences. Flags from the command line are applied last and win.
 */
static ChatConfig buildConfig( const CommandLine& line )
{
    std::filesystem::path config_path =
        line.settings_given ? line.settings : std::filesystem::path( "Data/session.json" );

    // Resolve the default config next to the executable when it is not present
    // relative to the working directory, so the default does not depend on where
    // the process was launched from. An explicit --settings is honored as-is.
    if ( !line.settings_given && !std::filesystem::exists( config_path ) )
    {
        const std::filesystem::path exe_relative = executable_directory() / "Data" / "session.json";

        if ( std::filesystem::exists( exe_relative ) )
            config_path = exe_relative;
    }

    nlohmann::json j;

    if ( std::filesystem::exists( config_path ) )
    {
        std::ifstream file( config_path );

        try
        {
            file >> j;
        }
        catch ( const nlohmann::json::parse_error& e )
        {
            throw ConfigError( std::format(
                "Session config: JSON parse error in '{}': {}", config_path.string(), e.what() ) );
        }
    }
    else if ( line.settings_given )
    {
        // A named file that is not there is a mistake, not a fallback: silently using defaults
        // would answer with settings the caller explicitly asked to replace.
        throw ConfigError( std::format(
            "Settings file not found: {}", std::filesystem::absolute( config_path ).string() ) );
    }
    else
    {
        std::cerr << "Session config: none found at "
            << std::filesystem::absolute( config_path ).string()
            << " - using defaults.\n";
    }

    // Resolve the model name against the store. There is no catalogue: what can be loaded is
    // what is installed, which is why this is a lookup rather than a table.
    //
    // THERE IS NO DEFAULT MODEL. The last model chosen wins, because choosing one with /model or
    // /install is an explicit act and should survive the session; a "model" key in the config is
    // the fallback for a store that has never been chosen from. When neither exists the session
    // opens with nothing selected, which is the honest description of a fresh install -- naming a
    // model here reported a 6+ GB artifact as missing to a user who had never asked for it.
    //
    // Explicitness of the invocation decides, and nothing here writes anything back: an
    // override is for this run, while only an in-session /model or /install SETS the
    // remembered choice (see switchModel).
    //
    //   --model                     always wins; the direct instruction for this run
    //   model key of --settings     wins next; naming that file is also an act of this run
    //   remembered choice           wins over the DEFAULT config, per the paragraph above
    //   model key of default config the fallback for a store never chosen from
    //
    // Without the second rule, pointing at a settings file and being handed a different model
    // than the one it names made the file mean different things depending on invisible state.
    const bool config_names_model = j.contains( "model" ) && j[ "model" ].is_string();
    std::string name;

    if ( line.model_given )
        name = line.model;
    else if ( line.settings_given && config_names_model )
        name = j[ "model" ].get<std::string>();
    else if ( const auto last_chosen = readLastChosenModel() )
        name = *last_chosen;
    else if ( config_names_model )
        name = j[ "model" ].get<std::string>();

    // Quantization is a deployment choice against an unquantized artifact, so it is settled
    // before the model resolves rather than being part of the name.
    std::optional<QuantizationMode> requested_quantization;

    if ( j.contains( "quantization" ) && j[ "quantization" ].is_string() )
    {
        const std::string value = j[ "quantization" ].get<std::string>();
        const auto parsed = parseQuantization( value );

        if ( !parsed )
            throw ConfigError( std::format(
                "Unknown quantization '{}'. Use none, fp8, or fp4.", value ) );

        requested_quantization = *parsed;
    }

    // A name that does not resolve is reported into the session rather than out of it. The
    // commands that fix it -- /install, /models, /model -- are all inside the session, so
    // exiting here is precisely what left a clean machine with no way to get its first model.
    std::optional<ResolvedModel> resolved;
    std::string no_model_reason;

    if ( name.empty() )
    {
        // Nothing has ever been chosen, and that needs no explanation -- the session banner
        // already tells an empty store what to do. A reason is for the other case, where a name
        // WAS asked for and did not resolve; printing one here just said it twice.
    }
    else
    {
        try
        {
            resolved = resolveModel( name, requested_quantization );
        }
        catch ( const std::exception& e )
        {
            // Except when the user named it on the command line. Opening a session that ignored
            // the one instruction it was given is worse than refusing, and a script that asked
            // for a specific model needs to know it did not get one.
            if ( line.model_given )
                throw ModelNotFound( e.what() );

            no_model_reason = e.what();
        }
    }

    ChatConfig config;
    config.no_model_reason = no_model_reason;

    if ( resolved )
    {
        config.model_name        = resolved->name;
        config.model_type        = resolved->family;
        config.precision         = resolved->precision;
        config.is_instruct       = resolved->instruct;
        config.base_model        = resolved->base_model;
        config.license           = resolved->license;
        config.streaming_capable = resolved->streaming_capable;

        // Thinking follows the model, not the config: a preference cannot give a model a
        // reasoning channel, and offering the switch only ever misreported the ones without one.
        config.thinking_capable  = resolved->thinking_capable;
        config.show_thinking     = resolved->thinking_capable;
        config.quantization_mode = resolved->quantization;
        config.quantization_applied_at_load = resolved->quantization_applied_at_load;
        config.model_path        = resolved->weights;
        config.tokenizer_path    = resolved->tokenizer;

        // Context length: explicit override, else the model's own default.
        config.context_length = resolved->default_context;
    }

    config.config_path    = config_path;

    // Clamped to what the architecture can address. One config serves every model the session may
    // load, so a context chosen for a 12B model reaches GPT-2 as well -- and GPT-2's positions are
    // a 1024-row learned table, so the oversized value is a failed load rather than a slow one.
    if ( j.contains( "context_length" ) && j[ "context_length" ].is_number_unsigned() )
    {
        const std::size_t requested = j[ "context_length" ].get<std::size_t>();

        config.configured_context_length = requested;

        if ( resolved )
        {
            const std::size_t ceiling = maxContextFor( config.model_type );

            if ( requested > ceiling )
            {
                std::cout << std::format(
                    "Session config: context_length {} exceeds what {} can address; using {}.\n",
                    requested, config.model_name, ceiling );
            }

            config.context_length = requested < ceiling ? requested : ceiling;
        }
        else
        {
            config.context_length = requested;
        }
    }

    if ( j.contains( "thinking_effort" ) && j[ "thinking_effort" ].is_number_integer() )
    {
        const int level = j[ "thinking_effort" ].get<int>();
        config.thinking_effort = level < 1 ? 1 : (level > 5 ? 5 : level);
    }

    // "verbose" sets the display-detail level: a string (off/thoughts/tools/all)
    // or a bool (true -> all, false -> off) for backward compatibility.
    if ( j.contains( "verbose" ) )
    {
        if ( j[ "verbose" ].is_string() )
        {
            const auto level = parseDetailLevel( j[ "verbose" ].get<std::string>() );

            if ( level )
                config.detail = *level;
        }
        else if ( j[ "verbose" ].is_boolean() )
        {
            config.detail = j[ "verbose" ].get<bool>() ? DetailLevel::All : DetailLevel::Off;
        }
    }

    if ( j.contains( "max_new_tokens" ) && j[ "max_new_tokens" ].is_number_unsigned() )
        config.max_new_tokens = j[ "max_new_tokens" ].get<std::size_t>();

    if ( j.contains( "temperature" ) && j[ "temperature" ].is_number() )
        config.temperature = j[ "temperature" ].get<float>();

    if ( j.contains( "top_k" ) && j[ "top_k" ].is_number_integer() )
        config.top_k = j[ "top_k" ].get<int>();

    if ( j.contains( "top_p" ) && j[ "top_p" ].is_number() )
        config.top_p = j[ "top_p" ].get<float>();

    if ( j.contains( "system_prompt_path" ) && j[ "system_prompt_path" ].is_string() )
    {
        std::filesystem::path system_prompt = j[ "system_prompt_path" ].get<std::string>();

        // Paths in the config assume the executable directory (where Data/ is copied);
        // resolve there when not found relative to the working directory.
        if ( system_prompt.is_relative() && !std::filesystem::exists( system_prompt ) )
        {
            const std::filesystem::path exe_relative = executable_directory() / system_prompt;

            if ( std::filesystem::exists( exe_relative ) )
                system_prompt = exe_relative;
        }

        config.system_prompt_path = system_prompt;
    }

    return config;
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
    auto sink = std::make_shared<Mila::Logging::ConsoleSink>( Mila::Logging::LogLevel::Warning );
    Mila::initialize( 0, std::move( sink ) );

    try
    {
        ChatConfig config = buildConfig( line );

        if ( config.detail == DetailLevel::All )
            Mila::Logging::Logger::defaultLogger().setLevel( Mila::Logging::LogLevel::Info );

        // Only when a model resolved: with nothing selected there are no paths to check, and
        // the session opens anyway so /install can be reached.
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

        if ( config.system_prompt_path.has_value() &&
             !std::filesystem::exists( *config.system_prompt_path ) )
        {
            std::cerr << "Error: System prompt file not found: "
                      << *config.system_prompt_path << "\n";

            return 3;
        }

        Chat chat( std::move( config ) );

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
