#include <string>
#include <string_view>
#include <optional>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <format>
#include <stdexcept>

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
 */
static std::filesystem::path executable_directory()
{
#ifdef _WIN32
    wchar_t buffer[ MAX_PATH ];
    const DWORD length = GetModuleFileNameW( nullptr, buffer, MAX_PATH );

    if ( length > 0 && length < MAX_PATH )
        return std::filesystem::path( buffer ).parent_path();
#endif
    return std::filesystem::current_path();
}

static void printUsage( const char* prog_name )
{
    std::cerr
        << "Usage: " << prog_name << " [--config <path>]\n"
        << "\n"
        << "All settings live in the session JSON config (default: Data/session.json).\n"
        << "  --config <path>   Use an alternate session JSON config file.\n"
        << "  --help, -h        Show this message.\n"
        << "\n"
        << "Session config keys:\n"
        << "  model              Installed model name. Default: " << kDefaultModelName << ".\n"
        << "  context_length     Maximum sequence length. Default: the model's own default.\n"
        << "  thinking           true to surface Gemma's reasoning channel.\n"
        << "  thinking_effort    1-5 token-budget scale for reasoning (default 3 = balanced).\n"
        << "  verbose            display detail: off | thoughts | all (default off).\n"
        << "  temperature, top_k, max_new_tokens, system_prompt_path.\n"
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
 * The single source of truth is the session JSON: it names a model alias, which
 * the catalog resolves into the family/size/precision/paths, plus optional
 * overrides (quantization, context length) and runtime preferences. The only
 * command-line option is --config, selecting which JSON file to load.
 */
static ChatConfig buildConfig( int argc, char* argv[] )
{
    std::filesystem::path config_path = "Data/session.json";
    bool explicit_config = false;

    for ( int i = 1; i < argc; ++i )
    {
        std::string_view arg = argv[ i ];

        if ( arg == "--config" )
        {
            if ( i + 1 >= argc )
                throw std::invalid_argument( "--config requires a value" );
            config_path = argv[ ++i ];
            explicit_config = true;
        }
        else
        {
            throw std::invalid_argument( std::format(
                "Unknown option: '{}'. Only --config <path> is supported; all settings "
                "live in the session JSON (run with --help).", arg ) );
        }
    }

    // Resolve the default config next to the executable when it is not present
    // relative to the working directory, so the default does not depend on where
    // the process was launched from. An explicit --config is honored as-is.
    if ( !explicit_config && !std::filesystem::exists( config_path ) )
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
            throw std::runtime_error( std::format(
                "Session config: JSON parse error in '{}': {}", config_path.string(), e.what() ) );
        }
    }
    else
    {
        std::cout << "Session config: none found at "
            << std::filesystem::absolute( config_path ).string()
            << " — using defaults.\n";
    }

    // Resolve the model name against the store. There is no catalogue: what can be loaded is
    // what is installed, which is why this is a lookup rather than a table.
    std::string name( kDefaultModelName );

    if ( j.contains( "model" ) && j[ "model" ].is_string() )
        name = j[ "model" ].get<std::string>();

    // Quantization is a deployment choice against an unquantized artifact, so it is settled
    // before the model resolves rather than being part of the name.
    std::optional<QuantizationMode> requested_quantization;

    if ( j.contains( "quantization" ) && j[ "quantization" ].is_string() )
    {
        const std::string value = j[ "quantization" ].get<std::string>();
        const auto parsed = parseQuantization( value );

        if ( !parsed )
            throw std::invalid_argument( std::format(
                "Unknown quantization '{}'. Use none, fp8, or fp4.", value ) );

        requested_quantization = *parsed;
    }

    // A name that does not resolve is reported into the session rather than out of it. The
    // commands that fix it -- /install, /models, /model -- are all inside the session, so
    // exiting here is precisely what left a clean machine with no way to get its first model.
    std::optional<ResolvedModel> resolved;
    std::string no_model_reason;

    try
    {
        resolved = resolveModel( name, requested_quantization );
    }
    catch ( const std::exception& e )
    {
        no_model_reason = e.what();
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
        config.quantization_mode = resolved->quantization;
        config.quantization_applied_at_load = resolved->quantization_applied_at_load;
        config.model_path        = resolved->weights;
        config.tokenizer_path    = resolved->tokenizer;

        // Context length: explicit override, else the model's own default.
        config.context_length = resolved->default_context;
    }

    config.config_path    = config_path;

    if ( j.contains( "context_length" ) && j[ "context_length" ].is_number_unsigned() )
        config.context_length = j[ "context_length" ].get<std::size_t>();

    if ( j.contains( "thinking" ) && j[ "thinking" ].is_boolean() )
        config.show_thinking = j[ "thinking" ].get<bool>();

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

    for ( int i = 1; i < argc; ++i )
    {
        const std::string_view arg = argv[ i ];

        if ( arg == "--help" || arg == "-h" )
        {
            printUsage( argv[ 0 ] );
            return 0;
        }
    }

    // Quiet by default — only warnings and errors. "verbose" in the session config
    // raises the level to Info to show tokenizer/model load logging.
    auto sink = std::make_shared<Mila::Logging::ConsoleSink>( Mila::Logging::LogLevel::Warning );
    Mila::initialize( 0, std::move( sink ) );

    try
    {
        ChatConfig config = buildConfig( argc, argv );

        if ( config.detail == DetailLevel::All )
            Mila::Logging::Logger::defaultLogger().setLevel( Mila::Logging::LogLevel::Info );

        // Only when a model resolved: with nothing selected there are no paths to check, and
        // the session opens anyway so /install can be reached.
        if ( !config.model_name.empty() )
        {
            if ( !std::filesystem::exists( config.model_path ) )
            {
                std::cerr << "Error: Model file not found: " << config.model_path << "\n";
                printUsage( argv[ 0 ] );
                return 1;
            }

            if ( !std::filesystem::exists( config.tokenizer_path ) )
            {
                std::cerr << "Error: Tokenizer file not found: " << config.tokenizer_path << "\n";
                printUsage( argv[ 0 ] );
                return 1;
            }
        }

        if ( config.system_prompt_path.has_value() &&
             !std::filesystem::exists( *config.system_prompt_path ) )
        {
            std::cerr << "Error: System prompt file not found: "
                      << *config.system_prompt_path << "\n";
            printUsage( argv[ 0 ] );
            return 1;
        }

        Chat chat( std::move( config ) );

        // Probe stub for the Gemma 4 native tool-call format experiment
        // (GemmaChatProtocol.md): returns a canned reading, no real lookup.
        chat.registerTool( "get_weather", []( const std::string& arguments ) -> std::string
        {
            return R"({"temperature_c": 18, "condition": "cloudy"})";
        } );

        chat.run();

        return 0;
    }
    catch ( const std::exception& e )
    {
        std::cerr << "Fatal error: " << e.what() << "\n";

        return 1;
    }
}
