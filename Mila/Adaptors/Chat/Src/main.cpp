#include <charconv>
#include <cstddef>
#include <string>
#include <string_view>
#include <optional>
#include <iostream>
#include <fstream>
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
import Mila.Chat;
import Chat.FamilyTraits;
import Chat.Footprint;
import Chat.Settings;
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
 * @brief Where this user's own configuration lives, mirroring resolveStoreRoot().
 *
 * %APPDATA% and not %LOCALAPPDATA%: settings are small and personal and should roam, while the
 * store correctly stays in Local because model blobs are large and machine-specific. A different
 * variable, deliberately. macOS takes the Linux path for symmetry -- Apple's own convention is
 * ~/Library/Application Support, worth revisiting only if a Metal backend ever makes macOS a real
 * target.
 *
 * Here rather than beside resolveStoreRoot() in the library, which is what
 * ChatConfiguration.md section 7 pictures: this resolves CHAT's configuration, and the library has
 * none of its own to place. The moment a second adaptor wants the same convention is the moment it
 * moves, and that is a library decision rather than this one.
 *
 * Empty when no variable answers, which callers read as "this platform has no config root" rather
 * than falling back to a directory that is nobody's idea of one.
 */
static std::filesystem::path resolveConfigRoot()
{
    using Mila::Distribution::readEnvironmentVariable;

    if ( const auto app_data = readEnvironmentVariable( "APPDATA" ) )
    {
        return std::filesystem::path( *app_data ) / "Mila";
    }

    if ( const auto xdg_config = readEnvironmentVariable( "XDG_CONFIG_HOME" ) )
    {
        return std::filesystem::path( *xdg_config ) / "mila";
    }

    if ( const auto home = readEnvironmentVariable( "HOME" ) )
    {
        return std::filesystem::path( *home ) / ".config" / "mila";
    }

    return {};
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
        << "  model              Installed model name, used until one is chosen with /model or\n"
        << "                     /install. There is no default: a fresh store has no model.\n"
        << "  context_length     Maximum sequence length, or \"auto\" (the default) to measure\n"
        << "                     the largest that fits this device. An explicit number is\n"
        << "                     honoured as written, with a warning if it will not fit.\n"
        << "  thinking_effort    1-5 token-budget scale for reasoning (default 3 = balanced).\n"
        << "                     Reasoning is surfaced whenever the model has that channel.\n"
        << "  verbose            display detail: off | thoughts | all (default off).\n"
        << "  system_prompt_path A prompt name (assistant, tools, tools-weather,\n"
        << "                     tools-single), or a path. A relative path resolves against\n"
        << "                     the directory of the file that set it.\n"
        << "  temperature, top_k, top_p, max_new_tokens.\n"
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
 * @brief One layer's file, parsed and checked to be a settings object.
 *
 * @throws ConfigError on a parse error, or on a document that is not an object. The second is not
 *         pedantry: an RFC 7386 merge patch that is not an object REPLACES what it is merged into,
 *         so a file holding an array would not misconfigure one key -- it would erase every layer
 *         beneath it.
 */
static nlohmann::json readSettingsFile( const std::filesystem::path& path )
{
    std::ifstream file( path );
    nlohmann::json document;

    try
    {
        file >> document;
    }
    catch ( const nlohmann::json::parse_error& e )
    {
        throw ConfigError( std::format(
            "Settings: JSON parse error in '{}': {}", path.string(), e.what() ) );
    }

    if ( !document.is_object() )
    {
        throw ConfigError( std::format(
            "Settings: '{}' must hold a JSON object of settings.", path.string() ) );
    }

    return document;
}

/**
 * @brief Layer 1: the compiled defaults.
 *
 * Rule one of ChatConfiguration.md section 3 -- a run with no configuration file anywhere must
 * still produce a Chat that can answer -- is this function. It carries the one key that has no
 * working value without it: a context of zero is not a small context, it is a failed load.
 *
 * Takes no family, which is the shape of the layer rather than an omission: the per-family facts
 * are in familyTraits(), and the one compiled SETTING is now the same for every architecture.
 *
 * The sampler defaults are deliberately not here. They already have a working value in
 * ChatConfig's member initializers, and repeating 0.8 in two places would be a second home for
 * one number rather than a second layer.
 */
static SettingsPatch familyInvariants()
{
    nlohmann::json values;

    // "auto", not a number. No constant compiled into an adaptor can be right for an 8 GB card, a
    // 12 GB card and a 24 GB one, which is what the 512 this replaces actually was: a constant
    // chosen to be wrong safely. The family's own default survives as the floor auto lands on
    // when there is no device to measure -- see familyTraits().
    values[ "context_length" ] = "auto";

    // The default persona, by NAME rather than by path, so it resolves the same from a checkout,
    // an install and a container. Compiled rather than shipped in a file beside the executable:
    // that file is layer 4, which OUTRANKS the user's own config, so naming the prompt there left
    // a user unable to choose a different one from their own chat.json. Measured, not reasoned.
    values[ "system_prompt_path" ] = "assistant";

    return SettingsPatch{ .layer = SettingsLayer::FamilyInvariants, .values = std::move( values ) };
}

/// @brief The merged value of a key, or nullptr when no layer set it.
static const nlohmann::json* findSetting( const MergedSettings& settings, const std::string& key )
{
    const auto found = settings.values().find( key );

    return found == settings.values().end() ? nullptr : &*found;
}

/**
 * @brief A key whose value is the wrong shape, named along with where it came from.
 *
 * The origin is in the message because a merged document has no single file to blame: a user told
 * only that context_length must be a positive integer has three files to search.
 */
static ConfigError badValue(
    const MergedSettings& settings,
    const std::string& key,
    std::string_view expected,
    const nlohmann::json& value )
{
    return ConfigError( std::format( "{} ({}): expected {}, but found {}.",
        key, settings.describeOrigin( key ), expected, value.dump() ) );
}

/// Empty when no layer set the key.
static std::string readString( const MergedSettings& settings, const std::string& key )
{
    const nlohmann::json* value = findSetting( settings, key );

    if ( !value )
        return {};

    if ( !value->is_string() )
        throw badValue( settings, key, "a string", *value );

    return value->get<std::string>();
}

/// A count that must be greater than zero -- context_length, max_new_tokens.
static std::optional<std::size_t> readCount( const MergedSettings& settings, const std::string& key )
{
    const nlohmann::json* value = findSetting( settings, key );

    if ( !value )
        return std::nullopt;

    if ( !value->is_number_unsigned() || value->get<std::size_t>() == 0 )
        throw badValue( settings, key, "a positive integer", *value );

    return value->get<std::size_t>();
}

/**
 * @brief What the merged `context_length` asks for: a length, or a measurement of the device.
 *
 * The only key with two shapes, because it is the only one whose right answer is a property of
 * the machine rather than of the model or the user. See ChatConfiguration.md section 6.
 */
struct ContextRequest
{
    bool present{ false };
    bool automatic{ false };
    std::size_t length{ 0 };
};

static ContextRequest readContextLength( const MergedSettings& settings )
{
    const nlohmann::json* value = findSetting( settings, "context_length" );

    if ( !value )
        return {};

    if ( value->is_string() && value->get<std::string>() == "auto" )
        return { .present = true, .automatic = true };

    if ( !value->is_number_unsigned() || value->get<std::size_t>() == 0 )
        throw badValue( settings, "context_length", "a positive integer or \"auto\"", *value );

    return { .present = true, .automatic = false, .length = value->get<std::size_t>() };
}

/// A knob that may legitimately be zero -- top_k 0 disables it, thinking_effort is clamped.
static std::optional<int> readInteger( const MergedSettings& settings, const std::string& key )
{
    const nlohmann::json* value = findSetting( settings, key );

    if ( !value )
        return std::nullopt;

    if ( !value->is_number_integer() || value->get<long long>() < 0 )
        throw badValue( settings, key, "a whole number of zero or more", *value );

    return value->get<int>();
}

/// temperature and top_p, where zero is greedy decoding rather than an omission.
static std::optional<float> readNumber( const MergedSettings& settings, const std::string& key )
{
    const nlohmann::json* value = findSetting( settings, key );

    if ( !value )
        return std::nullopt;

    if ( !value->is_number() || value->get<double>() < 0.0 )
        throw badValue( settings, key, "a number of zero or more", *value );

    return value->get<float>();
}

/**
 * @brief Resolve system_prompt_path to a file that exists, per ChatConfiguration.md section 8.
 *
 * Three rules, and the origin of the key is what selects between them.
 *
 * @return Nothing when the COMPILED default names a prompt this install does not have, which is
 *         not the user's mistake and must not stop a session: rule one of section 3 is that a run
 *         with no files anywhere still answers, and Prompts/ is a file.
 *
 * @throws ConfigError naming the prompt and where it was looked for, when any layer above the
 *         compiled default asked for it. Never silently replaced: a user who asked for a persona
 *         and got a different one has no way to discover that from the transcript.
 */
static std::optional<std::filesystem::path> resolveSystemPrompt(
    const std::string& value,
    const std::optional<SettingsOrigin>& origin,
    const std::filesystem::path& config_root,
    const std::string& described_origin )
{
    const std::filesystem::path written{ value };

    // A bare name -- no separator, no extension -- is a NAMED prompt rather than a path, resolved
    // through a search path. Written once, it works from a checkout, an install and a container
    // alike, and a user shadows a shipped prompt by putting their own beside their config.
    const bool is_named = value.find( '/' ) == std::string::npos
        && value.find( '\\' ) == std::string::npos
        && !written.has_extension();

    if ( is_named )
    {
        const std::string file_name = value + ".json";

        std::vector<std::filesystem::path> searched;

        if ( !config_root.empty() )
            searched.push_back( config_root / "prompts" / file_name );

        searched.push_back( executable_directory() / "Prompts" / file_name );

        for ( const auto& candidate : searched )
        {
            if ( std::filesystem::exists( candidate ) )
                return candidate;
        }

        // The compiled default naming a prompt this install does not carry is our omission, not
        // the user's request, so the session opens without a system prompt instead of refusing.
        if ( origin && origin->layer == SettingsLayer::FamilyInvariants )
        {
            return std::nullopt;
        }

        std::string looked_in;

        for ( const auto& candidate : searched )
        {
            looked_in += looked_in.empty() ? "" : ", ";
            looked_in += candidate.string();
        }

        throw ConfigError( std::format(
            "system_prompt_path ({}): no prompt named '{}'. Looked in {}.",
            described_origin, value, looked_in ) );
    }

    std::filesystem::path resolved = written;

    if ( resolved.is_relative() )
    {
        // A relative path resolves against the directory of the FILE that set the key -- the rule
        // tsconfig, Cargo and every include-path system settles on, and the only one that survives
        // the config root moving to %APPDATA%. A path typed on the command line is the exception:
        // the user typed it against the working directory and tab-completed there.
        const bool from_file = origin && !origin->file.empty();

        if ( from_file )
            resolved = origin->file.parent_path() / resolved;
    }

    if ( !std::filesystem::exists( resolved ) )
    {
        throw ConfigError( std::format(
            "system_prompt_path ({}): no such file: {}",
            described_origin, std::filesystem::absolute( resolved ).string() ) );
    }

    return resolved;
}

/**
 * @brief Resolve every setting for this run, from the layers of ChatConfiguration.md section 3.
 *
 * Each layer overrides the previous key by key, never file by file, so setting one key inherits
 * every other. The model NAME is an ordinary merged key like any other, which is what removes the
 * ladder of special cases that used to decide it -- the ranking of the layers already says which
 * name wins.
 *
 * The upper layers are merged twice, deliberately. Layers 1 and 2 describe a model, and which
 * model that is comes from the layers above them, so the name is read first and the whole order
 * is then merged in rank sequence. The layers are captured before either pass, so both see the
 * same bytes.
 */
static ChatConfig buildConfig( const CommandLine& line )
{
    std::vector<SettingsPatch> overrides;

    const std::filesystem::path config_root = resolveConfigRoot();

    // Layer 3 -- how this person likes Chat, wherever they run it from.
    if ( !config_root.empty() )
    {
        const std::filesystem::path user_config = config_root / "chat.json";

        if ( std::filesystem::exists( user_config ) )
        {
            overrides.push_back( SettingsPatch{
                .layer = SettingsLayer::UserConfig,
                .values = readSettingsFile( user_config ),
                .file = user_config } );
        }
    }

    // Layer 4 -- the config that is FOUND: this checkout's, or this image's. Named for what it
    // configures rather than for a session, and the same name the user config carries: one file
    // name, two ranks, which is the whole difference between them.
    std::filesystem::path local_path{ "chat.json" };

    // Resolved next to the executable when it is not present relative to the working directory,
    // so what is found does not depend on where the process was launched from.
    if ( !std::filesystem::exists( local_path ) )
    {
        const std::filesystem::path exe_relative = executable_directory() / "chat.json";

        if ( std::filesystem::exists( exe_relative ) )
            local_path = exe_relative;
    }

    if ( std::filesystem::exists( local_path ) )
    {
        overrides.push_back( SettingsPatch{
            .layer = SettingsLayer::LocalConfig,
            .values = readSettingsFile( local_path ),
            .file = local_path } );
    }

    // Layer 5 -- the model last chosen from inside a session, which is the one key it covers.
    // Choosing a model with /model load or /model install is an explicit act and should survive
    // a fresh store has nothing to remember, and that is the honest description of a new install.
    if ( const auto last_chosen = readLastChosenModel() )
    {
        nlohmann::json remembered;
        remembered[ "model" ] = *last_chosen;

        overrides.push_back( SettingsPatch{
            .layer = SettingsLayer::RememberedChoice, .values = std::move( remembered ) } );
    }

    // Layer 6 -- a file NAMED for this run was chosen exactly as a flag was, so it carries the
    // same rank and outranks the remembered choice. Pointing at a file that names a model and
    // being handed a different one made the file mean different things depending on invisible
    // state. It writes nothing back either: overriding is not setting.
    if ( line.settings_given )
    {
        // A named file that is not there is a mistake, not a fallback: silently using the layers
        // beneath it would answer with settings the caller explicitly asked to replace.
        if ( !std::filesystem::exists( line.settings ) )
        {
            throw ConfigError( std::format(
                "Settings file not found: {}",
                std::filesystem::absolute( line.settings ).string() ) );
        }

        overrides.push_back( SettingsPatch{
            .layer = SettingsLayer::CommandLine,
            .values = readSettingsFile( line.settings ),
            .file = line.settings } );
    }

    // ...and the flags outrank the file they were typed beside.
    if ( !line.overrides.empty() )
    {
        overrides.push_back( SettingsPatch{
            .layer = SettingsLayer::CommandLine, .values = line.overrides } );
    }

    // Which model, and how to deploy it: read from the layers above the model, because the two
    // layers below it cannot be built until the answer is known.
    MergedSettings chosen;
    chosen.applyAll( overrides );

    const std::string name = readString( chosen, "model" );

    // Quantization is a deployment choice against an unquantized artifact, so it is settled
    // before the model resolves rather than being part of the name.
    std::optional<QuantizationMode> requested_quantization;

    const std::string quantization = readString( chosen, "quantization" );

    if ( !quantization.empty() )
    {
        const auto parsed = parseQuantization( quantization );

        if ( !parsed )
            throw ConfigError( std::format(
                "quantization ({}): expected none, fp8 or fp4, but found '{}'.",
                chosen.describeOrigin( "quantization" ), quantization ) );

        requested_quantization = *parsed;
    }

    // A name that does not resolve is reported into the session rather than out of it. The
    // commands that fix it -- /model install, /model list, /model -- are all inside, so
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

    // The whole order, in rank sequence. Layers 1 and 2 describe the model that just resolved, so
    // with nothing resolved there is nothing for them to say and the session opens on what the
    // files asked for -- a state /model repairs, since switching re-derives from the family.
    MergedSettings settings;

    if ( resolved )
    {
        settings.apply( familyInvariants() );

        // Layer 2 -- what this checkpoint recommends, once ModelRecord carries
        // default_context_length and maximum_context_length. Publishing a model must not require
        // editing a switch statement in a chat adaptor. See ChatConfiguration.md section 5.
    }

    settings.applyAll( overrides );

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
    }

    // The one value resolved after the merge rather than by it, because the device is not a layer:
    // what fits the card is a value a key can TAKE, not a silent override that outranks what the
    // user asked for. A user who writes 8192 gets 8192; a user who writes nothing gets "auto" from
    // layer 1 and gets the card measured. ChatConfiguration.md section 6.
    if ( const ContextRequest request = readContextLength( settings ); request.present )
    {
        const FamilyTraits traits = familyTraits( config.model_type );

        // Carried into the session so /context can report where the number came from. Recorded
        // here because this is the only place that holds the merged document -- the origin knows
        // the FILE a layer wrote from, which the layer alone cannot say.
        config.context_origin = settings.describeOrigin( "context_length" );

        if ( request.automatic && resolved )
        {
            const ResolvedContext measured = resolveAutomaticContext(
                config.model_path, config.model_type, config.precision,
                config.quantization_mode, traits.max_context, traits.default_context );

            config.context_length = measured.context_length;
            config.context_is_automatic = true;
        }
        else if ( request.automatic )
        {
            // No model to measure, so nothing to measure it against. The session opens with
            // nothing selected and /model resolves the context when one is chosen.
            config.context_is_automatic = true;
        }
        else
        {
            config.context_length = request.length;

            // What was ASKED for, kept apart from the live value because a model switch rewrites
            // the latter. Only a value from above the defaults is a preference: a family default
            // that survived a switch would carry 512 into a family that opens at 4096.
            const auto origin = settings.originOf( "context_length" );

            if ( origin && origin->layer > SettingsLayer::ModelRecommendations )
                config.configured_context_length = request.length;

            // Clamped to what the architecture can address. One configuration serves every model
            // the session may load, so a context chosen for a 12B model reaches GPT-2 as well --
            // and GPT-2's positions are a 1024-row learned table, so the oversized value is a
            // failed load rather than a slow one.
            if ( resolved && request.length > traits.max_context )
            {
                std::cerr << std::format(
                    "context_length {} ({}) exceeds what {} can address; using {}.\n",
                    request.length, settings.describeOrigin( "context_length" ),
                    config.model_name, traits.max_context );

                config.context_length = traits.max_context;
            }
        }
    }

    if ( const auto effort = readInteger( settings, "thinking_effort" ) )
    {
        config.thinking_effort = *effort < 1 ? 1 : ( *effort > 5 ? 5 : *effort );
    }

    // "verbose" sets the display-detail level: a string (off/thoughts/all) or a bool
    // (true -> all, false -> off) for backward compatibility.
    if ( const nlohmann::json* verbose = findSetting( settings, "verbose" ) )
    {
        if ( verbose->is_boolean() )
        {
            config.detail = verbose->get<bool>() ? DetailLevel::All : DetailLevel::Off;
        }
        else if ( verbose->is_string() )
        {
            const auto level = parseDetailLevel( verbose->get<std::string>() );

            // Named rather than ignored: a user who asked to see the reasoning and silently got
            // the default has nothing in the transcript to tell them the key was misspelled.
            if ( !level )
                throw badValue( settings, "verbose", "off, thoughts or all", *verbose );

            config.detail = *level;
        }
        else
        {
            throw badValue( settings, "verbose", "off, thoughts or all", *verbose );
        }
    }

    if ( const auto tokens = readCount( settings, "max_new_tokens" ) )
        config.max_new_tokens = *tokens;

    if ( const auto temperature = readNumber( settings, "temperature" ) )
        config.temperature = *temperature;

    if ( const auto top_k = readInteger( settings, "top_k" ) )
        config.top_k = *top_k;

    if ( const auto top_p = readNumber( settings, "top_p" ) )
        config.top_p = *top_p;

    if ( const auto device = readInteger( settings, "device" ) )
        config.device_index = *device;

    if ( const std::string prompt_path = readString( settings, "system_prompt_path" );
         !prompt_path.empty() )
    {
        config.system_prompt_path = resolveSystemPrompt(
            prompt_path,
            settings.originOf( "system_prompt_path" ),
            config_root,
            settings.describeOrigin( "system_prompt_path" ) );
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
    auto sink = std::make_shared<SpinnerAwareConsoleSink>( Mila::Logging::LogLevel::Warning );
    Mila::initialize( 0, std::move( sink ) );

    try
    {
        ChatConfig config = buildConfig( line );

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
