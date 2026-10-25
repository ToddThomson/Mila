/**
 * @file Chat.SettingsResolution.ixx
 * @brief Every setting for one run of a session, merged from the layers of ChatConfiguration.md section 3.
 *
 * Shared by every front end over the session -- the console and the session process -- so a person's chat.json means
 * the same thing to both.
 */

export module Chat.SettingsResolution;

import std;
import Chat.Config;
import Chat.ConfigError;
import Chat.FamilyTraits;
import Chat.Json;
import Chat.ModelCatalog;
import Chat.ModelNotFound;
import Chat.Settings;

import Mila;

namespace Mila::ChatApp
{
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
    export std::filesystem::path resolveConfigRoot()
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
     * @brief One layer's file, parsed and checked to be a settings object.
     *
     * @throws ConfigError on a parse error, or on a document that is not an object. The second is not
     *         pedantry: an RFC 7386 merge patch that is not an object REPLACES what it is merged into,
     *         so a file holding an array would not misconfigure one key -- it would erase every layer
     *         beneath it.
     */
    nlohmann::json readSettingsFile( const std::filesystem::path& path )
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
    SettingsPatch familyInvariants()
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
    const nlohmann::json* findSetting( const MergedSettings& settings, const std::string& key )
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
    ConfigError badValue(
        const MergedSettings& settings,
        const std::string& key,
        std::string_view expected,
        const nlohmann::json& value )
    {
        return ConfigError( std::format( "{} ({}): expected {}, but found {}.",
            key, settings.describeOrigin( key ), expected, value.dump() ) );
    }

    /// Empty when no layer set the key.
    std::string readString( const MergedSettings& settings, const std::string& key )
    {
        const nlohmann::json* value = findSetting( settings, key );

        if ( !value )
            return {};

        if ( !value->is_string() )
            throw badValue( settings, key, "a string", *value );

        return value->get<std::string>();
    }

    /// A count that must be greater than zero -- context_length, max_new_tokens.
    std::optional<std::size_t> readCount( const MergedSettings& settings, const std::string& key )
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

    ContextRequest readContextLength( const MergedSettings& settings )
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
    std::optional<int> readInteger( const MergedSettings& settings, const std::string& key )
    {
        const nlohmann::json* value = findSetting( settings, key );

        if ( !value )
            return std::nullopt;

        if ( !value->is_number_integer() || value->get<long long>() < 0 )
            throw badValue( settings, key, "a whole number of zero or more", *value );

        return value->get<int>();
    }

    /// temperature and top_p, where zero is greedy decoding rather than an omission.
    std::optional<float> readNumber( const MergedSettings& settings, const std::string& key )
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
    std::optional<std::filesystem::path> resolveSystemPrompt(
        const std::string& value,
        const std::optional<SettingsOrigin>& origin,
        const std::filesystem::path& config_root,
        const std::filesystem::path& program_directory,
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

            searched.push_back( program_directory / "Prompts" / file_name );

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
     *
     * @param overrides The run's own flags, keyed as the settings they override (layer 6).
     * @param settings_file A file named for this run, merged at the flags' rank beneath them.
     * @param model_given Whether the model was named as an instruction for this run, which makes a name that does
     *        not resolve fatal instead of a session that opens on something else.
     * @param program_directory The running executable's directory, where the shipped prompts and a found
     *        chat.json live, independent of the working directory.
     *
     * @throws ConfigError when a layer says something impossible; ModelNotFound when an instructed model is not
     *         installed.
     */
    export ChatConfig resolveChatConfig(
        const nlohmann::json& overrides_from_flags,
        const std::optional<std::filesystem::path>& settings_file,
        bool model_given,
        const std::filesystem::path& program_directory )
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
            const std::filesystem::path exe_relative = program_directory / "chat.json";

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
        if ( settings_file.has_value() )
        {
            // A named file that is not there is a mistake, not a fallback: silently using the layers
            // beneath it would answer with settings the caller explicitly asked to replace.
            if ( !std::filesystem::exists( *settings_file ) )
            {
                throw ConfigError( std::format(
                    "Settings file not found: {}",
                    std::filesystem::absolute( *settings_file ).string() ) );
            }

            overrides.push_back( SettingsPatch{
                .layer = SettingsLayer::CommandLine,
                .values = readSettingsFile( *settings_file ),
                .file = *settings_file } );
        }

        // ...and the flags outrank the file they were typed beside.
        if ( !overrides_from_flags.empty() )
        {
            overrides.push_back( SettingsPatch{
                .layer = SettingsLayer::CommandLine, .values = overrides_from_flags } );
        }

        // Which model, and how to deploy it: read from the layers above the model, because the two
        // layers below it cannot be built until the answer is known.
        MergedSettings chosen;
        chosen.applyAll( overrides );

        const std::string name = readString( chosen, "model" );

        // Quantization is a deployment choice against unquantized weights, so it is settled
        // before the model resolves rather than being part of the name.
        std::optional<QuantizationMode> requested_quantization;

        const std::string quantization = readString( chosen, "quantization" );

        if ( !quantization.empty() )
        {
            const auto parsed = parseQuantization( quantization );

            if ( !parsed )
                throw ConfigError( std::format(
                    "quantization ({}): expected none, fp8, fp4 or q4_0, but found '{}'.",
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
                resolved = resolveModel(
                    name,
                    requested_quantization,
                    model_given ? RefusalAudience::Shell : RefusalAudience::Session );
            }
            catch ( const std::exception& e )
            {
                // Except when the user named it on the command line. Opening a session that ignored
                // the one instruction it was given is worse than refusing, and a script that asked
                // for a specific model needs to know it did not get one.
                if ( model_given )
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
            config.is_instruct       = resolved->instruct;
            config.base_model        = resolved->base_model;
            config.license           = resolved->license;
            config.streaming_capable = resolved->streaming_capable;

            // Thinking follows the model, not the config: a preference cannot give a model a
            // reasoning channel, and offering the switch only ever misreported the ones without one.
            config.thinking_capable  = resolved->thinking_capable;
            config.show_thinking     = resolved->thinking_capable;
            config.recommended_sampling = resolved->sampling;
            config.quantization_mode = resolved->quantization;
            config.quantization_applied_at_load = resolved->quantization_applied_at_load;
            config.model_path        = resolved->weights;
            config.tokenizer_path    = resolved->tokenizer;
        }

        // Read before the context below, which is measured on this device: read after it, "auto"
        // measured CUDA device 0 whatever --device named.
        if ( const auto device = readInteger( settings, "device" ) )
            config.device_index = *device;

        // Read before the context below for the same reason: "auto" measures what fits with this cache.
        if ( const std::string kv_cache = readString( settings, "kv_cache" ); !kv_cache.empty() )
        {
            const auto parsed = parseKvCache( kv_cache );

            if ( !parsed )
                throw ConfigError( std::format(
                    "kv_cache ({}): expected bf16 or fp8, but found '{}'.",
                    settings.describeOrigin( "kv_cache" ), kv_cache ) );

            config.kv_cache = *parsed;
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

            if ( request.automatic )
            {
                // Decided by the load, which plans the context for the model it loads against the card
                // as it stands then (Deployment.md). With no model yet, /model's load decides it.
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
                // the session may load, so a context chosen for one family reaches the next, and the
                // ceilings differ by a factor of two between Gemma and Qwen.
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

        // Signed, unlike the knobs above: a negative penalty rewards repeating a token, as OpenAI's does.
        if ( const nlohmann::json* penalty = findSetting( settings, "presence_penalty" ) )
        {
            if ( !penalty->is_number() || penalty->get<double>() < -2.0 || penalty->get<double>() > 2.0 )
                throw badValue( settings, "presence_penalty", "a number from -2 to 2", *penalty );

            config.presence_penalty = penalty->get<float>();
        }

        if ( const std::string prompt_path = readString( settings, "system_prompt_path" );
             !prompt_path.empty() )
        {
            config.system_prompt_path = resolveSystemPrompt(
                prompt_path,
                settings.originOf( "system_prompt_path" ),
                config_root,
                program_directory,
                settings.describeOrigin( "system_prompt_path" ) );
        }

        return config;
    }
}
