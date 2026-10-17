/**
 * @file ModelManifest.ixx
 * @brief The mila.json schema: what files compose a model, and for which Mila.
 *
 * One repository is one model is one name. Precision is part of that name rather than a key
 * inside this document, which is why there is no variant map here.
 * See Specifications/ModelDistribution.md.
 */

export module Distribution.ModelManifest;

import std;
export import Distribution.ModelFile;
export import Distribution.ModelCapabilities;
export import Distribution.SamplingRecommendation;

import nlohmann.json;
import Mila.Version;

namespace Mila::Distribution
{
    /**
     * @brief A model's published description.
     *
     * `variant` is descriptive, not a key: the name already carries the precision, and this states
     * what the bytes actually are so a name that lies is visible rather than load-bearing.
     *
     * `base_model` and `license` are lineage, and they are published on purpose -- every license
     * Mila redistributes under requires attribution to travel with the weights. Origin, by
     * contrast, is where a particular copy came from, belongs to one installation, and lives in
     * the store's record instead.
     */
    export struct ModelManifest
    {
        int manifest_version{ 1 };

        /// The model's name. Empty leaves the store to take it from the package directory.
        std::string name;

        /// The family, as every manifest since 0.20 has written it: "gemma", "llama", "qwen".
        std::string architecture;

        /**
         * @brief Which of the family's architectures the weights are: "3.1" or "3.2" for Llama.
         *
         * It names the chat template too, which ships with the checkpoint. Empty in every manifest
         * written before it existed; a family with one template reads empty as that one, and Llama,
         * which has two, refuses it rather than guess.
         */
        std::string architecture_version;

        std::string variant;
        std::string weight_quantization;
        std::string minimum_mila_version;

        std::string base_model;
        std::string license;

        /// What the model was trained to do. `instruct` decides the prompt template, which nothing else can infer.
        ModelCapabilities capabilities;

        /// The base model's publisher's sampling settings. Empty when the manifest carries none.
        std::vector<SamplingRecommendation> sampling;

        std::vector<ModelFile> files;

        const ModelFile* file( std::string_view role ) const
        {
            for ( const auto& entry : files )
            {
                if ( entry.role == role )
                {
                    return &entry;
                }
            }

            return nullptr;
        }
    };

    /// The role every model must declare, because a model without weights is not one.
    export inline constexpr std::string_view kWeightsRole = "weights";

    /// Optional, and separate from the weights so a reader can find it without knowing its name.
    export inline constexpr std::string_view kTokenizerRole = "tokenizer";

    /**
     * @brief The source model's license text, and any attribution file the license names.
     *
     * Declared rather than merely shipped in the package, because only a declared file is
     * fetched: every license Mila redistributes under requires its copy travel with the work,
     * and a pull that delivered weights alone would leave the recipient holding a copy that
     * carries neither. Llama 3.x 1.b.iii is the strictest of them -- the notice must sit in a
     * file distributed *as part of* the copy -- and Apache 2.0 4(a) and MIT ask the same of the
     * license text.
     */
    export inline constexpr std::string_view kLicenseRole = "license";
    export inline constexpr std::string_view kNoticeRole = "notice";

    /// The draft model a package carries for speculative decoding. Its presence is the MTP claim.
    export inline constexpr std::string_view kDrafterRole = "drafter";

    /// Every model reads text, so a manifest names it for the reader and no reader needs it named.
    export inline constexpr std::string_view kTextModality = "text";

    /**
     * @brief Read the capabilities and modalities a manifest or a store record declares.
     *
     * A document written before they existed carries only the top-level `instruct`, which writers keep emitting
     * so an older Mila still reads it; `capabilities.instruct` wins where both are present.
     */
    export template<typename TJson>
    ModelCapabilities readCapabilities( const TJson& json )
    {
        ModelCapabilities capabilities;
        capabilities.instruct = json.value( "instruct", false );

        if ( json.contains( "capabilities" ) && json[ "capabilities" ].is_object() )
        {
            const auto& declared = json[ "capabilities" ];

            capabilities.instruct = declared.value( "instruct", capabilities.instruct );
            capabilities.thinking = declared.value( "thinking", false );
            capabilities.tools = declared.value( "tools", false );
            capabilities.agentic = declared.value( "agentic", false );
        }

        if ( json.contains( "modalities" ) && json[ "modalities" ].is_array() )
        {
            for ( const auto& modality : json[ "modalities" ] )
            {
                if ( modality.is_string() && modality.template get<std::string>() != kTextModality )
                {
                    capabilities.modalities.push_back( modality.template get<std::string>() );
                }
            }
        }

        return capabilities;
    }

    /**
     * @brief Write the capabilities and modalities into a manifest or a store record.
     */
    export template<typename TJson>
    void writeCapabilities( TJson& json, const ModelCapabilities& capabilities )
    {
        json[ "instruct" ] = capabilities.instruct;
        json[ "capabilities" ] = {
            { "instruct", capabilities.instruct },
            { "thinking", capabilities.thinking },
            { "tools", capabilities.tools },
            { "agentic", capabilities.agentic } };

        TJson modalities = TJson::array();
        modalities.push_back( std::string( kTextModality ) );

        for ( const auto& modality : capabilities.modalities )
        {
            modalities.push_back( modality );
        }

        json[ "modalities" ] = modalities;
    }

    /**
     * @brief Read the sampling settings a manifest or a store record declares.
     *
     * A recommendation is never a reason to refuse a model, so a malformed setting is dropped whole rather than
     * applied in part, and an empty list leaves the model at its family's settings. A knob this build does not
     * know is ignored; a condition it does not know drops the setting, which would otherwise apply too widely.
     */
    export template<typename TJson>
    std::vector<SamplingRecommendation> readSampling( const TJson& json )
    {
        std::vector<SamplingRecommendation> settings;

        if ( !json.contains( "sampling" ) || !json[ "sampling" ].is_array() )
        {
            return settings;
        }

        const auto readFloat = []( const TJson& entry, const char* key, std::optional<float>& out ) -> bool
            {
                if ( !entry.contains( key ) )
                {
                    return true;
                }

                if ( !entry[ key ].is_number() || !std::isfinite( entry[ key ].template get<double>() ) )
                {
                    return false;
                }

                out = entry[ key ].template get<float>();

                return true;
            };

        for ( const auto& entry : json[ "sampling" ] )
        {
            if ( !entry.is_object() )
            {
                continue;
            }

            SamplingRecommendation setting;
            bool valid = true;

            if ( entry.contains( "when" ) )
            {
                const auto& when = entry[ "when" ];
                valid = when.is_object();

                for ( auto condition = when.begin(); valid && condition != when.end(); ++condition )
                {
                    valid = condition.key() == "thinking" && condition.value().is_boolean();

                    if ( valid )
                    {
                        setting.thinking = condition.value().template get<bool>();
                    }
                }
            }

            valid = valid
                && readFloat( entry, "temperature", setting.temperature )
                && readFloat( entry, "top_p", setting.top_p )
                && readFloat( entry, "presence_penalty", setting.presence_penalty );

            if ( valid && entry.contains( "top_k" ) )
            {
                valid = entry[ "top_k" ].is_number_integer() && entry[ "top_k" ].template get<long long>() >= 0;

                if ( valid )
                {
                    setting.top_k = entry[ "top_k" ].template get<int>();
                }
            }

            valid = valid
                && setting.temperature.value_or( 0.0f ) >= 0.0f
                && setting.top_p.value_or( 1.0f ) >= 0.0f && setting.top_p.value_or( 1.0f ) <= 1.0f;

            if ( valid )
            {
                settings.push_back( setting );
            }
        }

        return settings;
    }

    /**
     * @brief Write the sampling settings into a manifest or a store record; nothing when there are none.
     */
    export template<typename TJson>
    void writeSampling( TJson& json, std::span<const SamplingRecommendation> settings )
    {
        if ( settings.empty() )
        {
            return;
        }

        // The float's shortest spelling, so 0.95 is written as 0.95 rather than as the double nearest the float.
        const auto shortest = []( float value ) { return std::stod( std::format( "{}", value ) ); };

        TJson written = TJson::array();

        for ( const auto& setting : settings )
        {
            TJson entry = TJson::object();

            if ( setting.thinking.has_value() )
            {
                entry[ "when" ] = { { "thinking", *setting.thinking } };
            }

            if ( setting.temperature.has_value() )
            {
                entry[ "temperature" ] = shortest( *setting.temperature );
            }

            if ( setting.top_p.has_value() )
            {
                entry[ "top_p" ] = shortest( *setting.top_p );
            }

            if ( setting.top_k.has_value() )
            {
                entry[ "top_k" ] = *setting.top_k;
            }

            if ( setting.presence_penalty.has_value() )
            {
                entry[ "presence_penalty" ] = shortest( *setting.presence_penalty );
            }

            written.push_back( std::move( entry ) );
        }

        json[ "sampling" ] = std::move( written );
    }

    /**
     * @brief Parse a manifest, refusing anything a consumer would have to guess about.
     *
     * @param text The manifest's JSON text.
     * @param source What is being parsed, for messages only -- a name or a package path.
     * @throws std::runtime_error on malformed JSON, or a model that declares no usable files.
     */
    export ModelManifest parseModelManifest( const std::string& text, std::string_view source )
    {
        nlohmann::json json = nlohmann::json::parse( text, nullptr, false );

        if ( json.is_discarded() || !json.is_object() )
        {
            throw std::runtime_error( std::format(
                "{}: mila.json is not a JSON object", source ) );
        }

        ModelManifest manifest;
        manifest.manifest_version = json.value( "manifest_version", 1 );
        manifest.name = json.value( "name", std::string{} );
        manifest.architecture = json.value( "architecture", std::string{} );
        manifest.architecture_version = json.value( "architecture_version", std::string{} );
        manifest.variant = json.value( "variant", std::string{} );
        manifest.weight_quantization = json.value( "weight_quantization", std::string{} );
        manifest.minimum_mila_version = json.value( "minimum_mila_version", std::string{} );
        manifest.base_model = json.value( "base_model", std::string{} );
        manifest.license = json.value( "license", std::string{} );
        manifest.capabilities = readCapabilities( json );
        manifest.sampling = readSampling( json );

        if ( !json.contains( "files" ) || !json[ "files" ].is_object() )
        {
            throw std::runtime_error( std::format(
                "{}: mila.json declares no files", source ) );
        }

        for ( const auto& entry : json[ "files" ].items() )
        {
            const nlohmann::json& value = entry.value();

            if ( !value.is_object() || !value.contains( "path" ) || !value.contains( "sha256" ) )
            {
                throw std::runtime_error( std::format(
                    "{}: file '{}' needs both a path and a sha256", source, entry.key() ) );
            }

            ModelFile file;
            file.role = entry.key();
            file.path = value[ "path" ].get<std::string>();
            file.sha256 = value[ "sha256" ].get<std::string>();
            file.bytes = value.value( "bytes", std::uint64_t{ 0 } );

            manifest.files.push_back( std::move( file ) );
        }

        if ( manifest.file( kWeightsRole ) == nullptr )
        {
            throw std::runtime_error( std::format(
                "{}: mila.json declares no '{}' file", source, kWeightsRole ) );
        }

        return manifest;
    }

    /**
     * @brief Serialize a manifest.
     *
     * ordered_json rather than json: the emitted file is read by people, and a map that sorts its
     * keys puts the tokenizer above the weights and the files above the architecture.
     */
    export std::string toJsonText( const ModelManifest& manifest )
    {
        nlohmann::ordered_json files = nlohmann::ordered_json::object();

        // The weights lead, whatever order the vector holds: it is the file a reader looks for
        // first, and the rest are companions to it.
        for ( const auto& file : manifest.files )
        {
            if ( file.role == kWeightsRole )
            {
                files[ file.role ] = {
                    { "path", file.path }, { "sha256", file.sha256 }, { "bytes", file.bytes } };
            }
        }

        for ( const auto& file : manifest.files )
        {
            if ( file.role != kWeightsRole )
            {
                files[ file.role ] = {
                    { "path", file.path }, { "sha256", file.sha256 }, { "bytes", file.bytes } };
            }
        }

        nlohmann::ordered_json document = {
            { "manifest_version", manifest.manifest_version },
            { "name", manifest.name },
            { "architecture", manifest.architecture },
            { "variant", manifest.variant },
            { "weight_quantization", manifest.weight_quantization },
            { "minimum_mila_version", manifest.minimum_mila_version },
            { "base_model", manifest.base_model },
            { "license", manifest.license } };

        // Beside the family it qualifies, and only when known: a reader older than the field ignores it either way.
        if ( !manifest.architecture_version.empty() )
        {
            nlohmann::ordered_json placed = nlohmann::ordered_json::object();

            for ( auto entry = document.begin(); entry != document.end(); ++entry )
            {
                placed[ entry.key() ] = entry.value();

                if ( entry.key() == "architecture" )
                {
                    placed[ "architecture_version" ] = manifest.architecture_version;
                }
            }

            document = std::move( placed );
        }

        writeCapabilities( document, manifest.capabilities );
        writeSampling( document, std::span<const SamplingRecommendation>( manifest.sampling ) );
        document[ "files" ] = files;

        return document.dump( 2 ) + "\n";
    }

    /**
     * @brief Parse "<major>.<minor>[.<patch>]", with an optional prerelease or build suffix.
     *
     * std::from_chars rather than sscanf: no locale involvement, and trailing text is examined
     * rather than silently discarded, so a manifest whose version field holds something that is
     * not a version is refused instead of read as one this build happens to satisfy. A '-' or '+'
     * suffix is accepted because Mila's own versions carry one.
     */
    bool parseVersion( std::string_view text, int& major, int& minor, int& patch )
    {
        major = 0;
        minor = 0;
        patch = 0;

        int* const components[] = { &major, &minor, &patch };
        int parsed = 0;

        for ( int* component : components )
        {
            const char* const first = text.data();

            const auto result = std::from_chars( first, first + text.size(), *component );

            if ( result.ec != std::errc{} )
            {
                break;
            }

            text.remove_prefix( static_cast<std::size_t>( result.ptr - first ) );
            ++parsed;

            if ( text.empty() || text.front() != '.' )
            {
                break;
            }

            text.remove_prefix( 1 );
        }

        // Major and minor are the least a comparison can mean anything with.
        if ( parsed < 2 )
        {
            return false;
        }

        return text.empty() || text.front() == '-' || text.front() == '+';
    }

    /**
     * @brief Refuse an artifact that needs a newer Mila.
     *
     * A version comparison here beats a parse error somewhere inside the tensor index, which is
     * what a format change would otherwise look like.
     */
    export void requireCompatibleMilaVersion(
        const ModelManifest& manifest, std::string_view source )
    {
        if ( manifest.minimum_mila_version.empty() )
        {
            return;
        }

        int major = 0;
        int minor = 0;
        int patch = 0;

        if ( !parseVersion( manifest.minimum_mila_version, major, minor, patch ) )
        {
            throw std::runtime_error( std::format(
                "{}: unparseable minimum_mila_version '{}'",
                source, manifest.minimum_mila_version ) );
        }

        Version current = getAPIVersion();

        const long long required_ordinal =
            ( static_cast<long long>( major ) * 1000000 ) + ( minor * 1000 ) + patch;
        const long long current_ordinal =
            ( static_cast<long long>( current.getMajor() ) * 1000000 )
            + ( current.getMinor() * 1000 ) + current.getPatch();

        if ( current_ordinal < required_ordinal )
        {
            throw std::runtime_error( std::format(
                "{}: requires Mila {} or newer; this build is {}",
                source, manifest.minimum_mila_version, current.toString() ) );
        }
    }
}
