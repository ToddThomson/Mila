/**
 * @file ModelManifest.ixx
 * @brief The mila.json schema: what files compose a model, and for which Mila.
 *
 * One repository is one model is one name. Precision is part of that name rather than a key
 * inside this document, which is why there is no variant map here.
 * See Specifications/ModelDistribution.md.
 */

module;
#include <charconv>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

export module Distribution.ModelManifest;

import nlohmann.json;
import Mila.Version;

namespace Mila::Distribution
{
    /**
     * @brief One file composing a model, as the manifest declares it.
     *
     * `path` is the name the file has at its origin, kept for diagnostics and for republishing; the
     * bytes live at the digest.
     */
    export struct ModelFile
    {
        std::string role;
        std::string path;
        std::string sha256;
        uint64_t bytes{ 0 };
    };

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

        std::string architecture;
        std::string variant;
        std::string weight_quantization;
        std::string minimum_mila_version;

        std::string base_model;
        std::string license;

        /// Instruction-tuned. Decides the prompt template, which nothing else can infer.
        bool instruct{ false };

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
     * @brief Parse a manifest, refusing anything a consumer would have to guess about.
     *
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
        manifest.variant = json.value( "variant", std::string{} );
        manifest.weight_quantization = json.value( "weight_quantization", std::string{} );
        manifest.minimum_mila_version = json.value( "minimum_mila_version", std::string{} );
        manifest.base_model = json.value( "base_model", std::string{} );
        manifest.license = json.value( "license", std::string{} );
        manifest.instruct = json.value( "instruct", false );

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
            file.bytes = value.value( "bytes", uint64_t{ 0 } );

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
            { "license", manifest.license },
            { "instruct", manifest.instruct },
            { "files", files } };

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

            text.remove_prefix( static_cast<size_t>( result.ptr - first ) );
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
