/**
 * @file ModelResolver.ixx
 * @brief Pulls a published model into the store: manifest, variant selection, blobs, record.
 *
 * Composes a hub with a store and knows neither one's internals -- no URL appears here, and no
 * filesystem layout. The hub is an interface, so every case runs offline against a fake.
 * See Specifications/ModelDistribution.md.
 */

module;
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <format>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

export module Distribution.ModelResolver;

import nlohmann.json;
import Distribution.ModelCoordinate;
import Distribution.ModelStore;
import Distribution.ModelHub;
import Mila.Version;

namespace Mila::Distribution
{
    /**
     * @brief Pulls a published model into the store.
     *
     * Pull and load are separate verbs: this is the only thing that touches a hub, and what it
     * leaves behind is a store record that every later operation reads.
     */
    export class ModelResolver
    {
    public:

        ModelResolver( ModelStore& store, const IModelHub& hub )
            : store_( store ), hub_( hub )
        {}

        /**
         * @brief Pull the model a coordinate names.
         *
         * A path is refused rather than loaded. Installing is the operation that takes one, and
         * a load reads only the store -- an undescribed file is a model nothing knows the
         * quantization, the architecture or the integrity of.
         *
         * @throws std::runtime_error if the spec is not a coordinate, if the manifest is
         *         malformed, or if the artifact requires a newer Mila.
         */
        StoredModel pull( const std::string& spec )
        {
            const auto coordinate = parseCoordinate( spec );

            if ( !coordinate.has_value() )
            {
                std::error_code ignored;

                if ( std::filesystem::exists( spec, ignored ) )
                {
                    throw std::runtime_error( std::format(
                        "Model spec '{}' is a path, and a path cannot be loaded directly. "
                        "Install the file into the store first; only installed models load.",
                        spec ) );
                }

                throw std::runtime_error( std::format(
                    "Model spec '{}' is not a coordinate of the form "
                    "<organization>/<repository>[:<variant>][@<revision>]", spec ) );
            }

            return pull( *coordinate );
        }

        /**
         * @brief Pull a parsed coordinate, writing its record on success.
         *
         * The record is written last. Until every blob has verified there is nothing worth
         * recording, and a record naming a blob that never arrived would make a broken model
         * look installed.
         */
        StoredModel pull( const ModelCoordinate& coordinate )
        {
            if ( coordinate.isLocal() )
            {
                throw std::runtime_error( std::format(
                    "{}: '{}' names a model built on this machine, which no hub serves. "
                    "Install it from a package instead of pulling it.",
                    coordinate.toString(), kLocalOwner ) );
            }

            const std::string manifest_text = hub_.fetchManifest( coordinate );

            nlohmann::json manifest = nlohmann::json::parse( manifest_text, nullptr, false );

            if ( manifest.is_discarded() || !manifest.is_object() )
            {
                throw std::runtime_error( std::format(
                    "{}: mila.json is not a JSON object", coordinate.toString() ) );
            }

            const std::string variant_name = selectVariant( manifest, coordinate );
            const nlohmann::json& variant = manifest[ "variants" ][ variant_name ];

            requireCompatibleVersion( coordinate, variant_name, variant );

            if ( !variant.contains( "files" ) || !variant[ "files" ].is_object() )
            {
                throw std::runtime_error( std::format(
                    "{}: variant '{}' declares no files", coordinate.toString(), variant_name ) );
            }

            ModelRecord record;
            record.owner = coordinate.organization;
            record.repository = coordinate.repository;
            record.variant = variant_name;
            record.architecture = manifest.value( "architecture", std::string{} );
            record.weight_quantization = variant.value( "weight_quantization", std::string{} );
            record.minimum_mila_version = variant.value( "minimum_mila_version", std::string{} );
            record.hub = hub_.name();
            record.revision = coordinate.revision;

            const nlohmann::json& files = variant[ "files" ];

            fetchDeclaredFile( coordinate, files, "weights", true, record );
            fetchDeclaredFile( coordinate, files, "tokenizer", false, record );

            store_.writeRecord( record );

            return store_.describe( record );
        }

    private:

        static std::string selectVariant(
            const nlohmann::json& manifest, const ModelCoordinate& coordinate )
        {
            if ( !manifest.contains( "variants" ) || !manifest[ "variants" ].is_object() )
            {
                throw std::runtime_error( std::format(
                    "{}: mila.json declares no variants", coordinate.toString() ) );
            }

            const std::string requested = coordinate.variant.empty()
                ? manifest.value( "default_variant", std::string{} )
                : coordinate.variant;

            if ( requested.empty() )
            {
                throw std::runtime_error( std::format(
                    "{}: no variant given and mila.json declares no default_variant",
                    coordinate.toString() ) );
            }

            if ( !manifest[ "variants" ].contains( requested ) )
            {
                std::string available;

                for ( const auto& entry : manifest[ "variants" ].items() )
                {
                    available += available.empty() ? entry.key() : ", " + entry.key();
                }

                throw std::runtime_error( std::format(
                    "{}: no variant '{}'. Available: {}",
                    coordinate.toString(), requested, available ) );
            }

            return requested;
        }

        /**
         * @brief Parse "<major>.<minor>[.<patch>]", with an optional prerelease or build suffix.
         *
         * std::from_chars rather than sscanf: no locale involvement, and trailing text is
         * examined rather than silently discarded, so a manifest whose version field holds
         * something that is not a version is refused instead of read as one this build happens
         * to satisfy. A '-' or '+' suffix is accepted because Mila's own versions carry one.
         */
        static bool parseVersion( std::string_view text, int& major, int& minor, int& patch )
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
         * A version comparison here beats a parse error somewhere inside the tensor index,
         * which is what a format change would otherwise look like.
         */
        static void requireCompatibleVersion(
            const ModelCoordinate& coordinate,
            const std::string& variant_name,
            const nlohmann::json& variant )
        {
            const std::string minimum = variant.value( "minimum_mila_version", std::string{} );

            if ( minimum.empty() )
            {
                return;
            }

            int major = 0;
            int minor = 0;
            int patch = 0;

            if ( !parseVersion( minimum, major, minor, patch ) )
            {
                throw std::runtime_error( std::format(
                    "{}: variant '{}' has an unparseable minimum_mila_version '{}'",
                    coordinate.toString(), variant_name, minimum ) );
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
                    "{}: variant '{}' requires Mila {} or newer; this build is {}",
                    coordinate.toString(), variant_name, minimum, current.toString() ) );
            }
        }

        /**
         * @brief Fetch one declared file and append it to the record being built.
         */
        void fetchDeclaredFile(
            const ModelCoordinate& coordinate,
            const nlohmann::json& files,
            const std::string& role,
            bool required,
            ModelRecord& record )
        {
            if ( !files.contains( role ) )
            {
                if ( required )
                {
                    throw std::runtime_error( std::format(
                        "{}: manifest declares no '{}' file", coordinate.toString(), role ) );
                }

                return;
            }

            const nlohmann::json& entry = files[ role ];

            if ( !entry.is_object() || !entry.contains( "path" ) || !entry.contains( "sha256" ) )
            {
                throw std::runtime_error( std::format(
                    "{}: '{}' entry needs both a path and a sha256",
                    coordinate.toString(), role ) );
            }

            ModelFile file;
            file.role = role;
            file.path = entry[ "path" ].get<std::string>();
            file.sha256 = entry[ "sha256" ].get<std::string>();
            file.bytes = entry.value( "bytes", uint64_t{ 0 } );

            const IModelHub& hub = hub_;
            const ModelCoordinate& source = coordinate;
            const std::string path = file.path;

            store_.ensureBlob(
                std::format( "{} ({})", path, coordinate.toString() ),
                file.sha256,
                [&hub, &source, path]( uint64_t resume_from,
                    const std::function<bool( const char*, size_t )>& sink ) -> FetchReport
                {
                    return hub.fetchFile( source, path, resume_from, sink );
                } );

            record.files.push_back( std::move( file ) );
        }

        ModelStore& store_;
        const IModelHub& hub_;
    };
}
