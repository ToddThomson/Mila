/**
 * @file ModelResolver.ixx
 * @brief Pulls a published model into the store: manifest, variant selection, blobs, record.
 *
 * Composes a hub with a store and knows neither one's internals -- no URL appears here, and no
 * filesystem layout. The hub is an interface, so every case runs offline against a fake.
 * See Specifications/ModelDistribution.md.
 */

module;
#include <cstdint>
#include <filesystem>
#include <format>
#include <functional>
#include <stdexcept>
#include <string>
#include <system_error>

export module Distribution.ModelResolver;

import Distribution.ModelCoordinate;
import Distribution.ModelManifest;
import Distribution.ModelStore;
import Distribution.ModelHub;

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
         * @brief Pull a model by name.
         *
         * The name is all a user gives: the hub and the owner are supplied by the caller, because
         * a store that serves one publisher makes them a constant rather than a decision. A path
         * is refused -- installing is the operation that takes one, and a load reads only the
         * store, where an undescribed file is a model nothing knows the quantization, the
         * architecture or the integrity of.
         *
         * @throws std::runtime_error if the name is path-shaped, if the manifest is malformed, or
         *         if the artifact requires a newer Mila.
         */
        StoredModel pull( const std::string& name, const std::string& owner )
        {
            std::error_code ignored;

            if ( std::filesystem::exists( name, ignored ) )
            {
                throw std::runtime_error( std::format(
                    "'{}' is a path, and a path cannot be pulled. Install it into the store "
                    "instead; only installed models load.", name ) );
            }

            // Checked before it reaches a URL: a path-shaped name would otherwise become a
            // request against a repository that cannot exist, and the 404 would say nothing
            // about the actual mistake.
            if ( !parseCoordinate( owner + "/" + name ).has_value() )
            {
                throw std::runtime_error( std::format(
                    "'{}' is not a model name. Letters, digits, '.', '_' and '-' only.", name ) );
            }

            ModelCoordinate coordinate;
            coordinate.organization = owner;
            coordinate.repository = name;

            return pull( coordinate );
        }

        /**
         * @brief Pull an explicit coordinate, writing its record on success.
         *
         * The record is keyed by the repository name alone -- the owner is provenance, kept in the
         * record and never in the path, so what a user types to pull is what they type to load.
         *
         * The record is written last. Until every blob has verified there is nothing worth
         * recording, and a record naming a blob that never arrived would make a broken model
         * look installed.
         */
        StoredModel pull( const ModelCoordinate& coordinate )
        {
            const std::string source = coordinate.toString();

            const ModelManifest manifest =
                parseModelManifest( hub_.fetchManifest( coordinate ), source );

            requireCompatibleMilaVersion( manifest, source );

            ModelRecord record;
            record.name = coordinate.repository;
            record.architecture = manifest.architecture;
            record.variant = manifest.variant;
            record.weight_quantization = manifest.weight_quantization;
            record.minimum_mila_version = manifest.minimum_mila_version;
            record.base_model = manifest.base_model;
            record.license = manifest.license;
            record.instruct = manifest.instruct;
            record.hub = hub_.name();
            record.owner = coordinate.organization;
            record.repository = coordinate.repository;
            record.revision = coordinate.revision;

            // Every declared file, not a fixed pair of roles: what composes the model is the
            // manifest's statement, and a role this build does not recognize is still part of it.
            for ( const auto& file : manifest.files )
            {
                fetchDeclaredFile( coordinate, file );

                record.files.push_back( file );
            }

            return store_.describe( store_.writeRecord( std::move( record ) ) );
        }

    private:

        /**
         * @brief Fetch one declared file into the store.
         */
        void fetchDeclaredFile( const ModelCoordinate& coordinate, const ModelFile& file )
        {
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
        }

        ModelStore& store_;
        const IModelHub& hub_;
    };
}
