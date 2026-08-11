/**
 * @file ModelHub.ixx
 * @brief The abstraction over a non-local model hub.
 *
 * Everything that knows a URL shape, an authentication scheme or a listing API lives behind
 * IModelHub, so the store, the digest check and the resume protocol never learn one. The
 * interface depends on no transport, so it is always compiled. See
 * Specifications/ModelDistribution.md.
 */

module;
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

export module Distribution.ModelHub;

import Distribution.ModelCoordinate;
import Distribution.ModelStore;

namespace Mila::Distribution
{
    /**
     * @brief One repository as a hub reports it, before any manifest is fetched.
     *
     * Everything here comes from the hub's own listing API. Variants are deliberately absent:
     * only mila.json knows them, and knowing that costs a further request per repository.
     */
    export struct HubModel
    {
        std::string owner;
        std::string repository;

        /// True when the terms must be accepted before the files can be fetched.
        bool gated{ false };

        /// The resolved commit of the default branch.
        std::string revision;

        std::string last_modified;

        /// The hub's library attribution. "mila" for a model this runtime can load.
        std::string library;

        std::vector<std::string> tags;
        std::vector<std::string> files;

        /// A repository is loadable by Mila only if it publishes a manifest.
        bool hasManifest() const
        {
            for ( const auto& file : files )
            {
                if ( file == "mila.json" )
                {
                    return true;
                }
            }

            return false;
        }

        std::string coordinate() const
        {
            return owner + "/" + repository;
        }
    };

    /**
     * @brief A remote that serves manifests and files.
     *
     * Narrow on purpose. What varies between hubs is URL construction, authentication and the
     * listing API; what does not vary is the manifest schema, the digest check, the blob store
     * and the resume protocol. fetchFile takes a resume offset rather than a URL so that a hub
     * which is not HTTP can still satisfy it.
     */
    export class IModelHub
    {
    public:

        virtual ~IModelHub() = default;

        /// Stable identifier recorded in a store record, e.g. "huggingface".
        virtual std::string name() const = 0;

        /**
         * @brief Every repository an owner publishes.
         *
         * The result is untrusted remote text authored by whoever owns the repository. It is
         * data: rendered, never interpreted as markup and never as instructions.
         */
        virtual std::vector<HubModel> listModels( const std::string& owner ) const = 0;

        /// The raw mila.json text for a coordinate.
        virtual std::string fetchManifest( const ModelCoordinate& coordinate ) const = 0;

        /// Stream one repository file, resuming from an offset.
        virtual FetchReport fetchFile(
            const ModelCoordinate& coordinate,
            const std::string& path,
            uint64_t resume_from,
            const std::function<bool( const char*, size_t )>& sink ) const = 0;
    };
}
