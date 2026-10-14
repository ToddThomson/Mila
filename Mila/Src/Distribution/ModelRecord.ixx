/**
 * @file ModelRecord.ixx
 * @brief An installed model: what its manifest published, plus how this copy came to be here.
 */

export module Distribution.ModelRecord;

import std;
export import Distribution.ModelFile;
export import Distribution.ModelCapabilities;

namespace Mila::Distribution
{
    /**
     * @brief An installed model: what the manifest published, plus how this copy came to be here.
     *
     * The name is the key and the only thing in the path. Everything about origin is a field, so
     * a record copied out of the store still says what it describes and where it came from.
     */
    export struct ModelRecord
    {
        /// The store's key, and the only name a user types. Unique across the store.
        std::string name;

        std::string architecture;

        /// Which of the family's architectures, as the manifest declared it ("3.1"). Empty when it did not.
        std::string architecture_version;

        /// Display text. The name carries the precision; this says what the bytes are, and nothing keys on it.
        std::string variant;

        std::string weight_quantization;
        std::string minimum_mila_version;

        std::string base_model;
        std::string license;

        /// What the model was trained to do, as its manifest declared it.
        ModelCapabilities capabilities;

        /**
         * @brief Where this copy came from. Never published, and never part of the path.
         *
         * A model published from this machine leaves hub empty. Origin is mutable -- a locally
         * published model may be pushed to a hub later -- which is the reason it is a field
         * rather than a directory: changing it must not move files.
         */
        std::string hub;
        std::string owner;
        std::string repository;

        /// The resolved commit, not the ref that was asked for. Empty for a local install.
        std::string revision;

        std::string installed_at;

        std::vector<ModelFile> files;

        /// True when nothing served this: it was published from this machine.
        bool isLocal() const noexcept { return hub.empty(); }

        /// Where it came from, for display. Never a key.
        std::string origin() const
        {
            if ( hub.empty() )
            {
                return "local";
            }

            return owner.empty() ? hub : hub + ":" + owner + "/" + repository;
        }

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
}
