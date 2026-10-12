/**
 * @file InstallOptions.ixx
 * @brief How a package is to be installed into the store.
 */

module;
#include <string>

export module Distribution.InstallOptions;

namespace Mila::Distribution
{
    /**
     * @brief How a package is to be installed.
     *
     * A package carries its own name; these are the overrides. Nothing here names an origin,
     * because installing from a package is what "published from this machine" means.
     */
    export struct InstallOptions
    {
        /// Empty takes the manifest's name, then the package directory's name.
        std::string name;

        /**
         * @brief Replace an existing record of the same name.
         *
         * Off by default. A name is unique across the store, so installing over one is either a
         * refresh the caller meant or a collision they need to know about, and the store cannot
         * tell which.
         */
        bool replace{ false };

        /**
         * @brief Move the package's files into the store rather than copying them.
         *
         * On by default. A move is free on one volume, and it keeps a single integrity model in
         * which the path is the digest -- a copy leaves a second, unmanaged copy of a file that
         * may be several gigabytes. Turn it off to keep the package for a hub upload.
         */
        bool move_files{ true };
    };
}
