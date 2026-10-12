/**
 * @file InstallRequest.ixx
 * @brief Where a package is going: the local store, under a name.
 */

module;
#include <filesystem>
#include <string>

export module Tools.ExportArtifact.InstallRequest;

namespace Mila::Tools
{
    /**
     * @brief Where a package is going: the local store, with a coordinate to go under.
     */
    export struct InstallRequest
    {
        std::filesystem::path package_directory;

        /// Empty takes the manifest's name, then the package directory's name.
        std::string name;

        /// Replace an existing record of the same name rather than refusing the collision.
        bool replace{ false };

        /// Copy into the store rather than moving, leaving the package intact for a hub upload.
        bool keep_package{ false };
    };
}
