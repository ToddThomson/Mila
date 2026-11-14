/**
 * @file LocalDirectoryHub.ixx
 * @brief A folder of model packages standing in for a publisher: each subfolder with a mila.json is one model,
 *        served from disk as a hub serves it from the network.
 */

export module Distribution.LocalDirectoryHub;

import std;
import Distribution.ModelCoordinate;
import Distribution.FetchReport;
import Distribution.ModelHub;

namespace Mila::Distribution
{
    /**
     * @brief The environment variable that names a folder of packages to use in place of the publisher.
     *
     * Read by makeDefaultModelHub, so every program that installs a model -- Chat, mila-store, Mila Assistant --
     * reads the same folder. A package is laid out as ExportArtifact --package writes one.
     */
    export inline constexpr const char* kHubDirectoryVariable = "MILA_HUB_DIRECTORY";

    /**
     * @brief A folder of packages served as a hub: the listing is its subfolders, the manifest and files are read
     *        from disk.
     *
     * A folder has no owners, so every owner asked for lists the same packages. A path a manifest names is untrusted
     * text: one that leaves its package's folder is refused.
     */
    export class LocalDirectoryHub : public IModelHub
    {
    public:

        /// Reports each file's bytes as it is read, as the network hub reports them as they arrive.
        using Progress = std::function<bool( std::uint64_t received, std::uint64_t total )>;

        explicit LocalDirectoryHub( std::filesystem::path root, Progress progress = {} );

        std::string name() const override;

        std::vector<HubModel> listModels( const std::string& owner ) const override;

        std::string fetchManifest( const ModelCoordinate& coordinate ) const override;

        FetchReport fetchFile(
            const ModelCoordinate& coordinate,
            const std::string& path,
            std::uint64_t resume_from,
            const std::function<bool( const char*, std::size_t )>& sink ) const override;

        std::string fetchFilePart(
            const ModelCoordinate& coordinate,
            const std::string& path,
            std::uint64_t first,
            std::uint64_t count ) const override;

    private:

        /// The file a package names, refused when it would leave the package's folder.
        std::filesystem::path fileOf( const ModelCoordinate& coordinate, const std::string& path ) const;

        std::filesystem::path root_;
        Progress progress_;
    };
}
