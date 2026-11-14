/**
 * @file PublishedModel.ixx
 * @brief A model a publisher offers, as its listing and manifest describe it, and the start of its weights, which is
 *        what planning it before it is installed reads.
 */

export module Distribution.PublishedModel;

import std;
import Distribution.ModelCoordinate;
import Distribution.ModelManifest;
import Distribution.ModelHub;
import Distribution.ModelStore;

namespace Mila::Distribution
{
    /**
     * @brief One repository a publisher offers: where it is, and its manifest when that could be read.
     *
     * A model whose manifest could not be read is still listed, by name, so a listing that lost one request does
     * not lose the model; it cannot be planned or described. See Specifications/Apps/AssistantOnboarding.md.
     */
    export class PublishedModel
    {
    public:

        PublishedModel( ModelCoordinate coordinate, std::optional<ModelManifest> manifest );

        /**
         * @brief Every model the owner publishes with a manifest: one request for the listing and one for each
         *        manifest, which is its size, capabilities, licence and files.
         *
         * @throws std::exception when the listing itself cannot be read.
         */
        [[nodiscard]] static std::vector<PublishedModel> list( const IModelHub& hub, const std::string& owner );

        /// The name it installs under: one repository is one model at one format.
        [[nodiscard]] const std::string& name() const noexcept;

        [[nodiscard]] const ModelCoordinate& coordinate() const noexcept;

        /// Its manifest, or nothing when it could not be read.
        [[nodiscard]] const std::optional<ModelManifest>& manifest() const noexcept;

        /// The bytes an install downloads: every file the manifest lists. Zero when the manifest is unread.
        [[nodiscard]] std::uint64_t downloadBytes() const noexcept;

        /**
         * @brief The start of its weights as a file of its own -- the safetensors header and nothing after it --
         *        which a planner reads as it reads installed weights and which cannot be loaded.
         *
         * Kept in the store under the weights' digest, so a second plan of the same weights reads nothing from the
         * hub, and a republished model, whose digest changes, is read again.
         *
         * @throws std::runtime_error when the manifest is unread or names no weights, or the hub cannot serve the
         *         header, or what it serves is not one.
         */
        [[nodiscard]] std::filesystem::path weightsHeader( const IModelHub& hub, const ModelStore& store ) const;

    private:

        ModelCoordinate coordinate_;
        std::optional<ModelManifest> manifest_;
    };
}
