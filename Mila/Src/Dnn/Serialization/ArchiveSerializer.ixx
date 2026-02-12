/**
 * @file ArchiveSerializer.ixx
 * @brief Interface for hierarchical archive formats (ZIP, tar, etc.).
 *
 * Defines operations for reading/writing structured archives with
 * hierarchical paths like "components/layer1/tensors/weight/data.bin"
 */

module;
#include <string>
#include <vector>

export module Serialization.ArchiveSerializer;

import Serialization.Serializer;
import Serialization.OpenMode;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Interface for hierarchical archive serializers.
     *
     * Archive serializers work with hierarchical file structures where
     * resources are organized in directory trees. Used by ModelArchive
     * for Mila's internal checkpoint format.
     *
     * Examples: ZipSerializer, TarSerializer
     */
    export class ArchiveSerializer : public Serializer
    {
    public:
        virtual ~ArchiveSerializer() = default;

        /**
         * @brief Add a binary blob to the archive at a hierarchical path.
         *
         * @param path Hierarchical path (e.g., "network/components/layer1/weight.bin")
         * @param data Pointer to source bytes
         * @param size Number of bytes
         * @return true on success
         */
        [[nodiscard]] virtual bool addData(const std::string& path, const void* data, size_t size) = 0;

        /**
         * @brief Extract a file from the archive.
         *
         * @param path Hierarchical path
         * @param data Pre-allocated buffer
         * @param size Buffer size
         * @return Number of bytes extracted, or 0 on error
         */
        [[nodiscard]] virtual size_t extractData(const std::string& path, void* data, size_t size) = 0;

        /**
         * @brief Get file size at path.
         */
        [[nodiscard]] virtual size_t getFileSize(const std::string& path) const = 0;

        /**
         * @brief List all files in the archive.
         */
        [[nodiscard]] virtual std::vector<std::string> listFiles() const = 0;

        /**
         * @brief Add metadata entry.
         */
        [[nodiscard]] virtual bool addMetadata(const std::string& key, const std::string& value) = 0;

        /**
         * @brief Get metadata entry.
         */
        [[nodiscard]] virtual std::string getMetadata(const std::string& key) const = 0;

        /**
         * @brief Check if file exists at path.
         */
        [[nodiscard]] virtual bool hasFile(const std::string& path) const = 0;
    };
}