/**
 * @file ModelSerializer.ixx
 * @brief Abstract interface for model serialization.
 *
 * Provides a unified API for persisting and loading neural network models
 * to/from various archive formats (ZIP, tar, etc.). 
 */

module;
#include <string>
#include <vector>

export module Serialization.ModelSerializer;
import Serialization.OpenMode;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Abstract base class for model serialization backends.
     *
     * Defines the interface for reading and writing model data to archive
     * formats. Implementations must manage their own resource lifecycle
     * and maintain explicit open/closed state.
     *
     * Thread-safety: Implementations are not required to be thread-safe.
     * Callers must synchronize access if sharing instances across threads.
     */
    export class ModelSerializer
    {
    public:
        virtual ~ModelSerializer() = default;

        /**
         * @brief Open an archive.
         *
         * Combines read/write open operations under a single entry point.
         *
         * @param filename Filesystem path to the archive.
         * @param mode     Open mode (Read or Write).
         * @return true on success, false on failure.
         */
        [[nodiscard]] virtual bool open( const std::string& filename, OpenMode mode ) = 0;

        /**
         * @brief Finalize and close the archive.
         *
         * Must be safe to call multiple times and on already-closed instances.
         *
         * @return true on success, false on failure.
         */
        [[nodiscard]] virtual bool close() = 0;

        /**
         * @brief Check if the serializer is currently open.
         *
         * @return true if open for reading or writing, false if closed.
         */
        [[nodiscard]] virtual bool isOpen() const noexcept = 0;

        /**
         * @brief Check if the serializer is open for reading.
         *
         * @return true if open for reading.
         */
        [[nodiscard]] virtual bool isOpenForRead() const noexcept = 0;

        /**
         * @brief Check if the serializer is open for writing.
         *
         * @return true if open for writing.
         */
        [[nodiscard]] virtual bool isOpenForWrite() const noexcept = 0;

        /**
         * @brief Get the filename of the currently open archive.
         *
         * @return Filename, or empty string if no archive is open.
         */
        [[nodiscard]] virtual const std::string& getFilename() const noexcept = 0;

        /**
         * @brief Add a binary blob to the archive.
         *
         * Only valid when open for writing.
         *
         * @param path Archive-internal path (e.g., "network/weights.bin").
         * @param data Pointer to source bytes.
         * @param size Number of bytes to write.
         * @return true on success, false on failure.
         */
        [[nodiscard]] virtual bool addData( const std::string& path, const void* data, size_t size ) = 0;

        /**
         * @brief Extract a file from the archive.
         *
         * Only valid when open for reading.
         *
         * @param path Archive-internal path.
         * @param data Pre-allocated buffer to receive contents.
         * @param size Size of the buffer in bytes.
         * @return Number of bytes written, or 0 on error.
         */
        [[nodiscard]] virtual size_t extractData( const std::string& path, void* data, size_t size ) = 0;

        /**
         * @brief Get the uncompressed size of a file in the archive.
         *
         * Only valid when open for reading.
         *
         * @param path Archive-internal path.
         * @return File size in bytes, or 0 if missing or on error.
         */
        [[nodiscard]] virtual size_t getFileSize( const std::string& path ) const = 0;

        /**
         * @brief List all files in the archive.
         *
         * Only valid when open for reading.
         *
         * @return Vector of archive-internal file paths.
         */
        [[nodiscard]] virtual std::vector<std::string> listFiles() const = 0;

        /**
         * @brief Store metadata under the "metadata/" prefix.
         *
         * Only valid when open for writing.
         *
         * @param key Metadata key (stored as "metadata/<key>").
         * @param value Metadata value.
         * @return true on success, false on failure.
         */
        [[nodiscard]] virtual bool addMetadata( const std::string& key, const std::string& value ) = 0;

        /**
         * @brief Retrieve metadata.
         *
         * Only valid when open for reading.
         *
         * @param key Metadata key.
         * @return Metadata value, or empty string if missing or on error.
         */
        [[nodiscard]] virtual std::string getMetadata( const std::string& key ) const = 0;

        /**
         * @brief Check if the archive contains a file.
         *
         * Only valid when open for reading.
         *
         * @param path Archive-internal path.
         * @return true if the file exists.
         */
        [[nodiscard]] virtual bool hasFile( const std::string& path ) const = 0;
    };
}