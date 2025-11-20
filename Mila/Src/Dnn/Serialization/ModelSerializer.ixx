/**
 * @file ModelSerializer.ixx
 * @brief Defines the serialization interface for Mila models and modules
 */

module;
#include <string>
#include <vector>

export module Serialization.ModelSerializer;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Abstract interface for model and module serialization
     */
    export class ModelSerializer {
    public:
        virtual ~ModelSerializer() = default;

        /**
         * @brief Opens an archive for writing
         * @param filename Path to the archive file
         * @return True if archive was successfully opened
         */
        virtual bool openForWrite( const std::string& filename ) = 0;

        /**
         * @brief Opens an archive for reading
         * @param filename Path to the archive file
         * @return True if archive was successfully opened
         */
        virtual bool openForRead( const std::string& filename ) = 0;

        /**
         * @brief Closes the current archive
         * @return True if archive was successfully closed
         */
        virtual bool close() = 0;

        /**
         * @brief Adds a memory buffer to the archive
         * @param path Path within the archive
         * @param data Pointer to the data
         * @param size Size of the data in bytes
         * @return True if successful
         */
        virtual bool addData( const std::string& path, const void* data, size_t size ) = 0;

        /**
         * @brief Extracts data from the archive
         * @param path Path within the archive
         * @param data Pointer to the buffer where data should be stored
         * @param size Size of the buffer in bytes
         * @return Actual size of extracted data, 0 if failed
         */
        virtual size_t extractData( const std::string& path, void* data, size_t size ) = 0;

        /**
         * @brief Gets the size of a file in the archive without extracting
         * @param path Path within the archive
         * @return Size in bytes, or 0 if file doesn't exist
         */
        virtual size_t getFileSize( const std::string& path ) const = 0;

        /**
         * @brief Lists all files in the archive
         * @return Vector of file paths
         */
        virtual std::vector<std::string> listFiles() const = 0;

        /**
         * @brief Adds metadata (version, timestamp, etc.)
         * @param key Metadata key
         * @param value Metadata value
         * @return True if successful
         */
        virtual bool addMetadata( const std::string& key, const std::string& value ) = 0;

        /**
         * @brief Retrieves metadata
         * @param key Metadata key
         * @return Metadata value, empty if not found
         */
        virtual std::string getMetadata( const std::string& key ) const = 0;

        /**
         * @brief Checks if a file exists in the archive
         * @param path Path within the archive
         * @return True if file exists
         */
        virtual bool hasFile( const std::string& path ) const = 0;
    };
}