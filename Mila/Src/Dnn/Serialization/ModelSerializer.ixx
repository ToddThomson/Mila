/**
 * @file ModelSerializer.ixx
 * @brief Defines the serialization interface for Mila models and modules
 */

module;
#include <miniz.h>
#include <string>
#include <memory>
#include <filesystem>
#include <fstream>
#include <vector>

export module Dnn.Serialization.ModelSerializer;

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
         * @brief Checks if a file exists in the archive
         * @param path Path within the archive
         * @return True if file exists
         */
        virtual bool hasFile( const std::string& path ) const = 0;
    };
}