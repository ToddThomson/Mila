/**
 * @file ModelArchive.ixx
 * @brief Structured archive helper used by module save/load implementations.
 */

module;
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <utility>
#include <cstdint>
#include <exception>
#include <format>

export module Serialization.ModelArchive;

import Serialization.ModelSerializer;
import Serialization.OpenMode;
import nlohmann.json;

namespace Mila::Dnn::Serialization
{
	using json = nlohmann::json;

    /**
     * @brief ModelArchive provides high-level helpers for module serialization.
     *
     * Responsibilities:
     *  - Coordinate ModelSerializer lifecycle (open/close)
     *  - Write/read structured JSON metadata files
     *  - Write/read arbitrary binary blobs
     *  - Enforce ArchiveMode for type-safe operations
     *
     * ModelArchive takes ownership of a ModelSerializer and manages its lifecycle
     * based on the specified ArchiveMode. The serializer is automatically opened
     * in the constructor and closed in the destructor (RAII).
     */
    export class ModelArchive
    {
    public:
        /**
         * @brief Construct archive with serializer and automatically open for specified mode.
         *
         * The serializer is automatically opened for reading or writing based on the
         * specified mode. Throws if the serializer cannot be opened.
         *
         * @param serializer Owned serializer instance (typically ZipSerializer)
         * @param filepath Path to the archive file
         * @param mode Read or Write mode for the archive
         *
         * @throws std::invalid_argument if serializer is null
         * @throws std::runtime_error if serializer cannot be opened in the specified mode
         */
        explicit ModelArchive( const std::string& filepath, std::unique_ptr<ModelSerializer> serializer, OpenMode mode )
			: filepath_( filepath ), serializer_( std::move( serializer ) ), mode_( mode ), closed_( false )
        {
            if (!serializer_)
            {
                throw std::invalid_argument( "ModelArchive requires a non-null serializer" );
            }

            if (!serializer_->open( filepath, mode ))
            {
                throw std::runtime_error(
                    std::format( "ModelArchive: failed to open '{}' for {}", filepath, archiveModeToString( mode ) ) );
            }
        }

        /**
         * @brief Destructor automatically closes the serializer.
         */
        ~ModelArchive()
        {
            if (!closed_ && serializer_)
            {
                try
                {
                    close();
                }
                catch (...)
                {
                    // Suppress exceptions in destructor
                }
            }
        }

        // Non-copyable, movable
        ModelArchive( const ModelArchive& ) = delete;
        ModelArchive& operator=( const ModelArchive& ) = delete;

        ModelArchive( ModelArchive&& other ) noexcept
            : serializer_( std::move( other.serializer_ ) )
            , filepath_( std::move( other.filepath_ ) )
            , mode_( other.mode_ )
            , closed_( other.closed_ )
        {
            other.closed_ = true;  // Prevent double-close
        }

        ModelArchive& operator=( ModelArchive&& other ) noexcept
        {
            if (this != &other)
            {
                if (!closed_)
                {
                    try
                    {
                        close();
                    }
                    catch (...)
                    {
                    }
                }
                serializer_ = std::move( other.serializer_ );
                filepath_ = std::move( other.filepath_ );
                mode_ = other.mode_;
                closed_ = other.closed_;
                other.closed_ = true;
            }
            return *this;
        }

        /**
         * @brief Get the current archive mode.
         */
        OpenMode getMode() const noexcept
        {
            return mode_;
        }

        /**
         * @brief Get the filepath for this archive.
         */
        const std::string& getFilepath() const noexcept
        {
            return filepath_;
        }

        /**
         * @brief Check if archive is closed
         */
        bool isClosed() const noexcept
        {
            return closed_;
        }

        /**
     * @brief Finalize and close the archive
     *
     * For write mode, ensures all data is flushed.
     * For read mode, releases resources.
     *
     * @throws std::runtime_error if close fails
     */
        void close()
        {
            if (closed_) return;

            if (!serializer_)
            {
                closed_ = true;
                return;
            }

            try
            {
                serializer_->close();
                closed_ = true;
            }
            catch (const std::exception& e)
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::close failed for '{}': {}", filepath_, e.what() ) );
            }
        }

        // ====================================================================
        // File Query Operations
        // ====================================================================

        bool hasFile( const std::string& path ) const
        {
            requireOpen( "hasFile" );
            return serializer_->hasFile( path );
        }

        size_t getFileSize( const std::string& path ) const
        {
            requireOpen( "getFileSize" );
            return serializer_->getFileSize( path );
        }

        std::vector<std::string> listFiles() const
        {
            requireOpen( "listFiles" );
            return serializer_->listFiles();
        }

        // ====================================================================
        // Write Operations
        // ====================================================================

        /**
         * @brief Write a JSON object to the archive at `path`.
         *
         * @throws std::runtime_error if archive is not in Write mode
         * @throws std::runtime_error on serialization failure
         */
        void writeJson( const std::string& path, const json& j )
        {
            requireOpen( "writeJson" );
            requireMode( OpenMode::Write, "writeJson" );

            std::string s = j.dump(2);
            if (!serializer_->addData( path, s.data(), s.size() ))
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::writeJson failed for path: {}", path ) );
            }
        }

        /**
         * @brief Write raw binary blob to archive.
         *
         * @throws std::runtime_error if archive is not in Write mode
         * @throws std::runtime_error on write failure
         */
        void writeBlob( const std::string& path, const void* data, size_t size )
        {
            requireOpen( "writeBlob" );
            requireMode( OpenMode::Write, "writeBlob" );
            
            if (!data && size > 0)
            {
                throw std::invalid_argument( "ModelArchive::writeBlob: null data with non-zero size" );
            }

            if (!serializer_->addData( path, data, size ))
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::writeBlob failed for path: {}", path ) );
            }
        }

        // ====================================================================
        // Read Operations
        // ====================================================================

        /**
         * @brief Read and parse JSON object from `path`.
         *
         * @throws std::runtime_error if archive is not in Read mode
         * @throws std::runtime_error if file missing or parse fails
         */
        json readJson( const std::string& path ) const
        {
            requireMode( OpenMode::Read, "readJson" );

            size_t sz = serializer_->getFileSize( path );
            if (sz == 0)
            {
                throw std::runtime_error( "ModelArchive::readJson missing file: " + path );
            }

            std::string s( sz, '\0' );
            size_t read = serializer_->extractData( path, s.data(), sz );
            if (read != sz)
            {
                throw std::runtime_error( "ModelArchive::readJson failed to read entire file: " + path );
            }

            try
            {
                return json::parse( s );
            }
            catch (const std::exception& e)
            {
                throw std::runtime_error( std::string( "ModelArchive::readJson parse error: " ) + e.what() );
            }
        }

        /**
         * @brief Read raw binary blob from archive into returned vector.
         *
         * @throws std::runtime_error if archive is not in Read mode
         * @throws std::runtime_error if file missing or read fails
         */
        std::vector<uint8_t> readBlob( const std::string& path ) const
        {
            requireOpen( "readBlob" );
            requireMode( OpenMode::Read, "readBlob" );

            if (!serializer_->hasFile( path ))
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlob: file not found: {}", path ) );
            }

            size_t sz = serializer_->getFileSize( path );
            if (sz == 0)
            {
                return {};  // Empty file -> empty vector
                //throw std::runtime_error( "ModelArchive::readBlob missing file: " + path );
            }

            std::vector<uint8_t> buf( sz );
            size_t read = serializer_->extractData( path, buf.data(), sz );
            if (read != sz)
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlob: incomplete read from {}", path ) );
            }

            return buf;
        }

        /**
         * @brief Read binary blob directly into provided buffer
         *
         * @param path Path in archive
         * @param buffer Pre-allocated buffer
         * @param buffer_size Size of buffer
         * @return Number of bytes read
         * @throws std::runtime_error if file missing or buffer too small
         */
        size_t readBlobInto( const std::string& path, void* buffer, size_t buffer_size ) const
        {
            requireOpen( "readBlobInto" );
            requireMode( OpenMode::Read, "readBlobInto" );

            if (!serializer_->hasFile( path ))
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlobInto: file not found: {}", path ) );
            }

            size_t file_size = serializer_->getFileSize( path );
            if (file_size > buffer_size)
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlobInto: buffer too small ({} < {})",
                        buffer_size, file_size ) );
            }

            size_t read = serializer_->extractData( path, buffer, file_size );
            if (read != file_size)
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlobInto: incomplete read from {}", path ) );
            }

            return read;
        }

    private:
        std::unique_ptr<ModelSerializer> serializer_;
        std::string filepath_;
        OpenMode mode_;
		bool closed_{ false };

        void requireMode( OpenMode expected, const char* op ) const
        {
            if (mode_ != expected)
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::{}: operation requires {} mode, but archive is in {} mode",
                        op, archiveModeToString( expected ), archiveModeToString( mode_ ) ) );
            }
        }

        void requireOpen( const char* op ) const
        {
            if (closed_)
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::{}: archive is closed", op ) );
            }
        }
    };
}