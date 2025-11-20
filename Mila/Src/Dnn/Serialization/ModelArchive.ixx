/**
 * @file ModelArchive.ixx
 * @brief Structured archive helper used by module save/load implementations.
 *
 * Provides convenient JSON + blob helpers backed by a ModelSerializer
 * implementation (e.g. ZipSerializer). Designed to make module `save()` /
 * `load()` implementations concise and deterministic.
 */

module;
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <utility>
#include <cstdint>
#include <exception>

export module Serialization.ModelArchive;

import Serialization.ModelSerializer;
import nlohmann.json;

namespace Mila::Dnn::Serialization
{
    using nlohmann::json;

    /**
     * @brief ModelArchive provides high-level helpers for module serialization.
     *
     * Responsibilities:
     *  - Write/read structured JSON metadata files
     *  - Write/read arbitrary binary blobs
     *
     * Note: tensor-specific helpers and metadata types live in the
     *       `Serialization.Tensor` partition to avoid coupling ModelArchive
     *       to tensor implementation details.
     */
    export class ModelArchive
    {
    public:
        ModelArchive() = default;

        /**
         * @brief Construct with an existing serializer implementation.
         *
         * The serializer may already be opened for read/write.
         */
        explicit ModelArchive( std::shared_ptr<ModelSerializer> serializer )
            : serializer_( std::move( serializer ) )
        {
        }

        ~ModelArchive() = default;

        // ---------- Serializer management ----------

        /**
         * @brief Attach a serializer to the archive.
         *
         * The serializer must be opened (openForRead/openForWrite) before using
         * read/write helpers.
         */
        void setSerializer( std::shared_ptr<ModelSerializer> serializer )
        {
            serializer_ = std::move( serializer );
        }

        /**
         * @brief Returns true if a serializer is attached and appears ready.
         */
        bool hasSerializer() const noexcept
        {
            return static_cast<bool>(serializer_);
        }

        // ---------- Basic file helpers forwarded to serializer ----------

        bool hasFile( const std::string& path ) const
        {
            requireSerializer();
            return serializer_->hasFile( path );
        }

        size_t getFileSize( const std::string& path ) const
        {
            requireSerializer();
            return serializer_->getFileSize( path );
        }

        std::vector<std::string> listFiles() const
        {
            requireSerializer();
            return serializer_->listFiles();
        }

        // ---------- JSON helpers ----------

        /**
         * @brief Write a JSON object to the archive at `path`.
         *
         * Throws std::runtime_error on failure.
         */
        void writeJson( const std::string& path, const json& j ) const
        {
            requireSerializer();
            std::string s = j.dump();
            if (!serializer_->addData( path, s.data(), s.size() ))
            {
                throw std::runtime_error( "ModelArchive::writeJson failed for path: " + path );
            }
        }

        /**
         * @brief Read and parse JSON object from `path`.
         *
         * Throws std::runtime_error if file missing or parse fails.
         */
        json readJson( const std::string& path ) const
        {
            requireSerializer();
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

        // ---------- Binary blob helpers ----------

        /**
         * @brief Write raw binary blob to archive.
         *
         * Throws std::runtime_error on failure.
         */
        void writeBlob( const std::string& path, const void* data, size_t size ) const
        {
            requireSerializer();
            if (!serializer_->addData( path, data, size ))
            {
                throw std::runtime_error( "ModelArchive::writeBlob failed for path: " + path );
            }
        }

        /**
         * @brief Read raw binary blob from archive into returned vector.
         *
         * Throws std::runtime_error if file missing or read fails.
         */
        std::vector<uint8_t> readBlob( const std::string& path ) const
        {
            requireSerializer();
            size_t sz = serializer_->getFileSize( path );
            if (sz == 0)
            {
                throw std::runtime_error( "ModelArchive::readBlob missing file: " + path );
            }

            std::vector<uint8_t> buf( sz );
            size_t read = serializer_->extractData( path, buf.data(), sz );
            if (read != sz)
            {
                throw std::runtime_error( "ModelArchive::readBlob failed to read entire file: " + path );
            }

            return buf;
        }

    private:
        std::shared_ptr<ModelSerializer> serializer_{ nullptr };

        void requireSerializer() const
        {
            if (!serializer_)
            {
                throw std::runtime_error( "ModelArchive: no serializer attached" );
            }
        }
    };
}