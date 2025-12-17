/**
 * @file ModelArchive.ixx
 * @brief Structured archive helper used by component save/load implementations.
 *
 * Provides scoping API and type-safe metadata access without exposing JSON implementation.
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
import Serialization.Metadata;
import nlohmann.json;

namespace Mila::Dnn::Serialization
{
    using json = nlohmann::json;

    /**
     * @brief ModelArchive provides high-level helpers for component serialization.
     *
     * Responsibilities:
     *  - Coordinate ModelSerializer lifecycle (open/close)
     *  - Write/read type-safe metadata (abstracts JSON implementation)
     *  - Write/read arbitrary binary blobs
     *  - Enforce OpenMode for type-safe operations
     *
     * Metadata Abstraction:
     * - User code interacts with SerializationMetadata (type-safe, format-agnostic)
     * - ModelArchive handles conversion to/from JSON internally
     * - Future format changes (XML, binary, etc.) won't break user code
     *
     * A simple scoping API (pushScope/popScope / ScopedScope) allows callers to
     * set a logical directory prefix used by subsequent read/write operations.
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
         * @param filepath Path to the archive file
         * @param serializer Owned serializer instance (typically ZipSerializer)
         * @param mode Read or Write mode for the archive
         *
         * @throws std::invalid_argument if serializer is null
         * @throws std::runtime_error if serializer cannot be opened in the specified mode
         */
        explicit ModelArchive( const std::string& filepath, std::unique_ptr<ModelSerializer> serializer, OpenMode mode )
            : filepath_( filepath ), serializer_( std::move( serializer ) ), mode_( mode ), closed_( false )
        {
            if ( !serializer_ )
            {
                throw std::invalid_argument( "ModelArchive requires a non-null serializer" );
            }

            if ( !serializer_->open( filepath, mode ) )
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
            if ( !closed_ && serializer_ )
            {
                try
                {
                    close();
                }
                catch ( ... )
                {
                }
            }
        }

        ModelArchive( const ModelArchive& ) = delete;
        ModelArchive& operator=( const ModelArchive& ) = delete;

        ModelArchive( ModelArchive&& other ) noexcept
            : serializer_( std::move( other.serializer_ ) )
            , filepath_( std::move( other.filepath_ ) )
            , mode_( other.mode_ )
            , closed_( other.closed_ )
        {
            other.closed_ = true;
        }

        ModelArchive& operator=( ModelArchive&& other ) noexcept
        {
            if ( this != &other )
            {
                if ( !closed_ )
                {
                    try
                    {
                        close();
                    }
                    catch ( ... )
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

        OpenMode getMode() const noexcept
        {
            return mode_;
        }

        const std::string& getFilepath() const noexcept
        {
            return filepath_;
        }

        bool isClosed() const noexcept
        {
            return closed_;
        }

        void close()
        {
            if ( closed_ ) return;

            if ( !serializer_ )
            {
                closed_ = true;
                return;
            }

            try
            {
                serializer_->close();
                closed_ = true;
            }
            catch ( const std::exception& e )
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::close failed for '{}': {}", filepath_, e.what() ) );
            }
        }

        // ====================================================================
        // Scoping helpers
        // ====================================================================

        void pushScope( const std::string& scope )
        {
            std::string s = scope;
            while ( !s.empty() && s.front() == '/' ) s.erase( s.begin() );
            while ( !s.empty() && s.back() == '/' ) s.pop_back();

            if ( !s.empty() )
            {
                scope_stack_.push_back( s );
            }
        }

        void popScope()
        {
            if ( scope_stack_.empty() )
            {
                throw std::runtime_error( "ModelArchive::popScope: no scope to pop" );
            }
            scope_stack_.pop_back();
        }

        class ScopedScope
        {
        public:
            ScopedScope( ModelArchive& archive, const std::string& scope )
                : archive_( archive ), active_( true )
            {
                archive_.pushScope( scope );
            }

            ~ScopedScope()
            {
                if ( active_ )
                {
                    try
                    {
                        archive_.popScope();
                    }
                    catch ( ... )
                    {
                    }
                }
            }

            ScopedScope( const ScopedScope& ) = delete;
            ScopedScope& operator=( const ScopedScope& ) = delete;

            ScopedScope( ScopedScope&& other ) noexcept
                : archive_( other.archive_ ), active_( other.active_ )
            {
                other.active_ = false;
            }

        private:
            ModelArchive& archive_;
            bool active_;
        };

        // ====================================================================
        // File Query Operations
        // ====================================================================

        bool hasFile( const std::string& path ) const
        {
            requireOpen( "hasFile" );
            return serializer_->hasFile( scopedPath( path ) );
        }

        size_t getFileSize( const std::string& path ) const
        {
            requireOpen( "getFileSize" );
            return serializer_->getFileSize( scopedPath( path ) );
        }

        std::vector<std::string> listFiles() const
        {
            requireOpen( "listFiles" );
            return serializer_->listFiles();
        }

        // ====================================================================
        // Type-Safe Metadata Write/Read (USER-FACING API)
        // ====================================================================

        /**
         * @brief Write type-safe metadata to the archive.
         *
         * User-facing API that abstracts JSON serialization format.
         * Components use SerializationMetadata to build metadata without
         * knowledge of underlying format.
         *
         * @param path Archive path for metadata file
         * @param metadata Type-safe metadata container
         *
         * @throws std::runtime_error if archive is not in Write mode
         * @throws std::runtime_error on serialization failure
         *
         * @example
         * SerializationMetadata meta;
         * meta.set("type", "Linear")
         *     .set("version", 1)
         *     .set("input_features", 128);
         * archive.writeMetadata("meta.json", meta);
         */
        void writeMetadata( const std::string& path, const SerializationMetadata& metadata )
        {
            requireOpen( "writeMetadata" );
            requireMode( OpenMode::Write, "writeMetadata" );

            json j = metadata.toJson();

            const std::string full = scopedPath( path );

            std::string s = j.dump( 2 );
            if ( !serializer_->addData( full, s.data(), s.size() ) )
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::writeMetadata failed for path: {}", full ) );
            }
        }

        /**
         * @brief Read type-safe metadata from the archive.
         *
         * User-facing API that abstracts JSON deserialization format.
         * Components use SerializationMetadata to read metadata without
         * knowledge of underlying format.
         *
         * @param path Archive path for metadata file
         * @return Type-safe metadata container
         *
         * @throws std::runtime_error if archive is not in Read mode
         * @throws std::runtime_error if file missing or parse fails
         *
         * @example
         * auto meta = archive.readMetadata("meta.json");
         * std::string type = meta.getString("type");
         * int64_t version = meta.getInt("version");
         * int64_t features = meta.getInt("input_features");
         */
        SerializationMetadata readMetadata( const std::string& path ) const
        {
            requireMode( OpenMode::Read, "readMetadata" );

            const std::string full = scopedPath( path );

            size_t sz = serializer_->getFileSize( full );
            if ( sz == 0 )
            {
                throw std::runtime_error( "ModelArchive::readMetadata missing file: " + full );
            }

            std::string s( sz, '\0' );
            size_t read = serializer_->extractData( full, s.data(), sz );
            if ( read != sz )
            {
                throw std::runtime_error( "ModelArchive::readMetadata failed to read entire file: " + full );
            }

            try
            {
                json j = json::parse( s );
                return SerializationMetadata::fromJson( j );
            }
            catch ( const std::exception& e )
            {
                throw std::runtime_error( std::string( "ModelArchive::readMetadata parse error: " ) + e.what() );
            }
        }

        // ====================================================================
        // Binary Blob Operations
        // ====================================================================

        /**
         * @brief Write raw binary blob to archive.
         *
         * @param path Archive path for binary data
         * @param data Pointer to source bytes
         * @param size Number of bytes to write
         *
         * @throws std::runtime_error if archive is not in Write mode
         * @throws std::runtime_error on write failure
         */
        void writeBlob( const std::string& path, const void* data, size_t size )
        {
            requireOpen( "writeBlob" );
            requireMode( OpenMode::Write, "writeBlob" );

            if ( !data && size > 0 )
            {
                throw std::invalid_argument( "ModelArchive::writeBlob: null data with non-zero size" );
            }

            const std::string full = scopedPath( path );

            if ( !serializer_->addData( full, data, size ) )
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::writeBlob failed for path: {}", full ) );
            }
        }

        /**
         * @brief Read raw binary blob from archive into returned vector.
         *
         * @param path Archive path for binary data
         * @return Vector containing file contents
         *
         * @throws std::runtime_error if archive is not in Read mode
         * @throws std::runtime_error if file missing or read fails
         */
        std::vector<uint8_t> readBlob( const std::string& path ) const
        {
            requireOpen( "readBlob" );
            requireMode( OpenMode::Read, "readBlob" );

            const std::string full = scopedPath( path );

            if ( !serializer_->hasFile( full ) )
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlob: file not found: {}", full ) );
            }

            size_t sz = serializer_->getFileSize( full );
            if ( sz == 0 )
            {
                return {};
            }

            std::vector<uint8_t> buf( sz );
            size_t read = serializer_->extractData( full, buf.data(), sz );
            if ( read != sz )
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlob: incomplete read from {}", full ) );
            }

            return buf;
        }

        /**
         * @brief Read binary blob directly into provided buffer.
         *
         * @param path Archive path
         * @param buffer Pre-allocated buffer
         * @param buffer_size Size of buffer
         * @return Number of bytes read
         *
         * @throws std::runtime_error if file missing or buffer too small
         */
        size_t readBlobInto( const std::string& path, void* buffer, size_t buffer_size ) const
        {
            requireOpen( "readBlobInto" );
            requireMode( OpenMode::Read, "readBlobInto" );

            const std::string full = scopedPath( path );

            if ( !serializer_->hasFile( full ) )
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlobInto: file not found: {}", full ) );
            }

            size_t file_size = serializer_->getFileSize( full );
            if ( file_size > buffer_size )
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlobInto: buffer too small ({} < {})",
                        buffer_size, file_size ) );
            }

            size_t read = serializer_->extractData( full, buffer, file_size );
            if ( read != file_size )
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlobInto: incomplete read from {}", full ) );
            }

            return read;
        }

        /**
         * @brief Add simple metadata key-value pair.
         *
         * For simple string metadata (archive-level tags, format markers, etc.).
         * Use writeMetadata() for structured component metadata.
         */
        void addMetadata( const std::string& key, const std::string& value )
        {
            serializer_->addMetadata( key, value );
        }

        /**
         * @brief Retrieve simple metadata value.
         *
         * For simple string metadata (archive-level tags, format markers, etc.).
         * Use readMetadata() for structured component metadata.
         */
        std::string getMetadata( const std::string& key ) const
        {
            return serializer_->getMetadata( key );
        }

    private:
        std::unique_ptr<ModelSerializer> serializer_;
        std::string filepath_;
        OpenMode mode_;
        bool closed_{ false };
        std::vector<std::string> scope_stack_;

        static inline const std::vector<std::string> kAbsoluteRoots = { "network/", "components/", "modules/" };

        std::string currentPrefix() const
        {
            if ( scope_stack_.empty() ) return std::string();

            std::string p;
            for ( size_t i = 0; i < scope_stack_.size(); ++i )
            {
                if ( i ) p.push_back( '/' );
                p += scope_stack_[ i ];
            }

            if ( !p.empty() && p.back() != '/' ) p.push_back( '/' );

            return p;
        }

        bool isAbsolutePath( const std::string& path ) const noexcept
        {
            if ( path.empty() ) return false;
            if ( path.front() == '/' ) return true;

            for ( const auto& root : kAbsoluteRoots )
            {
                if ( path.rfind( root, 0 ) == 0 ) return true;
            }

            return false;
        }

        std::string scopedPath( const std::string& path ) const
        {
            if ( isAbsolutePath( path ) || scope_stack_.empty() )
            {
                return path;
            }

            return currentPrefix() + path;
        }

        void requireMode( OpenMode expected, const char* op ) const
        {
            if ( mode_ != expected )
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::{}: operation requires {} mode, but archive is in {} mode",
                        op, archiveModeToString( expected ), archiveModeToString( mode_ ) ) );
            }
        }

        void requireOpen( const char* op ) const
        {
            if ( closed_ )
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::{}: archive is closed", op ) );
            }
        }
    };
}