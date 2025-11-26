/**
 * @file ModelArchive.ixx
 * @brief Structured archive helper used by component save/load implementations.
 *
 * Provides a small scoping API so callers may set a logical namespace (scope)
 * for subsequent read/write calls. This keeps leaf `save_()` implementations
 * scope-relative and avoids repetitive path concatenation at each component.
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
     * @brief ModelArchive provides high-level helpers for component serialization.
     *
     * Responsibilities:
     *  - Coordinate ModelSerializer lifecycle (open/close)
     *  - Write/read structured JSON metadata files
     *  - Write/read arbitrary binary blobs
     *  - Enforce ArchiveMode for type-safe operations
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
        // Scoping helpers
        // ====================================================================

        /**
         * @brief Push a logical scope onto the archive scope stack.
         *
         * The scope is a path fragment (for example "components/<name>" or "network").
         * Subsequent read/write calls that use relative paths will be resolved under
         * the composed scope. Use popScope() to restore the previous scope.
         */
        void pushScope( const std::string& scope )
        {
            // Normalize: remove leading/trailing slashes to avoid double separators.
            std::string s = scope;
            while (!s.empty() && s.front() == '/') s.erase( s.begin() );
            while (!s.empty() && s.back() == '/') s.pop_back();

            if (!s.empty())
            {
                scope_stack_.push_back( s );
            }
            else
            {
                // Pushing an empty scope is a no-op.
            }
        }

        /**
         * @brief Pop the most recently pushed scope.
         *
         * @throws std::runtime_error if no scope to pop.
         */
        void popScope()
        {
            if (scope_stack_.empty())
            {
                throw std::runtime_error( "ModelArchive::popScope: no scope to pop" );
            }
            scope_stack_.pop_back();
        }

        /**
         * @brief RAII helper that pushes a scope and pops it on destruction.
         *
         * Use this to ensure scoped path is restored even in exception cases.
         */
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
                if (active_)
                {
                    try
                    {
                        archive_.popScope();
                    }
                    catch (...)
                    {
                        // Suppress exceptions from destructor
                    }
                }
            }

            // Disable copy
            ScopedScope( const ScopedScope& ) = delete;
            ScopedScope& operator=( const ScopedScope& ) = delete;

            // Allow move
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

            const std::string full = scopedPath( path );
            return serializer_->hasFile( full );
        }

        size_t getFileSize( const std::string& path ) const
        {
            requireOpen( "getFileSize" );

            const std::string full = scopedPath( path );
            return serializer_->getFileSize( full );
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
         * If a scope is active and `path` is relative, the path will be resolved
         * under the active scope. Paths that already begin with a top-level root
         * (for example "network/" or "components/") are treated as absolute and
         * are not prefixed.
         *
         * @throws std::runtime_error if archive is not in Write mode
         * @throws std::runtime_error on serialization failure
         */
        void writeJson( const std::string& path, const json& j )
        {
            requireOpen( "writeJson" );
            requireMode( OpenMode::Write, "writeJson" );

            const std::string full = scopedPath( path );

            std::string s = j.dump(2);
            if (!serializer_->addData( full, s.data(), s.size() ))
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::writeJson failed for path: {}", full ) );
            }
        }

        /**
         * @brief Write raw binary blob to archive.
         *
         * If a scope is active and `path` is relative, the path will be resolved
         * under the active scope. Paths that already begin with a top-level root
         * (for example "network/" or "components/") are treated as absolute.
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

            const std::string full = scopedPath( path );

            if (!serializer_->addData( full, data, size ))
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::writeBlob failed for path: {}", full ) );
            }
        }

        // ====================================================================
        // Read Operations
        // ====================================================================

        /**
         * @brief Read and parse JSON object from `path`.
         *
         * If a scope is active and `path` is relative, the path will be resolved
         * under the active scope. Paths that already begin with a top-level root
         * (for example "network/" or "components/") are treated as absolute.
         *
         * @throws std::runtime_error if archive is not in Read mode
         * @throws std::runtime_error if file missing or parse fails
         */
        json readJson( const std::string& path ) const
        {
            requireMode( OpenMode::Read, "readJson" );

            const std::string full = scopedPath( path );

            size_t sz = serializer_->getFileSize( full );
            if (sz == 0)
            {
                throw std::runtime_error( "ModelArchive::readJson missing file: " + full );
            }

            std::string s( sz, '\0' );
            size_t read = serializer_->extractData( full, s.data(), sz );
            if (read != sz)
            {
                throw std::runtime_error( "ModelArchive::readJson failed to read entire file: " + full );
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
         * If a scope is active and `path` is relative, the path will be resolved
         * under the active scope. Paths that already begin with a top-level root
         * (for example "network/" or "components/") are treated as absolute.
         *
         * @throws std::runtime_error if archive is not in Read mode
         * @throws std::runtime_error if file missing or read fails
         */
        std::vector<uint8_t> readBlob( const std::string& path ) const
        {
            requireOpen( "readBlob" );
            requireMode( OpenMode::Read, "readBlob" );

            const std::string full = scopedPath( path );

            if (!serializer_->hasFile( full ))
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlob: file not found: {}", full ) );
            }

            size_t sz = serializer_->getFileSize( full );
            if (sz == 0)
            {
                return {};  // Empty file -> empty vector
            }

            std::vector<uint8_t> buf( sz );
            size_t read = serializer_->extractData( full, buf.data(), sz );
            if (read != sz)
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlob: incomplete read from {}", full ) );
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

            const std::string full = scopedPath( path );

            if (!serializer_->hasFile( full ))
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlobInto: file not found: {}", full ) );
            }

            size_t file_size = serializer_->getFileSize( full );
            if (file_size > buffer_size)
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlobInto: buffer too small ({} < {})",
                        buffer_size, file_size ) );
            }

            size_t read = serializer_->extractData( full, buffer, file_size );
            if (read != file_size)
            {
                throw std::runtime_error(
                    std::format( "ModelArchive::readBlobInto: incomplete read from {}", full ) );
            }

            return read;
        }

        /**
     * @brief Add metadata key-value pair.
     */
        void addMetadata( const std::string& key, const std::string& value )
        {
            serializer_->addMetadata( key, value );
        }

        /**
         * @brief Retrieve metadata value.
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

        // Scope stack: top is back()
        std::vector<std::string> scope_stack_;

        // Top-level roots treated as absolute paths (do not prefix with scope)
        static inline const std::vector<std::string> kAbsoluteRoots = { "network/", "components/", "modules/" };

        // Compose current scope prefix, or empty string if no scope
        std::string currentPrefix() const
        {
            if (scope_stack_.empty()) return std::string();

            std::string p;
            for (size_t i = 0; i < scope_stack_.size(); ++i)
            {
                if (i) p.push_back( '/' );
                p += scope_stack_[i];
            }

            // Ensure trailing slash
            if (!p.empty() && p.back() != '/') p.push_back( '/' );

            return p;
        }

        // Determine if caller provided an absolute path that should not be prefixed
        bool isAbsolutePath( const std::string& path ) const noexcept
        {
            if (path.empty()) return false;
            if (path.front() == '/') return true;

            for (const auto& root : kAbsoluteRoots)
            {
                if (path.rfind( root, 0 ) == 0) return true;
            }

            return false;
        }

        // Resolve a user-specified path into the final archive path using current scope.
        std::string scopedPath( const std::string& path ) const
        {
            if (isAbsolutePath( path ) || scope_stack_.empty())
            {
                return path;
            }

            return currentPrefix() + path;
        }

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