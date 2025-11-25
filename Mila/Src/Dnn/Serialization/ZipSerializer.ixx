/**
 * @file ZipSerializer.ixx
 * @brief ZIP-based ModelSerializer implementation using miniz.
 *
 * Lightweight serializer providing read/write access to ZIP archives used
 * by the model persistence APIs.
 */

module;
#include <miniz.h>
#include <string>
#include <string_view>
#include <cstring>
#include <vector>
#include <span>
#include <algorithm>
#include <format>
#include <cassert>

export module Serialization.ZipSerializer;

import Serialization.ModelSerializer;
import Utils.Logger;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief ZIP archive serializer built on miniz.
     *
     * Responsibilities:
     *  - Write and read files (binary blobs and JSON) into a ZIP archive.
     *  - Provide simple metadata storage under the "metadata/" prefix.
     *
     * Notes:
     *  - Instances are not thread-safe; callers must synchronize access.
     *  - Move-only type; copying is disabled to prevent resource conflicts.
     *  - Maintains explicit open/closed state for robustness.
     */
    export class ZipSerializer : public ModelSerializer
    {
    public:
        /**
         * @brief Construct an empty, closed ZipSerializer.
         *
         * The underlying miniz archive structure is zero-initialized.
         */
        ZipSerializer() noexcept
        {
            std::memset( &zip_, 0, sizeof( zip_ ) );
        }

        /**
         * @brief Destructor ensures underlying archive is closed.
         *
         * Best-effort cleanup is performed and exceptions are suppressed.
         */
        ~ZipSerializer()
        {
            // Best-effort cleanup on destruction
            try
            {
                close();
            }
            catch (...)
            {
                // Suppress exceptions in destructor
            }
        }

        // Delete copy operations (resource-owning class)
        ZipSerializer( const ZipSerializer& ) = delete;
        ZipSerializer& operator=( const ZipSerializer& ) = delete;

        // Move operations
        ZipSerializer( ZipSerializer&& other ) noexcept
            : zip_( other.zip_ )
            , filename_( std::move( other.filename_ ) )
            , state_( other.state_ )
        {
            std::memset( &other.zip_, 0, sizeof( other.zip_ ) );
            other.filename_.clear();
            other.state_ = State::Closed;
        }

        ZipSerializer& operator=( ZipSerializer&& other ) noexcept
        {
            if (this != &other)
            {
                close();
                zip_ = other.zip_;
                filename_ = std::move( other.filename_ );
                state_ = other.state_;

                std::memset( &other.zip_, 0, sizeof( other.zip_ ) );
                other.filename_.clear();
                other.state_ = State::Closed;
            }
            return *this;
        }

        /**
         * @brief Open an archive for read or write (create/overwrite).
         *
         * Combines previous openForRead/openForWrite into one API using
         * ModelSerializer::OpenMode to select the behavior.
         *
         * @param filename Filesystem path to open.
         * @param mode     OpenMode::Read or OpenMode::Write.
         * @return true on success, false on failure.
         */
        [[nodiscard]] bool open( const std::string& filename, OpenMode mode ) override
        {
            // Ensure any prior archive is closed
            close();

            if (mode == OpenMode::Write)
            {
                if (!mz_zip_writer_init_file( &zip_, filename.c_str(), 0 ))
                {
                    Utils::Logger::info( std::format( "ZipSerializer: failed to init writer for {}", filename ) );
                    state_ = State::Closed;
                    return false;
                }

                filename_ = filename;
                state_ = State::OpenForWrite;
                return true;
            }
            else // OpenMode::Read
            {
                if (!mz_zip_reader_init_file( &zip_, filename.c_str(), 0 ))
                {
                    Utils::Logger::error_fmt( "ZipSerializer: failed to init reader for '{}'", filename );
                    state_ = State::Closed;
                    return false;
                }

                filename_ = filename;
                state_ = State::OpenForRead;
                return true;
            }
        }

        /**
         * @brief Finalize and close the archive.
         *
         * For write mode this ensures the archive is finalized and internal
         * miniz state is released. For read mode releases reader resources.
         *
         * @return true on success, false on internal miniz failure.
         */
        [[nodiscard]] bool close() override
        {
            if (state_ == State::Closed)
            {
                return true;
            }

            bool success = true;

            if (state_ == State::OpenForWrite)
            {
                // Call both finalize and end unconditionally to avoid leaks
                bool finalize_ok = mz_zip_writer_finalize_archive( &zip_ ) != 0;
                bool end_ok = mz_zip_writer_end( &zip_ ) != 0;
                success = finalize_ok && end_ok;

                if (!finalize_ok)
                {
                    Utils::Logger::warning_fmt( "ZipSerializer: finalize archive failed for '{}'", filename_ );
                }

                if (!end_ok)
                {
                    Utils::Logger::warning_fmt( "ZipSerializer: writer end failed for '{}'", filename_ );
                }
            }
            else if (state_ == State::OpenForRead)
            {
                bool reader_ok = mz_zip_reader_end( &zip_ ) != 0;
                success = reader_ok;

                if (!reader_ok)
                {
                    Utils::Logger::warning_fmt( "ZipSerializer: reader end failed for '{}'", filename_ );
                }
            }

            filename_.clear();
            std::memset( &zip_, 0, sizeof( zip_ ) );
            state_ = State::Closed;

            return success;
        }

        /**
         * @brief Check if the serializer is currently open.
         *
         * @return true if open for reading or writing, false if closed.
         */
        [[nodiscard]] bool isOpen() const noexcept override
        {
            return state_ != State::Closed;
        }

        /**
         * @brief Check if the serializer is open for reading.
         *
         * @return true if open for reading.
         */
        [[nodiscard]] bool isOpenForRead() const noexcept override
        {
            return state_ == State::OpenForRead;
        }

        /**
         * @brief Check if the serializer is open for writing.
         *
         * @return true if open for writing.
         */
        [[nodiscard]] bool isOpenForWrite() const noexcept override
        {
            return state_ == State::OpenForWrite;
        }

        /**
         * @brief Get the filename of the currently open archive.
         *
         * @return Filename, or empty string if no archive is open.
         */
        [[nodiscard]] const std::string& getFilename() const noexcept override
        {
            return filename_;
        }

        /**
         * @brief Add a binary blob to the archive at the given internal path.
         *
         * Paths are normalized to use forward slashes and stripped of leading
         * './' or '/'.
         *
         * @param path Archive-internal path (e.g., "network/weights.bin").
         * @param data Pointer to source bytes (may be null if size == 0).
         * @param size Number of bytes to write.
         * @return true on success, false on failure.
         */
        [[nodiscard]] bool addData( const std::string& path, const void* data, size_t size ) override
        {
            if (!isOpenForWrite())
            {
                Utils::Logger::error( "ZipSerializer: addData called but archive not open for writing" );
                return false;
            }

            std::string p = normalizeZipPath( path );

            if (!mz_zip_writer_add_mem( &zip_, p.c_str(), data, size, MZ_DEFAULT_COMPRESSION ))
            {
                Utils::Logger::error_fmt( "ZipSerializer: failed to add data '{}' to '{}'", p, filename_ );
                return false;
            }

            return true;
        }

        /**
         * @brief Add a binary blob using std::span (C++23 style).
         *
         * @param path Archive-internal path.
         * @param data Span of bytes to write.
         * @return true on success, false on failure.
         */
        [[nodiscard]] bool addData( const std::string& path, std::span<const std::byte> data )
        {
            return addData( path, data.data(), data.size() );
        }

        /**
         * @brief Extract a file from the archive into a caller-provided buffer.
         *
         * @param path Archive-internal path to extract.
         * @param data Pre-allocated buffer to receive file contents.
         * @param size Size of the provided buffer in bytes.
         * @return Number of bytes written into the buffer, or 0 on error.
         */
        [[nodiscard]] size_t extractData( const std::string& path, void* data, size_t size ) override
        {
            if (!isOpenForRead())
            {
                Utils::Logger::error( "ZipSerializer: extractData called but archive not open for reading" );
                return 0;
            }

            std::string p = normalizeZipPath( path );

            int fileIndex = mz_zip_reader_locate_file( &zip_, p.c_str(), nullptr, 0 );
            if (fileIndex < 0)
            {
                Utils::Logger::warning_fmt( "ZipSerializer: file not found in archive '{}': {}", filename_, p );
                return 0;
            }

            mz_zip_archive_file_stat stat;
            if (!mz_zip_reader_file_stat( &zip_, fileIndex, &stat ))
            {
                Utils::Logger::error_fmt( "ZipSerializer: failed to stat file '{}' in '{}'", p, filename_ );
                return 0;
            }

            if (stat.m_uncomp_size > size)
            {
                Utils::Logger::error_fmt( "ZipSerializer: buffer too small for '{}' (need {}, have {})",
                    p, static_cast<size_t>(stat.m_uncomp_size), size );
                return 0;
            }

            if (!mz_zip_reader_extract_to_mem( &zip_, fileIndex, data, size, 0 ))
            {
                Utils::Logger::error_fmt( "ZipSerializer: failed to extract '{}' from '{}'", p, filename_ );
                return 0;
            }

            return static_cast<size_t>(stat.m_uncomp_size);
        }

        /**
         * @brief Extract a file using std::span (C++23 style).
         *
         * @param path Archive-internal path to extract.
         * @param buffer Span to receive file contents.
         * @return Number of bytes written into the buffer, or 0 on error.
         */
        [[nodiscard]] size_t extractData( const std::string& path, std::span<std::byte> buffer )
        {
            return extractData( path, buffer.data(), buffer.size() );
        }

        /**
         * @brief Get the uncompressed size of the file stored at `path`.
         *
         * @param path Archive-internal path.
         * @return File size in bytes, or 0 if the file is missing or an error occurred.
         */
        [[nodiscard]] size_t getFileSize( const std::string& path ) const override
        {
            if (!isOpenForRead())
            {
                return 0;
            }

            std::string p = normalizeZipPath( path );

            int fileIndex = mz_zip_reader_locate_file( &zip_, p.c_str(), nullptr, 0 );
            if (fileIndex < 0)
            {
                return 0;
            }

            mz_zip_archive_file_stat stat;
            if (!mz_zip_reader_file_stat( &zip_, fileIndex, &stat ))
            {
                return 0;
            }

            return static_cast<size_t>(stat.m_uncomp_size);
        }

        /**
         * @brief List all files contained in the open archive.
         *
         * @return Vector of archive-internal file paths. Empty if archive is closed
         *         or opened for writing.
         */
        [[nodiscard]] std::vector<std::string> listFiles() const override
        {
            std::vector<std::string> files;

            if (!isOpenForRead())
            {
                return files;
            }

            int num_files = mz_zip_reader_get_num_files( &zip_ );
            files.reserve( std::max( 0, num_files ) );

            for (int i = 0; i < num_files; ++i)
            {
                mz_zip_archive_file_stat stat;
                if (mz_zip_reader_file_stat( &zip_, i, &stat ))
                {
                    files.emplace_back( stat.m_filename );
                }
            }

            return files;
        }

        /**
         * @brief Store small textual metadata under the "metadata/" prefix.
         *
         * @param key Metadata key (will be stored as "metadata/<key>").
         * @param value Metadata value bytes.
         * @return true on success.
         */
        [[nodiscard]] bool addMetadata( const std::string& key, const std::string& value ) override
        {
            return addData( "metadata/" + key, value.data(), value.size() );
        }

        /**
         * @brief Retrieve textual metadata previously stored with addMetadata().
         *
         * @param key Metadata key.
         * @return Metadata value string, or empty string if missing or on error.
         */
        [[nodiscard]] std::string getMetadata( const std::string& key ) const override
        {
            std::string p = "metadata/" + key;
            size_t size = getFileSize( p );
            if (size == 0)
            {
                return {};
            }

            std::string value( size, '\0' );
            // Need to extract data - const_cast required for miniz API
            size_t extracted = const_cast<ZipSerializer*>(this)->extractData( p, value.data(), size );

            if (extracted != size)
            {
                Utils::Logger::warning_fmt( "ZipSerializer: metadata extraction size mismatch for key '{}'", key );
                return {};
            }

            return value;
        }

        /**
         * @brief Check whether the archive contains the given path.
         *
         * @param path Archive-internal path to query.
         * @return true if the file exists and archive is open for read.
         */
        [[nodiscard]] bool hasFile( const std::string& path ) const override
        {
            if (!isOpenForRead())
            {
                return false;
            }

            std::string p = normalizeZipPath( path );

            return mz_zip_reader_locate_file( &zip_, p.c_str(), nullptr, 0 ) >= 0;
        }

    private:
        /**
         * @brief Archive state enumeration.
         */
        enum class State
        {
            Closed,
            OpenForRead,
            OpenForWrite
        };

        /**
         * @brief Normalize an archive-internal path to ZIP canonical form.
         *
         * Converts backslashes to forward slashes, collapses repeated slashes,
         * and strips a leading "./" or leading '/' if present.
         *
         * @param raw Raw path provided by caller.
         * @return Normalized path suitable for use with miniz.
         */
        static std::string normalizeZipPath( std::string_view raw )
        {
            if (raw.empty())
            {
                return {};
            }

            std::string out;
            out.reserve( raw.size() );

            // Combined pass: replace backslashes and skip repeated slashes
            bool last_was_slash = false;
            for (char c : raw)
            {
                if (c == '\\' || c == '/')
                {
                    if (!last_was_slash)
                    {
                        out.push_back( '/' );
                        last_was_slash = true;
                    }
                }
                else
                {
                    out.push_back( c );
                    last_was_slash = false;
                }
            }

            // Strip leading "./" or "/"
            std::string_view result = out;
            if (result.starts_with( "./" ))
            {
                result.remove_prefix( 2 );
            }
            else if (result.starts_with( "/" ))
            {
                result.remove_prefix( 1 );
            }

            return std::string( result );
        }

        mutable mz_zip_archive zip_;  // Mutable due to miniz API limitations
        std::string filename_;
        State state_ = State::Closed;
    };
}