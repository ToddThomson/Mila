/**
 * @file ZipSerializer.ixx
 * @brief Implements the ModelSerializer interface using miniz for ZIP archives
 */

module;
#include <miniz.h>
#include <string>
#include <string.h>

export module Serialization.ZipSerializer;

import Serialization.ModelSerializer;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Implementation of ModelSerializer using miniz for ZIP archives
     */
    export class ZipSerializer : public ModelSerializer {
    public:
        ZipSerializer() {
            memset( &zip_, 0, sizeof( zip_ ) );
        }

        ~ZipSerializer() {
            close();
        }

        bool openForWrite( const std::string& filename ) override {
            close();
            is_writing_ = true;

            if ( !mz_zip_writer_init_file( &zip_, filename.c_str(), 0 ) ) {
                return false;
            }

            filename_ = filename;
            return true;
        }

        bool openForRead( const std::string& filename ) override {
            close();
            is_writing_ = false;

            if ( !mz_zip_reader_init_file( &zip_, filename.c_str(), 0 ) ) {
                return false;
            }

            filename_ = filename;
            return true;
        }

        bool close() override {
            if ( filename_.empty() ) {
                return true;
            }

            bool success = true;

            if ( is_writing_ ) {
                success = mz_zip_writer_finalize_archive( &zip_ ) && mz_zip_writer_end( &zip_ );
            }
            else {
                success = mz_zip_reader_end( &zip_ );
            }

            filename_.clear();
            memset( &zip_, 0, sizeof( zip_ ) );
            return success;
        }

        bool addData( const std::string& path, const void* data, size_t size ) override {
            if ( !is_writing_ || filename_.empty() ) {
                return false;
            }

            return mz_zip_writer_add_mem( &zip_, path.c_str(), data, size, MZ_DEFAULT_COMPRESSION );
        }

        size_t extractData( const std::string& path, void* data, size_t size ) override {
            if ( is_writing_ || filename_.empty() ) {
                return 0;
            }

            int fileIndex = mz_zip_reader_locate_file( &zip_, path.c_str(), nullptr, 0 );
            if ( fileIndex < 0 ) {
                return 0;
            }

            mz_zip_archive_file_stat stat;
            if ( !mz_zip_reader_file_stat( &zip_, fileIndex, &stat ) ) {
                return 0;
            }

            if ( stat.m_uncomp_size > size ) {
                return 0;
            }

            if ( !mz_zip_reader_extract_to_mem( &zip_, fileIndex, data, size, 0 ) ) {
                return 0;
            }

            return stat.m_uncomp_size;
        }

        size_t getFileSize( const std::string& path ) const override
        {
            if (is_writing_ || filename_.empty())
            {
                return 0;
            }

            int fileIndex = mz_zip_reader_locate_file( const_cast<mz_zip_archive*>(&zip_), path.c_str(), nullptr, 0 );
            if (fileIndex < 0)
            {
                return 0;
            }

            mz_zip_archive_file_stat stat;
            if (!mz_zip_reader_file_stat( const_cast<mz_zip_archive*>( &zip_ ), fileIndex, &stat ))
            {
                return 0;
            }

            return stat.m_uncomp_size;
        }

        std::vector<std::string> listFiles() const override
        {
            std::vector<std::string> files;

            if (is_writing_ || filename_.empty())
            {
                return files;
            }

            int num_files = mz_zip_reader_get_num_files( const_cast<mz_zip_archive*>(&zip_) );
            files.reserve( num_files );

            for (int i = 0; i < num_files; ++i)
            {
                mz_zip_archive_file_stat stat;
                if (mz_zip_reader_file_stat( const_cast<mz_zip_archive*>( &zip_ ), i, &stat ))
                {
                    files.emplace_back( stat.m_filename );
                }
            }

            return files;
        }

        bool addMetadata( const std::string& key, const std::string& value ) override
        {
            return addData( "metadata/" + key, value.data(), value.size() );
        }

        std::string getMetadata( const std::string& key ) const override
        {
            size_t size = getFileSize( "metadata/" + key );
            if (size == 0) return "";

            std::string value( size, '\0' );
            const_cast<ZipSerializer*>(this)->extractData( "metadata/" + key, value.data(), size );
            
            return value;
        }

        bool hasFile( const std::string& path ) const override
        {
            if (is_writing_ || filename_.empty())
            {
                return false;
            }

            return mz_zip_reader_locate_file( const_cast<mz_zip_archive*>(&zip_),
                path.c_str(), nullptr, 0 ) >= 0;
        }


    private:
        mz_zip_archive zip_;
        std::string filename_;
        bool is_writing_ = false;
    };
}