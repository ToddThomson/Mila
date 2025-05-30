/**
 * @file ZipModelSerializer.ixx
 * @brief Implements the ModelSerializer interface using miniz for ZIP archives
 */

module;
#include <miniz.h>
#include <string>
#include <memory>
#include <stdexcept>
#include <vector>

export module Dnn.Serialization.ZipModelSerializer;

import Dnn.Serialization.ModelSerializer;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Implementation of ModelSerializer using miniz for ZIP archives
     */
    export class ZipModelSerializer : public ModelSerializer {
    public:
        ZipModelSerializer() {
            memset( &zip_, 0, sizeof( zip_ ) );
        }

        ~ZipModelSerializer() {
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

        bool hasFile( const std::string& path ) const override {
            if ( is_writing_ || filename_.empty() ) {
                return false;
            }

            return false; // FIXME: mz_zip_reader_locate_file( &zip_, path.c_str(), nullptr, 0 ) >= 0;
        }

    private:
        mz_zip_archive zip_;
        std::string filename_;
        bool is_writing_ = false;
    };
}