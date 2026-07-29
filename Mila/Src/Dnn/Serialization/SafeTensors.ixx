/**
 * @file SafeTensors.ixx
 * @brief safetensors container support: dtype naming and a two-phase writer.
 *
 * The distribution artifact is written in the safetensors layout so that any reader can
 * verify a Mila file without running Mila. Reading is handled by PretrainedModelReader,
 * which sniffs the container and shares one staging path across both formats.
 */

module;
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

export module Serialization.SafeTensors;

import nlohmann.json;
import Dnn.TensorDataType;
import Dnn.TensorTypes;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Key under which the Mila architecture config is stored in __metadata__.
     *
     * safetensors restricts __metadata__ to string-to-string, so the config rides as one
     * JSON-encoded string value. Foreign readers ignore the key; Mila parses it back with
     * the same reader the flat format uses.
     */
    export inline constexpr const char* kMilaConfigMetadataKey = "mila_config";

    /**
     * @brief Key marking an artifact whose weights are already quantized.
     *
     * Present only on pre-quantized artifacts. Its absence means the weights are in the
     * declared compute precision and the load path quantizes as it always has.
     */
    export inline constexpr const char* kMilaQuantizationMetadataKey = "mila_quantization";

    /**
     * @brief Byte width of one stored element of the given type.
     *
     * Packed 4-bit types report one byte because two elements share it; a packed tensor
     * therefore declares its physical (halved) column count, and the byte count follows
     * from the declared shape with no special case.
     */
    export inline size_t storageBytesPerElement( TensorDataType type )
    {
        switch ( type )
        {
            case TensorDataType::FP32:     return 4;
            case TensorDataType::FP16:     return 2;
            case TensorDataType::BF16:     return 2;
            case TensorDataType::FP8_E4M3: return 1;
            case TensorDataType::FP8_E5M2: return 1;
            case TensorDataType::FP4_E2M1: return 1;
            case TensorDataType::FP4_E3M0: return 1;
            case TensorDataType::INT8:     return 1;
            case TensorDataType::INT16:    return 2;
            case TensorDataType::INT32:    return 4;
            case TensorDataType::UINT8:    return 1;
            case TensorDataType::UINT16:   return 2;
            case TensorDataType::UINT32:   return 4;
            default:
                throw std::runtime_error(
                    std::format( "safetensors: no storage width for dtype {}",
                        tensorDataTypeToString( type ) ) );
        }
    }

    /**
     * @brief safetensors dtype spelling for a Mila type.
     *
     * Packed 4-bit weights are carried as U8, which is what the ecosystem does; the group
     * size and scale pairing live in the Mila config, not in the dtype.
     */
    export inline std::string toSafeTensorsDataTypeName( TensorDataType type )
    {
        switch ( type )
        {
            case TensorDataType::FP32:     return "F32";
            case TensorDataType::FP16:     return "F16";
            case TensorDataType::BF16:     return "BF16";
            case TensorDataType::FP8_E4M3: return "F8_E4M3";
            case TensorDataType::FP8_E5M2: return "F8_E5M2";
            case TensorDataType::INT8:     return "I8";
            case TensorDataType::INT16:    return "I16";
            case TensorDataType::INT32:    return "I32";
            case TensorDataType::UINT8:    return "U8";
            case TensorDataType::UINT16:   return "U16";
            case TensorDataType::UINT32:   return "U32";
            default:
                throw std::runtime_error(
                    std::format( "safetensors: dtype {} has no container spelling",
                        tensorDataTypeToString( type ) ) );
        }
    }

    /**
     * @brief Mila type for a safetensors dtype spelling.
     *
     * @throws std::runtime_error if the spelling is one Mila cannot represent.
     */
    export inline TensorDataType fromSafeTensorsDataTypeName( const std::string& name )
    {
        if ( name == "F32" )     return TensorDataType::FP32;
        if ( name == "F16" )     return TensorDataType::FP16;
        if ( name == "BF16" )    return TensorDataType::BF16;
        if ( name == "F8_E4M3" ) return TensorDataType::FP8_E4M3;
        if ( name == "F8_E5M2" ) return TensorDataType::FP8_E5M2;
        if ( name == "I8" )      return TensorDataType::INT8;
        if ( name == "I16" )     return TensorDataType::INT16;
        if ( name == "I32" )     return TensorDataType::INT32;
        if ( name == "U8" )      return TensorDataType::UINT8;
        if ( name == "U16" )     return TensorDataType::UINT16;
        if ( name == "U32" )     return TensorDataType::UINT32;

        throw std::runtime_error( std::format( "safetensors: unsupported dtype '{}'", name ) );
    }

    /**
     * @brief Which pass of a two-phase save a component is being driven through.
     *
     * Exists so a component visits its tensors from ONE ordered body rather than two.
     * The writer requires bodies in declaration order, and two separate walks would be
     * free to drift out of agreement with no diagnostic until the file is read back.
     */
    export enum class TensorSavePass
    {
        Declare,  ///< Reserve each tensor's byte range in the header.
        Write     ///< Stream each tensor's bytes, in declaration order.
    };

    /**
     * @brief Writes a safetensors file: u64 header length, JSON header, packed data.
     *
     * Two-phase by necessity. The header records every tensor's byte range, so all shapes
     * must be known before any data is written; declare everything, then stream the bodies
     * in declaration order. That ordering is not a convenience -- the container requires the
     * data region to be tiled contiguously, so a body written out of order corrupts the file.
     * Streaming rather than buffering is what lets a 22 GB model be written while only one
     * staged tensor is resident.
     *
     * @par Example
     * @code
     * SafeTensorsWriter writer( path );
     * writer.declareTensor( "weight", TensorDataType::BF16, shape_t{ 4096, 4096 } );
     * writer.setMetadata( kMilaConfigMetadataKey, config_json.dump() );
     * writer.beginData();
     * writer.writeTensorData( "weight", host_bytes, byte_count );
     * writer.close();
     * @endcode
     */
    export class SafeTensorsWriter
    {
    public:

        /**
         * @brief Open a path for writing.
         *
         * @throws std::runtime_error if the file cannot be created.
         */
        explicit SafeTensorsWriter( const std::filesystem::path& filepath )
            : filepath_( filepath )
        {
            std::filesystem::path parent = filepath.parent_path();

            if ( !parent.empty() )
            {
                std::filesystem::create_directories( parent );
            }

            file_.reset( std::fopen( filepath.string().c_str(), "wb" ) );

            if ( file_ == nullptr )
            {
                throw std::runtime_error(
                    "Cannot open safetensors file for writing: " + filepath.string() );
            }
        }

        ~SafeTensorsWriter()
        {
            file_.reset();
        }

        SafeTensorsWriter( const SafeTensorsWriter& ) = delete;
        SafeTensorsWriter& operator=( const SafeTensorsWriter& ) = delete;

        /**
         * @brief Declare a tensor and reserve its byte range. Phase one.
         *
         * @throws std::runtime_error if data writing has begun, or the name repeats.
         */
        void declareTensor( const std::string& name, TensorDataType dtype, const shape_t& shape )
        {
            if ( header_written_ )
            {
                throw std::runtime_error(
                    std::format( "SafeTensorsWriter: cannot declare '{}' after beginData()", name ) );
            }

            for ( const auto& entry : entries_ )
            {
                if ( entry.name == name )
                {
                    throw std::runtime_error(
                        std::format( "SafeTensorsWriter: duplicate tensor name '{}'", name ) );
                }
            }

            int64_t element_count = 1;

            for ( uint8_t d = 0; d < shape.ndim; ++d )
            {
                if ( shape[ d ] < 0 )
                {
                    throw std::runtime_error(
                        std::format( "SafeTensorsWriter: negative extent in shape of '{}'", name ) );
                }

                element_count *= shape[ d ];
            }

            Entry entry;
            entry.name = name;
            entry.dtype = dtype;
            entry.shape = shape;
            entry.begin = next_offset_;
            entry.nbytes = static_cast<uint64_t>( element_count ) * storageBytesPerElement( dtype );

            next_offset_ += entry.nbytes;
            entries_.push_back( std::move( entry ) );
        }

        /**
         * @brief Set a __metadata__ entry. Phase one.
         *
         * The container permits string values only; encode structured data as JSON text.
         */
        void setMetadata( const std::string& key, const std::string& value )
        {
            if ( header_written_ )
            {
                throw std::runtime_error(
                    std::format( "SafeTensorsWriter: cannot set metadata '{}' after beginData()", key ) );
            }

            metadata_[ key ] = value;
        }

        /**
         * @brief Emit the header and open the data region. Ends phase one.
         */
        void beginData()
        {
            if ( header_written_ )
            {
                throw std::runtime_error( "SafeTensorsWriter: beginData() called twice" );
            }

            nlohmann::json header = nlohmann::json::object();

            if ( !metadata_.empty() )
            {
                nlohmann::json meta = nlohmann::json::object();

                for ( const auto& [key, value] : metadata_ )
                {
                    meta[ key ] = value;
                }

                header[ "__metadata__" ] = std::move( meta );
            }

            for ( const auto& entry : entries_ )
            {
                nlohmann::json record = nlohmann::json::object();
                record[ "dtype" ] = toSafeTensorsDataTypeName( entry.dtype );

                nlohmann::json dims = nlohmann::json::array();

                for ( uint8_t d = 0; d < entry.shape.ndim; ++d )
                {
                    dims.push_back( entry.shape[ d ] );
                }

                record[ "shape" ] = std::move( dims );
                record[ "data_offsets" ] = nlohmann::json::array( { entry.begin, entry.begin + entry.nbytes } );

                header[ entry.name ] = std::move( record );
            }

            std::string header_text = header.dump();

            // The data region must start 8-byte aligned; pad the JSON itself, since the
            // container has no gap between header and data to pad instead.
            while ( ( sizeof( uint64_t ) + header_text.size() ) % 8 != 0 )
            {
                header_text.push_back( ' ' );
            }

            const uint64_t header_length = header_text.size();

            writeExact( &header_length, sizeof( header_length ), "header length" );
            writeExact( header_text.data(), header_text.size(), "header" );

            header_written_ = true;
            next_write_index_ = 0;
        }

        /**
         * @brief Append one tensor body. Phase two, in declaration order.
         *
         * @throws std::runtime_error on order or size mismatch, either of which would
         *         silently desynchronize the file from its own index.
         */
        void writeTensorData( const std::string& name, const void* data, size_t nbytes )
        {
            if ( !header_written_ )
            {
                throw std::runtime_error(
                    std::format( "SafeTensorsWriter: writeTensorData('{}') before beginData()", name ) );
            }

            if ( next_write_index_ >= entries_.size() )
            {
                throw std::runtime_error(
                    std::format( "SafeTensorsWriter: unexpected extra tensor '{}'", name ) );
            }

            const Entry& entry = entries_[ next_write_index_ ];

            if ( entry.name != name )
            {
                throw std::runtime_error(
                    std::format( "SafeTensorsWriter: out of order write; expected '{}', got '{}'",
                        entry.name, name ) );
            }

            if ( nbytes != entry.nbytes )
            {
                throw std::runtime_error(
                    std::format( "SafeTensorsWriter: '{}' declared {} bytes, got {}",
                        name, entry.nbytes, nbytes ) );
            }

            writeExact( data, nbytes, std::format( "body of '{}'", name ) );

            next_write_index_++;
        }

        /**
         * @brief Flush and close.
         *
         * @throws std::runtime_error if any declared tensor was never written -- the header
         *         would promise bytes the file does not contain.
         */
        void close()
        {
            if ( file_ == nullptr )
            {
                return;
            }

            if ( header_written_ && next_write_index_ != entries_.size() )
            {
                const std::string missing = entries_[ next_write_index_ ].name;
                file_.reset();

                throw std::runtime_error(
                    std::format( "SafeTensorsWriter: closed with {} of {} tensors written; '{}' missing",
                        next_write_index_, entries_.size(), missing ) );
            }

            file_.reset();
        }

        /**
         * @brief Number of declared tensors.
         */
        size_t getTensorCount() const noexcept
        {
            return entries_.size();
        }

    private:

        /**
         * @brief Write exactly nbytes, or throw.
         *
         * A short write leaves the data region misaligned with the header's byte ranges,
         * which no later step can detect.
         */
        void writeExact( const void* data, size_t nbytes, const std::string& description )
        {
            if ( nbytes == 0 )
            {
                return;
            }

            if ( std::fwrite( data, 1, nbytes, file_.get() ) != nbytes )
            {
                throw std::runtime_error(
                    std::format( "SafeTensorsWriter: failed writing {} to {}",
                        description, filepath_.string() ) );
            }
        }

        struct Entry
        {
            std::string name;
            TensorDataType dtype{ TensorDataType::FP32 };
            shape_t shape;
            uint64_t begin{ 0 };
            uint64_t nbytes{ 0 };
        };

        std::filesystem::path filepath_;

        // C stdio rather than std::ofstream: stream I/O instantiated from a module unit
        // trips MSVC's C2079 on basic_istream::sentry, transitively through <fstream>.
        // Same avoidance as PretrainedReader and TokenSequenceLoader.
        std::unique_ptr<std::FILE, int( * )( std::FILE* )> file_{ nullptr, &std::fclose };

        std::vector<Entry> entries_;
        std::map<std::string, std::string> metadata_;

        uint64_t next_offset_{ 0 };
        bool header_written_{ false };
        size_t next_write_index_{ 0 };
    };
}
