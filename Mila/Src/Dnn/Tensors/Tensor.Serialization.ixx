/**
 * @file Tensor.Serialization.ixx
 * @brief Tensor-specific serialization helpers and metadata.
 *
 * Provides TensorMetadata and free helpers to write/read tensor metadata + raw
 * bytes using ModelArchive. This partition keeps tensor concerns out of
 * ModelArchive and lets tensor implementations call these helpers.
 */

module;
#include <memory>
#include <string>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <utility>

export module Serialization.Tensor;

import Serialization.ModelArchive;
import Serialization.Metadata;
import Dnn.TensorTypes;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Lightweight metadata describing a serialized tensor.
     */
    export struct TensorMetadata
    {
        std::string dtype;
        shape_t shape;
        size_t byte_size{ 0 };
        std::string layout{ "row_major" };
        std::string byte_order{ "little" };
    };

    /**
     * @brief Write tensor metadata and raw bytes under the given prefix into archive.
     *
     * Writes:
     *   prefix + "/meta.json"   <-- TensorMetadata as SerializationMetadata
     *   prefix + "/data.bin"    <-- raw tensor bytes
     *
     * Caller is responsible for providing contiguous raw bytes in the
     * device-neutral layout expected by the runtime (row-major).
     *
     * @param archive ModelArchive to write to
     * @param prefix Path prefix for tensor files (e.g., "tensors/weight")
     * @param meta Tensor metadata
     * @param data Pointer to raw tensor bytes
     * @param size Number of bytes to write
     *
     * @throws std::runtime_error if write operations fail
     *
     * @example
     * TensorMetadata meta;
     * meta.dtype = "FP32";
     * meta.shape = {128, 64};
     * meta.byte_size = 128 * 64 * 4;
     * writeTensorBlob(archive, "tensors/weight", meta, weight_data, meta.byte_size);
     */
    export inline void writeTensorBlob( ModelArchive& archive, const std::string& prefix,
        const TensorMetadata& meta, const void* data, size_t size )
    {
        SerializationMetadata metadata;
        metadata.set( "dtype", meta.dtype )
                .set( "shape", meta.shape )
                .set( "byte_size", static_cast<int64_t>( meta.byte_size ) )
                .set( "layout", meta.layout )
                .set( "byte_order", meta.byte_order );

        archive.writeMetadata( prefix + "/meta.json", metadata );
        archive.writeBlob( prefix + "/data.bin", data, size );
    }

    /**
     * @brief Read tensor metadata and raw bytes from prefix using archive.
     *
     * Returns metadata and raw bytes vector. Throws on error.
     *
     * @param archive ModelArchive to read from
     * @param prefix Path prefix for tensor files (e.g., "tensors/weight")
     * @return Pair of TensorMetadata and raw bytes vector
     *
     * @throws std::runtime_error if read operations fail
     * @throws std::runtime_error if byte size mismatch detected
     *
     * @example
     * auto [meta, data] = readTensorBlob(archive, "tensors/weight");
     * // meta.dtype, meta.shape, etc. are populated
     * // data contains raw tensor bytes
     */
    export inline std::pair<TensorMetadata, std::vector<uint8_t>> readTensorBlob( const ModelArchive& archive, const std::string& prefix )
    {
        TensorMetadata meta;
        SerializationMetadata metadata = archive.readMetadata( prefix + "/meta.json" );

        meta.dtype = metadata.getString( "dtype" );
        meta.shape = metadata.getShape( "shape" );
        meta.byte_size = static_cast<size_t>( metadata.getInt( "byte_size" ) );

        // Optional fields with defaults
        auto layout_opt = metadata.tryGetString( "layout" );
        if ( layout_opt.has_value() )
        {
            meta.layout = layout_opt.value();
        }

        auto byte_order_opt = metadata.tryGetString( "byte_order" );
        if ( byte_order_opt.has_value() )
        {
            meta.byte_order = byte_order_opt.value();
        }

        std::vector<uint8_t> data = archive.readBlob( prefix + "/data.bin" );
        if ( data.size() != meta.byte_size )
        {
            throw std::runtime_error( "readTensorBlob size mismatch for prefix: " + prefix );
        }

        return { meta, std::move( data ) };
    }
}