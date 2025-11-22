/**
 * @file Tensor.Serialization.ixx
 * @brief Tensor-specific serialization helpers and metadata.
 *
 * Provides TensorMetadata and free helpers to write/read tensor metadata + raw
 * bytes using ModelArchive. This partition keeps tensor concerns out of
 * ModelArchive and lets tensor implementations call these helpers.
 */

module;
#include <memory> // for nlohmann::json
#include <string>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <utility>

export module Serialization.Tensor;

import Serialization.ModelArchive;
import Dnn.TensorTypes;
import nlohmann.json;

namespace Mila::Dnn::Serialization
{
    using nlohmann::json;

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
     *   prefix + "/meta.json"   <-- TensorMetadata as JSON
     *   prefix + "/data.bin"    <-- raw tensor bytes
     *
     * Caller is responsible for providing contiguous raw bytes in the
     * device-neutral layout expected by the runtime (row-major).
     */
    export inline void writeTensorBlob( ModelArchive& archive, const std::string& prefix,
        const TensorMetadata& meta, const void* data, size_t size )
    {
        json j;
        j["dtype"] = meta.dtype;
        j["shape"] = meta.shape;
        j["byte_size"] = meta.byte_size;
        j["layout"] = meta.layout;
        j["byte_order"] = meta.byte_order;

        archive.writeJson( prefix + "/meta.json", j );
        archive.writeBlob( prefix + "/data.bin", data, size );
    }

    /**
     * @brief Read tensor metadata and raw bytes from prefix using archive.
     *
     * Returns metadata and raw bytes vector. Throws on error.
     */
    export inline std::pair<TensorMetadata, std::vector<uint8_t>> readTensorBlob( const ModelArchive& archive, const std::string& prefix )
    {
        TensorMetadata meta;
        json j = archive.readJson( prefix + "/meta.json" );

        meta.dtype = j.at( "dtype" ).get<std::string>();
        meta.shape = j.at( "shape" ).get<shape_t>();
        meta.byte_size = j.at( "byte_size" ).get<size_t>();
        if (j.contains( "layout" )) meta.layout = j.at( "layout" ).get<std::string>();
        if (j.contains( "byte_order" )) meta.byte_order = j.at( "byte_order" ).get<std::string>();

        std::vector<uint8_t> data = archive.readBlob( prefix + "/data.bin" );
        if (data.size() != meta.byte_size)
        {
            throw std::runtime_error( "readTensorBlob size mismatch for prefix: " + prefix );
        }

        return { meta, std::move( data ) };
    }
}