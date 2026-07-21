/**
 * @file Tensor.Serialization.ixx
 * @brief Tensor-specific serialization helpers and metadata.
 *
 * Provides TensorMetadata, ITensorBlob, and free helpers to write/read tensor
 * metadata + raw bytes using ModelArchive. This partition keeps tensor concerns
 * out of ModelArchive and lets tensor implementations call these helpers.
 */

module;
#include <string>
#include <cstddef>
#include <stdexcept>
#include <utility>

export module Serialization.Tensor;

import Serialization.ModelArchive;
import Serialization.Metadata;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;
import Dnn.TensorBuffer;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Metadata describing a tensor in serialized form.
     *
     * Contains the minimum information required to interpret raw tensor bytes:
     * precision format, dimensional shape, and total size for validation.
     */
    export struct TensorMetadata
    {
        dtype_t dtype;
        shape_t shape;
        size_t total_bytes{ 0 };
    };

    /**
     * @brief Type-erased interface for a serialized tensor blob.
     *
     * Carries metadata alongside a raw host pointer to the underlying bytes.
     * Concrete TensorBlob<MR> implementations own the memory via TensorBuffer.
     * When MR is CudaPinnedMemoryResource, data() returns a pinned host pointer
     * enabling direct DMA in the CUDA copyFromBlob path without driver staging.
     *
     * This interface is the virtual boundary for loadParameter_ overrides --
     * component implementations receive an ITensorBlob regardless of the
     * originating memory resource.
     */
    export struct ITensorBlob
    {
        virtual ~ITensorBlob() = default;

        ITensorBlob() = default;
        ITensorBlob( const ITensorBlob& ) = delete;
        ITensorBlob& operator=( const ITensorBlob& ) = delete;
        ITensorBlob( ITensorBlob&& ) = default;
        ITensorBlob& operator=( ITensorBlob&& ) = default;

        virtual const TensorMetadata& getMetadata() const noexcept = 0;
        virtual const void* data() const noexcept = 0;
        virtual size_t sizeBytes() const noexcept = 0;
    };

    /**
     * @brief Concrete tensor blob owning a TensorBuffer-backed raw byte buffer.
     *
     * The memory resource controls the allocation strategy:
     *
     *   TensorBlob<CpuMemoryResource>        -- pageable host memory (default)
     *   TensorBlob<CudaPinnedMemoryResource> -- pinned host memory; copyFromBlob
     *                                          issues a direct DMA to device
     *                                          with no hidden driver staging
     *
     * @tparam MR Memory resource type. Must satisfy isValidTensor<UINT8, MR>.
     */
    export template<typename MR = Compute::CpuMemoryResource>
        requires isValidTensor<dtype_t::UINT8, MR>
    struct TensorBlob : ITensorBlob
    {
        TensorMetadata metadata;
        TensorBuffer<dtype_t::UINT8, MR> data_buffer;

        TensorBlob( TensorMetadata meta, TensorBuffer<dtype_t::UINT8, MR> buf )
            : metadata( std::move( meta ) ), data_buffer( std::move( buf ) )
        {}

        TensorBlob( TensorBlob&& ) = default;
        TensorBlob& operator=( TensorBlob&& ) = default;
        TensorBlob( const TensorBlob& ) = delete;
        TensorBlob& operator=( const TensorBlob& ) = delete;

        const TensorMetadata& getMetadata() const noexcept override { return metadata; }
        const void* data() const noexcept override { return data_buffer.data(); }
        size_t sizeBytes() const noexcept override { return data_buffer.storageBytes(); }
    };

    /**
     * @brief Non-owning ITensorBlob view over externally-owned bytes.
     *
     * Carries metadata plus a borrowed host pointer; it allocates and owns nothing.
     * Used by PretrainedModelReader to hand out blobs that point either into a
     * memory-mapped file region or into a reusable pinned staging buffer. The
     * referenced memory MUST outlive the view, and the view is only valid until
     * the owner reuses or releases that memory.
     */
    export struct TensorBlobView : ITensorBlob
    {
        TensorMetadata metadata;
        const void* ptr{ nullptr };
        size_t bytes{ 0 };

        TensorBlobView() = default;

        TensorBlobView( TensorMetadata meta, const void* data_ptr, size_t byte_count )
            : metadata( std::move( meta ) ), ptr( data_ptr ), bytes( byte_count )
        {}

        const TensorMetadata& getMetadata() const noexcept override { return metadata; }
        const void* data() const noexcept override { return ptr; }
        size_t sizeBytes() const noexcept override { return bytes; }
    };

    /**
     * @brief Write tensor metadata and raw bytes under the given prefix into archive.
     *
     * Writes:
     *   prefix + "/meta.json"  -- TensorMetadata as SerializationMetadata
     *   prefix + "/data.bin"   -- raw tensor bytes
     *
     * @param archive ModelArchive to write to
     * @param prefix Path prefix for tensor files (e.g., "tensors/weight")
     * @param meta Tensor metadata
     * @param data Pointer to raw tensor bytes
     * @param size Number of bytes to write
     *
     * @throws std::runtime_error if write operations fail
     */
    export inline void writeTensorBlob( ModelArchive& archive, const std::string& prefix,
        const TensorMetadata& meta, const void* data, size_t size )
    {
        SerializationMetadata sm;
        sm.set( "dtype", tensorDataTypeToString( meta.dtype ) )
          .set( "shape", meta.shape )
          .set( "total_bytes", static_cast<int64_t>( meta.total_bytes ) );

        archive.writeMetadata( prefix + "/meta.json", sm );
        archive.writeBlob( prefix + "/data.bin", data, size );
    }

    /**
     * @brief Read tensor metadata and raw bytes from prefix into a typed blob.
     *
     * Allocates a TensorBuffer<UINT8, MR> sized to total_bytes and reads
     * directly into it via readBlobInto -- no intermediate vector copy.
     * When MR is CudaPinnedMemoryResource the returned blob carries a pinned
     * host pointer ready for direct DMA in copyFromBlob.
     *
     * @tparam MR Memory resource for the blob data buffer. Defaults to CpuMemoryResource.
     * @param archive ModelArchive to read from
     * @param prefix Path prefix for tensor files (e.g., "tensors/weight")
     * @param device_id Device index passed to the memory resource constructor.
     * @return TensorBlob<MR> owning the metadata and raw byte buffer.
     *
     * @throws std::runtime_error if read operations fail or size mismatch detected
     */
    export template<typename MR = Compute::CpuMemoryResource>
        requires isValidTensor<dtype_t::UINT8, MR>
    TensorBlob<MR> readTensorBlob( const ModelArchive& archive, const std::string& prefix, int device_id = 0 )
    {
        SerializationMetadata sm = archive.readMetadata( prefix + "/meta.json" );

        TensorMetadata meta;
        meta.dtype = parseTensorDataType( sm.getString( "dtype" ) );
        meta.shape = sm.getShape( "shape" );
        meta.total_bytes = static_cast<size_t>( sm.getInt( "total_bytes" ) );

        TensorBuffer<dtype_t::UINT8, MR> buffer( device_id, meta.total_bytes );

        size_t read = archive.readBlobInto( prefix + "/data.bin", buffer.data(), buffer.storageBytes() );

        if ( read != meta.total_bytes )
        {
            throw std::runtime_error( "readTensorBlob size mismatch for prefix: " + prefix );
        }

        return TensorBlob<MR>( std::move( meta ), std::move( buffer ) );
    }
}