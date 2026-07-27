/**
 * @file PretrainedReader.ixx
 * @brief Reader for Mila pretrained binary format.
 *
 * Provides direct access to pretrained model weights stored in Mila's
 * flat binary format. Used by fromPretrained() factory methods.
 *
 * The whole file is memory-mapped at construction (CreateFileMapping/MapViewOfFile
 * on Windows, mmap on POSIX). streamTensorBlobs() consumes blobs in ascending file
 * offset order, turning the former 224+ random per-tensor reads into a single
 * sequential scan. On the CUDA path a background producer thread stages each blob
 * into a pinned host buffer (double-buffered) so disk I/O overlaps the H2D copy; all
 * CUDA work stays on the consuming thread. Staging reads each blob directly from the
 * file handle (positioned ReadFile/pread), not by faulting through the mapped view --
 * the map's 4 KB on-demand faults throttle a large model well below disk bandwidth
 * once the file no longer fits resident alongside the model's own GPU staging. The
 * mapping is kept for oversized blobs (consumed straight from the view) and the
 * legacy random-access readTensorBlob<MR>() fallback.
 */

module;
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <format>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <type_traits>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

export module Serialization.PretrainedReader;

import Serialization.Serializer;
import Serialization.OpenMode;
import Serialization.Tensor;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorBuffer;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;
import Compute.Device;
import Compute.CpuMemoryResource;
#ifdef MILA_HAS_CUDA
import Compute.CudaPinnedMemoryResource;
#endif

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Metadata for a tensor blob in pretrained model format.
     */
    struct TensorBlobMetadata
    {
        std::string name;
        uint32_t dtype;
        shape_t shape;
        uint64_t offset;
        uint64_t nbytes;
    };

    /**
     * @brief Metadata for pretrained model.
     */
    export struct PretrainedMetadata
    {
        std::string architecture;
        std::string model_name;
        uint32_t vocab_size;
        uint32_t max_seq_length;
        uint32_t embedding_dim;
        uint32_t num_layers;
        uint32_t num_heads;
        uint32_t num_kv_heads;
        uint32_t head_dim;          // explicit per-head width (Gemma decouples it from embedding_dim/num_heads); 0 = derive
        uint32_t hidden_dim;
        bool use_bias;
        bool tie_word_embeddings = false;

        std::string activation;
        std::string norm_type;
        std::string attention_type;
        std::string positional_encoding;

        float rope_theta;
        float norm_epsilon;

        // Gemma-specific geometry (0 / false for other architectures). The global
        // (full-attention) layers diverge from the sliding layers; the chassis fields
        // drive the 5:1 interleave, dual RoPE, and logit softcap. See GemmaConfig.
        uint32_t global_head_dim;
        uint32_t num_global_kv_heads;
        bool     key_equals_value;
        uint32_t window;
        uint32_t sliding_window_pattern;
        uint32_t global_rotary_dim;
        float    rope_theta_local;
        float    rope_theta_global;
        float    final_logit_softcapping;
    };

    enum class DType : uint32_t {
        Float32 = 0,
        Float16 = 1,
        BFloat16 = 2,
        Int32 = 3,
    };

    inline TensorDataType dtypeToTensorDataType( uint32_t dtype )
    {
        switch ( static_cast<DType>( dtype ) )
        {
            case DType::Float32:  return TensorDataType::FP32;
            case DType::Float16:  return TensorDataType::FP16;
            case DType::BFloat16: return TensorDataType::BF16;
            case DType::Int32:    return TensorDataType::INT32;
            default:
                throw std::runtime_error( "Unknown dtype code: " + std::to_string( dtype ) );
        }
    }

    /**
     * @brief Reader for Mila pretrained binary format.
     *
     * File format:
     *  - Header: MILA magic (0x4D494C41), version, num_tensors
     *  - Metadata: JSON string with model configuration
     *  - Tensor index: for each tensor: name, dtype, shape, offset, nbytes
     *  - Tensor data: concatenated binary blobs
     *
     * Provides flat key-value access to tensors by name:
     *  - "lenc.wte.weight"
     *  - "tf_layer_0.ln_1.bias"
     *  - "ln_final.weight"
     *
     * Usage:
     * @code
     * PretrainedModelReader reader( "gpt2_small.bin" );
     *
     * auto metadata = reader.getPretrainedMetadata();
     * auto names = reader.getTensorNames();
     *
     * for (const auto& name : names)
     * {
     *     auto blob = reader.readTensorBlob<CudaPinnedMemoryResource>( name, device_id );
     *     network->loadTensorByFlatName( name, blob );
     * }
     * @endcode
     */
    export class PretrainedModelReader
    {
    public:

        /**
         * @brief Open a Mila model file for reading.
         *
         * @param filepath Path to .bin model file.
         * @throws std::runtime_error if file cannot be opened or format is invalid.
         */
        explicit PretrainedModelReader( const std::filesystem::path& filepath )
            : filepath_( filepath )
        {
            file_.open( filepath, std::ios::binary );

            if ( !file_.is_open() )
            {
                throw std::runtime_error( "Cannot open pretrained model file: " + filepath.string() );
            }

            readHeader();
            readMetadata();
            readTensorIndex();

            mapFile();
            buildOffsetOrder();
        }

        ~PretrainedModelReader()
        {
            close();
        }

        // ================================================================
        // Serializer interface
        // ================================================================

        bool close()
        {
            if ( file_.is_open() )
            {
                file_.close();
            }

            unmapFile();

            // sorted_by_offset_ holds pointers into tensor_index_ nodes; drop them first.
            sorted_by_offset_.clear();
            filename_.clear();
            tensor_index_.clear();

            return true;
        }

        bool isOpen() const noexcept
        {
            return file_.is_open();
        }

        const std::string& getFilename() const noexcept
        {
            return filename_;
        }

        // ================================================================
        // PretrainedReader-specific API
        // ================================================================

        /**
         * @brief Get pretrained model metadata.
         */
        const PretrainedMetadata& getPretrainedMetadata() const
        {
            return metadata_;
        }

        /**
         * @brief Get list of all tensor names in the model.
         */
        std::vector<std::string> getTensorNames() const
        {
            std::vector<std::string> names;
            names.reserve( tensor_index_.size() );

            for ( const auto& [name, _] : tensor_index_ )
            {
                names.push_back( name );
            }

            return names;
        }

        /**
         * @brief Check if tensor exists.
         */
        bool hasTensor( const std::string& name ) const
        {
            return tensor_index_.find( name ) != tensor_index_.end();
        }

        /**
         * @brief Get the raw byte size of a named tensor.
         *
         * All sizes are known at construction time from the tensor index.
         * No I/O is performed.
         *
         * @param name Tensor name.
         * @return size_t Byte count of the tensor data.
         * @throws std::runtime_error if name is not found.
         */
        size_t getTensorSizeBytes( const std::string& name ) const
        {
            return static_cast<size_t>( getTensorBlobMetadata( name ).nbytes );
        }

        /**
         * @brief Get the maximum byte size across all tensors in the index.
         *
         * Returns the largest nbytes value in the tensor index. All sizes are
         * known at construction time. No I/O is performed.
         *
         * @return size_t Maximum tensor byte count, or 0 if the index is empty.
         */
        size_t getMaxTensorSizeBytes() const
        {
            size_t max_bytes = 0;

            for ( const auto& [name, meta] : tensor_index_ )
                max_bytes = std::max( max_bytes, static_cast<size_t>( meta.nbytes ) );

            return max_bytes;
        }

        /**
         * @brief Read raw tensor bytes by name into a memory-resource-typed blob.
         *
         * Allocates a TensorBuffer<UINT8, MR> of the exact tensor byte size and reads
         * directly from the file into it. No intermediate buffer is used.
         * When MR is CudaPinnedMemoryResource the returned blob data is page-locked,
         * enabling direct DMA to device in copyFromBlob without a staging copy.
         *
         * @tparam MR Memory resource for the blob data buffer. Defaults to CpuMemoryResource.
         * @param name Tensor name.
         * @param device_id Device index passed to the memory resource constructor.
         * @return TensorBlob<MR> owning the metadata and raw byte buffer.
         * @throws std::runtime_error if the tensor is not found or the read fails.
         */
        template<typename MR = Compute::CpuMemoryResource>
            requires isValidTensor<dtype_t::UINT8, MR>
        TensorBlob<MR> readTensorBlob( const std::string& name, int device_id = 0 )
        {
            const auto& blob_meta = getTensorBlobMetadata( name );

            TensorBuffer<dtype_t::UINT8, MR> buffer( device_id, static_cast<size_t>( blob_meta.nbytes ) );

            file_.seekg( static_cast<std::streamoff>( blob_meta.offset ) );
            file_.read(
                reinterpret_cast<char*>( buffer.data() ),
                static_cast<std::streamsize>( blob_meta.nbytes ) );

            if ( !file_.good() )
            {
                throw std::runtime_error( "Failed to read tensor: " + name );
            }

            TensorMetadata tensor_meta{
                .dtype = dtypeToTensorDataType( blob_meta.dtype ),
                .shape = blob_meta.shape,
                .total_bytes = blob_meta.nbytes
            };

            return TensorBlob<MR>( tensor_meta, std::move( buffer ) );
        }

        /**
         * @brief Stream every tensor blob to a consumer in file-offset order.
         *
         * Replaces the per-tensor seek+read loop. Because the whole file is mapped
         * once, consuming in ascending offset is a single sequential scan the OS can
         * read ahead, rather than 224+ random reads in hash-map order.
         *
         * When TStagingMemoryResource is CudaPinnedMemoryResource a background producer
         * thread stages each blob mmap -> pinned host buffer (double-buffered) while the
         * calling thread runs the consumer (H2D + quantize). All CUDA calls stay on the
         * calling thread; the producer does only host memcpy, matching the safe split in
         * TokenSequenceLoader. Blobs larger than the staging buffer (e.g. the token
         * embedding) bypass staging and are consumed directly from the mapped view.
         *
         * Contract: consume() MUST complete every device read of blob.data() before it
         * returns, because the pinned slot is handed back to the producer for reuse the
         * moment consume() returns. The non-quantized copyFromBlob path self-synchronizes
         * on the default stream, but the FP8/FP4 quantize path issues an async H2D on the
         * op stream and does NOT, so the model's consume callback must synchronize its
         * execution context after loadParameter. The producer's next memcpy overlaps that
         * synchronize, preserving the disk/H2D overlap.
         *
         * @tparam TStagingMemoryResource Staging resource. CudaPinnedMemoryResource selects
         *         the threaded pinned path; CpuMemoryResource consumes mapped views directly
         *         with no staging and no producer thread.
         * @param consume Callable invoked as consume(const std::string& name, const ITensorBlob&).
         * @param device_id Device index for the pinned staging buffers (CUDA path only).
         */
        template<typename TStagingMemoryResource = Compute::CpuMemoryResource, typename TConsumer>
        void streamTensorBlobs( TConsumer&& consume, int device_id = 0 )
        {
#ifdef MILA_HAS_CUDA
            if constexpr ( std::is_same_v<TStagingMemoryResource, Compute::CudaPinnedMemoryResource> )
            {
                streamTensorBlobsPinned( std::forward<TConsumer>( consume ), device_id );
            }
            else
#endif
            {
                (void)device_id;
                streamTensorBlobsDirect( std::forward<TConsumer>( consume ) );
            }
        }

    private:

        static constexpr uint32_t MAGIC = 0x4D494C41;  // "MILA"
        static constexpr uint32_t VERSION = 1;

        // Cap on each pinned staging buffer. Per-layer weights sit well under this;
        // larger tensors (token embedding / lm_head) bypass staging and are consumed
        // directly from the mapped view, bounding pinned memory to two of these.
        static constexpr size_t STAGING_BUFFER_CAPACITY_BYTES =
            static_cast<size_t>( 256 ) * 1024 * 1024;

        std::filesystem::path filepath_;
        std::ifstream file_;
        std::string filename_;

        PretrainedMetadata metadata_;
        std::unordered_map<std::string, TensorBlobMetadata> tensor_index_;
        uint32_t num_tensors_{ 0 };

        // Tensor metadata sorted by ascending file offset (pointers into tensor_index_).
        std::vector<const TensorBlobMetadata*> sorted_by_offset_;

        // Whole-file memory mapping. Owned for the reader's lifetime; every blob view
        // points into this region, so the mapping must outlive all handed-out views.
#ifdef _WIN32
        HANDLE file_handle_{ INVALID_HANDLE_VALUE };
        HANDLE mapping_handle_{ nullptr };
#else
        int fd_{ -1 };
#endif
        const uint8_t* map_base_{ nullptr };
        size_t map_size_{ 0 };

        // Positioned read of nbytes at file offset into dst. The pinned-staging producer
        // uses this instead of a memcpy through the mapped view: blobs are visited in
        // ascending file order, so these are sequential reads that stream off disk into
        // pinned memory at full bandwidth. It avoids the 4 KB on-demand page-fault path,
        // which throttles a large mmap once the whole file no longer fits resident (the
        // model's own GPU staging is multi-GB) -- the measured cause of a load running
        // ~5x below disk bandwidth.
        void readAt( uint64_t offset, void* dst, size_t nbytes ) const
        {
            auto* out = static_cast<uint8_t*>( dst );
            size_t done = 0;

            while ( done < nbytes )
            {
                const size_t remaining = nbytes - done;
#ifdef _WIN32
                // A synchronous handle honors OVERLAPPED.Offset as the read position and
                // completes synchronously; only the single producer thread touches the
                // handle, so there is no shared-file-pointer race.
                const uint64_t position = offset + done;
                OVERLAPPED overlapped{};
                overlapped.Offset = static_cast<DWORD>( position & 0xFFFFFFFFull );
                overlapped.OffsetHigh = static_cast<DWORD>( position >> 32 );

                const DWORD to_read = static_cast<DWORD>(
                    std::min<size_t>( remaining, 0x40000000ull ) );  // 1 GiB per call
                DWORD got = 0;

                if ( !ReadFile( file_handle_, out + done, to_read, &got, &overlapped ) || got == 0 )
                {
                    throw std::runtime_error( std::format(
                        "PretrainedReader: read failed at offset {} ({}/{} bytes) for {}",
                        offset, done, nbytes, filepath_.string() ) );
                }

                done += got;
#else
                const ssize_t got = ::pread(
                    fd_, out + done, remaining, static_cast<off_t>( offset + done ) );

                if ( got <= 0 )
                {
                    throw std::runtime_error( std::format(
                        "PretrainedReader: read failed at offset {} ({}/{} bytes) for {}",
                        offset, done, nbytes, filepath_.string() ) );
                }

                done += static_cast<size_t>( got );
#endif
            }
        }

        const TensorBlobMetadata& getTensorBlobMetadata( const std::string& name ) const
        {
            auto it = tensor_index_.find( name );

            if ( it == tensor_index_.end() )
            {
                throw std::runtime_error( "Tensor not found: " + name );
            }

            return it->second;
        }

        void buildOffsetOrder()
        {
            sorted_by_offset_.clear();
            sorted_by_offset_.reserve( tensor_index_.size() );

            for ( const auto& [name, meta] : tensor_index_ )
                sorted_by_offset_.push_back( &meta );

            std::sort( sorted_by_offset_.begin(), sorted_by_offset_.end(),
                []( const TensorBlobMetadata* a, const TensorBlobMetadata* b )
                {
                    return a->offset < b->offset;
                } );
        }

        void mapFile()
        {
#ifdef _WIN32
            file_handle_ = CreateFileW(
                filepath_.wstring().c_str(),
                GENERIC_READ,
                FILE_SHARE_READ,
                nullptr,
                OPEN_EXISTING,
                FILE_FLAG_SEQUENTIAL_SCAN,
                nullptr );

            if ( file_handle_ == INVALID_HANDLE_VALUE )
            {
                throw std::runtime_error( "Cannot memory-map model file: " + filepath_.string() );
            }

            LARGE_INTEGER file_size;

            if ( !GetFileSizeEx( file_handle_, &file_size ) )
            {
                CloseHandle( file_handle_ );
                file_handle_ = INVALID_HANDLE_VALUE;
                throw std::runtime_error( "Cannot query model file size: " + filepath_.string() );
            }

            map_size_ = static_cast<size_t>( file_size.QuadPart );

            mapping_handle_ = CreateFileMappingW( file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr );

            if ( !mapping_handle_ )
            {
                CloseHandle( file_handle_ );
                file_handle_ = INVALID_HANDLE_VALUE;
                throw std::runtime_error( "CreateFileMapping failed for: " + filepath_.string() );
            }

            void* view = MapViewOfFile( mapping_handle_, FILE_MAP_READ, 0, 0, 0 );

            if ( !view )
            {
                CloseHandle( mapping_handle_ );
                mapping_handle_ = nullptr;
                CloseHandle( file_handle_ );
                file_handle_ = INVALID_HANDLE_VALUE;
                throw std::runtime_error( "MapViewOfFile failed for: " + filepath_.string() );
            }

            map_base_ = static_cast<const uint8_t*>( view );

            // No whole-file prefetch: the staged path (readAt) streams each blob straight
            // off disk into pinned memory in file order, so priming the page cache with
            // all 22 GB here only competes with the model's own multi-GB GPU staging for
            // RAM and is evicted before it helps -- the measured cause of the fault-bound
            // load. FILE_FLAG_SEQUENTIAL_SCAN (above) keeps the OS read-ahead bias for
            // both those sequential reads and the oversized-blob mapped path.
#else
            fd_ = ::open( filepath_.c_str(), O_RDONLY );

            if ( fd_ < 0 )
            {
                throw std::runtime_error( "Cannot memory-map model file: " + filepath_.string() );
            }

            struct stat file_stat;

            if ( ::fstat( fd_, &file_stat ) != 0 )
            {
                ::close( fd_ );
                fd_ = -1;
                throw std::runtime_error( "Cannot stat model file: " + filepath_.string() );
            }

            map_size_ = static_cast<size_t>( file_stat.st_size );

            void* view = ::mmap( nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd_, 0 );

            if ( view == MAP_FAILED )
            {
                ::close( fd_ );
                fd_ = -1;
                throw std::runtime_error( "mmap failed for: " + filepath_.string() );
            }

            map_base_ = static_cast<const uint8_t*>( view );

            // MADV_SEQUENTIAL biases read-ahead for the oversized-blob mapped path; the
            // staged reads use pread (see readAt), so there is no whole-file MADV_WILLNEED
            // -- it would prime the page cache with the entire file only to be evicted
            // under the model's own multi-GB GPU staging before it helped.
            ::madvise( view, map_size_, MADV_SEQUENTIAL );
#endif
        }

        void unmapFile() noexcept
        {
#ifdef _WIN32
            if ( map_base_ )
            {
                UnmapViewOfFile( map_base_ );
                map_base_ = nullptr;
            }

            if ( mapping_handle_ )
            {
                CloseHandle( mapping_handle_ );
                mapping_handle_ = nullptr;
            }

            if ( file_handle_ != INVALID_HANDLE_VALUE )
            {
                CloseHandle( file_handle_ );
                file_handle_ = INVALID_HANDLE_VALUE;
            }
#else
            if ( map_base_ )
            {
                ::munmap( const_cast<uint8_t*>( map_base_ ), map_size_ );
                map_base_ = nullptr;
            }

            if ( fd_ >= 0 )
            {
                ::close( fd_ );
                fd_ = -1;
            }
#endif
            map_size_ = 0;
        }

        template<typename TConsumer>
        void streamTensorBlobsDirect( TConsumer&& consume )
        {
            for ( const TensorBlobMetadata* meta : sorted_by_offset_ )
            {
                TensorMetadata tensor_meta{
                    .dtype = dtypeToTensorDataType( meta->dtype ),
                    .shape = meta->shape,
                    .total_bytes = meta->nbytes
                };

                TensorBlobView view( tensor_meta, map_base_ + meta->offset, static_cast<size_t>( meta->nbytes ) );

                consume( meta->name, view );
            }
        }

#ifdef MILA_HAS_CUDA
        template<typename TConsumer>
        void streamTensorBlobsPinned( TConsumer&& consume, int device_id )
        {
            const size_t count = sorted_by_offset_.size();

            if ( count == 0 )
            {
                return;
            }

            const size_t staging_bytes = std::min<size_t>( STAGING_BUFFER_CAPACITY_BYTES, getMaxTensorSizeBytes() );

            // TensorBuffer's constructor is explicit, so direct-initialize the two slots
            // and index them through a pointer array.
            TensorBuffer<dtype_t::UINT8, Compute::CudaPinnedMemoryResource> staging_a( device_id, staging_bytes );
            TensorBuffer<dtype_t::UINT8, Compute::CudaPinnedMemoryResource> staging_b( device_id, staging_bytes );
            TensorBuffer<dtype_t::UINT8, Compute::CudaPinnedMemoryResource>* staging[ 2 ] = { &staging_a, &staging_b };

            const TensorBlobMetadata* slot_meta[ 2 ] = { nullptr, nullptr };
            const void* slot_ptr[ 2 ] = { nullptr, nullptr };
            bool slot_ready[ 2 ] = { false, false };
            bool slot_free[ 2 ] = { true, true };

            std::mutex mutex;
            std::condition_variable cv_ready;
            std::condition_variable cv_free;
            std::exception_ptr producer_exception;
            bool aborted = false;

            std::thread producer( [&]
            {
                try
                {
                    for ( size_t k = 0; k < count; ++k )
                    {
                        const int slot = static_cast<int>( k & 1 );

                        {
                            std::unique_lock<std::mutex> lock( mutex );
                            cv_free.wait( lock, [&] { return slot_free[ slot ] || aborted; } );

                            if ( aborted )
                            {
                                return;
                            }
                        }

                        const TensorBlobMetadata* meta = sorted_by_offset_[ k ];
                        const uint8_t* src = map_base_ + meta->offset;
                        const void* ptr;

                        if ( static_cast<size_t>( meta->nbytes ) <= staging_bytes )
                        {
                            readAt( static_cast<uint64_t>( meta->offset ),
                                staging[ slot ]->data(), static_cast<size_t>( meta->nbytes ) );
                            ptr = staging[ slot ]->data();
                        }
                        else
                        {
                            // Oversized blob (e.g. token embedding): skip staging and let the
                            // consumer copy straight from the mapped view (pageable H2D). Only
                            // a couple of tensors hit this; it still uses the slot for ordering.
                            ptr = src;
                        }

                        {
                            std::lock_guard<std::mutex> lock( mutex );
                            slot_meta[ slot ] = meta;
                            slot_ptr[ slot ] = ptr;
                            slot_ready[ slot ] = true;
                            slot_free[ slot ] = false;
                            cv_ready.notify_one();
                        }
                    }
                }
                catch ( ... )
                {
                    std::lock_guard<std::mutex> lock( mutex );
                    producer_exception = std::current_exception();
                    cv_ready.notify_all();
                }
            } );

            try
            {
                for ( size_t k = 0; k < count; ++k )
                {
                    const int slot = static_cast<int>( k & 1 );

                    std::unique_lock<std::mutex> lock( mutex );
                    cv_ready.wait( lock, [&] { return slot_ready[ slot ] || producer_exception; } );

                    if ( producer_exception )
                    {
                        std::rethrow_exception( producer_exception );
                    }

                    const TensorBlobMetadata* meta = slot_meta[ slot ];
                    const void* ptr = slot_ptr[ slot ];
                    slot_ready[ slot ] = false;
                    lock.unlock();

                    TensorMetadata tensor_meta{
                        .dtype = dtypeToTensorDataType( meta->dtype ),
                        .shape = meta->shape,
                        .total_bytes = meta->nbytes
                    };

                    TensorBlobView view( tensor_meta, ptr, static_cast<size_t>( meta->nbytes ) );

                    // consume() must complete its device read before returning (see the
                    // contract on streamTensorBlobs); only then is the slot safe to overwrite.
                    consume( meta->name, view );

                    lock.lock();
                    slot_free[ slot ] = true;
                    cv_free.notify_one();
                }
            }
            catch ( ... )
            {
                {
                    std::lock_guard<std::mutex> lock( mutex );
                    aborted = true;
                    cv_free.notify_all();
                }

                producer.join();
                throw;
            }

            producer.join();
        }
#endif

        void readHeader()
        {
            uint32_t magic, version, num_tensors;

            file_.read( reinterpret_cast<char*>( &magic ), sizeof( magic ) );
            file_.read( reinterpret_cast<char*>( &version ), sizeof( version ) );
            file_.read( reinterpret_cast<char*>( &num_tensors ), sizeof( num_tensors ) );

            if ( !file_.good() )
            {
                throw std::runtime_error( "Failed to read binary header" );
            }

            if ( magic != MAGIC )
            {
                throw std::runtime_error(
                    std::format( "Invalid file format: wrong magic number (expected 0x{:X}, got 0x{:X})",
                        MAGIC, magic ) );
            }

            if ( version != VERSION )
            {
                throw std::runtime_error( std::format( "Unsupported file version: {}", version ) );
            }

            num_tensors_ = num_tensors;
        }

        void readMetadata()
        {
            uint32_t metadata_size;
            file_.read( reinterpret_cast<char*>( &metadata_size ), sizeof( metadata_size ) );

            if ( !file_.good() || metadata_size == 0 )
            {
                throw std::runtime_error( "Failed to read metadata size" );
            }

            std::string json_str( metadata_size, '\0' );
            file_.read( json_str.data(), static_cast<std::streamsize>( metadata_size ) );

            if ( !file_.good() )
            {
                throw std::runtime_error( "Failed to read metadata JSON" );
            }

            parseMetadataJSON( json_str );
        }

        void parseMetadataJSON( const std::string& json )
        {
            auto extract_string = [&]( const std::string& key ) -> std::string
            {
                auto pos = json.find( "\"" + key + "\"" );
                if ( pos == std::string::npos ) return "";

                auto value_start = json.find( ":", pos ) + 1;
                auto quote_start = json.find( "\"", value_start );
                if ( quote_start == std::string::npos ) return "";

                auto quote_end = json.find( "\"", quote_start + 1 );
                if ( quote_end == std::string::npos ) return "";

                return json.substr( quote_start + 1, quote_end - quote_start - 1 );
            };

            auto extract_int = [&]( const std::string& key ) -> uint32_t
            {
                auto pos = json.find( "\"" + key + "\"" );
                if ( pos == std::string::npos ) return 0;

                auto value_start = json.find( ":", pos ) + 1;
                auto comma_or_brace = json.find_first_of( ",}", value_start );
                if ( comma_or_brace == std::string::npos ) return 0;

                std::string value_str = json.substr( value_start, comma_or_brace - value_start );
                value_str.erase( 0, value_str.find_first_not_of( " \t\n\r" ) );
                value_str.erase( value_str.find_last_not_of( " \t\n\r" ) + 1 );

                try { return std::stoul( value_str ); }
                catch ( ... ) { return 0; }
            };

            auto extract_bool = [&]( const std::string& key ) -> bool
            {
                auto pos = json.find( "\"" + key + "\"" );
                if ( pos == std::string::npos ) return false;

                auto true_pos = json.find( "true", pos );
                auto false_pos = json.find( "false", pos );

                return (true_pos != std::string::npos) &&
                    (false_pos == std::string::npos || true_pos < false_pos);
            };

            auto extract_float = [&]( const std::string& key ) -> float
            {
                auto pos = json.find( "\"" + key + "\"" );
                if ( pos == std::string::npos ) return 0.0f;

                auto value_start = json.find( ":", pos ) + 1;
                auto comma_or_brace = json.find_first_of( ",}", value_start );
                if ( comma_or_brace == std::string::npos ) return 0.0f;

                std::string value_str = json.substr( value_start, comma_or_brace - value_start );
                value_str.erase( 0, value_str.find_first_not_of( " \t\n\r" ) );
                value_str.erase( value_str.find_last_not_of( " \t\n\r" ) + 1 );

                try { return std::stof( value_str ); }
                catch ( ... ) { return 0.0f; }
            };

            metadata_.architecture        = extract_string( "architecture" );
            metadata_.model_name          = extract_string( "model_name" );
            metadata_.vocab_size          = extract_int( "vocab_size" );
            metadata_.max_seq_length      = extract_int( "max_seq_length" );
            metadata_.embedding_dim       = extract_int( "embedding_dim" );
            metadata_.num_layers          = extract_int( "num_layers" );
            metadata_.num_heads           = extract_int( "num_heads" );
            metadata_.num_kv_heads        = extract_int( "num_kv_heads" );
            metadata_.head_dim            = extract_int( "head_dim" );
            metadata_.hidden_dim          = extract_int( "hidden_dim" );
            metadata_.use_bias            = extract_bool( "use_bias" );
            metadata_.tie_word_embeddings = extract_bool( "tie_word_embeddings" );
            metadata_.activation          = extract_string( "activation" );
            metadata_.norm_type           = extract_string( "norm_type" );
            metadata_.attention_type      = extract_string( "attention_type" );
            metadata_.positional_encoding = extract_string( "positional_encoding" );
            metadata_.rope_theta          = extract_float( "rope_theta" );
            metadata_.norm_epsilon        = extract_float( "norm_epsilon" );

            metadata_.global_head_dim         = extract_int( "global_head_dim" );
            metadata_.num_global_kv_heads     = extract_int( "num_global_kv_heads" );
            metadata_.key_equals_value        = extract_bool( "key_equals_value" );
            metadata_.window                  = extract_int( "window" );
            metadata_.sliding_window_pattern  = extract_int( "sliding_window_pattern" );
            metadata_.global_rotary_dim       = extract_int( "global_rotary_dim" );
            metadata_.rope_theta_local        = extract_float( "rope_theta_local" );
            metadata_.rope_theta_global       = extract_float( "rope_theta_global" );
            metadata_.final_logit_softcapping = extract_float( "final_logit_softcapping" );
        }

        void readTensorIndex()
        {
            for ( uint32_t i = 0; i < num_tensors_; ++i )
            {
                TensorBlobMetadata meta;

                uint32_t name_length;
                file_.read( reinterpret_cast<char*>( &name_length ), sizeof( name_length ) );

                if ( !file_.good() || name_length == 0 || name_length > 1024 )
                {
                    throw std::runtime_error(
                        std::format( "Invalid tensor name length at index {}", i ) );
                }

                meta.name.resize( name_length );
                file_.read( meta.name.data(), static_cast<std::streamsize>( name_length ) );

                file_.read( reinterpret_cast<char*>( &meta.dtype ), sizeof( meta.dtype ) );

                uint32_t ndim;
                file_.read( reinterpret_cast<char*>( &ndim ), sizeof( ndim ) );

                if ( !file_.good() || ndim > TensorShape::MaxRank )
                {
                    throw std::runtime_error(
                        std::format( "Invalid tensor dimensionality for '{}'", meta.name ) );
                }

                meta.shape.ndim = static_cast<uint8_t>( ndim );

                for ( uint32_t d = 0; d < ndim; ++d )
                {
                    uint32_t dim32;
                    file_.read( reinterpret_cast<char*>( &dim32 ), sizeof( dim32 ) );
                    meta.shape[ d ] = static_cast<int64_t>( dim32 );
                }

                file_.read( reinterpret_cast<char*>( &meta.offset ), sizeof( meta.offset ) );
                file_.read( reinterpret_cast<char*>( &meta.nbytes ), sizeof( meta.nbytes ) );

                if ( !file_.good() )
                {
                    throw std::runtime_error(
                        std::format( "Failed to read tensor metadata for '{}'", meta.name ) );
                }

                tensor_index_[ meta.name ] = meta;
            }
        }
    };
}