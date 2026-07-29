/**
 * @file SafeTensors.Cpu.cpp
 * @brief safetensors writer/reader round trip, and a regression guard on the MILA path.
 *
 * PretrainedModelReader now accepts two containers, sniffed by the leading magic. The
 * risk that carries is not that the new path is wrong -- a wrong new path fails loudly
 * on its own tests -- but that adding it quietly breaks reading of the .bin files
 * already on disk, which no test covered because every one of them needs a converted
 * model this suite does not have.
 *
 * So this file writes both containers itself. The MILA case is built byte by byte from
 * the format definition rather than from a fixture, which pins the legacy path against
 * a written specification instead of against whatever a converter happened to emit.
 *
 * CPU device, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Serialization
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Serialization;

    namespace
    {
        /**
         * @brief Unique scratch path per test, removed on destruction.
         */
        class ScratchFile
        {
        public:

            explicit ScratchFile( const std::string& stem )
            {
                path_ = std::filesystem::temp_directory_path()
                    / std::format( "mila_safetensors_{}_{}.bin", stem, counter_++ );
            }

            ~ScratchFile()
            {
                std::error_code ignored;
                std::filesystem::remove( path_, ignored );
            }

            const std::filesystem::path& path() const { return path_; }

        private:

            std::filesystem::path path_;
            static inline int counter_ = 0;
        };

        using FilePointer = std::unique_ptr<std::FILE, int( * )( std::FILE* )>;

        FilePointer openFile( const std::filesystem::path& path, const char* mode )
        {
            return FilePointer( std::fopen( path.string().c_str(), mode ), &std::fclose );
        }

        /**
         * @brief Write a minimal MILA-format file from the format definition.
         *
         * Deliberately hand-rolled rather than produced by the writer under test: the
         * point is to assert that the legacy path still reads bytes laid out the way
         * MilaWeightWriter lays them out.
         *
         * C stdio rather than fstream throughout this file -- MSVC C++23 raises C2079 on
         * basic_istream::sentry when stream I/O meets `import Mila;` in a .cpp.
         */
        void writeMilaFormatFile( const std::filesystem::path& path,
            const std::string& metadata_json,
            const std::string& tensor_name,
            const std::vector<float>& values )
        {
            FilePointer file = openFile( path, "wb" );
            ASSERT_NE( file.get(), nullptr );

            const uint32_t magic = 0x4D494C41;
            const uint32_t version = 1;
            const uint32_t num_tensors = 1;

            std::fwrite( &magic, sizeof( magic ), 1, file.get() );
            std::fwrite( &version, sizeof( version ), 1, file.get() );
            std::fwrite( &num_tensors, sizeof( num_tensors ), 1, file.get() );

            const uint32_t metadata_size = static_cast<uint32_t>( metadata_json.size() );
            std::fwrite( &metadata_size, sizeof( metadata_size ), 1, file.get() );
            std::fwrite( metadata_json.data(), 1, metadata_json.size(), file.get() );

            const uint32_t name_length = static_cast<uint32_t>( tensor_name.size() );
            const uint32_t dtype_code = 0;  // float32
            const uint32_t ndim = 2;
            const uint32_t dim0 = 2;
            const uint32_t dim1 = static_cast<uint32_t>( values.size() / 2 );

            const long index_start = std::ftell( file.get() );
            const uint64_t index_size = 4 + name_length + 4 + 4 + 4 * ndim + 8 + 8;
            const uint64_t data_offset = static_cast<uint64_t>( index_start ) + index_size;
            const uint64_t nbytes = values.size() * sizeof( float );

            std::fwrite( &name_length, sizeof( name_length ), 1, file.get() );
            std::fwrite( tensor_name.data(), 1, tensor_name.size(), file.get() );
            std::fwrite( &dtype_code, sizeof( dtype_code ), 1, file.get() );
            std::fwrite( &ndim, sizeof( ndim ), 1, file.get() );
            std::fwrite( &dim0, sizeof( dim0 ), 1, file.get() );
            std::fwrite( &dim1, sizeof( dim1 ), 1, file.get() );
            std::fwrite( &data_offset, sizeof( data_offset ), 1, file.get() );
            std::fwrite( &nbytes, sizeof( nbytes ), 1, file.get() );

            ASSERT_EQ( static_cast<uint64_t>( std::ftell( file.get() ) ), data_offset );

            std::fwrite( values.data(), 1, static_cast<size_t>( nbytes ), file.get() );
        }
    }

    // ================================================================
    // safetensors container round trip
    // ================================================================

    TEST( SafeTensors, RoundTripsTensorsOfMixedDataTypes )
    {
        ScratchFile scratch( "mixed" );

        const std::vector<float> weights{ 1.0f, -2.5f, 3.25f, 4.75f, -5.5f, 6.0f };
        const std::vector<uint8_t> packed{ 0x0F, 0xA3, 0x71, 0xC2 };
        const std::vector<int32_t> counts{ 7, -11, 13 };

        {
            SafeTensorsWriter writer( scratch.path() );

            writer.declareTensor( "block.weight", TensorDataType::FP32, shape_t{ 2, 3 } );
            writer.declareTensor( "block.weight.packed", TensorDataType::UINT8, shape_t{ 4 } );
            writer.declareTensor( "block.counts", TensorDataType::INT32, shape_t{ 3 } );

            EXPECT_EQ( writer.getTensorCount(), 3u );

            writer.beginData();
            writer.writeTensorData( "block.weight", weights.data(), weights.size() * sizeof( float ) );
            writer.writeTensorData( "block.weight.packed", packed.data(), packed.size() );
            writer.writeTensorData( "block.counts", counts.data(), counts.size() * sizeof( int32_t ) );
            writer.close();
        }

        PretrainedModelReader reader( scratch.path() );

        EXPECT_TRUE( reader.hasTensor( "block.weight" ) );
        EXPECT_TRUE( reader.hasTensor( "block.weight.packed" ) );
        EXPECT_TRUE( reader.hasTensor( "block.counts" ) );
        EXPECT_EQ( reader.getTensorNames().size(), 3u );

        auto weight_blob = reader.readTensorBlob<Compute::CpuMemoryResource>( "block.weight" );
        const auto& weight_meta = weight_blob.getMetadata();

        EXPECT_EQ( weight_meta.dtype, TensorDataType::FP32 );
        ASSERT_EQ( weight_meta.shape.ndim, 2 );
        EXPECT_EQ( weight_meta.shape[ 0 ], 2 );
        EXPECT_EQ( weight_meta.shape[ 1 ], 3 );
        ASSERT_EQ( weight_blob.sizeBytes(), weights.size() * sizeof( float ) );
        EXPECT_EQ( std::memcmp( weight_blob.data(), weights.data(), weight_blob.sizeBytes() ), 0 );

        auto packed_blob = reader.readTensorBlob<Compute::CpuMemoryResource>( "block.weight.packed" );

        EXPECT_EQ( packed_blob.getMetadata().dtype, TensorDataType::UINT8 );
        ASSERT_EQ( packed_blob.sizeBytes(), packed.size() );
        EXPECT_EQ( std::memcmp( packed_blob.data(), packed.data(), packed_blob.sizeBytes() ), 0 );

        auto counts_blob = reader.readTensorBlob<Compute::CpuMemoryResource>( "block.counts" );

        EXPECT_EQ( counts_blob.getMetadata().dtype, TensorDataType::INT32 );
        ASSERT_EQ( counts_blob.sizeBytes(), counts.size() * sizeof( int32_t ) );
        EXPECT_EQ( std::memcmp( counts_blob.data(), counts.data(), counts_blob.sizeBytes() ), 0 );
    }

    TEST( SafeTensors, CarriesMilaConfigThroughMetadata )
    {
        ScratchFile scratch( "config" );

        const std::vector<float> values{ 1.0f, 2.0f };
        const std::string config =
            R"({"architecture":"llama","model_name":"tiny","vocab_size":128,)"
            R"("embedding_dim":64,"num_layers":2,"num_heads":4,"num_kv_heads":2,)"
            R"("rope_theta":10000.0,"tie_word_embeddings":true})";

        {
            SafeTensorsWriter writer( scratch.path() );
            writer.declareTensor( "w", TensorDataType::FP32, shape_t{ 2 } );
            writer.setMetadata( kMilaConfigMetadataKey, config );
            writer.beginData();
            writer.writeTensorData( "w", values.data(), values.size() * sizeof( float ) );
            writer.close();
        }

        PretrainedModelReader reader( scratch.path() );
        const auto& metadata = reader.getPretrainedMetadata();

        EXPECT_EQ( metadata.architecture, "llama" );
        EXPECT_EQ( metadata.model_name, "tiny" );
        EXPECT_EQ( metadata.vocab_size, 128u );
        EXPECT_EQ( metadata.embedding_dim, 64u );
        EXPECT_EQ( metadata.num_layers, 2u );
        EXPECT_EQ( metadata.num_heads, 4u );
        EXPECT_EQ( metadata.num_kv_heads, 2u );
        EXPECT_TRUE( metadata.tie_word_embeddings );
        EXPECT_FLOAT_EQ( metadata.rope_theta, 10000.0f );
    }

    TEST( SafeTensors, ReadsAFileThatCarriesNoMilaConfig )
    {
        ScratchFile scratch( "foreign" );

        const std::vector<float> values{ 3.0f, 4.0f };

        {
            SafeTensorsWriter writer( scratch.path() );
            writer.declareTensor( "w", TensorDataType::FP32, shape_t{ 2 } );
            writer.beginData();
            writer.writeTensorData( "w", values.data(), values.size() * sizeof( float ) );
            writer.close();
        }

        // A foreign safetensors file is still readable as tensors; only the architecture
        // metadata is absent, which a model factory rejects on its own terms.
        PretrainedModelReader reader( scratch.path() );

        EXPECT_TRUE( reader.hasTensor( "w" ) );
        EXPECT_TRUE( reader.getPretrainedMetadata().architecture.empty() );
    }

    // ================================================================
    // Writer contract
    // ================================================================

    TEST( SafeTensors, RejectsOutOfOrderBodyWrites )
    {
        ScratchFile scratch( "order" );

        const std::vector<float> a{ 1.0f };
        const std::vector<float> b{ 2.0f };

        SafeTensorsWriter writer( scratch.path() );
        writer.declareTensor( "first", TensorDataType::FP32, shape_t{ 1 } );
        writer.declareTensor( "second", TensorDataType::FP32, shape_t{ 1 } );
        writer.beginData();

        // Out of order would tile the data region against the wrong index entries.
        EXPECT_THROW(
            writer.writeTensorData( "second", b.data(), sizeof( float ) ),
            std::runtime_error );
    }

    TEST( SafeTensors, RejectsBodySizeMismatch )
    {
        ScratchFile scratch( "size" );

        const std::vector<float> values{ 1.0f, 2.0f, 3.0f };

        SafeTensorsWriter writer( scratch.path() );
        writer.declareTensor( "w", TensorDataType::FP32, shape_t{ 2 } );
        writer.beginData();

        EXPECT_THROW(
            writer.writeTensorData( "w", values.data(), 3 * sizeof( float ) ),
            std::runtime_error );
    }

    TEST( SafeTensors, RejectsDuplicateTensorNames )
    {
        ScratchFile scratch( "duplicate" );

        SafeTensorsWriter writer( scratch.path() );
        writer.declareTensor( "w", TensorDataType::FP32, shape_t{ 1 } );

        EXPECT_THROW(
            writer.declareTensor( "w", TensorDataType::FP32, shape_t{ 1 } ),
            std::runtime_error );
    }

    TEST( SafeTensors, RejectsDeclarationAfterHeaderIsWritten )
    {
        ScratchFile scratch( "late" );

        SafeTensorsWriter writer( scratch.path() );
        writer.declareTensor( "w", TensorDataType::FP32, shape_t{ 1 } );
        writer.beginData();

        EXPECT_THROW(
            writer.declareTensor( "late", TensorDataType::FP32, shape_t{ 1 } ),
            std::runtime_error );
    }

    TEST( SafeTensors, CloseRefusesWhenADeclaredTensorWasNeverWritten )
    {
        ScratchFile scratch( "incomplete" );

        const std::vector<float> a{ 1.0f };

        SafeTensorsWriter writer( scratch.path() );
        writer.declareTensor( "first", TensorDataType::FP32, shape_t{ 1 } );
        writer.declareTensor( "second", TensorDataType::FP32, shape_t{ 1 } );
        writer.beginData();
        writer.writeTensorData( "first", a.data(), sizeof( float ) );

        // The header promises bytes the file does not contain; silence here would
        // produce a file that only fails much later, at load.
        EXPECT_THROW( writer.close(), std::runtime_error );
    }

    // ================================================================
    // Reader robustness
    // ================================================================

    TEST( SafeTensors, RejectsAFileThatIsNeitherContainer )
    {
        ScratchFile scratch( "garbage" );

        {
            FilePointer file = openFile( scratch.path(), "wb" );
            ASSERT_NE( file.get(), nullptr );

            const std::string junk( 64, 'x' );
            std::fwrite( junk.data(), 1, junk.size(), file.get() );
        }

        EXPECT_THROW( PretrainedModelReader reader( scratch.path() ), std::runtime_error );
    }

    TEST( SafeTensors, RejectsATensorExtendingPastEndOfFile )
    {
        ScratchFile scratch( "truncated" );

        const std::vector<float> values{ 1.0f, 2.0f, 3.0f, 4.0f };

        {
            SafeTensorsWriter writer( scratch.path() );
            writer.declareTensor( "w", TensorDataType::FP32, shape_t{ 4 } );
            writer.beginData();
            writer.writeTensorData( "w", values.data(), values.size() * sizeof( float ) );
            writer.close();
        }

        const auto full_size = std::filesystem::file_size( scratch.path() );
        std::filesystem::resize_file( scratch.path(), full_size - sizeof( float ) );

        EXPECT_THROW( PretrainedModelReader reader( scratch.path() ), std::runtime_error );
    }

    // ================================================================
    // MILA format regression guard
    // ================================================================

    TEST( SafeTensors, StillReadsTheLegacyMilaContainer )
    {
        ScratchFile scratch( "legacy" );

        const std::vector<float> values{ 1.5f, -2.5f, 3.5f, -4.5f, 5.5f, -6.5f };
        const std::string metadata =
            R"({"architecture":"gpt2","model_name":"legacy","vocab_size":50257,)"
            R"("embedding_dim":768,"num_layers":12,"num_heads":12})";

        writeMilaFormatFile( scratch.path(), metadata, "lenc.wte.weight", values );

        PretrainedModelReader reader( scratch.path() );

        EXPECT_EQ( reader.getPretrainedMetadata().architecture, "gpt2" );
        EXPECT_EQ( reader.getPretrainedMetadata().vocab_size, 50257u );
        EXPECT_EQ( reader.getPretrainedMetadata().num_layers, 12u );

        ASSERT_TRUE( reader.hasTensor( "lenc.wte.weight" ) );

        auto blob = reader.readTensorBlob<Compute::CpuMemoryResource>( "lenc.wte.weight" );
        const auto& meta = blob.getMetadata();

        EXPECT_EQ( meta.dtype, TensorDataType::FP32 );
        ASSERT_EQ( meta.shape.ndim, 2 );
        EXPECT_EQ( meta.shape[ 0 ], 2 );
        EXPECT_EQ( meta.shape[ 1 ], 3 );
        ASSERT_EQ( blob.sizeBytes(), values.size() * sizeof( float ) );
        EXPECT_EQ( std::memcmp( blob.data(), values.data(), blob.sizeBytes() ), 0 );
    }

    TEST( SafeTensors, LegacyContainerStillRejectsAWrongVersion )
    {
        ScratchFile scratch( "legacy_version" );

        const std::vector<float> values{ 1.0f, 2.0f };
        writeMilaFormatFile( scratch.path(), R"({"architecture":"gpt2"})", "w", values );

        // Bump the version word in place; the magic still matches, so this must be
        // rejected by the legacy path rather than fall through to the safetensors parse.
        {
            FilePointer file = openFile( scratch.path(), "r+b" );
            ASSERT_NE( file.get(), nullptr );

            const uint32_t bad_version = 99;
            std::fseek( file.get(), sizeof( uint32_t ), SEEK_SET );
            std::fwrite( &bad_version, sizeof( bad_version ), 1, file.get() );
        }

        EXPECT_THROW( PretrainedModelReader reader( scratch.path() ), std::runtime_error );
    }
}
