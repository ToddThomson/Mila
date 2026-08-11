/**
 * @file GptModel.Cpu.cpp
 * @brief Checkpoint round trip through the GptModel public API.
 *
 * GptTransformer.Cpu.cpp covers the network-level save/load traversal. This file
 * covers the layer above it -- saveCheckpoint() and fromCheckpoint(), which are what
 * a caller actually uses and which the network-level tests bypass entirely.
 *
 * fromCheckpoint is the only GptModel factory reachable without a converted .bin:
 * fromPretrained needs a PretrainedModelReader over real weights, while a checkpoint
 * carries its own config, so a model can be reconstructed from nothing but an archive
 * this suite writes. That property is the whole point of the phase and is what is
 * asserted here.
 *
 * The model deliberately exposes no parameter accessor -- getLanguageNetwork() is
 * protected -- so weight fidelity is asserted by comparing ARCHIVES rather than
 * tensors: load a checkpoint, write one back out, and require the two to agree blob
 * for blob. That uses only the public API and is a stronger statement than reading
 * parameters would be, since it covers both legs of the trip.
 *
 * The generation context bound is covered in GptModel.Cuda.cpp instead: the CPU
 * operation layer takes its extents from build() rather than from the input, so CPU
 * inference cannot run a prompt shorter than the built context at all.
 *
 * CPU device, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <stdexcept>
#include <system_error>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    namespace
    {
        using GptCpu = Mila::Dnn::GptTransformer<DeviceType::Cpu, TensorDataType::FP32>;
        using GptModelCpu = Mila::Dnn::GptModel<DeviceType::Cpu, TensorDataType::FP32>;
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        constexpr int64_t kEmbedding = 16;
        constexpr int64_t kLayers = 2;
        constexpr int64_t kHeads = 2;
        constexpr int64_t kVocab = 32;
        constexpr int64_t kMaxSeq = 16;
        constexpr int64_t kHidden = 64;
        constexpr int64_t kSeq = 4;

        GptConfig smallConfig()
        {
            GptConfig config( kEmbedding, kLayers );
            config.withVocabSize( kVocab )
                .withNumHeads( kHeads )
                .withMaxSequenceLength( kMaxSeq )
                .withHiddenSize( kHidden )
                .withBias( true );

            return config;
        }

        std::filesystem::path makeTempArchivePath( const std::string& tag )
        {
            const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();

            return std::filesystem::temp_directory_path()
                / std::format( "mila_test_gptmodel_{}_{}.mila", tag, stamp );
        }

        // A built network with a distinct value per tensor, written straight to an
        // archive -- the same bytes saveCheckpoint produces, since that is a thin
        // wrapper over Network::save.
        void writeCheckpoint( const std::filesystem::path& path )
        {
            auto net = std::make_unique<GptCpu>( "gpt", smallConfig(), Device::Cpu() );
            net->build( BuildContext( shape_t{ 1, kSeq }, RuntimeMode::Inference ) );

            float value = 0.5f;

            for ( auto* parameter : net->getParameters() )
            {
                fill( *static_cast<HostTensor*>( parameter ), value );
                value += 0.25f;
            }

            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            net->save( archive, SerializationMode::Checkpoint );
        }

        std::vector<std::string> tensorBlobPaths( const ModelArchive& archive )
        {
            std::vector<std::string> blobs;

            for ( const auto& name : archive.listFiles() )
            {
                if ( name.ends_with( "/data.bin" ) )
                {
                    blobs.push_back( name );
                }
            }

            std::sort( blobs.begin(), blobs.end() );

            return blobs;
        }
    }

    class GptModelCpuTests : public ::testing::Test
    {
    };

    // ====================================================================
    // fromCheckpoint
    // ====================================================================

    TEST_F( GptModelCpuTests, FromCheckpoint_ReconstructsTheModelFromTheArchiveAlone )
    {
        const auto path = makeTempArchivePath( "reconstruct" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        writeCheckpoint( path );

        // No config passed: the archive carries it. That is the property separating
        // fromCheckpoint from fromPretrained, which must be handed the geometry.
        auto model = GptModelCpu::fromCheckpoint( path );

        ASSERT_NE( model, nullptr );

        const auto& config = model->getConfig();

        EXPECT_EQ( config.getVocabSize(), kVocab );
        EXPECT_EQ( config.getNumLayers(), kLayers );
        EXPECT_EQ( config.getEmbeddingSize(), kEmbedding );
        EXPECT_EQ( config.getNumHeads(), kHeads );

        // The two fields the previous hand-rolled metadata block got wrong: hidden size
        // was written under a key fromMetadata does not read, and use_bias was omitted.
        EXPECT_EQ( config.getHiddenSize(), kHidden );
        EXPECT_EQ( config.getUseBias(), true );

        std::filesystem::remove( path, ec );
    }

    TEST_F( GptModelCpuTests, FromCheckpoint_ThrowsOnAnArchiveItDidNotWrite )
    {
        const auto path = makeTempArchivePath( "foreign" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            archive.writeBlob( "unrelated.bin", "x", 1 );
        }

        EXPECT_THROW( (void)GptModelCpu::fromCheckpoint( path ), std::runtime_error );

        std::filesystem::remove( path, ec );
    }

    // ====================================================================
    // saveCheckpoint -- the full public cycle
    // ====================================================================

    TEST_F( GptModelCpuTests, SaveCheckpoint_RoundTripsEveryBlobUnchanged )
    {
        const auto source_path = makeTempArchivePath( "src" );
        const auto resaved_path = makeTempArchivePath( "resaved" );
        std::error_code ec;
        std::filesystem::remove( source_path, ec );
        std::filesystem::remove( resaved_path, ec );

        writeCheckpoint( source_path );

        auto model = GptModelCpu::fromCheckpoint( source_path );
        ASSERT_NE( model, nullptr );

        model->saveCheckpoint( resaved_path );

        ModelArchive source( source_path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );
        ModelArchive resaved( resaved_path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );

        const auto source_blobs = tensorBlobPaths( source );
        const auto resaved_blobs = tensorBlobPaths( resaved );

        // Same set of paths: anything the load leg dropped would be absent on the way
        // back out, and anything mis-scoped would land under a different name.
        ASSERT_EQ( source_blobs, resaved_blobs );
        ASSERT_GT( source_blobs.size(), 8u ) << "fixture too small to exercise nesting";

        // Same bytes: the values survived load and save.
        for ( const auto& blob : source_blobs )
        {
            const size_t size = source.getFileSize( blob );

            ASSERT_EQ( resaved.getFileSize( blob ), size ) << blob;
            ASSERT_GT( size, 0u ) << blob;

            std::vector<std::uint8_t> expected( size );
            std::vector<std::uint8_t> actual( size );

            ASSERT_EQ( source.readBlobInto( blob, expected.data(), size ), size ) << blob;
            ASSERT_EQ( resaved.readBlobInto( blob, actual.data(), size ), size ) << blob;

            EXPECT_EQ( expected, actual ) << blob;
        }

        std::filesystem::remove( source_path, ec );
        std::filesystem::remove( resaved_path, ec );
    }
}
