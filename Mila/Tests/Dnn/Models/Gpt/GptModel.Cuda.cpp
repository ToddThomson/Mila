/**
 * @file GptModel.Cuda.cpp
 * @brief GptModel's checkpoint round trip and the deployment context bound in generate().
 *
 * The fixture carries no real weights: a checkpoint written from a 16-position toy config reaches the context
 * bound in single-digit steps, and checkpoint fidelity is asserted by comparing archives blob for blob.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
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
        using GptCuda = Mila::Dnn::GptTransformer<DeviceType::Cuda, TensorDataType::FP32>;
        using GptModelCuda = Mila::Dnn::GptModel<DeviceType::Cuda, TensorDataType::FP32>;

        constexpr int64_t kEmbedding = 16;
        constexpr int64_t kLayers = 2;
        constexpr int64_t kHeads = 2;
        constexpr int64_t kVocab = 32;
        constexpr int64_t kMaxSeq = 16;
        constexpr int64_t kHidden = 64;
        constexpr int64_t kSeq = 4;

        // Small enough that the boundary is reached in single-digit steps, and well
        // inside kMaxSeq so the bound under test is the deployment context rather than
        // the trained maximum.
        constexpr dim_t kDeploymentContext = 8;
        constexpr int32_t kPromptToken = 1;
        constexpr int kPromptLength = 2;

        // Tokens the loop emits before the bound stops it: one from prefill, then one per
        // decode at positions kPromptLength .. kDeploymentContext - 1.
        constexpr int kTokensBeforeBound =
            1 + static_cast<int>( kDeploymentContext ) - kPromptLength;

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
                / std::format( "mila_test_gptmodel_cuda_{}_{}.mila", tag, stamp );
        }

        // A built network with a distinct value per tensor, written straight to an archive -- the same bytes
        // saveCheckpoint produces, since that is a thin wrapper over Network::save.
        void writeCheckpoint( const std::filesystem::path& path )
        {
            auto net = std::make_unique<GptCuda>( "gpt", smallConfig(), Device::Cuda( 0 ) );
            net->build( BuildContext( shape_t{ 1, kSeq }, RuntimeMode::Inference ) );

            float value = 0.5f;

            for ( auto* parameter : net->getParameters() )
            {
                fill( *static_cast<GptCuda::TensorType*>( parameter ), value );
                value += 0.25f;
            }

            net->synchronize();

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

        std::unique_ptr<GptModelCuda> loadBoundedModel( const std::filesystem::path& path )
        {
            writeCheckpoint( path );

            return GptModelCuda::fromCheckpoint(
                path, DeviceId{ DeviceType::Cuda, 0 }, kDeploymentContext );
        }
    }

    class GptModelCudaGenerateTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }
        }
    };

    class GptModelCudaCheckpointTests : public GptModelCudaGenerateTests
    {
    };

    // No config passed: the archive carries it. That is what separates fromCheckpoint from load.
    TEST_F( GptModelCudaCheckpointTests, FromCheckpoint_ReconstructsTheModelFromTheArchiveAlone )
    {
        const auto path = makeTempArchivePath( "reconstruct" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        writeCheckpoint( path );

        auto model = GptModelCuda::fromCheckpoint( path );

        ASSERT_NE( model, nullptr );

        const auto& config = model->getConfig();

        EXPECT_EQ( config.getVocabSize(), kVocab );
        EXPECT_EQ( config.getNumLayers(), kLayers );
        EXPECT_EQ( config.getEmbeddingSize(), kEmbedding );
        EXPECT_EQ( config.getNumHeads(), kHeads );
        EXPECT_EQ( config.getHiddenSize(), kHidden );
        EXPECT_EQ( config.getUseBias(), true );

        std::filesystem::remove( path, ec );
    }

    TEST_F( GptModelCudaCheckpointTests, FromCheckpoint_ThrowsOnAnArchiveItDidNotWrite )
    {
        const auto path = makeTempArchivePath( "foreign" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            archive.writeBlob( "unrelated.bin", "x", 1 );
        }

        EXPECT_THROW( (void)GptModelCuda::fromCheckpoint( path ), std::runtime_error );

        std::filesystem::remove( path, ec );
    }

    // Load a checkpoint, write one back out, and require the two archives to agree blob for blob: anything the
    // load leg dropped is absent on the way out, and anything mis-scoped lands under a different name.
    TEST_F( GptModelCudaCheckpointTests, SaveCheckpoint_RoundTripsEveryBlobUnchanged )
    {
        const auto source_path = makeTempArchivePath( "src" );
        const auto resaved_path = makeTempArchivePath( "resaved" );
        std::error_code ec;
        std::filesystem::remove( source_path, ec );
        std::filesystem::remove( resaved_path, ec );

        writeCheckpoint( source_path );

        auto model = GptModelCuda::fromCheckpoint( source_path );
        ASSERT_NE( model, nullptr );

        model->saveCheckpoint( resaved_path );

        {
            ModelArchive source( source_path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );
            ModelArchive resaved( resaved_path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );

            const auto source_blobs = tensorBlobPaths( source );
            const auto resaved_blobs = tensorBlobPaths( resaved );

            ASSERT_EQ( source_blobs, resaved_blobs );
            ASSERT_GT( source_blobs.size(), 8u ) << "fixture too small to exercise nesting";

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
        }

        std::filesystem::remove( source_path, ec );
        std::filesystem::remove( resaved_path, ec );
    }

    // The fixture's vocabulary is 32 while eos_token_ is 50256, so no sampled token can
    // end the run early -- the bound is the only exit, which is what makes the emitted
    // count an assertion rather than an observation.
    TEST_F( GptModelCudaGenerateTests, Generate_StopsAtTheContextBoundInsteadOfReadingPastIt )
    {
        const auto path = makeTempArchivePath( "context_bound" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto model = loadBoundedModel( path );
        ASSERT_NE( model, nullptr );

        const std::vector<int32_t> prompt( kPromptLength, kPromptToken );

        int emitted = 0;

        GenerateParams params;
        // Above the bound on purpose: the budget must not be what stops the loop, or the
        // case would pass with no guard at all.
        params.max_new_tokens = 64;

        const auto status = model->generate(
            prompt, [&emitted]( int32_t ) { ++emitted; }, params );

        EXPECT_EQ( status, GenerateStatus::ContextOverflow );
        EXPECT_EQ( emitted, kTokensBeforeBound );

        std::filesystem::remove( path, ec );
    }

    // The other half of the boundary: a run that fits must not report the overflow.
    // Without this, a guard that fired one position early would still pass the case above.
    TEST_F( GptModelCudaGenerateTests, Generate_ReportsTheBudgetWhenTheRunFitsInsideTheContext )
    {
        const auto path = makeTempArchivePath( "within_bound" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto model = loadBoundedModel( path );
        ASSERT_NE( model, nullptr );

        const std::vector<int32_t> prompt( kPromptLength, kPromptToken );

        int emitted = 0;

        GenerateParams params;
        params.max_new_tokens = kTokensBeforeBound - 1;

        const auto status = model->generate(
            prompt, [&emitted]( int32_t ) { ++emitted; }, params );

        EXPECT_EQ( status, GenerateStatus::MaxNewTokensReached );
        EXPECT_EQ( emitted, *params.max_new_tokens );

        std::filesystem::remove( path, ec );
    }

    // A prompt that cannot fit is rejected before any device work, rather than truncated
    // -- the caller is told, and the distinction matters because a silently truncated
    // prompt produces a plausible answer to a question nobody asked.
    TEST_F( GptModelCudaGenerateTests, Generate_RejectsAPromptLongerThanTheContext )
    {
        const auto path = makeTempArchivePath( "long_prompt" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto model = loadBoundedModel( path );
        ASSERT_NE( model, nullptr );

        const std::vector<int32_t> prompt(
            static_cast<size_t>( kDeploymentContext ) + 1, kPromptToken );

        EXPECT_THROW(
            (void)model->generate( prompt, []( int32_t ) {} ),
            std::invalid_argument );

        std::filesystem::remove( path, ec );
    }
}
