/**
 * @file GptTransformer.Cuda.cpp
 * @brief Concrete-network tests for GptTransformer<DeviceType::Cuda, FP32>.
 *
 * GptTransformer is the GPT-2 decoder-only language network:
 *   Lpe (token + positional embedding) -> N x GptBlock -> final LayerNorm -> lm_head
 * Covers the build contract, the runtime modes, forward and backward, and the save and load round trip across
 * nested blocks.
 *
 * CUDA device tests -- skipped when no CUDA device is present.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <stdexcept>
#include <system_error>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::Transformers::Gpt
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    namespace
    {
        using GptCuda = Mila::Dnn::GptTransformer<DeviceType::Cuda, TensorDataType::FP32>;
        using HostTokenTensor = Tensor<TensorDataType::INT32, CpuMemoryResource>;
        using HostLogitsTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        constexpr int64_t kEmbedding = 16;
        constexpr int64_t kLayers = 2;
        constexpr int64_t kHeads = 2;
        constexpr int64_t kVocab = 32;
        constexpr int64_t kMaxSeq = 16;
        constexpr int64_t kHidden = 64;

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
    }

    class GptTransformerCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }
        }

        std::unique_ptr<GptCuda> builtNet( int64_t batch, int64_t seq, RuntimeMode mode )
        {
            auto net = std::make_unique<GptCuda>( "gpt", smallConfig(), Device::Cuda( 0 ) );
            net->build( BuildContext( shape_t{ batch, seq }, mode ) );

            return net;
        }

        // Deterministic in-range token ids on the device.
        GptCuda::TokenIndexType makeTokens( int64_t batch, int64_t seq )
        {
            HostTokenTensor host( Device::Cpu(), shape_t{ batch, seq } );
            auto* data = host.data();

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                data[ i ] = static_cast<int32_t>( i % static_cast<size_t>( kVocab ) );
            }

            GptCuda::TokenIndexType device_tokens( Device::Cuda( 0 ), shape_t{ batch, seq } );
            copy( host, device_tokens );

            return device_tokens;
        }

        static constexpr int64_t batch_ = 1;
        static constexpr int64_t seq_ = 4;
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( GptTransformerCudaTests, Construct_StandaloneSucceeds )
    {
        GptCuda net( "gpt", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( net.getName(), "gpt" );
        EXPECT_EQ( net.getDeviceId().type, DeviceType::Cuda );
    }

    TEST_F( GptTransformerCudaTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW( GptCuda( "gpt", smallConfig(), Device::Cpu() ), std::invalid_argument );
    }

    TEST_F( GptTransformerCudaTests, Construct_InvalidConfigThrows )
    {
        // embedding_size 0 fails GptConfig::validate(), invoked in the ctor.
        GptConfig bad( 0, kLayers );

        EXPECT_THROW( GptCuda( "gpt", bad, Device::Cuda( 0 ) ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle
    // ====================================================================

    TEST_F( GptTransformerCudaTests, Build_SetsIsBuilt )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        EXPECT_TRUE( net->isBuilt() );
    }

    TEST_F( GptTransformerCudaTests, Build_ThrowsOnNonRank2Input )
    {
        GptCuda net( "gpt", smallConfig(), Device::Cuda( 0 ) );

        // Rank-3 input violates the [B, T] token-index contract.
        EXPECT_THROW( net.build( BuildContext( shape_t{ batch_, seq_, kEmbedding }, RuntimeMode::Inference ) ),
            std::invalid_argument );
    }

    TEST_F( GptTransformerCudaTests, Forward_ThrowsBeforeBuild )
    {
        GptCuda net( "gpt", smallConfig(), Device::Cuda( 0 ) );
        auto input = makeTokens( batch_, seq_ );

        EXPECT_THROW( net.forward( input ), std::runtime_error );
    }

    TEST_F( GptTransformerCudaTests, Backward_ThrowsBeforeBuild )
    {
        GptCuda net( "gpt", smallConfig(), Device::Cuda( 0 ) );
        auto input = makeTokens( batch_, seq_ );
        GptCuda::TensorType output_grad( Device::Cuda( 0 ), shape_t{ batch_, seq_, kVocab } );

        EXPECT_THROW( net.backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // C. Execution Context / Device
    // ====================================================================

    TEST_F( GptTransformerCudaTests, Synchronize_Succeeds )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        EXPECT_NO_THROW( net->synchronize() );
    }

    // ====================================================================
    // D. Runtime + Training Mode
    // ====================================================================

    TEST_F( GptTransformerCudaTests, BuiltForTraining_IsTrainingMode )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Training );

        EXPECT_TRUE( net->isTrainingMode() );
        EXPECT_FALSE( net->isInferenceMode() );
    }

    TEST_F( GptTransformerCudaTests, BuiltForInference_IsInferenceMode )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        EXPECT_TRUE( net->isInferenceMode() );
        EXPECT_FALSE( net->isTrainingMode() );
    }

    TEST_F( GptTransformerCudaTests, Backward_ThrowsWhenBuiltForInference )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        auto input = makeTokens( batch_, seq_ );
        GptCuda::TensorType output_grad( Device::Cuda( 0 ), shape_t{ batch_, seq_, kVocab } );

        net->forward( input );

        EXPECT_THROW( net->backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // E. Forward (logits shape + finiteness)
    // ====================================================================

    TEST_F( GptTransformerCudaTests, Forward_ProducesLogitsShape )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Training );

        auto input = makeTokens( batch_, seq_ );
        auto& logits = net->forward( input );

        EXPECT_EQ( logits.shape(), ( shape_t{ batch_, seq_, kVocab } ) );
    }

    TEST_F( GptTransformerCudaTests, Forward_ProducesFiniteLogits )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Training );

        auto input = makeTokens( batch_, seq_ );
        auto& logits = net->forward( input );

        HostLogitsTensor host_logits( Device::Cpu(), shape_t{ batch_, seq_, kVocab } );
        copy( logits, host_logits );
        net->synchronize();

        const auto* data = host_logits.data();
        for ( dim_t i = 0; i < host_logits.size(); ++i )
        {
            ASSERT_TRUE( std::isfinite( data[ i ] ) ) << "non-finite logit at index " << i;
        }
    }

    // ====================================================================
    // F. Backward (runs in training mode after forward)
    // ====================================================================

    TEST_F( GptTransformerCudaTests, Backward_RunsAfterForwardInTrainingMode )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Training );

        auto input = makeTokens( batch_, seq_ );

        HostLogitsTensor host_grad( Device::Cpu(), shape_t{ batch_, seq_, kVocab } );
        for ( dim_t i = 0; i < host_grad.size(); ++i )
        {
            host_grad.data()[ i ] = 0.01f * static_cast<float>( i + 1 );
        }

        GptCuda::TensorType output_grad( Device::Cuda( 0 ), shape_t{ batch_, seq_, kVocab } );
        copy( host_grad, output_grad );

        net->forward( input );

        EXPECT_NO_THROW( net->backward( input, output_grad ) );
    }

    TEST_F( GptTransformerCudaTests, Backward_ThrowsWhenForwardNotCalled )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Training );

        auto input = makeTokens( batch_, seq_ );
        GptCuda::TensorType output_grad( Device::Cuda( 0 ), shape_t{ batch_, seq_, kVocab } );

        EXPECT_THROW( net->backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // G. Parameters & Components
    // ====================================================================

    TEST_F( GptTransformerCudaTests, GetComponents_ReturnsLayersPlusThree )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        // encoder + kLayers blocks + final_layernorm + lm_head.
        EXPECT_EQ( net->getComponents().size(), static_cast<size_t>( kLayers + 3 ) );
    }

    TEST_F( GptTransformerCudaTests, ParameterCount_Positive )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        EXPECT_GT( net->parameterCount(), 0 );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( GptTransformerCudaTests, ToString_DescribesNetwork )
    {
        GptCuda net( "my_gpt", smallConfig(), Device::Cuda( 0 ) );

        const std::string text = net.toString();

        EXPECT_NE( text.find( "Gpt" ), std::string::npos );
        EXPECT_NE( text.find( "my_gpt" ), std::string::npos );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TEST_F( GptTransformerCudaTests, GetType_IsGpt2 )
    {
        GptCuda net( "gpt", smallConfig(), Device::Cuda( 0 ) );

        // Structural kind is Network; the GPT-2 architecture identity is ModelType.
        EXPECT_EQ( net.getType(), ComponentType::Network );
        EXPECT_EQ( net.getModelType(), ModelType::Gpt2 );
    }

    // ====================================================================
    // K. Serialization round trip
    // ====================================================================

    namespace
    {
        std::filesystem::path makeTempArchivePath( const std::string& tag )
        {
            const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();

            return std::filesystem::temp_directory_path()
                / std::format( "mila_test_gpt_cuda_{}_{}.mila", tag, stamp );
        }

        void fillParametersDistinctly( GptCuda& net, float first, float step )
        {
            float value = first;

            for ( auto* parameter : net.getParameters() )
            {
                fill( *static_cast<GptCuda::TensorType*>( parameter ), value );
                value += step;
            }

            net.synchronize();
        }

        HostLogitsTensor parameterOnHost( GptCuda& net, const ITensor* parameter )
        {
            const auto& device = *static_cast<const GptCuda::TensorType*>( parameter );
            HostLogitsTensor host( Device::Cpu(), device.shape() );
            copy( device, host );
            net.synchronize();

            return host;
        }
    }

    TEST_F( GptTransformerCudaTests, SaveThenLoad_RestoresEveryParameterAcrossNestedBlocks )
    {
        const auto path = makeTempArchivePath( "roundtrip" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto source = builtNet( batch_, seq_, RuntimeMode::Inference );

        // A distinct value per tensor, so a collision or a cross-wired restore shows as one parameter holding
        // another's value.
        fillParametersDistinctly( *source, 0.5f, 0.25f );

        const auto parameter_count = source->getParameters().size();
        ASSERT_GT( parameter_count, 8u ) << "fixture too small to exercise nesting";

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            source->save( archive, SerializationMode::Checkpoint );
        }

        ModelArchive reader( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );

        const auto files = reader.listFiles();
        const auto blob_count = std::count_if( files.begin(), files.end(),
            []( const std::string& name )
            {
                return name.ends_with( "/data.bin" );
            } );

        EXPECT_EQ( static_cast<size_t>( blob_count ), parameter_count );

        // Poisoned, so "restored" cannot be confused with "happened to already match".
        auto target = builtNet( batch_, seq_, RuntimeMode::Inference );
        fillParametersDistinctly( *target, -1.0f, 0.0f );

        target->load( reader, SerializationMode::Checkpoint );

        const auto source_parameters = source->getParameters();
        const auto target_parameters = target->getParameters();

        ASSERT_EQ( source_parameters.size(), target_parameters.size() );

        for ( size_t i = 0; i < source_parameters.size(); ++i )
        {
            const auto expected = parameterOnHost( *source, source_parameters[ i ] );
            const auto actual = parameterOnHost( *target, target_parameters[ i ] );

            ASSERT_EQ( expected.size(), actual.size() ) << "parameter " << i;

            for ( dim_t element = 0; element < expected.size(); ++element )
            {
                ASSERT_EQ( expected.data()[ element ], actual.data()[ element ] )
                    << "parameter " << i << " element " << element;
            }
        }

        std::filesystem::remove( path, ec );
    }

    TEST_F( GptTransformerCudaTests, SaveThenLoad_RecoversTheConfigThatWroteIt )
    {
        const auto path = makeTempArchivePath( "config" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto source = builtNet( batch_, seq_, RuntimeMode::Inference );

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            source->save( archive, SerializationMode::Checkpoint );
        }

        ModelArchive reader( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );
        const GptConfig recovered = GptCuda::configFromArchive( reader );
        const GptConfig expected = smallConfig();

        EXPECT_EQ( recovered.getVocabSize(), expected.getVocabSize() );
        EXPECT_EQ( recovered.getNumLayers(), expected.getNumLayers() );
        EXPECT_EQ( recovered.getEmbeddingSize(), expected.getEmbeddingSize() );
        EXPECT_EQ( recovered.getNumHeads(), expected.getNumHeads() );
        EXPECT_EQ( recovered.getMaxSequenceLength(), expected.getMaxSequenceLength() );
        EXPECT_EQ( recovered.getHiddenSize(), expected.getHiddenSize() );
        EXPECT_EQ( recovered.getUseBias(), expected.getUseBias() );

        EXPECT_EQ( GptCuda::buildSequenceLengthFromArchive( reader ), seq_ );

        std::filesystem::remove( path, ec );
    }

    TEST_F( GptTransformerCudaTests, Load_ThrowsWhenTheArchiveDescribesADifferentNetwork )
    {
        const auto path = makeTempArchivePath( "mismatch" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto source = builtNet( batch_, seq_, RuntimeMode::Inference );

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            source->save( archive, SerializationMode::Checkpoint );
        }

        GptConfig wider = smallConfig();
        wider.withVocabSize( kVocab * 2 );

        auto target = std::make_unique<GptCuda>( "gpt", wider, Device::Cuda( 0 ) );
        target->build( BuildContext( shape_t{ batch_, seq_ }, RuntimeMode::Inference ) );

        ModelArchive reader( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );

        EXPECT_THROW( target->load( reader, SerializationMode::Checkpoint ), std::runtime_error );

        std::filesystem::remove( path, ec );
    }
}
