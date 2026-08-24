/**
 * @file Llama.Cuda.cpp
 * @brief Concrete-network tests for LlamaTransformer<DeviceType::Cuda, FP32>.
 *
 * LlamaTransformer is the LLaMA decoder-only language network:
 *   TokenEmbedding -> RoPE -> N x LlamaBlock -> final RmsNorm -> lm_head (Linear)
 *
 * This tests the LlamaTransformer DELTA over the LanguageModelNetwork base: construction/
 * validation, the rank-2 [B, T] build contract, the RuntimeMode/TrainingMode axis,
 * forward producing logits of shape [B, T, vocab], the component graph, and the
 * backward preconditions. The constituent components (TokenEmbedding, RoPE, GQA,
 * RmsNorm, SwiGLU, Linear) are covered in their own suites.
 *
 * NOTE: numerics are intentionally NOT asserted here. The GQA standalone forward()
 * path is a no-op stub (BACKLOG: GroupedQueryAttention::forward), so the validated
 * compute path is prefill/decode driven by the network with the shared GqaState
 * workspace. This suite therefore covers the network's public contract and logits
 * shape only -- a value/convergence oracle belongs with the loss-on-device work.
 *
 * fromPretrained is NOT tested here: LlamaTransformer::fromPretrained is retired
 * (commented out in Src); LlamaModel::fromPretrained is the supported load path.
 *
 * CUDA device tests -- skipped when no CUDA device is present.
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

import Mila;

namespace Mila::Tests::Dnn::Components::Transformers::Llama
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    namespace
    {
        using LlamaCuda = Mila::Dnn::LlamaTransformer<DeviceType::Cuda, TensorDataType::FP32>;
        using TokenTensor = Tensor<TensorDataType::INT32, CpuMemoryResource>;

        constexpr int64_t kEmbedding = 64;
        constexpr int64_t kLayers = 2;
        constexpr int64_t kHeads = 4;
        constexpr int64_t kKVHeads = 2;
        constexpr int64_t kVocab = 128;
        constexpr int64_t kMaxSeq = 32;
        constexpr int64_t kHidden = 128;

        LlamaConfig smallConfig()
        {
            return LlamaConfig( kEmbedding, kLayers )
                .withVocabularyLength( kVocab )
                .withNumHeads( kHeads )
                .withNumKVHeads( kKVHeads )
                .withHiddenDimension( kHidden )
                .withMaxSequenceLength( kMaxSeq )
                .withRoPETheta( 10000.0f )
                .withRoPEScalingFactor( 1.0f )
                .withBias( false );
        }
    }

    class LlamaTransformerCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }
        }

        std::unique_ptr<LlamaCuda> builtNet( int64_t batch, int64_t seq, RuntimeMode mode )
        {
            auto net = std::make_unique<LlamaCuda>( "llama", smallConfig(), Device::Cuda( 0 ) );
            net->build( BuildContext( shape_t{ batch, seq }, mode ) );

            return net;
        }

        // Deterministic in-range token ids on the device.
        LlamaCuda::TokenIndexType makeTokens( int64_t batch, int64_t seq )
        {
            TokenTensor host( Device::Cpu(), shape_t{ batch, seq } );
            auto* data = host.data();

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                data[ i ] = static_cast<int32_t>( i % static_cast<size_t>( kVocab ) );
            }

            LlamaCuda::TokenIndexType device_tokens( Device::Cuda( 0 ), shape_t{ batch, seq } );
            copy( host, device_tokens );

            return device_tokens;
        }

        static constexpr int64_t batch_ = 1;
        static constexpr int64_t seq_ = 4;
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( LlamaTransformerCudaTests, Construct_StandaloneSucceeds )
    {
        LlamaCuda net( "llama", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( net.getName(), "llama" );
        EXPECT_EQ( net.getDeviceId().type, DeviceType::Cuda );
    }

    TEST_F( LlamaTransformerCudaTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW( LlamaCuda( "llama", smallConfig(), Device::Cpu() ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle
    // ====================================================================

    TEST_F( LlamaTransformerCudaTests, Build_SetsIsBuilt )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        EXPECT_TRUE( net->isBuilt() );
    }

    TEST_F( LlamaTransformerCudaTests, Build_AllocatesParameters )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        EXPECT_GT( net->parameterCount(), 0 );
    }

    TEST_F( LlamaTransformerCudaTests, Build_ThrowsOnNonRank2Input )
    {
        LlamaCuda net( "llama", smallConfig(), Device::Cuda( 0 ) );

        // Rank-3 input violates the [B, T] token-index contract.
        EXPECT_THROW( net.build( BuildContext( shape_t{ batch_, seq_, kEmbedding }, RuntimeMode::Inference ) ),
            std::invalid_argument );
    }

    TEST_F( LlamaTransformerCudaTests, Forward_ThrowsBeforeBuild )
    {
        LlamaCuda net( "llama", smallConfig(), Device::Cuda( 0 ) );
        auto input = makeTokens( batch_, seq_ );

        EXPECT_THROW( net.forward( input ), std::runtime_error );
    }

    TEST_F( LlamaTransformerCudaTests, Backward_ThrowsBeforeBuild )
    {
        LlamaCuda net( "llama", smallConfig(), Device::Cuda( 0 ) );
        auto input = makeTokens( batch_, seq_ );
        LlamaCuda::TensorType output_grad( Device::Cuda( 0 ), shape_t{ batch_, seq_, kVocab } );

        EXPECT_THROW( net.backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // C. Execution Context / Device
    // ====================================================================

    TEST_F( LlamaTransformerCudaTests, Synchronize_Succeeds )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        EXPECT_NO_THROW( net->synchronize() );
    }

    // ====================================================================
    // D. Runtime + Training Mode
    // ====================================================================

    TEST_F( LlamaTransformerCudaTests, BuiltForTraining_IsTrainingMode )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Training );

        EXPECT_TRUE( net->isTrainingMode() );
        EXPECT_FALSE( net->isInferenceMode() );
    }

    TEST_F( LlamaTransformerCudaTests, BuiltForInference_IsInferenceMode )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        EXPECT_TRUE( net->isInferenceMode() );
        EXPECT_FALSE( net->isTrainingMode() );
    }

    TEST_F( LlamaTransformerCudaTests, Backward_ThrowsWhenBuiltForInference )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        auto input = makeTokens( batch_, seq_ );
        LlamaCuda::TensorType output_grad( Device::Cuda( 0 ), shape_t{ batch_, seq_, kVocab } );

        EXPECT_THROW( net->backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // E. Forward (logits shape; numerics deferred -- see file note)
    // ====================================================================

    TEST_F( LlamaTransformerCudaTests, Forward_ProducesLogitsShape )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Training );

        auto input = makeTokens( batch_, seq_ );
        auto& logits = net->forward( input );

        EXPECT_EQ( logits.shape(), ( shape_t{ batch_, seq_, kVocab } ) );
    }

    // ====================================================================
    // G. Parameters & Components
    // ====================================================================

    TEST_F( LlamaTransformerCudaTests, GetComponents_ReturnsLayersPlusThree )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        // token_embedding + kLayers blocks + final_rmsnorm + lm_head.
        EXPECT_EQ( net->getComponents().size(), static_cast<size_t>( kLayers + 3 ) );
    }

    TEST_F( LlamaTransformerCudaTests, ParameterCount_Positive )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        EXPECT_GT( net->parameterCount(), 0 );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( LlamaTransformerCudaTests, ToString_ContainsExpectedFields )
    {
        LlamaCuda net( "my_llama", smallConfig(), Device::Cuda( 0 ) );

        const std::string text = net.toString();

        EXPECT_NE( text.find( "Llama Network" ), std::string::npos );
        EXPECT_NE( text.find( "my_llama" ), std::string::npos );
        EXPECT_NE( text.find( "Number of KV heads" ), std::string::npos );
        EXPECT_NE( text.find( "RoPE theta" ), std::string::npos );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TEST_F( LlamaTransformerCudaTests, GetType_IsLlama )
    {
        LlamaCuda net( "llama", smallConfig(), Device::Cuda( 0 ) );

        // Structural kind is Network; the Llama architecture identity is ModelType.
        EXPECT_EQ( net.getType(), ComponentType::Network );
        EXPECT_EQ( net.getModelType(), ModelType::Llama );
    }

    // ====================================================================
    // K. Serialization round trip
    // ====================================================================
    //
    // LlamaBlock had a hand-rolled save_ that pushed no scope per child, so every
    // child of every block wrote its tensors to one path; it also listed ten children
    // where eleven are registered, so one was never written at all. Both are invisible
    // to a mock-composite test -- only a real network shows them. This is also the
    // only coverage of RmsNorm's and TokenEmbedding's getParameterNames(), which the
    // GPT stack does not use.

    namespace
    {
        std::filesystem::path makeTempArchivePath( const std::string& tag )
        {
            const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();

            return std::filesystem::temp_directory_path()
                / std::format( "mila_test_llama_{}_{}.mila", tag, stamp );
        }
    }

    TEST_F( LlamaTransformerCudaTests, SaveThenLoad_RestoresEveryParameterAcrossBlocks )
    {
        const auto path = makeTempArchivePath( "roundtrip" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto source = builtNet( batch_, seq_, RuntimeMode::Inference );

        // Distinct value per tensor so a collision or cross-wired restore names itself.
        float value = 0.5f;

        for ( auto* parameter : source->getParameters() )
        {
            fill( *static_cast<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>*>( parameter ), value );
            value += 0.25f;
        }

        const auto parameter_count = source->getParameters().size();
        ASSERT_GT( parameter_count, 8u ) << "fixture too small to exercise nesting";

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            source->save( archive, SerializationMode::Checkpoint );
        }

        ModelArchive reader( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );

        // One blob per parameter. The unscoped hand-rolled walk produced a single blob
        // regardless of how many parameters the network held.
        const auto files = reader.listFiles();
        const auto blob_count = std::count_if( files.begin(), files.end(),
            []( const std::string& name )
            {
                return name.ends_with( "/data.bin" );
            } );

        EXPECT_EQ( static_cast<size_t>( blob_count ), parameter_count );

        auto target = builtNet( batch_, seq_, RuntimeMode::Inference );

        for ( auto* parameter : target->getParameters() )
        {
            fill( *static_cast<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>*>( parameter ), -1.0f );
        }

        target->load( reader, SerializationMode::Checkpoint );

        const auto source_parameters = source->getParameters();
        const auto target_parameters = target->getParameters();

        ASSERT_EQ( source_parameters.size(), target_parameters.size() );

        // Device tensors: compare through a host copy of each pair.
        for ( size_t i = 0; i < source_parameters.size(); ++i )
        {
            using DeviceTensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;
            using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;

            const auto* expected = static_cast<const DeviceTensor*>( source_parameters[ i ] );
            const auto* actual = static_cast<const DeviceTensor*>( target_parameters[ i ] );

            ASSERT_EQ( expected->size(), actual->size() ) << "parameter " << i;

            HostTensor expected_host( Device::Cpu(), expected->shape() );
            HostTensor actual_host( Device::Cpu(), actual->shape() );

            copy( *expected, expected_host );
            copy( *actual, actual_host );

            for ( dim_t element = 0; element < expected->size(); ++element )
            {
                ASSERT_EQ( expected_host.data()[ element ], actual_host.data()[ element ] )
                    << "parameter " << i << " element " << element;
            }
        }

        std::filesystem::remove( path, ec );
    }
}
