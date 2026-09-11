/**
 * @file GptTransformer.Cuda.cpp
 * @brief Concrete-network tests for GptTransformer<DeviceType::Cuda, FP32>.
 *
 * The CUDA mirror of GptTransformer.Cpu.cpp. GptTransformer is the GPT-2
 * decoder-only language network:
 *   Lpe (token + positional embedding) -> N x GptBlock -> final LayerNorm -> lm_head
 *
 * Covers the GptTransformer DELTA over the LanguageModelNetwork base: construction/
 * validation, the rank-2 [B, T] build contract, the RuntimeMode/TrainingMode axis,
 * forward producing finite logits of shape [B, T, vocab], the component graph, and
 * backward. Unlike the Llama CUDA path, the GPT-2 CUDA forward/backward kernels are
 * exercised by the Bard training sample, so finiteness and training-mode backward
 * are asserted here (not just the contract).
 *
 * CUDA device tests -- skipped when no CUDA device is present.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::Transformers::Gpt
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

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
}
