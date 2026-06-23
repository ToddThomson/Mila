/**
 * @file GptTransformer.Cpu.cpp
 * @brief Concrete-network tests for GptTransformer<DeviceType::Cpu, FP32>.
 *
 * GptTransformer is the GPT-2 decoder-only language network:
 *   Lpe (token + positional embedding) -> N x GptBlock -> final LayerNorm -> lm_head (Linear)
 *
 * This tests the GptTransformer DELTA over the LanguageNetwork base: construction/
 * validation, the rank-2 [B, T] build contract, the RuntimeMode/TrainingMode axis,
 * forward producing finite logits of shape [B, T, vocab], and the backward
 * preconditions. The constituent components (Lpe, GptBlock, LayerNorm, Linear) are
 * covered in their own suites.
 *
 * NOTE: a training-convergence oracle (loss strictly decreases over a small step
 * budget) is the net-new keystone for this network, but it depends on a correct
 * composed backward. The GptBlock finite-difference gradient check currently fails
 * (BACKLOG: MHA CPU backward suspect), so the convergence oracle is deferred until
 * that is triaged -- it would red for the same reason. Backward here is exercised
 * only for its preconditions (it runs shape-correctly regardless).
 *
 * CPU device, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Transformers::Gpt
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using GptCpu = Mila::Dnn::GptTransformer<DeviceType::Cpu, TensorDataType::FP32>;
        using TokenTensor = Tensor<TensorDataType::INT32, CpuMemoryResource>;
        using LogitsTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;

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

        // Deterministic in-range token ids.
        void fillTokens( TokenTensor& t )
        {
            auto* data = t.data();

            for ( size_t i = 0; i < t.size(); ++i )
            {
                data[ i ] = static_cast<int32_t>( i % static_cast<size_t>( kVocab ) );
            }
        }
    }

    class GptTransformerCpuTests : public ::testing::Test
    {
    protected:
        std::unique_ptr<GptCpu> builtNet( int64_t batch, int64_t seq, RuntimeMode mode )
        {
            auto net = std::make_unique<GptCpu>( "gpt", smallConfig(), Device::Cpu() );
            net->build( BuildContext( shape_t{ batch, seq }, mode ) );

            return net;
        }

        static constexpr int64_t batch_ = 1;
        static constexpr int64_t seq_ = 4;
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( GptTransformerCpuTests, Construct_StandaloneSucceeds )
    {
        GptCpu net( "gpt", smallConfig(), Device::Cpu() );

        EXPECT_EQ( net.getName(), "gpt" );
        EXPECT_EQ( net.getDeviceId().type, DeviceType::Cpu );
    }

    // Sentinel: GptTransformer's ctor creates the ExecutionContext in its
    // initializer list BEFORE the device-type-mismatch check in the body (unlike
    // MLP/GptBlock, which validate the device first). On the CUDA dev build this
    // still throws std::invalid_argument as expected; under MILA_ENABLE_CUDA=OFF it
    // may surface the ctor-ordering wart (a CUDA context construction precedes the
    // type check). Kept as a sentinel -- the fix is to check the device type before
    // creating the context. See BACKLOG.
    TEST_F( GptTransformerCpuTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW( GptCpu( "gpt", smallConfig(), Device::Cuda( 0 ) ), std::invalid_argument );
    }

    TEST_F( GptTransformerCpuTests, Construct_InvalidConfigThrows )
    {
        // embedding_size 0 fails GptConfig::validate(), invoked in the ctor.
        GptConfig bad( 0, kLayers );

        EXPECT_THROW( GptCpu( "gpt", bad, Device::Cpu() ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle
    // ====================================================================

    TEST_F( GptTransformerCpuTests, Build_SetsIsBuilt )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        EXPECT_TRUE( net->isBuilt() );
    }

    TEST_F( GptTransformerCpuTests, Build_ThrowsOnNonRank2Input )
    {
        GptCpu net( "gpt", smallConfig(), Device::Cpu() );

        // Rank-3 input violates the [B, T] token-index contract.
        EXPECT_THROW( net.build( BuildContext( shape_t{ batch_, seq_, kEmbedding }, RuntimeMode::Inference ) ),
            std::invalid_argument );
    }

    TEST_F( GptTransformerCpuTests, Forward_ThrowsBeforeBuild )
    {
        GptCpu net( "gpt", smallConfig(), Device::Cpu() );
        TokenTensor input( Device::Cpu(), shape_t{ batch_, seq_ } );
        fillTokens( input );

        EXPECT_THROW( net.forward( input ), std::runtime_error );
    }

    TEST_F( GptTransformerCpuTests, Backward_ThrowsBeforeBuild )
    {
        GptCpu net( "gpt", smallConfig(), Device::Cpu() );
        TokenTensor input( Device::Cpu(), shape_t{ batch_, seq_ } );
        LogitsTensor output_grad( Device::Cpu(), shape_t{ batch_, seq_, kVocab } );
        fillTokens( input );

        EXPECT_THROW( net.backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // C. Execution Context / Device
    // ====================================================================

    TEST_F( GptTransformerCpuTests, Synchronize_Succeeds )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        EXPECT_NO_THROW( net->synchronize() );
    }

    // ====================================================================
    // D. Runtime + Training Mode
    // ====================================================================

    TEST_F( GptTransformerCpuTests, BuiltForTraining_IsTrainingMode )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Training );

        EXPECT_TRUE( net->isTrainingMode() );
        EXPECT_FALSE( net->isInferenceMode() );
    }

    TEST_F( GptTransformerCpuTests, BuiltForInference_IsInferenceMode )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        EXPECT_TRUE( net->isInferenceMode() );
        EXPECT_FALSE( net->isTrainingMode() );
    }

    TEST_F( GptTransformerCpuTests, Backward_ThrowsWhenBuiltForInference )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        TokenTensor input( Device::Cpu(), shape_t{ batch_, seq_ } );
        LogitsTensor output_grad( Device::Cpu(), shape_t{ batch_, seq_, kVocab } );
        fillTokens( input );

        net->forward( input );

        EXPECT_THROW( net->backward( input, output_grad ), std::runtime_error );
    }

    TEST_F( GptTransformerCpuTests, Backward_ThrowsWhenForwardNotCalled )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Training );

        TokenTensor input( Device::Cpu(), shape_t{ batch_, seq_ } );
        LogitsTensor output_grad( Device::Cpu(), shape_t{ batch_, seq_, kVocab } );
        fillTokens( input );

        EXPECT_THROW( net->backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // E. Forward (logits shape + finiteness)
    // ====================================================================

    TEST_F( GptTransformerCpuTests, Forward_ProducesLogitsShape )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Training );

        TokenTensor input( Device::Cpu(), shape_t{ batch_, seq_ } );
        fillTokens( input );

        auto& logits = net->forward( input );

        EXPECT_EQ( logits.shape(), ( shape_t{ batch_, seq_, kVocab } ) );
    }

    TEST_F( GptTransformerCpuTests, Forward_ProducesFiniteLogits )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Training );

        TokenTensor input( Device::Cpu(), shape_t{ batch_, seq_ } );
        fillTokens( input );

        auto& logits = net->forward( input );

        const auto* data = logits.data();
        for ( size_t i = 0; i < logits.size(); ++i )
        {
            ASSERT_TRUE( std::isfinite( data[ i ] ) ) << "non-finite logit at index " << i;
        }
    }

    // ====================================================================
    // F. Backward (preconditions; runs shape-correctly)
    // ====================================================================

    TEST_F( GptTransformerCpuTests, Backward_RunsAfterForwardInTrainingMode )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Training );

        TokenTensor input( Device::Cpu(), shape_t{ batch_, seq_ } );
        LogitsTensor output_grad( Device::Cpu(), shape_t{ batch_, seq_, kVocab } );
        fillTokens( input );
        for ( size_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = 0.01f * static_cast<float>( i + 1 );
        }

        net->forward( input );

        EXPECT_NO_THROW( net->backward( input, output_grad ) );
    }

    // ====================================================================
    // G. Parameters & Components
    // ====================================================================

    TEST_F( GptTransformerCpuTests, GetComponents_ReturnsLayersPlusThree )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        // encoder + kLayers blocks + final_layernorm + lm_head.
        EXPECT_EQ( net->getComponents().size(), static_cast<size_t>( kLayers + 3 ) );
    }

    TEST_F( GptTransformerCpuTests, ParameterCount_Positive )
    {
        auto net = builtNet( batch_, seq_, RuntimeMode::Inference );

        EXPECT_GT( net->parameterCount(), 0u );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( GptTransformerCpuTests, ToString_DescribesNetwork )
    {
        GptCpu net( "my_gpt", smallConfig(), Device::Cpu() );

        const std::string text = net.toString();

        EXPECT_NE( text.find( "Gpt" ), std::string::npos );
        EXPECT_NE( text.find( "my_gpt" ), std::string::npos );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TEST_F( GptTransformerCpuTests, GetType_IsGpt2 )
    {
        GptCpu net( "gpt", smallConfig(), Device::Cpu() );

        // Structural kind is Network; the GPT-2 architecture identity is ModelType.
        EXPECT_EQ( net.getType(), ComponentType::Network );
        EXPECT_EQ( net.getModelType(), ModelType::Gpt2 );
    }
}
