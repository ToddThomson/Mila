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

import Mila;

namespace Mila::Tests::Dnn::Components::Transformers::Gpt
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

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

        EXPECT_GT( net->parameterCount(), 0 );
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

    // ====================================================================
    // K. Serialization round trip
    // ====================================================================
    //
    // The coverage the doubles in Core/CompositeComponent.cpp cannot provide. That
    // suite proves CompositeComponent's scoped traversal works; this proves the real
    // composites USE it. They did not: GptBlock and MLP each overrode save_ with an
    // unscoped hand-rolled walk, so every child of every block wrote its tensors to
    // one path and overwrote the last. A mock composite that does not override save_
    // is structurally incapable of catching that -- only a real network is.

    namespace
    {
        std::filesystem::path makeTempArchivePath( const std::string& tag )
        {
            const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();

            return std::filesystem::temp_directory_path()
                / std::format( "mila_test_gpt_{}_{}.mila", tag, stamp );
        }

        void fillParametersDistinctly( GptCpu& net, float first, float step )
        {
            float value = first;

            for ( auto* parameter : net.getParameters() )
            {
                fill( *static_cast<LogitsTensor*>( parameter ), value );
                value += step;
            }
        }
    }

    TEST_F( GptTransformerCpuTests, SaveThenLoad_RestoresEveryParameterAcrossNestedBlocks )
    {
        const auto path = makeTempArchivePath( "roundtrip" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto source = builtNet( batch_, seq_, RuntimeMode::Inference );

        // A distinct value per tensor: with two blocks, each holding an attention, two
        // LayerNorms, two projections and an MLP, a collision or a cross-wired restore
        // shows up as a specific parameter holding another's value rather than as a
        // uniform pass.
        fillParametersDistinctly( *source, 0.5f, 0.25f );

        const auto parameter_count = source->getParameters().size();
        ASSERT_GT( parameter_count, 8u ) << "fixture too small to exercise nesting";

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            source->save( archive, SerializationMode::Checkpoint );
        }

        ModelArchive reader( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );

        // One blob per parameter. This is the assertion the unscoped walk fails outright:
        // it produced a single blob no matter how many parameters the network held.
        const auto files = reader.listFiles();
        const auto blob_count = std::count_if( files.begin(), files.end(),
            []( const std::string& name )
            {
                return name.ends_with( "/data.bin" );
            } );

        EXPECT_EQ( static_cast<size_t>( blob_count ), parameter_count );

        // A separately built network, poisoned so "restored" cannot be confused with
        // "happened to already match".
        auto target = builtNet( batch_, seq_, RuntimeMode::Inference );
        fillParametersDistinctly( *target, -1.0f, 0.0f );

        target->load( reader, SerializationMode::Checkpoint );

        const auto source_parameters = source->getParameters();
        const auto target_parameters = target->getParameters();

        ASSERT_EQ( source_parameters.size(), target_parameters.size() );

        for ( size_t i = 0; i < source_parameters.size(); ++i )
        {
            const auto* expected = static_cast<const LogitsTensor*>( source_parameters[ i ] );
            const auto* actual = static_cast<const LogitsTensor*>( target_parameters[ i ] );

            ASSERT_EQ( expected->size(), actual->size() ) << "parameter " << i;

            const float* expected_data = static_cast<const float*>( expected->rawData() );
            const float* actual_data = static_cast<const float*>( actual->rawData() );

            for ( dim_t element = 0; element < expected->size(); ++element )
            {
                ASSERT_EQ( expected_data[ element ], actual_data[ element ] )
                    << "parameter " << i << " element " << element;
            }
        }

        std::filesystem::remove( path, ec );
    }

    TEST_F( GptTransformerCpuTests, SaveThenLoad_RecoversTheConfigThatWroteIt )
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
        const GptConfig recovered = GptCpu::configFromArchive( reader );
        const GptConfig expected = smallConfig();

        EXPECT_EQ( recovered.getVocabSize(), expected.getVocabSize() );
        EXPECT_EQ( recovered.getNumLayers(), expected.getNumLayers() );
        EXPECT_EQ( recovered.getEmbeddingSize(), expected.getEmbeddingSize() );
        EXPECT_EQ( recovered.getNumHeads(), expected.getNumHeads() );
        EXPECT_EQ( recovered.getMaxSequenceLength(), expected.getMaxSequenceLength() );

        // The two fields the hand-rolled metadata block got wrong: hidden size was written
        // under a key fromMetadata does not read (so it fell back to 4x embedding, right
        // for GPT-2 by coincidence), and use_bias was never written at all.
        EXPECT_EQ( recovered.getHiddenSize(), expected.getHiddenSize() );
        EXPECT_EQ( recovered.getUseBias(), expected.getUseBias() );

        EXPECT_EQ( GptCpu::buildSequenceLengthFromArchive( reader ), seq_ );

        std::filesystem::remove( path, ec );
    }

    TEST_F( GptTransformerCpuTests, Load_ThrowsWhenTheArchiveDescribesADifferentNetwork )
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

        auto target = std::make_unique<GptCpu>( "gpt", wider, Device::Cpu() );
        target->build( BuildContext( shape_t{ batch_, seq_ }, RuntimeMode::Inference ) );

        ModelArchive reader( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );

        EXPECT_THROW( target->load( reader, SerializationMode::Checkpoint ), std::runtime_error );

        std::filesystem::remove( path, ec );
    }
}
