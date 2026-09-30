/**
 * @file Gemma.MixtureOfExperts.Cuda.cpp
 * @brief GemmaTransformer with a routed feed-forward against a tiny HuggingFace Gemma 4 MoE model.
 *
 * The converted weights and the reference logits come from
 * Mila/Tools/Converters/Gemma/gemma_4_26b_moe/hf_gemma_moe_model_reference.py, which converts its own
 * checkpoint with Gemma/convert_weights.py -- so one capture holds the converter's names and the block's
 * wiring together. Tolerances are the ones Gemma4MoE.md Phase 8 fixed before the first run; the gates
 * that read the capture skip without it. The sequence log-likelihood gates are ModelFamilyParity.md 8.2, G1.
 *
 * CUDA device tests -- skipped when no CUDA device is present.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

import Mila;

#include "Common/LogLikelihoodHarness.h"

namespace Mila::Tests::Dnn::Components::Transformers::Gemma
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        namespace fs = std::filesystem;

        using NoKvCompression = Mila::Dnn::Quant::KvCache::NoKvCompression;
        using NoWeightQuant = Mila::Dnn::Quant::Weight::NoWeightQuant;
        using HostTokens = Tensor<TensorDataType::INT32, CpuMemoryResource>;
        using DeviceTokens = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>;

        template<TensorDataType TPrecision>
        using RoutedNetwork = GemmaTransformer<DeviceType::Cuda, TPrecision, NoWeightQuant, NoKvCompression, GemmaFeedForward::Routed>;

        template<TensorDataType TPrecision>
        using DenseNetwork = GemmaTransformer<DeviceType::Cuda, TPrecision>;

        constexpr int64_t kBatch = 1;
        constexpr int64_t kContext = 16;
        constexpr int64_t kModelDim = 128;
        constexpr int64_t kLayers = 2;
        constexpr int64_t kExperts = 8;
        constexpr int64_t kTopK = 2;
        constexpr int64_t kExpertHidden = 64;

        // Gemma4MoE.md Phase 8, "Wiring gate". The BF16 figure replaced the original 1e-2 after
        // HuggingFace's own bfloat16 run measured 1.4e-2 to 4.1e-2 from its float32 run on this model.
        constexpr double kFp32LogitTolerance = 1e-4;
        constexpr double kBf16RelativeL2 = 1e-1;

        fs::path captureDirectory()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_moe_tiny";
        }

        fs::path weightsPath( TensorDataType precision )
        {
            return captureDirectory()
                / ( precision == TensorDataType::BF16 ? "gemma4_moe_tiny_bf16.bin" : "gemma4_moe_tiny_fp32.bin" );
        }

        fs::path referencePath()
        {
            return captureDirectory() / "gemma4_moe_tiny_reference.safetensors";
        }

        bool captureExists()
        {
            return fs::exists( referencePath() )
                && fs::exists( weightsPath( TensorDataType::FP32 ) )
                && fs::exists( weightsPath( TensorDataType::BF16 ) );
        }

        // The capture's geometry, for the gates that need no capture.
        GemmaConfig routedConfig()
        {
            return GemmaConfig( kModelDim, kLayers )
                .withVocabularyLength( 128 )
                .withNumHeads( 4 )
                .withNumKVHeads( 2 )
                .withHeadDim( 64 )
                .withGlobalHeadDim( 128 )
                .withNumGlobalKVHeads( 1 )
                .withKeyEqualsValue( true )
                .withHiddenDimension( 128 )
                .withMaxSequenceLength( 32 )
                .withRMSNormEpsilon( 1e-6f )
                .withWindow( 8 )
                .withSlidingWindowPattern( 2 )
                .withGlobalRotaryDim( 32 )
                .withRoPETheta( 10000.0f )
                .withGlobalRoPETheta( 250000.0f )
                .withFinalLogitSoftcapping( 30.0f )
                .withMixtureOfExperts( kExperts, kTopK, kExpertHidden );
        }

        // The mapping GemmaModel applies to the same metadata; the network is not reachable through it.
        GemmaConfig configFromMetadata( const Serialization::WeightsMetadata& metadata )
        {
            GemmaConfig config( static_cast<dim_t>( metadata.embedding_dim ), static_cast<dim_t>( metadata.num_layers ) );

            config.withVocabularyLength( static_cast<dim_t>( metadata.vocab_size ) )
                .withMaxSequenceLength( static_cast<dim_t>( metadata.max_seq_length ) )
                .withNumHeads( static_cast<dim_t>( metadata.num_heads ) )
                .withNumKVHeads( static_cast<dim_t>( metadata.num_kv_heads ) )
                .withHeadDim( static_cast<dim_t>( metadata.head_dim ) )
                .withGlobalHeadDim( static_cast<dim_t>( metadata.global_head_dim ) )
                .withNumGlobalKVHeads( static_cast<dim_t>( metadata.num_global_kv_heads ) )
                .withKeyEqualsValue( metadata.key_equals_value )
                .withHiddenDimension( static_cast<dim_t>( metadata.hidden_dim ) )
                .withRMSNormEpsilon( metadata.norm_epsilon )
                .withWindow( static_cast<dim_t>( metadata.window ) )
                .withSlidingWindowPattern( static_cast<dim_t>( metadata.sliding_window_pattern ) )
                .withGlobalRotaryDim( static_cast<dim_t>( metadata.global_rotary_dim ) )
                .withRoPETheta( metadata.rope_theta_local )
                .withGlobalRoPETheta( metadata.rope_theta_global )
                .withFinalLogitSoftcapping( metadata.final_logit_softcapping )
                .withTieWordEmbeddings( metadata.tie_word_embeddings )
                .withMixtureOfExperts( static_cast<dim_t>( metadata.num_experts ),
                    static_cast<dim_t>( metadata.top_k_experts ), static_cast<dim_t>( metadata.expert_hidden_dim ) );

            return config;
        }

        DeviceTokens deviceTokens( const std::vector<std::int32_t>& ids )
        {
            HostTokens host( Device::Cpu(), shape_t{ kBatch, static_cast<int64_t>( ids.size() ) } );
            std::copy( ids.begin(), ids.end(), host.data() );

            DeviceTokens device( Device::Cuda( 0 ), shape_t{ kBatch, static_cast<int64_t>( ids.size() ) } );
            copy( host, device );

            return device;
        }

        // Half the context, so the 15-token capture crosses a chunk boundary.
        constexpr int64_t kLogLikelihoodChunk = 8;

        // The FP32 logit tolerance, summed over the capture's 14 scored positions.
        constexpr double kFp32LogLikelihoodTolerance = 1e-4;

        struct LogLikelihoodReference
        {
            std::vector<std::int32_t> tokens;
            double total{ 0.0 };
            dim_t positions{ 0 };
        };

        LogLikelihoodReference readLogLikelihoodReference()
        {
            Serialization::WeightsReader reference( referencePath() );
            auto token_blob = reference.readTensorBlob<CpuMemoryResource>( "tokens" );
            auto log_probability_blob = reference.readTensorBlob<CpuMemoryResource>( "next_token_log_probabilities" );

            LogLikelihoodReference result;
            result.tokens.resize( static_cast<std::size_t>( token_blob.getMetadata().shape[ 0 ] ) );
            std::memcpy( result.tokens.data(), token_blob.data(), result.tokens.size() * sizeof( std::int32_t ) );

            std::vector<float> per_position( static_cast<std::size_t>( log_probability_blob.getMetadata().shape[ 0 ] ) );
            std::memcpy( per_position.data(), log_probability_blob.data(), per_position.size() * sizeof( float ) );

            for ( const float log_probability : per_position )
            {
                result.total += log_probability;
            }

            result.positions = static_cast<dim_t>( per_position.size() );

            return result;
        }

        std::unique_ptr<RoutedNetwork<TensorDataType::FP32>> loadedRoutedFp32( const GemmaConfig& config )
        {
            Serialization::WeightsReader weights( weightsPath( TensorDataType::FP32 ) );

            auto network = std::make_unique<RoutedNetwork<TensorDataType::FP32>>( "gemma", config, Device::Cuda( 0 ) );
            network->build( BuildContext( shape_t{ kBatch, kContext }, RuntimeMode::Inference ).withPrefillSize( kLogLikelihoodChunk ) );
            network->loadParameters( weights );

            return network;
        }

        GemmaConfig capturedConfigWithWindow( dim_t window )
        {
            Serialization::WeightsReader weights( weightsPath( TensorDataType::FP32 ) );

            GemmaConfig config = configFromMetadata( weights.getWeightsMetadata() );
            config.withLogLikelihoodWindow( window );

            return config;
        }

        void constructRouted( const GemmaConfig& config )
        {
            RoutedNetwork<TensorDataType::BF16> network( "gemma", config, Device::Cuda( 0 ) );
        }

        void constructDense( const GemmaConfig& config )
        {
            DenseNetwork<TensorDataType::BF16> network( "gemma", config, Device::Cuda( 0 ) );
        }
    }

    class GemmaMixtureOfExpertsCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            context_ = createExecutionContext( Device::Cuda( 0 ) );
        }

        template<TensorDataType TPrecision>
        void expectLogitsMatchReference()
        {
            const TensorDataType precision = TPrecision;

            Serialization::WeightsReader weights( weightsPath( precision ) );
            const GemmaConfig config = configFromMetadata( weights.getWeightsMetadata() );

            ASSERT_TRUE( config.hasMixtureOfExperts() ) << "the converted weights carry no expert geometry";

            RoutedNetwork<TPrecision> network( "gemma", config, Device::Cuda( 0 ) );
            network.build( BuildContext( shape_t{ kBatch, kContext }, RuntimeMode::Inference ).withPrefillSize( kContext ) );
            network.loadParameters( weights );

            Serialization::WeightsReader reference( referencePath() );
            auto token_blob = reference.readTensorBlob<CpuMemoryResource>( "tokens" );
            auto logits_blob = reference.readTensorBlob<CpuMemoryResource>( "logits" );

            const int64_t total = static_cast<int64_t>( token_blob.getMetadata().shape[ 0 ] );
            const int64_t steps = static_cast<int64_t>( logits_blob.getMetadata().shape[ 0 ] );
            const int64_t vocab = static_cast<int64_t>( logits_blob.getMetadata().shape[ 1 ] );
            const int64_t prompt = total - ( steps - 1 );

            ASSERT_LE( total, kContext );

            std::vector<std::int32_t> tokens( static_cast<std::size_t>( total ) );
            std::memcpy( tokens.data(), token_blob.data(), tokens.size() * sizeof( std::int32_t ) );

            std::vector<float> expected( static_cast<std::size_t>( steps * vocab ) );
            ASSERT_EQ( logits_blob.sizeBytes(), expected.size() * sizeof( float ) );
            std::memcpy( expected.data(), logits_blob.data(), logits_blob.sizeBytes() );

            std::vector<float> actual;

            auto append = [&]( const auto& logits )
            {
                network.synchronize();

                auto host = toHost<TensorDataType::FP32>( logits, context_.get() );
                actual.insert( actual.end(), host.data(), host.data() + host.size() );
            };

            append( network.prefill( deviceTokens( std::vector<std::int32_t>( tokens.begin(), tokens.begin() + prompt ) ) ) );

            for ( int64_t step = 1; step < steps; ++step )
            {
                const int64_t position = prompt + step - 1;

                append( network.decode( deviceTokens( { tokens[ static_cast<std::size_t>( position ) ] } ), position ) );
            }

            ASSERT_EQ( actual.size(), expected.size() );

            for ( int64_t step = 0; step < steps; ++step )
            {
                const float* got = actual.data() + step * vocab;
                const float* want = expected.data() + step * vocab;

                int64_t mismatches = 0;
                double worst = 0.0;
                double difference_squares = 0.0;
                double reference_squares = 0.0;

                for ( int64_t column = 0; column < vocab; ++column )
                {
                    const double difference = static_cast<double>( got[ column ] ) - want[ column ];

                    worst = std::max( worst, std::fabs( difference ) );
                    difference_squares += difference * difference;
                    reference_squares += static_cast<double>( want[ column ] ) * want[ column ];

                    if ( std::fabs( difference ) > kFp32LogitTolerance )
                    {
                        ++mismatches;
                    }
                }

                const double relative_l2 = std::sqrt( difference_squares / reference_squares );
                const auto got_argmax = std::max_element( got, got + vocab ) - got;
                const auto want_argmax = std::max_element( want, want + vocab ) - want;

                std::cout << std::format( "[ reference ] {} step {}: worst {:.3e}, relative L2 {:.3e}, argmax {} vs {}\n",
                    precision == TensorDataType::BF16 ? "BF16" : "FP32", step, worst, relative_l2, got_argmax, want_argmax );

                if ( precision == TensorDataType::BF16 )
                {
                    EXPECT_LE( relative_l2, kBf16RelativeL2 ) << "step " << step;
                    EXPECT_EQ( got_argmax, want_argmax ) << "step " << step;
                }
                else
                {
                    EXPECT_EQ( mismatches, 0 ) << "step " << step << " of " << vocab << " logits; worst " << worst;
                }
            }
        }

        std::unique_ptr<IExecutionContext> context_;
    };

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( GemmaMixtureOfExpertsCudaTests, Fp32_LogitsMatchHuggingFace )
    {
        if ( !captureExists() )
        {
            GTEST_SKIP() << "tiny MoE capture not present at: " << captureDirectory().string();
        }

        expectLogitsMatchReference<TensorDataType::FP32>();
    }

    TEST_F( GemmaMixtureOfExpertsCudaTests, Bf16_LogitsMatchHuggingFace )
    {
        if ( !captureExists() )
        {
            GTEST_SKIP() << "tiny MoE capture not present at: " << captureDirectory().string();
        }

        expectLogitsMatchReference<TensorDataType::BF16>();
    }

    // ====================================================================
    // E2. Sequence log-likelihood against HuggingFace's own (ModelFamilyParity.md 8.2, G1)
    //
    // The logits gates above compare the head BEFORE the softcap, because the sampler applies it and
    // argmax is blind to it. A probability is not, so these compare what HuggingFace's loss computes:
    // the log-probability of each next token after the softcap.
    // ====================================================================

    // Window 1 is decode's path, 3 leaves a partial final window in each chunk, and 8 is the whole chunk.
    TEST_F( GemmaMixtureOfExpertsCudaTests, Fp32_SequenceLogLikelihood_MatchesHuggingFace )
    {
        if ( !captureExists() )
        {
            GTEST_SKIP() << "tiny MoE capture not present at: " << captureDirectory().string();
        }

        const LogLikelihoodReference reference = readLogLikelihoodReference();

        for ( const dim_t window : { 1, 3, 8 } )
        {
            auto network = loadedRoutedFp32( capturedConfigWithWindow( window ) );

            const SequenceLogLikelihood measured = Common::sequenceLogLikelihoodOf( *network, reference.tokens );

            std::cout << std::format( "[ log-likelihood ] window {}: {:.9f} against HuggingFace {:.9f}, difference {:.3e}\n",
                window, measured.total_log_probability, reference.total,
                measured.total_log_probability - reference.total );

            EXPECT_EQ( measured.scored_positions, reference.positions ) << "window " << window;
            EXPECT_NEAR( measured.total_log_probability, reference.total, kFp32LogLikelihoodTolerance ) << "window " << window;
        }
    }

    // The gate can fail: without the softcap, the same weights must miss HuggingFace's number. Were the logits
    // too small for the softcap to matter, the gate above would pass whether or not it is applied.
    TEST_F( GemmaMixtureOfExpertsCudaTests, Fp32_SequenceLogLikelihood_WithoutTheSoftcap_MissesHuggingFace )
    {
        if ( !captureExists() )
        {
            GTEST_SKIP() << "tiny MoE capture not present at: " << captureDirectory().string();
        }

        const LogLikelihoodReference reference = readLogLikelihoodReference();

        GemmaConfig uncapped = capturedConfigWithWindow( 3 );
        uncapped.withFinalLogitSoftcapping( 0.0f );

        auto network = loadedRoutedFp32( uncapped );

        const SequenceLogLikelihood measured = Common::sequenceLogLikelihoodOf( *network, reference.tokens );

        std::cout << std::format( "[ log-likelihood ] without the softcap: difference {:.3e}\n",
            measured.total_log_probability - reference.total );

        EXPECT_GT( std::fabs( measured.total_log_probability - reference.total ), kFp32LogLikelihoodTolerance );
    }

    // Each position's most likely next token, read from the rows the log-likelihood evaluated, is the token
    // greedy generation chose there. The rows come from observing the head, which publishes every window.
    TEST_F( GemmaMixtureOfExpertsCudaTests, Fp32_SequenceLogLikelihood_ArgmaxIsTheGreedyToken )
    {
        if ( !captureExists() )
        {
            GTEST_SKIP() << "tiny MoE capture not present at: " << captureDirectory().string();
        }

        constexpr int kGenerated = 3;

        const LogLikelihoodReference reference = readLogLikelihoodReference();
        const std::vector<std::int32_t> prompt( reference.tokens.begin(), reference.tokens.begin() + 12 );

        auto network = loadedRoutedFp32( capturedConfigWithWindow( 3 ) );

        const Common::GreedyContinuation greedy =
            Common::greedyContinuationOf( *network, prompt, kGenerated, {}, kContext );

        ASSERT_EQ( greedy.tokens.size(), static_cast<std::size_t>( kGenerated ) );

        std::vector<std::int32_t> sequence = prompt;
        sequence.insert( sequence.end(), greedy.tokens.begin(), greedy.tokens.end() );

        using DeviceLogits = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;

        std::vector<std::vector<float>> rows;
        const int64_t vocab = static_cast<int64_t>( capturedConfigWithWindow( 1 ).getVocabSize() );

        const std::size_t observed = network->observe( "*.lm_head", ComputePassMask::inference(),
            [&]( std::string_view, ComputePass, std::string_view stage, const ITensor& value )
            {
                const auto* typed = dynamic_cast<const DeviceLogits*>( &value );

                if ( stage != "output" || typed == nullptr )
                {
                    return;
                }

                // Published as soon as the head is enqueued; the rows exist once the stream reaches it.
                network->synchronize();

                auto host = toHost<TensorDataType::FP32>( *typed );

                for ( int64_t row = 0; row * vocab < static_cast<int64_t>( host.size() ); ++row )
                {
                    rows.emplace_back( host.data() + row * vocab, host.data() + ( row + 1 ) * vocab );
                }
            } );

        ASSERT_EQ( observed, 1u ) << "the head was not selected, so no rows will arrive";

        (void)Common::sequenceLogLikelihoodOf( *network, sequence );

        network->stopObserving();

        // A row for every scored position, and for the final position too when it shares a window with one.
        ASSERT_GE( rows.size(), sequence.size() - 1 );

        for ( int generated = 0; generated < kGenerated; ++generated )
        {
            const std::size_t position = prompt.size() - 1 + static_cast<std::size_t>( generated );

            EXPECT_EQ( Common::argMax( rows[ position ] ), greedy.tokens[ static_cast<std::size_t>( generated ) ] )
                << "position " << position;
        }
    }

    // ====================================================================
    // Names -- the converter's vocabulary is the network's
    // ====================================================================

    TEST_F( GemmaMixtureOfExpertsCudaTests, FlatNames_AreTheConvertedFileNames )
    {
        if ( !captureExists() )
        {
            GTEST_SKIP() << "tiny MoE capture not present at: " << captureDirectory().string();
        }

        const fs::path saved = fs::temp_directory_path() / "mila_gemma_moe_names.safetensors";
        std::vector<std::string> expected;

        {
            Serialization::WeightsReader weights( weightsPath( TensorDataType::FP32 ) );
            expected = weights.getTensorNames();

            RoutedNetwork<TensorDataType::FP32> network( "gemma", configFromMetadata( weights.getWeightsMetadata() ), Device::Cuda( 0 ) );
            network.build( BuildContext( shape_t{ kBatch, kContext }, RuntimeMode::Inference ).withPrefillSize( kContext ) );
            network.loadParameters( weights );

            Serialization::SafeTensorsWriter writer( saved );
            network.saveFlatTensors( writer, "", Serialization::TensorSavePass::Declare );
            writer.beginData();
            network.saveFlatTensors( writer, "", Serialization::TensorSavePass::Write );
            writer.close();
        }

        std::vector<std::string> actual;
        {
            Serialization::WeightsReader reader( saved );
            actual = reader.getTensorNames();
        }

        std::error_code ignored;
        fs::remove( saved, ignored );

        std::sort( expected.begin(), expected.end() );
        std::sort( actual.begin(), actual.end() );

        EXPECT_EQ( actual, expected );

        const auto expert_tensors = std::count_if( actual.begin(), actual.end(),
            []( const std::string& name ) { return name.find( ".experts." ) != std::string::npos; } );

        EXPECT_EQ( expert_tensors, 2 * kLayers ) << "gate_up_proj and down_proj per layer";
    }

    // ====================================================================
    // G. Footprint
    // ====================================================================

    // One network alive at a time: the process-wide RoPE cache makes a second one under-report.
    TEST_F( GemmaMixtureOfExpertsCudaTests, Bf16_FootprintPredictedAndInactiveBytesReported )
    {
        const BuildContext context = BuildContext( shape_t{ kBatch, kContext }, RuntimeMode::Inference )
            .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) )
            .withPrefillSize( kContext );

        MemoryStats predicted;
        {
            RoutedNetwork<TensorDataType::BF16> predictor( "gemma", routedConfig(), Device::Cuda( 0 ) );
            predicted = predictor.getRequiredMemory( context );
        }

        MemoryStats built;
        {
            RoutedNetwork<TensorDataType::BF16> network( "gemma", routedConfig(), Device::Cuda( 0 ) );
            network.build( context );
            built = network.getMemoryStats();
        }

        EXPECT_EQ( predicted.device_parameter_bytes, built.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, built.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_gradient_bytes, built.device_gradient_bytes ) << "gradients";
        EXPECT_EQ( predicted.device_inactive_parameter_bytes, built.device_inactive_parameter_bytes ) << "inactive parameters";

        // Each layer's bank is 8 experts x 3 x 64 x 128 at BF16; top-2 leaves six of the eight unread.
        const std::size_t bank_bytes = static_cast<std::size_t>( kExperts * 3 * kExpertHidden * kModelDim ) * 2;

        EXPECT_EQ( built.device_inactive_parameter_bytes,
            static_cast<std::size_t>( kLayers ) * bank_bytes / kExperts * ( kExperts - kTopK ) );
    }

    /**
     * @brief A log-likelihood window is priced exactly as it is built, and costs more than the one-row default.
     *
     * Build and footprint resolve the window through one function, so a measurement build cannot allocate a
     * head the prediction never named. The inequality keeps this from passing for a window both paths ignore.
     */
    TEST_F( GemmaMixtureOfExpertsCudaTests, Bf16_FootprintPredicted_WidenedLogLikelihoodWindow )
    {
        const BuildContext context = BuildContext( shape_t{ kBatch, kContext }, RuntimeMode::Inference )
            .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) )
            .withPrefillSize( kContext );

        GemmaConfig widened = routedConfig();
        widened.withLogLikelihoodWindow( kContext );

        MemoryStats predicted;
        {
            RoutedNetwork<TensorDataType::BF16> predictor( "gemma", widened, Device::Cuda( 0 ) );
            predicted = predictor.getRequiredMemory( context );
        }

        MemoryStats built;
        {
            RoutedNetwork<TensorDataType::BF16> network( "gemma", widened, Device::Cuda( 0 ) );
            network.build( context );
            built = network.getMemoryStats();
        }

        MemoryStats one_row;
        {
            RoutedNetwork<TensorDataType::BF16> network( "gemma", routedConfig(), Device::Cuda( 0 ) );
            network.build( context );
            one_row = network.getMemoryStats();
        }

        EXPECT_EQ( predicted.device_parameter_bytes, built.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, built.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_scratch_bytes, built.device_scratch_bytes ) << "scratch";
        EXPECT_EQ( predicted.device_gradient_bytes, built.device_gradient_bytes ) << "gradients";

        EXPECT_GT( built.device_state_bytes, one_row.device_state_bytes )
            << "a wider head must cost more state than the one-row default";
    }

    // A window above what a prefill pass supplies resolves to the pass: the head reads at most a chunk of rows.
    TEST_F( GemmaMixtureOfExpertsCudaTests, Bf16_LogLikelihoodWindow_ClampsToThePrefillChunk )
    {
        const BuildContext context = BuildContext( shape_t{ kBatch, kContext }, RuntimeMode::Inference )
            .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) )
            .withPrefillSize( kLogLikelihoodChunk );

        GemmaConfig at_chunk = routedConfig();
        at_chunk.withLogLikelihoodWindow( kLogLikelihoodChunk );

        GemmaConfig above_chunk = routedConfig();
        above_chunk.withLogLikelihoodWindow( kLogLikelihoodChunk + 100 );

        MemoryStats bounded;
        {
            RoutedNetwork<TensorDataType::BF16> network( "gemma", at_chunk, Device::Cuda( 0 ) );
            network.build( context );
            bounded = network.getMemoryStats();
        }

        MemoryStats over;
        {
            RoutedNetwork<TensorDataType::BF16> network( "gemma", above_chunk, Device::Cuda( 0 ) );
            network.build( context );
            over = network.getMemoryStats();
        }

        MemoryStats over_predicted;
        {
            RoutedNetwork<TensorDataType::BF16> predictor( "gemma", above_chunk, Device::Cuda( 0 ) );
            over_predicted = predictor.getRequiredMemory( context );
        }

        EXPECT_EQ( over.device_state_bytes, bounded.device_state_bytes );
        EXPECT_EQ( over_predicted.device_state_bytes, over.device_state_bytes )
            << "prediction must clamp exactly as the build does";
    }

    TEST_F( GemmaMixtureOfExpertsCudaTests, DeploymentFootprint_ReadsExpertGeometryFromTheWeights )
    {
        if ( !captureExists() )
        {
            GTEST_SKIP() << "tiny MoE capture not present at: " << captureDirectory().string();
        }

        const DeploymentFootprint footprint = GemmaModel<DeviceType::Cuda, TensorDataType::FP32>::getDeploymentFootprint(
            weightsPath( TensorDataType::FP32 ), GemmaModelConfig( kContext ), Device::Cuda( 0 ) );

        EXPECT_GT( footprint.memory.device_inactive_parameter_bytes, 0u );
    }

    // The routed chassis builds FP4 at group 64: the dispatch, the stored scheme name and the model all agree.
    TEST_F( GemmaMixtureOfExpertsCudaTests, Fp4Load_UsesTheRoutedGroup )
    {
        if ( !captureExists() )
        {
            GTEST_SKIP() << "tiny MoE capture not present at: " << captureDirectory().string();
        }

        GemmaModelConfig config( kContext );
        config.withWeightQuantization( WeightQuantization::FP4 );

        const auto model = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::load(
            weightsPath( TensorDataType::BF16 ), config, Device::Cuda( 0 ) );

        EXPECT_EQ( model->weightQuantizationScheme(), "per_group_fp4_64" );
    }

    // The model refuses a policy the expert bank does not implement, naming the bank, before a routed network of
    // that policy is instantiated; the bank's own construction check is never reached.
    TEST_F( GemmaMixtureOfExpertsCudaTests, Fp8AndQ4_0_RefusedByTheModelBeforeTheBank )
    {
        if ( !captureExists() )
        {
            GTEST_SKIP() << "tiny MoE capture not present at: " << captureDirectory().string();
        }

        for ( WeightQuantization quantization : { WeightQuantization::FP8, WeightQuantization::Q4_0 } )
        {
            GemmaModelConfig config( kContext );
            config.withWeightQuantization( quantization );

            try
            {
                GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::getDeploymentFootprint(
                    weightsPath( TensorDataType::BF16 ), config, Device::Cuda( 0 ) );

                ADD_FAILURE() << weightQuantizationName( quantization ) << " was not refused";
            }
            catch ( const std::runtime_error& error )
            {
                // The bank's own refusal also names the expert bank, so the test asks for the model's words.
                EXPECT_NE( std::string( error.what() ).find(
                    "GemmaModel::getDeploymentFootprint: a mixture-of-experts Gemma cannot run" ), std::string::npos )
                    << weightQuantizationName( quantization ) << ": " << error.what();
            }
        }
    }

    // ====================================================================
    // Refusals
    // ====================================================================

    TEST_F( GemmaMixtureOfExpertsCudaTests, DenseConfig_RefusedByRoutedTransformer )
    {
        GemmaConfig dense = routedConfig();
        dense.withMixtureOfExperts( 0, 0, 0 );

        EXPECT_THROW( constructRouted( dense ), std::invalid_argument );
    }

    TEST_F( GemmaMixtureOfExpertsCudaTests, RoutedConfig_RefusedByDenseTransformer )
    {
        EXPECT_THROW( constructDense( routedConfig() ), std::invalid_argument );
    }
}
