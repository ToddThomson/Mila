/**
 * @file Llama.HuggingFaceReference.Cuda.cpp
 * @brief LlamaTransformer against a tiny HuggingFace Llama: logits, decode, and sequence log-likelihood.
 *
 * The converted weights and the reference come from Mila/Tools/Converters/Llama/hf_llama_tiny_reference.py,
 * which converts its own checkpoint with Llama/convert_weights.py, so one capture holds the converter's names
 * and the network's wiring together. Two captures of the same weights, without and with Llama 3's frequency
 * scaling: ModelFamilyParity.md 8.4, L1 and L2. The gates skip without them.
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
#include <vector>

import Mila;

#include "Measurement/LogLikelihoodHarness.h"

namespace Mila::Tests::Dnn::Components::Transformers::Llama
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        namespace fs = std::filesystem;

        template<TensorDataType TPrecision>
        using Network = LlamaTransformer<DeviceType::Cuda, TPrecision>;

        constexpr int64_t kBatch = 1;
        constexpr int64_t kContext = 32;

        // Four chunks over the 32-token capture, so every chunk after the first attends cached history.
        constexpr int64_t kChunk = 8;

        // Fixed before the first run: Gemma's tiny-model tolerances (Gemma4MoE.md Phase 8), where the BF16
        // figure is what HuggingFace's own bfloat16 run measured against its float32 one.
        constexpr double kFp32LogitTolerance = 1e-4;
        constexpr double kBf16RelativeL2 = 1e-1;
        constexpr double kFp32LogLikelihoodTolerance = 1e-4;

        /// The spacing of BF16 values at `value`'s magnitude: 7 stored mantissa bits.
        double bf16Step( float value )
        {
            return std::ldexp( 1.0, std::ilogb( std::fabs( value ) ) - 7 );
        }

        /// One capture of the script: the same weights, with and without Llama 3's frequency scaling.
        enum class Capture
        {
            Unscaled,
            Scaled
        };

        fs::path captureDirectory( Capture capture )
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "llama"
                / ( capture == Capture::Scaled ? "llama_tiny_scaled" : "llama_tiny" );
        }

        fs::path weightsPath( Capture capture, TensorDataType precision )
        {
            return captureDirectory( capture )
                / ( precision == TensorDataType::BF16 ? "llama_tiny_bf16.bin" : "llama_tiny_fp32.bin" );
        }

        fs::path referencePath( Capture capture )
        {
            return captureDirectory( capture ) / "llama_tiny_reference.safetensors";
        }

        bool captureExists( Capture capture )
        {
            return fs::exists( referencePath( capture ) )
                && fs::exists( weightsPath( capture, TensorDataType::FP32 ) )
                && fs::exists( weightsPath( capture, TensorDataType::BF16 ) );
        }

        // The geometry LlamaModel builds for the same file.
        LlamaConfig capturedConfig( Capture capture, TensorDataType precision, dim_t window = 1 )
        {
            Serialization::WeightsReader weights( weightsPath( capture, precision ) );

            LlamaConfig config = LlamaModel<DeviceType::Cuda, TensorDataType::FP32>::configFromMetadata(
                weights.getWeightsMetadata() );
            config.withLogLikelihoodWindow( window );

            return config;
        }

        template<TensorDataType TPrecision>
        std::unique_ptr<Network<TPrecision>> loadedNetwork( Capture capture, const LlamaConfig& config, int64_t chunk )
        {
            Serialization::WeightsReader weights( weightsPath( capture, TPrecision ) );

            auto network = std::make_unique<Network<TPrecision>>( "llama", config, Device::Cuda( 0 ) );
            network->build( BuildContext( shape_t{ kBatch, kContext }, RuntimeMode::Inference ).withPrefillSize( chunk ) );
            network->loadParameters( weights );

            return network;
        }

        std::vector<std::int32_t> referenceTokens( Capture capture )
        {
            Serialization::WeightsReader reference( referencePath( capture ) );
            auto blob = reference.readTensorBlob<CpuMemoryResource>( "tokens" );

            std::vector<std::int32_t> tokens( static_cast<std::size_t>( blob.getMetadata().shape[ 0 ] ) );
            std::memcpy( tokens.data(), blob.data(), tokens.size() * sizeof( std::int32_t ) );

            return tokens;
        }

        struct LogLikelihoodReference
        {
            double total{ 0.0 };
            dim_t positions{ 0 };
        };

        LogLikelihoodReference readLogLikelihoodReference( Capture capture )
        {
            Serialization::WeightsReader reference( referencePath( capture ) );
            auto blob = reference.readTensorBlob<CpuMemoryResource>( "next_token_log_probabilities" );

            std::vector<float> per_position( static_cast<std::size_t>( blob.getMetadata().shape[ 0 ] ) );
            std::memcpy( per_position.data(), blob.data(), per_position.size() * sizeof( float ) );

            LogLikelihoodReference result;

            for ( const float log_probability : per_position )
            {
                result.total += log_probability;
            }

            result.positions = static_cast<dim_t>( per_position.size() );

            return result;
        }
    }

    class LlamaHuggingFaceReferenceCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            for ( const Capture capture : { Capture::Unscaled, Capture::Scaled } )
            {
                if ( !captureExists( capture ) )
                {
                    GTEST_SKIP() << "tiny Llama capture not present at: " << captureDirectory( capture ).string();
                }
            }
        }

        /// Prefill the prompt, then decode the reference's own tokens one at a time, against HuggingFace's logits.
        template<TensorDataType TPrecision>
        void expectLogitsMatchReference( Capture capture )
        {
            auto network = loadedNetwork<TPrecision>( capture, capturedConfig( capture, TPrecision ), kContext );

            Serialization::WeightsReader reference( referencePath( capture ) );
            auto logits_blob = reference.readTensorBlob<CpuMemoryResource>( "logits" );

            const std::vector<std::int32_t> tokens = referenceTokens( capture );
            const int64_t total = static_cast<int64_t>( tokens.size() );
            const int64_t steps = static_cast<int64_t>( logits_blob.getMetadata().shape[ 0 ] );
            const int64_t vocab = static_cast<int64_t>( logits_blob.getMetadata().shape[ 1 ] );
            const int64_t prompt = total - ( steps - 1 );

            ASSERT_LE( total, kContext );

            std::vector<float> expected( static_cast<std::size_t>( steps * vocab ) );
            ASSERT_EQ( logits_blob.sizeBytes(), expected.size() * sizeof( float ) );
            std::memcpy( expected.data(), logits_blob.data(), logits_blob.sizeBytes() );

            std::vector<float> actual;

            auto append = [&]( const auto& logits )
            {
                const std::vector<float> host = Measurement::hostLogits( *network, logits );
                actual.insert( actual.end(), host.begin(), host.end() );
            };

            append( network->prefill( Measurement::deviceTokens( *network,
                std::vector<std::int32_t>( tokens.begin(), tokens.begin() + prompt ) ) ) );

            for ( int64_t step = 1; step < steps; ++step )
            {
                const int64_t position = prompt + step - 1;

                append( network->decode( Measurement::deviceTokens( *network, { tokens[ static_cast<std::size_t>( position ) ] } ), position ) );
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

                std::cout << std::format( "[ reference ] {} {} step {}: worst {:.3e}, relative L2 {:.3e}, argmax {} vs {}\n",
                    capture == Capture::Scaled ? "scaled" : "unscaled", TPrecision == TensorDataType::BF16 ? "BF16" : "FP32",
                    step, worst, relative_l2, got_argmax, want_argmax );

                if constexpr ( TPrecision == TensorDataType::BF16 )
                {
                    EXPECT_LE( relative_l2, kBf16RelativeL2 ) << "step " << step;

                    // The head emits BF16, so two tokens closer in HuggingFace's own logits than one BF16 step at
                    // that magnitude cannot be ordered by it: the first run's step 1 is such a tie (0.0087 against a
                    // step of 0.0156). Any other disagreement is a defect.
                    const double reference_gap = static_cast<double>( want[ want_argmax ] ) - want[ got_argmax ];

                    EXPECT_LT( reference_gap, bf16Step( want[ want_argmax ] ) )
                        << "step " << step << ": argmax " << got_argmax << " where HuggingFace chose " << want_argmax;
                }
                else
                {
                    EXPECT_EQ( mismatches, 0 ) << "step " << step << " of " << vocab << " logits; worst " << worst;
                }
            }
        }

        /// The total at windows 1, 3 and 8 against HuggingFace's, chunked so later chunks attend cached history.
        void expectSequenceLogLikelihoodMatchesReference( Capture capture )
        {
            const LogLikelihoodReference reference = readLogLikelihoodReference( capture );
            const std::vector<std::int32_t> tokens = referenceTokens( capture );

            for ( const dim_t window : { 1, 3, 8 } )
            {
                auto network = loadedNetwork<TensorDataType::FP32>(
                    capture, capturedConfig( capture, TensorDataType::FP32, window ), kChunk );

                const SequenceLogLikelihood measured = Measurement::sequenceLogLikelihoodOf( *network, tokens );

                std::cout << std::format( "[ log-likelihood ] {} window {}: {:.9f} against HuggingFace {:.9f}, difference {:.3e}\n",
                    capture == Capture::Scaled ? "scaled" : "unscaled", window, measured.total_log_probability,
                    reference.total, measured.total_log_probability - reference.total );

                EXPECT_EQ( measured.scored_positions, reference.positions ) << "window " << window;
                EXPECT_NEAR( measured.total_log_probability, reference.total, kFp32LogLikelihoodTolerance ) << "window " << window;
            }
        }
    };

    // ====================================================================
    // L1: logits, decode and log-likelihood without frequency scaling
    // ====================================================================

    TEST_F( LlamaHuggingFaceReferenceCudaTests, Fp32_LogitsMatchHuggingFace )
    {
        expectLogitsMatchReference<TensorDataType::FP32>( Capture::Unscaled );
    }

    TEST_F( LlamaHuggingFaceReferenceCudaTests, Bf16_LogitsMatchHuggingFace )
    {
        expectLogitsMatchReference<TensorDataType::BF16>( Capture::Unscaled );
    }

    // Window 1 is decode's path, 3 leaves a partial final window in each chunk, and 8 is the whole chunk.
    TEST_F( LlamaHuggingFaceReferenceCudaTests, Fp32_SequenceLogLikelihood_MatchesHuggingFace )
    {
        expectSequenceLogLikelihoodMatchesReference( Capture::Unscaled );
    }

    // ====================================================================
    // L2: the same gates on a model trained with Llama 3's frequency scaling
    // ====================================================================

    TEST_F( LlamaHuggingFaceReferenceCudaTests, Scaled_Fp32_LogitsMatchHuggingFace )
    {
        expectLogitsMatchReference<TensorDataType::FP32>( Capture::Scaled );
    }

    TEST_F( LlamaHuggingFaceReferenceCudaTests, Scaled_Bf16_LogitsMatchHuggingFace )
    {
        expectLogitsMatchReference<TensorDataType::BF16>( Capture::Scaled );
    }

    TEST_F( LlamaHuggingFaceReferenceCudaTests, Scaled_Fp32_SequenceLogLikelihood_MatchesHuggingFace )
    {
        expectSequenceLogLikelihoodMatchesReference( Capture::Scaled );
    }

    // The gate can fail: the scaled model's weights read without their scaling must miss HuggingFace. Were the
    // scaling too small to show at 32 tokens, the gates above would pass whether or not it is applied.
    TEST_F( LlamaHuggingFaceReferenceCudaTests, Scaled_Fp32_WithoutTheScaling_MissesHuggingFace )
    {
        const LogLikelihoodReference reference = readLogLikelihoodReference( Capture::Scaled );

        LlamaConfig unscaled = capturedConfig( Capture::Scaled, TensorDataType::FP32, 3 );

        ASSERT_TRUE( unscaled.getRoPEFrequencyScaling().isScaled() ) << "the scaled capture's weights carry no scaling";

        unscaled.withRoPEFrequencyScaling( RopeFrequencyScaling{} );

        auto network = loadedNetwork<TensorDataType::FP32>( Capture::Scaled, unscaled, kChunk );

        const SequenceLogLikelihood measured = Measurement::sequenceLogLikelihoodOf( *network, referenceTokens( Capture::Scaled ) );

        std::cout << std::format( "[ log-likelihood ] scaled weights without the scaling: difference {:.3e}\n",
            measured.total_log_probability - reference.total );

        EXPECT_GT( std::fabs( measured.total_log_probability - reference.total ), kFp32LogLikelihoodTolerance );
    }

    // ====================================================================
    // L2: a Llama file that does not say how its positions are scaled is refused
    // ====================================================================

    namespace
    {
        Serialization::WeightsMetadata llamaMetadata( const std::string& rope_scaling )
        {
            Serialization::WeightsMetadata metadata{};
            metadata.architecture = "llama";
            metadata.model_name = "llama_under_test";
            metadata.vocab_size = 256;
            metadata.max_seq_length = 64;
            metadata.embedding_dim = 512;
            metadata.num_layers = 2;
            metadata.num_heads = 4;
            metadata.num_kv_heads = 2;
            metadata.hidden_dim = 1024;
            metadata.rope_theta = 10000.0f;
            metadata.rope_scaling = rope_scaling;
            metadata.rope_scaling_factor = 8.0f;
            metadata.rope_low_frequency_factor = 1.0f;
            metadata.rope_high_frequency_factor = 4.0f;
            metadata.rope_original_context_length = 8192;

            return metadata;
        }

        using LlamaFp32 = LlamaModel<DeviceType::Cuda, TensorDataType::FP32>;
    }

    TEST( LlamaFrequencyScalingMetadataTests, Llama3_ReadsAllFourValues )
    {
        const LlamaConfig config = LlamaFp32::configFromMetadata( llamaMetadata( "llama3" ) );

        EXPECT_EQ( config.getRoPEFrequencyScaling(), ( RopeFrequencyScaling{ 8.0f, 1.0f, 4.0f, 8192 } ) );
    }

    TEST( LlamaFrequencyScalingMetadataTests, None_IsUnscaled )
    {
        EXPECT_FALSE( LlamaFp32::configFromMetadata( llamaMetadata( "none" ) ).getRoPEFrequencyScaling().isScaled() );
    }

    TEST( LlamaFrequencyScalingMetadataTests, Absent_IsRefused )
    {
        EXPECT_THROW( LlamaFp32::configFromMetadata( llamaMetadata( "" ) ), std::runtime_error );
    }

    TEST( LlamaFrequencyScalingMetadataTests, UnknownRule_IsRefused )
    {
        EXPECT_THROW( LlamaFp32::configFromMetadata( llamaMetadata( "yarn" ) ), std::runtime_error );
    }

    // Each position's most likely next token, read from the rows the log-likelihood evaluated, is the token
    // greedy generation chose there.
    TEST_F( LlamaHuggingFaceReferenceCudaTests, Fp32_SequenceLogLikelihood_ArgmaxIsTheGreedyToken )
    {
        constexpr int kGenerated = 6;
        constexpr std::size_t kPrompt = 20;

        const std::vector<std::int32_t> tokens = referenceTokens( Capture::Unscaled );
        const std::vector<std::int32_t> prompt( tokens.begin(), tokens.begin() + kPrompt );

        auto network = loadedNetwork<TensorDataType::FP32>(
            Capture::Unscaled, capturedConfig( Capture::Unscaled, TensorDataType::FP32, 3 ), kChunk );

        const Measurement::GreedyContinuation greedy =
            Measurement::greedyContinuationOf( *network, prompt, kGenerated, {}, kContext );

        ASSERT_EQ( greedy.tokens.size(), static_cast<std::size_t>( kGenerated ) );

        std::vector<std::int32_t> sequence = prompt;
        sequence.insert( sequence.end(), greedy.tokens.begin(), greedy.tokens.end() );

        using DeviceLogits = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;

        std::vector<std::vector<float>> rows;
        const int64_t vocab = static_cast<int64_t>( capturedConfig( Capture::Unscaled, TensorDataType::FP32 ).getVocabSize() );

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

        (void)Measurement::sequenceLogLikelihoodOf( *network, sequence );

        network->stopObserving();

        ASSERT_GE( rows.size(), sequence.size() - 1 );

        for ( int generated = 0; generated < kGenerated; ++generated )
        {
            const std::size_t position = prompt.size() - 1 + static_cast<std::size_t>( generated );

            EXPECT_EQ( Measurement::argMax( rows[ position ] ), greedy.tokens[ static_cast<std::size_t>( generated ) ] )
                << "position " << position;
        }
    }
}
