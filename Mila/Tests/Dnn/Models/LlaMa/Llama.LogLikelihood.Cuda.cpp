/**
 * @file Llama.LogLikelihood.Cuda.cpp
 * @brief Llama's sequence log-likelihood on real weights: window 1 against window 64 on the 3.1 8B, and the 3.2 1B
 *        against HuggingFace on both sides of its original context.
 *
 * ModelFamilyParity.md 8.4, L1 and L2. Needs exported weights, the Llama tokenizer, the wikitext-2 test split and a
 * HuggingFace capture, so it never runs in CI.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
// C stdio rather than <fstream>: an input-stream header in a TU that does `import Mila;` leaves
// std::basic_istream::sentry incomplete.
#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

import Mila;

#include "Common/LogLikelihoodHarness.h"

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace fs = std::filesystem;

    namespace
    {
        using LlamaBf16 = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;

        // What LlamaModel builds for the published FP4 weights.
        using MeasuredLlama = LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupFp4<128>, LlamaBf16::LlamaKvPolicy>;

        // ModelFamilyParity.md section 9, items 11 and 13: G1's bound, taken for Llama unchanged.
        constexpr double kWindowRelativePerplexityTolerance = 2e-3;

        // <|begin_of_text|>. The tokenizer does not add it; Llama is trained with it at the start of every sequence.
        constexpr std::int32_t kBeginOfText = 128000;

        fs::path weightsPath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama31_8b_instruct_fp4.safetensors";
        }

        fs::path tokenizerPath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama32_tokenizer.bin";
        }

        fs::path corpusPath()
        {
            return fs::path( TEST_DATA_DIR ).parent_path() / "Mila" / "Tools" / "Quantization" / "corpus" / "wiki.test.raw";
        }

        bool inputsPresent()
        {
            return fs::exists( weightsPath() ) && fs::exists( tokenizerPath() ) && fs::exists( corpusPath() );
        }

        std::unique_ptr<MeasuredLlama> buildMeasuredLlama( dim_t window, dim_t context_length )
        {
            Serialization::WeightsReader reader( weightsPath() );

            LlamaConfig config = LlamaBf16::configFromMetadata( reader.getWeightsMetadata() );
            config.withLogLikelihoodWindow( window );

            PrefillChunking chunking;

            auto network = Common::buildMeasuredNetwork<MeasuredLlama>(
                weightsPath(), config, DeviceId{ DeviceType::Cuda, 0 }, context_length, &chunking );

            std::cout << std::format( "  {}: window {}, context {}, prefill chunk {}\n",
                weightsPath().filename().string(), window, context_length, chunking.chunk_rows ) << std::flush;

            return network;
        }

        /// The first `token_budget` tokens of wikitext-2 test. Four characters per token over-reads, then truncates.
        std::vector<std::int32_t> corpusTokens( dim_t token_budget )
        {
            std::FILE* corpus_file = std::fopen( corpusPath().string().c_str(), "rb" );

            if ( corpus_file == nullptr )
            {
                return {};
            }

            std::string text( static_cast<std::size_t>( token_budget ) * 4, '\0' );
            text.resize( std::fread( text.data(), 1, text.size(), corpus_file ) );
            std::fclose( corpus_file );

            auto tokens = Mila::Data::BpeTokenizer::loadLlama32( tokenizerPath() )->encode( text );

            if ( tokens.size() > static_cast<std::size_t>( token_budget ) )
            {
                tokens.resize( static_cast<std::size_t>( token_budget ) );
            }

            return tokens;
        }

        struct CorpusLogLikelihood
        {
            SequenceLogLikelihood total;
            double seconds{ 0.0 };

            double perplexity() const
            {
                return std::exp( -total.total_log_probability / static_cast<double>( total.scored_positions ) );
            }
        };

        /// Non-overlapping segments of the context length, each <|begin_of_text|> then text, scored from a cold cache.
        CorpusLogLikelihood corpusLogLikelihood( MeasuredLlama& network, const std::vector<std::int32_t>& tokens,
            dim_t context_length )
        {
            CorpusLogLikelihood result;

            const std::size_t text_per_segment = static_cast<std::size_t>( context_length ) - 1;
            const auto start = std::chrono::steady_clock::now();

            for ( std::size_t offset = 0; offset < tokens.size(); offset += text_per_segment )
            {
                const std::size_t length = std::min<std::size_t>( text_per_segment, tokens.size() - offset );

                std::vector<std::int32_t> segment{ kBeginOfText };
                segment.insert( segment.end(),
                    tokens.begin() + static_cast<std::ptrdiff_t>( offset ),
                    tokens.begin() + static_cast<std::ptrdiff_t>( offset + length ) );

                const SequenceLogLikelihood scored = Common::sequenceLogLikelihoodOf( network, segment );

                result.total.total_log_probability += scored.total_log_probability;
                result.total.scored_positions += scored.scored_positions;
            }

            result.seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();

            return result;
        }
    }

    // ====================================================================
    // Window 1 against window 64 (ModelFamilyParity.md 8.4, L1)
    //   MilaTests --gtest_also_run_disabled_tests
    //       --gtest_filter=LlamaLogLikelihoodCudaTests.DISABLED_WindowsAgree_Fp4
    //
    // One build is alive at a time, and both score the same 16K tokens at the same context. Pin the card by
    // UUID; two cards disagree in the last digits.
    // ====================================================================
    TEST( LlamaLogLikelihoodCudaTests, DISABLED_WindowsAgree_Fp4 )
    {
        if ( getDeviceCount( DeviceType::Cuda ) == 0 || !inputsPresent() )
        {
            GTEST_SKIP() << "Needs a CUDA device, " << weightsPath().string() << ", the Llama tokenizer and wikitext-2";
        }

        constexpr dim_t kContextLength = 1024;
        constexpr dim_t kTokenBudget = 16384;

        const std::vector<std::int32_t> tokens = corpusTokens( kTokenBudget );

        ASSERT_GT( tokens.size(), static_cast<std::size_t>( kContextLength ) );

        CorpusLogLikelihood arms[ 2 ];
        const dim_t windows[ 2 ] = { 1, 64 };

        for ( int arm = 0; arm < 2; ++arm )
        {
            auto network = buildMeasuredLlama( windows[ arm ], kContextLength );

            arms[ arm ] = corpusLogLikelihood( *network, tokens, kContextLength );

            std::cout << std::format(
                "  window {:>2}: {} positions, mean negative log-likelihood {:.17g}, PERPLEXITY {:.4f}, {:.1f} s ({:.0f} positions/s)\n",
                windows[ arm ], arms[ arm ].total.scored_positions,
                -arms[ arm ].total.total_log_probability / static_cast<double>( arms[ arm ].total.scored_positions ),
                arms[ arm ].perplexity(), arms[ arm ].seconds,
                static_cast<double>( arms[ arm ].total.scored_positions ) / arms[ arm ].seconds ) << std::flush;
        }

        ASSERT_EQ( arms[ 0 ].total.scored_positions, arms[ 1 ].total.scored_positions );

        const double relative = std::fabs( arms[ 1 ].perplexity() - arms[ 0 ].perplexity() ) / arms[ 0 ].perplexity();

        std::cout << std::format( "  relative perplexity difference: {:.3e} (bound {:.0e})\n",
            relative, kWindowRelativePerplexityTolerance ) << std::flush;

        EXPECT_LE( relative, kWindowRelativePerplexityTolerance );
    }

    // ====================================================================
    // Llama 3.2 1B against HuggingFace past its original context (ModelFamilyParity.md 8.4, L2)
    //   MilaTests --gtest_also_run_disabled_tests --gtest_filter=LlamaLongContextCudaTests.*
    //
    // The reference is Mila/Tools/Converters/Llama/hf_llama_long_context_reference.py: HuggingFace's per-position
    // log-probability of the first 10240 wikitext-2 tokens at FP32, and a greedy continuation. Llama 3's frequency
    // scaling has an original context of 8192, so the positions are compared in two bands, below it and past it.
    // ====================================================================

    namespace
    {
        using LlamaFp32 = LlamaModel<DeviceType::Cuda, TensorDataType::FP32>;
        using MeasuredLlama1B = LlamaTransformer<DeviceType::Cuda, TensorDataType::FP32>;

        // Written before the first run: FP32 against FP32, summation order the only difference. The tiny model
        // agrees to 2e-7 nats per token (Llama.HuggingFaceReference.Cuda.cpp); a real one is sixteen layers deep.
        constexpr double kLongContextMeanTolerance = 1e-3;

        // Positions scored at or past this index predict a token at or past Llama 3's original context.
        constexpr std::size_t kOriginalContext = 8192;

        fs::path oneBillionWeightsPath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama32_1b_fp32.bin";
        }

        fs::path longContextReferencePath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama32_1b_long_context_reference.safetensors";
        }

        std::vector<std::int32_t> readTokens( Serialization::WeightsReader& reader, const std::string& name )
        {
            auto blob = reader.readTensorBlob<CpuMemoryResource>( name );

            std::vector<std::int32_t> tokens( static_cast<std::size_t>( blob.getMetadata().shape[ 0 ] ) );
            std::memcpy( tokens.data(), blob.data(), tokens.size() * sizeof( std::int32_t ) );

            return tokens;
        }

        std::unique_ptr<MeasuredLlama1B> buildMeasuredLlama1B( bool scaled, dim_t context_length )
        {
            Serialization::WeightsReader reader( oneBillionWeightsPath() );

            LlamaConfig config = LlamaFp32::configFromMetadata( reader.getWeightsMetadata() );
            config.withLogLikelihoodWindow( 64 );

            if ( !scaled )
            {
                config.withRoPEFrequencyScaling( RopeFrequencyScaling{} );
            }

            return Common::buildMeasuredNetwork<MeasuredLlama1B>(
                oneBillionWeightsPath(), config, DeviceId{ DeviceType::Cuda, 0 }, context_length );
        }

        /// Mean negative log-likelihood of the positions in [begin, end), per position.
        double meanNegativeLogLikelihood( const std::vector<double>& per_position, std::size_t begin, std::size_t end )
        {
            double sum = 0.0;

            for ( std::size_t position = begin; position < end; ++position )
            {
                sum -= per_position[ position ];
            }

            return sum / static_cast<double>( end - begin );
        }

        /// Each scored position's log-probability, read from the head rows the log-likelihood evaluates.
        std::vector<double> perPositionLogProbabilities( MeasuredLlama1B& network, const std::vector<std::int32_t>& tokens )
        {
            using DeviceLogits = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;

            std::vector<double> per_position;
            per_position.reserve( tokens.size() );

            const std::int64_t vocab = 128256;

            const std::size_t observed = network.observe( "*.lm_head", ComputePassMask::inference(),
                [&]( std::string_view, ComputePass, std::string_view stage, const ITensor& value )
                {
                    const auto* typed = dynamic_cast<const DeviceLogits*>( &value );

                    if ( stage != "output" || typed == nullptr )
                    {
                        return;
                    }

                    network.synchronize();

                    auto host = toHost<TensorDataType::FP32>( *typed );

                    for ( std::int64_t row = 0; row * vocab < static_cast<std::int64_t>( host.size() ); ++row )
                    {
                        const std::size_t position = per_position.size();

                        if ( position + 1 >= tokens.size() )
                        {
                            break;
                        }

                        const float* logits = host.data() + row * vocab;
                        const float largest = *std::max_element( logits, logits + vocab );

                        double partition = 0.0;

                        for ( std::int64_t column = 0; column < vocab; ++column )
                        {
                            partition += std::exp( static_cast<double>( logits[ column ] ) - largest );
                        }

                        per_position.push_back( static_cast<double>( logits[ tokens[ position + 1 ] ] ) - largest - std::log( partition ) );
                    }
                } );

            EXPECT_EQ( observed, 1u ) << "the head was not selected, so no rows will arrive";

            (void)Common::sequenceLogLikelihoodOf( network, tokens );

            network.stopObserving();

            return per_position;
        }

        bool longContextInputsPresent()
        {
            return fs::exists( oneBillionWeightsPath() ) && fs::exists( longContextReferencePath() );
        }
    }

    TEST( LlamaLongContextCudaTests, DISABLED_BandsMatchHuggingFace_1B )
    {
        if ( getDeviceCount( DeviceType::Cuda ) == 0 || !longContextInputsPresent() )
        {
            GTEST_SKIP() << "Needs a CUDA device, " << oneBillionWeightsPath().string() << " and "
                << longContextReferencePath().string();
        }

        Serialization::WeightsReader reference( longContextReferencePath() );

        const std::vector<std::int32_t> tokens = readTokens( reference, "tokens" );

        auto reference_blob = reference.readTensorBlob<CpuMemoryResource>( "next_token_log_probabilities" );
        std::vector<float> reference_floats( static_cast<std::size_t>( reference_blob.getMetadata().shape[ 0 ] ) );
        std::memcpy( reference_floats.data(), reference_blob.data(), reference_floats.size() * sizeof( float ) );

        const std::vector<double> expected( reference_floats.begin(), reference_floats.end() );

        ASSERT_EQ( expected.size(), tokens.size() - 1 );

        const dim_t context_length = static_cast<dim_t>( tokens.size() );
        const std::size_t below = kOriginalContext - 1;
        const std::size_t end = expected.size();

        const double expected_below = meanNegativeLogLikelihood( expected, 0, below );
        const double expected_above = meanNegativeLogLikelihood( expected, below, end );

        double measured_above[ 2 ] = {};

        for ( const bool scaled : { true, false } )
        {
            auto network = buildMeasuredLlama1B( scaled, context_length );

            const std::vector<double> measured = perPositionLogProbabilities( *network, tokens );

            ASSERT_EQ( measured.size(), expected.size() );

            const double measured_below = meanNegativeLogLikelihood( measured, 0, below );
            measured_above[ scaled ? 0 : 1 ] = meanNegativeLogLikelihood( measured, below, end );

            std::cout << std::format(
                "  {}: positions 1-{} {:.6f} (HuggingFace {:.6f}, {:+.2e}); {}-{} {:.6f} (HuggingFace {:.6f}, {:+.2e})\n",
                scaled ? "scaled  " : "unscaled", below, measured_below, expected_below, measured_below - expected_below,
                below + 1, end, measured_above[ scaled ? 0 : 1 ], expected_above,
                measured_above[ scaled ? 0 : 1 ] - expected_above ) << std::flush;

            if ( scaled )
            {
                EXPECT_NEAR( measured_below, expected_below, kLongContextMeanTolerance );
                EXPECT_NEAR( measured_above[ 0 ], expected_above, kLongContextMeanTolerance );
            }
        }

        // The gate can fail: without the scaling the band past the original context must miss.
        EXPECT_GT( std::fabs( measured_above[ 1 ] - expected_above ), kLongContextMeanTolerance );
    }

    // ====================================================================
    // Flash prefill against the cuBLASLt pipeline, one build (ModelFamilyParity.md 8.4, L4)
    //   MilaTests --gtest_also_run_disabled_tests --gtest_filter=LlamaFlashPrefillCudaTests.*
    //
    // Every BF16 Llama prefills through flash. The cuBLASLt arm is the same network with a full-width score buffer
    // installed on every block and flash switched off -- what the network built before flash was wired.
    // ====================================================================

    namespace
    {
        using CublasLtWorkspace = GqaWorkspace<DeviceType::Cuda, TensorDataType::BF16>;

        // Written before the first run: Gemma's flash A/B agreed within 0.016 nats per token (GqaFlashAttention.md).
        constexpr double kFlashAgreementTolerance = 0.02;

        /// Switch every block of `network` to the cuBLASLt prefill; the returned workspace must outlive its use.
        CublasLtWorkspace useCublasLtPrefill( MeasuredLlama& network, const LlamaConfig& config, dim_t context_length,
            dim_t prefill_chunk )
        {
            const dim_t heads = config.getNumHeads();

            CublasLtWorkspace workspace = makeGqaWorkspace<DeviceType::Cuda, TensorDataType::BF16>(
                DeviceId{ DeviceType::Cuda, 0 }, 1, heads, config.getModelDim() / heads, context_length, prefill_chunk,
                context_length, "cublaslt_arm.gqa_ws." );

            for ( dim_t layer = 0; layer < config.getNumLayers(); ++layer )
            {
                auto block = std::dynamic_pointer_cast<MeasuredLlama::TransformerBlockType>(
                    network.findComponent( std::format( "{}.tf_layer_{}", network.getName(), layer ) ) );

                if ( block == nullptr )
                {
                    throw std::runtime_error( std::format( "layer {} is not a Llama block", layer ) );
                }

                block->setState( workspace.state() );
                block->setUseFlashPrefill( false );
            }

            return workspace;
        }

        LlamaConfig measuredConfig( dim_t window )
        {
            Serialization::WeightsReader reader( weightsPath() );

            LlamaConfig config = LlamaBf16::configFromMetadata( reader.getWeightsMetadata() );
            config.withLogLikelihoodWindow( window );

            return config;
        }
    }

    TEST( LlamaFlashPrefillCudaTests, DISABLED_FlashAgreesWithCublasLt_Fp4 )
    {
        if ( getDeviceCount( DeviceType::Cuda ) == 0 || !inputsPresent() )
        {
            GTEST_SKIP() << "Needs a CUDA device, " << weightsPath().string() << ", the Llama tokenizer and wikitext-2";
        }

        constexpr dim_t kContextLength = 4096;
        constexpr dim_t kTokenBudget = 16384;

        const std::vector<std::int32_t> tokens = corpusTokens( kTokenBudget );

        CorpusLogLikelihood arms[ 2 ];

        for ( const bool flash : { true, false } )
        {
            const LlamaConfig config = measuredConfig( 64 );

            PrefillChunking chunking;

            auto network = Common::buildMeasuredNetwork<MeasuredLlama>(
                weightsPath(), config, DeviceId{ DeviceType::Cuda, 0 }, kContextLength, &chunking );

            std::optional<CublasLtWorkspace> cublaslt;

            if ( !flash )
            {
                cublaslt = useCublasLtPrefill( *network, config, kContextLength, chunking.chunk_rows );
            }

            CorpusLogLikelihood& arm = arms[ flash ? 0 : 1 ];
            arm = corpusLogLikelihood( *network, tokens, kContextLength );

            std::cout << std::format( "  {}: chunk {}, {} positions, mean negative log-likelihood {:.9f}, perplexity {:.4f}\n",
                flash ? "flash   " : "cuBLASLt", chunking.chunk_rows, arm.total.scored_positions,
                -arm.total.total_log_probability / static_cast<double>( arm.total.scored_positions ), arm.perplexity() )
                << std::flush;
        }

        ASSERT_EQ( arms[ 0 ].total.scored_positions, arms[ 1 ].total.scored_positions );

        const double positions = static_cast<double>( arms[ 0 ].total.scored_positions );
        const double difference =
            ( arms[ 1 ].total.total_log_probability - arms[ 0 ].total.total_log_probability ) / positions;

        std::cout << std::format( "  mean difference {:+.3e} nats per token (bound {})\n", difference, kFlashAgreementTolerance );

        EXPECT_LE( std::fabs( difference ), kFlashAgreementTolerance );
    }

    // Prefill alone -- no head windows, no host reduction -- at each prompt length, flash and cuBLASLt, best of three.
    TEST( LlamaFlashPrefillCudaTests, DISABLED_PrefillRate_Fp4 )
    {
        if ( getDeviceCount( DeviceType::Cuda ) == 0 || !inputsPresent() )
        {
            GTEST_SKIP() << "Needs a CUDA device, " << weightsPath().string() << ", the Llama tokenizer and wikitext-2";
        }

        // Four characters per token over-reads for most text, but not by enough to promise 32K tokens.
        const std::vector<std::int32_t> corpus = corpusTokens( 40960 );

        for ( const dim_t prompt_length : { dim_t{ 8192 }, dim_t{ 32768 } } )
        {
            if ( corpus.size() + 1 < static_cast<std::size_t>( prompt_length ) )
            {
                std::cout << std::format( "  {}: the corpus is shorter than the prompt\n", prompt_length );
                continue;
            }

            std::vector<std::int32_t> prompt{ kBeginOfText };
            prompt.insert( prompt.end(), corpus.begin(), corpus.begin() + ( prompt_length - 1 ) );

            for ( const bool flash : { true, false } )
            {
                const LlamaConfig config = measuredConfig( 1 );

                PrefillChunking chunking;

                auto network = Common::buildMeasuredNetwork<MeasuredLlama>(
                    weightsPath(), config, DeviceId{ DeviceType::Cuda, 0 }, prompt_length, &chunking );

                std::optional<CublasLtWorkspace> cublaslt;

                if ( !flash )
                {
                    try
                    {
                        cublaslt = useCublasLtPrefill( *network, config, prompt_length, chunking.chunk_rows );
                    }
                    catch ( const std::exception& error )
                    {
                        std::cout << std::format( "  {}: cuBLASLt arm does not fit ({})\n", prompt_length, error.what() );
                        continue;
                    }
                }

                const auto device_prompt = Common::deviceTokens( *network, prompt );

                double best = 0.0;

                for ( int run = 0; run < 4; ++run )
                {
                    network->synchronize();
                    const auto start = std::chrono::steady_clock::now();

                    (void)network->prefill( device_prompt );
                    network->synchronize();

                    const double seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();

                    // The first run builds the cuBLASLt plans and warms the caches.
                    if ( run > 0 && ( best == 0.0 || seconds < best ) )
                    {
                        best = seconds;
                    }
                }

                std::cout << std::format( "  prompt {:>6}, {}: chunk {}, {:.3f} s, {:.0f} tokens/s\n", prompt_length,
                    flash ? "flash   " : "cuBLASLt", chunking.chunk_rows, best, static_cast<double>( prompt_length ) / best )
                    << std::flush;
            }
        }
    }

    // Flash prefill at each chunk, the planner bypassed, and what each chunk costs in context (ModelFamilyParity.md
    // 8.4, L4). The planner prefers the largest context at which the top rung fits, so a larger top rung buys
    // throughput with context -- both sides are printed. 2048 is above every family's top rung and is measured to
    // show what the next one would buy.
    TEST( LlamaFlashPrefillCudaTests, DISABLED_PrefillRateByChunk_Fp4 )
    {
        if ( getDeviceCount( DeviceType::Cuda ) == 0 || !inputsPresent() )
        {
            GTEST_SKIP() << "Needs a CUDA device, " << weightsPath().string() << ", the Llama tokenizer and wikitext-2";
        }

        constexpr dim_t kChunks[] = { 2048, 1024, 512, 256, 128 };
        const DeviceId device{ DeviceType::Cuda, 0 };
        const LlamaConfig config = measuredConfig( 1 );

        {
            Serialization::WeightsReader reader( weightsPath() );
            MeasuredLlama network( reader.getWeightsMetadata().model_name, config, device );

            const Mila::Deployment::DeviceReading reading = Mila::Deployment::DeviceReading::take( device );

            // One context step of footprint, for converting a chunk's extra bytes into context tokens.
            for ( const dim_t context_length : { dim_t{ 19456 }, dim_t{ 50176 } } )
            {
                const auto priced = [&]( dim_t length, dim_t chunk ) {
                    return network.getRequiredMemory( BuildContext( shape_t{ 1, length }, RuntimeMode::Inference, false )
                        .withAllocationGranularity( reading.allocation_granularity )
                        .withPrefillSize( chunk ) ).totalDeviceBytes();
                };

                const double bytes_per_token = static_cast<double>( priced( context_length + 1024, 512 ) - priced( context_length, 512 ) ) / 1024.0;

                for ( const dim_t chunk : kChunks )
                {
                    const std::size_t bytes = priced( context_length, chunk );
                    const double extra = static_cast<double>( bytes ) - static_cast<double>( priced( context_length, 512 ) );

                    std::cout << std::format( "  context {:>6}, chunk {:>4}: footprint {:.0f} MiB, {:+.0f} MiB against 512 = {:+.0f} context tokens\n",
                        context_length, chunk, static_cast<double>( bytes ) / ( 1024.0 * 1024.0 ), extra / ( 1024.0 * 1024.0 ),
                        -extra / bytes_per_token ) << std::flush;
                }
            }

            std::cout << std::format( "  free at reading {:.0f} MiB\n", static_cast<double>( reading.free_bytes ) / ( 1024.0 * 1024.0 ) );
        }

        const std::vector<std::int32_t> corpus = corpusTokens( 40960 );

        for ( const dim_t prompt_length : { dim_t{ 8192 }, dim_t{ 32768 } } )
        {
            if ( corpus.size() + 1 < static_cast<std::size_t>( prompt_length ) )
            {
                std::cout << std::format( "  {}: the corpus is shorter than the prompt\n", prompt_length );
                continue;
            }

            std::vector<std::int32_t> prompt{ kBeginOfText };
            prompt.insert( prompt.end(), corpus.begin(), corpus.begin() + ( prompt_length - 1 ) );

            for ( const dim_t chunk : kChunks )
            {
                Serialization::WeightsReader reader( weightsPath() );

                auto network = std::make_unique<MeasuredLlama>( reader.getWeightsMetadata().model_name, config, device );
                const Mila::Deployment::DeviceReading reading = Mila::Deployment::DeviceReading::take( device );

                network->build( BuildContext( shape_t{ 1, prompt_length }, RuntimeMode::Inference, false )
                    .withAllocationGranularity( reading.allocation_granularity )
                    .withPrefillSize( chunk ) );
                network->loadParameters( reader );

                const auto device_prompt = Common::deviceTokens( *network, prompt );

                double best = 0.0;

                for ( int run = 0; run < 4; ++run )
                {
                    network->synchronize();
                    const auto start = std::chrono::steady_clock::now();

                    (void)network->prefill( device_prompt );
                    network->synchronize();

                    const double seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();

                    if ( run > 0 && ( best == 0.0 || seconds < best ) )
                    {
                        best = seconds;
                    }
                }

                std::cout << std::format( "  prompt {:>6}, chunk {:>4}: {:.3f} s, {:.0f} tokens/s\n", prompt_length, chunk,
                    best, static_cast<double>( prompt_length ) / best ) << std::flush;
            }
        }
    }

    // Decode tokens per second with the fused decode-attention kernel and with the cuBLASLt pipeline, one build. The
    // fused kernel reads the live cache and cuBLASLt the whole allocated context, so both a short and a long prompt
    // are measured inside one 32K deployment.
    TEST( LlamaFlashPrefillCudaTests, DISABLED_DecodeRate_Fp4 )
    {
        if ( getDeviceCount( DeviceType::Cuda ) == 0 || !inputsPresent() )
        {
            GTEST_SKIP() << "Needs a CUDA device, " << weightsPath().string() << ", the Llama tokenizer and wikitext-2";
        }

        constexpr dim_t kContextLength = 32768;
        constexpr int kDecodeSteps = 128;

        const std::vector<std::int32_t> corpus = corpusTokens( 20480 );
        const LlamaConfig config = measuredConfig( 1 );

        auto network = Common::buildMeasuredNetwork<MeasuredLlama>(
            weightsPath(), config, DeviceId{ DeviceType::Cuda, 0 }, kContextLength );

        auto setFusedDecode = [&]( bool fused )
        {
            for ( dim_t layer = 0; layer < config.getNumLayers(); ++layer )
            {
                auto block = std::dynamic_pointer_cast<MeasuredLlama::TransformerBlockType>(
                    network->findComponent( std::format( "{}.tf_layer_{}", network->getName(), layer ) ) );

                ASSERT_NE( block, nullptr ) << "layer " << layer;

                block->setUseFlashDecode( fused );
            }
        };

        for ( const dim_t prompt_length : { dim_t{ 512 }, dim_t{ 16384 } } )
        {
            ASSERT_GE( corpus.size() + 1, static_cast<std::size_t>( prompt_length ) );

            std::vector<std::int32_t> prompt{ kBeginOfText };
            prompt.insert( prompt.end(), corpus.begin(), corpus.begin() + ( prompt_length - 1 ) );

            for ( const bool fused : { true, false } )
            {
                setFusedDecode( fused );

                (void)network->prefill( Common::deviceTokens( *network, prompt ) );

                const auto token = Common::deviceTokens( *network, { 128000 } );

                // One untimed step: the first decode builds the cuBLASLt plans.
                (void)network->decode( token, prompt_length );
                network->synchronize();

                const auto start = std::chrono::steady_clock::now();

                for ( int step = 1; step <= kDecodeSteps; ++step )
                {
                    (void)network->decode( token, prompt_length + step );
                }

                network->synchronize();

                const double seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();

                std::cout << std::format( "  prompt {:>5}, {}: {:.1f} tokens/s\n", prompt_length,
                    fused ? "fused   " : "cuBLASLt", kDecodeSteps / seconds ) << std::flush;
            }
        }
    }

    TEST( LlamaLongContextCudaTests, DISABLED_GreedyMatchesHuggingFace_1B )
    {
        if ( getDeviceCount( DeviceType::Cuda ) == 0 || !longContextInputsPresent() )
        {
            GTEST_SKIP() << "Needs a CUDA device, " << oneBillionWeightsPath().string() << " and "
                << longContextReferencePath().string();
        }

        Serialization::WeightsReader reference( longContextReferencePath() );

        const std::vector<std::int32_t> prompt = readTokens( reference, "greedy_prompt" );
        const std::vector<std::int32_t> expected = readTokens( reference, "greedy_tokens" );

        constexpr dim_t kContextLength = 1024;

        auto network = buildMeasuredLlama1B( true, kContextLength );

        // <|end_of_text|>: the base model's stop token, which HuggingFace's continuation keeps and Mila's does not.
        const Common::GreedyContinuation greedy = Common::greedyContinuationOf(
            *network, prompt, static_cast<int>( expected.size() ), { 128001 }, kContextLength );

        ASSERT_LE( greedy.tokens.size(), expected.size() );
        ASSERT_GE( greedy.tokens.size() + 1, expected.size() ) << "Mila stopped early";

        for ( std::size_t index = 0; index < greedy.tokens.size(); ++index )
        {
            EXPECT_EQ( greedy.tokens[ index ], expected[ index ] ) << "greedy divergence at generated token " << index;
        }
    }
}
