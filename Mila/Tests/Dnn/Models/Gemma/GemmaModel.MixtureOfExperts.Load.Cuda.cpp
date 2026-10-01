/**
 * @file GemmaModel.MixtureOfExperts.Load.Cuda.cpp
 * @brief Gemma 4 26B-A4B loaded at FP4 and at Q4_0: its footprint against the driver and Specifications/Gemma.md s10.5,
 * and its greedy tokens against HuggingFace.
 *
 * Quantizes the 47 GiB BF16 conversion of Google's quantization-aware checkpoint on load, so it needs the weights and
 * a 16 GiB card and never runs in CI. The HuggingFace tokens come from
 * `Tools/Converters/Gemma/gemma_4_26b_moe/hf_gemma_layer_stream.py --generate` on the same checkpoint.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
// C stdio rather than <fstream>: an input-stream header in a TU that does `import Mila;` leaves
// std::basic_istream::sentry incomplete.
#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Common/CudaDeviceScope.h"

import Mila;

#include "Common/LogLikelihoodHarness.h"

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace fs = std::filesystem;

    namespace
    {
        using GemmaCudaBf16 = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;

        template<typename TWeightQuantization>
        using ExpertBankType = MixtureOfExperts<DeviceType::Cuda, TensorDataType::BF16, ActivationType::Gelu, TWeightQuantization>;

        // "The capital of France is" in the instruct chat turn, thinking off -- the prompt of the BF16 parity gate.
        const std::vector<int32_t> kPromptIds = { 2, 105, 2364, 107, 818, 5279, 529, 7001, 563, 106, 107, 105, 4368, 107, 100, 45518, 107, 101 };

        // Pasted from `hf_gemma_layer_stream.py --generate 8` on the quantization-aware checkpoint (BF16, RTX 5060 Ti,
        // 2026-09-30): "The capital of France is **Paris**.". The narrowest top-1 margin is 12.1 logits, at the first
        // token; the non-quantization-aware checkpoint gave the same tokens at a narrowest margin of 4.25.
        const std::vector<int32_t> kExpectedGen = { 818, 5279, 529, 7001, 563, 5213, 50429, 84750 };

        constexpr dim_t kContextLength = 8192;
        constexpr dim_t kLayers = 30;

        // One layer's packed bank, from the layout alone, each of its four allocations rounded up to the 2 MiB
        // granularity of both cards (MemoryFootprint.md 11.8). PerGroupFp4<64>: gate_up_proj [128, 1408, 1408] bytes
        // (253,755,392, already a multiple) + [128, 1408, 44] FP32 scales (31,719,424 -> 33,554,432), and down_proj
        // [128, 2816, 352] (126,877,696 -> 127,926,272) + [128, 2816, 11] (15,859,712 -> 16,777,216). Q4_0's FP16
        // scales per 32 are the same bytes as FP32 scales per 64, so both policies give this figure.
        constexpr std::size_t kLayerBankBytes = 432'013'312;
        constexpr std::size_t kLayerInactiveBytes = kLayerBankBytes / 128 * 120;

        // Gemma.md s10.5, the PerGroupFp4<64> row: bank 11.96 + Linears 0.86 + FP8 table 0.69 + router 0.02. The Q4_0
        // row is the same bytes.
        constexpr double kSection8WeightsGiB = 13.54;

        // Below this much free after the load, cudaMemGetInfo has stopped measuring: WDDM places the rest in host
        // memory and reports the card as full, so "consumed" saturates at what was free.
        constexpr std::size_t kSaturatedFreeBytes = std::size_t{ 256 } * 1024 * 1024;

        fs::path weightsPath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_26b_a4b_it_qat_bf16.bin";
        }

        std::size_t freeDeviceBytes()
        {
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;

            if ( cudaMemGetInfo( &free_bytes, &total_bytes ) != cudaSuccess )
            {
                return 0;
            }

            return free_bytes;
        }

        double toGiB( std::size_t bytes )
        {
            return static_cast<double>( bytes ) / ( 1024.0 * 1024.0 * 1024.0 );
        }
    }

    class GemmaMixtureOfExpertsLoadCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
                GTEST_SKIP() << "No CUDA device available";

            weights_ = weightsPath();

            if ( !fs::exists( weights_ ) )
                GTEST_SKIP() << "Not present: " << weights_.string();
        }

        // One load serves both halves: quantizing 47 GiB on load is the expensive part, and the footprint it leaves is
        // the one generation runs in.
        //
        // It runs on whatever device is current. The s8 row it checks against was measured on a 16 GiB card; a card
        // with materially less will not hold this model and the load is what says so.
        template<typename TWeightQuantization>
        void loadFitAndGenerate( WeightQuantization quantization, const char* label )
        {
            int ordinal = 0;
            ASSERT_EQ( cudaGetDevice( &ordinal ), cudaSuccess );
            const DeviceId device{ DeviceType::Cuda, ordinal };
            const Common::ScopedCurrentCudaDevice current( ordinal );

            ASSERT_TRUE( current.selected() );

            cudaFree( nullptr );

            // One layer's bank, asked of the component before any model exists, so a model-level mismatch is
            // attributable to the bank or to what surrounds it.
            {
                ExpertBankType<TWeightQuantization> bank( "probe.experts", MixtureOfExpertsConfig( 2816, 704, 128, 8 ), device );
                const MemoryStats layer = bank.getRequiredMemory(
                    BuildContext( shape_t{ 1, 1, 2816 }, RuntimeMode::Inference, false )
                        .withAllocationGranularity( allocationGranularity( device ) ) );

                std::cout << std::format( "[{}] one bank  parameters {} bytes  inactive {} bytes  (layout {} / {})\n",
                    label, layer.device_parameter_bytes, layer.device_inactive_parameter_bytes, kLayerBankBytes,
                    kLayerInactiveBytes ) << std::flush;

                EXPECT_EQ( layer.device_parameter_bytes, kLayerBankBytes );
                EXPECT_EQ( layer.device_inactive_parameter_bytes, kLayerInactiveBytes );
            }

            GemmaModelConfig config;
            config.withContextLength( kContextLength )
                .withWeightQuantization( quantization );

            const DeploymentFootprint footprint = GemmaCudaBf16::getDeploymentFootprint( weights_, config, device );
            const MemoryStats& predicted = footprint.memory;
            const std::size_t free_before = freeDeviceBytes();

            // The prefill chunk shrinks to fit the device; the weights cannot. Past this, a device that holds the
            // weights is one the model must fit, which the last assertion holds.
            if ( predicted.device_parameter_bytes >= free_before )
            {
                GTEST_SKIP() << std::format( "weights need {} bytes and CUDA device {} has {} free",
                    predicted.device_parameter_bytes, ordinal, free_before );
            }

            auto model = GemmaCudaBf16::load( weights_, config, device );
            ASSERT_NE( model, nullptr );

            const std::size_t free_after_load = freeDeviceBytes();
            const std::size_t consumed = free_before - free_after_load;
            const bool saturated = free_after_load < kSaturatedFreeBytes;
            const MemoryStats reported = model->getMemoryStats();
            const std::size_t residual = consumed > predicted.totalDeviceBytes() ? consumed - predicted.totalDeviceBytes() : 0;

            std::cout << std::format( "[{}] CUDA device {}  scheme {}\n", label, ordinal, model->weightQuantizationScheme() );

            std::cout << std::format(
                "[{}] context {}  prefill chunk {} of {} rows  free before load {:.3f} GiB  free after {:.3f} GiB{}\n"
                "  weights   reported {:.3f} GiB  (s8 row {:.2f})  inactive {:.3f} GiB  active {:.3f} GiB\n"
                "  state     reported {:.3f} GiB  ({} bytes)  predicted total {} bytes\n"
                "  predicted {:.3f} GiB  reported {:.3f} GiB  consumed {:.3f} GiB  residual {:.3f} GiB  scratch {:.3f} GiB\n",
                label, kContextLength, footprint.prefill.chunk_rows, footprint.prefill.unconstrained_chunk_rows,
                toGiB( free_before ), toGiB( free_after_load ), saturated ? "  (SATURATED)" : "",
                toGiB( reported.device_parameter_bytes ), kSection8WeightsGiB,
                toGiB( reported.device_inactive_parameter_bytes ), toGiB( reported.activeDeviceParameterBytes() ),
                toGiB( reported.device_state_bytes ), reported.device_state_bytes, predicted.totalDeviceBytes(),
                toGiB( predicted.totalDeviceBytes() ), toGiB( reported.totalDeviceBytes() ), toGiB( consumed ),
                toGiB( residual ), toGiB( reported.device_scratch_bytes ) ) << std::flush;

            EXPECT_EQ( reported.device_inactive_parameter_bytes, kLayers * kLayerInactiveBytes );

            // ModelFamilyParity.md 8.2, G5: at 8192 the 16 GB card plans the chunk the 12B does, not one narrowed to fit.
            EXPECT_EQ( footprint.prefill.chunk_rows, footprint.prefill.unconstrained_chunk_rows )
                << "the prefill chunk was narrowed to fit the device";

            const double weights_ratio = toGiB( reported.device_parameter_bytes ) / kSection8WeightsGiB;
            EXPECT_GT( weights_ratio, 0.97 ) << "weights below the s8 row";
            EXPECT_LT( weights_ratio, 1.03 ) << "weights above the s8 row";

            EXPECT_EQ( predicted.device_parameter_bytes, reported.device_parameter_bytes );
            EXPECT_EQ( predicted.device_state_bytes, reported.device_state_bytes );
            EXPECT_EQ( predicted.device_scratch_bytes, reported.device_scratch_bytes );

            if ( !saturated )
            {
                EXPECT_LE( predicted.totalDeviceBytes(), consumed ) << "prediction exceeded actual consumption";
                EXPECT_LT( residual, consumed / 4 ) << "unmodelled memory exceeded 25% of what was consumed";
            }

            GenerateParams generate_params;
            generate_params.max_new_tokens = static_cast<int>( kExpectedGen.size() );
            generate_params.sampling.temperature = 0.0f;
            generate_params.sampling.top_k = 1;

            std::vector<int32_t> generated;
            [[maybe_unused]] const auto status = model->generate(
                kPromptIds, [&]( int32_t token ) { generated.push_back( token ); }, generate_params );

            std::string line = std::format( "[{}] generated:", label );

            for ( const int32_t token : generated )
                line += std::format( " {}", token );

            std::cout << line << "\n" << std::flush;

            // Mila omits a trailing stop token that HuggingFace's list includes, so Mila's tokens are a prefix of it.
            ASSERT_LE( generated.size(), kExpectedGen.size() );
            ASSERT_GE( generated.size() + 1, kExpectedGen.size() ) << "Mila stopped early";

            for ( size_t i = 0; i < generated.size(); ++i )
            {
                EXPECT_EQ( generated[ i ], kExpectedGen[ i ] ) << "greedy divergence at generated token " << i;
            }

            // Fitting is judged on the prediction, because consumption cannot be read past a full card. The prefill
            // chunk rule picks the largest rung that fits the free memory, so a card this model fits at all is left
            // nearly full.
            EXPECT_LT( predicted.totalDeviceBytes(), free_before ) << std::format(
                "does not fit this card: {} bytes predicted against {} free", predicted.totalDeviceBytes(), free_before );
        }

        fs::path weights_;
    };

    TEST_F( GemmaMixtureOfExpertsLoadCudaTests, Fp4Load_FitsSection8AndMatchesHuggingFaceGreedy )
    {
        loadFitAndGenerate<Mila::Dnn::Quant::Weight::PerGroupFp4<64>>( WeightQuantization::FP4, "fp4" );
    }

    // Gemma4MoE.md Phase 9, G5 item 7: the trained format.
    TEST_F( GemmaMixtureOfExpertsLoadCudaTests, Q4_0Load_FitsSection8AndMatchesHuggingFaceGreedy )
    {
        loadFitAndGenerate<Mila::Dnn::Quant::Weight::PerGroupInt4<32>>( WeightQuantization::Q4_0, "q4_0" );
    }

    // Diagnostic: the network's predicted footprint over context lengths and prefill chunks, allocating nothing,
    // against the free memory the RTX 5060 Ti reported after a 26B load's context was created (15,904,800,768 bytes,
    // the refusals of 2026-09-30). Differences along a row are the per-token terms; down a column, the per-chunk ones.
    TEST_F( GemmaMixtureOfExpertsLoadCudaTests, DISABLED_Q4_0_PredictedFootprintByContextAndChunk )
    {
        using RoutedQ4_0 = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16, Mila::Dnn::Quant::Weight::PerGroupInt4<32>,
            GemmaCudaBf16::GemmaSlidingKvPolicy, GemmaFeedForward::Routed>;
        using RoutedQ4_0Fp8Global = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16, Mila::Dnn::Quant::Weight::PerGroupInt4<32>,
            GemmaCudaBf16::GemmaSlidingKvPolicy, GemmaFeedForward::Routed, Mila::Dnn::Quant::KvCache::PerTokenKvFp8<>>;
        using RoutedQ4_0Fp8 = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16, Mila::Dnn::Quant::Weight::PerGroupInt4<32>,
            Mila::Dnn::Quant::KvCache::SlidingWindowKvFp8, GemmaFeedForward::Routed, Mila::Dnn::Quant::KvCache::PerTokenKvFp8<>>;

        constexpr std::size_t kFreeBytes = 15'904'800'768;
        constexpr double kMiB = 1024.0 * 1024.0;

        Serialization::WeightsReader reader( weights_ );
        GemmaConfig config = GemmaCudaBf16::configFromMetadata( reader.getWeightsMetadata() );
        const DeviceId device{ DeviceType::Cuda, 0 };
        const RoutedQ4_0 network( reader.getWeightsMetadata().model_name, config, device );

        const auto print = [&]( const auto& priced, const char* caches )
        {
            for ( const dim_t chunk : { dim_t{ 1024 }, dim_t{ 128 } } )
            {
                for ( const dim_t context : { dim_t{ 8192 }, dim_t{ 16384 }, dim_t{ 32768 }, dim_t{ 65536 }, dim_t{ 98304 } } )
                {
                    const MemoryStats stats = priced.getRequiredMemory(
                        BuildContext( shape_t{ 1, context }, RuntimeMode::Inference, false )
                            .withAllocationGranularity( allocationGranularity( device ) )
                            .withPrefillSize( chunk ) );
                    const std::size_t total = stats.totalDeviceBytes();

                    std::cout << std::format(
                        "[footprint] {:<10} chunk {:>4} context {:>6}: parameters {:>8.1f} MiB  state {:>7.1f} MiB  "
                        "scratch {:>5.1f} MiB  total {:>8.1f} MiB  {}\n",
                        caches, chunk, context, stats.device_parameter_bytes / kMiB, stats.device_state_bytes / kMiB,
                        stats.device_scratch_bytes / kMiB, total / kMiB,
                        total <= kFreeBytes ? std::format( "fits, {:.0f} MiB spare", ( kFreeBytes - total ) / kMiB )
                                            : std::format( "over by {:.0f} MiB", ( total - kFreeBytes ) / kMiB ) ) << std::flush;
                }
            }
        };

        print( network, "bf16" );
        print( RoutedQ4_0Fp8Global( reader.getWeightsMetadata().model_name, config, device ), "fp8-global" );
        print( RoutedQ4_0Fp8( reader.getWeightsMetadata().model_name, config, device ), "fp8" );

        // Per child, with the context GemmaTransformer::requiredMemoryAtChunk gives a block, at a 1024-row chunk:
        // what each costs at 8K and what 8K more tokens add.
        const auto blockContext = [&]( dim_t context )
        {
            return BuildContext( shape_t{ 1, context }, RuntimeMode::Inference, false )
                .withAllocationGranularity( allocationGranularity( device ) )
                .forChild( shape_t{ 1, context, config.getModelDim() } )
                .withPrefillSize( 1024 )
                .withInstalledOutput( true )
                .withFusedDecode( true );
        };

        std::size_t global_growth = 0;
        std::size_t local_growth = 0;
        std::size_t global_first = 0;
        std::size_t local_first = 0;
        int layer = 0;

        for ( const auto& child : network.getComponents() )
        {
            if ( child->getName().find( ".tf_layer_" ) == std::string::npos )
                continue;

            const std::size_t at8k = child->getRequiredMemory( blockContext( 8192 ) ).device_state_bytes;
            const std::size_t at16k = child->getRequiredMemory( blockContext( 16384 ) ).device_state_bytes;
            const bool global = config.isGlobalLayer( static_cast<dim_t>( layer++ ) );

            ( global ? global_growth : local_growth ) += at16k - at8k;
            ( global ? global_first : local_first ) = ( global ? global_first : local_first ) == 0 ? at8k : ( global ? global_first : local_first );
        }

        std::cout << std::format(
            "[footprint] one global block at 8K: state {:.1f} MiB; the global blocks add {:.0f} bytes a token\n"
            "[footprint] one local block at 8K: state {:.1f} MiB; the local blocks add {:.0f} bytes a token\n",
            global_first / kMiB, static_cast<double>( global_growth ) / 8192.0,
            local_first / kMiB, static_cast<double>( local_growth ) / 8192.0 ) << std::flush;
    }

    // Diagnostic, G5b: the same wikitext segments scored through the planner's prefill chunk -- the grouped INT8
    // prefill and Linear's INT8 GEMM -- and one row at a time, where every projection takes the BF16-activation
    // matvec and the bank its gather decode. The two differ by what INT8 activations do, which raw wikitext cannot
    // grade (ModelFamilyParity.md 8.2, G1); a broken prefill it does show. Printed, not gated.
    TEST_F( GemmaMixtureOfExpertsLoadCudaTests, DISABLED_Q4_0_ChunkedPrefillAgainstOneRowAtATime_Wikitext )
    {
        using RoutedQ4_0 = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16, Mila::Dnn::Quant::Weight::PerGroupInt4<32>,
            GemmaCudaBf16::GemmaSlidingKvPolicy, GemmaFeedForward::Routed>;

        constexpr dim_t kSegmentLength = 2048;
        constexpr int kSegments = 4;
        constexpr std::int32_t kBos = 2;

        const fs::path tokenizer = fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma_tokenizer.bin";
        const fs::path corpus = fs::path( TEST_DATA_DIR ).parent_path() / "Mila" / "Tools" / "Quantization" / "corpus" / "wiki.test.raw";

        if ( !fs::exists( tokenizer ) || !fs::exists( corpus ) )
            GTEST_SKIP() << "Needs " << tokenizer.string() << " and " << corpus.string();

        std::vector<std::int32_t> text;
        {
            std::FILE* file = std::fopen( corpus.string().c_str(), "rb" );
            ASSERT_NE( file, nullptr );

            // Raw wikitext runs past four characters a token; six over-reads.
            std::string raw( static_cast<std::size_t>( kSegments * kSegmentLength ) * 6, '\0' );
            raw.resize( std::fread( raw.data(), 1, raw.size(), file ) );
            std::fclose( file );

            text = Mila::Data::BpeTokenizer::loadGemma( tokenizer )->encode( raw );
        }

        ASSERT_GE( text.size(), static_cast<std::size_t>( kSegments * ( kSegmentLength - 1 ) ) );

        GemmaConfig config = GemmaCudaBf16::configFromMetadata( Serialization::WeightsReader( weights_ ).getWeightsMetadata() );
        config.withLogLikelihoodWindow( 1 );

        const DeviceId device{ DeviceType::Cuda, 0 };

        // Each segment is <bos> and 2047 text tokens from a cold cache, as G1 scores.
        auto score = [&]( std::optional<dim_t> chunk_rows, const char* label )
        {
            Serialization::WeightsReader reader( weights_ );
            auto network = std::make_unique<RoutedQ4_0>( reader.getWeightsMetadata().model_name, config, device );

            const Mila::Deployment::DeviceReading reading = Mila::Deployment::DeviceReading::take( device );
            const BuildContext context = BuildContext( shape_t{ 1, kSegmentLength }, RuntimeMode::Inference, false )
                .withAllocationGranularity( reading.allocation_granularity );
            const dim_t rows = chunk_rows.value_or(
                Mila::Deployment::choosePrefillChunk( *network, context, reading.free_bytes ).chunk_rows );

            network->build( context.withPrefillSize( rows ) );
            network->loadParameters( reader );

            double total = 0.0;
            dim_t positions = 0;

            for ( int segment = 0; segment < kSegments; ++segment )
            {
                std::vector<std::int32_t> tokens{ kBos };
                const auto first = text.begin() + static_cast<std::ptrdiff_t>( segment ) * ( kSegmentLength - 1 );
                tokens.insert( tokens.end(), first, first + ( kSegmentLength - 1 ) );

                const SequenceLogLikelihood scored = Common::sequenceLogLikelihoodOf( *network, tokens );
                total += scored.total_log_probability;
                positions += scored.scored_positions;
            }

            const double mean_nats = -total / static_cast<double>( positions );

            std::cout << std::format( "[q4_0 {}] prefill chunk {}: {} positions, mean {:.5f} nats, perplexity {:.3f}\n",
                label, rows, positions, mean_nats, std::exp( mean_nats ) ) << std::flush;

            return mean_nats;
        };

        const double chunked = score( std::nullopt, "chunked" );
        const double one_row = score( dim_t{ 1 }, "one row" );

        std::cout << std::format( "[q4_0] chunked against one row at a time: {:+.5f} nats ({:+.3f}%)\n",
            chunked - one_row, 100.0 * ( chunked - one_row ) / one_row ) << std::flush;
    }
}
