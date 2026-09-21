/**
 * @file GemmaModel.MixtureOfExperts.Fp4.Cuda.cpp
 * @brief Gemma 4 26B-A4B loaded at FP4: its footprint against the driver and Specifications/MixtureOfExperts.md s8, and its greedy tokens against HuggingFace.
 *
 * Quantizes the 47 GiB BF16 weights on load, so it needs the weights and a 16 GiB card and never runs in CI. The
 * HuggingFace tokens come from `Tools/Converters/Gemma/gemma_4_26b_moe/hf_gemma_layer_stream.py --generate`.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Common/CudaDeviceScope.h"

import Mila;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace fs = std::filesystem;

    namespace
    {
        using GemmaCudaBf16 = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;
        using ExpertBankType = MixtureOfExperts<DeviceType::Cuda, TensorDataType::BF16, ActivationType::Gelu,
            Mila::Dnn::Quant::Weight::PerGroupFp4<64>>;

        // "The capital of France is" in the instruct chat turn, thinking off -- the prompt of the BF16 parity gate.
        const std::vector<int32_t> kPromptIds = { 2, 105, 2364, 107, 818, 5279, 529, 7001, 563, 106, 107, 105, 4368, 107, 100, 45518, 107, 101 };

        // Pasted from `hf_gemma_layer_stream.py --generate 8` (BF16, RTX 4070): "The capital of France is **Paris**.".
        // The narrowest top-1 margin is 4.25 logits, at the first token.
        const std::vector<int32_t> kExpectedGen = { 818, 5279, 529, 7001, 563, 5213, 50429, 84750 };

        constexpr dim_t kContextLength = 8192;
        constexpr dim_t kLayers = 30;

        // The packed PerGroupFp4<64> bank of one layer, from the layout alone, each of its four allocations rounded up
        // to the 2 MiB granularity of both cards (MemoryFootprint.md 11.8): gate_up_proj [128, 1408, 1408] bytes
        // (253,755,392, already a multiple) + [128, 1408, 44] FP32 scales (31,719,424 -> 33,554,432), and down_proj
        // [128, 2816, 352] (126,877,696 -> 127,926,272) + [128, 2816, 11] (15,859,712 -> 16,777,216).
        constexpr std::size_t kLayerBankBytes = 432'013'312;
        constexpr std::size_t kLayerInactiveBytes = kLayerBankBytes / 128 * 120;

        // MixtureOfExperts.md s8, the PerGroupFp4<64> row: bank 11.96 + Linears 0.86 + FP8 table 0.69 + router 0.02.
        constexpr double kSection8WeightsGiB = 13.54;

        // Below this much free after the load, cudaMemGetInfo has stopped measuring: WDDM places the rest in host
        // memory and reports the card as full, so "consumed" saturates at what was free.
        constexpr std::size_t kSaturatedFreeBytes = std::size_t{ 256 } * 1024 * 1024;

        fs::path weightsPath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_26b_a4b_it_bf16.bin";
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

    class GemmaMixtureOfExpertsFp4CudaTests : public ::testing::Test
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

        fs::path weights_;
    };

    // One load serves both halves: quantizing 47 GiB on load is the expensive part, and the footprint it leaves is
    // the one generation runs in.
    //
    // It runs on whatever device is current. The s8 row it checks against was measured on a 16 GiB
    // card; a card with materially less will not hold this model and the load is what says so.
    TEST_F( GemmaMixtureOfExpertsFp4CudaTests, Fp4Load_FitsSection8AndMatchesHuggingFaceGreedy )
    {
        int ordinal = 0;
        ASSERT_EQ( cudaGetDevice( &ordinal ), cudaSuccess );
        const DeviceId device{ DeviceType::Cuda, ordinal };
        const Common::ScopedCurrentCudaDevice current( ordinal );

        ASSERT_TRUE( current.selected() );

        cudaFree( nullptr );

        // One layer's bank, asked of the component before any model exists, so a model-level mismatch is attributable
        // to the bank or to what surrounds it.
        {
            ExpertBankType bank( "probe.experts", MixtureOfExpertsConfig( 2816, 704, 128, 8 ), device );
            const MemoryStats layer = bank.getRequiredMemory(
                BuildContext( shape_t{ 1, 1, 2816 }, RuntimeMode::Inference, false ) );

            std::cout << std::format( "[fp4] one bank  parameters {} bytes  inactive {} bytes  (layout {} / {})\n",
                layer.device_parameter_bytes, layer.device_inactive_parameter_bytes, kLayerBankBytes, kLayerInactiveBytes )
                << std::flush;

            EXPECT_EQ( layer.device_parameter_bytes, kLayerBankBytes );
            EXPECT_EQ( layer.device_inactive_parameter_bytes, kLayerInactiveBytes );
        }

        GemmaModelConfig config;
        config.withContextLength( kContextLength )
            .withWeightQuantization( WeightQuantization::FP4 );

        const DeploymentFootprint footprint = GemmaCudaBf16::getDeploymentFootprint( weights_, config, device );
        const MemoryStats& predicted = footprint.memory;
        const std::size_t free_before = freeDeviceBytes();

        // The prefill chunk shrinks to fit the device; the weights cannot. Past this, a device that holds the weights
        // is one the model must fit, which the last assertion holds.
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

        std::cout << std::format( "[fp4] CUDA device {}\n", ordinal );

        std::cout << std::format(
            "[fp4] context {}  prefill chunk {} of {} rows  free before load {:.3f} GiB  free after {:.3f} GiB{}\n"
            "  weights   reported {:.3f} GiB  (s8 row {:.2f})  inactive {:.3f} GiB  active {:.3f} GiB\n"
            "  state     reported {:.3f} GiB  ({} bytes)  predicted total {} bytes\n"
            "  predicted {:.3f} GiB  reported {:.3f} GiB  consumed {:.3f} GiB  residual {:.3f} GiB  scratch {:.3f} GiB\n",
            kContextLength, footprint.prefill.chunk_rows, footprint.prefill.unconstrained_chunk_rows,
            toGiB( free_before ), toGiB( free_after_load ), saturated ? "  (SATURATED)" : "",
            toGiB( reported.device_parameter_bytes ), kSection8WeightsGiB,
            toGiB( reported.device_inactive_parameter_bytes ), toGiB( reported.activeDeviceParameterBytes() ),
            toGiB( reported.device_state_bytes ), reported.device_state_bytes, predicted.totalDeviceBytes(),
            toGiB( predicted.totalDeviceBytes() ), toGiB( reported.totalDeviceBytes() ), toGiB( consumed ),
            toGiB( residual ), toGiB( reported.device_scratch_bytes ) ) << std::flush;

        EXPECT_EQ( reported.device_inactive_parameter_bytes, kLayers * kLayerInactiveBytes );

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
        generate_params.max_new_tokens = kExpectedGen.empty() ? 8 : static_cast<int>( kExpectedGen.size() );
        generate_params.sampling.temperature = 0.0f;
        generate_params.sampling.top_k = 1;

        std::vector<int32_t> generated;
        [[maybe_unused]] const auto status = model->generate(
            kPromptIds, [&]( int32_t token ) { generated.push_back( token ); }, generate_params );

        std::string line = "[fp4] generated:";

        for ( const int32_t token : generated )
            line += std::format( " {}", token );

        std::cout << line << "\n" << std::flush;

        if ( kExpectedGen.empty() )
            GTEST_SKIP() << "Populate kExpectedGen from hf_gemma_layer_stream.py --generate";

        // Mila omits a trailing stop token that HuggingFace's list includes, so Mila's tokens are a prefix of it.
        ASSERT_LE( generated.size(), kExpectedGen.size() );
        ASSERT_GE( generated.size() + 1, kExpectedGen.size() ) << "Mila stopped early";

        for ( size_t i = 0; i < generated.size(); ++i )
        {
            EXPECT_EQ( generated[ i ], kExpectedGen[ i ] ) << "greedy divergence at generated token " << i;
        }

        // Fitting is judged on the prediction, because consumption cannot be read past a full card. The prefill chunk
        // rule picks the largest rung that fits the free memory, so a card this model fits at all is left nearly full.
        EXPECT_LT( predicted.totalDeviceBytes(), free_before ) << std::format(
            "does not fit this card: {} bytes predicted against {} free", predicted.totalDeviceBytes(), free_before );
    }
}
