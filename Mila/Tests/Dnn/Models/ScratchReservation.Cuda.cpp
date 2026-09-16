/**
 * @file ScratchReservation.Cuda.cpp
 * @brief A built model reserves the forward scratch its footprint predicts, and generation stays inside it.
 *
 * Specifications/MemoryFootprint.md Phase 6 step 2, criteria 1 and 3. Needs exported weights of each model,
 * so it never runs in CI.
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

#include "Common/DeviceWithoutDisplay.h"

import Mila;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace fs = std::filesystem;

    namespace
    {
        constexpr int kDecodeSteps = 8;

        // Under the run-to-run noise in consumed memory (MemoryFootprint.md 6.4), and under the smallest
        // scratch any of these models reserves, so growth past it is scratch allocated during generation.
        constexpr std::size_t kGenerationGrowthBytes = std::size_t{ 16 } * 1024 * 1024;

        fs::path modelsDirectory()
        {
            return fs::path( TEST_DATA_DIR ) / "models";
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

        // A prompt that fills one prefill chunk makes every operation request its largest scratch, and
        // the decode steps add the fused decode attention's request. Any request above the reservation
        // throws inside generate(); a scratch allocated during generation instead shows as growth.
        //
        // Growth is measured on a device that drives no display. On one that does, Windows lowers the
        // process's video memory budget as it writes (MemoryFootprint.md 11.5), and free memory moved
        // 29.1 MiB (cb2-3) and 47.6 MiB (Llama 3.1 8B) during generation on the RTX 4070 with nothing in
        // Mila allocating.
        template<typename TModel, typename TConfig>
        void expectScratchReservedAndSufficient( const std::string& label, const fs::path& weights, const TConfig& config )
        {
            if ( !fs::exists( weights ) )
            {
                GTEST_SKIP() << "Not present: " << weights.string();
            }

            const std::optional<int> without_display = Common::findCudaDeviceWithoutDisplay();
            const int ordinal = without_display.value_or( 0 );
            const DeviceId device{ DeviceType::Cuda, ordinal };
            const Common::ScopedCurrentCudaDevice current( ordinal );

            ASSERT_TRUE( current.selected() );

            cudaFree( nullptr );

            const DeploymentFootprint footprint = TModel::getDeploymentFootprint( weights, config, device );
            const std::size_t free_before_load = freeDeviceBytes();

            // The prefill chunk shrinks to fit the device; the weights cannot.
            if ( footprint.memory.device_parameter_bytes >= free_before_load )
            {
                GTEST_SKIP() << std::format( "weights need {} bytes and CUDA device {} has {} free",
                    footprint.memory.device_parameter_bytes, ordinal, free_before_load );
            }

            auto model = TModel::fromPretrained( weights, config, device );
            ASSERT_NE( model, nullptr );

            const MemoryStats reported = model->getMemoryStats();

            std::cout << std::format( "[{}] prefill chunk {}  scratch predicted {} bytes  reported {} bytes\n",
                label, footprint.prefill.chunk_rows, footprint.memory.device_scratch_bytes,
                reported.device_scratch_bytes ) << std::flush;

            EXPECT_EQ( footprint.memory.device_scratch_bytes, reported.device_scratch_bytes );
            EXPECT_GT( reported.device_scratch_bytes, std::size_t{ 0 } );

            std::vector<int32_t> prompt( static_cast<std::size_t>( footprint.prefill.chunk_rows ) );

            for ( std::size_t index = 0; index < prompt.size(); ++index )
            {
                prompt[ index ] = static_cast<int32_t>( 1000 + index % 7000 );
            }

            GenerateParams params;
            params.max_new_tokens = kDecodeSteps;
            params.sampling.temperature = 0.0f;
            params.sampling.top_k = 1;

            std::size_t generated = 0;
            const std::size_t free_before_generation = freeDeviceBytes();

            EXPECT_NO_THROW( {
                [[maybe_unused]] const auto status = model->generate(
                    prompt, [&]( int32_t ) { ++generated; }, params );
            } );

            const std::size_t free_after_generation = freeDeviceBytes();
            const std::size_t growth = free_before_generation > free_after_generation
                ? free_before_generation - free_after_generation : 0;

            std::cout << std::format( "[{}] CUDA device {}{}  device memory growth during generation {:.1f} MiB  "
                "free after {:.1f} MiB\n", label, ordinal, without_display ? "" : " (drives a display)",
                static_cast<double>( growth ) / ( 1024.0 * 1024.0 ),
                static_cast<double>( free_after_generation ) / ( 1024.0 * 1024.0 ) ) << std::flush;

            EXPECT_GT( generated, std::size_t{ 0 } );

            if ( !without_display )
            {
                GTEST_SKIP() << "every visible CUDA device drives a display, so growth measures the Windows "
                                "budget as well as Mila; measured " << growth << " bytes";
            }

            // A process that sees one GPU spills past the card rather than failing, and a card with nothing left
            // reports no growth however much is allocated.
            if ( free_after_generation < kGenerationGrowthBytes )
            {
                GTEST_SKIP() << std::format( "CUDA device {} is saturated ({} bytes free), so growth cannot be read",
                    ordinal, free_after_generation );
            }

            EXPECT_LT( growth, kGenerationGrowthBytes ) << std::format(
                "device memory grew {} bytes during generation: the scratch was not reserved at build", growth );
        }

        GemmaModelConfig gemmaConfig( dim_t context_length )
        {
            GemmaModelConfig config;
            config.withContextLength( context_length ).withWeightQuantization( WeightQuantization::FP4 );

            return config;
        }
    }

    TEST( ScratchReservationCudaTests, Gemma4_12B_Fp4_Context8192 )
    {
        expectScratchReservedAndSufficient<GemmaModel<DeviceType::Cuda, TensorDataType::BF16>>(
            "gemma 4 12b fp4, context 8192", modelsDirectory() / "gemma" / "gemma4_12b_it_fp4.safetensors",
            gemmaConfig( 8192 ) );
    }

    // A context of 512 caps the prefill chunk at 512 rows where 8192 takes 1024, so the FP8-activation
    // request comes from a different row bucket.
    TEST( ScratchReservationCudaTests, Gemma4_12B_Fp4_Context512 )
    {
        expectScratchReservedAndSufficient<GemmaModel<DeviceType::Cuda, TensorDataType::BF16>>(
            "gemma 4 12b fp4, context 512", modelsDirectory() / "gemma" / "gemma4_12b_it_fp4.safetensors",
            gemmaConfig( 512 ) );
    }

    TEST( ScratchReservationCudaTests, Llama31_8B_Fp4_Context8192 )
    {
        LlamaModelConfig config( 8192 );
        config.withWeightQuantization( WeightQuantization::FP4 );

        expectScratchReservedAndSufficient<LlamaModel<DeviceType::Cuda, TensorDataType::BF16>>(
            "llama 3.1 8b fp4, context 8192", modelsDirectory() / "llama" / "llama31_8b_instruct_fp4.safetensors",
            config );
    }

    // About 15 GiB of weights, so it runs only where a device that large is visible.
    TEST( ScratchReservationCudaTests, Qwen38_27B_Fp4_Context8192 )
    {
        QwenModelConfig config( 8192 );
        config.withWeightQuantization( WeightQuantization::FP4 );

        expectScratchReservedAndSufficient<QwenModel<DeviceType::Cuda, TensorDataType::BF16>>(
            "qwen 3.8 27b fp4, context 8192", modelsDirectory() / "qwen" / "qwen38_27b_fp4.safetensors", config );
    }

    TEST( ScratchReservationCudaTests, Qwen38_27B_Codebook_Context4096 )
    {
        QwenModelConfig config( 4096 );
        config.withPrecisionPlan();

        expectScratchReservedAndSufficient<QwenModel<DeviceType::Cuda, TensorDataType::BF16>>(
            "qwen 3.8 27b cb2-3, context 4096", modelsDirectory() / "qwen" / "qwen38_27b_cb2-3.safetensors", config );
    }
}
