/**
 * @file QuantizeOnLoad.Footprint.Cuda.cpp
 * @brief A model fitted on load from BF16 weights settles at the device footprint of its exported form.
 *
 * Specifications/MemoryFootprint.md Phase 6 step 1, criterion 3. Needs both the BF16 weights and the exported
 * FP4 weights of each model, so it never runs in CI.
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
        constexpr dim_t kContextLength = 8192;
        constexpr int kDecodeSteps = 8;

        // Half the load staging buffer. The run-to-run noise in consumed memory is 50-70 MiB
        // (MemoryFootprint.md 6.4), so a difference under this separates staging returned from staging kept.
        constexpr std::size_t kStagingKeptBytes = std::size_t{ 128 } * 1024 * 1024;

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

        std::size_t freeDeviceBytesOn( int ordinal )
        {
            if ( cudaSetDevice( ordinal ) != cudaSuccess )
            {
                return 0;
            }

            return freeDeviceBytes();
        }

        double toMiB( std::size_t bytes )
        {
            return static_cast<double>( bytes ) / ( 1024.0 * 1024.0 );
        }

        std::size_t difference( std::size_t first, std::size_t second )
        {
            return first > second ? first - second : second - first;
        }

        struct LoadMeasurement
        {
            std::size_t consumed_after_load{ 0 };
            std::size_t consumed_after_generation{ 0 };
            std::vector<int32_t> generated;
        };

        // The prompt fills one prefill chunk, so the forward scratch of a full chunk is in the second reading.
        template<typename TModel, typename TConfig>
        LoadMeasurement measureLoad( const fs::path& weights, TConfig config, int ordinal )
        {
            const DeviceId device{ DeviceType::Cuda, ordinal };
            const DeploymentFootprint footprint = TModel::getDeploymentFootprint( weights, config, device );
            const std::size_t free_before = freeDeviceBytesOn( ordinal );

            LoadMeasurement measurement;

            {
                auto model = TModel::load( weights, config, device );
                measurement.consumed_after_load = free_before - freeDeviceBytesOn( ordinal );

                // The scratch a fitted load reserves is the one its exported form predicts.
                EXPECT_EQ( footprint.memory.device_scratch_bytes, model->getMemoryStats().device_scratch_bytes );

                std::vector<int32_t> prompt( static_cast<std::size_t>( footprint.prefill.chunk_rows ) );

                for ( std::size_t index = 0; index < prompt.size(); ++index )
                {
                    prompt[ index ] = static_cast<int32_t>( 1000 + index % 7000 );
                }

                GenerateParams params;
                params.max_new_tokens = kDecodeSteps;
                params.sampling.temperature = 0.0f;
                params.sampling.top_k = 1;

                [[maybe_unused]] const auto status = model->generate(
                    prompt, [&]( int32_t token ) { measurement.generated.push_back( token ); }, params );

                measurement.consumed_after_generation = free_before - freeDeviceBytesOn( ordinal );
            }

            return measurement;
        }

        template<typename TModel, typename TConfig>
        void expectFittedLoadSettlesAtExport(
            const std::string& label, const fs::path& bf16_weights, const fs::path& exported_weights, TConfig config )
        {
            if ( !fs::exists( bf16_weights ) || !fs::exists( exported_weights ) )
            {
                GTEST_SKIP() << "Not present: " << bf16_weights.string() << " and " << exported_weights.string();
            }

            config.withWeightQuantization( WeightQuantization::FP4 );

            // Consumption is measured on a device that drives no display, for the reason
            // GemmaModel.Footprint.Cuda.cpp gives: on one that does, the Llama 3.1 8B exported arm read
            // 10389.6 MiB after generation in one run and 10679.0 MiB in the next, with Mila unchanged.
            const std::optional<int> without_display = Common::findCudaDeviceWithoutDisplay();
            const int ordinal = without_display.value_or( 0 );
            const Common::ScopedCurrentCudaDevice current( ordinal );

            ASSERT_TRUE( current.selected() );

            const LoadMeasurement exported = measureLoad<TModel>( exported_weights, config, ordinal );
            const LoadMeasurement fitted = measureLoad<TModel>( bf16_weights, config, ordinal );

            std::cout << std::format( "[{}] CUDA device {}{}\n",
                label, ordinal, without_display ? "" : " (drives a display)" );

            std::cout << std::format(
                "[{}] consumed after load: exported {:.1f} MiB, fitted {:.1f} MiB, difference {:.1f} MiB\n"
                "[{}] consumed after generation: exported {:.1f} MiB, fitted {:.1f} MiB, difference {:.1f} MiB\n",
                label, toMiB( exported.consumed_after_load ), toMiB( fitted.consumed_after_load ),
                toMiB( difference( exported.consumed_after_load, fitted.consumed_after_load ) ),
                label, toMiB( exported.consumed_after_generation ), toMiB( fitted.consumed_after_generation ),
                toMiB( difference( exported.consumed_after_generation, fitted.consumed_after_generation ) ) ) << std::flush;

            EXPECT_EQ( fitted.generated, exported.generated );

            if ( !without_display )
            {
                GTEST_SKIP() << "every visible CUDA device drives a display, so consumed memory measures the "
                                "Windows budget as well as Mila";
            }

            EXPECT_LT( difference( exported.consumed_after_load, fitted.consumed_after_load ), kStagingKeptBytes )
                << "a load that fits BF16 weights keeps device memory its exported form does not";
            EXPECT_LT( difference( exported.consumed_after_generation, fitted.consumed_after_generation ), kStagingKeptBytes );
        }
    }

    TEST( QuantizeOnLoadFootprintCudaTests, Llama32_3B_FittedLoadSettlesAtExport )
    {
        expectFittedLoadSettlesAtExport<LlamaModel<DeviceType::Cuda, TensorDataType::BF16>>(
            "llama 3.2 3b",
            modelsDirectory() / "llama" / "llama32_3b_instruct_bf16.bin",
            modelsDirectory() / "llama" / "llama32_3b_instruct_fp4.safetensors",
            LlamaModelConfig( kContextLength ) );
    }

    TEST( QuantizeOnLoadFootprintCudaTests, Llama31_8B_FittedLoadSettlesAtExport )
    {
        expectFittedLoadSettlesAtExport<LlamaModel<DeviceType::Cuda, TensorDataType::BF16>>(
            "llama 3.1 8b",
            modelsDirectory() / "llama" / "llama31_8b_instruct_bf16.bin",
            modelsDirectory() / "llama" / "llama31_8b_instruct_fp4.safetensors",
            LlamaModelConfig( kContextLength ) );
    }

    TEST( QuantizeOnLoadFootprintCudaTests, Gemma4_12B_FittedLoadSettlesAtExport )
    {
        GemmaModelConfig config;
        config.withContextLength( kContextLength );

        expectFittedLoadSettlesAtExport<GemmaModel<DeviceType::Cuda, TensorDataType::BF16>>(
            "gemma 4 12b",
            modelsDirectory() / "gemma" / "gemma4_12b_it_bf16.bin",
            modelsDirectory() / "gemma" / "gemma4_12b_it_fp4.safetensors",
            config );
    }
}
