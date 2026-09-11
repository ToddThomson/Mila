/**
 * @file GemmaModel.Rates.Cuda.cpp
 * @brief Prefill and decode rates for Gemma 4 12B FP4, timed the way Qwen's are.
 *
 * These exist to be read against QwenPackedArtifactTests' rates. A number that has no
 * comparator says only that the machine ran; the question these answer is what the 27B
 * chassis costs against the 12B one on the same card, by the same subtraction, in the
 * same build. DISABLED because they are measurements, and they need a real checkpoint.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <stop_token>
#include <vector>

import Mila;

#include "Common/GenerationRates.h"

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    using Mila::Tests::Common::decodeSecondsPerToken;
    using Mila::Tests::Common::prefillSecondsPerToken;

    namespace fs = std::filesystem;

    namespace
    {
        using GemmaBf16 = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;

        fs::path gemmaCheckpoint()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_12b_it_bf16.bin";
        }
    }

    class GemmaRateCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            checkpoint_ = gemmaCheckpoint();

            if ( !fs::exists( checkpoint_ ) )
            {
                GTEST_SKIP() << "Gemma checkpoint not present at: " << checkpoint_.string();
            }
        }

        fs::path checkpoint_;
    };

    // Same context and prompt lengths as QwenPackedArtifactTests.DISABLED_PrefillRate, so
    // the two are directly comparable:
    //   MilaTests --gtest_also_run_disabled_tests
    //       --gtest_filter=GemmaRateCudaTests.DISABLED_PrefillRate
    TEST_F( GemmaRateCudaTests, DISABLED_PrefillRate )
    {
        constexpr dim_t kContextLength = 4096;
        constexpr dim_t kLongPrompt = 2048;
        constexpr dim_t kShortPrompt = 128;

        GemmaModelConfig config;
        config.withContextLength( kContextLength )
            .withWeightQuantization( WeightQuantization::FP4 );

        auto model = GemmaBf16::fromPretrained( checkpoint_, config );

        ASSERT_NE( model, nullptr );

        const double per_token_seconds =
            prefillSecondsPerToken( model, kLongPrompt, kShortPrompt );

        std::cout << std::format(
            "\n  Gemma 4 12B FP4, context {}, prompts {} vs {}\n"
            "  PREFILL: {:.2f} ms/token, {:.0f} tok/s\n",
            kContextLength, kLongPrompt, kShortPrompt,
            per_token_seconds * 1000.0, 1.0 / per_token_seconds ) << std::flush;

        EXPECT_GT( per_token_seconds, 0.0 );
    }

    //   MilaTests --gtest_also_run_disabled_tests
    //       --gtest_filter=GemmaRateCudaTests.DISABLED_DecodeRate
    TEST_F( GemmaRateCudaTests, DISABLED_DecodeRate )
    {
        constexpr dim_t kContextLength = 512;
        constexpr int kShort = 8;
        constexpr int kLong = 72;

        GemmaModelConfig config;
        config.withContextLength( kContextLength )
            .withWeightQuantization( WeightQuantization::FP4 );

        auto model = GemmaBf16::fromPretrained( checkpoint_, config );

        ASSERT_NE( model, nullptr );

        const std::vector<std::int32_t> prompt{ 760, 6511, 314, 9338, 369 };
        const double per_token = decodeSecondsPerToken( model, prompt, kLong, kShort );

        const MemoryStats stats = model->getMemoryStats();
        const double weight_gib =
            static_cast<double>( stats.device_parameter_bytes ) / ( 1024.0 * 1024 * 1024 );

        // Decode reads every resident weight once per token, so this is the figure that
        // says whether a rate difference is bytes or kernels.
        const double achieved_gb_per_second =
            ( weight_gib * 1024 * 1024 * 1024 ) / per_token / 1e9;

        std::cout << std::format(
            "\n  Gemma 4 12B FP4, context {}\n"
            "  DECODE: {:.2f} ms/token, {:.1f} tok/s\n"
            "  device weights {:.2f} GiB, implied bandwidth {:.0f} GB/s\n",
            kContextLength, per_token * 1000.0, 1.0 / per_token,
            weight_gib, achieved_gb_per_second ) << std::flush;

        EXPECT_GT( per_token, 0.0 );
    }
}
