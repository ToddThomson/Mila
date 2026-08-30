/**
 * @file ObservationCost.Cuda.cpp
 * @brief Observability's Section 10 cost criterion, measured on a whole model.
 *
 * Publication is a Component facility, so its cost is (publish calls per token) x (cost per
 * call) and nothing about it is per-family. That is why this file is not under Models/: the
 * two chassis below are two samples of one measurement, not two measurements.
 *
 * WHICH CONFIGURATION MEASURES IT. The overhead is FIXED per token, so its share is largest
 * where a decode step is CHEAPEST -- a small model, not the largest one. Section 10 names
 * "the 27B at 16K", which is the configuration that hides the effect best: ~33 ms/token on a
 * 64-layer stack, and on the 12 GiB card that build also sits on the spill boundary
 * (10.27 GiB predicted against ~10.85 free), so the number would carry the pager rather than
 * the branch. Measuring where the effect would be LARGEST and finding nothing is the
 * stronger claim, and it settles the 27B a fortiori.
 *
 * Run both arms on the 16 GiB card, where every model here is DRAM-resident with room:
 *   set CUDA_VISIBLE_DEVICES=GPU-9a81c7d1-9db2-16b3-c256-2f991ec2a22c
 *   MilaTests --gtest_also_run_disabled_tests --gtest_filter=ObservationCostCudaTests.*
 *
 * TWO ARMS, ONE LINE APART. The arms differ only by kObservabilityPublishes in Component.ixx
 * -- a temporary instrument, not shipped. Run baseline / instrumented / baseline: Section 7
 * measured a stub on one component and found two IDENTICAL builds further apart than either
 * was from the instrumented one, so build drift is bracketed rather than assumed away.
 *
 * REPEATS, NOT ONE PAIR. decodeSecondsPerToken is a single subtraction and cannot separate a
 * small effect from run-to-run noise, which is exactly where Section 7's "+2.2%" artifact
 * came from. The median is the number; the min-max spread is what says whether the median
 * can resolve anything at all.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <stop_token>
#include <string_view>
#include <vector>

import Mila;

#include "Common/GenerationRates.h"

namespace Mila::Tests::Dnn::Observation
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    using Mila::Tests::Common::decodeSecondsPerToken;

    namespace fs = std::filesystem;

    namespace
    {
        constexpr int kShortTokens = 8;
        constexpr int kLongTokens = 72;
        constexpr int kRepeats = 7;

        constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

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

        /**
         * @brief Repeat the subtraction, then report median, spread and every sample.
         *
         * Headroom is reported because it is the precondition for believing any of it: a
         * model that fills the card measures the pager, and cudaMemGetInfo cannot see WDDM's
         * shared allocation, so a spill reads as free memory that never returns.
         */
        template<typename TModel>
        void measureAndReport( TModel& model, std::string_view label, dim_t context_length )
        {
            const std::vector<std::int32_t> prompt{ 760, 6511, 314, 9338, 369 };

            const MemoryStats stats = model->getMemoryStats();
            const double weight_gib = static_cast<double>( stats.device_parameter_bytes ) / kGiB;
            const double headroom_gib = static_cast<double>( freeDeviceBytes() ) / kGiB;

            std::vector<double> samples;
            samples.reserve( kRepeats );

            for ( int repeat = 0; repeat < kRepeats; ++repeat )
            {
                samples.push_back(
                    decodeSecondsPerToken( model, prompt, kLongTokens, kShortTokens ) );
            }

            std::vector<double> sorted = samples;
            std::sort( sorted.begin(), sorted.end() );

            const double median = sorted[ sorted.size() / 2 ];
            const double fastest = sorted.front();
            const double slowest = sorted.back();

            std::cout << std::format(
                "\n  {}, context {}\n"
                "  device weights {:.2f} GiB, {:.2f} GiB still free after load\n"
                "  DECODE over {} repeats ({} vs {} tokens, by subtraction):\n"
                "    median  {:.4f} ms/token  ({:.2f} tok/s)\n"
                "    fastest {:.4f}  slowest {:.4f}  spread {:.4f} ms ({:.2f}% of median)\n",
                label, static_cast<long long>( context_length ),
                weight_gib, headroom_gib,
                kRepeats, kShortTokens, kLongTokens,
                median * 1000.0, 1.0 / median,
                fastest * 1000.0, slowest * 1000.0,
                ( slowest - fastest ) * 1000.0,
                ( slowest - fastest ) / median * 100.0 ) << std::flush;

            std::cout << "    samples:";

            for ( const double sample : samples )
            {
                std::cout << std::format( " {:.4f}", sample * 1000.0 );
            }

            std::cout << " (ms/token)\n" << std::flush;

            EXPECT_GT( median, 0.0 );
        }
    }

    class ObservationCostCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }
        }

        static void skipUnlessPresent( const fs::path& path )
        {
            if ( !fs::exists( path ) )
            {
                GTEST_SKIP() << "Checkpoint not present at: " << path.string();
            }
        }
    };

    // The sensitive arm. 28 layers and a few ms per decode step, so a fixed per-token
    // overhead is a far larger share here than on any larger model -- this is the one that
    // would show an effect if there were one.
    TEST_F( ObservationCostCudaTests, DISABLED_Llama32_3B_Fp4 )
    {
        constexpr dim_t kContextLength = 4096;

        const fs::path checkpoint = fs::path( TEST_DATA_DIR ) / "models" / "llama"
            / "llama32_3b_instruct_fp4.safetensors";

        skipUnlessPresent( checkpoint );

        LlamaModelConfig config( kContextLength );
        config.withWeightQuantization( WeightQuantization::FP4 );

        auto model = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>::fromPretrained(
            checkpoint, config );

        ASSERT_NE( model, nullptr );

        measureAndReport( model, "Llama 3.2 3B Instruct FP4", kContextLength );
    }

    // The deployment-realistic arm, and a second chassis: if publication were costing
    // anything structural rather than incidental, two unrelated block designs would not both
    // read as noise.
    TEST_F( ObservationCostCudaTests, DISABLED_Gemma4_12B_Fp4 )
    {
        constexpr dim_t kContextLength = 4096;

        const fs::path checkpoint = fs::path( TEST_DATA_DIR ) / "models" / "gemma"
            / "gemma4_12b_it_bf16.bin";

        skipUnlessPresent( checkpoint );

        GemmaModelConfig config;
        config.withContextLength( kContextLength )
            .withWeightQuantization( WeightQuantization::FP4 );

        auto model = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::fromPretrained(
            checkpoint, config );

        ASSERT_NE( model, nullptr );

        measureAndReport( model, "Gemma 4 12B Instruct FP4", kContextLength );
    }
}
