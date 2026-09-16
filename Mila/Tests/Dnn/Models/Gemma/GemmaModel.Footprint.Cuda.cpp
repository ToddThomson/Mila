/**
 * @file GemmaModel.Footprint.Cuda.cpp
 * @brief Gate B -- the reported footprint against what the driver says was consumed.
 *
 * Gate A (Gemma.Cuda.cpp) compares getRequiredMemory against getMemoryStats: two Mila
 * accountings of the same allocations, which agree exactly or one of them is wrong.
 * Gate B compares against cudaMemGetInfo, which is a different question -- it sees what
 * MemoryStats cannot: allocator rounding, and the grow-on-demand execution-context
 * scratch that no build-time contract observes.
 *
 * Gate B is therefore NOT an equality test. Its job is to bound and attribute the
 * residual: the prediction must not exceed what was actually consumed (an overestimate
 * refuses configurations that fit), and the shortfall must stay within a stated margin
 * so a regression in the unmodelled terms is visible rather than absorbed.
 *
 * Requires a real checkpoint and is skipped without one, so it does not run in CI.
 * See Specifications/MemoryFootprint.md section 7.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>

import Mila;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace fs = std::filesystem;

    namespace
    {
        fs::path gemmaCheckpointPath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_12b_it_bf16.bin";
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

    class GemmaFootprintCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            checkpoint_ = gemmaCheckpointPath();

            if ( !fs::exists( checkpoint_ ) )
            {
                GTEST_SKIP() << "Gemma checkpoint not present at: " << checkpoint_.string();
            }
        }

        fs::path checkpoint_;
    };

    // The prediction runs before the model exists, so it must cost nothing on the device.
    // If asking the question allocated, the answer could not be trusted on a machine that
    // is already close to full -- which is precisely where the question gets asked.
    TEST_F( GemmaFootprintCudaTests, GetRequiredMemory_AllocatesNothing )
    {
        // Force CUDA context creation first, or its one-time cost lands inside the window.
        cudaFree( nullptr );

        const std::size_t free_before = freeDeviceBytes();

        GemmaModelConfig config;
        config.withContextLength( 8192 )
            .withWeightQuantization( WeightQuantization::FP4 );

        const MemoryStats predicted =
            GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::getRequiredMemory(
                checkpoint_, config );

        const std::size_t free_after = freeDeviceBytes();

        EXPECT_GT( predicted.totalDeviceBytes(), 0u );
        EXPECT_EQ( free_before, free_after )
            << "asking for the footprint consumed device memory";
    }

    // The prediction must scale with context, and scale in the KV cache rather than the
    // weights -- Gemma's local layers hold a bounded ring, so the growth is sub-linear
    // and the weights term is flat. A prediction that ignored context would pass Gate A
    // and still be useless.
    TEST_F( GemmaFootprintCudaTests, GetRequiredMemory_GrowsWithContext )
    {
        auto predictAt = []( const fs::path& path, dim_t context_length )
        {
            GemmaModelConfig config;
            config.withContextLength( context_length )
                .withWeightQuantization( WeightQuantization::FP4 );

            return GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::getRequiredMemory(
                path, config );
        };

        const MemoryStats small = predictAt( checkpoint_, 4096 );
        const MemoryStats large = predictAt( checkpoint_, 32768 );

        EXPECT_EQ( small.device_parameter_bytes, large.device_parameter_bytes )
            << "weights must not depend on context length";
        EXPECT_GT( large.device_state_bytes, small.device_state_bytes )
            << "KV cache and activations must grow with context length";

        std::cout << std::format(
            "[footprint] ctx 4096  weights {:.2f} GiB  state {:.2f} GiB  total {:.2f} GiB\n"
            "[footprint] ctx 32768 weights {:.2f} GiB  state {:.2f} GiB  total {:.2f} GiB\n",
            toGiB( small.device_parameter_bytes ), toGiB( small.device_state_bytes ),
            toGiB( small.totalDeviceBytes() ),
            toGiB( large.device_parameter_bytes ), toGiB( large.device_state_bytes ),
            toGiB( large.totalDeviceBytes() ) );
    }

    // The chunk the rule picks against the device's free memory, reported beside the memory
    // (MemoryFootprint.md section 11). What it picks depends on the card and on what else is
    // resident, so this holds the rule's properties rather than a chunk: it is no larger than the
    // context permits, a pick that fits keeps the prediction within the memory that was free, and a
    // longer context never buys a larger chunk.
    TEST_F( GemmaFootprintCudaTests, GetDeploymentFootprint_ReportsPrefillChunkAndAgreesOnMemory )
    {
        auto footprintAt = []( const fs::path& path, dim_t context_length )
        {
            GemmaModelConfig config;
            config.withContextLength( context_length )
                .withWeightQuantization( WeightQuantization::FP4 );

            return GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::getDeploymentFootprint(
                path, config );
        };

        GemmaModelConfig config;
        config.withContextLength( 8192 )
            .withWeightQuantization( WeightQuantization::FP4 );

        cudaFree( nullptr );

        const std::size_t free_before = freeDeviceBytes();
        const DeploymentFootprint short_context = footprintAt( checkpoint_, 8192 );
        const MemoryStats memory_only =
            GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::getRequiredMemory(
                checkpoint_, config );

        EXPECT_EQ( short_context.memory.totalDeviceBytes(), memory_only.totalDeviceBytes() )
            << "the two entry points must answer from the same arithmetic";

        EXPECT_GT( short_context.prefill.chunk_rows, 0 );
        EXPECT_LE( short_context.prefill.chunk_rows,
                   short_context.prefill.unconstrained_chunk_rows );

        if ( short_context.prefill.fits_available_memory )
        {
            EXPECT_LE( short_context.memory.totalDeviceBytes(), free_before )
                << "a chunk that fits must keep the prediction within the free memory";
        }

        const DeploymentFootprint long_context = footprintAt( checkpoint_, 131072 );

        EXPECT_LE( long_context.prefill.chunk_rows, short_context.prefill.chunk_rows )
            << "a longer context must never buy a larger prefill chunk";

        std::cout << std::format(
            "[prefill] ctx 8192   chunk {} of {} rows\n"
            "[prefill] ctx 131072 chunk {} of {} rows\n",
            short_context.prefill.chunk_rows, short_context.prefill.unconstrained_chunk_rows,
            long_context.prefill.chunk_rows, long_context.prefill.unconstrained_chunk_rows );
    }

    // Gate B proper. Predict, then actually load, and hold the two against the driver's
    // own accounting. The printed residual is the deliverable as much as the assertions:
    // it is the size of everything a build-time contract cannot see.
    TEST_F( GemmaFootprintCudaTests, GetRequiredMemory_BoundsActualConsumption )
    {
        constexpr dim_t kContextLength = 8192;

        cudaFree( nullptr );

        GemmaModelConfig config;
        config.withContextLength( kContextLength )
            .withWeightQuantization( WeightQuantization::FP4 );

        const MemoryStats predicted =
            GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::getRequiredMemory(
                checkpoint_, config );

        const std::size_t free_before = freeDeviceBytes();

        auto model = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::fromPretrained(
            checkpoint_, config );

        ASSERT_NE( model, nullptr );

        const std::size_t free_after_load = freeDeviceBytes();
        const std::size_t consumed = free_before - free_after_load;

        const MemoryStats reported = model->getMemoryStats();
        const std::size_t residual = consumed > predicted.totalDeviceBytes()
            ? consumed - predicted.totalDeviceBytes()
            : 0;

        std::cout << std::format(
            "[gate B] context {}\n"
            "  predicted (getRequiredMemory) {:.3f} GiB\n"
            "  reported  (getMemoryStats)    {:.3f} GiB\n"
            "  consumed  (cudaMemGetInfo)    {:.3f} GiB\n"
            "  residual  (unmodelled)        {:.3f} GiB  ({:.1f}% of consumed)\n",
            kContextLength,
            toGiB( predicted.totalDeviceBytes() ),
            toGiB( reported.totalDeviceBytes() ),
            toGiB( consumed ),
            toGiB( residual ),
            consumed > 0 ? ( 100.0 * static_cast<double>( residual ) / consumed ) : 0.0 );

        // Scratch is reserved at build and reported, so it is inside the prediction rather than the
        // residual.
        std::cout << std::format( "  of which scratch (reported)   {:.3f} GiB\n", toGiB( reported.device_scratch_bytes ) );

        // The prediction is an accounting of the same allocations getMemoryStats counts, so
        // after a real build they must still agree exactly -- Gate A, restated on a real
        // model rather than a synthetic config.
        EXPECT_EQ( predicted.device_parameter_bytes, reported.device_parameter_bytes );
        EXPECT_EQ( predicted.device_state_bytes, reported.device_state_bytes );
        EXPECT_EQ( predicted.device_scratch_bytes, reported.device_scratch_bytes );

        // Never promise more room than exists.
        EXPECT_LE( predicted.totalDeviceBytes(), consumed )
            << "prediction exceeded actual consumption -- an overestimate refuses "
               "configurations that fit";

        // What Mila does not predict: the share of packed small allocations and the fixed remainder,
        // together under 30 MiB once rounding is predicted (MemoryFootprint.md 11.8), plus -- on a card
        // that drives a display -- the Windows budget cut, measured at 224 MiB on the RTX 4070 (11.5).
        // The bound covers a display card because this runs wherever the suite runs; the 64 MiB figure
        // is checked on a headless card in the Phase 6 step 3 gate runs.
        constexpr std::size_t kResidualBoundBytes = std::size_t{ 512 } * 1024 * 1024;

        EXPECT_LT( residual, kResidualBoundBytes )
            << "unmodelled memory exceeded 512 MiB";
    }
}
