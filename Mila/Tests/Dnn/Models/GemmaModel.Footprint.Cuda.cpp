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

    // The chunk the prediction resolves on its way to sizing the activation workspaces, and
    // used to discard. `context_length: "auto"` bounds on it, because memory alone cannot say
    // that the largest context which fits is one where prefill has walked down toward its floor.
    //
    // Device-independent by construction: the budget is a fixed constant less the global KV
    // term, both computed from the checkpoint's geometry, so these numbers do not depend on
    // which card runs the test. See Specifications/ChatConfiguration.md section 6.
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

        const DeploymentFootprint short_context = footprintAt( checkpoint_, 8192 );
        const MemoryStats memory_only =
            GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::getRequiredMemory(
                checkpoint_, config );

        EXPECT_EQ( short_context.memory.totalDeviceBytes(), memory_only.totalDeviceBytes() )
            << "the two entry points must answer from the same arithmetic";

        EXPECT_GT( short_context.prefill.chunk_rows, 0 );
        EXPECT_LE( short_context.prefill.chunk_rows,
                   short_context.prefill.unconstrained_chunk_rows );

        // The property auto depends on. Without it the throughput bound would be a constant and
        // the scan would have nothing to find.
        const DeploymentFootprint long_context = footprintAt( checkpoint_, 131072 );

        EXPECT_LT( long_context.prefill.chunk_rows, short_context.prefill.chunk_rows )
            << "the prefill chunk must walk down as the KV cache eats the activation budget";
        EXPECT_TRUE( long_context.prefill.isBudgetConstrained() )
            << "at the trained maximum the chunk is expected to be budget-bound";

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

        // Attribute the residual. Scratch is the largest term a build-time contract cannot
        // see: it is allocated lazily during forward passes, sized by whichever operation
        // needed the most, and never shrunk. Whatever is left after subtracting it is
        // allocator rounding plus anything the load path leaves behind.
        const std::size_t scratch = model->getScratchHighWaterBytes();

        std::cout << std::format(
            "  of which scratch high-water   {:.3f} GiB  ({:.1f}% of the residual)\n"
            "  unattributed remainder        {:.3f} GiB\n",
            toGiB( scratch ),
            residual > 0 ? ( 100.0 * static_cast<double>( scratch ) / residual ) : 0.0,
            toGiB( residual > scratch ? residual - scratch : 0 ) );

        // The prediction is an accounting of the same allocations getMemoryStats counts, so
        // after a real build they must still agree exactly -- Gate A, restated on a real
        // model rather than a synthetic config.
        EXPECT_EQ( predicted.device_parameter_bytes, reported.device_parameter_bytes );
        EXPECT_EQ( predicted.device_state_bytes, reported.device_state_bytes );

        // Never promise more room than exists.
        EXPECT_LE( predicted.totalDeviceBytes(), consumed )
            << "prediction exceeded actual consumption -- an overestimate refuses "
               "configurations that fit";

        // Bound the unmodelled terms. Generous by design: this is not an equality test,
        // and the number that matters is the one printed above. Tighten only when the
        // residual has been attributed.
        EXPECT_LT( residual, consumed / 4 )
            << "unmodelled memory exceeded 25% of what was consumed";
    }
}
