/**
 * @file LlamaModel.Footprint.Cuda.cpp
 * @brief Gate B for the Llama chassis, and a measurement of its missing memory gates.
 *
 * The companion to GemmaModel.Footprint.Cuda.cpp. The invariants are the same -- asking
 * costs nothing, the answer grows with context, and it never promises more room than the
 * load actually consumes.
 *
 * What differs is what the numbers show. Llama pools no per-block activations, does not
 * tie its embedding to its head, and quantizes neither of those two tables, so its
 * footprint carries costs Gemma's does not. That is a known limitation (BACKLOG, Models)
 * rather than a bug in the reporting, and QuantizationRatio_ExposesUnquantizedTables
 * turns it from a suspicion into a figure derived from the checkpoint itself.
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
#include <optional>

#include "Common/DeviceWithoutDisplay.h"

import Mila;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace fs = std::filesystem;

    namespace
    {
        fs::path llamaCheckpointPath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama31_8b_instruct_bf16.bin";
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

        double toGiB( std::size_t bytes )
        {
            return static_cast<double>( bytes ) / ( 1024.0 * 1024.0 * 1024.0 );
        }

        MemoryStats predictAt( const fs::path& path, dim_t context_length, WeightQuantization quantization )
        {
            return LlamaModel<DeviceType::Cuda, TensorDataType::BF16>::getRequiredMemory(
                path, LlamaModelConfig( context_length ).withWeightQuantization( quantization ) );
        }
    }

    class LlamaFootprintCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            checkpoint_ = llamaCheckpointPath();

            if ( !fs::exists( checkpoint_ ) )
            {
                GTEST_SKIP() << "Llama checkpoint not present at: " << checkpoint_.string();
            }
        }

        fs::path checkpoint_;
    };

    TEST_F( LlamaFootprintCudaTests, GetRequiredMemory_AllocatesNothing )
    {
        const Common::ScopedCurrentCudaDevice current( 0 );

        cudaFree( nullptr );

        const std::size_t free_before = freeDeviceBytesOn( 0 );

        const MemoryStats predicted = predictAt( checkpoint_, 8192, WeightQuantization::FP4 );

        const std::size_t free_after = freeDeviceBytesOn( 0 );

        EXPECT_GT( predicted.totalDeviceBytes(), 0u );
        EXPECT_EQ( free_before, free_after )
            << "asking for the footprint consumed device memory";
    }

    // Llama's attention scratch spans the full context rather than collapsing to a ring
    // width -- there is no flash-prefill reclaim here and no sliding-window KV policy --
    // so state is expected to climb considerably faster with context than Gemma's does.
    // The figures are the point; the assertion only pins the direction.
    TEST_F( LlamaFootprintCudaTests, GetRequiredMemory_GrowsWithContext )
    {
        const MemoryStats small = predictAt( checkpoint_, 4096, WeightQuantization::FP4 );
        const MemoryStats large = predictAt( checkpoint_, 32768, WeightQuantization::FP4 );

        EXPECT_EQ( small.device_parameter_bytes, large.device_parameter_bytes )
            << "weights must not depend on context length";
        EXPECT_GT( large.device_state_bytes, small.device_state_bytes )
            << "KV cache and activations must grow with context length";

        std::cout << std::format(
            "[footprint] ctx 4096  weights {:.2f} GiB  state {:.2f} GiB  total {:.2f} GiB\n"
            "[footprint] ctx 32768 weights {:.2f} GiB  state {:.2f} GiB  total {:.2f} GiB\n"
            "[footprint] state growth across an 8x context increase: {:.2f} GiB\n",
            toGiB( small.device_parameter_bytes ), toGiB( small.device_state_bytes ),
            toGiB( small.totalDeviceBytes() ),
            toGiB( large.device_parameter_bytes ), toGiB( large.device_state_bytes ),
            toGiB( large.totalDeviceBytes() ),
            toGiB( large.device_state_bytes - small.device_state_bytes ) );
    }

    // The Llama half of the prefill-chunk question: the same rule and the same properties as the
    // Gemma test, since what the rule picks depends on the card and on what else is resident.
    TEST_F( LlamaFootprintCudaTests, GetDeploymentFootprint_ReportsPrefillChunkAndAgreesOnMemory )
    {
        auto footprintAt = []( const fs::path& path, dim_t context_length )
        {
            return LlamaModel<DeviceType::Cuda, TensorDataType::BF16>::getDeploymentFootprint(
                path,
                LlamaModelConfig( context_length )
                    .withWeightQuantization( WeightQuantization::FP4 ) );
        };

        const Common::ScopedCurrentCudaDevice current( 0 );

        cudaFree( nullptr );

        const std::size_t free_before = freeDeviceBytesOn( 0 );
        const DeploymentFootprint short_context = footprintAt( checkpoint_, 8192 );

        EXPECT_EQ( short_context.memory.totalDeviceBytes(),
                   predictAt( checkpoint_, 8192, WeightQuantization::FP4 ).totalDeviceBytes() )
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

    // The memory-gate limitation, measured rather than asserted from geometry.
    //
    // If every weight participated in FP4, quantized parameters would be about 0.266 of
    // the BF16 figure -- 0.53125 bytes per parameter against 2. The token embedding and
    // lm_head carry no quantization policy on this chassis and neither is tied, so two
    // full-precision [vocab, model_dim] tables survive quantization and pull the ratio up.
    // The gap between the observed ratio and 0.266 is the cost of the missing gates.
    //
    // Deliberately not asserted against a threshold: this documents a known limitation, and
    // a test that fails when the limitation is fixed is a test nobody thanks you for.
    TEST_F( LlamaFootprintCudaTests, QuantizationRatio_ExposesUnquantizedTables )
    {
        const MemoryStats unquantized = predictAt( checkpoint_, 8192, WeightQuantization::None );
        const MemoryStats quantized = predictAt( checkpoint_, 8192, WeightQuantization::FP4 );

        ASSERT_GT( unquantized.device_parameter_bytes, 0u );

        const double ratio = static_cast<double>( quantized.device_parameter_bytes )
            / static_cast<double>( unquantized.device_parameter_bytes );

        constexpr double kIdealFp4Ratio = 0.53125 / 2.0;

        const std::size_t unquantized_residue = quantized.device_parameter_bytes
            - static_cast<std::size_t>( kIdealFp4Ratio * unquantized.device_parameter_bytes );

        std::cout << std::format(
            "[gates] BF16 weights           {:.3f} GiB\n"
            "[gates] FP4 weights            {:.3f} GiB\n"
            "[gates] observed ratio         {:.3f}  (fully quantized would be {:.3f})\n"
            "[gates] cost of missing gates  {:.3f} GiB\n",
            toGiB( unquantized.device_parameter_bytes ),
            toGiB( quantized.device_parameter_bytes ),
            ratio, kIdealFp4Ratio,
            toGiB( unquantized_residue ) );

        EXPECT_LT( quantized.device_parameter_bytes, unquantized.device_parameter_bytes )
            << "FP4 weights must be smaller than BF16 weights";
    }

    // Runs on a device that drives no display, for the reason GemmaModel.Footprint.Cuda.cpp gives.
    TEST_F( LlamaFootprintCudaTests, GetRequiredMemory_BoundsActualConsumption )
    {
        constexpr dim_t kContextLength = 8192;

        const std::optional<int> without_display = Common::findCudaDeviceWithoutDisplay();
        const int ordinal = without_display.value_or( 0 );
        const DeviceId device{ DeviceType::Cuda, ordinal };

        const Common::ScopedCurrentCudaDevice current( ordinal );

        ASSERT_TRUE( current.selected() );

        cudaFree( nullptr );

        const LlamaModelConfig config =
            LlamaModelConfig( kContextLength ).withWeightQuantization( WeightQuantization::FP4 );

        const MemoryStats predicted =
            LlamaModel<DeviceType::Cuda, TensorDataType::BF16>::getRequiredMemory( checkpoint_, config, device );

        const std::size_t free_before = freeDeviceBytesOn( ordinal );

        auto model = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>::fromPretrained(
            checkpoint_, config, device );

        ASSERT_NE( model, nullptr );

        const std::size_t free_after_load = freeDeviceBytesOn( ordinal );
        const std::size_t consumed = free_before - free_after_load;

        const MemoryStats reported = model->getMemoryStats();
        const std::size_t residual = consumed > predicted.totalDeviceBytes()
            ? consumed - predicted.totalDeviceBytes()
            : 0;

        std::cout << std::format(
            "[gate B] context {}, CUDA device {}{}\n"
            "  predicted (getRequiredMemory) {:.3f} GiB\n"
            "  reported  (getMemoryStats)    {:.3f} GiB\n"
            "  consumed  (cudaMemGetInfo)    {:.3f} GiB\n"
            "  residual  (unmodelled)        {:.3f} GiB  ({:.1f}% of consumed)\n",
            kContextLength, ordinal, without_display ? "" : " (drives a display)",
            toGiB( predicted.totalDeviceBytes() ),
            toGiB( reported.totalDeviceBytes() ),
            toGiB( consumed ),
            toGiB( residual ),
            consumed > 0 ? ( 100.0 * static_cast<double>( residual ) / consumed ) : 0.0 );

        // Scratch is reserved at build and reported, so it is inside the prediction rather than the
        // residual.
        std::cout << std::format( "  of which scratch (reported)   {:.3f} GiB\n", toGiB( reported.device_scratch_bytes ) );

        EXPECT_EQ( predicted.device_parameter_bytes, reported.device_parameter_bytes );
        EXPECT_EQ( predicted.device_state_bytes, reported.device_state_bytes );
        EXPECT_EQ( predicted.device_scratch_bytes, reported.device_scratch_bytes );

        EXPECT_LE( predicted.totalDeviceBytes(), consumed )
            << "prediction exceeded actual consumption -- an overestimate refuses "
               "configurations that fit";

        if ( !without_display )
        {
            GTEST_SKIP() << "every visible CUDA device drives a display, so the residual measures the "
                            "Windows budget as well as Mila; measured " << residual / ( 1024 * 1024 ) << " MiB";
        }

        // What Mila does not predict on a device without a display: the share of packed small allocations
        // and the fixed remainder, together under 30 MiB once rounding is predicted (MemoryFootprint.md
        // 11.8). The bound is Phase 6 step 3's criterion 2.
        constexpr std::size_t kResidualBoundBytes = std::size_t{ 64 } * 1024 * 1024;

        EXPECT_LT( residual, kResidualBoundBytes )
            << "unmodelled memory exceeded 64 MiB";
    }
}
