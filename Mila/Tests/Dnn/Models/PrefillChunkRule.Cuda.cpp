/**
 * @file PrefillChunkRule.Cuda.cpp
 * @brief The chunk a network is built with is chosen once, from one reading, and the build executes it.
 *
 * Specifications/Deployment.md Phase 2, on a synthetic geometry per family: the rule, G1 (plan equals
 * build, at every rung) and N1 (a build that read free memory itself would build a different rung than it
 * was given). Runs wherever CUDA does. G1 on the real models is PlanEqualsBuild.Cuda.cpp.
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        // Long enough that every family's table admits every rung.
        constexpr dim_t kContext = 1024;

        using GemmaNetwork = GemmaTransformer<DeviceType::Cuda, TensorDataType::FP32>;
        using LlamaNetwork = LlamaTransformer<DeviceType::Cuda, TensorDataType::FP32>;
        using QwenNetwork = QwenTransformer<DeviceType::Cuda, TensorDataType::FP32>;

        // Pattern 2 over two layers makes layer 1 global, so both block kinds are priced and built.
        GemmaConfig gemmaConfig()
        {
            return GemmaConfig( 64, 2 )
                .withVocabularyLength( 128 )
                .withNumHeads( 4 )
                .withNumKVHeads( 2 )
                .withHeadDim( 32 )
                .withGlobalHeadDim( 64 )
                .withNumGlobalKVHeads( 1 )
                .withKeyEqualsValue( true )
                .withHiddenDimension( 128 )
                .withMaxSequenceLength( kContext )
                .withRMSNormEpsilon( 1e-6f )
                .withWindow( 8 )
                .withSlidingWindowPattern( 2 )
                .withGlobalRotaryDim( 32 )
                .withRoPETheta( 10000.0f )
                .withGlobalRoPETheta( 1000000.0f )
                .withFinalLogitSoftcapping( 30.0f );
        }

        LlamaConfig llamaConfig()
        {
            return LlamaConfig( 64, 2 )
                .withVocabularyLength( 128 )
                .withNumHeads( 4 )
                .withNumKVHeads( 2 )
                .withHiddenDimension( 128 )
                .withMaxSequenceLength( kContext )
                .withRoPETheta( 10000.0f )
                .withRoPEScalingFactor( 1.0f )
                .withBias( false );
        }

        // The published 3:1 interleave: DeltaNet layers and a full-attention layer.
        QwenConfig qwenConfig()
        {
            return QwenConfig( 64, 4 )
                .withVocabularyLength( 128 )
                .withNumHeads( 4 )
                .withNumKVHeads( 2 )
                .withHeadDim( 32 )
                .withAttentionOutputGate( true )
                .withHiddenDimension( 128 )
                .withMaxSequenceLength( kContext )
                .withRMSNormEpsilon( 1e-6f )
                .withRoPETheta( 1e7f )
                .withPartialRotaryFactor( 0.25f )
                .withFullAttentionInterval( 4 );
        }

        BuildContext pricedFor( DeviceId device, dim_t context_length )
        {
            return BuildContext( shape_t{ 1, context_length }, RuntimeMode::Inference, false )
                .withAllocationGranularity( allocationGranularity( device ) );
        }

        // One network alive at a time: the process-wide RoPE cache makes a second one report less than
        // it allocates.
        template<typename TNetwork, typename TConfig>
        MemoryStats predictedAt( const TConfig& config, DeviceId device, const BuildContext& context )
        {
            TNetwork predictor( "network", config, device );

            return predictor.getRequiredMemory( context );
        }

        template<typename TNetwork, typename TConfig>
        MemoryStats builtAt( const TConfig& config, DeviceId device, const BuildContext& context )
        {
            TNetwork network( "network", config, device );
            network.build( context );

            return network.getMemoryStats();
        }

        void expectCategoriesEqual( const MemoryStats& predicted, const MemoryStats& built, const std::string& label )
        {
            EXPECT_EQ( predicted.device_parameter_bytes, built.device_parameter_bytes ) << label << ": parameters";
            EXPECT_EQ( predicted.device_state_bytes, built.device_state_bytes ) << label << ": state";
            EXPECT_EQ( predicted.device_scratch_bytes, built.device_scratch_bytes ) << label << ": scratch";
            EXPECT_EQ( predicted.device_gradient_bytes, built.device_gradient_bytes ) << label << ": gradients";
        }

        // Predicted totals at each rung the context admits, largest rung first.
        template<typename TNetwork, typename TConfig>
        std::vector<std::size_t> totalsByRung( const TConfig& config, DeviceId device, const BuildContext& priced )
        {
            TNetwork predictor( "network", config, device );
            std::vector<std::size_t> totals;

            for ( dim_t rung : TNetwork::kPrefillChunkRungs )
            {
                if ( rung <= priced.inputShape()[ 1 ] )
                    totals.push_back( predictor.getRequiredMemory( priced.withPrefillSize( rung ) ).totalDeviceBytes() );
            }

            return totals;
        }

        // G1 on a synthetic geometry: at every rung, the prediction is the build, category by category. The
        // totals at adjacent rungs differ, so equal totals identify the rung that was built.
        template<typename TNetwork, typename TConfig>
        void expectEveryRungBuildsWhatItPrices( const TConfig& config )
        {
            const DeviceId device = Device::Cuda( 0 );
            const BuildContext priced = pricedFor( device, kContext );

            std::size_t larger_rung_total = 0;

            for ( dim_t rung : TNetwork::kPrefillChunkRungs )
            {
                const BuildContext context = priced.withPrefillSize( rung );
                const MemoryStats predicted = predictedAt<TNetwork>( config, device, context );
                const MemoryStats built = builtAt<TNetwork>( config, device, context );

                expectCategoriesEqual( predicted, built, std::format( "rung {}", rung ) );

                if ( larger_rung_total != 0 )
                {
                    EXPECT_LT( predicted.totalDeviceBytes(), larger_rung_total )
                        << "rung " << rung << " prices the same as the rung above it";
                }

                larger_rung_total = predicted.totalDeviceBytes();
            }
        }

        // N1: the plan is chosen against a reading that admits only a middle rung, while the device itself has
        // room for the largest. A build that read free memory itself would build the largest rung and report its
        // footprint; a build that executes its context builds the rung it was given.
        template<typename TNetwork, typename TConfig>
        void expectTheBuildExecutesThePlannedChunk( const TConfig& config )
        {
            const DeviceId device = Device::Cuda( 0 );
            const BuildContext priced = pricedFor( device, kContext );
            const std::vector<std::size_t> totals = totalsByRung<TNetwork>( config, device, priced );

            ASSERT_GE( totals.size(), 3u ) << "the table needs a rung either side of the planned one";

            const std::size_t planned_index = 1;
            const dim_t planned_rung = TNetwork::kPrefillChunkRungs[ planned_index ];

            ASSERT_GT( readFreeDeviceBytes( device ), totals.front() )
                << "the device must have room for the largest rung, or a build that read it would agree by accident";

            PrefillChunking plan;
            {
                TNetwork planner( "network", config, device );
                plan = choosePrefillChunk( planner, priced, totals[ planned_index ] );
            }

            ASSERT_EQ( plan.chunk_rows, planned_rung );
            EXPECT_TRUE( plan.isMemoryConstrained() );

            const BuildContext context = priced.withPrefillSize( plan.chunk_rows );
            const MemoryStats built = builtAt<TNetwork>( config, device, context );

            expectCategoriesEqual( predictedAt<TNetwork>( config, device, context ), built, "planned rung" );
            EXPECT_NE( built.totalDeviceBytes(), totals.front() ) << "the build took the rung the device had room for";
        }
    }

    class PrefillChunkRuleCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }
        }
    };

    // The rule, stated against readings placed exactly on and just below each rung's price.
    TEST_F( PrefillChunkRuleCudaTests, ChoosesTheLargestRungWhoseTotalFitsTheReading )
    {
        const DeviceId device = Device::Cuda( 0 );
        const BuildContext priced = pricedFor( device, kContext );
        const std::vector<std::size_t> totals = totalsByRung<GemmaNetwork>( gemmaConfig(), device, priced );
        const std::size_t rung_count = std::size( GemmaNetwork::kPrefillChunkRungs );

        ASSERT_EQ( totals.size(), rung_count );

        GemmaNetwork planner( "network", gemmaConfig(), device );

        for ( std::size_t index = 0; index < rung_count; ++index )
        {
            const dim_t rung = GemmaNetwork::kPrefillChunkRungs[ index ];
            const PrefillChunking at = choosePrefillChunk( planner, priced, totals[ index ] );

            EXPECT_EQ( at.chunk_rows, rung ) << "a reading equal to rung " << rung << "'s total";
            EXPECT_EQ( at.unconstrained_chunk_rows, GemmaNetwork::kPrefillChunkRungs[ 0 ] );
            EXPECT_TRUE( at.fits_available_memory );

            const PrefillChunking below = choosePrefillChunk( planner, priced, totals[ index ] - 1 );

            if ( index + 1 < rung_count )
            {
                EXPECT_EQ( below.chunk_rows, GemmaNetwork::kPrefillChunkRungs[ index + 1 ] )
                    << "a reading one byte under rung " << rung << "'s total";
                EXPECT_TRUE( below.fits_available_memory );
            }
            else
            {
                EXPECT_EQ( below.chunk_rows, rung ) << "under the smallest rung, the smallest rung is used anyway";
                EXPECT_FALSE( below.fits_available_memory );
            }
        }
    }

    TEST_F( PrefillChunkRuleCudaTests, AReadingOfZeroTakesTheLargestRungTheContextPermits )
    {
        const DeviceId device = Device::Cuda( 0 );
        GemmaNetwork planner( "network", gemmaConfig(), device );

        const PrefillChunking at_1024 = choosePrefillChunk( planner, pricedFor( device, 1024 ), 0 );
        const PrefillChunking at_700 = choosePrefillChunk( planner, pricedFor( device, 700 ), 0 );

        EXPECT_EQ( at_1024.chunk_rows, 1024 );
        EXPECT_EQ( at_700.chunk_rows, 512 );
        EXPECT_EQ( at_700.unconstrained_chunk_rows, 512 );
        EXPECT_FALSE( at_700.isMemoryConstrained() );
    }

    // Below the smallest rung the context is one chunk, and memory is not consulted.
    TEST_F( PrefillChunkRuleCudaTests, AContextShorterThanTheSmallestRungIsOneChunk )
    {
        const DeviceId device = Device::Cuda( 0 );
        LlamaNetwork planner( "network", llamaConfig(), device );

        const PrefillChunking chunking = choosePrefillChunk( planner, pricedFor( device, 100 ), 1 );

        EXPECT_EQ( chunking.chunk_rows, 100 );
        EXPECT_EQ( chunking.unconstrained_chunk_rows, 100 );
        EXPECT_TRUE( chunking.fits_available_memory );
    }

    TEST_F( PrefillChunkRuleCudaTests, Gemma_EveryRungBuildsWhatItPrices )
    {
        expectEveryRungBuildsWhatItPrices<GemmaNetwork>( gemmaConfig() );
    }

    TEST_F( PrefillChunkRuleCudaTests, Llama_EveryRungBuildsWhatItPrices )
    {
        expectEveryRungBuildsWhatItPrices<LlamaNetwork>( llamaConfig() );
    }

    TEST_F( PrefillChunkRuleCudaTests, Qwen_EveryRungBuildsWhatItPrices )
    {
        expectEveryRungBuildsWhatItPrices<QwenNetwork>( qwenConfig() );
    }

    TEST_F( PrefillChunkRuleCudaTests, Gemma_TheBuildExecutesThePlannedChunk )
    {
        expectTheBuildExecutesThePlannedChunk<GemmaNetwork>( gemmaConfig() );
    }

    TEST_F( PrefillChunkRuleCudaTests, Llama_TheBuildExecutesThePlannedChunk )
    {
        expectTheBuildExecutesThePlannedChunk<LlamaNetwork>( llamaConfig() );
    }

    TEST_F( PrefillChunkRuleCudaTests, Qwen_TheBuildExecutesThePlannedChunk )
    {
        expectTheBuildExecutesThePlannedChunk<QwenNetwork>( qwenConfig() );
    }
}
