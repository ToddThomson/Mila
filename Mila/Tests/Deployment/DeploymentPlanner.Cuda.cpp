/**
 * @file DeploymentPlanner.Cuda.cpp
 * @brief The deployment decision on one device, against readings the test chooses.
 *
 * Specifications/Deployment.md sections 5 and 6 on a synthetic geometry per family: the objective at every
 * budget, each limit and refusal reason, and G4 (a request fixing a plan's values reproduces it). Nothing
 * here reads the device it plans for, so every case runs wherever CUDA does. The recorded readings of G2 are
 * DeploymentPlanner.G2.Cuda.cpp.
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Deployment
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;

    namespace
    {
        constexpr dim_t kTrainedMaximum = 8192;
        constexpr std::size_t kPlenty = std::size_t{ 1 } << 40;

        using GemmaNetwork = GemmaTransformer<DeviceType::Cuda, TensorDataType::FP32>;
        using LlamaNetwork = LlamaTransformer<DeviceType::Cuda, TensorDataType::FP32>;

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
                .withMaxSequenceLength( kTrainedMaximum )
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
                .withMaxSequenceLength( kTrainedMaximum )
                .withRoPETheta( 10000.0f )
                .withRoPEScalingFactor( 1.0f )
                .withBias( false );
        }

        DeviceReading readingOf( std::size_t free_bytes )
        {
            DeviceReading reading;
            reading.device = Device::Cuda( 0 );
            reading.free_bytes = free_bytes;
            reading.total_bytes = free_bytes == 0 ? 0 : kPlenty;
            reading.allocation_granularity = allocationGranularity( Device::Cuda( 0 ) );

            return reading;
        }

        template<typename TNetwork>
        std::expected<DeploymentPlans, DeploymentRefusal> plan(
            const TNetwork& network, const DeploymentRequest& request, std::size_t free_bytes )
        {
            return planOnDevice( network, request, readingOf( free_bytes ), kTrainedMaximum,
                Serialization::WeightsMetadata{}, "" );
        }

        // What one context length costs against a budget: the chunk the rule picks and the whole footprint.
        struct Candidate
        {
            dim_t context_length;
            PrefillChunking prefill;
            MemoryStats footprint;
        };

        template<typename TNetwork>
        Candidate priceAt( const TNetwork& network, dim_t context_length, std::size_t budget )
        {
            const BuildContext context = BuildContext( shape_t{ 1, context_length }, RuntimeMode::Inference, false )
                .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) );
            const PrefillChunking prefill = choosePrefillChunk( network, context, budget );

            return { context_length, prefill, network.getRequiredMemory( context.withPrefillSize( prefill.chunk_rows ) ) };
        }

        /**
         * @brief Section 5 stated directly: every candidate priced, then the objective's choice read off.
         *
         * The planner searches and stops early; this prices every context on the grid and reads the answer off
         * the whole table, so a search that stops in the wrong place, or labels its stop wrongly, disagrees.
         */
        template<typename TNetwork>
        void expectTheObjectiveAt( const TNetwork& network, std::size_t budget )
        {
            std::vector<Candidate> fitting;
            std::optional<Candidate> weights_exceed;

            for ( dim_t context_length = kTrainedMaximum; context_length >= DeploymentRequest::kContextStep;
                  context_length -= DeploymentRequest::kContextStep )
            {
                const Candidate candidate = priceAt( network, context_length, budget );

                if ( candidate.footprint.device_parameter_bytes > budget )
                {
                    weights_exceed = candidate;
                    break;
                }

                if ( candidate.prefill.fits_available_memory && candidate.footprint.totalDeviceBytes() <= budget )
                    fitting.push_back( candidate );
            }

            const auto planned = plan( network, DeploymentRequest{}, budget );
            const std::string at = std::format( "budget {}", budget );

            if ( weights_exceed )
            {
                ASSERT_FALSE( planned.has_value() ) << at;
                EXPECT_EQ( planned.error().reason(), DeploymentRefusal::Reason::WeightsExceedDevice ) << at;

                return;
            }

            if ( fitting.empty() )
            {
                ASSERT_FALSE( planned.has_value() ) << at;
                EXPECT_EQ( planned.error().reason(), DeploymentRefusal::Reason::NothingAboveTheFloorFits ) << at;

                return;
            }

            ASSERT_TRUE( planned.has_value() ) << at << ": " << planned.error().toString();

            const DeploymentPlan& best = planned->best();
            const Candidate& largest = fitting.front();

            const Candidate* chosen = &largest;
            DeploymentPlan::ContextLimit limit = largest.context_length == kTrainedMaximum
                ? DeploymentPlan::ContextLimit::TrainedMaximum
                : DeploymentPlan::ContextLimit::DeviceMemory;

            for ( const Candidate& candidate : fitting )
            {
                if ( !candidate.prefill.isMemoryConstrained() )
                {
                    chosen = &candidate;

                    if ( candidate.context_length < largest.context_length )
                        limit = DeploymentPlan::ContextLimit::FullPrefillChunk;

                    break;
                }
            }

            EXPECT_EQ( best.contextLength(), chosen->context_length ) << at;
            EXPECT_EQ( best.prefillChunkRows(), chosen->prefill.chunk_rows ) << at;
            EXPECT_EQ( best.contextLimitedBy(), limit ) << at << ": " << DeploymentPlan::nameOf( best.contextLimitedBy() );
            EXPECT_EQ( best.footprint().totalDeviceBytes(), chosen->footprint.totalDeviceBytes() ) << at;
            EXPECT_EQ( planned->rankedPlans().size(), 1u ) << at << ": one device, one plan";
        }

        // Every budget where the answer can change: each candidate's weights, its total at every rung, and one
        // byte either side.
        template<typename TNetwork>
        std::vector<std::size_t> budgetsWhereTheAnswerChanges( const TNetwork& network )
        {
            std::vector<std::size_t> budgets;

            for ( dim_t context_length = kTrainedMaximum; context_length >= DeploymentRequest::kContextStep;
                  context_length -= DeploymentRequest::kContextStep )
            {
                const BuildContext context = BuildContext( shape_t{ 1, context_length }, RuntimeMode::Inference, false )
                    .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) );

                for ( dim_t rung : TNetwork::kPrefillChunkRungs )
                {
                    if ( rung > context_length )
                        continue;

                    const MemoryStats stats = network.getRequiredMemory( context.withPrefillSize( rung ) );

                    for ( std::size_t edge : { stats.totalDeviceBytes(), stats.device_parameter_bytes } )
                    {
                        budgets.push_back( edge - 1 );
                        budgets.push_back( edge );
                        budgets.push_back( edge + 1 );
                    }
                }
            }

            return budgets;
        }
    }

    class DeploymentPlannerCudaTests : public ::testing::Test
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

    // ====================================================================
    // Section 5: the objective
    // ====================================================================

    TEST_F( DeploymentPlannerCudaTests, AmpleMemoryPlansTheTrainedMaximumAtTheLargestChunk )
    {
        const GemmaNetwork network( "gemma", gemmaConfig(), Device::Cuda( 0 ) );
        const auto planned = plan( network, DeploymentRequest{}, kPlenty );

        ASSERT_TRUE( planned.has_value() );

        const DeploymentPlan& best = planned->best();

        EXPECT_EQ( best.contextLength(), kTrainedMaximum );
        EXPECT_EQ( best.contextLimitedBy(), DeploymentPlan::ContextLimit::TrainedMaximum );
        EXPECT_EQ( best.prefillChunkRows(), 1024 );
        EXPECT_EQ( best.prefillChunkLimitedBy(), DeploymentPlan::PrefillChunkLimit::LargestTheContextPermits );
        EXPECT_EQ( &planned->best(), &planned->rankedPlans().front() );
    }

    TEST_F( DeploymentPlannerCudaTests, TheRequestCeilingCapsTheContext )
    {
        const GemmaNetwork network( "gemma", gemmaConfig(), Device::Cuda( 0 ) );
        const auto planned = plan( network, DeploymentRequest{}.withAutomaticContextLength( 1024, 5000 ), kPlenty );

        ASSERT_TRUE( planned.has_value() );
        EXPECT_EQ( planned->best().contextLength(), 4096 ) << "the grid below the ceiling";
        EXPECT_EQ( planned->best().contextLimitedBy(), DeploymentPlan::ContextLimit::TrainedMaximum );
    }

    // The whole objective against an oracle that prices every candidate, at every budget where the answer can
    // change. This is the test that fails when rules 3 and 4 are swapped.
    TEST_F( DeploymentPlannerCudaTests, Gemma_AgreesWithTheObjectiveAtEveryBudget )
    {
        const GemmaNetwork network( "gemma", gemmaConfig(), Device::Cuda( 0 ) );

        for ( std::size_t budget : budgetsWhereTheAnswerChanges( network ) )
        {
            expectTheObjectiveAt( network, budget );
        }
    }

    TEST_F( DeploymentPlannerCudaTests, Llama_AgreesWithTheObjectiveAtEveryBudget )
    {
        const LlamaNetwork network( "llama", llamaConfig(), Device::Cuda( 0 ) );

        for ( std::size_t budget : budgetsWhereTheAnswerChanges( network ) )
        {
            expectTheObjectiveAt( network, budget );
        }
    }

    // ====================================================================
    // Refusals and fixed values (section 6, 12.2)
    // ====================================================================

    TEST_F( DeploymentPlannerCudaTests, WeightsThatExceedTheDeviceAreRefusedWithTheirSize )
    {
        const GemmaNetwork network( "gemma", gemmaConfig(), Device::Cuda( 0 ) );
        const Candidate floor = priceAt( network, 1024, kPlenty );

        const auto planned = plan( network, DeploymentRequest{}, floor.footprint.device_parameter_bytes - 1 );

        ASSERT_FALSE( planned.has_value() );
        EXPECT_EQ( planned.error().reason(), DeploymentRefusal::Reason::WeightsExceedDevice );
        EXPECT_EQ( planned.error().footprint().device_parameter_bytes, floor.footprint.device_parameter_bytes );
        EXPECT_EQ( planned.error().reading().free_bytes, floor.footprint.device_parameter_bytes - 1 );
    }

    TEST_F( DeploymentPlannerCudaTests, AFixedContextThatFitsIsPlannedAsFixed )
    {
        const GemmaNetwork network( "gemma", gemmaConfig(), Device::Cuda( 0 ) );
        const auto planned = plan( network, DeploymentRequest{}.withContextLength( 3000 ), kPlenty );

        ASSERT_TRUE( planned.has_value() );
        EXPECT_EQ( planned->best().contextLength(), 3000 );
        EXPECT_EQ( planned->best().contextLimitedBy(), DeploymentPlan::ContextLimit::FixedByCaller );
        EXPECT_EQ( planned->best().prefillChunkRows(), 1024 );
    }

    // Deployment.md 12.2: refused, where it used to be attempted with a warning.
    TEST_F( DeploymentPlannerCudaTests, AFixedContextThatFitsAtNoChunkIsRefused )
    {
        const GemmaNetwork network( "gemma", gemmaConfig(), Device::Cuda( 0 ) );
        const Candidate at_floor_rung = [&]
        {
            const BuildContext context = BuildContext( shape_t{ 1, 4096 }, RuntimeMode::Inference, false )
                .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) );

            return Candidate{ 4096, {}, network.getRequiredMemory( context.withPrefillSize( 64 ) ) };
        }();

        const auto planned = plan( network, DeploymentRequest{}.withContextLength( 4096 ),
            at_floor_rung.footprint.totalDeviceBytes() - 1 );

        ASSERT_FALSE( planned.has_value() );
        EXPECT_EQ( planned.error().reason(), DeploymentRefusal::Reason::FixedContextDoesNotFit );
        EXPECT_EQ( planned.error().contextLength(), 4096 );
    }

    TEST_F( DeploymentPlannerCudaTests, HeadroomIsLeftFree )
    {
        const GemmaNetwork network( "gemma", gemmaConfig(), Device::Cuda( 0 ) );
        const Candidate top = priceAt( network, kTrainedMaximum, kPlenty );
        const std::size_t exact = top.footprint.totalDeviceBytes();

        const auto without = plan( network, DeploymentRequest{}, exact );
        const auto with = plan( network, DeploymentRequest{}.withHeadroom( 1 ), exact );

        ASSERT_TRUE( without.has_value() );
        ASSERT_TRUE( with.has_value() );
        EXPECT_EQ( without->best().contextLength(), kTrainedMaximum );
        EXPECT_EQ( without->best().prefillChunkRows(), 1024 );
        EXPECT_TRUE( with->best().contextLength() < kTrainedMaximum || with->best().prefillChunkRows() < 1024 )
            << "one byte of headroom must cost the plan that fitted exactly";
    }

    // A device that cannot report memory (the CPU, a failed query) cannot choose a context length, and does
    // not stop a caller who chose one.
    TEST_F( DeploymentPlannerCudaTests, ADeviceWithoutAReadingPlansAFixedContextAndRefusesAnAutomaticOne )
    {
        const GemmaNetwork network( "gemma", gemmaConfig(), Device::Cuda( 0 ) );

        const auto automatic = plan( network, DeploymentRequest{}, 0 );
        const auto fixed = plan( network, DeploymentRequest{}.withContextLength( 2048 ), 0 );

        ASSERT_FALSE( automatic.has_value() );
        EXPECT_EQ( automatic.error().reason(), DeploymentRefusal::Reason::DeviceDoesNotReportMemory );

        ASSERT_TRUE( fixed.has_value() );
        EXPECT_EQ( fixed->best().contextLength(), 2048 );
        EXPECT_EQ( fixed->best().prefillChunkRows(), 1024 );
    }

    TEST_F( DeploymentPlannerCudaTests, AContextPastTheTrainedMaximumIsMalformedNotRefused )
    {
        const GemmaNetwork network( "gemma", gemmaConfig(), Device::Cuda( 0 ) );

        EXPECT_THROW( (void)plan( network, DeploymentRequest{}.withContextLength( kTrainedMaximum + 1 ), kPlenty ),
            std::invalid_argument );
        EXPECT_THROW( (void)plan( network, DeploymentRequest{}.withAutomaticContextLength( kTrainedMaximum + 1024 ), kPlenty ),
            std::invalid_argument );
    }

    // ====================================================================
    // G4: a fixed request agrees with the planner
    // ====================================================================

    TEST_F( DeploymentPlannerCudaTests, AFixedRequestAtAPlansValuesReproducesThePlan )
    {
        const GemmaNetwork network( "gemma", gemmaConfig(), Device::Cuda( 0 ) );

        for ( std::size_t budget : budgetsWhereTheAnswerChanges( network ) )
        {
            const auto automatic = plan( network, DeploymentRequest{}, budget );

            if ( !automatic )
            {
                const auto fixed = plan( network,
                    DeploymentRequest{}.withContextLength( automatic.error().contextLength() ), budget );

                ASSERT_FALSE( fixed.has_value() ) << "budget " << budget;

                if ( automatic.error().reason() == DeploymentRefusal::Reason::WeightsExceedDevice )
                    EXPECT_EQ( fixed.error().reason(), DeploymentRefusal::Reason::WeightsExceedDevice ) << "budget " << budget;

                continue;
            }

            const DeploymentPlan& chosen = automatic->best();
            const auto fixed = plan( network, DeploymentRequest{}.withContextLength( chosen.contextLength() ), budget );

            ASSERT_TRUE( fixed.has_value() ) << "budget " << budget;
            EXPECT_EQ( fixed->best().contextLength(), chosen.contextLength() ) << "budget " << budget;
            EXPECT_EQ( fixed->best().prefillChunkRows(), chosen.prefillChunkRows() ) << "budget " << budget;
            EXPECT_EQ( fixed->best().footprint().totalDeviceBytes(), chosen.footprint().totalDeviceBytes() ) << "budget " << budget;
            EXPECT_EQ( fixed->best().contextLimitedBy(), DeploymentPlan::ContextLimit::FixedByCaller ) << "budget " << budget;
        }
    }

    // ====================================================================
    // The types' own contracts
    // ====================================================================

    TEST( DeploymentTypesTests, AnEmptyRankingIsRefusedBecauseNoPlanIsARefusal )
    {
        EXPECT_THROW( DeploymentPlans( std::vector<DeploymentPlan>{} ), std::logic_error );
    }

    TEST( DeploymentTypesTests, AnAutomaticFloorBelowOneStepOrAboveItsCeilingIsMalformed )
    {
        EXPECT_THROW( DeploymentRequest{}.withAutomaticContextLength( 512 ), std::invalid_argument );
        EXPECT_THROW( DeploymentRequest{}.withAutomaticContextLength( 8192, 4096 ), std::invalid_argument );
        EXPECT_NO_THROW( DeploymentRequest{}.withAutomaticContextLength( 4096, 8192 ) );
    }

    TEST( DeploymentTypesTests, ARequestIsAutomaticUntilAContextLengthIsFixed )
    {
        DeploymentRequest request;

        EXPECT_TRUE( request.isContextLengthAutomatic() );

        request.withContextLength( 4096 );
        EXPECT_FALSE( request.isContextLengthAutomatic() );

        request.withAutomaticContextLength();
        EXPECT_TRUE( request.isContextLengthAutomatic() );
    }

    TEST( DeploymentTypesTests, AModelConfigBecomesAFixedRequestWithItsFormats )
    {
        GemmaModelConfig config( 8192 );
        config.withFP4Quantization().withLanguageModelHeadPositions( 4 );

        const DeploymentRequest request = DeploymentRequest::fromModelConfig( config );

        EXPECT_FALSE( request.isContextLengthAutomatic() );
        EXPECT_EQ( request.getContextLength(), 8192 );
        EXPECT_EQ( request.getWeightQuantization(), WeightQuantization::FP4 );
        EXPECT_EQ( request.getKvCacheCompression(), KvCacheCompression::FP8 );
        EXPECT_EQ( request.getLanguageModelHeadPositions(), 4 );
    }
}
