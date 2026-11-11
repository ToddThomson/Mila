/**
 * @file PrefillScoreRowsRule.Cpu.cpp
 * @brief The block of prefill score rows a plan takes: from the memory its context and chunk left, changing neither.
 *
 * Specifications/Deployment.md section 5, against a network the test makes up, so nothing reads or builds on a
 * device and every case runs in any build.
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <expected>
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
        constexpr std::size_t kMiB = std::size_t{ 1 } << 20;

        /// A network whose footprint is its weights, bytes a context position, bytes a chunk row and bytes a score row.
        struct MadeUpNetwork
        {
            static constexpr dim_t kPrefillChunkRungs[] = { 1024, 512, 256, 128, 64 };

            bool wants_score_rows{ true };
            std::size_t weight_bytes{ 100 * kMiB };
            std::size_t bytes_a_position{ 1024 };
            std::size_t bytes_a_chunk_row{ 4096 };

            // A score row is one value a context position for each of 16 heads, in 16 bits.
            std::size_t bytes_a_score_cell{ 32 };

            MemoryStats getRequiredMemory( const BuildContext& context ) const
            {
                const auto positions = static_cast<std::size_t>( context.inputShape()[ 1 ] );

                MemoryStats stats;
                stats.device_parameter_bytes = weight_bytes;
                stats.device_state_bytes = positions * bytes_a_position
                    + static_cast<std::size_t>( context.getResolvedPrefillSize() ) * bytes_a_chunk_row
                    + static_cast<std::size_t>( context.getPrefillScoreRows() ) * positions * bytes_a_score_cell;

                return stats;
            }

            bool wantsPrefillScoreRows() const noexcept
            {
                return wants_score_rows;
            }
        };

        BuildContext contextOf( dim_t context_length, dim_t chunk_rows )
        {
            return BuildContext( shape_t{ 1, context_length }, RuntimeMode::Inference, false )
                .withAllocationGranularity( 0 )
                .withPrefillSize( chunk_rows );
        }

        std::size_t bytesAt( const MadeUpNetwork& network, dim_t context_length, dim_t chunk_rows, dim_t score_rows )
        {
            return network.getRequiredMemory( contextOf( context_length, chunk_rows ).withPrefillScoreRows( score_rows ) ).totalDeviceBytes();
        }

        /// The least budget a block of `score_rows` is taken from: its footprint, and what the block adds again.
        std::size_t budgetFor( const MadeUpNetwork& network, dim_t context_length, dim_t chunk_rows, dim_t score_rows )
        {
            const std::size_t with = bytesAt( network, context_length, chunk_rows, score_rows );

            return with + ( with - bytesAt( network, context_length, chunk_rows, 0 ) );
        }

        DeviceReading readingOf( std::size_t free_bytes )
        {
            DeviceReading reading{ .device = DeviceId{ DeviceType::Cpu, 0 } };
            reading.free_bytes = free_bytes;
            reading.total_bytes = free_bytes;

            return reading;
        }

        std::expected<DeploymentPlan, DeviceShortfall> planOf( const MadeUpNetwork& network, DeploymentRequest request, std::size_t free_bytes )
        {
            request.withHeadroom( 0 );

            return planOnDevice( network, request, readingOf( free_bytes ), 65536, Serialization::WeightsMetadata{}, "" );
        }
    }

    TEST( PrefillScoreRowsRule, TakesTheLargestBlockThatFitsTwiceOver )
    {
        const MadeUpNetwork network;
        const BuildContext context = contextOf( 16384, 1024 );

        for ( const dim_t rows : kPrefillScoreRowRungs )
        {
            const std::size_t exactly = budgetFor( network, 16384, 1024, rows );

            EXPECT_EQ( choosePrefillScoreRows( network, context, exactly ), rows ) << "a budget of exactly " << rows << " rows";

            // One byte short of a block is the block below it, and under the floor no block at all.
            const dim_t below = rows == 32 ? 0 : rows / 2;

            EXPECT_EQ( choosePrefillScoreRows( network, context, exactly - 1 ), below ) << "one byte short of " << rows << " rows";
        }

        // A budget the block fits once, and not twice, buys the block below it.
        EXPECT_EQ( choosePrefillScoreRows( network, context, bytesAt( network, 16384, 1024, 1024 ) ), 512 );
    }

    TEST( PrefillScoreRowsRule, IsNeverLongerThanTheChunk )
    {
        const MadeUpNetwork network;
        const std::size_t plenty = std::size_t{ 1 } << 40;

        EXPECT_EQ( choosePrefillScoreRows( network, contextOf( 16384, 1024 ), plenty ), 1024 );
        EXPECT_EQ( choosePrefillScoreRows( network, contextOf( 16384, 256 ), plenty ), 256 );
        EXPECT_EQ( choosePrefillScoreRows( network, contextOf( 16384, 64 ), plenty ), 64 );

        // A context shorter than every chunk rung is one chunk: 48 rows hold a block of 32, 24 hold none.
        EXPECT_EQ( choosePrefillScoreRows( network, contextOf( 48, 48 ), plenty ), 32 );
        EXPECT_EQ( choosePrefillScoreRows( network, contextOf( 24, 24 ), plenty ), 0 );
    }

    TEST( PrefillScoreRowsRule, GivesNoBlockToANetworkThatWantsNone )
    {
        MadeUpNetwork network;
        network.wants_score_rows = false;

        EXPECT_EQ( choosePrefillScoreRows( network, contextOf( 16384, 1024 ), std::size_t{ 1 } << 40 ), 0 );
        EXPECT_EQ( choosePrefillScoreRows( network, contextOf( 16384, 1024 ), 0 ), 0 );
    }

    TEST( PrefillScoreRowsRule, TakesTheLargestWhereTheDeviceCannotSay )
    {
        const MadeUpNetwork network;

        EXPECT_EQ( choosePrefillScoreRows( network, contextOf( 16384, 512 ), 0 ), 512 );
    }

    TEST( PrefillScoreRowsRule, APlansContextAndChunkAreThoseOfANetworkThatWantsNoBlock )
    {
        const MadeUpNetwork wanting;
        MadeUpNetwork fused;
        fused.wants_score_rows = false;

        DeploymentRequest automatic;
        automatic.withAutomaticContextLength();

        dim_t plans_with_a_block = 0;
        dim_t plans_without = 0;

        // From a budget nothing fits to one every context fits with a whole chunk as its block.
        for ( std::size_t free_bytes = 96 * kMiB; free_bytes <= 2560 * kMiB; free_bytes += 8 * kMiB )
        {
            const auto with = planOf( wanting, automatic, free_bytes );
            const auto without = planOf( fused, automatic, free_bytes );

            ASSERT_EQ( with.has_value(), without.has_value() ) << free_bytes;

            if ( !with )
                continue;

            EXPECT_EQ( with->contextLength(), without->contextLength() ) << free_bytes;
            EXPECT_EQ( with->prefillChunkRows(), without->prefillChunkRows() ) << free_bytes;
            EXPECT_EQ( with->contextLimitedBy(), without->contextLimitedBy() ) << free_bytes;
            EXPECT_EQ( without->prefillScoreRows(), 0 );

            // The plan's footprint is the network's at the block it records, it fits twice over, and the next block up would not.
            const dim_t rows = with->prefillScoreRows();
            const dim_t context_length = with->contextLength();
            const dim_t chunk_rows = with->prefillChunkRows();

            EXPECT_EQ( with->footprint().totalDeviceBytes(), bytesAt( wanting, context_length, chunk_rows, rows ) ) << free_bytes;
            EXPECT_LE( budgetFor( wanting, context_length, chunk_rows, rows ), free_bytes ) << free_bytes;
            EXPECT_LE( rows, chunk_rows );

            const dim_t next = rows == 0 ? 32 : rows * 2;

            if ( next <= chunk_rows )
            {
                EXPECT_GT( budgetFor( wanting, context_length, chunk_rows, next ), free_bytes ) << free_bytes;
            }

            EXPECT_EQ( with->buildContext().getPrefillScoreRows(), rows );
            EXPECT_EQ( with->buildContext().getPrefillSize(), chunk_rows );

            ( rows > 0 ? plans_with_a_block : plans_without ) += 1;
        }

        // Both kinds were met, or the comparison above compared nothing.
        EXPECT_GT( plans_with_a_block, 0 );
        EXPECT_GT( plans_without, 0 );
    }

    TEST( PrefillScoreRowsRule, AFixedContextTakesABlockFromWhatItLeaves )
    {
        const MadeUpNetwork network;

        DeploymentRequest fixed;
        fixed.withContextLength( 8192 );

        const std::size_t fused_bytes = bytesAt( network, 8192, 1024, 0 );
        const std::size_t with_128 = budgetFor( network, 8192, 1024, 128 );

        const auto tight = planOf( network, fixed, fused_bytes );
        const auto roomy = planOf( network, fixed, with_128 );

        ASSERT_TRUE( tight.has_value() );
        ASSERT_TRUE( roomy.has_value() );

        EXPECT_EQ( tight->prefillScoreRows(), 0 );
        EXPECT_EQ( roomy->prefillScoreRows(), 128 );
        EXPECT_EQ( roomy->contextLength(), 8192 );
        EXPECT_EQ( roomy->prefillChunkRows(), tight->prefillChunkRows() );
    }

    TEST( PrefillScoreRowsRule, ABuildContextRefusesNegativeRowsAndCarriesNoneToAChild )
    {
        const BuildContext context = contextOf( 4096, 512 ).withPrefillScoreRows( 64 );

        EXPECT_EQ( context.getPrefillScoreRows(), 64 );
        EXPECT_EQ( context.withShape( shape_t{ 1, 2048 } ).getPrefillScoreRows(), 64 );
        EXPECT_EQ( context.forChild( shape_t{ 1, 4096, 8 } ).getPrefillScoreRows(), 0 );
        EXPECT_EQ( BuildContext().getPrefillScoreRows(), 0 );
        EXPECT_THROW( (void)context.withPrefillScoreRows( -1 ), std::invalid_argument );
    }
}
