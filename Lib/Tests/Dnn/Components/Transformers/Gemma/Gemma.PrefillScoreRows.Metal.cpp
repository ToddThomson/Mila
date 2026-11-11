/**
 * @file Gemma.PrefillScoreRows.Metal.cpp
 * @brief A Gemma network built with a block of prefill score rows on Metal: its global layers prefill unfused a block at
 *        a time, and it allocates what it was priced at (Metal/MetalGlobalAttentionPrefill.md section 8).
 *
 * The tiny network has a sliding layer (head size 256) and a global one (head size 512), the head size the unfused
 * pipeline is preferred at. The same weights are built without a block, where every layer prefills fused, and the two
 * builds' logits are compared: the unfused scores are rounded to FP16 before the softmax and the fused kernel's are not,
 * so they agree within that rounding, and a block that attends from the wrong offset moves a logit by the size of the
 * logits.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

import Mila;

#include "Measurement/LogLikelihoodHarness.h"
#include "Common/TinyDecodeNetworks.h"

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        constexpr TensorDataType kPrecision = TensorDataType::FP16;

        using TinyGemma = GemmaTransformer<DeviceType::Metal, kPrecision,
            Quant::Weight::NoWeightQuant, GemmaModel<DeviceType::Metal, kPrecision>::GemmaSlidingKvPolicy>;

        constexpr dim_t kContextLength = 256;

        // Two whole chunks of 64 and a part of a third, so that blocks start at a held depth and one is cut short.
        constexpr dim_t kPrompt = 150;

        // The logits of the two builds differ by the rounding of a score to FP16: 2.3e-3 of the logits' RMS at its largest, measured
        // on an M1 Pro at every block size, 2026-11-10. A wrong block is a difference of the order of one.
        constexpr double kRoundingBar = 1e-2;

        std::vector<std::int32_t> promptTokens()
        {
            std::vector<std::int32_t> tokens( static_cast<std::size_t>( kPrompt ) );

            for ( std::size_t i = 0; i < tokens.size(); ++i )
                tokens[ i ] = static_cast<std::int32_t>( ( 7 + 31 * i ) % Common::kTinyVocabulary );

            return tokens;
        }

        std::unique_ptr<TinyGemma> buildTiny( dim_t score_rows )
        {
            return Common::buildTinyNetwork<TinyGemma>(
                Common::tinyGemmaConfig( kContextLength ), kContextLength, Device::Metal( 0 ), score_rows );
        }

        BuildContext pricedContext( dim_t score_rows )
        {
            return BuildContext( shape_t{ 1, kContextLength }, RuntimeMode::Inference, true )
                .withPrefillSize( Common::kTinyPrefillRows )
                .withPrefillScoreRows( score_rows )
                .withAllocationGranularity( allocationGranularity( Device::Metal( 0 ) ) );
        }

        /// State and scratch, where a block's buffers are. This network's parameters are priced 9,216 bytes under what it allocates
        /// on Metal with or without a block (Issues/Untriaged.md), and are not compared here.
        void expectAllocatesWhatItWasPricedAt( const TinyGemma& built, const TinyGemma& pricing, dim_t score_rows )
        {
            const MemoryStats predicted = pricing.getRequiredMemory( pricedContext( score_rows ) );
            const MemoryStats allocated = built.getMemoryStats();

            EXPECT_EQ( predicted.device_state_bytes, allocated.device_state_bytes ) << score_rows << " score rows: state";
            EXPECT_EQ( predicted.device_scratch_bytes, allocated.device_scratch_bytes ) << score_rows << " score rows: scratch";
        }

        double largestDifferenceOverRms( const std::vector<float>& produced, const std::vector<float>& expected )
        {
            double square_sum = 0.0;
            double largest = 0.0;

            for ( std::size_t i = 0; i < expected.size(); ++i )
            {
                square_sum += static_cast<double>( expected[ i ] ) * expected[ i ];
                largest = std::max( largest, std::abs( static_cast<double>( produced[ i ] ) - expected[ i ] ) );
            }

            return largest / std::sqrt( square_sum / static_cast<double>( expected.size() ) );
        }
    }

    class GemmaPrefillScoreRowsMetal : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Metal ) == 0 )
            {
                GTEST_SKIP() << "Needs a Metal device";
            }
        }
    };

    TEST_F( GemmaPrefillScoreRowsMetal, WantsABlockForItsGlobalLayers )
    {
        const auto network = buildTiny( 0 );

        EXPECT_TRUE( network->wantsPrefillScoreRows() );
    }

    TEST_F( GemmaPrefillScoreRowsMetal, ABuildWithABlockAllocatesWhatItWasPricedAtAndTheScoreBuffersMore )
    {
        const auto fused = buildTiny( 0 );
        const auto blocked = buildTiny( 32 );
        const auto whole_chunk = buildTiny( Common::kTinyPrefillRows );

        expectAllocatesWhatItWasPricedAt( *fused, *fused, 0 );
        expectAllocatesWhatItWasPricedAt( *blocked, *blocked, 32 );
        expectAllocatesWhatItWasPricedAt( *whole_chunk, *whole_chunk, Common::kTinyPrefillRows );

        // One graph prices every block, as the planner's rule asks it to.
        expectAllocatesWhatItWasPricedAt( *blocked, *fused, 32 );

        // One score buffer of four heads across the context, in FP16, for each row of the block: under the size a device rounds.
        const std::size_t bytes_a_row = 4 * static_cast<std::size_t>( kContextLength ) * 2;

        EXPECT_EQ( blocked->getMemoryStats().device_state_bytes - fused->getMemoryStats().device_state_bytes, 32 * bytes_a_row );
        EXPECT_EQ( whole_chunk->getMemoryStats().device_state_bytes - fused->getMemoryStats().device_state_bytes,
            static_cast<std::size_t>( Common::kTinyPrefillRows ) * bytes_a_row );
    }

    TEST_F( GemmaPrefillScoreRowsMetal, ItsLogitsAreTheFusedBuildsWithinAScoresRounding )
    {
        const auto prompt = promptTokens();

        auto fused = buildTiny( 0 );
        const std::vector<float> expected = Measurement::hostLogits( *fused, fused->prefill( Measurement::deviceTokens( *fused, prompt ) ) );

        for ( const dim_t score_rows : { dim_t{ 32 }, dim_t{ 40 }, Common::kTinyPrefillRows } )
        {
            auto blocked = buildTiny( score_rows );
            const std::vector<float> produced =
                Measurement::hostLogits( *blocked, blocked->prefill( Measurement::deviceTokens( *blocked, prompt ) ) );

            ASSERT_EQ( produced.size(), expected.size() );

            const double difference = largestDifferenceOverRms( produced, expected );

            std::printf( "[blocks] %lld score rows: largest difference %.3e of the logits' RMS\n", static_cast<long long>( score_rows ), difference );

            EXPECT_LE( difference, kRoundingBar ) << "blocks of " << score_rows;
            EXPECT_GT( difference, 0.0 ) << "blocks of " << score_rows << " ran the fused kernel: the logits are the fused build's to the bit";
        }
    }

    TEST_F( GemmaPrefillScoreRowsMetal, RefusesABlockLongerThanTheChunk )
    {
        TinyGemma network( "tiny", Common::tinyGemmaConfig( kContextLength ), Device::Metal( 0 ) );

        EXPECT_THROW( network.build( BuildContext( shape_t{ 1, kContextLength }, RuntimeMode::Inference, true )
            .withPrefillSize( Common::kTinyPrefillRows ).withPrefillScoreRows( Common::kTinyPrefillRows + 1 ) ), std::invalid_argument );
    }
}
