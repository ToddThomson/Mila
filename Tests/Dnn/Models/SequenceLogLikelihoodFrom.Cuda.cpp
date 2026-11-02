/**
 * @file SequenceLogLikelihoodFrom.Cuda.cpp
 * @brief A sequence scored from a rewound position against the same positions scored from position 0, per family.
 *
 * Scoring from an offset is what lets a continuation be scored after a cached conversation without prefilling it
 * again (ContextProfile.md 4.3). The reference is G2's arithmetic: the log-likelihood of a sequence less that of its
 * prefix one token past the offset leaves exactly the positions the offset call scores.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <memory>
#include <unordered_set>
#include <vector>

import Mila;

#include "Measurement/LogLikelihoodHarness.h"
#include "Common/TinyDecodeNetworks.h"

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using TinyLlama = LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16>;
        using TinyGemma = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::NoWeightQuant, GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::GemmaSlidingKvPolicy>;
        using TinyQwen = QwenTransformer<DeviceType::Cuda, TensorDataType::BF16>;

        constexpr dim_t kContextLength = 256;
        constexpr dim_t kOffset = 40;
        constexpr dim_t kLength = 57;

        std::vector<std::int32_t> sequence( dim_t length )
        {
            std::vector<std::int32_t> tokens( static_cast<std::size_t>( length ) );

            for ( std::size_t i = 0; i < tokens.size(); ++i )
                tokens[ i ] = static_cast<std::int32_t>( ( 11 + 29 * i ) % Common::kTinyVocabulary );

            return tokens;
        }

        /**
         * The offset call after a rewind against two calls from position 0. The two differ only in where the chunk
         * boundaries fall, so they agree to BF16 rounding of the positions scored, not bit for bit. Qwen's rewind
         * is a return to the position it saved, so every family saves first; on the others the save keeps nothing.
         */
        template<typename TNetwork, typename TConfig>
        void expectOffsetScoreMatchesTheDifference( const TConfig& config )
        {
            const auto tokens = sequence( kLength );
            const std::vector<std::int32_t> prefix( tokens.begin(), tokens.begin() + kOffset );
            const std::vector<std::int32_t> prefix_and_one( tokens.begin(), tokens.begin() + kOffset + 1 );

            auto network = Common::buildTinyNetwork<TNetwork>( config, kContextLength );

            const double whole = Measurement::sequenceLogLikelihoodOf( *network, tokens ).total_log_probability;
            const double before = Measurement::sequenceLogLikelihoodOf( *network, prefix_and_one ).total_log_probability;

            (void)Measurement::hostLogits( *network, network->prefill( Measurement::deviceTokens( *network, prefix ) ) );
            ASSERT_EQ( network->savePosition(), kOffset );
            ASSERT_TRUE( network->rewindKvCache( kOffset ) );

            const SequenceLogLikelihood resumed = network->sequenceLogLikelihoodFrom( Measurement::deviceTokens( *network, tokens ), kOffset );

            ASSERT_TRUE( std::isfinite( whole ) && std::isfinite( before ) && std::isfinite( resumed.total_log_probability ) );
            EXPECT_EQ( resumed.scored_positions, kLength - 1 - kOffset );

            const double expected = whole - before;

            // Measured 2026-10-03 on these networks: gaps of 0.012 (Llama), 0 (Gemma) and 0.020 (Qwen) nats in about
            // 105. A position scored twice or not at all moves the total by several nats a position.
            EXPECT_NEAR( resumed.total_log_probability, expected, 0.002 * std::fabs( expected ) )
                << "scoring from the rewound offset disagrees with the same positions scored from position 0";
        }
    }

    class SequenceLogLikelihoodFromCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "Needs a CUDA device";
            }
        }
    };

    TEST_F( SequenceLogLikelihoodFromCudaTests, Llama_MatchesTheDifferenceOfTwoWholeScores )
    {
        expectOffsetScoreMatchesTheDifference<TinyLlama>( Common::tinyLlamaConfig( kContextLength ) );
    }

    TEST_F( SequenceLogLikelihoodFromCudaTests, Gemma_MatchesTheDifferenceOfTwoWholeScores )
    {
        expectOffsetScoreMatchesTheDifference<TinyGemma>( Common::tinyGemmaConfig( kContextLength ) );
    }

    TEST_F( SequenceLogLikelihoodFromCudaTests, Qwen_MatchesTheDifferenceOfTwoWholeScores )
    {
        expectOffsetScoreMatchesTheDifference<TinyQwen>( Common::tinyQwenConfig( kContextLength ) );
    }

    TEST_F( SequenceLogLikelihoodFromCudaTests, AnOffsetThatLeavesNothingToScoreIsRefused )
    {
        auto network = Common::buildTinyNetwork<TinyLlama>( Common::tinyLlamaConfig( kContextLength ), kContextLength );
        const auto tokens = Measurement::deviceTokens( *network, sequence( 8 ) );

        EXPECT_THROW( (void)network->sequenceLogLikelihoodFrom( tokens, 7 ), std::invalid_argument );
        EXPECT_THROW( (void)network->sequenceLogLikelihoodFrom( tokens, -1 ), std::invalid_argument );
    }
}
