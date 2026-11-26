/**
 * @file KvCacheChoice.cpp
 * @brief The harness runs a model with the FP8 cache only where the uncompressed cache leaves it short of the context
 *        agentic work is measured at and the FP8 cache reaches it (AssistantOnboarding.md 4.4, round 12).
 */

#include <gtest/gtest.h>
#include <optional>

import Mila;
import Assistant.KvCacheChoice;

namespace Mila::Tests::Apps::Assistant
{
    using Mila::Assistant::chooseKvCache;
    using Mila::Dnn::KvCacheCompression;
    using Mila::Deployment::kAgenticContextLength;

    // Gemma 4 26B-A4B Q4_0 on the 16 GB card, as the planner priced it on 2026-11-25: 37,888 uncompressed, 65,536 FP8.
    TEST( KvCacheChoiceTest, AModelTheFp8CacheTakesToTheAgenticDepthRunsWithIt )
    {
        EXPECT_EQ( chooseKvCache( 37888, 65536 ), KvCacheCompression::FP8 );
        EXPECT_EQ( chooseKvCache( std::nullopt, 65536 ), KvCacheCompression::FP8 ) << "fitting only with the FP8 cache";
    }

    // Gemma 4 12B Q4_0 there plans 262,144 uncompressed.
    TEST( KvCacheChoiceTest, AModelThatReachesTheAgenticDepthUncompressedStaysUncompressed )
    {
        EXPECT_EQ( chooseKvCache( 262144, 262144 ), KvCacheCompression::None );
        EXPECT_EQ( chooseKvCache( kAgenticContextLength, kAgenticContextLength + 1024 ), KvCacheCompression::None );
    }

    TEST( KvCacheChoiceTest, AnFp8CacheThatStillFallsShortIsNotTaken )
    {
        EXPECT_EQ( chooseKvCache( 16384, 32768 ), KvCacheCompression::None ) << "longer, but still no agent";
        EXPECT_EQ( chooseKvCache( 16384, std::nullopt ), KvCacheCompression::None ) << "no FP8 cache for it";
        EXPECT_EQ( chooseKvCache( std::nullopt, std::nullopt ), KvCacheCompression::None );
    }
}
