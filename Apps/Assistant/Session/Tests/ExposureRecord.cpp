/**
 * @file ExposureRecord.cpp
 * @brief What a chat read from outside survives being stored, and a damaged record never brings a chat back clear
 *        (AssistantInterface.md 5.6).
 */

#include <gtest/gtest.h>
#include <chrono>
#include <optional>
#include <string>
#include <vector>

import nlohmann.json;
import Mila;
import Assistant.ExposureRecord;

namespace Mila::Assistant::Tests
{
    using Mila::Agents::ExposureSource;

    TEST( ExposureRecord, AStoredExposureComesBackAsItWas )
    {
        const auto when = std::chrono::floor<std::chrono::milliseconds>( std::chrono::system_clock::now() );
        const std::vector<ExposureSource> exposure{
            { "read_web_page", R"({"url":"https://example.com/"})", when },
            { "read_file", R"({"path":"notes.md"})", std::nullopt } };

        // Through text, as the store writes and reads it.
        const nlohmann::json stored = nlohmann::json::parse( exposureToJson( exposure ).dump() );

        EXPECT_EQ( exposureFromJson( stored ), exposure );
    }

    TEST( ExposureRecord, AChatThatReadNothingStoresAnEmptyList )
    {
        EXPECT_EQ( exposureToJson( {} ), nlohmann::json::array() );
        EXPECT_TRUE( exposureFromJson( nlohmann::json::array() ).empty() );
    }

    TEST( ExposureRecord, ADamagedRecordStillCountsAsReadFromOutside )
    {
        EXPECT_EQ( exposureFromJson( "not a list" ).size(), 1u );

        const std::vector<ExposureSource> read = exposureFromJson( nlohmann::json::array( { 7, { { "tool", 3 } }, { { "time", "yesterday" } } } ) );

        ASSERT_EQ( read.size(), 3u );
        EXPECT_EQ( read[ 0 ].tool, "unknown" );
        EXPECT_EQ( read[ 1 ].tool, "unknown" );
        EXPECT_FALSE( read[ 2 ].when.has_value() );
    }
}
