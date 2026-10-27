/**
 * @file SettingsFile.cpp
 * @brief The Mila app keeps its settings in a file of its own: what it writes it reads back, and a setting it cannot
 *        read is said and left at its default while the rest still apply (MilaApp.md 3.7).
 */

#include <gtest/gtest.h>
#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

import Mila;
import Assistant.Settings;
import Assistant.SettingsFile;

namespace Mila::Tests::Apps::Assistant
{
    using namespace Mila::Assistant;

    namespace
    {
        class SettingsFileTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                static std::atomic<int> counter{ 0 };

                folder_ = std::filesystem::temp_directory_path() / std::format( "mila_settings_test_{}", counter++ );
                std::filesystem::remove_all( folder_ );
            }

            void TearDown() override
            {
                std::error_code ignored;
                std::filesystem::remove_all( folder_, ignored );
            }

            std::filesystem::path path() const
            {
                return folder_ / "settings.json";
            }

            void store( std::string_view text ) const
            {
                std::filesystem::create_directories( folder_ );
                std::ofstream( path(), std::ios::binary ) << text;
            }

            static bool mentions( const std::vector<std::string>& problems, std::string_view key )
            {
                for ( const std::string& problem : problems )
                {
                    if ( problem.find( key ) != std::string::npos )
                    {
                        return true;
                    }
                }

                return false;
            }

            std::filesystem::path folder_;
        };
    }

    TEST_F( SettingsFileTest, NoFileIsEverySettingAtItsDefault )
    {
        std::vector<std::string> problems;
        const Settings read = SettingsFile( path() ).read( problems );

        EXPECT_TRUE( problems.empty() );
        EXPECT_TRUE( read.model.empty() );
        EXPECT_FALSE( read.context_length.has_value() );
        EXPECT_TRUE( read.thinking );
        EXPECT_TRUE( read.effort.empty() );
        EXPECT_EQ( read.detail, "off" );
        EXPECT_FALSE( read.sampling.temperature.has_value() );
        EXPECT_FALSE( read.sampling.top_k.has_value() );
    }

    TEST_F( SettingsFileTest, WhatIsWrittenIsReadBack )
    {
        Settings settings;
        settings.model = "gemma-4-12b-it-q4_0";
        settings.context_length = 32768;
        settings.thinking = false;
        settings.effort = "medium";
        settings.detail = "all";
        settings.sampling.temperature = 0.5f;
        settings.sampling.top_k = 20;
        settings.sampling.top_p = 0.9f;
        settings.sampling.presence_penalty = -1.5f;

        SettingsFile( path() ).write( settings );

        std::vector<std::string> problems;
        const Settings read = SettingsFile( path() ).read( problems );

        EXPECT_TRUE( problems.empty() );
        EXPECT_EQ( read.model, settings.model );
        EXPECT_EQ( read.context_length, settings.context_length );
        EXPECT_EQ( read.thinking, settings.thinking );
        EXPECT_EQ( read.effort, settings.effort );
        EXPECT_EQ( read.detail, settings.detail );
        EXPECT_EQ( read.sampling.temperature, settings.sampling.temperature );
        EXPECT_EQ( read.sampling.top_k, settings.sampling.top_k );
        EXPECT_EQ( read.sampling.top_p, settings.sampling.top_p );
        EXPECT_EQ( read.sampling.presence_penalty, settings.sampling.presence_penalty );
    }

    TEST_F( SettingsFileTest, AnAutomaticContextIsWrittenAsAutoAndReadAsNone )
    {
        SettingsFile( path() ).write( Settings{} );

        std::ifstream file( path(), std::ios::binary );
        const std::string text( ( std::istreambuf_iterator<char>( file ) ), std::istreambuf_iterator<char>() );

        EXPECT_NE( text.find( R"("context_length": "auto")" ), std::string::npos ) << text;

        std::vector<std::string> problems;
        EXPECT_FALSE( SettingsFile( path() ).read( problems ).context_length.has_value() );
        EXPECT_TRUE( problems.empty() );
    }

    TEST_F( SettingsFileTest, ASettingThatCannotBeReadIsSaidAndTheRestStillApply )
    {
        store( R"({ "model": "qwen3.8-27b-fp4", "detail": "loud", "context_length": 0,
            "sampling": { "temperature": 0.5, "top_p": 3, "top_k": 1.5 } })" );

        std::vector<std::string> problems;
        const Settings read = SettingsFile( path() ).read( problems );

        EXPECT_EQ( problems.size(), 4u );
        EXPECT_TRUE( mentions( problems, "detail" ) );
        EXPECT_TRUE( mentions( problems, "context_length" ) );
        EXPECT_TRUE( mentions( problems, "sampling.top_p" ) );
        EXPECT_TRUE( mentions( problems, "sampling.top_k" ) );

        EXPECT_EQ( read.model, "qwen3.8-27b-fp4" );
        EXPECT_EQ( read.sampling.temperature, 0.5f );
        EXPECT_EQ( read.detail, "off" );
        EXPECT_FALSE( read.context_length.has_value() );
        EXPECT_FALSE( read.sampling.top_p.has_value() );
        EXPECT_FALSE( read.sampling.top_k.has_value() );
    }

    TEST_F( SettingsFileTest, AFileThatIsNotJsonIsSaidAndEverySettingIsAtItsDefault )
    {
        store( "{ not json" );

        std::vector<std::string> problems;
        const Settings read = SettingsFile( path() ).read( problems );

        ASSERT_EQ( problems.size(), 1u );
        EXPECT_NE( problems.front().find( "settings.json" ), std::string::npos ) << problems.front();
        EXPECT_TRUE( read.model.empty() );
    }

    TEST_F( SettingsFileTest, AWriteReplacesTheFileWholeAndLeavesNoPartOfItBehind )
    {
        Settings first;
        first.model = "first";
        SettingsFile( path() ).write( first );

        Settings second;
        second.model = "second";
        SettingsFile( path() ).write( second );

        std::vector<std::string> problems;
        EXPECT_EQ( SettingsFile( path() ).read( problems ).model, "second" );

        std::filesystem::path partial = path();
        partial += ".partial";
        EXPECT_FALSE( std::filesystem::exists( partial ) );
    }
}
