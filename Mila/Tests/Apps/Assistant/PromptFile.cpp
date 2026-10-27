/**
 * @file PromptFile.cpp
 * @brief The Mila app's activity prompts load as shipped, each tool's schema reaches the model as its file writes it,
 *        a prompt that is wrong is refused by name, and a call is shown by its tool's summary (MilaApp.md 3.1).
 */

#include <gtest/gtest.h>
#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

import Assistant.PromptTool;
import Assistant.PromptFile;

namespace Mila::Tests::Apps::Assistant
{
    using namespace Mila::Assistant;

    namespace
    {
        /// The prompts the app ships, found from this file's own place in the tree.
        std::filesystem::path shippedPrompts()
        {
            const std::filesystem::path here = std::filesystem::path( std::source_location::current().file_name() ).parent_path();

            return here.parent_path().parent_path().parent_path() / "Apps" / "Assistant" / "Prompts";
        }

        class PromptFileTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                static std::atomic<int> counter{ 0 };

                folder_ = std::filesystem::temp_directory_path() / std::format( "mila_prompt_test_{}", counter++ );
                std::filesystem::remove_all( folder_ );
                std::filesystem::create_directories( folder_ );
            }

            void TearDown() override
            {
                std::error_code ignored;
                std::filesystem::remove_all( folder_, ignored );
            }

            std::filesystem::path store( std::string_view text ) const
            {
                const std::filesystem::path path = folder_ / "prompt.json";
                std::ofstream( path, std::ios::binary ) << text;

                return path;
            }

            /// The message the load refused with, or empty when it loaded.
            std::string refusal( std::string_view text ) const
            {
                try
                {
                    (void)PromptFile::load( store( text ) );
                }
                catch ( const std::runtime_error& error )
                {
                    return error.what();
                }

                return {};
            }

            std::filesystem::path folder_;
        };
    }

    TEST( PromptFileShipped, EveryActivitysPromptLoadsWithItsTools )
    {
        const std::vector<std::pair<std::string_view, std::size_t>> expected{
            { "ask.json", 0 }, { "files.json", 6 }, { "research.json", 1 }, { "research-search.json", 2 },
            { "tasks.json", 9 }, { "tasks-search.json", 10 } };

        for ( const auto& [name, tools] : expected )
        {
            const PromptFile prompt = PromptFile::load( shippedPrompts() / name );

            EXPECT_FALSE( prompt.system_prompt.empty() ) << name;
            EXPECT_EQ( prompt.tools.size(), tools ) << name;

            for ( const PromptTool& tool : prompt.tools )
            {
                EXPECT_TRUE( tool.parameters.starts_with( R"({"type":"object")" ) ) << name << " " << tool.name;
                EXPECT_FALSE( tool.summary.empty() ) << name << " " << tool.name;
            }
        }
    }

    TEST_F( PromptFileTest, AToolsSchemaKeepsTheKeysInTheOrderTheFileWritesThem )
    {
        const PromptFile prompt = PromptFile::load( store( R"({ "system_prompt": "Help.", "tools": [ { "name": "pick",
            "description": "Pick one.", "parameters": { "type": "object", "properties": { "zebra": { "type": "string",
            "description": "Last." }, "apple": { "type": "string", "description": "First." } }, "required": [ "zebra" ] } } ] })" ) );

        ASSERT_EQ( prompt.tools.size(), 1u );
        EXPECT_EQ( prompt.tools.front().parameters,
            R"({"type":"object","properties":{"zebra":{"type":"string","description":"Last."},"apple":{"type":"string","description":"First."}},"required":["zebra"]})" );
    }

    TEST_F( PromptFileTest, APromptWithNoSystemPromptIsRefused )
    {
        EXPECT_NE( refusal( R"({ "tools": [] })" ).find( "system_prompt" ), std::string::npos );
    }

    TEST_F( PromptFileTest, AToolWithNoDescriptionIsRefusedByName )
    {
        const std::string message = refusal( R"({ "system_prompt": "Help.", "tools": [ { "name": "pick" } ] })" );

        EXPECT_NE( message.find( "pick" ), std::string::npos ) << message;
        EXPECT_NE( message.find( "description" ), std::string::npos ) << message;
    }

    TEST_F( PromptFileTest, AToolWhoseParametersAreNotAnObjectIsRefusedByName )
    {
        const std::string message = refusal(
            R"({ "system_prompt": "Help.", "tools": [ { "name": "pick", "description": "Pick.", "parameters": [] } ] })" );

        EXPECT_NE( message.find( "pick" ), std::string::npos ) << message;
    }

    TEST_F( PromptFileTest, AFileThatIsNotJsonIsRefused )
    {
        EXPECT_FALSE( refusal( "{ not json" ).empty() );
    }

    TEST( CallSummary, ACallIsShownByItsToolsSummaryWithTheArgumentsInIt )
    {
        const std::vector<PromptTool> tools{
            { "read_file", "Read a file.", "{}", "Reading {path} from line {start_line}" },
            { "list_folder", "List a folder.", "{}", "" } };

        EXPECT_EQ( callSummary( tools, "read_file", R"({"path":"notes.txt","start_line":3})" ), "Reading notes.txt from line 3" );
        EXPECT_EQ( callSummary( tools, "read_file", "not json" ), "Reading {path} from line {start_line}" );
        EXPECT_EQ( callSummary( tools, "list_folder", R"({"path":"."})" ), "Calling list_folder" );
        EXPECT_EQ( callSummary( tools, "no_such_tool", "{}" ), "Calling no_such_tool" );
    }
}
