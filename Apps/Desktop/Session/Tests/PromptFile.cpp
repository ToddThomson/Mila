/**
 * @file PromptFile.cpp
 * @brief Mila Assistant's prompt loads as shipped, its passages follow the tools offered, each tool's schema reaches
 *        the model as its file writes it, a prompt that is wrong is refused by name, and a call is shown by its tool's
 *        summary (AssistantInterface.md 5.5).
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <set>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

import Desktop.PromptTool;
import Desktop.PromptFile;
import Mila;

namespace Mila::Tests::Apps::Desktop
{
    using namespace Mila::Desktop;

    namespace
    {
        /// The prompts the app ships, found from this file's own place in the tree.
        std::filesystem::path shippedPrompts()
        {
            const std::filesystem::path here = std::filesystem::path( std::source_location::current().file_name() ).parent_path();

            return here.parent_path().parent_path() / "Prompts";
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

    TEST( PromptFileShipped, TheChatsPromptLoadsWithEveryTool )
    {
        std::vector<Mila::Agents::Tool> provided = Mila::Agents::FileTools::tools();
        provided.push_back( Mila::Agents::WebPageReader::tool() );

        const PromptFile prompt = PromptFile::load( shippedPrompts() / "general.json" );

        EXPECT_EQ( prompt.tools.size(), 14u );

        for ( const PromptTool& tool : prompt.tools )
        {
            // A tool the file only names is one Mila::Agents provides; the session takes the rest from it.
            if ( tool.description.empty() )
            {
                EXPECT_NE( std::ranges::find( provided, tool.name, &Mila::Agents::Tool::name ), provided.end() ) << tool.name;

                continue;
            }

            EXPECT_TRUE( tool.parameters.starts_with( R"({"type":"object")" ) ) << tool.name;
            EXPECT_FALSE( tool.summary.empty() ) << tool.name;
        }
    }

    // A prompt naming a tool not offered draws calls of it, so no passage names a tool the chat was not given.
    TEST( PromptFileShipped, ThePromptNamesOnlyTheToolsOffered )
    {
        const PromptFile prompt = PromptFile::load( shippedPrompts() / "general.json" );
        const std::set<std::string, std::less<>> mac{ "propose_plan", "update_step", "read_web_page" };
        const std::set<std::string, std::less<>> windows{ "propose_plan", "update_step", "read_web_page", "web_search",
            "request_folder", "list_folder", "read_file", "find_files", "search_text", "write_file", "edit_file" };

        const std::string on_mac = prompt.systemPromptFor( mac );
        const std::string on_windows = prompt.systemPromptFor( windows );

        for ( const std::string_view absent : { "web_search", "request_folder", "list_folder", "edit_file", "run_command",
            "open_place", "find_records" } )
        {
            EXPECT_EQ( on_mac.find( absent ), std::string::npos ) << absent;
        }

        EXPECT_NE( on_mac.find( "read_web_page" ), std::string::npos );
        EXPECT_NE( on_mac.find( "You cannot search the web" ), std::string::npos );
        EXPECT_NE( on_windows.find( "request_folder" ), std::string::npos );
        EXPECT_NE( on_windows.find( "web_search" ), std::string::npos );
        EXPECT_EQ( on_windows.find( "You cannot search the web" ), std::string::npos );
        EXPECT_EQ( on_windows.find( "run_command" ), std::string::npos );

        // The workspace's own tools are named only where they are offered (AssistantInterface.md 5.2).
        std::set<std::string, std::less<>> workspace = mac;
        workspace.insert( { "open_place", "find_records" } );

        const std::string with_workspace = prompt.systemPromptFor( workspace );

        EXPECT_NE( with_workspace.find( "call open_place" ), std::string::npos );
        EXPECT_NE( with_workspace.find( "call find_records" ), std::string::npos );
    }

    TEST_F( PromptFileTest, APassageIsKeptByTheToolItNamesAndLeftOutByTheOneItStandsInFor )
    {
        const PromptFile prompt = PromptFile::load( store( R"({ "system_prompt": [ "Always.",
            { "when": "read", "text": "Read." }, { "when": "read", "unless": "search", "text": "No search." } ] })" ) );

        EXPECT_EQ( prompt.systemPromptFor( {} ), "Always." );
        EXPECT_EQ( prompt.systemPromptFor( { "read" } ), "Always. Read. No search." );
        EXPECT_EQ( prompt.systemPromptFor( { "read", "search" } ), "Always. Read." );
    }

    TEST_F( PromptFileTest, ASystemPromptWrittenAsOneTextIsOnePassage )
    {
        const PromptFile prompt = PromptFile::load( store( R"({ "system_prompt": "Help." })" ) );

        EXPECT_EQ( prompt.systemPromptFor( { "read" } ), "Help." );
    }

    TEST_F( PromptFileTest, APassageWithNoTextIsRefused )
    {
        EXPECT_NE( refusal( R"({ "system_prompt": [ { "when": "read" } ] })" ).find( "passage" ), std::string::npos );
        EXPECT_NE( refusal( R"({ "system_prompt": [ { "when": 3, "text": "Read." } ] })" ).find( "when" ), std::string::npos );
        EXPECT_NE( refusal( R"({ "system_prompt": [] })" ).find( "system_prompt" ), std::string::npos );
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

    TEST_F( PromptFileTest, AToolNamedByAStringHasOnlyItsName )
    {
        const PromptFile prompt = PromptFile::load( store( R"({ "system_prompt": "Help.", "tools": [ "read_file" ] })" ) );

        ASSERT_EQ( prompt.tools.size(), 1u );
        EXPECT_EQ( prompt.tools.front().name, "read_file" );
        EXPECT_TRUE( prompt.tools.front().description.empty() );
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
