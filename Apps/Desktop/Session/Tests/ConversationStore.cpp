/**
 * @file ConversationStore.cpp
 * @brief A conversation names its project by one field, and one stored before projects is in none
 *        (AssistantInterface.md 8).
 */

#include <gtest/gtest.h>
#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <system_error>

import nlohmann.json;
import Desktop.ConversationStore;

namespace Mila::Tests::Apps::Desktop
{
    using namespace Mila::Desktop;

    namespace
    {
        class ConversationStoreTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                static std::atomic<int> counter{ 0 };

                folder_ = std::filesystem::temp_directory_path() / std::format( "mila_conversation_store_test_{}", counter++ );
                std::filesystem::remove_all( folder_ );
            }

            void TearDown() override
            {
                std::error_code ignored;
                std::filesystem::remove_all( folder_, ignored );
            }

            static nlohmann::json conversation( const std::string& id, const std::string& updated )
            {
                return { { "v", 1 }, { "id", id }, { "title", "A chat" }, { "model", "m" }, { "folder", nullptr },
                    { "created", updated }, { "updated", updated }, { "messages", nlohmann::json::array() } };
            }

            std::filesystem::path folder_;
        };
    }

    TEST_F( ConversationStoreTest, AConversationStoredBeforeProjectsIsInNone )
    {
        const ConversationStore store( folder_ );
        const std::string id = store.newIdentifier();

        store.save( conversation( id, "2026-11-19T10:00:00.000Z" ) );

        const nlohmann::json items = store.list();

        ASSERT_EQ( items.size(), 1u );
        EXPECT_TRUE( items[ 0 ][ "project" ].is_null() );
        EXPECT_EQ( items[ 0 ][ "folder_from_project" ], false );
    }

    TEST_F( ConversationStoreTest, ItsProjectIsListedAndKeptAsItWasSaved )
    {
        const ConversationStore store( folder_ );
        const std::string id = store.newIdentifier();

        nlohmann::json stored = conversation( id, "2026-11-19T10:00:00.000Z" );
        stored[ "project" ] = "project-20261119100000-abcdef";
        stored[ "folder_from_project" ] = true;
        store.save( stored );

        EXPECT_EQ( store.list()[ 0 ][ "project" ], "project-20261119100000-abcdef" );
        EXPECT_EQ( store.list()[ 0 ][ "folder_from_project" ], true );
        EXPECT_EQ( store.load( id ).value(), stored );
        EXPECT_TRUE( id.starts_with( "conversation-" ) );
    }

    TEST_F( ConversationStoreTest, AnIdItDidNotGiveNamesNoConversation )
    {
        const ConversationStore store( folder_ );

        EXPECT_THROW( store.save( conversation( "../outside", "2026-11-19T10:00:00.000Z" ) ), std::runtime_error );
        EXPECT_FALSE( store.load( "../outside" ).has_value() );
        EXPECT_FALSE( store.remove( "../outside" ) );
    }
}
