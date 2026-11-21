/**
 * @file ProjectStore.cpp
 * @brief The workspace keeps each project in a folder of its own, named only by an id it gave, in the folder each
 *        platform keeps an application's data in (AssistantInterface.md 8).
 */

#include <gtest/gtest.h>
#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

import nlohmann.json;
import Assistant.ProjectStore;
import Assistant.StoreFile;
import Assistant.WorkspaceFolder;

namespace Mila::Tests::Apps::Assistant
{
    using namespace Mila::Assistant;

    namespace
    {
        class ProjectStoreTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                static std::atomic<int> counter{ 0 };

                folder_ = std::filesystem::temp_directory_path() / std::format( "mila_project_store_test_{}", counter++ );
                std::filesystem::remove_all( folder_ );
            }

            void TearDown() override
            {
                std::error_code ignored;
                std::filesystem::remove_all( folder_, ignored );
            }

            static nlohmann::json project( const std::string& id, const std::string& name, const std::string& updated )
            {
                return { { "v", 1 }, { "id", id }, { "name", name }, { "purpose", "" }, { "instructions", "" },
                    { "folders", nlohmann::json::array() }, { "created", updated }, { "updated", updated } };
            }

            std::filesystem::path folder_;
        };
    }

    TEST_F( ProjectStoreTest, AProjectsUnderstandingIsKeptBesideItAndGoesWithIt )
    {
        const ProjectStore store( folder_ );
        const std::string id = store.newIdentifier();

        EXPECT_TRUE( store.understanding( id )[ "current" ].is_null() ) << "a project not stored has an understanding";
        EXPECT_THROW( store.saveUnderstanding( id, { { "v", 1 } } ), std::runtime_error ) << "a note brought a project back";

        store.save( project( id, "Garden", "2026-11-21T10:00:00.000Z" ) );

        EXPECT_TRUE( store.understanding( id )[ "current" ].is_null() );
        EXPECT_TRUE( store.understanding( id )[ "waiting" ].is_null() );

        const nlohmann::json kept = { { "v", 1 }, { "current", { { "id", "note-1" }, { "summary", "Tomatoes." } } }, { "waiting", nullptr } };
        store.saveUnderstanding( id, kept );

        EXPECT_EQ( store.understanding( id ), kept );
        EXPECT_TRUE( std::filesystem::exists( folder_ / id / "understanding.json" ) );

        ASSERT_TRUE( store.remove( id ) );
        EXPECT_TRUE( store.understanding( id )[ "current" ].is_null() );
    }

    TEST_F( ProjectStoreTest, WhatIsSavedIsListedMostRecentlyChangedFirst )
    {
        const ProjectStore store( folder_ );
        const std::string first = store.newIdentifier();
        const std::string second = first + "b";

        store.save( project( first, "Kyoto in November", "2026-11-19T10:00:00.000Z" ) );
        store.save( project( second, "Calculator app", "2026-11-19T11:00:00.000Z" ) );

        const auto projects = store.list();

        ASSERT_EQ( projects.size(), 2u );
        EXPECT_EQ( projects[ 0 ][ "name" ], "Calculator app" );
        EXPECT_EQ( projects[ 1 ][ "name" ], "Kyoto in November" );
        EXPECT_EQ( store.load( first ).value()[ "name" ], "Kyoto in November" );
        EXPECT_TRUE( std::filesystem::exists( folder_ / first / "project.json" ) );
        EXPECT_FALSE( std::filesystem::exists( folder_ / first / "project.json.partial" ) );
    }

    TEST_F( ProjectStoreTest, AnIdItDidNotGiveNamesNoProject )
    {
        const ProjectStore store( folder_ );

        EXPECT_THROW( store.save( project( "../outside", "x", "2026-11-19T10:00:00.000Z" ) ), std::runtime_error );
        EXPECT_THROW( store.save( project( "Project-1", "x", "2026-11-19T10:00:00.000Z" ) ), std::runtime_error );
        EXPECT_FALSE( store.load( "../outside" ).has_value() );
        EXPECT_FALSE( store.remove( ".." ) );
        EXPECT_THROW( store.history( "a/b" ), std::invalid_argument );
        EXPECT_FALSE( std::filesystem::exists( folder_.parent_path() / "outside" ) );
    }

    TEST_F( ProjectStoreTest, AFolderWhoseRecordNamesAnotherIdIsNotThatProject )
    {
        const ProjectStore store( folder_ );
        const std::string id = store.newIdentifier();

        writeStoreFile( folder_ / id / "project.json", project( "project-other", "Moved in", "2026-11-19T10:00:00.000Z" ).dump() );
        std::filesystem::create_directories( folder_ / "not-a-project" );

        EXPECT_FALSE( store.load( id ).has_value() );
        EXPECT_TRUE( store.list().empty() );
    }

    TEST_F( ProjectStoreTest, RemovingAProjectRemovesItsFolderAndHistory )
    {
        const ProjectStore store( folder_ );
        const std::string id = store.newIdentifier();

        store.save( project( id, "Kyoto in November", "2026-11-19T10:00:00.000Z" ) );
        store.history( id ).append( { { "kind", "project.made" }, { "time", "2026-11-19T10:00:00.000Z" } } );

        ASSERT_EQ( store.history( id ).read().size(), 1u );
        EXPECT_TRUE( store.remove( id ) );
        EXPECT_FALSE( std::filesystem::exists( folder_ / id ) );
        EXPECT_FALSE( store.load( id ).has_value() );
        EXPECT_FALSE( store.remove( id ) );
    }

    TEST( StoreFileTest, AnIdIsItsKindTheTimeAndHexInTheCharactersAnIdMayHave )
    {
        const std::string id = newStoreIdentifier( "project" );

        EXPECT_TRUE( id.starts_with( "project-" ) );
        EXPECT_EQ( id.size(), std::string( "project-20261119100000-abcdef" ).size() );
        EXPECT_TRUE( isStoreIdentifier( id ) );
        EXPECT_FALSE( isStoreIdentifier( "" ) );
        EXPECT_FALSE( isStoreIdentifier( "project/1" ) );
        EXPECT_FALSE( isStoreIdentifier( "project.1" ) );
        EXPECT_FALSE( isStoreIdentifier( std::string( 65, 'a' ) ) );
    }

    TEST( WorkspaceFolderTest, EachPlatformNamesItsFoldersAsItsOwnAre )
    {
        const std::filesystem::path workspace = std::filesystem::path( "workspace" );

#if defined( _WIN32 ) || defined( __APPLE__ )
        EXPECT_EQ( conversationsFolder( workspace ), workspace / "Conversations" );
        EXPECT_EQ( projectsFolder( workspace ), workspace / "Projects" );
#else
        EXPECT_EQ( conversationsFolder( workspace ), workspace / "conversations" );
        EXPECT_EQ( projectsFolder( workspace ), workspace / "projects" );
#endif

        EXPECT_EQ( activityFile( workspace ), workspace / "activity.jsonl" );

#if defined( __APPLE__ )
        const std::filesystem::path expected = std::filesystem::path( "Library" ) / "Application Support" / "Mila" / "Assistant";
        EXPECT_TRUE( defaultWorkspaceFolder().relative_path().string().ends_with( expected.string() ) );
#elif defined( _WIN32 )
        EXPECT_EQ( defaultWorkspaceFolder().filename(), "Assistant" );
        EXPECT_EQ( defaultWorkspaceFolder().parent_path().filename(), "Mila" );
#else
        EXPECT_EQ( defaultWorkspaceFolder().filename(), "assistant" );
        EXPECT_EQ( defaultWorkspaceFolder().parent_path().filename(), "mila" );
#endif
    }
}
