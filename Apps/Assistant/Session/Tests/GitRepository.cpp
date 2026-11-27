/**
 * @file GitRepository.cpp
 * @brief Whether a folder's git repository has a commit yet, read from its files (Todd, 2026-11-27: Mila offers the
 *        first commit of a project it started).
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>

import Assistant.GitRepository;

namespace Mila::Tests::Apps::Assistant
{
    using namespace Mila::Assistant;

    namespace
    {
        class GitRepositoryTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                static int counter = 0;
                folder_ = std::filesystem::temp_directory_path() / std::format( "mila_git_repository_test_{}", counter++ );
                std::filesystem::remove_all( folder_ );
                std::filesystem::create_directories( folder_ );
            }

            void TearDown() override
            {
                std::error_code ignored;
                std::filesystem::remove_all( folder_, ignored );
            }

            /// What `git init` leaves: HEAD naming a branch, and no ref for it.
            void initialize( const char* head = "ref: refs/heads/master\n" ) const
            {
                std::filesystem::create_directories( folder_ / ".git" / "refs" / "heads" );
                std::ofstream( folder_ / ".git" / "HEAD", std::ios::binary ) << head;
            }

            void write( const std::filesystem::path& relative, const char* text ) const
            {
                std::filesystem::create_directories( ( folder_ / relative ).parent_path() );
                std::ofstream( folder_ / relative, std::ios::binary ) << text;
            }

            std::filesystem::path folder_;
        };
    }

    TEST_F( GitRepositoryTest, AFolderWithNoRepositoryHasNoFirstCommitToOffer )
    {
        EXPECT_FALSE( repositoryWithoutCommits( folder_ ) );
    }

    TEST_F( GitRepositoryTest, ARepositoryJustStartedHasNoCommit )
    {
        initialize();
        EXPECT_TRUE( repositoryWithoutCommits( folder_ ) );

        // As git for Windows may write it.
        initialize( "ref: refs/heads/main\r\n" );
        EXPECT_TRUE( repositoryWithoutCommits( folder_ ) );
    }

    TEST_F( GitRepositoryTest, ABranchWithACommitHasOne )
    {
        initialize();
        write( ".git/refs/heads/master", "4b825dc642cb6eb9a060e54bf8d69288fbee4904\n" );

        EXPECT_FALSE( repositoryWithoutCommits( folder_ ) );
    }

    TEST_F( GitRepositoryTest, ABranchKeptInPackedRefsHasACommit )
    {
        initialize();
        write( ".git/packed-refs", "# pack-refs with: peeled fully-peeled sorted\r\n4b825dc642cb6eb9a060e54bf8d69288fbee4904 refs/heads/master\r\n" );

        EXPECT_FALSE( repositoryWithoutCommits( folder_ ) );
    }

    TEST_F( GitRepositoryTest, ADetachedHeadIsNotAFirstCommitToOffer )
    {
        initialize( "4b825dc642cb6eb9a060e54bf8d69288fbee4904\n" );

        EXPECT_FALSE( repositoryWithoutCommits( folder_ ) );
    }
}
