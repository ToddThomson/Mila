/**
 * @file SecondNameWatch.cpp
 * @brief Which files under a folder have a name elsewhere, found by the first walk and kept current by the watcher
 *        (ContainedPrograms.md 4.3): each such file found beside a file with one name, not recorded. Windows only.
 *
 * The folders are made under the system's temporary folder, inside and outside on one volume, as a hard link needs.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <random>
#include <stop_token>
#include <string>
#include <thread>

import Mila;
import AI.SecondNameWatch;

namespace Mila::Tests::AI
{
    using namespace Mila::AI;

    namespace
    {
        void write( const std::filesystem::path& file, const std::string& content )
        {
            std::filesystem::create_directories( file.parent_path() );
            std::ofstream( file, std::ios::binary | std::ios::trunc ) << content;
        }

        /// A second name for a file, its folder made first.
        void link( const std::filesystem::path& file, const std::filesystem::path& name )
        {
            std::filesystem::create_directories( name.parent_path() );
            std::filesystem::create_hard_link( file, name );
        }

        std::chrono::steady_clock::time_point soon()
        {
            return std::chrono::steady_clock::now() + std::chrono::seconds( 10 );
        }

        class SecondNameWatchTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                std::random_device seed;
                root_ = std::filesystem::canonical( std::filesystem::temp_directory_path() ) / std::format( "MilaSecondName-{:08x}", seed() );
                inside_ = root_ / "inside";
                outside_ = root_ / "outside";

                write( inside_ / "plain.txt", "one name" );
                write( outside_ / "secret.txt", "secret" );
            }

            void TearDown() override
            {
                std::error_code error;
                std::filesystem::remove_all( root_, error );
            }

            /// Refreshes until the watcher has reported a name, or two seconds pass.
            bool refreshUntil( SecondNameWatch& watch, const std::filesystem::path& name, bool recorded )
            {
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds( 2 );

                do
                {
                    EXPECT_TRUE( watch.refresh( soon(), {} ) );

                    if ( watch.files().contains( name ) == recorded )
                    {
                        return true;
                    }

                    std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
                } while ( std::chrono::steady_clock::now() < until );

                return false;
            }

            std::filesystem::path root_;
            std::filesystem::path inside_;
            std::filesystem::path outside_;
        };
    }

    TEST_F( SecondNameWatchTest, TheFirstWalkFindsAFileWithANameOutsideAndNotOneWithASingleName )
    {
        link( outside_ / "secret.txt", inside_ / "sub" / "link.txt" );

        SecondNameWatch watch( inside_ );
        ASSERT_TRUE( watch.refresh( soon(), {} ) );

        ASSERT_TRUE( watch.files().contains( inside_ / "sub" / "link.txt" ) ) << "the file with a name outside";
        EXPECT_FALSE( watch.files().contains( inside_ / "plain.txt" ) ) << "the file with one name";

        const auto& names = watch.files().at( inside_ / "sub" / "link.txt" );
        EXPECT_EQ( names.size(), 2u );
        EXPECT_NE( std::ranges::find( names, outside_ / "secret.txt" ), names.end() ) << "its name outside is listed";
    }

    TEST_F( SecondNameWatchTest, ANameMadeInsideAfterTheWalkIsFound )
    {
        SecondNameWatch watch( inside_ );
        ASSERT_TRUE( watch.refresh( soon(), {} ) );
        ASSERT_TRUE( watch.files().empty() );

        link( outside_ / "secret.txt", inside_ / "late.txt" );
        write( inside_ / "late-plain.txt", "one name" );

        EXPECT_TRUE( refreshUntil( watch, inside_ / "late.txt", true ) );
        EXPECT_FALSE( watch.files().contains( inside_ / "late-plain.txt" ) );
    }

    TEST_F( SecondNameWatchTest, ANameMadeOutsideForAFileInsideAfterTheWalkIsFound )
    {
        SecondNameWatch watch( inside_ );
        ASSERT_TRUE( watch.refresh( soon(), {} ) );

        link( inside_ / "plain.txt", outside_ / "copy-of-plain.txt" );

        EXPECT_TRUE( refreshUntil( watch, inside_ / "plain.txt", true ) );
    }

    TEST_F( SecondNameWatchTest, AFileWhoseNameOutsideIsRemovedIsDropped )
    {
        link( outside_ / "secret.txt", inside_ / "link.txt" );

        SecondNameWatch watch( inside_ );
        ASSERT_TRUE( watch.refresh( soon(), {} ) );
        ASSERT_TRUE( watch.files().contains( inside_ / "link.txt" ) );

        std::filesystem::remove( outside_ / "secret.txt" );

        EXPECT_TRUE( refreshUntil( watch, inside_ / "link.txt", false ) );
    }

    TEST_F( SecondNameWatchTest, AFolderMovedInBringsItsLinkedFileWithIt )
    {
        SecondNameWatch watch( inside_ );
        ASSERT_TRUE( watch.refresh( soon(), {} ) );

        link( outside_ / "secret.txt", outside_ / "bundle" / "deep" / "link.txt" );
        std::filesystem::create_directories( outside_ / "plain-bundle" );
        write( outside_ / "plain-bundle" / "one.txt", "one name" );
        std::filesystem::rename( outside_ / "bundle", inside_ / "bundle" );
        std::filesystem::rename( outside_ / "plain-bundle", inside_ / "plain-bundle" );

        EXPECT_TRUE( refreshUntil( watch, inside_ / "bundle" / "deep" / "link.txt", true ) );
        EXPECT_FALSE( watch.files().contains( inside_ / "plain-bundle" / "one.txt" ) );
    }

    TEST_F( SecondNameWatchTest, ARefreshPastItsDeadlineOrStoppedReportsItself )
    {
        SecondNameWatch watch( inside_ );

        EXPECT_FALSE( watch.refresh( std::chrono::steady_clock::now() - std::chrono::seconds( 1 ), {} ) );

        std::stop_source stop;
        stop.request_stop();
        EXPECT_FALSE( watch.refresh( soon(), stop.get_token() ) );

        EXPECT_TRUE( watch.refresh( soon(), {} ) ) << "a refresh with time and no stop finishes";
    }
}
