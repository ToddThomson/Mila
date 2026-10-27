/**
 * @file ToolGrant.Files.cpp
 * @brief What a grant's files reach: each access the grant refuses beside the same kind of access inside it,
 *        allowed (ToolCalling.md 3.4). Windows only, where a folder can be granted.
 *
 * Each test builds its folders under the system's temporary folder: a granted project, a sibling whose name begins
 * with the project's, and a folder outside holding a secret. Junctions need no privilege; a symbolic link needs
 * Developer Mode or an administrator, so those tests skip where one cannot be made. A mount point needs an
 * administrator and is not tested: it leads to a final path as a junction does.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

import Mila;

namespace Mila::Tests::AI
{
    using namespace Mila::AI;

    namespace
    {
        std::string text( const std::filesystem::path& path )
        {
            const std::u8string utf8 = path.generic_u8string();

            return std::string( utf8.begin(), utf8.end() );
        }

        void write( const std::filesystem::path& file, std::string_view content )
        {
            std::filesystem::create_directories( file.parent_path() );
            std::ofstream stream( file, std::ios::binary | std::ios::trunc );
            stream.write( content.data(), static_cast<std::streamsize>( content.size() ) );
        }

        std::string contentOf( const std::filesystem::path& file )
        {
            std::ifstream stream( file, std::ios::binary );

            return std::string( std::istreambuf_iterator<char>( stream ), std::istreambuf_iterator<char>() );
        }

        /// That an access is refused for the reason given, not for another such as the file not being there.
        template <typename TAccess>
        void expectRefused( TAccess&& access, std::string_view reason )
        {
            try
            {
                access();
                ADD_FAILURE() << "allowed; expected refused because it " << reason;
            }
            catch ( const std::runtime_error& error )
            {
                EXPECT_NE( std::string_view( error.what() ).find( reason ), std::string_view::npos ) << error.what();
            }
        }

        constexpr std::string_view kOutside = "lies outside";

        /// A junction, which any account can make; true when it was made.
        bool makeJunction( const std::filesystem::path& link, const std::filesystem::path& target )
        {
            const std::string command = std::format( "mklink /J \"{}\" \"{}\" >nul 2>&1", link.string(), target.string() );

            return std::system( command.c_str() ) == 0 && std::filesystem::exists( std::filesystem::symlink_status( link ) );
        }

        class ToolGrantFilesTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                std::random_device seed;
                root_ = std::filesystem::temp_directory_path() / std::format( "MilaToolGrant-{:08x}", seed() );

                project_ = root_ / "project";
                sibling_ = root_ / "project2";
                outside_ = root_ / "outside";

                write( project_ / "inside.txt", "inside" );
                write( project_ / "sub" / "a.txt", "a" );
                write( sibling_ / "secret.txt", "sibling secret" );
                write( outside_ / "secret.txt", "secret" );
            }

            void TearDown() override
            {
                std::error_code error;

                // A junction is removed as a link, never by entering it.
                std::filesystem::remove_all( root_, error );
            }

            ToolGrant grant( FolderAccess access = FolderAccess::ReadWrite ) const
            {
                return ToolGrant( { GrantFolder{ project_, access } }, false );
            }

            std::filesystem::path root_;
            std::filesystem::path project_;
            std::filesystem::path sibling_;
            std::filesystem::path outside_;
        };
    }

    // ================================================================
    // Paths
    // ================================================================

    TEST_F( ToolGrantFilesTest, AnAbsolutePathOutsideIsRefused )
    {
        expectRefused( [&] { (void)grant().openFile( text( outside_ / "secret.txt" ) ); }, kOutside );
    }

    TEST_F( ToolGrantFilesTest, AnAbsolutePathInsideIsAllowed )
    {
        EXPECT_EQ( grant().openFile( text( project_ / "inside.txt" ) ).read(), "inside" );
        EXPECT_EQ( grant().openFile( "sub/a.txt" ).read(), "a" ) << "a relative path is read against the folder";
    }

    TEST_F( ToolGrantFilesTest, ASiblingWhoseNameBeginsWithTheFoldersIsRefused )
    {
        expectRefused( [&] { (void)grant().openFile( text( sibling_ / "secret.txt" ) ); }, kOutside );
    }

    TEST_F( ToolGrantFilesTest, ADifferenceOfCaseInsideIsAllowed )
    {
        EXPECT_EQ( grant().openFile( text( root_ / "PROJECT" / "INSIDE.TXT" ) ).read(), "inside" );
        EXPECT_EQ( grant().openFile( "SUB/A.TXT" ).read(), "a" );
    }

    TEST_F( ToolGrantFilesTest, AFolderThatDoesNotExistIsNeverGranted )
    {
        EXPECT_THROW( ToolGrant( { GrantFolder{ root_ / "missing", FolderAccess::Read } }, false ), std::invalid_argument );
        EXPECT_THROW( ToolGrant( { GrantFolder{ project_ / "inside.txt", FolderAccess::Read } }, false ), std::invalid_argument );
    }

    TEST_F( ToolGrantFilesTest, AFolderThatExistsIsGranted )
    {
        const ToolGrant granted = grant( FolderAccess::Read );

        ASSERT_EQ( granted.folders().size(), 1u );
        EXPECT_EQ( granted.folders().front().access, FolderAccess::Read );
        EXPECT_FALSE( granted.network() );
    }

    // ================================================================
    // Links
    // ================================================================

    TEST_F( ToolGrantFilesTest, AJunctionLeadingOutsideIsRefused )
    {
        ASSERT_TRUE( makeJunction( project_ / "out", outside_ ) );

        expectRefused( [&] { (void)grant().openFile( "out/secret.txt" ); }, kOutside );
        expectRefused( [&] { (void)grant().list( "out" ); }, kOutside );
    }

    TEST_F( ToolGrantFilesTest, AJunctionInsideIsAllowed )
    {
        ASSERT_TRUE( makeJunction( project_ / "in", project_ / "sub" ) );

        EXPECT_EQ( grant().openFile( "in/a.txt" ).read(), "a" );
        EXPECT_EQ( grant().list( "in" ).size(), 1u );
    }

    TEST_F( ToolGrantFilesTest, ASymbolicLinkLeadingOutsideIsRefused )
    {
        std::error_code error;
        std::filesystem::create_symlink( outside_ / "secret.txt", project_ / "link.txt", error );

        if ( error )
        {
            GTEST_SKIP() << "a symbolic link needs Developer Mode or an administrator: " << error.message();
        }

        expectRefused( [&] { (void)grant().openFile( "link.txt" ); }, kOutside );
    }

    TEST_F( ToolGrantFilesTest, ASymbolicLinkInsideIsAllowed )
    {
        std::error_code error;
        std::filesystem::create_symlink( project_ / "inside.txt", project_ / "link.txt", error );

        if ( error )
        {
            GTEST_SKIP() << "a symbolic link needs Developer Mode or an administrator: " << error.message();
        }

        EXPECT_EQ( grant().openFile( "link.txt" ).read(), "inside" );
    }

    TEST_F( ToolGrantFilesTest, AHardLinkedFileIsRefused )
    {
        std::filesystem::create_hard_link( outside_ / "secret.txt", project_ / "hard.txt" );

        expectRefused( [&] { (void)grant().openFile( "hard.txt" ); }, "more than one name" );
        expectRefused( [&] { (void)grant().openFileToWrite( "hard.txt", false ); }, "more than one name" );
        EXPECT_EQ( contentOf( outside_ / "secret.txt" ), "secret" );
    }

    TEST_F( ToolGrantFilesTest, AFileWithOneNameIsAllowed )
    {
        EXPECT_EQ( grant().openFile( "inside.txt" ).read(), "inside" );
        EXPECT_NO_THROW( (void)grant().openFileToWrite( "inside.txt", false ) );
    }

    // ================================================================
    // Writing
    // ================================================================

    TEST_F( ToolGrantFilesTest, AWriteInAFolderGrantedToReadIsRefused )
    {
        expectRefused( [&] { (void)grant( FolderAccess::Read ).openFileToWrite( "inside.txt", false ); }, "only to read" );
        expectRefused( [&] { (void)grant( FolderAccess::Read ).openFileToWrite( "new.txt", true ); }, "only to read" );
        EXPECT_FALSE( std::filesystem::exists( project_ / "new.txt" ) );
    }

    TEST_F( ToolGrantFilesTest, AWriteInAFolderGrantedToReadAndWriteIsAllowed )
    {
        grant().openFileToWrite( "inside.txt", false ).replace( "changed" );

        EXPECT_EQ( contentOf( project_ / "inside.txt" ), "changed" );
    }

    TEST_F( ToolGrantFilesTest, AReplaceThroughAJunctionLeadingOutsideLeavesTheFileWhole )
    {
        ASSERT_TRUE( makeJunction( project_ / "out", outside_ ) );

        expectRefused( [&] { (void)grant().openFileToWrite( "out/secret.txt", false ); }, kOutside );
        expectRefused( [&] { (void)grant().openFileToWrite( "out/secret.txt", true ); }, kOutside );
        EXPECT_EQ( contentOf( outside_ / "secret.txt" ), "secret" ) << "nothing truncated before the check";
    }

    TEST_F( ToolGrantFilesTest, AReplaceInsideWritesTheFile )
    {
        grant().openFileToWrite( "sub/a.txt", true ).replace( "replaced" );

        EXPECT_EQ( contentOf( project_ / "sub" / "a.txt" ), "replaced" );
    }

    TEST_F( ToolGrantFilesTest, ACreateUnderAJunctionLeadingOutsideIsRefused )
    {
        ASSERT_TRUE( makeJunction( project_ / "out", outside_ ) );
        ASSERT_TRUE( makeJunction( project_ / "dangling", outside_ / "not-there" ) );

        expectRefused( [&] { (void)grant().openFileToWrite( "out/new.txt", true ); }, kOutside );
        expectRefused( [&] { (void)grant().openFileToWrite( "dangling/new.txt", true ); }, "already exists" );
        EXPECT_FALSE( std::filesystem::exists( outside_ / "new.txt" ) );
        EXPECT_FALSE( std::filesystem::exists( outside_ / "not-there" ) ) << "a junction to nothing is never created through";
    }

    TEST_F( ToolGrantFilesTest, ACreateInANewFolderInsideIsAllowed )
    {
        grant().openFileToWrite( "new/deeper/note.txt", true ).replace( "made" );

        EXPECT_EQ( contentOf( project_ / "new" / "deeper" / "note.txt" ), "made" );
    }

    TEST_F( ToolGrantFilesTest, ACreateThroughALinkToAFileNotThereIsRefused )
    {
        std::error_code error;
        std::filesystem::create_symlink( outside_ / "made.txt", project_ / "dangling.txt", error );

        if ( error )
        {
            GTEST_SKIP() << "a symbolic link needs Developer Mode or an administrator: " << error.message();
        }

        expectRefused( [&] { (void)grant().openFileToWrite( "dangling.txt", true ); }, "already exists" );
        EXPECT_FALSE( std::filesystem::exists( outside_ / "made.txt" ) );
    }

    // ================================================================
    // Walking
    // ================================================================

    TEST_F( ToolGrantFilesTest, AWalkDoesNotEnterAJunctionLeadingOutside )
    {
        ASSERT_TRUE( makeJunction( project_ / "out", outside_ ) );

        std::set<std::string> files;
        const std::size_t not_entered = grant().walk( ".", [&files]( std::string_view relative, const FolderEntry& entry )
            {
                if ( !entry.is_folder )
                {
                    files.emplace( relative );
                }

                return true;
            } );

        EXPECT_EQ( not_entered, 1u );
        EXPECT_FALSE( files.contains( "out/secret.txt" ) );
    }

    TEST_F( ToolGrantFilesTest, AWalkEntersAFolderInside )
    {
        ASSERT_TRUE( makeJunction( project_ / "in", project_ / "sub" ) );

        std::set<std::string> files;
        const std::size_t not_entered = grant().walk( ".", [&files]( std::string_view relative, const FolderEntry& entry )
            {
                if ( !entry.is_folder )
                {
                    files.emplace( relative );
                }

                return true;
            } );

        EXPECT_EQ( not_entered, 0u );
        EXPECT_EQ( files, ( std::set<std::string>{ "in/a.txt", "inside.txt", "sub/a.txt" } ) );
    }

    // ================================================================
    // Narrowing
    // ================================================================

    TEST_F( ToolGrantFilesTest, AToolThatOnlyReadsIsRefusedAWrite )
    {
        ToolNeeds reads;
        reads.reads_files = true;

        expectRefused( [&] { (void)grant().narrowedTo( reads ).openFileToWrite( "inside.txt", false ); }, "only to read" );
        // A tool that declares no files is given no folder.
        expectRefused( [&] { (void)grant().narrowedTo( ToolNeeds{} ).openFile( "inside.txt" ); }, "granted no folder" );
    }

    TEST_F( ToolGrantFilesTest, AToolThatWritesIsAllowedAWrite )
    {
        ToolNeeds writes;
        writes.writes_files = true;

        EXPECT_NO_THROW( grant().narrowedTo( writes ).openFileToWrite( "inside.txt", false ).replace( "written" ) );
        EXPECT_EQ( contentOf( project_ / "inside.txt" ), "written" );
    }

    TEST_F( ToolGrantFilesTest, TheGateGivesEachCallItsToolsNarrowedGrant )
    {
        std::optional<FolderAccess> seen;

        Tool reader{ "read_notes", "Read a file.", R"({"type":"object"})",
            [&seen]( std::string_view, const ToolGrant& given, std::stop_token )
            {
                seen = given.folders().empty() ? std::nullopt : std::optional( given.folders().front().access );

                return std::string( "read" );
            } };

        reader.needs.reads_files = true;

        ToolGate gate;
        gate.add( std::move( reader ) );
        gate.setGrant( grant() );
        gate.setRules( ToolRules{}.add( { "*", ToolPermission::Allow, {} } ) );

        const ToolOutcome outcome = gate.answer( "read_notes", "{}", {}, {} );

        EXPECT_TRUE( outcome.ran );
        EXPECT_EQ( seen, FolderAccess::Read ) << "a read-write folder is given read-only to a tool that only reads";
    }
}
