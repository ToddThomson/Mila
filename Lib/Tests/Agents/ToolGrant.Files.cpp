/**
 * @file ToolGrant.Files.cpp
 * @brief What a grant's files reach: each access the grant refuses beside the same kind of access inside it,
 *        allowed (ToolCalling.md 3.4). Windows, macOS and Linux, each by its own ways to the same access.
 *
 * Each test builds its folders under the system's temporary folder: a granted project, a sibling whose name begins
 * with the project's, and a folder outside holding a secret. A link to a folder is a junction on Windows, which needs
 * no privilege, and a symbolic link elsewhere. A symbolic link on Windows needs Developer Mode or an administrator,
 * so those tests skip there where one cannot be made. A mount point needs an administrator and is not tested: it
 * leads to a final path as a junction does. macOS's temporary folder is under /var, a link to /private/var, so every
 * full path a test gives on a Mac goes through a link. On Linux TMPDIR chooses the volume, so the same tests run on
 * one that heeds case and one that ignores it.
 */

#include <gtest/gtest.h>
#include <algorithm>
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

#ifndef _WIN32
#  include <cerrno>
#  include <fcntl.h>
#  include <sys/socket.h>
#  include <sys/stat.h>
#  include <sys/un.h>
#  include <unistd.h>
#endif

import Mila;

namespace Mila::Tests::AI
{
    using namespace Mila::Agents;

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

        /// A link to a folder any account can make: a junction on Windows, a symbolic link elsewhere. True when made.
        bool makeFolderLink( const std::filesystem::path& link, const std::filesystem::path& target )
        {
#ifdef _WIN32
            const std::string command = std::format( "mklink /J \"{}\" \"{}\" >nul 2>&1", link.string(), target.string() );

            return std::system( command.c_str() ) == 0 && std::filesystem::exists( std::filesystem::symlink_status( link ) );
#else
            std::error_code error;
            std::filesystem::create_directory_symlink( target, link, error );

            return !error;
#endif
        }

        std::filesystem::path uniqueName( std::string_view prefix )
        {
            std::random_device seed;

            return std::format( "{}-{:08x}", prefix, seed() );
        }

        class ToolGrantFilesTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                root_ = std::filesystem::temp_directory_path() / uniqueName( "MilaToolGrant" );

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

                // A link is removed as a link, never by entering it.
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
        if ( !std::filesystem::exists( root_ / "PROJECT" ) )
        {
            GTEST_SKIP() << "the temporary folder is on a volume whose names differ by case";
        }

        EXPECT_EQ( grant().openFile( text( root_ / "PROJECT" / "INSIDE.TXT" ) ).read(), "inside" );
        EXPECT_EQ( grant().openFile( "SUB/A.TXT" ).read(), "a" );
        grant().openFileToWrite( text( root_ / "PROJECT" / "SUB" / "NEW.TXT" ), true ).replace( "made" );
        EXPECT_EQ( contentOf( project_ / "sub" / "NEW.TXT" ), "made" );

        ToolGrant granted = grant();
        std::size_t files = 0;
        const std::size_t not_entered = granted.walk( "SUB", [&files]( std::string_view, const FolderEntry& entry )
            {
                files += entry.is_folder ? 0 : 1;

                return true;
            } );

        EXPECT_EQ( not_entered, 0u );
        EXPECT_EQ( files, 2u );
    }

    TEST_F( ToolGrantFilesTest, AFolderWhoseNameDiffersOnlyByCaseIsAnotherFolderWhereCaseIsHeeded )
    {
        if ( std::filesystem::exists( root_ / "PROJECT" ) )
        {
            GTEST_SKIP() << "the temporary folder is on a volume that ignores case";
        }

        write( root_ / "PROJECT" / "secret.txt", "another folder" );

        expectRefused( [&] { (void)grant().openFile( text( root_ / "PROJECT" / "secret.txt" ) ); }, kOutside );
        expectRefused( [&] { (void)grant().openFileToWrite( text( root_ / "PROJECT" / "new.txt" ), true ); }, kOutside );
        expectRefused( [&] { (void)grant().list( text( root_ / "PROJECT" ) ); }, kOutside );
        EXPECT_FALSE( std::filesystem::exists( root_ / "PROJECT" / "new.txt" ) );
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

    TEST_F( ToolGrantFilesTest, AFolderLinkLeadingOutsideIsRefused )
    {
        ASSERT_TRUE( makeFolderLink( project_ / "out", outside_ ) );

        expectRefused( [&] { (void)grant().openFile( "out/secret.txt" ); }, kOutside );
        expectRefused( [&] { (void)grant().list( "out" ); }, kOutside );
    }

    TEST_F( ToolGrantFilesTest, AFolderLinkInsideIsAllowed )
    {
        ASSERT_TRUE( makeFolderLink( project_ / "in", project_ / "sub" ) );

        EXPECT_EQ( grant().openFile( "in/a.txt" ).read(), "a" );
        EXPECT_EQ( grant().list( "in" ).size(), 1u );
    }

    TEST_F( ToolGrantFilesTest, AFolderReplacedByALinkOutsideOnceGrantedIsRefused )
    {
        const ToolGrant granted = grant();
        ASSERT_EQ( granted.openFile( "sub/a.txt" ).read(), "a" );

        std::filesystem::remove_all( project_ / "sub" );
        ASSERT_TRUE( makeFolderLink( project_ / "sub", outside_ ) );

        expectRefused( [&] { (void)granted.openFile( "sub/secret.txt" ); }, kOutside );
        expectRefused( [&] { (void)granted.openFileToWrite( "sub/new.txt", true ); }, kOutside );
        expectRefused( [&] { (void)granted.list( "sub" ); }, kOutside );
        EXPECT_FALSE( std::filesystem::exists( outside_ / "new.txt" ) );
    }

    TEST_F( ToolGrantFilesTest, AFolderReplacedByAFolderInsideOnceGrantedIsAllowed )
    {
        const ToolGrant granted = grant();
        ASSERT_EQ( granted.openFile( "sub/a.txt" ).read(), "a" );

        std::filesystem::remove_all( project_ / "sub" );
        write( project_ / "sub" / "b.txt", "b" );

        EXPECT_EQ( granted.openFile( "sub/b.txt" ).read(), "b" );
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
        expectRefused( [&] { (void)grant().openFileToWrite( "link.txt", false ); }, kOutside );
        EXPECT_EQ( contentOf( outside_ / "secret.txt" ), "secret" );
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

    TEST_F( ToolGrantFilesTest, AReplaceThroughAFolderLinkLeadingOutsideLeavesTheFileWhole )
    {
        ASSERT_TRUE( makeFolderLink( project_ / "out", outside_ ) );

        expectRefused( [&] { (void)grant().openFileToWrite( "out/secret.txt", false ); }, kOutside );
        expectRefused( [&] { (void)grant().openFileToWrite( "out/secret.txt", true ); }, kOutside );
        EXPECT_EQ( contentOf( outside_ / "secret.txt" ), "secret" ) << "nothing truncated before the check";
    }

    TEST_F( ToolGrantFilesTest, AReplaceInsideWritesTheFile )
    {
        grant().openFileToWrite( "sub/a.txt", true ).replace( "replaced" );

        EXPECT_EQ( contentOf( project_ / "sub" / "a.txt" ), "replaced" );
    }

    TEST_F( ToolGrantFilesTest, ACreateUnderAFolderLinkLeadingOutsideIsRefused )
    {
        ASSERT_TRUE( makeFolderLink( project_ / "out", outside_ ) );
        ASSERT_TRUE( makeFolderLink( project_ / "dangling", outside_ / "not-there" ) );

        expectRefused( [&] { (void)grant().openFileToWrite( "out/new.txt", true ); }, kOutside );
        expectRefused( [&] { (void)grant().openFileToWrite( "dangling/new.txt", true ); }, "already exists" );
        EXPECT_FALSE( std::filesystem::exists( outside_ / "new.txt" ) );
        EXPECT_FALSE( std::filesystem::exists( outside_ / "not-there" ) ) << "a link to nothing is never created through";
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

    TEST_F( ToolGrantFilesTest, AWalkDoesNotEnterAFolderLinkLeadingOutside )
    {
        ASSERT_TRUE( makeFolderLink( project_ / "out", outside_ ) );

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
        ASSERT_TRUE( makeFolderLink( project_ / "in", project_ / "sub" ) );

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

#ifndef _WIN32
    // ================================================================
    // Pipes, sockets and long paths: macOS's and Linux's ways in
    // ================================================================

    namespace
    {
        /// Makes a named pipe in the project; false where the volume makes none, as a Windows drive in WSL does not.
        bool makePipe( const std::filesystem::path& pipe )
        {
            return ::mkfifo( pipe.c_str(), 0644 ) == 0;
        }

#ifdef __APPLE__
        // F_GETPATH reports up to MAXPATHLEN (ToolCalling.md 3.4, Files on macOS).
        constexpr std::size_t kReportedPathLimit = 1024;
#else
        // /proc reports up to 4,096 bytes (ToolCalling.md 3.4, Files on Linux).
        constexpr std::size_t kReportedPathLimit = 4096;
#endif

        /// Folders of 200 characters each under the project, with a file name: just past the limit, or within it.
        std::string deepPath( const std::filesystem::path& project, bool past )
        {
            const std::string level( 200, 'd' );
            const std::size_t room = kReportedPathLimit - project.native().size() - std::string_view( "/deep.txt" ).size();
            const std::size_t depth = past ? room / ( level.size() + 1 ) + 2 : room / ( level.size() + 1 ) - 1;
            std::string path;

            for ( std::size_t index = 0; index < depth; ++index )
            {
                path += level + "/";
            }

            return path + "deep.txt";
        }
    }

    TEST_F( ToolGrantFilesTest, ANamedPipeIsRefusedWithoutWaiting )
    {
        if ( !makePipe( project_ / "pipe" ) )
        {
            GTEST_SKIP() << "this volume makes no named pipe: " << std::generic_category().message( errno );
        }

        // With no writer, an open that waited would never return.
        expectRefused( [&] { (void)grant().openFile( "pipe" ); }, "not a file or folder" );
        expectRefused( [&] { (void)grant().openFileToWrite( "pipe", false ); }, "not a file or folder" );
    }

    TEST_F( ToolGrantFilesTest, ARegularFileBesideAPipeIsAllowed )
    {
        if ( !makePipe( project_ / "pipe" ) )
        {
            GTEST_SKIP() << "this volume makes no named pipe: " << std::generic_category().message( errno );
        }

        EXPECT_EQ( grant().openFile( "inside.txt" ).read(), "inside" );
        EXPECT_EQ( grant().list( "." ).size(), 3u ) << "inside.txt, sub and the pipe are all listed";
    }

    TEST_F( ToolGrantFilesTest, ASocketFileIsRefused )
    {
        const std::string path = ( project_ / "socket" ).native();
        sockaddr_un address{};

        if ( path.size() >= sizeof( address.sun_path ) )
        {
            GTEST_SKIP() << "the temporary folder's path is too long for a socket's name";
        }

        const int listening = ::socket( AF_UNIX, SOCK_STREAM, 0 );
        ASSERT_NE( listening, -1 );
        address.sun_family = AF_UNIX;
        std::ranges::copy( path, address.sun_path );

        if ( ::bind( listening, reinterpret_cast<const sockaddr*>( &address ), sizeof( address ) ) != 0 )
        {
            const int error = errno;
            ::close( listening );
            GTEST_SKIP() << "this volume makes no socket file: " << std::generic_category().message( error );
        }

        expectRefused( [&] { (void)grant().openFile( "socket" ); }, "not a file or folder" );
        expectRefused( [&] { (void)grant().openFileToWrite( "socket", false ); }, "not a file or folder" );
        EXPECT_EQ( grant().openFile( "inside.txt" ).read(), "inside" ) << "a file beside it is read";
        ::close( listening );
    }

    TEST_F( ToolGrantFilesTest, APathPastWhatTheSystemCanReportIsRefused )
    {
        expectRefused( [&] { (void)grant().openFileToWrite( deepPath( project_, true ), true ); }, "longer than" );
    }

    TEST_F( ToolGrantFilesTest, ALongPathWithinWhatTheSystemCanReportIsAllowed )
    {
        const std::string path = deepPath( project_, false );

        grant().openFileToWrite( path, true ).replace( "deep" );

        EXPECT_EQ( grant().openFile( path ).read(), "deep" );
    }
#endif

#ifdef __linux__
    // ================================================================
    // Linux's own ways in
    // ================================================================

    TEST_F( ToolGrantFilesTest, ALinkThroughProcToAFileOutsideIsRefused )
    {
        // The test's own descriptor on the secret: /proc/self/fd/<n> leads to it from anywhere.
        const int held = ::open( ( outside_ / "secret.txt" ).c_str(), O_RDONLY | O_CLOEXEC );
        ASSERT_NE( held, -1 );

        const std::string magic = std::format( "/proc/self/fd/{}", held );
        std::filesystem::create_symlink( magic, project_ / "held.txt" );

        expectRefused( [&] { (void)grant().openFile( "held.txt" ); }, "link in /proc" );
        expectRefused( [&] { (void)grant().openFileToWrite( "held.txt", false ); }, "link in /proc" );
        expectRefused( [&] { (void)grant().openFile( magic ); }, "link in /proc" );
        EXPECT_EQ( contentOf( outside_ / "secret.txt" ), "secret" );
        ::close( held );
    }

    TEST_F( ToolGrantFilesTest, AnOrdinaryLinkToTheSameKindOfFileInsideIsAllowed )
    {
        std::filesystem::create_symlink( project_ / "inside.txt", project_ / "held.txt" );

        EXPECT_EQ( grant().openFile( "held.txt" ).read(), "inside" );
    }
#endif

#ifdef __APPLE__
    // ================================================================
    // macOS's own ways in
    // ================================================================

    TEST_F( ToolGrantFilesTest, AResourceForkIsRefused )
    {
        EXPECT_THROW( (void)grant().openFile( "inside.txt/..namedfork/rsrc" ), std::invalid_argument );
        EXPECT_THROW( (void)grant().openFileToWrite( text( project_ / "inside.txt" / "..namedfork" / "rsrc" ), true ),
            std::invalid_argument );
    }

    TEST_F( ToolGrantFilesTest, TheFilesOwnContentIsAllowed )
    {
        EXPECT_EQ( grant().openFile( text( project_ / "inside.txt" ) ).read(), "inside" );
    }

    /// /tmp is a link to /private/tmp, as /var is to /private/var: a folder granted by one name is reached by both.
    class ToolGrantFilesTmpTest : public ::testing::Test
    {
    protected:

        void SetUp() override
        {
            name_ = uniqueName( "MilaToolGrant" );
            write( std::filesystem::path( "/tmp" ) / name_ / "project" / "inside.txt", "inside" );
            write( std::filesystem::path( "/tmp" ) / name_ / "outside" / "secret.txt", "secret" );
        }

        void TearDown() override
        {
            std::error_code error;
            std::filesystem::remove_all( std::filesystem::path( "/private/tmp" ) / name_, error );
        }

        std::filesystem::path name_;
    };

    TEST_F( ToolGrantFilesTmpTest, AFileOutsideReachedThroughTmpIsRefused )
    {
        const ToolGrant granted( { GrantFolder{ std::filesystem::path( "/tmp" ) / name_ / "project", FolderAccess::Read } }, false );

        expectRefused( [&] { (void)granted.openFile( text( std::filesystem::path( "/tmp" ) / name_ / "outside" / "secret.txt" ) ); }, kOutside );
        expectRefused( [&] { (void)granted.openFile( text( std::filesystem::path( "/private/tmp" ) / name_ / "outside" / "secret.txt" ) ); }, kOutside );
    }

    TEST_F( ToolGrantFilesTmpTest, AFileInsideReachedByEitherNameIsAllowed )
    {
        const ToolGrant granted( { GrantFolder{ std::filesystem::path( "/tmp" ) / name_ / "project", FolderAccess::Read } }, false );

        EXPECT_EQ( granted.openFile( text( std::filesystem::path( "/tmp" ) / name_ / "project" / "inside.txt" ) ).read(), "inside" );
        EXPECT_EQ( granted.openFile( text( std::filesystem::path( "/private/tmp" ) / name_ / "project" / "inside.txt" ) ).read(), "inside" );
    }
#endif
}
