/**
 * @file ToolGrant.Programs.cpp
 * @brief What a program run through a grant reaches: each access the grant's container refuses beside the same
 *        access inside it, accepted (ContainedPrograms.md 8, phase 2). Windows only, where a folder can be granted.
 *
 * The grant's own part -- what it hands its container, and to which tools -- is tested against a container that
 * records the run. The container's part runs MXC itself, loaded from the file MILA_MXC_LIBRARY names; those tests
 * skip, saying why, when this build has no container or the file is not given or not usable here.
 */

#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

import Mila;

namespace Mila::Tests::AI
{
    using namespace Mila::AI;

    namespace
    {
        void write( const std::filesystem::path& file, std::string_view content )
        {
            std::filesystem::create_directories( file.parent_path() );
            std::ofstream stream( file, std::ios::binary | std::ios::trunc );
            stream.write( content.data(), static_cast<std::streamsize>( content.size() ) );
        }

        std::string text( const std::filesystem::path& path )
        {
            const std::u8string utf8 = path.u8string();

            return std::string( utf8.begin(), utf8.end() );
        }

        bool contains( std::string_view haystack, std::string_view needle )
        {
            return haystack.find( needle ) != std::string_view::npos;
        }

        /// A container that runs nothing and keeps what it was asked.
        class RecordingContainer final : public ProgramContainer
        {
        public:

            ProgramResult run( const ContainedRun& run, std::stop_token ) const override
            {
                last = run;
                ++runs;

                return ProgramResult{ 7 };
            }

            mutable std::optional<ContainedRun> last;
            mutable int runs{ 0 };
        };

        class ToolGrantProgramsTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                std::random_device seed;
                root_ = std::filesystem::canonical( std::filesystem::temp_directory_path() ) / std::format( "MilaPrograms-{:08x}", seed() );
                project_ = root_ / "project";
                reference_ = root_ / "reference";
                tools_ = root_ / "tools";
                outside_ = root_ / "outside";

                write( project_ / "inside.txt", "inside" );
                write( reference_ / "read-only.txt", "reference" );
                write( tools_ / "tool.txt", "a tool" );
                write( outside_ / "secret.txt", "secret" );
            }

            void TearDown() override
            {
                std::error_code error;
                std::filesystem::remove_all( root_, error );
            }

            std::filesystem::path root_;
            std::filesystem::path project_;
            std::filesystem::path reference_;
            std::filesystem::path tools_;
            std::filesystem::path outside_;
        };

        ToolNeeds commandNeeds()
        {
            ToolNeeds needs;
            needs.reads_files = true;
            needs.writes_files = true;
            needs.runs_programs = true;

            return needs;
        }
    }

    // ================================================================
    // What the grant hands its container
    // ================================================================

    TEST_F( ToolGrantProgramsTest, AGrantWithNoContainerRefusesBesideOneWithAContainerThatRuns )
    {
        const ToolGrant without( { GrantFolder{ project_, FolderAccess::ReadWrite } }, false );
        const auto container = std::make_shared<RecordingContainer>();
        const ToolGrant with( { GrantFolder{ project_, FolderAccess::ReadWrite } }, false, {}, container );

        EXPECT_FALSE( without.runsPrograms() );

        try
        {
            (void)without.runProgram( { "cmd.exe /c echo hi" }, {} );
            ADD_FAILURE() << "ran with no container";
        } catch ( const std::runtime_error& error ) {
            EXPECT_TRUE( contains( error.what(), "no way to run programs" ) ) << error.what();
        }

        EXPECT_EQ( with.runProgram( { "cmd.exe /c echo hi" }, {} ).exit_code, 7 );
        EXPECT_EQ( container->runs, 1 );
    }

    TEST_F( ToolGrantProgramsTest, TheContainerIsGivenOnlyToAToolThatRunsPrograms )
    {
        const auto container = std::make_shared<RecordingContainer>();
        const ToolGrant grant( { GrantFolder{ project_, FolderAccess::ReadWrite } }, false, { tools_ }, container );

        ToolNeeds files;
        files.reads_files = true;
        files.writes_files = true;

        const ToolGrant file_tool = grant.narrowedTo( files );
        const ToolGrant command_tool = grant.narrowedTo( commandNeeds() );

        EXPECT_FALSE( file_tool.runsPrograms() );
        EXPECT_TRUE( file_tool.programFolders().empty() );
        EXPECT_TRUE( command_tool.runsPrograms() );
        EXPECT_EQ( command_tool.programFolders().size(), 1u );

        EXPECT_THROW( (void)file_tool.runProgram( { "cmd.exe /c echo hi" }, {} ), std::runtime_error );
        EXPECT_NO_THROW( (void)command_tool.runProgram( { "cmd.exe /c echo hi" }, {} ) );
    }

    // Decision 9: the program folders reach only the program.
    TEST_F( ToolGrantProgramsTest, NoFileToolReadsAProgramFolderBesideAFileInTheProject )
    {
        const ToolGrant grant( { GrantFolder{ project_, FolderAccess::ReadWrite } }, false, { tools_ },
            std::make_shared<RecordingContainer>() );

        try
        {
            (void)grant.openFile( text( tools_ / "tool.txt" ) );
            ADD_FAILURE() << "a file tool read a program folder";
        } catch ( const std::runtime_error& error ) {
            EXPECT_TRUE( contains( error.what(), "lies outside" ) ) << error.what();
        }

        EXPECT_NO_THROW( (void)grant.openFile( "inside.txt" ) );
    }

    TEST_F( ToolGrantProgramsTest, TheRunIsGivenFinalPathsAndTheGrantsOwnAccessAndNetwork )
    {
        const auto container = std::make_shared<RecordingContainer>();
        const ToolGrant grant( { GrantFolder{ project_, FolderAccess::ReadWrite }, GrantFolder{ reference_, FolderAccess::Read } },
            false, { tools_ }, container );
        std::filesystem::create_directories( project_ / "sub" );

        ProgramRequest request{ "cmd.exe /c echo hi", "sub", std::chrono::seconds( 5 ), 100 };
        (void)grant.narrowedTo( commandNeeds() ).runProgram( request, {} );

        ASSERT_TRUE( container->last.has_value() );
        const ContainedRun& run = *container->last;

        EXPECT_EQ( run.command_line, "cmd.exe /c echo hi" );
        EXPECT_TRUE( text( run.start_folder ).ends_with( "project\\sub" ) ) << text( run.start_folder );
        ASSERT_EQ( run.folders.size(), 2u );
        EXPECT_EQ( run.folders[ 0 ].access, FolderAccess::ReadWrite );
        EXPECT_EQ( run.folders[ 1 ].access, FolderAccess::Read );
        EXPECT_TRUE( text( run.folders[ 0 ].folder ).starts_with( "\\\\?\\" ) ) << "a final path";
        EXPECT_EQ( run.program_folders.size(), 1u );
        EXPECT_FALSE( run.network );
        EXPECT_EQ( run.time, std::chrono::seconds( 5 ) );
        EXPECT_EQ( run.output_bytes, 100u );
    }

    TEST_F( ToolGrantProgramsTest, AStartFolderOutsideIsRefusedBesideOneInside )
    {
        const ToolGrant grant( { GrantFolder{ project_, FolderAccess::ReadWrite } }, false, {}, std::make_shared<RecordingContainer>() );

        try
        {
            (void)grant.runProgram( { "cmd.exe /c echo hi", text( outside_ ) }, {} );
            ADD_FAILURE() << "started outside";
        } catch ( const std::runtime_error& error ) {
            EXPECT_TRUE( contains( error.what(), "lies outside" ) ) << error.what();
        }

        EXPECT_NO_THROW( (void)grant.runProgram( { "cmd.exe /c echo hi", text( project_ ) }, {} ) );
    }

    TEST_F( ToolGrantProgramsTest, AToolThatReadsButDoesNotWriteRunsWithItsFoldersReadOnly )
    {
        const auto container = std::make_shared<RecordingContainer>();
        const ToolGrant grant( { GrantFolder{ project_, FolderAccess::ReadWrite } }, false, {}, container );

        ToolNeeds reads;
        reads.reads_files = true;
        reads.runs_programs = true;

        (void)grant.narrowedTo( reads ).runProgram( { "cmd.exe /c echo hi" }, {} );

        ASSERT_TRUE( container->last.has_value() );
        EXPECT_EQ( container->last->folders.at( 0 ).access, FolderAccess::Read );
    }

    // ================================================================
    // What the container holds the run to: MXC itself
    // ================================================================

    namespace
    {
        std::shared_ptr<const MxcProgramContainer> openMxc( std::string& why_not )
        {
            static const auto opened = []() -> std::expected<std::shared_ptr<const MxcProgramContainer>, std::string>
                {
                    const char* path = std::getenv( "MILA_MXC_LIBRARY" );

                    if ( path == nullptr || *path == '\0' )
                    {
                        // The absent container says so whatever it is given.
                        auto absent = MxcProgramContainer::open( {} );

                        if ( !absent )
                        {
                            return std::unexpected( absent.error() + "; MILA_MXC_LIBRARY names no file either" );
                        }

                        return std::unexpected( std::string( "MILA_MXC_LIBRARY names no mxc_ffi.dll" ) );
                    }

                    return MxcProgramContainer::open( path );
                }();

            if ( !opened )
            {
                why_not = opened.error();

                return nullptr;
            }

            return *opened;
        }

        class ContainedRunTest : public ToolGrantProgramsTest
        {
        protected:

            void SetUp() override
            {
                std::string why_not;
                container_ = openMxc( why_not );

                if ( !container_ )
                {
                    GTEST_SKIP() << why_not;
                }

                ToolGrantProgramsTest::SetUp();
            }

            ToolGrant grant( bool network = false, std::vector<std::filesystem::path> program_folders = {} ) const
            {
                return ToolGrant( { GrantFolder{ project_, FolderAccess::ReadWrite }, GrantFolder{ reference_, FolderAccess::Read } },
                    network, std::move( program_folders ), container_ ).narrowedTo( withNetwork( network ) );
            }

            static ToolNeeds withNetwork( bool network )
            {
                ToolNeeds needs = commandNeeds();
                needs.uses_network = network;

                return needs;
            }

            ProgramResult run( const ToolGrant& grant, std::string command ) const
            {
                return grant.runProgram( { std::move( command ), "", std::chrono::seconds( 60 ), 64 * 1024 }, {} );
            }

            std::shared_ptr<const MxcProgramContainer> container_;
        };

        /// Answers one HTTP request on a loopback port with 200, on a thread of its own.
        class LoopbackServer
        {
        public:

            LoopbackServer()
            {
                WSADATA data{};
                WSAStartup( MAKEWORD( 2, 2 ), &data );

                listener_ = socket( AF_INET, SOCK_STREAM, IPPROTO_TCP );
                sockaddr_in address{};
                address.sin_family = AF_INET;
                address.sin_addr.s_addr = htonl( INADDR_LOOPBACK );
                bind( listener_, reinterpret_cast<sockaddr*>( &address ), sizeof( address ) );
                listen( listener_, 4 );

                int length = sizeof( address );
                getsockname( listener_, reinterpret_cast<sockaddr*>( &address ), &length );
                port_ = ntohs( address.sin_port );

                server_ = std::thread( [this]
                    {
                        for ( ;; )
                        {
                            const SOCKET client = accept( listener_, nullptr, nullptr );

                            if ( client == INVALID_SOCKET )
                            {
                                return;
                            }

                            char buffer[ 1024 ];
                            recv( client, buffer, sizeof( buffer ), 0 );
                            const std::string_view reply = "HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\nok";
                            send( client, reply.data(), static_cast<int>( reply.size() ), 0 );
                            closesocket( client );
                        }
                    } );
            }

            ~LoopbackServer()
            {
                closesocket( listener_ );
                server_.join();
                WSACleanup();
            }

            [[nodiscard]] unsigned short port() const noexcept
            {
                return port_;
            }

        private:

            SOCKET listener_{ INVALID_SOCKET };
            unsigned short port_{ 0 };
            std::thread server_;
        };

        /// What curl prints for a status, on the host, outside any container.
        std::string hostStatus( const std::string& url, const std::filesystem::path& scratch )
        {
            const std::filesystem::path out = scratch / "host-status.txt";
            const std::string command = std::format( "curl.exe -s -m 10 -o NUL -w \"%{{http_code}}\" {} > \"{}\" 2>nul", url, text( out ) );
            std::system( command.c_str() );
            std::ifstream stream( out );

            return std::string( std::istreambuf_iterator<char>( stream ), std::istreambuf_iterator<char>() );
        }

        /// The next hop of this machine's default route when it is a private IPv4 address, such as a home router.
        /// Asking for the route sends nothing.
        std::optional<std::string> privateGateway()
        {
            MIB_IPFORWARDROW row{};
            in_addr anywhere{};
            inet_pton( AF_INET, "192.0.2.1", &anywhere );

            if ( GetBestRoute( anywhere.s_addr, 0, &row ) != NO_ERROR )
            {
                return std::nullopt;
            }

            const std::uint32_t hop = ntohl( row.dwForwardNextHop );
            const bool is_private = ( hop >> 24 ) == 10 || ( hop >> 20 ) == 0xAC1 || ( hop >> 16 ) == 0xC0A8;

            if ( !is_private )
            {
                return std::nullopt;
            }

            in_addr address{};
            address.s_addr = row.dwForwardNextHop;
            char buffer[ INET_ADDRSTRLEN ]{};
            inet_ntop( AF_INET, &address, buffer, sizeof( buffer ) );

            return std::string( buffer );
        }
    }

    TEST_F( ContainedRunTest, AFileOutsideIsRefusedBesideAFileInsideAccepted )
    {
        const ToolGrant granted = grant();

        const ProgramResult inside = run( granted, "cmd.exe /c type inside.txt" );
        EXPECT_EQ( inside.exit_code, 0 ) << inside.errors;
        EXPECT_TRUE( contains( inside.output, "inside" ) ) << inside.output;

        const ProgramResult outside = run( granted, std::format( "cmd.exe /c type \"{}\"", text( outside_ / "secret.txt" ) ) );
        EXPECT_NE( outside.exit_code, 0 );
        EXPECT_FALSE( contains( outside.output, "secret" ) ) << outside.output;
    }

    TEST_F( ContainedRunTest, AWriteToAReadOnlyFolderIsRefusedBesideOneToTheProjectAccepted )
    {
        const ToolGrant granted = grant();

        const ProgramResult read_only = run( granted, std::format( "cmd.exe /c type \"{}\"", text( reference_ / "read-only.txt" ) ) );
        EXPECT_TRUE( contains( read_only.output, "reference" ) ) << "the read-only folder is read";

        (void)run( granted, std::format( "cmd.exe /c echo written> \"{}\"", text( reference_ / "new.txt" ) ) );
        (void)run( granted, "cmd.exe /c echo written> new.txt" );

        EXPECT_FALSE( std::filesystem::exists( reference_ / "new.txt" ) ) << "written into the read-only folder";
        EXPECT_TRUE( std::filesystem::exists( project_ / "new.txt" ) ) << "not written into the project";
    }

    TEST_F( ContainedRunTest, ANameInsideForAFileOutsideIsWithheldBesideAFileWhoseNamesAreAllInside )
    {
        std::filesystem::create_hard_link( outside_ / "secret.txt", project_ / "link.txt" );
        write( project_ / "both.txt", "both inside" );
        std::filesystem::create_directories( project_ / "sub" );
        std::filesystem::create_hard_link( project_ / "both.txt", project_ / "sub" / "both-again.txt" );

        const ToolGrant granted = grant();
        const ProgramResult linked = run( granted, "cmd.exe /c type link.txt" );

        EXPECT_FALSE( contains( linked.output, "secret" ) ) << linked.output;
        ASSERT_EQ( linked.withheld.size(), 1u );
        EXPECT_TRUE( linked.withheld.front().ends_with( "project\\link.txt" ) ) << linked.withheld.front();

        const ProgramResult both = run( granted, "cmd.exe /c type sub\\both-again.txt" );
        EXPECT_TRUE( contains( both.output, "both inside" ) ) << both.output;
    }

    TEST_F( ContainedRunTest, ANameMadeBetweenRunsForAFileOutsideIsWithheldBesideTheFileBeforeIt )
    {
        const ToolGrant granted = grant();

        const ProgramResult before = run( granted, "cmd.exe /c type inside.txt" );
        EXPECT_TRUE( before.withheld.empty() );
        EXPECT_TRUE( contains( before.output, "inside" ) );

        std::filesystem::create_hard_link( outside_ / "secret.txt", project_ / "late.txt" );

        ProgramResult after;

        // The watcher reports the new name a moment after it is made.
        for ( int attempt = 0; attempt < 20 && after.withheld.empty(); ++attempt )
        {
            std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );
            after = run( granted, "cmd.exe /c type late.txt" );
        }

        EXPECT_FALSE( contains( after.output, "secret" ) ) << after.output;
        ASSERT_EQ( after.withheld.size(), 1u );
        EXPECT_TRUE( after.withheld.front().ends_with( "late.txt" ) );
    }

    TEST_F( ContainedRunTest, AFileAlsoNamedInAReadOnlyFolderIsWithheldBesideOneNamedOnlyInTheProject )
    {
        std::filesystem::create_hard_link( reference_ / "read-only.txt", project_ / "writable-name.txt" );

        const ProgramResult result = run( grant(), "cmd.exe /c type inside.txt" );

        ASSERT_EQ( result.withheld.size(), 1u ) << "a write through the project's name would change the read-only folder";
        EXPECT_TRUE( result.withheld.front().ends_with( "writable-name.txt" ) );
        EXPECT_TRUE( contains( result.output, "inside" ) );
    }

    TEST_F( ContainedRunTest, AProgramFolderIsReadByTheRunOnlyWhenGranted )
    {
        const std::string command = std::format( "cmd.exe /c type \"{}\"", text( tools_ / "tool.txt" ) );

        const ProgramResult without = run( grant(), command );
        EXPECT_FALSE( contains( without.output, "a tool" ) ) << without.output;

        const ProgramResult with = run( grant( false, { tools_ } ), command );
        EXPECT_TRUE( contains( with.output, "a tool" ) ) << with.errors;
    }

    TEST_F( ContainedRunTest, ALoopbackServerIsRefusedWithTheNetworkGrantedBesideTheHostReachingIt )
    {
        const LoopbackServer server;
        const std::string url = std::format( "http://127.0.0.1:{}/", server.port() );

        ASSERT_EQ( hostStatus( url, root_ ), "200" ) << "the host reaches its own server";

        const std::string command = std::format( "curl.exe -s -m 5 -o NUL -w \"%{{http_code}}\" {}", url );
        const ProgramResult contained = run( grant( true ), command );

        EXPECT_NE( contained.output, "200" ) << "a private address was reached with the network granted";
    }

    // Loopback is refused by the container's own rule for the host; a router on the local network is refused only by
    // the ranges the grant sends, so this is the test that they are sent.
    TEST_F( ContainedRunTest, TheLocalNetworksGatewayIsRefusedWithTheNetworkGrantedBesideTheHostReachingIt )
    {
        const std::optional<std::string> gateway = privateGateway();

        if ( !gateway )
        {
            GTEST_SKIP() << "this machine's default route has no private gateway";
        }

        const std::string url = std::format( "http://{}/", *gateway );

        if ( hostStatus( url, root_ ) != "200" )
        {
            GTEST_SKIP() << "the gateway " << *gateway << " answers no web page to the host";
        }

        const std::string command = std::format( "curl.exe -s -m 5 -o NUL -w \"%{{http_code}}\" {}", url );
        const ProgramResult contained = run( grant( true ), command );

        EXPECT_NE( contained.output, "200" ) << "the local network was reached with the network granted";
    }

    TEST_F( ContainedRunTest, ThePublicNetworkIsRefusedWithNoneGrantedBesideItAcceptedWhenGranted )
    {
        const std::string url = "https://example.com/";

        if ( hostStatus( url, root_ ) != "200" )
        {
            GTEST_SKIP() << "this machine cannot reach " << url;
        }

        const std::string command = std::format( "curl.exe -s -m 10 -o NUL -w \"%{{http_code}}\" {}", url );

        EXPECT_NE( run( grant( false ), command ).output, "200" ) << "the network was reached with none granted";
        EXPECT_EQ( run( grant( true ), command ).output, "200" ) << "the public network granted was not reached";
    }

    // A copy of the screen made on Mila's own desktop copies that desktop, which holds nothing, so the test is what it
    // shows: how many colours a grid of points across the whole screen holds.
    TEST_F( ContainedRunTest, ThePersonsScreenIsNotSeenBesideTheHostSeeingIt )
    {
        const std::string script = "Add-Type -AssemblyName System.Drawing,System.Windows.Forms; "
            "$r = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; $b = New-Object System.Drawing.Bitmap $r.Width,$r.Height; "
            "try { [System.Drawing.Graphics]::FromImage($b).CopyFromScreen(0,0,0,0,$b.Size) } catch { 'colours 0'; exit }; "
            "$c = @{}; for ($x = 0; $x -lt 16; $x++) { for ($y = 0; $y -lt 16; $y++) { "
            "$c[$b.GetPixel([int]($x * ($r.Width - 1) / 15), [int]($y * ($r.Height - 1) / 15)).ToArgb()] = 1 } }; 'colours ' + $c.Count";
        const auto colours = []( const std::string& output )
            {
                const auto at = output.find( "colours " );

                return at == std::string::npos ? -1 : std::stoi( output.substr( at + 8 ) );
            };

        const std::filesystem::path out = root_ / "host-screen.txt";
        std::system( std::format( "powershell.exe -NoProfile -Command \"{}\" > \"{}\" 2>nul", script, text( out ) ).c_str() );
        std::ifstream stream( out );
        const int host = colours( std::string( std::istreambuf_iterator<char>( stream ), std::istreambuf_iterator<char>() ) );

        if ( host < 2 )
        {
            GTEST_SKIP() << "this session's screen shows too little to tell it apart from an empty one";
        }

        const ProgramResult contained = run( grant(), std::format( "powershell.exe -NoProfile -Command \"{}\"", script ) );
        const int seen = colours( contained.output );

        EXPECT_GE( seen, 0 ) << "the program did not run: " << contained.errors;
        EXPECT_LE( seen, 1 ) << "the program saw " << seen << " colours where the host's screen shows " << host;
    }

    TEST_F( ContainedRunTest, OutputPastTheCapIsDroppedAndCounted )
    {
        const ProgramResult result = grant().runProgram(
            { "cmd.exe /c for /L %i in (1,1,5000) do @echo 0123456789", "", std::chrono::seconds( 60 ), 1000 }, {} );

        EXPECT_EQ( result.output.size(), 1000u );
        EXPECT_GT( result.output_dropped, 50'000u );
    }

    TEST_F( ContainedRunTest, ARunIsEndedWhenStoppedAndWhenItsTimeRunsOut )
    {
        const std::string forever = "cmd.exe /c for /L %i in (1,0,2) do @rem";
        const ToolGrant granted = grant();

        std::stop_source stop;
        std::thread stopper( [&] { std::this_thread::sleep_for( std::chrono::milliseconds( 500 ) ); stop.request_stop(); } );
        const auto started = std::chrono::steady_clock::now();
        const ProgramResult stopped = granted.runProgram( { forever, "", std::chrono::seconds( 60 ), 1000 }, stop.get_token() );
        stopper.join();

        EXPECT_TRUE( stopped.stopped );
        EXPECT_LT( std::chrono::steady_clock::now() - started, std::chrono::seconds( 10 ) );

        const ProgramResult timed = granted.runProgram( { forever, "", std::chrono::seconds( 1 ), 1000 }, {} );

        EXPECT_TRUE( timed.timed_out );
        EXPECT_FALSE( timed.stopped );
    }
}
