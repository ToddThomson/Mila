#define NOMINMAX
#include <windows.h>
#include <gtest/gtest.h>

import std;
import ToolBroker.LineChannel;
import ToolBroker.Worker;
import ToolBroker.WorkerLimits;
import ToolBroker.WorkerProgram;

namespace Mila::Tests::Apps::ToolBroker
{
    using Mila::ToolBroker::launchWorker;
    using Mila::ToolBroker::Worker;
    using Mila::ToolBroker::WorkerLimits;
    using Mila::ToolBroker::WorkerProgram;

    namespace
    {
        WorkerProgram echoWorker( std::vector<std::wstring> arguments = {} )
        {
            return { std::filesystem::path( TOOL_BROKER_ECHO_WORKER ), std::move( arguments ) };
        }

        std::string ask( Worker& worker, std::string_view request )
        {
            worker.channel().send( request );

            return worker.channel().receive().value_or( "<closed>" );
        }

        /** Whether the process has ended, waiting up to `timeout` for it to. */
        bool hasEnded( std::uint32_t process_id, std::chrono::milliseconds timeout )
        {
            HANDLE process = ::OpenProcess( SYNCHRONIZE, FALSE, process_id );

            if ( process == nullptr )
            {
                return ::GetLastError() == ERROR_INVALID_PARAMETER;
            }

            const bool ended = ::WaitForSingleObject( process, static_cast<DWORD>( timeout.count() ) ) == WAIT_OBJECT_0;
            ::CloseHandle( process );

            return ended;
        }

        /** A fresh folder in the build directory, beside the echo worker, removed when the test ends. */
        class TemporaryFolder
        {
        public:
            TemporaryFolder()
                : path_( std::filesystem::path( TOOL_BROKER_ECHO_WORKER ).parent_path()
                    / std::format( "tool-broker-test-{}-{}", ::GetCurrentProcessId(), counter_++ ) )
            {
                std::filesystem::create_directories( path_ );
            }

            ~TemporaryFolder()
            {
                std::error_code ignored;
                std::filesystem::remove_all( path_, ignored );
            }

            const std::filesystem::path& path() const noexcept
            {
                return path_;
            }

        private:
            static inline int counter_ = 0;
            std::filesystem::path path_;
        };
    }

    TEST( WorkerTests, AnswersOverItsChannel )
    {
        auto worker = launchWorker( echoWorker(), {} );

        EXPECT_EQ( ask( worker, "hello" ), "hello" );
        EXPECT_EQ( ask( worker, R"({"jsonrpc":"2.0","id":7})" ), R"({"jsonrpc":"2.0","id":7})" );
    }

    TEST( WorkerTests, SeesNoEnvironmentWhenGivenNone )
    {
        auto worker = launchWorker( echoWorker(), {} );

        EXPECT_EQ( ask( worker, "environment" ), "" );
    }

    TEST( WorkerTests, SeesExactlyTheEnvironmentItIsGiven )
    {
        WorkerLimits limits;
        limits.environment = { { L"MILA_SECOND", L"2" }, { L"MILA_FIRST", L"1" } };

        auto worker = launchWorker( echoWorker(), limits );

        EXPECT_EQ( ask( worker, "environment" ), "MILA_FIRST,MILA_SECOND" );
    }

    TEST( WorkerTests, RefusesAnEnvironmentNameHoldingAnEqualsSign )
    {
        WorkerLimits limits;
        limits.environment = { { L"NAME=VALUE", L"1" } };

        EXPECT_THROW( launchWorker( echoWorker(), limits ), std::invalid_argument );
    }

    TEST( WorkerTests, StartsInItsProjectArea )
    {
        TemporaryFolder folder;
        WorkerLimits limits;
        limits.project_area = folder.path();

        auto worker = launchWorker( echoWorker(), limits );

        EXPECT_TRUE( std::filesystem::equivalent( ask( worker, "directory" ), folder.path() ) );
    }

    TEST( WorkerTests, ReceivesItsArgumentsUnchanged )
    {
        auto worker = launchWorker( echoWorker( { L"plain", L"with space", L"say \"hi\"", L"trailing\\", L"" } ), {} );

        EXPECT_EQ( ask( worker, "arguments" ), "plain|with space|say \"hi\"|trailing\\|" );
    }

    TEST( WorkerTests, ExitsOnItsOwnWhenItsInputCloses )
    {
        std::uint32_t process_id = 0;
        auto worker = std::make_optional( launchWorker( echoWorker(), {} ) );
        process_id = worker->processId();
        ASSERT_EQ( ask( *worker, "ready" ), "ready" );

        // Well inside the grace period, so the worker left by itself rather than when its job closed.
        const auto start = std::chrono::steady_clock::now();
        worker.reset();
        const auto elapsed = std::chrono::steady_clock::now() - start;

        EXPECT_LT( elapsed, Worker::kExitGrace / 2 );
        EXPECT_TRUE( hasEnded( process_id, std::chrono::milliseconds( 0 ) ) );
    }

    TEST( WorkerTests, IsEndedWhenItIgnoresTheEndOfItsInput )
    {
        std::uint32_t process_id = 0;

        {
            auto worker = launchWorker( echoWorker(), {} );
            process_id = worker.processId();
            ASSERT_EQ( ask( worker, "keep-running" ), "ok" );
        }

        EXPECT_TRUE( hasEnded( process_id, std::chrono::milliseconds( 5000 ) ) );
    }

    TEST( WorkerTests, EndsWhatItStartedWhenItEnds )
    {
        std::uint32_t child_id = 0;

        {
            auto worker = launchWorker( echoWorker(), {} );
            const auto reply = ask( worker, "start-child" );
            ASSERT_FALSE( reply.starts_with( "failed" ) ) << reply;
            child_id = static_cast<std::uint32_t>( std::stoul( reply ) );
            ASSERT_FALSE( hasEnded( child_id, std::chrono::milliseconds( 0 ) ) );
        }

        EXPECT_TRUE( hasEnded( child_id, std::chrono::milliseconds( 5000 ) ) );
    }

    TEST( WorkerTests, ReportsAProgramThatCannotStart )
    {
        WorkerProgram missing{ std::filesystem::path( TOOL_BROKER_ECHO_WORKER ).parent_path() / "no-such-worker.exe", {} };

        EXPECT_THROW( launchWorker( missing, {} ), std::system_error );
    }
}
