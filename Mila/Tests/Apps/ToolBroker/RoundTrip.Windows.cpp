/**
 * @file RoundTrip.Windows.cpp
 * @brief Phase 3's measurement (Notebooks/LocalToolService.md section 11): a client's call through the broker to
 *        a resident worker and back, against the same call made to the worker directly.
 */

#define NOMINMAX
#include <windows.h>

import std;
import ToolBroker.Json;
import ToolBroker.LineChannel;
import ToolBroker.PipeServer;
import ToolBroker.Relay;
import ToolBroker.Worker;
import ToolBroker.WorkerProgram;

namespace
{
    using Mila::ToolBroker::LineChannel;

    constexpr int kWarmUp = 1000;
    constexpr int kCalls = 20000;
    constexpr std::size_t kRequestBytes = 256;
    constexpr std::size_t kReplyBytes = 1024;

    std::string request( int id, std::size_t reply_text, std::size_t padding )
    {
        nlohmann::json message = {
            { "jsonrpc", "2.0" },
            { "id", id },
            { "method", "tools/call" },
            { "params", {
                { "name", "echo" },
                { "arguments", { { "bytes", reply_text }, { "padding", std::string( padding, 'p' ) } } },
                { "_meta", {
                    { "io.modelcontextprotocol/protocolVersion", "2026-07-28" },
                    { "io.modelcontextprotocol/clientCapabilities", nlohmann::json::object() } } } } }
        };

        return message.dump();
    }

    /** One call: send, read the answer, parse it as the client would. Returns the answer's length. */
    std::size_t call( LineChannel& channel, const std::string& message )
    {
        channel.send( message );
        const auto answer = channel.receive().value();
        const auto parsed = nlohmann::json::parse( answer );

        if ( !parsed.contains( "result" ) )
        {
            throw std::runtime_error( "The call failed: " + answer );
        }

        return answer.size();
    }

    struct Summary
    {
        double median;
        double p99;
        double max;
    };

    Summary summarize( std::vector<double> microseconds )
    {
        std::ranges::sort( microseconds );
        const auto at = [&]( double fraction )
            {
                return microseconds[ static_cast<std::size_t>( fraction * ( microseconds.size() - 1 ) ) ];
            };

        return { at( 0.5 ), at( 0.99 ), microseconds.back() };
    }

    /** Sizes the request and reply to the gate's 256 bytes and 1 KiB, then times `kCalls` calls after warm-up. */
    Summary measure( LineChannel& channel, std::size_t& request_size, std::size_t& reply_size )
    {
        const auto bare_reply = call( channel, request( 0, 0, 0 ) );
        const auto reply_text = kReplyBytes - bare_reply;
        const auto bare_request = request( 0, reply_text, 0 ).size();
        const auto padding = kRequestBytes - bare_request;

        for ( int index = 0; index < kWarmUp; ++index )
        {
            call( channel, request( index, reply_text, padding ) );
        }

        std::vector<std::string> messages;
        messages.reserve( kCalls );

        for ( int index = 0; index < kCalls; ++index )
        {
            messages.push_back( request( kWarmUp + index, reply_text, padding ) );
        }

        request_size = messages.front().size();
        std::vector<double> microseconds;
        microseconds.reserve( kCalls );

        for ( const auto& message : messages )
        {
            const auto start = std::chrono::steady_clock::now();
            reply_size = call( channel, message );
            const auto elapsed = std::chrono::steady_clock::now() - start;
            microseconds.push_back( std::chrono::duration<double, std::micro>( elapsed ).count() );
        }

        return summarize( std::move( microseconds ) );
    }

    Mila::ToolBroker::Worker startWorker()
    {
        return Mila::ToolBroker::launchWorker(
            { std::filesystem::path( TOOL_BROKER_ECHO_WORKER ), { L"--json" } }, {} );
    }

    LineChannel openPipe( const std::wstring& name )
    {
        HANDLE pipe = ::CreateFileW( name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr );

        if ( pipe == INVALID_HANDLE_VALUE )
        {
            throw std::runtime_error( "Opening the broker's pipe failed" );
        }

        HANDLE duplicate = nullptr;
        ::DuplicateHandle( ::GetCurrentProcess(), pipe, ::GetCurrentProcess(), &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS );

        return LineChannel( pipe, duplicate );
    }

    void report( std::string_view arm, const Summary& summary, std::size_t request_size, std::size_t reply_size )
    {
        std::cout << std::format( "{:<8} median {:8.1f} us   p99 {:8.1f} us   max {:9.1f} us   request {} B, reply {} B\n",
            arm, summary.median, summary.p99, summary.max, request_size, reply_size );
    }
}

int main()
{
    std::size_t request_size = 0;
    std::size_t reply_size = 0;

    // The direct arm: no broker, the client is the worker's own client.
    auto direct_worker = startWorker();
    const auto direct = measure( direct_worker.channel(), request_size, reply_size );
    report( "direct", direct, request_size, reply_size );

    // The broker arm: client, pipe, broker (parse, check, forward), worker, and back.
    const auto name = std::format( L"\\\\.\\pipe\\mila-tool-broker-round-trip-{}", ::GetCurrentProcessId() );
    Mila::ToolBroker::PipeServer server( name );
    auto broker_worker = startWorker();

    std::jthread broker( [&server, &broker_worker]
        {
            auto client = server.accept();

            while ( Mila::ToolBroker::relayOne( client, broker_worker ) )
            {
            }
        } );

    auto client = openPipe( name );
    const auto through_broker = measure( client, request_size, reply_size );
    report( "broker", through_broker, request_size, reply_size );
    client.closeSending();

    std::cout << std::format( "gate (broker median < 100 us): {}   forced failure (direct < broker): {}\n",
        through_broker.median < 100.0 ? "pass" : "FAIL", direct.median < through_broker.median ? "holds" : "VIOLATED" );

    return through_broker.median < 100.0 && direct.median < through_broker.median ? 0 : 1;
}
