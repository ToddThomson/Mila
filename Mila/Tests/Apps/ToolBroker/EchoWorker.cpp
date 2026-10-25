/**
 * @file EchoWorker.cpp
 * @brief A worker for the tool broker's tests: answers each line on standard input with one line on standard output.
 *
 * `environment`, `directory`, `arguments` and `start-child` report what the worker was given; `keep-running` makes
 * it ignore the end of its input, as a worker that does not shut down would; anything else is echoed. Started with
 * `--json [label]`, it is an MCP server instead, whose replies carry the label, and its `test/` methods misbehave
 * on request.
 */

#define NOMINMAX
#include <windows.h>

import std;
import ToolBroker.Json;

namespace
{
    void sendJson( const nlohmann::json& message )
    {
        std::cout << message.dump() << '\n' << std::flush;
    }

    /**
     * Answers `tools/list` with one tool, `echo`; and every other request with a complete result whose text is
     * `arguments.text` after `label:` and the tool's name, or `arguments.bytes` long, so a caller chooses the reply's
     * size -- except for the `test/` methods.
     */
    int runJson( std::string_view label )
    {
        std::string line;

        while ( std::getline( std::cin, line ) )
        {
            const auto request = nlohmann::json::parse( line, nullptr, false );

            if ( request.is_discarded() || !request.contains( "id" ) )
            {
                continue;
            }

            const auto& id = request[ "id" ];
            const auto method = request.value( "method", "" );

            if ( method == "test/notify-first" )
            {
                sendJson( { { "jsonrpc", "2.0" }, { "method", "notifications/progress" },
                    { "params", { { "progress", 1 } } } } );
            } else if ( method == "test/wrong-id" ) {
                sendJson( { { "jsonrpc", "2.0" }, { "id", "another" }, { "result", { { "resultType", "complete" } } } } );
                continue;
            } else if ( method == "test/send-request" ) {
                sendJson( { { "jsonrpc", "2.0" }, { "id", 99 }, { "method", "roots/list" } } );
                continue;
            } else if ( method == "test/not-json" ) {
                std::cout << "not json\n" << std::flush;
                continue;
            } else if ( method == "test/exit" ) {
                return 0;
            } else if ( method == "tools/list" ) {
                const nlohmann::json tool = { { "name", "echo" }, { "description", "Repeats its text" },
                    { "inputSchema", { { "type", "object" }, { "properties", { { "text", { { "type", "string" } } } } } } } };
                sendJson( { { "jsonrpc", "2.0" }, { "id", id },
                    { "result", { { "resultType", "complete" }, { "tools", nlohmann::json::array( { tool } ) } } } } );
                continue;
            }

            std::string text;

            if ( request.contains( "params" ) && request[ "params" ].contains( "arguments" ) )
            {
                const auto& params = request[ "params" ];
                const auto& arguments = params[ "arguments" ];

                if ( arguments.contains( "text" ) )
                {
                    text = std::format( "{}:{}:{}", label, params.value( "name", "" ), arguments[ "text" ].get<std::string>() );
                } else {
                    text = std::string( arguments.value( "bytes", std::size_t{ 0 } ), 'x' );
                }
            }

            sendJson( { { "jsonrpc", "2.0" }, { "id", id },
                { "result", { { "resultType", "complete" },
                    { "content", nlohmann::json::array( { { { "type", "text" }, { "text", text } } } ) } } } } );
        }

        return 0;
    }

    std::string narrow( std::wstring_view text )
    {
        if ( text.empty() )
        {
            return {};
        }

        const int size = ::WideCharToMultiByte( CP_UTF8, 0, text.data(), static_cast<int>( text.size() ), nullptr, 0,
            nullptr, nullptr );
        std::string result( static_cast<std::size_t>( size ), '\0' );
        ::WideCharToMultiByte( CP_UTF8, 0, text.data(), static_cast<int>( text.size() ), result.data(), size, nullptr,
            nullptr );

        return result;
    }

    std::string environmentNames()
    {
        std::vector<std::string> names;
        wchar_t* block = ::GetEnvironmentStringsW();

        for ( const wchar_t* entry = block; *entry != L'\0'; entry += std::wcslen( entry ) + 1 )
        {
            const std::wstring_view variable( entry );
            names.push_back( narrow( variable.substr( 0, variable.find( L'=', 1 ) ) ) );
        }

        ::FreeEnvironmentStringsW( block );
        std::ranges::sort( names );

        std::string joined;

        for ( const auto& name : names )
        {
            joined += joined.empty() ? name : "," + name;
        }

        return joined;
    }

    std::string startChild()
    {
        std::array<wchar_t, 32768> path{};
        ::GetModuleFileNameW( nullptr, path.data(), static_cast<DWORD>( path.size() ) );

        std::wstring command = L"\"" + std::wstring( path.data() ) + L"\" --wait";
        STARTUPINFOW startup{ sizeof( STARTUPINFOW ) };
        PROCESS_INFORMATION process{};

        if ( !::CreateProcessW( nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr,
                &startup, &process ) )
        {
            return "failed " + std::to_string( ::GetLastError() );
        }

        ::CloseHandle( process.hThread );
        ::CloseHandle( process.hProcess );

        return std::to_string( process.dwProcessId );
    }
}

int main( int argc, char** argv )
{
    if ( argc == 2 && std::string_view( argv[ 1 ] ) == "--wait" )
    {
        ::Sleep( INFINITE );
    }

    if ( argc >= 2 && std::string_view( argv[ 1 ] ) == "--json" )
    {
        return runJson( argc >= 3 ? argv[ 2 ] : "echo" );
    }

    bool keep_running = false;
    std::string line;

    while ( std::getline( std::cin, line ) )
    {
        std::string reply;

        if ( line == "environment" )
        {
            reply = environmentNames();
        } else if ( line == "directory" ) {
            reply = std::filesystem::current_path().string();
        } else if ( line == "arguments" ) {
            for ( int index = 1; index < argc; ++index )
            {
                reply += index == 1 ? std::string( argv[ index ] ) : "|" + std::string( argv[ index ] );
            }
        } else if ( line == "start-child" ) {
            reply = startChild();
        } else if ( line == "keep-running" ) {
            keep_running = true;
            reply = "ok";
        } else {
            reply = line;
        }

        std::cout << reply << '\n' << std::flush;
    }

    if ( keep_running )
    {
        ::Sleep( INFINITE );
    }

    return 0;
}
