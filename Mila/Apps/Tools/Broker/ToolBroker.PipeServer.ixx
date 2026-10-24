/**
 * @file ToolBroker.PipeServer.ixx
 * @brief The broker's front door: a named pipe only the current user can open.
 */

export module ToolBroker.PipeServer;

import std;
import ToolBroker.LineChannel;

namespace Mila::ToolBroker
{
    /**
     * @brief Listens on a named pipe whose access list names the current user alone, and which refuses clients
     *        on other machines.
     *
     * It is created as the first instance of its name, so a program that took the name first makes the broker
     * refuse to start rather than hand its clients to that program. Another program running as the same user can
     * still open the pipe, which the threat model accepts (`Notebooks/LocalToolService.md` section 6.1).
     */
    export class PipeServer
    {
    public:
        /**
         * @param name The full pipe path, `\\.\pipe\...`.
         * @throws std::system_error if the name is already in use or the pipe cannot be created.
         */
        explicit PipeServer( std::wstring name );

        ~PipeServer();

        PipeServer( PipeServer&& other ) noexcept;
        PipeServer& operator=( PipeServer&& other ) noexcept;

        PipeServer( const PipeServer& ) = delete;
        PipeServer& operator=( const PipeServer& ) = delete;

        /**
         * @brief Waits for the next client and returns its channel, and keeps listening on a fresh instance.
         *
         * The channel's two handles name one synchronous pipe, so it is used request then reply from one thread:
         * a read blocked on it holds up a write. Overlapped input and output come with a broker that must send
         * while it waits to receive.
         *
         * @throws std::system_error if the connection or the next instance fails.
         */
        LineChannel accept();

        const std::wstring& name() const noexcept;

    private:
        void* createInstance( bool first );

        std::wstring name_;
        void* listening_ = nullptr;
    };
}
