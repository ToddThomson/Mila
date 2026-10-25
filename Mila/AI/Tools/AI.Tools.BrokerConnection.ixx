/**
 * @file AI.Tools.BrokerConnection.ixx
 * @brief The client's end of the broker's named pipe: one MCP message per line.
 */

export module AI.Tools.BrokerConnection;

import std;

namespace Mila::AI
{
    /**
     * @brief Sends and receives newline-delimited messages over the broker's pipe, MCP's stdio framing.
     *
     * One handle carries both directions synchronously, so it is used request then reply from one thread.
     */
    export class BrokerConnection
    {
    public:
        /** The cap on one received message, so a broker that never ends a line cannot exhaust memory. */
        static constexpr std::size_t kMessageLimit = 16 * 1024 * 1024;

        /**
         * @param pipe_name The full pipe path, `\\.\pipe\...`.
         * @throws std::system_error if the pipe cannot be opened.
         */
        explicit BrokerConnection( const std::wstring& pipe_name );

        ~BrokerConnection();

        BrokerConnection( BrokerConnection&& other ) noexcept;
        BrokerConnection& operator=( BrokerConnection&& other ) noexcept;

        BrokerConnection( const BrokerConnection& ) = delete;
        BrokerConnection& operator=( const BrokerConnection& ) = delete;

        /**
         * @throws std::invalid_argument if the message holds a newline.
         * @throws std::system_error if the write fails.
         */
        void send( std::string_view message );

        /**
         * @return The next message, or nothing once the broker has closed.
         * @throws std::length_error if a message exceeds `kMessageLimit`.
         * @throws std::system_error if the read fails.
         */
        std::optional<std::string> receive();

    private:
        void* pipe_ = nullptr;
        std::string buffer_;
        bool end_of_stream_ = false;
    };
}
