/**
 * @file ToolBroker.LineChannel.ixx
 * @brief One MCP message per line over a byte stream: the framing of a worker's pipes and of the broker's own.
 */

export module ToolBroker.LineChannel;

import std;

namespace Mila::ToolBroker
{
    /**
     * @brief Sends and receives newline-delimited messages over two operating-system handles it owns.
     *
     * MCP's stdio binding delimits messages by newlines and forbids one inside a message, and a custom transport
     * over a byte stream should reuse that framing (MCP 2026-07-28, *Transports*), so the named pipe uses it too.
     * The channel moves bytes; it does not parse JSON.
     */
    export class LineChannel
    {
    public:
        /** The default cap on one received message, so a worker that never ends a line cannot exhaust memory. */
        static constexpr std::size_t kDefaultMessageLimit = 16 * 1024 * 1024;

        /**
         * Takes ownership of both handles. They must be distinct handles, though they may name one duplex pipe.
         */
        LineChannel( void* read_handle, void* write_handle, std::size_t message_limit = kDefaultMessageLimit );

        ~LineChannel();

        LineChannel( LineChannel&& other ) noexcept;
        LineChannel& operator=( LineChannel&& other ) noexcept;

        LineChannel( const LineChannel& ) = delete;
        LineChannel& operator=( const LineChannel& ) = delete;

        /**
         * @brief Writes one message and the newline that ends it.
         * @throws std::invalid_argument if the message holds a newline.
         * @throws std::logic_error if the sending side was closed.
         * @throws std::system_error if the write fails.
         */
        void send( std::string_view message );

        /**
         * @brief The next message, without its newline or a carriage return before it.
         * @return Nothing once the other end has closed. A last line the other end never finished is not a
         *         message, and is dropped.
         * @throws std::length_error if a message exceeds the channel's limit.
         * @throws std::system_error if the read fails.
         */
        std::optional<std::string> receive();

        /** Closes the sending side, which is how a worker is told to exit (MCP stdio, *Shutdown*). */
        void closeSending() noexcept;

    private:
        void close() noexcept;

        void* read_handle_ = nullptr;
        void* write_handle_ = nullptr;
        std::size_t message_limit_;
        std::string buffer_;
        bool end_of_stream_ = false;
    };
}
