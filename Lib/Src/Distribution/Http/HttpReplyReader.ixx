/**
 * @file HttpReplyReader.ixx
 * @brief An HTTP/1.1 reply read as its bytes arrive: the head, then the body by length, by chunks or to the close.
 *
 * For a transport that speaks HTTP itself over a connection it holds (HttpClient.md section 9).
 */

export module Distribution.HttpReplyReader;

import std;

namespace Mila::Distribution
{
    /**
     * @brief Reads one reply to one request on a connection that closes after it.
     *
     * An interim 1xx reply is passed over; the head kept is the final one. The head is bounded, as are a chunk's size
     * line and the trailers, so a hostile server cannot grow what is held. A reply to a HEAD request is never read
     * here: no transport sends one.
     */
    export class HttpReplyReader
    {
    public:

        /// Called with each run of body bytes, the chunk framing removed. Return false to stop reading.
        using BodyCallback = std::function<bool( const char* data, std::size_t length )>;

        /// The most a reply's head, a chunk's size line or its trailers may hold.
        static constexpr std::size_t kMaximumHeadBytes = 64 * 1024;

        /**
         * @brief Reads the next bytes of the reply.
         *
         * @returns false when the reply is malformed (error() says how) or the callback stopped it (stopped());
         *          true otherwise, complete() saying whether the reply has ended. Bytes after the end are ignored.
         */
        bool read( std::string_view bytes, const BodyCallback& body );

        /**
         * @brief The connection closed: true when that ends the reply, false when the reply was cut short.
         */
        bool finish();

        /// Whether the final head has been read, so status(), location() and contentLength() are known.
        [[nodiscard]] bool headComplete() const noexcept;

        /// Whether the whole reply has been read.
        [[nodiscard]] bool complete() const noexcept;

        /// Whether the body callback stopped the reading.
        [[nodiscard]] bool stopped() const noexcept;

        [[nodiscard]] long status() const noexcept;

        /// The Location header verbatim, empty when absent.
        [[nodiscard]] const std::string& location() const noexcept;

        /// The Content-Length header, zero when absent or when the body comes in chunks.
        [[nodiscard]] std::uint64_t contentLength() const noexcept;

        /// What was malformed, empty when nothing was.
        [[nodiscard]] const std::string& error() const noexcept;

    private:

        enum class Stage
        {
            Head,
            BodyByLength,
            BodyToClose,
            ChunkSize,
            ChunkData,
            ChunkEnd,
            Trailers,
            Complete,
            Failed
        };

        bool fail( std::string reason );
        bool readHead( std::string_view& bytes );
        bool parseHead();
        bool readLine( std::string_view& bytes, bool& ended );
        bool readChunkSize( std::string_view& bytes );
        bool readChunkEnd( std::string_view& bytes );
        bool readTrailers( std::string_view& bytes );
        bool deliver( std::string_view bytes, const BodyCallback& body );

        Stage stage_{ Stage::Head };
        std::string pending_;
        std::size_t trailer_bytes_{ 0 };
        bool head_complete_{ false };
        long status_{ 0 };
        std::string location_;
        std::uint64_t content_length_{ 0 };
        std::uint64_t remaining_{ 0 };
        bool stopped_{ false };
        std::string error_;
    };
}
