/**
 * @file HttpTransport.ixx
 * @brief One HTTP request, moved. No interpretation of the answer.
 *
 * The narrow half of the split described in Specifications/HttpClient.md: a transport moves
 * bytes for one request; HttpClient decides how to ask and how to read the reply. Nothing
 * security-relevant is decided here, which is why a transport can be supplied by a host
 * language without being trusted with a token.
 */

export module Distribution.HttpTransport;

import std;

namespace Mila::Distribution
{
    export struct HttpHeader
    {
        std::string name;
        std::string value;
    };

    export struct HttpFetch
    {
        std::string url;

        /// Exactly what to send. A transport adds nothing and removes nothing.
        std::vector<HttpHeader> headers;

        /// When set, the request is a POST carrying exactly these bytes, and its Content-Type is one of the headers;
        /// a GET when not.
        std::optional<std::string> body;

        /// Seconds with no data before the transfer is abandoned. Zero disables.
        long low_speed_timeout_seconds{ 60 };
    };

    export struct HttpResponse
    {
        /// Zero when the request never completed.
        long http_code{ 0 };

        /// The Location header verbatim, empty when absent. Resolving it is the client's job.
        std::string location;

        /// Server-reported length of this response body. Zero if absent.
        std::uint64_t content_length{ 0 };

        /// Connection, TLS or I/O failure -- distinct from any HTTP status.
        bool transport_failed{ false };

        /// Detail on failure, or a bounded prefix of a non-2xx body. Never contains a token.
        std::string message;
    };

    /**
     * @brief Called with each chunk of body as it arrives. Return false to abort.
     *
     * The transport neither buffers nor hashes; a caller writing a 6 GB artifact hashes and
     * writes here so the bytes are touched exactly once.
     */
    export using SinkCallback =
        std::function<bool( const char* data, std::size_t length )>;

    /**
     * @brief Called once when the response line and headers have arrived, before any body.
     *
     * Exists so a caller can report progress against a total it would otherwise only learn
     * after the transfer finished.
     */
    export using HeadersCallback =
        std::function<void( long http_code, std::uint64_t content_length )>;

    /**
     * @brief Called with the address of each connection a transport is about to make, before it is made; false
     *        refuses it.
     *
     * The address is 4 bytes for IPv4 or 16 for IPv6, in network order -- the one connected to, after the name was
     * resolved, so a check here decides on the connection and not on a host name.
     */
    export using ConnectionCheck =
        std::function<bool( std::span<const std::uint8_t> address )>;

    /**
     * @brief Performs one HTTP GET, or a POST when the request carries a body.
     *
     * A transport that cannot send a body answers a request with one as a transport failure, never as a GET.
     *
     * Three obligations, and they are the whole contract:
     *
     * **Do not follow redirects.** Report the status and the Location header; the client
     * decides whether to follow and what may be sent to the next host.
     *
     * **Send exactly the headers given.** No token discovery, no Range construction, no
     * additions of any kind. A transport is never told a token -- it is handed headers the
     * client has already decided are safe for this exact host, which is what makes an
     * untrusted implementation safe.
     *
     * **Do not deliver a non-2xx body to the sink.** A redirect or error body must never
     * reach a caller that is hashing bytes into a blob; capture a bounded prefix into
     * message instead.
     */
    export class IHttpTransport
    {
    public:

        virtual ~IHttpTransport() = default;

        /// Stable identifier, for messages. e.g. "libcurl".
        virtual std::string name() const = 0;

        virtual HttpResponse fetch(
            const HttpFetch& request,
            const SinkCallback& sink,
            const HeadersCallback& on_headers ) const = 0;
    };
}
