/**
 * @file HttpTransport.ixx
 * @brief One HTTP request, moved. No interpretation of the answer.
 *
 * The narrow half of the split described in Specifications/HttpClient.md: a transport moves
 * bytes for one request; HttpClient decides how to ask and how to read the reply. Nothing
 * security-relevant is decided here, which is why a transport can be supplied by a host
 * language without being trusted with a token.
 */

module;
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

export module Distribution.HttpTransport;

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
        uint64_t content_length{ 0 };

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
        std::function<bool( const char* data, size_t length )>;

    /**
     * @brief Called once when the response line and headers have arrived, before any body.
     *
     * Exists so a caller can report progress against a total it would otherwise only learn
     * after the transfer finished.
     */
    export using HeadersCallback =
        std::function<void( long http_code, uint64_t content_length )>;

    /**
     * @brief Performs one HTTP GET.
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
