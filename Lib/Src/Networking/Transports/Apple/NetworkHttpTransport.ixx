/**
 * @file NetworkHttpTransport.ixx
 * @brief Network.framework as an IHttpTransport that makes only the connections a check allows (HttpClient.md 9).
 */

export module Networking.NetworkHttpTransport;

import std;
import Networking.HttpTransport;

namespace Mila::Networking
{
    /**
     * @brief The checked transport on macOS: a tool grant's network.
     *
     * Resolves the URL's host itself, puts each address to the check, and connects to an allowed address, never to
     * the name, so the address checked is the one reached. Before a byte is sent the connection must say that it goes
     * to that address and through no proxy. TLS is verified against the URL's host with the system's trust. Speaks
     * HTTP/1.1, one request to a connection, and only http and https.
     */
    export class NetworkHttpTransport : public IHttpTransport
    {
    public:

        explicit NetworkHttpTransport( ConnectionCheck check );

        std::string name() const override;

        HttpResponse fetch(
            const HttpFetch& request,
            const SinkCallback& sink,
            const HeadersCallback& on_headers ) const override;

    private:

        ConnectionCheck check_;
    };
}
