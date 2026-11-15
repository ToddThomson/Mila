/**
 * @file UrlSessionHttpTransport.ixx
 * @brief Apple's URL loading as an IHttpTransport: Keychain trust and the Mac's proxy settings (HttpClient.md 9).
 */

export module Distribution.UrlSessionHttpTransport;

import std;
import Distribution.HttpTransport;

namespace Mila::Distribution
{
    /**
     * @brief The default transport on macOS: one request through an ephemeral URL session, with no cache, no cookies
     *        and no redirect followed.
     *
     * The given headers are sent as given. When the request names none, Mila's User-Agent is sent, and
     * Accept-Encoding: identity, so the body is the resource's own bytes, as curl's request brings them.
     */
    export class UrlSessionHttpTransport : public IHttpTransport
    {
    public:

        std::string name() const override;

        HttpResponse fetch(
            const HttpFetch& request,
            const SinkCallback& sink,
            const HeadersCallback& on_headers ) const override;
    };
}
