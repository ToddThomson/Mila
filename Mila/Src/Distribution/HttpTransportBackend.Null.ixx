/**
 * @file HttpTransportBackend.Null.ixx
 * @brief The transport for a build with no HTTP client of its own: one that refuses, by name.
 *
 * One of two candidate files for Distribution.HttpTransportBackend; CMake compiles this one
 * when MILA_ENABLE_LIBCURL is OFF. Only moving bytes is gone -- the store, packaging, install,
 * pull, and every HuggingFace URL, token and status rule are still compiled, so a caller that
 * supplies its own transport loses nothing at all. That is how the Python wheel works.
 */

module;
#include <memory>
#include <string>

export module Distribution.HttpTransportBackend;

import Distribution.HttpTransport;

namespace Mila::Distribution
{
    /**
     * @brief Whether this build can perform an HTTP request itself.
     *
     * A consumer tests this to say so plainly rather than reporting an empty listing, which
     * would claim a publisher has nothing when the truth is that nothing can be asked.
     */
    export inline constexpr bool kHttpTransportAvailable = false;

    /**
     * @brief The transport of a build compiled without one.
     *
     * Fails with the same fact every time rather than a network error, because there is no
     * network call to have failed. Supplying a transport is the supported answer, not a
     * workaround -- this build knows everything about HuggingFace except how to reach it.
     */
    export class NullHttpTransport : public IHttpTransport
    {
    public:

        std::string name() const override { return "none"; }

        HttpResponse fetch(
            const HttpFetch&,
            const SinkCallback&,
            const HeadersCallback& ) const override
        {
            HttpResponse response;
            response.transport_failed = true;
            response.message =
                "This build was compiled with MILA_ENABLE_LIBCURL=OFF and has no HTTP client. "
                "Supply a transport, or install the model from a package on disk.";

            return response;
        }
    };

    /**
     * @brief The transport a consumer gets when it does not supply its own.
     *
     * Never null, so a caller need not check; what varies between builds is whether the
     * object it returns can reach a network.
     */
    export std::shared_ptr<const IHttpTransport> makeDefaultHttpTransport()
    {
        return std::make_shared<const NullHttpTransport>();
    }
}
