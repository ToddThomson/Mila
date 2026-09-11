/**
 * @file HttpTransportBackend.Curl.ixx
 * @brief The HTTP transport this build has: libcurl.
 *
 * One of two candidate files for Distribution.HttpTransportBackend; CMake compiles this one
 * when MILA_ENABLE_LIBCURL is ON. Selecting by source file rather than by preprocessor keeps
 * both alternatives compiled code and keeps `import Mila;` free of an `#ifdef`.
 */

module;
#include <memory>

export module Distribution.HttpTransportBackend;

import Distribution.HttpTransport;

export import Distribution.CurlHttpTransport;

namespace Mila::Distribution
{
    /**
     * @brief Whether this build can perform an HTTP request itself.
     *
     * A consumer tests this to say so plainly rather than reporting an empty listing, which
     * would claim a publisher has nothing when the truth is that nothing can be asked.
     */
    export inline constexpr bool kHttpTransportAvailable = true;

    /**
     * @brief The transport a consumer gets when it does not supply its own.
     *
     * Never null, so a caller need not check; what varies between builds is whether the
     * object it returns can reach a network.
     */
    export std::shared_ptr<const IHttpTransport> makeDefaultHttpTransport()
    {
        return std::make_shared<const CurlHttpTransport>();
    }
}
