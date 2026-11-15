/**
 * @file HttpTransportBackend.Apple.ixx
 * @brief The HTTP transports a macOS build has: Apple's URL loading, and Network.framework for a checked network.
 *
 * One of three candidate files for Networking.HttpTransportBackend; CMake compiles this one on Apple when
 * MILA_ENABLE_LIBCURL is ON, and no libcurl is built there (HttpClient.md section 9).
 */

export module Networking.HttpTransportBackend;

import std;
import Networking.HttpTransport;

export import Networking.UrlSessionHttpTransport;
export import Networking.NetworkHttpTransport;

namespace Mila::Networking
{
    /**
     * @brief Whether this build can perform an HTTP request itself.
     *
     * A consumer tests this to say so plainly rather than reporting an empty listing, which
     * would claim a publisher has nothing when the truth is that nothing can be asked.
     */
    export inline constexpr bool kHttpTransportAvailable = true;

    /**
     * @brief The transport a consumer gets when it does not supply its own: Apple's URL loading, with the Keychain's
     *        trust and the Mac's proxy settings.
     */
    export std::shared_ptr<const IHttpTransport> makeDefaultHttpTransport()
    {
        return std::make_shared<const UrlSessionHttpTransport>();
    }

    /**
     * @brief A transport that makes only the connections a check allows, and none through a proxy.
     */
    export std::shared_ptr<const IHttpTransport> makeCheckedHttpTransport( ConnectionCheck check )
    {
        return std::make_shared<const NetworkHttpTransport>( std::move( check ) );
    }
}
