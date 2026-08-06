/**
 * @file ModelHubBackend.HuggingFace.ixx
 * @brief The hub backend this build has: HuggingFace over Mila's HTTP client.
 *
 * One of two candidate files for Distribution.ModelHubBackend; CMake compiles this one when
 * MILA_ENABLE_MODEL_HUB is ON. Selecting the backend by source file rather than by preprocessor
 * keeps both alternatives compiled code and keeps `import Mila;` free of an #ifdef.
 */

module;
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

export module Distribution.ModelHubBackend;

import Distribution.ModelHub;

export import Distribution.HttpClient;
export import Distribution.HuggingFaceHub;

namespace Mila::Distribution
{
    /**
     * @brief Whether this build can reach a hub at all.
     *
     * A consumer tests this to say so plainly rather than reporting an empty listing, which
     * would claim a publisher has nothing when the truth is that nothing can be asked.
     */
    export inline constexpr bool kModelHubAvailable = true;

    /**
     * @brief The hub a consumer gets when it does not care which one.
     *
     * Returns an object in every build, so a caller never null-checks; what varies is whether
     * that object can reach a network. The token is discovered here rather than passed in
     * because it is the hub's own concern and one publisher makes it a constant.
     */
    export std::unique_ptr<IModelHub> makeDefaultModelHub(
        std::function<bool( uint64_t bytes_so_far, uint64_t total_bytes )> progress = {} )
    {
        return std::make_unique<HuggingFaceHub>(
            discoverHuggingFaceToken(), std::move( progress ) );
    }
}
