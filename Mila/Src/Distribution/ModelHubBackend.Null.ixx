/**
 * @file ModelHubBackend.Null.ixx
 * @brief The hub backend for a build with no transport: one that refuses, by name.
 *
 * One of two candidate files for Distribution.ModelHubBackend; CMake compiles this one when
 * MILA_ENABLE_MODEL_HUB is OFF. The store, packaging and install are unaffected -- only
 * reaching a remote is gone, so that is the only thing this refuses.
 */

module;
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

export module Distribution.ModelHubBackend;

import Distribution.ModelCoordinate;
import Distribution.ModelStore;
import Distribution.ModelHub;

namespace Mila::Distribution
{
    /**
     * @brief Whether this build can reach a hub at all.
     *
     * A consumer tests this to say so plainly rather than reporting an empty listing, which
     * would claim a publisher has nothing when the truth is that nothing can be asked.
     */
    export inline constexpr bool kModelHubAvailable = false;

    /**
     * @brief The hub of a build compiled without one.
     *
     * Every operation fails with the same fact rather than a transport error, because there is
     * no transport to have failed. Installing a package directory still works, which is how a
     * build like this is meant to acquire a model.
     */
    export class NullModelHub : public IModelHub
    {
    public:

        std::string name() const override { return "none"; }

        std::vector<HubModel> listModels( const std::string& ) const override
        {
            throw std::runtime_error( refusal() );
        }

        std::string fetchManifest( const ModelCoordinate& ) const override
        {
            throw std::runtime_error( refusal() );
        }

        FetchReport fetchFile(
            const ModelCoordinate&,
            const std::string&,
            uint64_t,
            const std::function<bool( const char*, size_t )>& ) const override
        {
            return { FetchOutcome::Failed, refusal() };
        }

    private:

        static std::string refusal()
        {
            return "This build was compiled with MILA_ENABLE_MODEL_HUB=OFF and cannot reach a "
                "model hub. Install a model package from disk instead; installed models load "
                "normally.";
        }
    };

    /**
     * @brief The hub a consumer gets when it does not care which one.
     *
     * Returns an object in every build, so a caller never null-checks; what varies is whether
     * that object can reach a network. The progress callback is accepted and ignored, so a
     * consumer needs no second call shape for a build without a hub.
     */
    export std::unique_ptr<IModelHub> makeDefaultModelHub(
        std::function<bool( uint64_t bytes_so_far, uint64_t total_bytes )> = {} )
    {
        return std::make_unique<NullModelHub>();
    }
}
