/**
 * @file DeploymentRequest.ixx
 * @brief What a caller asks the deployment planner for, the same on every surface.
 *
 * Each plannable value is either fixed or left to the planner. See Specifications/Deployment.md section 3.3.
 */

module;
#include <cstddef>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>

export module Deployment.DeploymentRequest;

export import Dnn.LanguageModelConfig;
import Dnn.TensorTypes;
import Compute.DeviceId;

namespace Mila::Deployment
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief A deployment request: the context length fixed or automatic, the device, and the headroom.
     *
     * The weight format, KV-cache compression and head width are inherited from LanguageModelConfig and
     * are always the caller's; the planner never chooses them (Deployment.md 12.1). The context length is
     * automatic until withContextLength() fixes it. The library default is every plannable value automatic
     * and zero headroom (12.6).
     */
    export struct DeploymentRequest : LanguageModelConfig<DeploymentRequest>
    {
        /// The grid the planner searches automatic context lengths on, and the smallest it will choose.
        static constexpr dim_t kContextStep = 1024;

        DeploymentRequest() = default;

        /**
         * @brief The same deployment a family's model config describes, with its context length fixed.
         */
        template<typename TConfig>
        static DeploymentRequest fromModelConfig( const LanguageModelConfig<TConfig>& config )
        {
            DeploymentRequest request;
            request.withWeightQuantization( config.getWeightQuantization() )
                .withKvCacheCompression( config.getKvCacheCompression() )
                .withLanguageModelHeadPositions( config.getLanguageModelHeadPositions() );

            if ( config.getContextLength() > 0 )
                request.withContextLength( config.getContextLength() );

            return request;
        }

        /**
         * @brief Leave the context length to the planner, between a floor and a ceiling.
         *
         * @param floor   The shortest context the caller accepts. Nothing fitting at or above it is a
         *                refusal.
         * @param ceiling The longest context worth considering; the model's trained maximum caps it
         *                anyway. Zero means that maximum.
         * @throws std::invalid_argument when the floor is below one grid step or above a nonzero ceiling.
         */
        DeploymentRequest& withAutomaticContextLength( dim_t floor = kContextStep, dim_t ceiling = 0 )
        {
            if ( floor < kContextStep || ( ceiling != 0 && floor > ceiling ) )
            {
                throw std::invalid_argument( std::format(
                    "DeploymentRequest: automatic context floor {} must be at least {} and not above the ceiling {}",
                    floor, kContextStep, ceiling ) );
            }

            context_length_ = 0;
            context_floor_ = floor;
            context_ceiling_ = ceiling;

            return *this;
        }

        /**
         * @brief Bytes to leave free on the device: what another process or a display needs, which only the
         *        caller knows.
         */
        DeploymentRequest& withHeadroom( std::size_t bytes )
        {
            headroom_bytes_ = bytes;

            return *this;
        }

        /// The one device to plan on (Phase 3). Unset means the family's device type at ordinal 0.
        DeploymentRequest& withDevice( DeviceId device )
        {
            device_ = device;

            return *this;
        }

        [[nodiscard]] bool isContextLengthAutomatic() const noexcept
        {
            return context_length_ == 0;
        }

        [[nodiscard]] dim_t getContextFloor() const noexcept
        {
            return context_floor_;
        }

        /// Zero when the model's trained maximum is the ceiling.
        [[nodiscard]] dim_t getContextCeiling() const noexcept
        {
            return context_ceiling_;
        }

        [[nodiscard]] std::size_t getHeadroom() const noexcept
        {
            return headroom_bytes_;
        }

        [[nodiscard]] const std::optional<DeviceId>& getDevice() const noexcept
        {
            return device_;
        }

        std::string toString() const
        {
            std::string result = "DeploymentRequest:\n";
            result += baseToString();
            result += std::format( "  context:             {}\n", isContextLengthAutomatic()
                ? std::format( "automatic, {} to {}", context_floor_,
                    context_ceiling_ == 0 ? std::string( "trained maximum" ) : std::to_string( context_ceiling_ ) )
                : std::to_string( context_length_ ) );
            result += std::format( "  headroom_bytes:      {}\n", headroom_bytes_ );

            return result;
        }

    private:

        dim_t context_floor_{ kContextStep };
        dim_t context_ceiling_{ 0 };
        std::size_t headroom_bytes_{ 0 };
        std::optional<DeviceId> device_;
    };
}
