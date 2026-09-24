/**
 * @file DeploymentRefusal.ixx
 * @brief The planner's answer when nothing fits: why, and the numbers that say so.
 *
 * A returned value rather than an exception: "does not fit" is an answer, not an error. See
 * Specifications/Deployment.md sections 3.3 and 6.
 */

module;
#include <cstddef>
#include <format>
#include <string>
#include <string_view>

export module Deployment.DeploymentRefusal;

import Dnn.Component;
import Dnn.TensorTypes;
import Deployment.DeviceReading;
import Compute.DeviceId;

namespace Mila::Deployment
{
    using namespace Mila::Dnn;

    /**
     * @brief Why a request has no plan, the reading it was decided against, and the priced footprint that did
     *        not fit.
     *
     * The library never phrases a reason for a user; each adaptor explains it in its own voice. toString() is
     * for a developer reading a log or an exception.
     */
    export class DeploymentRefusal
    {
    public:

        enum class Reason
        {
            /// The weights alone exceed every device the request allows. No context length helps.
            WeightsExceedDevice,

            /// The context length the request fixed fits at no prefill chunk.
            FixedContextDoesNotFit,

            /// No automatic context length at or above the request's floor fits.
            NothingAboveTheFloorFits,

            /// The device reported no memory, so an automatic context length has nothing to be chosen against.
            DeviceDoesNotReportMemory
        };

        DeploymentRefusal(
            Reason reason, const DeviceReading& reading, std::size_t headroom_bytes,
            dim_t context_length, const MemoryStats& footprint )
            : reason_( reason ), reading_( reading ), headroom_bytes_( headroom_bytes ),
              context_length_( context_length ), footprint_( footprint )
        {
        }

        [[nodiscard]] Reason reason() const noexcept
        {
            return reason_;
        }

        [[nodiscard]] const DeviceReading& reading() const noexcept
        {
            return reading_;
        }

        [[nodiscard]] std::size_t headroom() const noexcept
        {
            return headroom_bytes_;
        }

        /// The context length the refused footprint was priced at: the fixed one, or the floor.
        [[nodiscard]] dim_t contextLength() const noexcept
        {
            return context_length_;
        }

        /// What that context would allocate at its smallest prefill chunk. Empty when the device reported no
        /// memory and nothing was priced.
        [[nodiscard]] const MemoryStats& footprint() const noexcept
        {
            return footprint_;
        }

        static std::string_view nameOf( Reason reason ) noexcept
        {
            switch ( reason )
            {
                case Reason::WeightsExceedDevice: return "WeightsExceedDevice";
                case Reason::FixedContextDoesNotFit: return "FixedContextDoesNotFit";
                case Reason::NothingAboveTheFloorFits: return "NothingAboveTheFloorFits";
                case Reason::DeviceDoesNotReportMemory: return "DeviceDoesNotReportMemory";
            }

            return "Unknown";
        }

        std::string toString() const
        {
            if ( reason_ == Reason::DeviceDoesNotReportMemory )
            {
                return std::format( "{}: {} reports no memory to plan an automatic context length against",
                    nameOf( reason_ ), reading_.device.toString() );
            }

            return std::format(
                "{}: at context {} the weights need {} bytes and the whole deployment {}, against {} free on {}{}",
                nameOf( reason_ ), context_length_, footprint_.device_parameter_bytes,
                footprint_.totalDeviceBytes(), reading_.free_bytes, reading_.device.toString(),
                headroom_bytes_ == 0 ? std::string{} : std::format( " less {} of headroom", headroom_bytes_ ) );
        }

    private:

        Reason reason_;
        DeviceReading reading_;
        std::size_t headroom_bytes_;
        dim_t context_length_;
        MemoryStats footprint_;
    };
}
