/**
 * @file DeploymentRefusal.ixx
 * @brief The planner's answer when nothing fits what the request allows: why, each allowed device's shortfall,
 *        and the devices it was not allowed to use.
 *
 * A returned value rather than an exception: "does not fit" is an answer, not an error. See
 * Specifications/Deployment.md sections 3.3, 6 and 12.8.
 */

export module Deployment.DeploymentRefusal;

import std;
import Deployment.DeviceReading;
export import Deployment.DeviceShortfall;

namespace Mila::Deployment
{
    /**
     * @brief Why a request has no plan, with what each allowed device said and the present devices it did not allow.
     *
     * The devices not allowed are named so that "nothing fits on what you allowed" is never reported as "nothing
     * fits on this machine". They are described, never read (DeviceReading::describe), so they report no memory.
     * The library never phrases a reason for a user; each adaptor explains it in its own voice. toString() is for a
     * developer reading a log or an exception.
     */
    export class DeploymentRefusal
    {
    public:

        enum class Reason
        {
            /// Every allowed device was planned and none holds the request; shortfalls() says why for each.
            NoAllowedDeviceFits,

            /// A device the request named is not present; absent() lists them. A named device is never replaced by
            /// another.
            AllowedDeviceAbsent,

            /// The request named no device and the build sees none this model runs on.
            NoDeviceToPlanOn
        };

        DeploymentRefusal(
            Reason reason,
            std::vector<DeviceShortfall> shortfalls,
            std::vector<DeviceReading> not_allowed,
            std::vector<std::string> absent = {} )
            : reason_( reason ), shortfalls_( std::move( shortfalls ) ), not_allowed_( std::move( not_allowed ) ),
              absent_( std::move( absent ) )
        {
        }

        [[nodiscard]] Reason reason() const noexcept
        {
            return reason_;
        }

        /// One per allowed device planned, in device order; empty unless the reason is NoAllowedDeviceFits.
        [[nodiscard]] std::span<const DeviceShortfall> shortfalls() const noexcept
        {
            return shortfalls_;
        }

        /// The present devices this model runs on that the request did not allow, described and not read.
        [[nodiscard]] std::span<const DeviceReading> notAllowed() const noexcept
        {
            return not_allowed_;
        }

        /// The names the request gave that match no present device, as written.
        [[nodiscard]] std::span<const std::string> absent() const noexcept
        {
            return absent_;
        }

        static std::string_view nameOf( Reason reason ) noexcept
        {
            switch ( reason )
            {
                case Reason::NoAllowedDeviceFits: return "NoAllowedDeviceFits";
                case Reason::AllowedDeviceAbsent: return "AllowedDeviceAbsent";
                case Reason::NoDeviceToPlanOn: return "NoDeviceToPlanOn";
            }

            return "Unknown";
        }

        std::string toString() const
        {
            std::string text( nameOf( reason_ ) );

            if ( !absent_.empty() )
                text += std::format( ": not present: {}", joined( absent_, []( const std::string& name ) { return name; } ) );

            for ( const DeviceShortfall& shortfall : shortfalls_ )
            {
                text += "; " + shortfall.toString();
            }

            if ( !not_allowed_.empty() )
            {
                text += std::format( "; not allowed: {}", joined( not_allowed_,
                    []( const DeviceReading& device ) { return device.device.toString(); } ) );
            }

            return text;
        }

    private:

        template<typename TItem, typename TName>
        static std::string joined( const std::vector<TItem>& items, TName&& name )
        {
            std::string text;

            for ( const TItem& item : items )
            {
                if ( !text.empty() )
                    text += ", ";

                text += name( item );
            }

            return text;
        }

        Reason reason_;
        std::vector<DeviceShortfall> shortfalls_;
        std::vector<DeviceReading> not_allowed_;
        std::vector<std::string> absent_;
    };
}
