/**
 * @file ExecutionContext.Traits.ixx
 * @brief Maps a device type to its concrete execution context class.
 */

export module Compute.ExecutionContextTraits;

import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @internal
     * @brief Device-to-context map behind the ExecutionContext alias template.
     *
     * Each backend specializes this in its own module with a single `type` alias. The
     * indirection exists because MSVC 14.51 fails to complete a reachable explicit
     * specialization when the dereference is a dependent expression instantiated in a
     * consumer's translation unit, while a plain class in the same position completes
     * correctly. Keeping the specializations here -- and the context classes plain --
     * lets the aggregator publish this map without publishing the classes themselves.
     *
     * @tparam TDeviceType The device type to map.
     */
    export template<DeviceType TDeviceType>
    struct ExecutionContextTraits;
}
