module;
#include <type_traits>

export module Compute.DeviceTypeTraits;

import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    // Primary unspecialized template -- will fail to instantiate for unsupported devices.
    export template <DeviceType TDevice>
    struct DeviceTypeTraits;
}