module;
#include <memory_resource>

export module Compute.MemoryResource;

import Compute.MemoryResourceProperties;

namespace Mila::Dnn::Compute
{
    export using MemoryResource = std::pmr::memory_resource;

    template<typename T>
    concept IsHostAccessible = requires {
        { T::is_host_accessible() } -> std::convertible_to<bool>;
    };

    template<typename T>
    concept IsDeviceAccessible = requires {
        { T::is_device_accessible() } -> std::convertible_to<bool>;
    };

    /*export bool isHostAccessible( const MemoryResource& resource ) {
        if constexpr ( IsHostAccessible<decltype(resource)> ) {
            return resource.is_host_accessible();
        }
        return false;
    };

    export bool isDeviceAccessible( const MemoryResource& resource ) {
        if constexpr ( IsDeviceAccessible<decltype(resource)> ) {
            return resource.is_device_accessible();
        }
        return false;
    };*/
}
