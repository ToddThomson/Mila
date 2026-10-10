/**
 * @file Mila_py.Runtime.cpp
 * @brief Implementation unit for Mila.Bindings.Runtime: reads the device registry and the library version.
 */

module;

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

module Mila.Bindings.Runtime;

import Mila;

namespace Mila::Bindings
{
    using namespace Mila::Dnn::Compute;

    std::vector<CudaDeviceInfo> cudaDevices()
    {
        std::vector<CudaDeviceInfo> devices;
        auto& registry = DeviceRegistry::instance();

        for ( const auto& id : registry.listDeviceIds() )
        {
            if ( id.type != DeviceType::Cuda )
            {
                continue;
            }

            const auto cuda = std::dynamic_pointer_cast<CudaDevice>( registry.getDevice( id ) );

            if ( !cuda )
            {
                continue;
            }

            const auto& properties = cuda->getProperties();
            const auto [ major, minor ] = properties.getComputeCapability();

            devices.push_back( CudaDeviceInfo{
                .index = id.index,
                .name = properties.getName(),
                .compute_capability_major = major,
                .compute_capability_minor = minor,
                .total_memory_bytes = properties.totalGlobalMem,
                .pci_domain = properties.pciDomainID,
                .pci_bus = properties.pciBusID,
                .pci_device = properties.pciDeviceID,
            } );
        }

        std::ranges::sort( devices, {}, &CudaDeviceInfo::index );

        return devices;
    }

    std::string version()
    {
        return Mila::getAPIVersion().toString();
    }
}
