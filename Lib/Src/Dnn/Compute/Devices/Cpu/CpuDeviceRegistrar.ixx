/**
 * @file CpuDeviceRegistrar.ixx
 * @brief Registers the host's CPU device with the DeviceRegistry.
 */

export module Compute.CpuDeviceRegistrar;

import std;
import Compute.CpuDevice;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceRegistry;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU device plugin for device-agnostic registration.
     *
     * Encapsulates CPU-specific logic for device registration,
     * providing a clean static interface.
     */
    export class CpuDeviceRegistrar
    {
    public:

        /**
         * @brief Register CPU support with the DeviceRegistry.
         *
         * Registers the CPU device with the global DeviceRegistry.
         */
        static void registerDevices()
        {
            auto& registry = DeviceRegistry::instance();

            DeviceId device_id = Device::Cpu();

            registry.registerDevice( device_id, []( DeviceConstructionKey key ) -> std::shared_ptr<Device> {
                return std::make_shared<CpuDevice>( key );
            } );
        }
    };
}
