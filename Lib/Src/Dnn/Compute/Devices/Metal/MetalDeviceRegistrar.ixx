/**
 * @file MetalDeviceRegistrar.ixx
 * @brief Registers a Mac's Metal device with the DeviceRegistry.
 */

export module Compute.MetalDeviceRegistrar;

import std;
import Compute.Device;
import Compute.DeviceRegistry;
import Compute.MetalDevice;

namespace Mila::Dnn::Compute
{
    export class MetalDeviceRegistrar
    {
    public:

        /// A Mac has one Metal device. A Mac below Mila's floor registers too, and is refused when the device is first asked for.
        static void registerDevices()
        {
            if ( !MetalDevice::isPresent() )
            {
                return;
            }

            DeviceRegistry::instance().registerDevice( Device::Metal( 0 ), []( DeviceConstructionKey key ) -> std::shared_ptr<Device>
                {
                    return std::make_shared<MetalDevice>( key );
                } );
        }
    };
}
