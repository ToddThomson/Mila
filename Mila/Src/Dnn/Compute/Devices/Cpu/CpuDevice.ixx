/**
 * @file CpuDevice.ixx
 * @brief Implementation of CPU-based compute device for the Mila framework.
 */

module;
#include <string>
#include <memory>
#include <thread>

export module Compute.CpuDevice;

import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceRegistry;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Class representing a CPU compute device.
     *
     * Provides an interface to interact with CPU compute resources.
     * CPU is treated as a single logical device without numeric indexing
     * (always index 0).
     *
     * Device instances are created exclusively by DeviceRegistry.
     * Users should obtain devices through DeviceRegistry::getDevice().
     */
    export class CpuDevice : public Device
    {
    public:

        /**
         * @brief Construct CPU device.
         *
         * @param key Construction key ensuring only DeviceRegistry can create instances
         */
        explicit CpuDevice( DeviceConstructionKey key )
        {
            (void)key;
        }

        /**
         * @brief Gets the device type.
         *
         * @return DeviceType The device type (Cpu).
         */
        constexpr DeviceType getDeviceType() const override
        {
            return DeviceType::Cpu;
        }

        /**
         * @brief Gets the device name.
         *
         * @return std::string The device name ("CPU:0").
         */
        std::string getDeviceName() const override
        {
            return "CPU:0";
        }

        /**
         * @brief Gets the device identifier.
         *
         * For CPU, this always returns Device::Cpu() (type=Cpu, index=0).
         *
         * @return DeviceId The CPU device identifier.
         */
        constexpr DeviceId getDeviceId() const override
        {
            return Device::Cpu();
        }

        /**
         * @brief Gets the number of logical CPU cores available.
         *
         * Returns the number of hardware threads available on the system.
         *
         * @return Number of logical CPU cores/threads available
         */
        static unsigned int getLogicalCoreCount()
        {
            unsigned int coreCount = std::thread::hardware_concurrency();

            return (coreCount > 0) ? coreCount : 1;
        }

        /**
         * @brief Gets system memory information for CPU operations.
         *
         * @return Available system memory in bytes, or 0 if detection not implemented
         */
        static size_t getAvailableMemory()
        {
            return 0;
        }
    };

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