/**
 * @file Device.ixx
 * @brief Abstract compute device interface and device identifier factory.
 */

module;
#include <string>

export module Compute.Device;

import Compute.DeviceType;
import Compute.DeviceId;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Abstract interface for compute device implementations.
     *
     * Defines the common interface for all compute device types (CPU, CUDA,
     * Metal, ROCm, etc.). Concrete device implementations inherit from this
     * class and are registered with DeviceRegistry for runtime discovery.
     *
     * This class also serves as the primary factory for DeviceId values via
     * static methods, providing a clean API: Device::Cuda(0), Device::Cpu().
     */
    export class Device
    {
    public:
        virtual ~Device() = default;

        /**
         * @brief Gets the device identifier.
         *
         * @return DeviceId The identifier for this device (type + index).
         */
        virtual DeviceId getDeviceId() const = 0;

        /**
         * @brief Gets the device type.
         *
         * @return DeviceType The type of this device (Cpu, Cuda, Metal, Rocm).
         */
        virtual constexpr DeviceType getDeviceType() const = 0;

        /**
         * @brief Gets the human-readable device name.
         *
         * @return std::string Device name (e.g., "CPU:0", "CUDA:0", "Metal:1").
         */
        virtual std::string getDeviceName() const = 0;

        // ====================================================================
        // Static Factory Methods - Primary API for Device Identification
        // ====================================================================

        /**
         * @brief Create CPU device identifier.
         *
         * Returns a DeviceId representing the CPU device.
         * CPU is always index 0 (single logical CPU device).
         *
         * @return DeviceId CPU device identifier.
         */
        static constexpr DeviceId Cpu() noexcept
        {
            return DeviceId{ DeviceType::Cpu, 0 };
        }

        /**
         * @brief Create CUDA device identifier.
         *
         * @param index Zero-based CUDA device index (0, 1, 2, ...).
         * @return DeviceId CUDA device identifier.
         */
        static constexpr DeviceId Cuda( int index ) noexcept
        {
            return DeviceId{ DeviceType::Cuda, index };
        }

        /**
         * @brief Create Metal device identifier.
         *
         * @param index Zero-based Metal device index.
         * @return DeviceId Metal device identifier.
         */
        static constexpr DeviceId Metal( int index ) noexcept
        {
            return DeviceId{ DeviceType::Metal, index };
        }

        /**
         * @brief Create ROCm device identifier.
         *
         * @param index Zero-based ROCm device index.
         * @return DeviceId ROCm device identifier.
         */
        static constexpr DeviceId Rocm( int index ) noexcept
        {
            return DeviceId{ DeviceType::Rocm, index };
        }
    };
}