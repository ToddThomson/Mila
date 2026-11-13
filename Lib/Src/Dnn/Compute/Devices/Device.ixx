/**
 * @file Device.ixx
 * @brief Abstract compute device interface and device identifier factory.
 */

export module Compute.Device;

import std;
import Compute.DeviceType;
import Compute.DeviceId;
export import Compute.DeviceMemoryInfo;
export import Compute.DevicePerformance;
export import Compute.MemoryHold;

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
         * @brief The device's place in this process, as text: "CPU:0", "CUDA:1", "Metal:0".
         *
         * A coordinate, not a name for people; that is getHardwareName.
         */
        virtual std::string getDeviceName() const = 0;

        /**
         * @brief The hardware's own name, such as "NVIDIA GeForce RTX 4070" or "Apple M1 Pro".
         *
         * Empty when the device does not report one.
         */
        virtual std::string getHardwareName() const
        {
            return {};
        }

        /**
         * @brief A name for this device that stays the same between runs, which a stored setting names.
         *
         * A device's index can move from one run to the next -- CUDA's with CUDA_VISIBLE_DEVICES, and it cannot
         * tell two cards of one model apart -- so a setting saved for a device names this instead
         * (Specifications/Deployment.md 12.8). CUDA's is the UUID as nvidia-smi -L prints it,
         * "GPU-8932f937-d72c-4106-c12f-20bd9faed9f6"; a device type with one device names it by its type.
         */
        virtual std::string getStableIdentity() const = 0;

        /**
         * @brief What the device reports about its speed, raw. Empty when it reports nothing.
         */
        virtual DevicePerformance getPerformance() const
        {
            return {};
        }

        /**
         * @brief Current free and total device memory.
         *
         * Answers "will this model fit" together with a footprint from
         * Model::getRequiredMemory. Zeroed by default for devices with no distinct memory
         * pool to report; a caller seeing total_bytes == 0 should treat the figure as
         * unavailable rather than as a device with no memory.
         *
         * NOTE on Windows: WDDM oversubscribes into shared host memory rather than failing,
         * so free_bytes running out does not produce an allocation error -- it produces a
         * model that runs pathologically slowly. Callers should warn rather than refuse.
         */
        virtual DeviceMemoryInfo getMemoryInfo() const
        {
            return {};
        }

        /**
         * @brief The multiple the driver rounds a device allocation larger than 1 MiB up to.
         *
         * Zero for a device whose allocations are not rounded. See Specifications/MemoryFootprint.md 11.8.
         */
        virtual std::size_t getAllocationGranularity() const
        {
            return 0;
        }

        /**
         * @brief Keep Mila's memory on this device as firmly as `hold` says, now and for what it allocates later.
         *
         * Changed while running: an application holds firmly while it is in use and lets go when it is not
         * (Specifications/Deployment.md 12.10). Nothing to do for a device whose memory is never paged, which is the
         * default.
         */
        virtual void setMemoryHold( [[maybe_unused]] MemoryHold hold )
        {
        }

        /// The hold last set, `MemoryHold::None` until one is.
        virtual MemoryHold getMemoryHold() const
        {
            return MemoryHold::None;
        }

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

        template<DeviceType TDeviceType>
        static constexpr DeviceId getDeviceId( int index ) noexcept
        {
            return DeviceId{ TDeviceType, index };
        }
    };
}