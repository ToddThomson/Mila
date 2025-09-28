/**
 * @file DeviceContext.ixx
 * @brief Abstract device context framework for multi-device compute operations.
 *
 * This file provides the DeviceContext base class that defines the unified interface
 * for managing compute operations across heterogeneous hardware platforms. Each
 * supported device type has its own specialized implementation in separate files.
 */

module;
#include <memory>
#include <string>
#include <stdexcept>
#include <algorithm>

export module Compute.DeviceContext;

import Compute.ComputeDevice;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Abstract base class for device-specific compute contexts.
     *
     * DeviceContext provides a unified interface for managing compute devices and their
     * associated resources across different hardware platforms. Each supported device type
     * (CPU, CUDA, Metal, OpenCL, Vulkan) has its own specialized implementation that handles
     * device-specific resource management and operations.
     */
    export class DeviceContext {
    public:
        /**
         * @brief Virtual destructor for proper cleanup of derived classes.
         */
        virtual ~DeviceContext() = default;

        /**
         * @brief Copy constructor (deleted).
         * @note DeviceContext is not copyable due to unique resource ownership.
         */
        DeviceContext(const DeviceContext&) = delete;

        /**
         * @brief Copy assignment operator (deleted).
         * @note DeviceContext is not copyable due to unique resource ownership.
         */
        DeviceContext& operator=(const DeviceContext&) = delete;

        /**
         * @brief Move constructor.
         */
        DeviceContext(DeviceContext&& other) noexcept = default;

        /**
         * @brief Move assignment operator.
         */
        DeviceContext& operator=(DeviceContext&& other) noexcept = default;

        // ====================================================================
        // Pure Virtual Interface - Must be implemented by derived classes
        // ====================================================================

        /**
         * @brief Gets the device type for this context.
         * @return DeviceType enumeration value for this device context.
         */
        virtual DeviceType getDeviceType() const = 0;

        /**
         * @brief Gets the device name (e.g., "CPU", "CUDA:0", "Metal:0").
         * @return String identifier for this device.
         */
        virtual std::string getDeviceName() const = 0;

        /**
         * @brief Gets the device ID for numbered devices (-1 for devices without numbering).
         * @return Device ID or -1 if not applicable.
         */
        virtual int getDeviceId() const = 0;

        /**
         * @brief Sets this device as active in the current thread.
         * @throws std::runtime_error If device activation fails.
         */
        virtual void makeCurrent() = 0;

        /**
         * @brief Synchronizes the device, waiting for all operations to complete.
         */
        virtual void synchronize() = 0;

        /**
         * @brief Gets the compute device associated with this context.
         * @return Shared pointer to the compute device.
         */
        virtual std::shared_ptr<ComputeDevice> getDevice() const = 0;

        // ====================================================================
        // Common Interface with Default Implementations
        // ====================================================================

        /**
         * @brief Checks if the current device is of a specific type.
         * @param type The device type to check against.
         * @return True if the device matches the specified type.
         */
        bool isDeviceType(DeviceType type) const {
            return getDeviceType() == type;
        }

        /**
         * @brief Checks if the current device is a CPU device.
         */
        bool isCpuDevice() const {
            return isDeviceType(DeviceType::Cpu);
        }

        /**
         * @brief Checks if the current device is a CUDA device.
         */
        bool isCudaDevice() const {
            return isDeviceType(DeviceType::Cuda);
        }

        /**
         * @brief Checks if the current device is a Metal device.
         */
        bool isMetalDevice() const {
            return isDeviceType(DeviceType::Metal);
        }

        /**
         * @brief Checks if the current device is an OpenCL device.
         */
        bool isOpenCLDevice() const {
            return isDeviceType(DeviceType::OpenCL);
        }

        /**
         * @brief Checks if the current device is a Vulkan device.
         */
        bool isVulkanDevice() const {
            return isDeviceType(DeviceType::Vulkan);
        }

        /**
         * @brief Factory method to create device context from device name.
         * @param device_name Device identifier string (e.g., "cpu", "cuda:0").
         * @return Shared pointer to appropriate device context implementation.
         * @throws std::runtime_error If device name is invalid or device creation fails.
         */
        static std::shared_ptr<DeviceContext> create(const std::string& device_name);

    protected:
        /**
         * @brief Protected default constructor for derived classes.
         */
        DeviceContext() = default;
    };
}