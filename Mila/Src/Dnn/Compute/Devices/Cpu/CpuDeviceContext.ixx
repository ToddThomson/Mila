/**
 * @file CpuDeviceContext.ixx
 * @brief CPU-specific device context implementation.
 */

module;
#include <memory>
#include <string>
#include <stdexcept>

export module Compute.CpuDeviceContext;

import Compute.DeviceContext;
import Compute.ComputeDevice;
import Compute.DeviceRegistry;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU-specific device context implementation.
     *
     * Provides a device context abstraction for CPU-based compute operations.
     * Represents the system CPU as a unified compute device for tensor operations
     * and neural network computations. Unlike GPU contexts, CPU contexts require
     * no device switching or synchronization operations.
     *
     * The CPU device context maintains consistency with the device abstraction
     * layer while providing appropriate no-op implementations for operations
     * that are specific to discrete compute devices like GPUs.
     */
    export class CpuDeviceContext : public DeviceContext {
    public:

        /**
         * @brief Default constructor for CPU device context.
         *
         * Creates a CPU device context for the system's CPU. Since there is
         * only one logical CPU device per system, no device identifier is required.
         * The context automatically registers with the device registry and validates
         * successful CPU device creation.
         *
         * @throws std::runtime_error If CPU device creation fails
         */
        CpuDeviceContext() {
            device_ = DeviceRegistry::instance().createDevice( "CPU" );

            if (!device_) {
                throw std::runtime_error( "Failed to create CPU device" );
            }
        }

        /**
         * @brief Returns the device type for this context.
         * @return Always returns DeviceType::Cpu
         */
        DeviceType getDeviceType() const override {
            return DeviceType::Cpu;
        }

        /**
         * @brief Returns the device name for this context.
         * @return Always returns "CPU"
         */
        std::string getDeviceName() const override {
            return "CPU";
        }

        /**
         * @brief Returns the device identifier.
         * @return Always returns -1 to indicate no discrete device ID (CPU is not enumerated like GPUs)
         */
        int getDeviceId() const override {
            return -1; // CPU doesn't have device ID
        }

        /**
         * @brief Makes this device context current.
         * No-op for CPU since there's no concept of "current CPU device".
         */
        void makeCurrent() override {
            // No-op for CPU
        }

        /**
         * @brief Returns the underlying compute device.
         * @return Shared pointer to the CPU compute device instance
         */
        std::shared_ptr<ComputeDevice> getDevice() const override {
            return device_;
        }

    private:
        std::shared_ptr<ComputeDevice> device_; ///< CPU compute device instance
    };
}