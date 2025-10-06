/**
 * @file MetalDeviceContext.ixx
 * @brief Metal-specific device context implementation (future).
 */

module;
#include <memory>
#include <string>
#include <stdexcept>

export module Compute.MetalDeviceContext;

import Compute.DeviceContext;
import Compute.ComputeDevice;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Metal-specific device context (future implementation).
     *
     * Will manage Apple Metal devices and their associated command queues,
     * shader libraries, and device-specific resources.
     */
    export class MetalDeviceContext : public DeviceContext {
    public:
        explicit MetalDeviceContext(const std::string& device_name) {
            throw std::runtime_error("Metal device support not yet implemented");
        }

        DeviceType getDeviceType() const override {
            return DeviceType::Metal;
        }

        std::string getDeviceName() const override {
            return "Metal:UNIMPLEMENTED";
        }

        int getDeviceId() const override {
            return -1;
        }

        void makeCurrent() override {
            // Future implementation
        }

        std::shared_ptr<ComputeDevice> getDevice() const override {
            return nullptr;
        }

    private:
        std::shared_ptr<ComputeDevice> device_;
        // Future: Metal-specific resources
        // id<MTLDevice> metal_device_;
        // id<MTLCommandQueue> command_queue_;
        // id<MTLLibrary> default_library_;
    };
}