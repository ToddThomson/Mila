/**
 * @file OpenCLDeviceContext.ixx
 * @brief OpenCL-specific device context implementation (future).
 */

module;
#include <memory>
#include <string>
#include <stdexcept>

export module Compute.OpenCLDeviceContext;

import Compute.DeviceContext;
import Compute.ComputeDevice;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief OpenCL-specific device context (future implementation).
     *
     * Will manage OpenCL devices and their associated contexts, command queues,
     * and program caches for cross-platform GPU compute operations.
     */
    export class OpenCLDeviceContext : public DeviceContext {
    public:
        explicit OpenCLDeviceContext(const std::string& device_name) {
            throw std::runtime_error("OpenCL device support not yet implemented");
        }

        DeviceType getDeviceType() const override {
            return DeviceType::OpenCL;
        }

        std::string getDeviceName() const override {
            return "OpenCL:UNIMPLEMENTED";
        }

        int getDeviceId() const override {
            return -1;
        }

        void makeCurrent() override {
            // Future implementation
        }

        void synchronize() override {
            // Future implementation
        }

        std::shared_ptr<ComputeDevice> getDevice() const override {
            return nullptr;
        }

    private:
        std::shared_ptr<ComputeDevice> device_;
        // Future: OpenCL-specific resources
        // cl_context context_;
        // cl_command_queue queue_;
        // cl_device_id device_id_;
        // cl_program program_cache_;
    };
}