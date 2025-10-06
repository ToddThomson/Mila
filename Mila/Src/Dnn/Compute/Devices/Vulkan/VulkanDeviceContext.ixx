/**
 * @file VulkanDeviceContext.ixx
 * @brief Vulkan-specific device context implementation (future).
 */

module;
#include <memory>
#include <string>
#include <stdexcept>

export module Compute.VulkanDeviceContext;

import Compute.DeviceContext;
import Compute.ComputeDevice;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Vulkan-specific device context (future implementation).
     *
     * Will manage Vulkan devices and their associated command pools, queues,
     * and descriptor sets for high-performance compute operations.
     */
    export class VulkanDeviceContext : public DeviceContext {
    public:
        explicit VulkanDeviceContext(const std::string& device_name) {
            throw std::runtime_error("Vulkan device support not yet implemented");
        }

        DeviceType getDeviceType() const override {
            return DeviceType::Vulkan;
        }

        std::string getDeviceName() const override {
            return "Vulkan:UNIMPLEMENTED";
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
        // Future: Vulkan-specific resources
        // VkDevice logical_device_;
        // VkQueue compute_queue_;
        // VkCommandPool command_pool_;
        // uint32_t queue_family_index_;
    };
}