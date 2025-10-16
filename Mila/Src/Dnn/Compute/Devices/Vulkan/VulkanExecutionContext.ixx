/**
 * @file VulkanExecutionContext.ixx
 * @brief Vulkan-specific execution context specialization.
 */

    module;
#include <memory>
#include <string>
#include <stdexcept>
#include <mutex>

export module Compute.VulkanExecutionContext;

import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.ComputeDevice;
import Compute.VulkanDevice;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Vulkan execution context specialization.
     *
     * Mirrors the API surface of other ExecutionContext specializations
     * while leaving Vulkan-specific runtime calls as placeholders.
     */
    export template<>
        class ExecutionContext<DeviceType::Vulkan> : public IExecutionContext
    {
    public:
        /**
         * @brief Construct Vulkan execution context for a device id.
         *
         * @param device_id Vulkan device id (0-based)
         * @throws std::invalid_argument If device_id is invalid (handled by VulkanDevice)
         */
        explicit ExecutionContext( int device_id )
            : IExecutionContext( DeviceType::Vulkan ), device_( std::make_shared<VulkanDevice>( device_id ) ) {

            initializeExecutionResources();
        }

        /**
         * @brief Construct Vulkan execution context from existing device.
         *
         * @param device Shared ComputeDevice instance; must be a Vulkan device
         * @throws std::invalid_argument If device is null or not Vulkan
         */
        explicit ExecutionContext( std::shared_ptr<ComputeDevice> device )
            : IExecutionContext( DeviceType::Vulkan ), device_( validateDevice( device ) ) {

            initializeExecutionResources();
        }

        /**
         * @brief Destructor with proper resource cleanup.
         */
        ~ExecutionContext() {
            releaseResources();
        }

        // Non-copyable, non-movable (owns device-specific resources)
        ExecutionContext( const ExecutionContext& ) = delete;
        ExecutionContext& operator=( const ExecutionContext& ) = delete;
        ExecutionContext( ExecutionContext&& ) = delete;
        ExecutionContext& operator=( ExecutionContext&& ) = delete;

        /**
         * @brief Synchronizes the Vulkan command queue.
         *
         * Placeholder: replace with vkQueueWaitIdle or similar when Vulkan support is implemented.
         *
         * @throws std::runtime_error If synchronization cannot be performed
         */
        void synchronize() {
            if (!queue_created_)
            {
                // Nothing to synchronize
                return;
            }

            // Real synchronization is not implemented yet.
            throw std::runtime_error( "Vulkan synchronization not implemented" );
        }

        /**
         * @brief Gets the associated device.
         */
        std::shared_ptr<ComputeDevice> getDevice() const {
            return device_;
        }

        /**
         * @brief Gets the device type (always Vulkan).
         */
        static constexpr DeviceType getDeviceType() {
            return DeviceType::Vulkan;
        }

        /**
         * @brief Gets the device name.
         */
        std::string getDeviceName() const {
            return device_->getDeviceName();
        }

        /**
         * @brief Gets the device id.
         */
        int getDeviceId() const {
            return device_->getDeviceId();
        }

        /**
         * @brief Checks if this is a Vulkan device (always true).
         */
        static constexpr bool isVulkanDevice() {
            return true;
        }

        /**
         * @brief Checks if this is a CPU device (always false).
         */
        static constexpr bool isCpuDevice() {
            return false;
        }

        /**
         * @brief Returns an opaque pointer representing the Vulkan queue.
         *
         * Placeholder: real Vulkan support should return a typed VkQueue or wrapper.
         *
         * @throws std::runtime_error If Vulkan queue is not available
         */
        void* getQueue() const {
            std::lock_guard<std::mutex> lock( handle_mutex_ );

            if (!queue_created_)
            {
                throw std::runtime_error( "Vulkan queue is not available; Vulkan support not implemented" );
            }

            return queue_;
        }

    private:
        std::shared_ptr<ComputeDevice> device_;
        mutable void* queue_{ nullptr }; // opaque placeholder for platform queue (VkQueue)
        bool queue_created_{ false };
        mutable std::mutex handle_mutex_;

        /**
         * @brief Validates device for construction.
         */
        static std::shared_ptr<ComputeDevice> validateDevice(
            std::shared_ptr<ComputeDevice> device ) {

            if (!device)
            {
                throw std::invalid_argument( "Device cannot be null" );
            }

            if (device->getDeviceType() != DeviceType::Vulkan)
            {
                throw std::invalid_argument(
                    "Vulkan ExecutionContext requires Vulkan device, got: " +
                    std::string( deviceToString( device->getDeviceType() ) )
                );
            }

            return device;
        }

        /**
         * @brief Initializes execution resources.
         *
         * Placeholder: in a full implementation this would create Vulkan logical device,
         * command pools and queues.
         */
        void initializeExecutionResources() {
            // Placeholder: real Vulkan initialization would go here.
            queue_created_ = false;
        }

        /**
         * @brief Releases all resources.
         *
         * Called by destructor - must not throw.
         */
        void releaseResources() noexcept {
            std::lock_guard<std::mutex> lock( handle_mutex_ );
            queue_ = nullptr;
            queue_created_ = false;
        }
    };
}