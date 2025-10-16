/**
 * @file MetalExecutionContext.ixx
 * @brief Metal-specific execution context specialization.
 */

    module;
#include <memory>
#include <string>
#include <stdexcept>
#include <mutex>

export module Compute.MetalExecutionContext;

import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.ComputeDevice;
import Compute.MetalDevice;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Metal execution context specialization.
     *
     * Provides the ExecutionContext<DeviceType::Metal> specialization.
     * This class mirrors the shape and usage patterns of the CUDA specialization
     * but currently provides placeholders where platform-specific Metal
     * implementation would be required.
     *
     * The implementation uses RAII, is non-copyable/non-movable and defers
     * platform-specific behavior (command queues, synchronization) until Metal
     * support is implemented.
     */
    export template<>
        class ExecutionContext<DeviceType::Metal> : public IExecutionContext
    {
    public:
        /**
         * @brief Constructs Metal execution context for a device id.
         *
         * Creates a MetalDevice instance and initializes execution resources.
         *
         * @param device_id Metal device id (0-based)
         * @throws std::invalid_argument If device_id is invalid (handled by MetalDevice)
         */
        explicit ExecutionContext( int device_id )
            : IExecutionContext( DeviceType::Metal ), device_( std::make_shared<MetalDevice>( device_id ) ) {

            initializeExecutionResources();
        }

        /**
         * @brief Constructs Metal execution context from existing device.
         *
         * Shares device ownership and initializes execution resources.
         *
         * @param device Shared ComputeDevice instance; must be a Metal device
         * @throws std::invalid_argument If device is null or not Metal
         */
        explicit ExecutionContext( std::shared_ptr<ComputeDevice> device )
            : IExecutionContext( DeviceType::Metal ), device_( validateDevice( device ) ) {

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
         * @brief Synchronizes the execution queue.
         *
         * Currently this implementation is a placeholder: if a real Metal
         * command queue is available and Metal support is implemented this
         * method should wait for completion of submitted commands.
         *
         * @throws std::runtime_error If synchronization cannot be performed
         */
        void synchronize() {
            // No-op placeholder: Metal synchronization is not implemented yet.
            // If future implementation creates a platform-specific command queue,
            // replace this with the appropriate synchronization call.
            if (!queue_created_)
            {
                // Nothing to synchronize
                return;
            }

            // If Metal support is added, perform actual synchronization here.
            throw std::runtime_error( "Metal synchronization not implemented" );
        }

        /**
         * @brief Gets the associated device.
         */
        std::shared_ptr<ComputeDevice> getDevice() const {
            return device_;
        }

        /**
         * @brief Gets the device type (always Metal).
         */
        static constexpr DeviceType getDeviceType() {
            return DeviceType::Metal;
        }

        /**
         * @brief Gets the device name (e.g., "Metal:0" or device-provided name).
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
         * @brief Checks if this is a Metal device (always true).
         */
        static constexpr bool isMetalDevice() {
            return true;
        }

        /**
         * @brief Checks if this is a CPU device (always false).
         */
        static constexpr bool isCpuDevice() {
            return false;
        }

        /**
         * @brief Returns an opaque pointer representing the Metal command queue.
         *
         * This is a placeholder. Real Metal support should return a typed
         * command queue handle (Objective-C id<MTLCommandQueue> or wrapper).
         *
         * @throws std::runtime_error If Metal command queue is not available
         */
        void* getCommandQueue() const {
            std::lock_guard<std::mutex> lock( handle_mutex_ );

            if (!queue_created_)
            {
                throw std::runtime_error( "Metal command queue is not available; Metal support not implemented" );
            }

            return command_queue_;
        }

    private:
        std::shared_ptr<ComputeDevice> device_;
        mutable void* command_queue_{ nullptr }; // opaque placeholder for platform queue
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

            if (device->getDeviceType() != DeviceType::Metal)
            {
                throw std::invalid_argument(
                    "Metal ExecutionContext requires Metal device, got: " +
                    std::string( deviceToString( device->getDeviceType() ) )
                );
            }

            return device;
        }

        /**
         * @brief Initializes execution resources.
         *
         * Currently leaves platform-specific resources uncreated. In a full
         * implementation this would create the Metal device, command queue and
         * any per-context resources.
         */
        void initializeExecutionResources() {
            // Placeholder: real Metal initialization would go here.
            // For now, leave command_queue_ null to indicate unimplemented.
            queue_created_ = false;
        }

        /**
         * @brief Releases all resources.
         *
         * Called by destructor - must not throw.
         */
        void releaseResources() noexcept {
            // If a real Metal command queue was created it should be released here.
            std::lock_guard<std::mutex> lock( handle_mutex_ );
            command_queue_ = nullptr;
            queue_created_ = false;
        }
    };
}