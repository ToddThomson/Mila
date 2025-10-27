/**
 * @file OpenCLExecutionContext.ixx
 * @brief OpenCL-specific execution context specialization.
 */

    module;
#include <memory>
#include <string>
#include <stdexcept>
#include <mutex>

export module Compute.OpenCLExecutionContext;

import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.ComputeDevice;
import Compute.OpenCLDevice;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief OpenCL execution context specialization.
     *
     * Mirrors the API surface of other ExecutionContext specializations
     * (e.g., CUDA) while leaving OpenCL-specific runtime calls as placeholders.
     */
    export template<>
        class ExecutionContext<DeviceType::OpenCL> : public IExecutionContext
    {
    public:
        /**
         * @brief Construct OpenCL execution context for a device id.
         *
         * @param device_id OpenCL device id (0-based)
         * @throws std::invalid_argument If device_id is invalid (handled by OpenCLDevice)
         */
        explicit ExecutionContext( int device_id )
            : IExecutionContext( DeviceType::OpenCL ), device_( std::make_shared<OpenCLDevice>( device_id ) ) {

            initializeExecutionResources();
        }

        /**
         * @brief Construct OpenCL execution context from existing device.
         *
         * @param device Shared ComputeDevice instance; must be an OpenCL device
         * @throws std::invalid_argument If device is null or not OpenCL
         */
        explicit ExecutionContext( std::shared_ptr<ComputeDevice> device )
            : IExecutionContext( DeviceType::OpenCL ), device_( validateDevice( device ) ) {

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
         * @brief Synchronizes the OpenCL command queue.
         *
         * Currently this is a placeholder. Replace with real OpenCL clFinish
         * or equivalent once platform support is implemented.
         *
         * @throws std::runtime_error If synchronization cannot be performed
         */
        void synchronize() {
            if (!queue_created_)
            {
                // Nothing to synchronize
                return;
            }

            // Actual OpenCL synchronization is not implemented yet.
            throw std::runtime_error( "OpenCL synchronization not implemented" );
        }

        /**
         * @brief Gets the associated device.
         */
        std::shared_ptr<ComputeDevice> getDevice() const {
            return device_;
        }

        /**
         * @brief Gets the device type (always OpenCL).
         */
        static constexpr DeviceType getDeviceType() {
            return DeviceType::OpenCL;
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
         * @brief Checks if this is an OpenCL device (always true).
         */
        static constexpr bool isOpenCLDevice() {
            return true;
        }

        /**
         * @brief Checks if this is a CPU device (always false).
         */
        static constexpr bool isCpuDevice() {
            return false;
        }

        /**
         * @brief Returns an opaque pointer representing the OpenCL command queue.
         *
         * This is a placeholder. Real OpenCL support should return a typed
         * cl_command_queue or a wrapper type.
         *
         * @throws std::runtime_error If OpenCL command queue is not available
         */
        void* getCommandQueue() const {
            std::lock_guard<std::mutex> lock( handle_mutex_ );

            if (!queue_created_)
            {
                throw std::runtime_error( "OpenCL command queue is not available; OpenCL support not implemented" );
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

            if (device->getDeviceType() != DeviceType::OpenCL)
            {
                throw std::invalid_argument(
                    "OpenCL ExecutionContext requires OpenCL device, got: " +
                    std::string( deviceTypeToString( device->getDeviceType() ) )
                );
            }

            return device;
        }

        /**
         * @brief Initializes execution resources.
         *
         * Placeholder: in a full implementation this would create OpenCL
         * contexts and command queues.
         */
        void initializeExecutionResources() {
            // Placeholder: real OpenCL initialization would go here.
            queue_created_ = false;
        }

        /**
         * @brief Releases all resources.
         *
         * Called by destructor - must not throw.
         */
        void releaseResources() noexcept {
            std::lock_guard<std::mutex> lock( handle_mutex_ );
            command_queue_ = nullptr;
            queue_created_ = false;
        }
    };
}