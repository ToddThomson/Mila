/**
 * @file ExecutionContext.ixx
 * @brief Abstract execution context framework for compute operations and stream management.
 *
 * ExecutionContext provides the interface for managing execution streams, synchronization,
 * and compute library handles across different hardware platforms. This is separate from
 * DeviceContext, which handles device identification and activation for memory allocation.
 *
 * Responsibilities:
 * - DeviceContext: Device selection and activation (used for memory allocation)
 * - ExecutionContext: Stream management and synchronization (used for compute operations)
 */

module;
#include <memory>
#include <string>

export module Compute.ExecutionContext;

import Compute.DeviceContext;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Abstract base class for execution-specific contexts.
     *
     * ExecutionContext manages the resources needed for executing operations:
     * streams for asynchronous execution, library handles (cuBLAS, cuDNN),
     * and synchronization primitives. Multiple ExecutionContexts can share
     * the same DeviceContext, allowing different modules/graphs to have
     * independent execution streams on the same device.
     *
     * Design rationale:
     * - Tensors hold device_name strings and use DeviceContext transiently for allocation
     * - Modules own ExecutionContexts and pass them to TensorOps for compute operations
     * - This separation ensures proper stream management in multi-module scenarios
     */
    export class ExecutionContext {
    public:
        /**
         * @brief Virtual destructor for proper cleanup of derived classes.
         */
        virtual ~ExecutionContext() = default;

        /**
         * @brief Copy constructor (deleted).
         * @note ExecutionContext is not copyable due to unique resource ownership.
         */
        ExecutionContext( const ExecutionContext& ) = delete;

        /**
         * @brief Copy assignment operator (deleted).
         */
        ExecutionContext& operator=( const ExecutionContext& ) = delete;

        /**
         * @brief Move constructor.
         */
        ExecutionContext( ExecutionContext&& other ) noexcept = default;

        /**
         * @brief Move assignment operator.
         */
        ExecutionContext& operator=( ExecutionContext&& other ) noexcept = default;

        // ====================================================================
        // Pure Virtual Interface - Must be implemented by derived classes
        // ====================================================================

        /**
         * @brief Synchronizes execution, waiting for all queued operations to complete.
         *
         * Blocks the calling thread until all operations submitted to this execution
         * context have completed. For CUDA, this synchronizes the associated stream.
         * For CPU, this is typically a no-op.
         */
        virtual void synchronize() = 0;

        /**
         * @brief Gets the underlying device context.
         *
         * Returns the DeviceContext that this execution context is bound to.
         * The device context is used to query device properties and ensure
         * operations are executed on the correct device.
         *
         * @return Shared pointer to the associated device context
         */
        virtual std::shared_ptr<DeviceContext> getDeviceContext() const = 0;

        // ====================================================================
        // Common Interface with Default Implementations
        // ====================================================================

        /**
         * @brief Gets the device type for this execution context.
         * @return DeviceType enumeration value
         */
        DeviceType getDeviceType() const {
            return getDeviceContext()->getDeviceType();
        }

        /**
         * @brief Gets the device name (e.g., "CPU", "CUDA:0").
         * @return String identifier for the device
         */
        std::string getDeviceName() const {
            return getDeviceContext()->getDeviceName();
        }

        /**
         * @brief Gets the device ID (-1 for devices without numbering).
         * @return Device ID or -1 if not applicable
         */
        int getDeviceId() const {
            return getDeviceContext()->getDeviceId();
        }

        /**
         * @brief Checks if this execution context is for a CUDA device.
         */
        bool isCudaDevice() const {
            return getDeviceType() == DeviceType::Cuda;
        }

        /**
         * @brief Checks if this execution context is for a CPU device.
         */
        bool isCpuDevice() const {
            return getDeviceType() == DeviceType::Cpu;
        }

        /**
         * @brief Factory method to create execution context from device context.
         *
         * Creates an appropriate execution context implementation based on the
         * device type. Each execution context owns its own stream and library
         * handles, allowing independent execution even when sharing a device.
         *
         * @param device_context Device context to bind this execution context to
         * @return Shared pointer to appropriate execution context implementation
         * @throws std::runtime_error If device type is unsupported or creation fails
         *
         * @note Multiple execution contexts can share the same device context
         * @note Each execution context has independent streams for asynchronous execution
         */
        static std::shared_ptr<ExecutionContext> create(
            std::shared_ptr<DeviceContext> device_context
        );

        /**
         * @brief Factory method to create execution context from device name.
         *
         * Convenience method that creates both device context and execution context
         * from a device name string. Equivalent to:
         * create(DeviceContext::create(device_name))
         *
         * @param device_name Device identifier string (e.g., "CPU", "CUDA:0")
         * @return Shared pointer to appropriate execution context implementation
         * @throws std::runtime_error If device name is invalid or creation fails
         */
        static std::shared_ptr<ExecutionContext> create( const std::string& device_name );

    protected:
        /**
         * @brief Protected default constructor for derived classes.
         */
        ExecutionContext() = default;
    };
}