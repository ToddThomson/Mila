/**
 * @file ExecutionContext.ixx
 * @brief Templated execution context framework for compute operations and stream management.
 *
 * ExecutionContext provides the interface for managing execution streams, synchronization,
 * and compute library handles across different hardware platforms. The template parameter
 * provides compile-time device type safety and eliminates runtime dispatch overhead.
 */

module;
#include <cassert>
#include <memory>
#include <string>

export module Compute.ExecutionContext;

import Compute.IExecutionContext;
import Compute.ComputeDevice;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Templated execution context for device-specific operations.
     *
     * ExecutionContext manages the resources needed for executing operations:
     * streams for asynchronous execution, library handles (cuBLAS, cuDNN),
     * and synchronization primitives. The template parameter provides compile-time
     * device type checking and enables device-specific optimizations.
     *
     * Design rationale:
     * - Template parameter eliminates runtime type checking overhead
     * - Each device type has a specialized implementation
     * - Modules are templated on device type, ensuring type safety throughout
     * - No virtual function overhead for device-specific operations
     *
     * @tparam TDeviceType The device type (Cpu, Cuda, Metal, etc.)
     */
    export template<DeviceType TDeviceType>
    class ExecutionContext : public IExecutionContext {
    public:

        ExecutionContext() : IExecutionContext( TDeviceType ) {}

        /**
         * @brief Destructor for proper cleanup of derived classes.
         */
        ~ExecutionContext() = default;

        /**
         * @brief Copy constructor (deleted).
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
        // Interface - Implemented by specializations
        // ====================================================================

        /**
         * @brief Synchronizes execution, waiting for all queued operations to complete.
         *
         * Blocks the calling thread until all operations submitted to this execution
         * context have completed. Implementation is device-specific.
         */
        void synchronize();

        /**
         * @brief Gets the underlying device.
         * @return Shared pointer to the associated device
         */
        std::shared_ptr<ComputeDevice> getDevice() const;

        /**
         * @brief Gets the device type for this execution context.
         * @return DeviceType enumeration value (known at compile time)
         */
        static constexpr DeviceType getDeviceType() noexcept {
            return TDeviceType;
        }

        /**
         * @brief Gets the device name (e.g., "CPU", "CUDA:0").
         * @return String identifier for the device
         */
        std::string getDeviceName() const {
            return getDevice()->getDeviceName();
        }

        /**
         * @brief Gets the device ID (-1 for devices without numbering).
         * @return Device ID or -1 if not applicable
         */
        int getDeviceId() const {
            return getDevice()->getDeviceId();
        }

        /**
         * @brief Checks if this execution context is for a CUDA device.
         */
        static constexpr bool isCudaDevice() {
            return TDeviceType == DeviceType::Cuda;
        }

        /**
         * @brief Checks if this execution context is for a CPU device.
         */
        static constexpr bool isCpuDevice() {
            return TDeviceType == DeviceType::Cpu;
        }

    //protected:
    //    /**
    //     * @brief Protected default constructor for specialized implementations.
    //     */
    //    ExecutionContext() = default;
    };

    /**
     * @brief Safe cast from IExecutionContext to concrete ExecutionContext<Device>.
     *
     * Performs a debug assertion to verify the device type matches, then
     * does a zero-cost static_cast in release builds.
     *
     * @tparam Device The device trait type to cast to
     * @param ctx The type-erased context pointer
     * @return Pointer to the concrete context, or nullptr if ctx is nullptr
     */
    export template<DeviceType TDeviceType>
        [[nodiscard]] ExecutionContext<TDeviceType>* cast_context( IExecutionContext* ctx ) noexcept {
        if (!ctx) 
            return nullptr;

        assert( ctx->getDeviceType() == TDeviceType && "Device type mismatch in context cast" );

        return static_cast<ExecutionContext<TDeviceType>*>(ctx);
    }

    /**
     * @brief Safe cast from IExecutionContext to concrete ExecutionContext<Device> (const version).
     */
    export template<DeviceType TDeviceType>
        [[nodiscard]] const ExecutionContext<TDeviceType>* cast_context( const IExecutionContext* ctx ) noexcept {
        if (!ctx) 
            return nullptr;
        
        assert( ctx->getDeviceType() == TDeviceType && "Device type mismatch in context cast" );
        
        return static_cast<const ExecutionContext<TDeviceType>*>(ctx);
    }

    // ====================================================================
    // Type Aliases for Common Device Types
    // ====================================================================

    export using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;
    export using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
}