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

export import Compute.ExecutionContextBase;

export import :Cpu;

#ifdef MILA_HAS_CUDA
export import :Cuda;
#endif

// Conditionally export Metal backend
#ifdef MILA_HAS_METAL
export import :Metal;
#endif

// Conditionally export ROCm backend
#ifdef MILA_HAS_ROCM
export import :Rocm;
#endif

import Compute.IExecutionContext;
import Compute.ComputeDevice;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Safe cast from IExecutionContext to concrete ExecutionContext<Device>.
     *
     * Performs a debug assertion to verify the device type matches, then
     * does a zero-cost static_cast in release builds.
     *
     * @tparam TDeviceType The device type to cast to
     * @param ctx The type-erased context pointer
     * @return Pointer to the concrete context, or nullptr if ctx is nullptr
     */
    export template<DeviceType TDeviceType>
        [[nodiscard]] ExecutionContext<TDeviceType>* cast_context( IExecutionContext* ctx ) noexcept
    {
        if ( !ctx )
            return nullptr;

        assert( ctx->getDeviceType() == TDeviceType && "Device type mismatch in context cast" );
        return static_cast<ExecutionContext<TDeviceType>*>(ctx);
    }

    /**
     * @brief Safe cast from IExecutionContext to concrete ExecutionContext<Device> (const version).
     */
    export template<DeviceType TDeviceType>
        [[nodiscard]] const ExecutionContext<TDeviceType>* cast_context( const IExecutionContext* ctx ) noexcept
    {
        if ( !ctx )
            return nullptr;

        assert( ctx->getDeviceType() == TDeviceType && "Device type mismatch in context cast" );
        return static_cast<const ExecutionContext<TDeviceType>*>(ctx);
    }

    /**
     * @brief Query if a backend is available at runtime.
     */
    export constexpr bool hasBackend( DeviceType type ) noexcept
    {
        switch ( type )
        {
            case DeviceType::Cpu:
                return true;  // Always available
            case DeviceType::Cuda:
#ifdef MILA_HAS_CUDA
                return true;
#else
                return false;
#endif
            case DeviceType::Metal:
#ifdef MILA_HAS_METAL
                return true;
#else
                return false;
#endif
            case DeviceType::Rocm:
#ifdef MILA_HAS_ROCM
                return true;
#else
                return false;
#endif
            default:
                return false;
        }
    }

    // ====================================================================
    // Type Aliases for Common Device Types
    // ====================================================================

    //export using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;
    //export using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
}