/**
 * @file ExecutionContext.ixx
 * @brief Templated execution context framework for compute operations and stream management.
 *
 * ExecutionContext provides the interface for managing execution streams, synchronization,
 * and compute library handles across different hardware platforms. The template parameter
 * provides compile-time device type safety and eliminates runtime dispatch overhead.
 */

module;
#include <cuda_runtime.h>
#include <cublasLt.h>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#include <cassert>
#include <memory>
#include <string>
#include <stdexcept>
#include <format>

export module Compute.ExecutionContext;

import Compute.ExecutionContextTemplate;
export import Compute.IExecutionContext;

// REVIEW: Only import the specializations
import :Cpu;

#ifdef MILA_HAS_CUDA
import :Cuda;
#endif

// Conditionally export Metal backend
#ifdef MILA_HAS_METAL
//import :Metal;
#endif

// Conditionally export ROCm backend
#ifdef MILA_HAS_ROCM
//import :Rocm;
#endif

import Compute.DeviceType;
import Compute.DeviceId;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Create execution context for specified device.
     *
     * Factory function returning type-erased IExecutionContext.
     * Hides device-specific implementation details from users.
     */

     // REVIEW: Should be templated on TDeviceType to avoid runtime switch?
    export std::shared_ptr<IExecutionContext> createExecutionContext( DeviceId device_id )
    {
        switch ( device_id.type )
        {
            case DeviceType::Cpu:
                return std::make_shared<ExecutionContext<DeviceType::Cpu>>( device_id );

            #ifdef MILA_HAS_CUDA
            case DeviceType::Cuda:
                return std::make_shared<ExecutionContext<DeviceType::Cuda>>( device_id );
            #endif

            default:
                throw std::invalid_argument(
                    std::format( "Unsupported device type: {}",
                        deviceTypeToString( device_id.type ) )
                );
        }
    }

    // REVIEW: These casts are internal only. The user needs only deal with IExecutionContext pointers.

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
    /*export template<DeviceType TDeviceType>
        [[nodiscard]] ExecutionContext<TDeviceType>* cast_context( IExecutionContext* ctx ) noexcept
    {
        if ( !ctx )
            return nullptr;

        assert( ctx->getDeviceId().type == TDeviceType && "Device type mismatch in context cast" );
        return static_cast<ExecutionContext<TDeviceType>*>(ctx);
    }*/

    /**
     * @brief Safe cast from IExecutionContext to concrete ExecutionContext<Device> (const version).
     */
    /*export template<DeviceType TDeviceType>
        [[nodiscard]] const ExecutionContext<TDeviceType>* cast_context( const IExecutionContext* ctx ) noexcept
    {
        if ( !ctx )
            return nullptr;

        assert( ctx->getDeviceId().type == TDeviceType && "Device type mismatch in context cast" );
        return static_cast<const ExecutionContext<TDeviceType>*>(ctx);
    }*/

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
}