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

export import Compute.ExecutionContextTemplate;
export import Compute.IExecutionContext;

export import :Cpu;

#ifdef MILA_HAS_CUDA
export import :Cuda;
#endif

#ifdef MILA_HAS_METAL
export import :Metal;
#endif

#ifdef MILA_HAS_ROCM
export import :Rocm;
#endif

import Compute.DeviceType;
import Compute.DeviceId;

namespace Mila::Dnn::Compute
{
    /**
     * @internal
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
    [[nodiscard]] ExecutionContext<TDeviceType>* cast_context_( IExecutionContext* ctx ) noexcept
    {
        if ( !ctx )
            return nullptr;

        assert( ctx->getDeviceId().type == TDeviceType && "Device type mismatch in context cast" );
        
        return static_cast<ExecutionContext<TDeviceType>*>(ctx);
    }

    /**
     * @internal
     * @brief Safe cast from IExecutionContext to concrete ExecutionContext<Device> (const version).
     */
    export template<DeviceType TDeviceType>
    [[nodiscard]] const ExecutionContext<TDeviceType>* cast_context_( const IExecutionContext* ctx ) noexcept
    {
        if ( !ctx )
            return nullptr;

        assert( ctx->getDeviceId().type == TDeviceType && "Device type mismatch in context cast" );
        
        return static_cast<const ExecutionContext<TDeviceType>*>(ctx);
    }

    /**
     * @brief Validate and cast IExecutionContext to device-specific execution context.
     *
     * Generic helper for operation constructors. Validates that the provided context
     * matches the expected device type and casts it to the concrete type.
     *
     * @tparam TDeviceType The expected device type
     * @param context The execution context to validate
     * @param op_name Operation name for error messages
     * @return Validated and cast execution context
     * @throws std::invalid_argument if context is null or device type doesn't match
     *
     * @example
     * CudaGeluOp(IExecutionContext* context, const GeluConfig& config)
     *     : cuda_context_(validateExecutionContext<DeviceType::Cuda>(context, "CudaGeluOp"))
     *     , config_(config)
     * {}
     */
    export template<DeviceType TDeviceType>
    ExecutionContext<TDeviceType>* validateExecutionContext_(
        IExecutionContext* context,
        const std::string& op_name )
    {
        if ( !context ) {
            throw std::invalid_argument( "{} requires a non-null execution context" );
            // FIXME
            //    std::format( "{} requires a non-null execution context", op_name )
            //);
        }

        if ( context->getDeviceId().type != TDeviceType ) {
            throw std::invalid_argument( "{} requires {} execution context, got {}" );
            //    std::format( "{} requires {} execution context, got {}",
            //        op_name,
            //        deviceTypeToString( TDeviceType ),
            //        deviceTypeToString( context->getDeviceId().type ) )
            //);
        }

        return static_cast<ExecutionContext<TDeviceType>*>(context);
    }
}