/**
 * @file ExecutionContext.ixx
 * @brief Execution context framework for compute operations and stream management.
 *
 * ExecutionContext manages execution streams, synchronization, and compute library handles
 * across hardware platforms. Selecting the backend by device type at compile time gives
 * type safety and eliminates runtime dispatch overhead.
 */

module;
#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#include <cassert>
#include <string>
#include <stdexcept>
#include <format>

export module Compute.ExecutionContext;

export import Compute.IExecutionContext;

// The traits modules export no entities -- an explicit specialization cannot carry
// `export` -- so re-exporting them publishes nothing. They are re-exported rather than
// imported plainly because MSVC 14.51 will not complete a merely reachable explicit
// specialization when the dereference is dependent. The context classes below stay on
// plain imports, which is what keeps them out of `import Mila;`.
export import Compute.ExecutionContextTraits;
export import Compute.ExecutionContextTraits.Cpu;

import Compute.CpuExecutionContext;

#ifdef MILA_HAS_CUDA
export import Compute.ExecutionContextTraits.Cuda;
import Compute.CudaExecutionContext;
#endif

// Metal and Rocm have no binding here: neither backend has ever been compiled, and their
// context files declare standalone modules rather than partitions of this one.

import Compute.DeviceType;
import Compute.DeviceId;

namespace Mila::Dnn::Compute
{
    /**
     * @brief The execution context for a device type.
     *
     * Resolves to the backend's concrete context class -- CpuExecutionContext,
     * CudaExecutionContext -- each of which implements IExecutionContext and adds the
     * stream and library handles its backend needs.
     *
     * @tparam TDeviceType The device type (Cpu, Cuda, ...).
     */
    export template<DeviceType TDeviceType>
    using ExecutionContext = typename ExecutionContextTraits<TDeviceType>::type;

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
     * @internal
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
            throw std::invalid_argument( std::format( "{} requires a non-null execution context", op_name ) );
        }

        if ( context->getDeviceId().type != TDeviceType ) {
            throw std::invalid_argument(
                std::format( "{} requires {} execution context, got {}",
                    op_name,
                    deviceTypeToString( TDeviceType ),
                    deviceTypeToString( context->getDeviceId().type ) )
            );
        }

        return static_cast<ExecutionContext<TDeviceType>*>(context);
    }
}
