module;
#include <cassert>
#include <memory>
#include <string>
#include <stdexcept>
#include <format>

export module Compute.ExecutionContextFactory;

import Compute.ExecutionContext;
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
    export std::unique_ptr<IExecutionContext> createExecutionContext( DeviceId device_id )
    {
        switch ( device_id.type )
        {
            case DeviceType::Cpu:
                return std::make_unique<ExecutionContext<DeviceType::Cpu>>( device_id );

            #ifdef MILA_HAS_CUDA
            case DeviceType::Cuda:
                return std::make_unique<ExecutionContext<DeviceType::Cuda>>( device_id );
            #endif

            default:
                throw std::invalid_argument(
                    std::format( "Unsupported device type: {}", deviceTypeToString( device_id.type ) )
                );
        }
    }

    /**
     * @brief Query if a backend is available at runtime.
     */
    //export constexpr bool hasBackend( DeviceType type ) noexcept
    //{
    //    switch ( type )
    //    {
    //        case DeviceType::Cpu:
    //            return true;  // Always available
    //        case DeviceType::Cuda:
    //        #ifdef MILA_HAS_CUDA
    //            return true;
    //        #else
    //            return false;
    //        #endif
    //        case DeviceType::Metal:
    //        #ifdef MILA_HAS_METAL
    //            return true;
    //        #else
    //            return false;
    //        #endif
    //        case DeviceType::Rocm:
    //        #ifdef MILA_HAS_ROCM
    //            return true;
    //        #else
    //            return false;
    //        #endif
    //        default:
    //            return false;
    //    }
    //}
}