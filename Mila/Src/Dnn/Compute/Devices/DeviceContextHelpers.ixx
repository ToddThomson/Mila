/**
 * @file DeviceContextHelpers.ixx
 * @brief Utility functions for device context validation.
 *
 * This file defines utility functions for creating and validating device contexts
 * that can be shared across different operation types.
 */

module;
#include <memory>
#include <stdexcept>
#include <type_traits>

export module Compute.DeviceContextHelpers;

import Compute.ComputeDevice;
import Compute.CudaDevice;
import Compute.DeviceContext;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Creates a device context compatible with the specified device type.
     *
     * @tparam TDevice The device type to create a context for.
     * @return std::shared_ptr<DeviceContext> A new context of the appropriate type.
     */
    export template <DeviceType TDeviceType>
    std::shared_ptr<DeviceContext> CreateCompatibleContext() {
        if constexpr ( TDeviceType == DeviceType::Cuda ) {
            return std::make_shared<DeviceContext>( "CUDA:0" );
        }
        else {
            return std::make_shared<DeviceContext>( "CPU" );
        }
    }

    /**
     * @brief Validates that the provided context is compatible with the specified device type.
     *
     * @tparam TDevice The device type to validate against.
     * @param context The context to validate.
     * @return std::shared_ptr<DeviceContext> The validated context.
     * @throws std::invalid_argument If the context is null.
     * @throws std::runtime_error If the context is incompatible with TDevice.
     */
    export template <DeviceType TDeviceType>
    std::shared_ptr<DeviceContext> ValidateContext( std::shared_ptr<DeviceContext> context ) {
        if ( !context ) {
            throw std::invalid_argument( "Device context cannot be null" );
        }

        if ( context->getDevice()->getDeviceType() != TDeviceType ) {
            throw std::runtime_error( "The provided device context is incompatible with the operation's device type." );
        }

        return context;
    }
}