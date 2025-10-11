/**
 * @file ExecutionContext.ixx
 * @brief Templated execution context framework for compute operations and stream management.
 *
 * ExecutionContext provides the interface for managing execution streams, synchronization,
 * and compute library handles across different hardware platforms. The template parameter
 * provides compile-time device type safety and eliminates runtime dispatch overhead.
 */

module;
#include <memory>
#include <string>

export module Compute.IExecutionContext;

import Compute.ComputeDevice;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Type-erased execution context interface.
     *
     * Provides a minimal virtual interface to query runtime properties of an
     * execution context. Implementations (templated specializations) should
     * inherit from this interface to enable type-erased usage.
     */
    export class IExecutionContext
    {
    public:

        virtual ~IExecutionContext() = default;

        /**
         * @brief Get the runtime device type for this execution context.
         *
         * This is a non-virtual, zero-overhead accessor that returns the
         * device type set during construction.
         */
        [[nodiscard]] DeviceType getDeviceType() const noexcept {
            return deviceType_;
        }

    protected:
        /**
         * @brief Constructor for derived classes to set the device type.
         * @param type The device type for this context.
         */
        explicit IExecutionContext( DeviceType type ) noexcept
            : deviceType_( type ) {
        }

    private:

        DeviceType deviceType_;
    };
}