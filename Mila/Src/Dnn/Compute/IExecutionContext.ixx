/**
 * @file IExecutionContext.ixx
 * @brief Minimal type-erased execution context interface.
 */

export module Compute.IExecutionContext;

import Compute.DeviceId;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Type-erased execution context interface.
     *
     * Provides a minimal virtual interface for execution contexts. Specializations
     * (CPU, CUDA, Metal, ROCm) inherit from this to enable polymorphic usage when
     * the device type is not known at compile time.
     *
     * For performance-critical code where the device type is known statically,
     * use the templated ExecutionContext<TDeviceType> directly or cast_context<>() to avoid
     * runtime overhead.
     */
    export class IExecutionContext
    {
    public:
        virtual ~IExecutionContext() = default;

        /**
         * @brief Get the device identifier.
         *
         * @return DeviceId Device identifier (type + index).
         */
        [[nodiscard]] virtual DeviceId getDeviceId() const noexcept = 0;

        /**
         * @brief Synchronize all pending operations.
         *
         * Blocks until all operations submitted to this context complete.
         * For CPU contexts, this is typically a no-op.
         */
        virtual void synchronize() = 0;

    protected:
        IExecutionContext() = default;
    };
}