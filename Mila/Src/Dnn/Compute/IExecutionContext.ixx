/**
 * @file IExecutionContext.ixx
 * @brief Minimal type-erased execution context interface.
 */

module;
#include <cstddef>
#include <utility>

export module Compute.IExecutionContext;

import Compute.DeviceId;
import Compute.Observation;

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

        /**
         * @brief High-water mark of context-owned scratch device memory, in bytes.
         *
         * Scratch is allocated lazily during forward passes and grows without shrinking,
         * so no build-time contract sees it -- it is the largest identified component of
         * the gap between a reported footprint and what the driver says was consumed
         * (Specifications/MemoryFootprint.md section 6.4).
         *
         * Zero for contexts that allocate no scratch, which includes every CPU context.
         */
        [[nodiscard]] virtual std::size_t getScratchHighWaterBytes() const noexcept
        {
            return 0;
        }

        /**
         * @brief Install the observer components publish their activations to.
         *
         * The context is the transport because it is already shared parent to child across a
         * whole model tree, so an observer installed here reaches every component without
         * further plumbing. Pass an empty function to detach.
         *
         * ONE CONTEXT, ONE MODEL TREE. Every transformer creates its own context from a
         * DeviceId and no constructor accepts an existing one, so an observer installed here
         * sees exactly one model. A future overload that accepted a context -- a shared CUDA
         * stream across two models is a reasonable thing to want -- would make this a
         * cross-model observation leak, so the contract must hold or this must change with it.
         *
         * Not virtual and stored here rather than per backend: the behaviour is identical for
         * every device.
         */
        void setActivationObserver( ActivationObserver observer )
        {
            activation_observer_ = std::move( observer );
        }

        [[nodiscard]] const ActivationObserver& getActivationObserver() const noexcept
        {
            return activation_observer_;
        }

        [[nodiscard]] bool hasActivationObserver() const noexcept
        {
            return static_cast<bool>( activation_observer_ );
        }

    protected:
        IExecutionContext() = default;

    private:
        ActivationObserver activation_observer_{};
    };
}