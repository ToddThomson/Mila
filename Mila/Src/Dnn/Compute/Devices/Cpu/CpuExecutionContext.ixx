/**
 * @file CpuExecutionContext.ixx
 * @brief CPU-specific execution context implementation.
 *
 * Provides a minimal execution context for CPU operations. Unlike CUDA, CPU does not
 * require stream management or library handles, so this implementation is primarily
 * for API consistency with other device types.
 */

module;
#include <memory>
#include <string>
#include <stdexcept>

export module Compute.CpuExecutionContext;

import Compute.ExecutionContext;
import Compute.DeviceContext;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU-specific execution context implementation.
     *
     * Minimal execution context for CPU operations. CPU execution does not require
     * streams or library handles like CUDA, so this implementation provides the
     * ExecutionContext interface without additional overhead.
     *
     * Design rationale:
     * - No streams: CPU operations execute sequentially
     * - No library handles: CPU uses standard libraries (Eigen, BLAS, etc.) directly
     * - synchronize() is a no-op: CPU operations complete synchronously
     * - Maintains API consistency with CUDA execution contexts
     */
    export class CpuExecutionContext : public ExecutionContext {
    public:
        /**
         * @brief Constructs CPU execution context from device context.
         *
         * @param device_context Device context for this execution context
         * @throws std::invalid_argument If device_context is null or not a CPU device
         */
        explicit CpuExecutionContext( std::shared_ptr<DeviceContext> device_context )
            : device_context_( device_context ) {

            if (!device_context_) {
                throw std::invalid_argument( "Device context cannot be null" );
            }

            if (!device_context_->isCpuDevice()) {
                throw std::invalid_argument(
                    "CpuExecutionContext requires CPU device context"
                );
            }
        }

        /**
         * @brief Destructor.
         *
         * No resources to clean up for CPU execution context.
         */
        ~CpuExecutionContext() override = default;

        /**
         * @brief Synchronizes CPU execution.
         *
         * No-op for CPU since operations execute synchronously. Provided for
         * API consistency with CUDA execution contexts.
         */
        void synchronize() override {
            // CPU operations are synchronous, no action needed
        }

        /**
         * @brief Gets the underlying device context.
         * @return Shared pointer to the CPU device context
         */
        std::shared_ptr<DeviceContext> getDeviceContext() const override {
            return device_context_;
        }

    private:
        std::shared_ptr<DeviceContext> device_context_;
    };
}