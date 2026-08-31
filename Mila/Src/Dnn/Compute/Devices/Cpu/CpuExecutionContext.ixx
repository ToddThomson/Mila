/**
 * @file CpuExecutionContext.ixx
 * @brief CPU-specific execution context specialization.
 */

module;
#include <memory>
#include <string>
#include <stdexcept>

export module Compute.CpuExecutionContext;

import Compute.IExecutionContext;
import Compute.Device;
import Compute.DeviceId;
import Compute.CpuDevice;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU execution context specialization.
     *
     * Minimal execution context for CPU operations. CPU execution does not
     * require streams or library handles, so this implementation provides
     * the interface without additional overhead.
     */
    export class CpuExecutionContext : public IExecutionContext
    {
    public:
        /**
         * @brief Constructs CPU execution context.
         *
         * @param device_id CPU device identifier (typically Device::Cpu()).
         */
        explicit CpuExecutionContext( DeviceId device_id = Device::Cpu() )
            : device_id_( device_id )
        {
        }

        CpuExecutionContext( const CpuExecutionContext& ) = delete;
        CpuExecutionContext& operator=( const CpuExecutionContext& ) = delete;
        CpuExecutionContext( CpuExecutionContext&& ) = delete;
        CpuExecutionContext& operator=( CpuExecutionContext&& ) = delete;

        /**
         * @brief Gets the device identifier.
         *
         * @return DeviceId The CPU device identifier.
         */
        [[nodiscard]] DeviceId getDeviceId() const noexcept override
        {
            return device_id_;
        }

        /**
         * @brief Synchronizes CPU execution (no-op).
         *
         * CPU operations are synchronous, so this is a no-op.
         */
        void synchronize() override
        {
        }

    private:
        DeviceId device_id_;
    };
}