export module Compute.ExecutionContextBase;

import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Templated execution context for device-specific operations.
     *
     * ExecutionContext manages the resources needed for executing operations:
     * streams for asynchronous execution, library handles (cuBLAS, cuDNN),
     * and synchronization primitives. The template parameter provides compile-time
     * device type checking and enables device-specific optimizations.
     *
     * Design rationale:
     * - Template parameter eliminates runtime type checking overhead
     * - Each device type has a specialized implementation
     * - Modules are templated on device type, ensuring type safety throughout
     * - No virtual function overhead for device-specific operations
     *
     * @tparam TDeviceType The device type (Cpu, Cuda, Metal, etc.)
     */
    export template<DeviceType TDeviceType>
        class ExecutionContext;
}