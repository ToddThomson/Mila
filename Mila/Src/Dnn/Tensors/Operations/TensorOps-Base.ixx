/**
 * @file TensorOps-Base.ixx
 * @brief Base declaration for device-specific TensorOps specializations.
 *
 * Declares the `TensorOps` template used as the entry point for device-specific
 * tensor operation implementations. Specializations must provide the concrete
 * operations for their device.
 */

export module Dnn.TensorOps.Base;

import Compute.DeviceType;

namespace Mila::Dnn
{
    /**
     * @brief Device-dispatched TensorOps interface template.
     *
     * Specialize `TensorOps<TDevice>` for each supported `Compute::DeviceType` to
     * provide backend implementations of tensor operations (elementwise, reductions,
     * copy, fill, etc.).
     *
     * Requirements for specializations:
     * - Provide the operations used by the framework (static or instance methods),
     *   matching the signatures expected by TensorOps callers.
     * - Use the device's memory resource and execution context types to access
     *   device-specific APIs and streams.
     * - Respect host/device accessibility guarantees: CPU specializations must
     *   operate on host-accessible memory, CUDA specializations on device memory.
     *
     * Usage example:
     * @code
     * template<>
     * struct TensorOps<Compute::DeviceType::Cpu>
     * {
     *     static void copy(const ITensor& src, ITensor& dst);
     *     // ...
     * };
     * @endcode
     *
     * @tparam TDevice Compute device type to specialize for (DeviceType::Cpu, DeviceType::Cuda, ...)
     */
    export template<Compute::DeviceType TDevice> 
    struct TensorOps;
}