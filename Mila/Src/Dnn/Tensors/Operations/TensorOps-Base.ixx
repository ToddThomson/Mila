/**
 * @file TensorOps-Base.ixx
 * @brief The device-dispatched entry point for tensor operations.
 *
 * Declares the `TensorOps` alias template that resolves a device type to its backend
 * implementation through Dnn::TensorOpsTraits.
 */

export module Dnn.TensorOps.Base;

import Compute.DeviceType;
export import Dnn.TensorOpsTraits;

namespace Mila::Dnn
{
    /**
     * @brief Device-dispatched tensor operations.
     *
     * Resolves to the backend implementation for a device -- CpuTensorOps,
     * CudaTensorOps -- each of which supplies the elementwise, reduction, copy and fill
     * operations the device-neutral entry points in Dnn.TensorOps dispatch to.
     *
     * A backend is added by defining its operations class and binding it in a traits
     * module of its own: `template<> struct TensorOpsTraits<DeviceType::X> { using type = XTensorOps; };`.
     * Backends respect host/device accessibility -- CPU on host-accessible memory,
     * CUDA on device memory.
     *
     * @tparam TDevice Compute device type (DeviceType::Cpu, DeviceType::Cuda, ...)
     */
    export template<Compute::DeviceType TDevice>
    using TensorOps = typename TensorOpsTraits<TDevice>::type;
}