/**
 * @file CpuTensorOps.Traits.ixx
 * @brief Binds DeviceType::Cpu to CpuTensorOps.
 */

export module Dnn.TensorOpsTraits.Cpu;

import Compute.DeviceType;
import Dnn.TensorOpsTraits;
import Compute.CpuTensorOps;

namespace Mila::Dnn
{
    // An explicit specialization cannot carry `export`, which is what lets Dnn.TensorOps
    // re-export this module without publishing the CPU backend.
    template<>
    struct TensorOpsTraits<Compute::DeviceType::Cpu>
    {
        using type = CpuTensorOps;
    };
}
