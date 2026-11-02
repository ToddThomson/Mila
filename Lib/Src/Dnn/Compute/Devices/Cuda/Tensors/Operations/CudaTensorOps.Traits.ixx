/**
 * @file CudaTensorOps.Traits.ixx
 * @brief Binds DeviceType::Cuda to CudaTensorOps.
 */

export module Dnn.TensorOpsTraits.Cuda;

import Compute.DeviceType;
import Dnn.TensorOpsTraits;
import Compute.CudaTensorOps;

namespace Mila::Dnn
{
    // An explicit specialization cannot carry `export`, which is what lets Dnn.TensorOps
    // re-export this module without publishing the CUDA backend.
    template<>
    struct TensorOpsTraits<Compute::DeviceType::Cuda>
    {
        using type = CudaTensorOps;
    };
}
