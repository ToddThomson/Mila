/**
 * @file CudaExecutionContext.Traits.ixx
 * @brief Binds DeviceType::Cuda to CudaExecutionContext.
 */

export module Compute.ExecutionContextTraits.Cuda;

import Compute.DeviceType;
import Compute.ExecutionContextTraits;
import Compute.CudaExecutionContext;

namespace Mila::Dnn::Compute
{
    // An explicit specialization cannot carry `export`, which is what lets
    // Compute.ExecutionContext re-export this module without publishing anything.
    template<>
    struct ExecutionContextTraits<DeviceType::Cuda>
    {
        using type = CudaExecutionContext;
    };
}
