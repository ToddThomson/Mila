/**
 * @file CpuExecutionContext.Traits.ixx
 * @brief Binds DeviceType::Cpu to CpuExecutionContext.
 */

export module Compute.ExecutionContextTraits.Cpu;

import Compute.DeviceType;
import Compute.ExecutionContextTraits;
import Compute.CpuExecutionContext;

namespace Mila::Dnn::Compute
{
    // An explicit specialization cannot carry `export`, which is what lets
    // Compute.ExecutionContext re-export this module without publishing anything.
    template<>
    struct ExecutionContextTraits<DeviceType::Cpu>
    {
        using type = CpuExecutionContext;
    };
}
