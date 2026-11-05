/**
 * @file MetalExecutionContext.Traits.ixx
 * @brief Binds DeviceType::Metal to MetalExecutionContext.
 */

export module Compute.ExecutionContextTraits.Metal;

import Compute.DeviceType;
import Compute.ExecutionContextTraits;
import Compute.MetalExecutionContext;

namespace Mila::Dnn::Compute
{
    // An explicit specialization cannot carry `export`, which is what lets
    // Compute.ExecutionContext re-export this module without publishing anything.
    template<>
    struct ExecutionContextTraits<DeviceType::Metal>
    {
        using type = MetalExecutionContext;
    };
}
