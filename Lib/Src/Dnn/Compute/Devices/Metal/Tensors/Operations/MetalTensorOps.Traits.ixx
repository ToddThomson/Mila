/**
 * @file MetalTensorOps.Traits.ixx
 * @brief Binds DeviceType::Metal to MetalTensorOps.
 */

export module Dnn.TensorOpsTraits.Metal;

import Compute.DeviceType;
import Dnn.TensorOpsTraits;
import Compute.MetalTensorOps;

namespace Mila::Dnn
{
    // An explicit specialization cannot carry `export`, which is what lets Dnn.TensorOps
    // re-export this module without publishing the Metal backend.
    template<>
    struct TensorOpsTraits<Compute::DeviceType::Metal>
    {
        using type = MetalTensorOps;
    };
}
