/**
 * @file CpuTensorOps.Fill.ixx
 * @brief CPU tensor fill operations partition
 */

module;
#include <cstring>
#include <algorithm>
#include <span>
#include <type_traits>
#include <stdexcept>

export module Dnn.TensorOps:Transfer.Cpu;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceTraits;
import Compute.CpuMemoryResource;
import Compute.CpuTensorDataTypeTraits;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief CPU specialization of TensorOps for memory transfer operations.
     *
     */
    template<typename TComputeDeviceTag> struct TensorOps;

    export template<>
        struct TensorOps<Compute::CpuComputeDeviceTag>
    {
    };
}