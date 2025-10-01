module;
#include <cstring>
#include <algorithm>
#include <span>
#include <type_traits>
#include <stdexcept>

export module Dnn.TensorOps:Fill.Cuda;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceTraits;
import Compute.CudaMemoryResource;
import Compute.CudaTensorDataType;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    template<typename TComputeDeviceTag> struct TensorOps;

    export template<>
        struct TensorOps<Compute::CudaComputeDeviceTag>
    {
    };
}