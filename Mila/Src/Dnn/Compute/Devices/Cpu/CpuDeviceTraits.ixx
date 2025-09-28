export module Compute.CpuComputeDeviceTraits;

import Compute.DeviceTraits;
import Compute.CpuMemoryResource;
import Compute.CpuDevice;

namespace Mila::Dnn::Compute
{
    template<>
    struct ComputeDeviceTraits<CpuMemoryResource>
    {
        using type = CpuDevice;
    };
}