export module Compute.CudaDeviceTraits;

import Compute.DeviceTraits;
import Compute.CudaMemoryResource;
import Compute.CudaManagedMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    template<>
    struct ComputeDeviceTraits<CudaMemoryResource>
    {
        using type = CudaDevice;
    };

    template<>
    struct ComputeDeviceTraits<CudaManagedMemoryResource>
    {
        using type = CudaDevice;
    };

    template<>
    struct ComputeDeviceTraits<CudaPinnedMemoryResource>
    {
        using type = CudaDevice;
    };
}