export module Compute.CudaDeviceTraits;

import Compute.DeviceTraits;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaManagedMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    /*template<>
    struct ComputeDeviceTraits<CudaDeviceMemoryResource>
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
    };*/
}