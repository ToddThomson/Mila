module;
#include <iostream>
#include <memory>
#include <cuda_runtime.h>

export module Compute.DeviceFactory;

import Compute.DeviceInterface;
import Compute.CpuDevice;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    export class DeviceFactory {
    public:
        static std::shared_ptr<DeviceInterface> createDevice( const std::string& name ) {
            int deviceCount = 0;
            cudaGetDeviceCount( &deviceCount );

            if ( deviceCount > 0 ) {
                return std::make_shared<Cuda::CudaDevice>();
            } else {
                return std::make_shared<Cpu::CpuDevice>();
            }
        }
    };
}
