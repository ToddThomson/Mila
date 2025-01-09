module;
#include <iostream>
#include <memory>
#include <cuda_runtime.h>

export module Compute.BackendFactory;

import Compute.BackendInterface;
import Compute.CpuBackend;
import Compute.CudaBackend;

namespace Mila::Dnn::Compute
{
    export class BackendFactory {
    public:
        static std::unique_ptr<BackendInterface> createBackend( const std::string& name ) {
            int deviceCount = 0;
            cudaGetDeviceCount( &deviceCount );

            if ( deviceCount > 0 ) {
                return std::make_unique<Cuda::CudaBackend>();
            } else {
                return std::make_unique<Cpu::CpuBackend>();
            }
        }
    };
}
