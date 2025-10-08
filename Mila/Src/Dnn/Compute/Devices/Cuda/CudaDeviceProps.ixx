module;
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

export module Compute.CudaDeviceProps;

import Cuda.Helpers;
import Cuda.Error;

namespace Mila::Dnn::Compute
{
	export class CudaDeviceProps
	{
	public:

        CudaDeviceProps( int deviceId )
        {
            cudaCheckStatus( cudaGetDeviceProperties( &props_, deviceId ) );
        }

        const cudaDeviceProp* getProperties() const
        {
            return &props_;
        }

        std::string toString() const
        {
            // TJT: REVIEW Include CUDA driver and runtime versions
            int driver_version = Cuda::getDriverVersion();
            int runtime_version = Cuda::getRuntimeVersion();
            
            std::stringstream ss;

            ss << "Device Properties: " << std::endl
                << "Device: " << props_.name << std::endl
                << " CUDA Driver Version: " << (driver_version / 1000) << ", " << (driver_version % 100) / 10 << std::endl
                << " CUDA Runtime Version: " << (runtime_version / 1000) << "," << (runtime_version % 100) / 10 << std::endl;
            ss << " CUDA Capability Major/Minor version number: " << props_.major << "/" << props_.minor << std::endl;
            ss << " Total amount of global memory (bytes): " << props_.totalGlobalMem << std::endl;

            return ss.str();
        }

        const std::string getName() const {
            return props_.name;
        }

        const std::pair<int, int> getComputeCapability() const {
            return std::pair<int, int>( props_.major, props_.minor );
        }

        const size_t getTotalGlobalMem() const {
            return props_.totalGlobalMem;
        }

	private:

		cudaDeviceProp props_;
	};
}